"""THE DISTRIBUTED SWEEP ( `grill/calcul-distribue/decisions.md` ): the seeds split between BOXES, each box an ordinary diagram of its
seeds whose cells start from the whole domain, and the cells that a box cannot finish TRAVELLING to the boxes that can still cut them.

    PowerDiagram( positions, weights, boundaries = box_half_spaces( mi, ma ), nb_boxes = 4 ).measures   # boxes in turn, one device
    PowerDiagram( loom.spread( positions, ranks ), weights, boundaries = ... ).measures              # one box per rank

`ranks` is any group of loom ranks ( `loom.JaxRanks( mesh )`, `loom.TorchRanks()`... ), and a Jax array already sharded along the seeds
counts as spread. `PowerDiagram` then hands the work to `PowerDiagram_Distributed`, which runs `_program` on every rank through
`loom.spmd` -- the same code for every framework.

A round ( `diagram/Distributed.h::dist_round` ) cuts every cell a box holds with the seeds of that box, then finishes it ( its measure
goes back to the owner of its seed ) or sends it to the nearest box it has not visited that can still cut it. Rounds go on until no cell
travels. No ghost seed, no halo: what crosses the boundaries are cells, of variable size, in rows of fixed width ( a `loom.Capacity`:
a cell larger than a row, or more cells for a rank than its slots, runs the program again with more room ).

= Internals

The ids of the cuts are global ( the offset of the box + the rank in the box ), so that the adjoint can send a derivative to the owner of
any cut. `K <= 52`: the visited boxes are the bits of an integer stored in a double. 2D only, forward only, for now. The boxes ( the
split, the trees, their rows ) are still made on the host, from the whole cloud: the redistribution by exchanges is a later step.
"""
import numpy as np

import loom
from loom import IntTensor, RealTensor, Axis, ShapeVar
from loom.compilation.FfiCode import FfiCode
from loom.tensor.batch import new_batch_axis
from loom.tensor.Dtype import Dtype

from .AaBsp import AaBsp
from .CellScratch import CellScratch
from .PowerDiagram import PowerDiagram, _on_item

HEADER = 4 + 2          # owner, visited, weight, position ( 2D ), nb


def split_seeds( positions, nb_boxes ):
    """`nb_boxes` lists of indices, the seeds cut by recursive median bisections along the widest extent ( a kd-tree's top levels:
    boxes of about the same number of seeds, compact in space; the same number exactly when `nb_boxes` divides it evenly all the way
    down, e.g. a power of two dividing `n` )"""
    positions = np.asarray( positions )

    def rec( idx, k ):
        if k == 1:
            return [ idx ]
        p = positions[ idx ]
        d = int( np.argmax( p.max( axis = 0 ) - p.min( axis = 0 ) ) )
        order = idx[ np.argsort( p[ :, d ], kind = "stable" ) ]
        kl = k // 2
        cut = len( idx ) * kl // k
        return rec( order[ :cut ], kl ) + rec( order[ cut: ], k - kl )

    return rec( np.arange( len( positions ) ), int( nb_boxes ) )


def box_row( pd ):
    """what the other boxes know of box `pd`: the box of its seeds and the affine majorant of their weights, `[ lo, hi, a, b ]`"""
    t = pd.tree
    host = lambda x: np.asarray( loom.to_numpy( x.value ) )
    box = host( t.node_box )[ 0 ]
    return np.concatenate( [ box[ 0 ], box[ 1 ], host( t.node_wa )[ 0 ], [ float( host( t.node_wb )[ 0 ] ) ] ] )


def _round( pd, first, inbox, boxes, my_box, id_offset, gids, width ):
    """one round on the box of `pd` ( any framework, traced or not ): `first` ( a host bool ) its own seeds, else the rows of `inbox`.
    `my_box`, `id_offset`: `[ 1 ]` int arrays. Returns the values `( dest, msgs, gid, val, status )`"""
    dom = pd._domain_cell()
    m = int( pd.positions.shape[ 0 ] ) if first else int( inbox.shape[ 0 ] )
    nb_words = pd._scratch_words( pd._scratch_capacity, 1, False )
    kp = loom.kernel_parameters_for( "power_diagram_dist_round" )
    nt = max( 1, min( 64, m ) )
    num_thread = new_batch_axis( nt, prefix = "thread" )
    num_item = Axis( ShapeVar( m ), name = "num_item" )
    num_in = Axis( ShapeVar( int( inbox.shape[ 0 ] ) ), name = "num_in" )
    num_w = Axis( ShapeVar( width ), name = "msg_width" )
    num_box = Axis( ShapeVar( int( boxes.shape[ 0 ] ) ), name = "num_box" )
    num_bw = Axis( ShapeVar( int( boxes.shape[ 1 ] ) ), name = "box_width" )
    one = Axis( ShapeVar( 1 ), name = "one" )
    num_gid = Axis( ShapeVar( int( gids.shape[ 0 ] ) ), name = "num_gid" )

    out_msgs = RealTensor[ num_item, num_w ]()
    out_dest = IntTensor[ num_item ]()
    out_val  = RealTensor[ num_item ]()
    out_gid  = IntTensor[ num_item ]()
    status   = IntTensor[ one ]()
    rank = "SI( inputs.thread_rank( batch_index ) ), SI( inputs.thread_rank.shape( 0 ) )"
    kp.ffi_call(
        FfiCode.per_item( _on_item( "sdot::distributed::dist_round( inputs.power_diagram, inputs.dom_cell, scratch.pool( batch_index ), " )
                          + rank + ", " + _on_item( "inputs.first, inputs.inbox, inputs.boxes, inputs.my_box, inputs.id_offset, "
                                                    "inputs.gids, outputs.out_msgs, outputs.out_dest, outputs.out_val, "
                                                    "outputs.out_gid, outputs.status );" ),
                          includes = [ "sdot/diagram/Distributed.h" ] ),
        power_diagram = pd,
        dom_cell = dom,
        pool = CellScratch.for_call( nb_words, dom.kernel_dtype, batch_axes = [ num_thread ] ),
        thread_rank = IntTensor[ num_thread ]( np.arange( nt, dtype = np.int32 ) ),
        first = IntTensor[ one ]( np.array( [ int( first ) ], dtype = np.int32 ) ),
        inbox = RealTensor[ num_in, num_w ]( inbox ),
        boxes = RealTensor[ num_box, num_bw ]( boxes ),
        my_box = IntTensor[ one ]( my_box ),
        id_offset = IntTensor[ one ]( id_offset ),
        gids = IntTensor[ num_gid ]( gids ),
        out_msgs = loom.out( out_msgs ),
        out_dest = loom.out( out_dest ),
        out_val = loom.out( out_val ),
        out_gid = loom.out( out_gid ),
        status = loom.out( status ),
    )
    return out_dest.value, out_msgs.value, out_gid.value, out_val.value, status.value


# ---- the boxes in turn, from Python, on one device ---------------------------------------------------------------------------------

class DistributedDiagram:
    """the boxes of a distributed diagram driven from Python, one box after the other: one `PowerDiagram` per box, the global ids of
    their seeds, the table of the boxes. What validates the algorithm, and the shape of the out-of-core mode."""

    def __init__( self, positions, weights, boundaries, nb_boxes, msg_capacity = 32 ):
        positions = np.asarray( positions, dtype = np.float64 )
        weights = np.zeros( len( positions ) ) if weights is None else np.asarray( weights, dtype = np.float64 )
        if not 1 <= nb_boxes <= 52:
            raise ValueError( f"{ nb_boxes } boxes: 1 to 52 for now ( the visited set is the bits of a double )" )
        if positions.shape[ 1 ] != 2:
            raise ValueError( "the distributed sweep is 2D only for now" )
        self.n = len( positions )
        self.parts = split_seeds( positions, nb_boxes )
        if any( len( p ) == 0 for p in self.parts ):
            raise ValueError( f"{ nb_boxes } boxes for { self.n } seeds: a box would be empty" )
        self.diagrams = [ PowerDiagram( positions[ p ], weights[ p ], boundaries = boundaries ) for p in self.parts ]
        counts = np.array( [ len( p ) for p in self.parts ] )
        self.offsets = np.concatenate( [ [ 0 ], np.cumsum( counts )[ :-1 ] ] )
        self.msg_capacity = int( msg_capacity )
        self.boxes = np.array( [ box_row( pd ) for pd in self.diagrams ] )

    @property
    def nb_boxes( self ):
        return len( self.diagrams )

    def round( self, b, inbox ):
        """one round on box `b`: `inbox` the rows it receives ( `None`: its own seeds ). Returns `( dest, msgs, gid, val, overflow )`"""
        W = HEADER + 3 * self.msg_capacity
        first = inbox is None
        res = _round( self.diagrams[ b ], first, np.zeros( ( 1, W ) ) if first else inbox, self.boxes,
                      np.array( [ b ], dtype = np.int32 ), np.array( [ self.offsets[ b ] ], dtype = np.int32 ),
                      self.parts[ b ].astype( np.int32 ), W )
        dest, msgs, gid, val, status = ( np.asarray( loom.to_numpy( r ) ) for r in res )
        return dest, msgs, gid, val, bool( status[ 0 ] )

    def measures( self, max_rounds = 1000, stats = None ):
        """the measure of every cell, `[ n ]` in the order of the given seeds: rounds on every box until no cell travels"""
        while True:
            res = np.zeros( self.n )
            seen = np.zeros( self.n, dtype = np.int64 )
            inboxes = [ None ] * self.nb_boxes
            first = True
            overflow = False
            nb_rounds = 0
            while True:
                outs = [ [] for _ in range( self.nb_boxes ) ]
                travelling = 0
                for b in range( self.nb_boxes ):
                    if not first and inboxes[ b ] is None:
                        continue
                    dest, msgs, gid, val, ovf = self.round( b, inboxes[ b ] )
                    overflow |= ovf
                    done = dest == -1
                    g = gid[ done ].astype( np.int64 )
                    np.add.at( res, g, val[ done ] )
                    np.add.at( seen, g, 1 )
                    for j in np.unique( dest[ dest >= 0 ] ):
                        outs[ j ].append( msgs[ dest == j ] )
                        travelling += int( ( dest == j ).sum() )
                inboxes = [ np.concatenate( o ) if o else None for o in outs ]
                first = False
                nb_rounds += 1
                if travelling == 0 or overflow or nb_rounds >= max_rounds:
                    break
            if not overflow:
                break
            self.msg_capacity *= 2
        if stats is not None:
            stats[ "rounds" ] = nb_rounds
        assert ( seen == 1 ).all(), "a cell was lost or finished twice"
        return res


# ---- one box per rank: a program for `loom.spmd` ------------------------------------------------------------------------------------

_TREE_FIELDS = ( "seed_indices", "node_left", "node_right", "node_begin", "node_end", "node_box", "node_wa", "node_wb" )


def _tree_from_arrays( arrays, n, d, leaf ):
    """an `AaBsp` over `n` seeds whose tensors are `arrays` ( the tree of THIS rank's box, traced or not ), as the card builds one
    ( `AaBsp._init_on_card` ): the shape is fixed by `n` and `leaf`, the order and its inverse are values"""
    t = object.__new__( AaBsp )
    t.max_depth = AaBsp.max_depth_for( n, leaf )
    t.max_seeds_per_leaf = leaf
    t._nb_leaves = None
    t._rank = arrays[ "rank_of" ]
    t.__base_init__( nb_dims = d, nb_lohi = 2, nb_bsp_seeds = n, nb_bsp_nodes = 2 ** t.max_depth - 1,
                     **{ k: arrays[ k ] for k in _TREE_FIELDS } )
    t._majorant_weights = None
    t._sorted_for = None
    return t


def _program( ranks, pos, w, boxes, tree, *, nloc, leaf, boundaries, rows, slots ):
    """what each rank runs ( `loom.spmd` ): `pos` / `w` the seeds of its box, `tree` its tree ( blocks `[ 1, ... ]` ), `boxes` the
    rows of all the boxes. Returns the measures of the cells of its seeds, `[ nloc ]`, in the order of its box. `rows` ( vertices per
    travelling cell ) and `slots` ( cells per pair of ranks and per round ) are `loom.Capacity`s."""
    o = loom.ops( pos )
    K = ranks.size
    W = HEADER + 3 * rows.value
    C = slots.value
    i32, f64 = Dtype.si( 32 ), Dtype.fp( 64 )

    t = _tree_from_arrays( { k: v[ 0 ] for k, v in tree.items() }, nloc, 2, leaf )
    t._majorant_weights = w           # ( its majorant was built on these weights: nothing to refresh )
    pd = PowerDiagram( pos, w, boundaries = boundaries, accelerator = t )
    me = ranks.rank()
    gids = me * nloc + o.arange( nloc, i32 )            # owner code: box * nloc + the seed's index in its box

    def step( first, inbox ):
        dest, msgs, gid, val, status = _round( pd, first, inbox, boxes, me, me * nloc, gids, W )
        rows.report( status )
        return dest, msgs, gid, val

    def settle( res, dest, gid, val ):
        """the finished cells: those of this rank's seeds added to `res`, the others sent back to their owner"""
        done = dest == -1
        owner = gid // nloc
        mine = done & ( owner == me )
        res = loom.scatter_add( res, o.where( mine, gid % nloc, -1 ), val )
        back = loom.route( o.where( done & ~mine, owner, -1 ), o.stack( [ o.astype( gid % nloc, f64 ), val ], axis = 1 ), K, slots )
        back = o.reshape( ranks.exchange( back ), ( K * C, 2 ) )
        return loom.scatter_add( res, o.astype( back[ :, 0 ], i32 ), back[ :, 1 ] )

    def send( dest, msgs ):
        return o.reshape( ranks.exchange( loom.route( dest, msgs, K, slots ) ), ( K * C, W ) )

    def moving( dest ):
        return ranks.all_sum( o.sum( o.astype( dest >= 0, i32 ) ) )

    dest, msgs, gid, val = step( True, o.zeros( ( 1, W ), f64 ) )
    res = settle( o.zeros( ( nloc, ), f64 ), dest, gid, val )

    def loop( c ):
        res, inbox, _ = c
        dest, msgs, gid, val = step( False, inbox )
        return settle( res, dest, gid, val ), send( dest, msgs ), moving( dest )

    res, _, _ = loom.while_loop( lambda c: c[ 2 ] > 0, loop, ( res, send( dest, msgs ), moving( dest ) ) )
    return res


# ---- what the user sees: a `PowerDiagram` -----------------------------------------------------------------------------------------

def is_spread( x ):
    """whether `x` is split between ranks ( `loom.spread`, or a Jax array sharded along its first axis ): what makes `PowerDiagram`
    distributed"""
    return loom.spread_of( x ) is not None


class PowerDiagram_Distributed( PowerDiagram ):
    """A `PowerDiagram` whose seeds are split between BOXES ( `PowerDiagram( positions, ... )` chooses it, see `diagram_class_for` ):
    one box per rank when the positions are spread ( `_program`, through `loom.spmd` ), or `nb_boxes` boxes in turn on one device
    ( `DistributedDiagram` ). Only `measures`, in 2D, with a domain, for now: the rest of the arguments of `PowerDiagram` are refused
    rather than ignored."""

    def __init__( self, positions, weights = None, boundaries = None, accelerator = None, nb_boxes = None,
                  msg_capacity = 16, inbox_capacity = 64, **kwargs ):
        given = sorted( k for k, v in kwargs.items() if v is not None )
        if accelerator is not None or given:
            raise NotImplementedError( f"a distributed diagram takes positions, weights and boundaries only for now ( got { given or [ 'accelerator' ] } )" )
        if boundaries is None:
            raise NotImplementedError( "a distributed diagram needs a domain ( `boundaries` ) for now" )
        s = loom.spread_of( positions )
        self._ranks = None if s is None else s.ranks
        if self._ranks is not None and nb_boxes is not None and nb_boxes != self._ranks.size:
            raise ValueError( "spread positions: one box per rank, `nb_boxes` cannot say otherwise" )
        self._given_positions = positions
        self._given_weights = weights
        self._boundaries = boundaries
        self._nb_boxes = nb_boxes
        # ( kept from one call to the next: what an overflow taught is not learnt again )
        self._rows = loom.Capacity( msg_capacity )
        self._slots = loom.Capacity( inbox_capacity )
        self._boxes = None

    @property
    def positions( self ):
        return self._given_positions

    @property
    def weights( self ):
        return self._given_weights

    @property
    def dim_count( self ):
        return int( self._given_positions.shape[ 1 ] )

    @property
    def measures( self ):
        """the measure of each cell, `[ n ]` in the order of the positions -- spread like them when they are"""
        # THE BOXES, on the host for now ( from the whole cloud: a collective when every rank holds its block only )
        pos = loom.gather_to_host( self._given_positions ).astype( np.float64 )
        w = np.zeros( len( pos ) ) if self._given_weights is None else loom.gather_to_host( self._given_weights ).astype( np.float64 )
        if self._ranks is None:
            if self._boxes is None:
                self._boxes = DistributedDiagram( pos, w, self._boundaries, self._nb_boxes, msg_capacity = self._rows.value )
            return RealTensor( self._boxes.measures() )

        ranks = self._ranks
        K, ( n, d ) = ranks.size, pos.shape
        if d != 2:
            raise NotImplementedError( "the distributed sweep is 2D only for now" )
        if n % K:
            raise ValueError( f"{ n } seeds cannot be split evenly between { K } ranks" )
        nloc, leaf = n // K, 10
        parts = split_seeds( pos, K )
        if any( len( p ) != nloc for p in parts ):
            raise ValueError( f"the bisections did not give { nloc } seeds to every box ( take a power of two ranks dividing n )" )
        trees = [ AaBsp( pos[ p ], w[ p ], max_seeds_per_leaf = leaf ) for p in parts ]
        boxes = np.array( [ box_row( PowerDiagram( pos[ p ], w[ p ], boundaries = self._boundaries, accelerator = t ) )
                            for p, t in zip( parts, trees ) ] )
        stacked = { k: np.stack( [ np.asarray( loom.to_numpy( getattr( t, k ).value ) ) for t in trees ] ) for k in _TREE_FIELDS }
        stacked[ "rank_of" ] = np.stack( [ np.asarray( t.rank_of_seeds() ) for t in trees ] )
        order = np.concatenate( parts )

        names = list( stacked )
        program = lambda rk, p, ww, bx, *tr: _program( rk, p, ww, bx, dict( zip( names, tr ) ), nloc = nloc, leaf = leaf,
                                                       boundaries = self._boundaries, rows = self._rows, slots = self._slots )
        res = loom.spmd( program, ranks )( loom.spread_host( pos[ order ], ranks ), loom.spread_host( w[ order ], ranks ),
                                           _replicated( boxes, self._given_positions ),
                                           *( loom.spread_host( stacked[ k ], ranks ) for k in names ) )
        # back to the order of the positions, spread like them ( on the host for now, as the boxes )
        out = np.empty( n )
        out[ order ] = loom.gather_to_host( res )
        return loom.spread_host( out, ranks )


def _replicated( array, like ):
    """`array` as a value of the framework of `like`, the same on every rank"""
    o = loom.ops( getattr( like, "value", like ) )
    return o.array( array, Dtype.fp( 64 ) )
