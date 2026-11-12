"""THE DISTRIBUTED SWEEP ( `grill/calcul-distribue/decisions.md` ): the seeds split between BOXES, each box an ordinary diagram of its
seeds whose cells start from the whole domain, and the cells that a box cannot finish TRAVELLING to the boxes that can still cut them.

    PowerDiagram( positions, weights, boundaries = box_half_spaces( mi, ma ), nb_boxes = 4 ).measures   # boxes in turn, one device
    PowerDiagram( loom.spread( positions, ranks ), weights, boundaries = ... ).measures              # one box per rank
    PowerDiagram( loom.spread( positions, ranks ), weights, boundaries = ..., nb_boxes = 8 )         # several boxes per rank

`ranks` is any group of loom ranks ( `loom.JaxRanks( mesh )`, `loom.TorchRanks()`... ), and a Jax array already sharded along the seeds
counts as spread. `PowerDiagram` then hands the work to `PowerDiagram_Distributed`: ONE algorithm, a `loom.BoxProgram` ( `_sweep` ),
which loom runs on whatever the operands say -- the boxes in turn on one rank, Jax devices, Torch processes, MPI ranks.

A round ( `diagram/Distributed.h::dist_round` ) cuts every cell a box holds with the seeds of that box, then finishes it ( its measure
goes back to the owner of its seed ) or sends it to the nearest box it has not visited that can still cut it. Rounds go on until no cell
travels. No ghost seed, no halo: what crosses the boundaries are cells, of variable size, in rows of fixed width ( a `loom.Capacity`:
a cell larger than a row, or more cells for a rank than its slots, runs the program again with more room ).

= Internals

The ids of the cuts are global ( the offset of the box + the rank in the box ), so that the adjoint can send a derivative to the owner of
any cut. `K <= 52`: the visited boxes are the bits of an integer stored in a double. 2D only for now.

DIFFERENTIABLE ( Jax, Torch ) with respect to the positions and the weights when they are not spread: the seeds of each box are taken
from the given ones by the framework, the measures put back the same way, and `loom.BoxProgram` replays the rounds in reverse with
`dist_round_bwd` as the adjoint of a round. Spread positions ( or out of core ) give the measures only: their boxes are still made on
the host. The boxes ( the
split, the trees, their rows ) are still made on the host, from the whole cloud: the redistribution by exchanges is a later step.
Boxes of different sizes ( `n` not a multiple of `K` ) go to loom as lists, one value per box: not on Jax devices, whose single trace
needs one shape.
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
    nb_words = pd._scratch_words( pd._scratch_capacity, 1, True )       # ( sized for the adjoint, which cuts the same cells )
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
        FfiCode.per_item( _on_item( "sdot::distributed::dist_round_bwd( inputs.power_diagram, inputs.dom_cell, scratch.pool( batch_index ), " )
                          + rank + ", " + _on_item( "inputs.first, inputs.inbox, inputs.boxes, inputs.my_box, inputs.id_offset, "
                                                    "inputs.gids, outputs.out_msgs, outputs.out_dest, grad_of_outputs.out_msgs, "
                                                    f"grad_of_outputs.out_val, { pd._grad_seeds_expr() }, grad_of_inputs.inbox );" ),
                          # the seeds of the box and the rows received, from the rows sent and from the measures
                          computed_grads = "inputs.power_diagram.measures_computed_grads( computed_grads.from_out_msgs.power_diagram ); "
                                           "inputs.power_diagram.measures_computed_grads( computed_grads.from_out_val.power_diagram ); "
                                           "computed_grads.from_out_msgs.inbox = true; computed_grads.from_out_val.inbox = true;",
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


# ---- one algorithm for every runtime: a `loom.BoxProgram` ----------------------------------------------------------------------------

_TREE_FIELDS = ( "seed_indices", "node_left", "node_right", "node_begin", "node_end", "node_box", "node_wa", "node_wb" )
_BOX_FIELDS = _TREE_FIELDS + ( "rank_of", )
LEAF = 10


def _tree_from_arrays( arrays, n, d, leaf ):
    """an `AaBsp` over `n` seeds whose tensors are `arrays` ( the tree of a box, traced or not ), as the card builds one
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


def _sweep( boundaries, L, rows, send_capacity, add_capacity ):
    """the `BoxProgram` of the sweep. Per box: its seeds `pos` `[ n, 2 ]`, `w` `[ n ]`, the arrays of its tree; shared: `boxes`, the rows
    of all the boxes. The owner of a seed is `box * L + its index in the box`: its measure is added to `measures[ box, index ]`.
    `rows` ( vertices per travelling cell ) is a `loom.Capacity`, and so the width of a row follows it"""
    i32 = Dtype.si( 32 )

    def width():
        return HEADER + 3 * rows.value

    def run( box, first, inbox, pos, w, *rest ):
        *tree, boxes = rest
        o = loom.ops( pos )
        # ( the tree and the box rows say where to look for the cuts, not what they give: constants for the adjoint )
        tree, boxes = [ o.stop_gradient( a ) for a in tree ], o.stop_gradient( boxes )
        n = int( pos.shape[ 0 ] )
        t = _tree_from_arrays( dict( zip( _BOX_FIELDS, tree ) ), n, 2, LEAF )
        t._majorant_weights = w           # ( its majorant was built on these weights: nothing to refresh )
        pd = PowerDiagram( pos, w, boundaries = boundaries, accelerator = t )
        off = o.astype( box.index, i32 ) * L
        dest, msgs, gid, val, status = _round( pd, first, inbox, boxes, box.index, off, off + o.arange( n, i32 ), width() )
        rows.report( status )
        none = gid * 0 - 1
        return loom.Outbox( send = ( dest, msgs ), add = { "measures": ( o.where( dest == -1, gid // L, none ), gid % L, val ) } )

    def start( box, pos, w, *rest ):
        o = loom.ops( pos )
        return run( box, True, o.zeros( ( 1, width() ), Dtype.fp( 64 ) ), pos, w, *rest )

    def step( box, inbox, pos, w, *rest ):
        o = loom.ops( pos )
        # ( an empty slot of the inbox: an owner < 0, as the kernel reads it )
        rows_in = o.where( o.reshape( inbox.valid, ( -1, 1 ) ), inbox.rows, inbox.rows * 0 - 1 )
        return run( box, False, rows_in, pos, w, *rest )

    return loom.BoxProgram( start, step, row_width = width, outputs = { "measures": L },
                            send_capacity = send_capacity, add_capacity = add_capacity )


# ---- what the user sees: a `PowerDiagram` -----------------------------------------------------------------------------------------

def is_spread( x ):
    """whether `x` is split between ranks ( `loom.spread`, or a Jax array sharded along its first axis ): what makes `PowerDiagram`
    distributed"""
    return loom.spread_of( x ) is not None


class PowerDiagram_Distributed( PowerDiagram ):
    """A `PowerDiagram` whose seeds are split between BOXES ( `PowerDiagram( positions, ... )` chooses it, see `diagram_class_for` ):
    `nb_boxes` of them ( by default one per rank of the spread positions ), run by `_sweep` on the ranks of the positions, or in turn on
    one rank when they are not spread. Only `measures`, in 2D, with a domain, for now: the rest of the arguments of `PowerDiagram` are
    refused rather than ignored."""

    def __init__( self, positions, weights = None, boundaries = None, accelerator = None, nb_boxes = None,
                  msg_capacity = 16, inbox_capacity = 64, max_resident = None, **kwargs ):
        given = sorted( k for k, v in kwargs.items() if v is not None )
        if accelerator is not None or given:
            raise NotImplementedError( f"a distributed diagram takes positions, weights and boundaries only for now ( got { given or [ 'accelerator' ] } )" )
        if boundaries is None:
            raise NotImplementedError( "a distributed diagram needs a domain ( `boundaries` ) for now" )
        s = loom.spread_of( positions )
        self._ranks = None if s is None else s.ranks
        K = nb_boxes if nb_boxes is not None else self._ranks.size if self._ranks is not None else 1
        if self._ranks is not None and K % self._ranks.size:
            raise ValueError( f"{ K } boxes on { self._ranks.size } ranks: every rank takes the same number of boxes" )
        if not 1 <= K <= 52:
            raise ValueError( f"{ K } boxes: 1 to 52 for now ( the visited set is the bits of a double )" )
        self._given_positions = positions
        self._given_weights = weights
        self._boundaries = boundaries
        self._nb_boxes = K
        if max_resident is not None and self._ranks is not None:
            raise NotImplementedError( "out of core ( `max_resident` ) on one rank only for now: positions not spread, `nb_boxes`" )
        self._max_resident = max_resident
        # ( kept from one call to the next: what an overflow taught is not learnt again )
        self._rows = loom.Capacity( msg_capacity )
        self._send = loom.Capacity( inbox_capacity )
        self._add = loom.Capacity( 64 )
        self._program = None
        self.last_rounds = None

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
        pos = _host_values( self._given_positions )
        w = np.zeros( len( pos ) ) if self._given_weights is None else _host_values( self._given_weights )
        n, d = pos.shape
        if d != 2:
            raise NotImplementedError( "the distributed sweep is 2D only for now" )
        K = self._nb_boxes
        parts = split_seeds( pos, K )
        if any( len( p ) == 0 for p in parts ):
            raise ValueError( f"{ K } boxes for { n } seeds: a box would be empty" )
        trees = [ AaBsp( pos[ p ], w[ p ], max_seeds_per_leaf = LEAF ) for p in parts ]
        boxes = np.array( [ box_row( PowerDiagram( pos[ p ], w[ p ], boundaries = self._boundaries, accelerator = t ) )
                            for p, t in zip( parts, trees ) ] )
        host = lambda x: np.asarray( loom.to_numpy( x ) )
        per_box = [ [ pos[ p ] for p in parts ], [ w[ p ] for p in parts ] ]
        per_box += [ [ host( getattr( t, k ).value ) for t in trees ] for k in _TREE_FIELDS ]
        per_box += [ [ np.asarray( t.rank_of_seeds() ) for t in trees ] ]
        like = getattr( self._given_positions, "value", self._given_positions )
        L = max( len( p ) for p in parts )
        # one rank, in memory: the seeds of the boxes TAKEN from the given ones by the framework, and the measures put back the same way,
        # so that the measures are differentiable ( `loom.BoxProgram`'s adjoint, `dist_round_bwd` ). The split, the trees and the box rows
        # are constants: where the cuts are looked for, not what they give
        local = self._ranks is None and self._max_resident is None
        if local:
            o = loom.ops( like )
            i32 = Dtype.si( 32 )
            P_ = like
            W_ = getattr( self._given_weights, "value", self._given_weights ) if self._given_weights is not None else o.zeros( ( n, ), o.dtype_of( like ) )
            take = lambda x, p: x[ o.array( np.asarray( p, dtype = np.int32 ), i32 ) ]
            if all( len( p ) == L for p in parts ):
                per_box[ 0 ], per_box[ 1 ] = take( P_, np.stack( parts ) ), take( W_, np.stack( parts ) )
            else:
                per_box[ 0 ], per_box[ 1 ] = [ take( P_, p ) for p in parts ], [ take( W_, p ) for p in parts ]
        if self._max_resident is not None:
            # out of core: the boxes stay on the host, read by loom when they have mail
            args = [ loom.BoxStore.of( a ) for a in per_box ]
        elif all( len( p ) == L for p in parts ):
            # boxes of one size: arrays `[ K, ... ]`, which every runtime takes ( Jax devices included )
            args = [ a if local and k < 2 else _native( np.stack( a ), like ) for k, a in enumerate( per_box ) ]
        else:
            args = [ a if local and k < 2 else [ _native( b, like ) for b in a ] for k, a in enumerate( per_box ) ]

        if self._program is None:
            self._program = _sweep( self._boundaries, L, self._rows, self._send, self._add )
        ranks = self._ranks or ( None if self._max_resident is not None else loom.LocalRanks() )
        out = self._program( *args, shared = ( _native( boxes, like ), ), ranks = ranks, max_resident = self._max_resident )[ "measures" ]
        self.last_rounds = self._program.last_rounds

        if local:
            inv = np.empty( n, dtype = np.int32 )
            for b, p in enumerate( parts ):
                inv[ p ] = b * L + np.arange( len( p ) )
            M = o.reshape( out.value, ( -1, ) )
            return RealTensor( M[ o.array( inv, i32 ) ] )

        # back to the order of the positions, spread like them ( on the host for now, as the boxes )
        M = np.asarray( loom.gather_to_host( out ) )
        res = np.empty( n )
        for b, p in enumerate( parts ):
            res[ p ] = M[ b, :len( p ) ]
        return loom.spread_host( res, self._ranks ) if self._ranks is not None else RealTensor( res )


def _host_values( x ):
    """the values of `x` on the host, in float64, whatever traces it ( a gradient: the split of the boxes is a constant )"""
    s = loom.spread_of( x )
    v = getattr( x, "value", x )
    if s is None:
        v = loom.ops( v ).stop_gradient( v )
        if hasattr( v, "primal" ):          # ( a Jax JVP tracer of a concrete value )
            v = v.primal
    return np.asarray( loom.gather_to_host( v if s is None else x ) ).astype( np.float64 )


def _native( a, like ):
    """the host array `a` as a value of the framework of `like`, its dtype kept"""
    a = np.asarray( a )
    return loom.ops( like ).array( a, Dtype.from_numpy( a.dtype ) )
