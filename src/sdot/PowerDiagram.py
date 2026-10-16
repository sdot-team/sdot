"""Power diagram ( Laguerre ) -- THE VIEW, and what all the storages have in common.

The cell of seed `i` is where its POWER DISTANCE wins:

    |x - d_i|² - w_i  <=  |x - d_j|² - w_j   for every other j

Expanded, the inequality loses its `|x|²` on both sides and becomes a half-space: a power
diagram costs exactly what a Voronoi diagram costs, one plane per rival and the same cut. Only the
DIFFERENCES of weights reach the planes: "all equal" and "no weights at all" are the same
object, and the Euclidean case is called `Voronoi` ( see `Voronoi.py` ).

`PowerDiagram( positions, weights, ... )` does not carry a diagram: it carries its SEEDS and the
convex domain that bounds them, and rebuilds what it is asked for, cell by cell, in the
scratch of a work-item ( `diagram/Ops.h` ). This file is the CONTRACT -- what a user reads and
writes: `positions`, `weights`, `measures`, `cells`, `cell( i )` -- and the common trunk: the domain,
the distribution, the scratch, the three kernels. HOW the seeds are laid out is up to a
specialization, chosen at construction:

  * `PowerDiagram_Plain` -- the seeds as they came, each cell cut by the `n - 1`
    bisectors. The floor, and what remains when the positions are a tracer;
  * `PowerDiagram_Bsp`   -- the seeds in the order of a BSP tree ( `AaBsp` ), a leaf being
    read in one piece, and each cell cut only by the seeds the tree could not
    rule out. The default as soon as the positions are concrete.

The neighborhood is ACCELERABLE, not the result: an accelerator can only silence cuts that
would have removed nothing, so the cells are the SAME, up to rounding errors.
"""

import numpy as np

import loom
from loom.compilation.FfiCode import FfiCode
import loom
from loom.tensor import Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar, Tensor, new_batch_axis
from loom.util import Aggregate

from .Cell import BOUNDARY, Cell
from .CellScratch import CellScratch, fp_size
from .StartCell import start_cell
from .viz.Displayable import Displayable


def diagram_class_for( positions, weights, accelerator ):
    """The specialization that lays out these seeds: the BSP tree unless it is not wanted
    ( `accelerator = "plain"` ) or one cannot be built here -- TRACED seeds, positions
    or weights: the tree is built host-side, and under a `jit` even a constant comes out of a
    kernel traced. Whoever wants the tree under a trace builds it outside and passes it ( `accelerator = tree` )."""
    from .PowerDiagram_Bsp import PowerDiagram_Bsp
    from .PowerDiagram_Plain import PowerDiagram_Plain
    if accelerator == "plain":
        return PowerDiagram_Plain
    if accelerator is None and any( loom.is_traced( getattr( x, "raw", x ) ) for x in ( positions, weights ) if x is not None ):
        return PowerDiagram_Plain
    return PowerDiagram_Bsp


def _engine_unit( dim, seeds ):
    """The cell engine for these seeds, as a `loom.FfiModule` (`sdot/engine/Cells.h` declares `engine_run`; both floats of the
    kernel are instantiated: the float of the cells is that of the call's domain, which the seeds do not know)."""
    inst = "".join( f'template LOOM_EXPORT int engine_run<LocalFor<{ dim },{ tk }>,{ seeds }>( LocalFor<{ dim },{ tk }> &, const { seeds } &, SI );\n'
                    for tk in ( "float", "double" ) )
    return loom.FfiModule(
        "sdot_engine", includes = [ "sdot/engine/Cells.h", "sdot/cell/Engine.h" ],
        source = ( "namespace sdot { namespace engine {\n"
                   "template<class Local,class Seeds>\n"
                   "int engine_run( Local &cell, const Seeds &seeds, SI k0 ) {\n"
                   "    using TK = typename Local::TKernel;\n"
                   "    auto f = seeds.template provider<TK>( k0 );\n"
                   "    return run<true>( cell, f );\n"
                   "}\n" + inst + "} }\n" ) )



def _on_item( code ):
    """`code` with every value of the call taken at `batch_index`: `inputs.x` becomes `inputs.x( batch_index )`, and so on for the
    outputs and the cotangents. A value selects the axes of `batch_index` it has and lets the others through, so without a `vmap`
    nothing changes -- and under one ( a batch of seeds, of densities, or only of COTANGENTS: `jax.jacrev`, a `vmap` of a `vjp` ),
    each item reads and writes its own slice, while the work-items of the sweep ( the thread axis ) share them."""
    import re
    return re.sub( r"\b(inputs|outputs|grad_of_outputs|grad_of_inputs)\.(\w+)(?!\w|\s*\()", r"\1.\2( batch_index )", code )

class PowerDiagram( Aggregate, Displayable ):
    # ---- what a specialization provides -------------------------------------------------------------
    #
    #   _init_seeds( positions, weights, accelerator )   lays out the seeds in its tensors
    #   positions / weights            properties, readable and writable, in the USER's ORDER
    #   _ranks_of_items()              for `cells`: the rank ( storage order ) of seed `i`
    #
    # and C++ side ( `diagram/Ops.h` ): `point( k )`, `weight( k )`, `user_id( k )`, `provider( k0 )`.

    # THE STARTING CELL, when a better one than "all of space" is known: the box
    # `box_min <= x <= box_max`, laid down in one go by `Cell.init_as_hypercube`. Absent ( `Unbound` ),
    # each cell is born as an unbounded REPLACEMENT SIMPLEX whose every cut must first
    # push back the infinite planes. This is not a second domain: it is the same one, expressed in
    # the form that can be laid down directly; what a box does not say stays in `bnd_*`.
    box_min        : RealTensor[ "dim" ]
    box_max        : RealTensor[ "dim" ]

    # the domain: a list of half-spaces, hence any polyhedral convex set. Absent, the
    # cells that run off to infinity stay there -- and are measured as such ( `TF::max` ).
    bnd_directions : RealTensor[ "num_boundary", "dim" ]
    bnd_offsets    : RealTensor[ "num_boundary" ]

    # THE CELL THE CUTS START FROM, when the domain is a convex polytope rather than a box ( `StartCell.py` ): its vertices
    # and, in 3D, their topology ( three facets and three neighbours each ). The facets are then `bnd_directions` /
    # `bnd_offsets`, in the order of the cuts' identifiers, and `box_min` / `box_max` the box of the vertices -- the scale.
    # Absent, the cells start from the box and `bnd_*` are cuts made on it.
    start_vertices : RealTensor[ "num_start", "dim" ]
    start_topo     : IntTensor[ "num_start", "num_topo", dict( size = 32 ) ]     # ( a few hundred vertices: no reason for 64 bits )

    # PERIODICITY ( `periodicity = ...` ): the translations of the seeds' IMAGES, row 0 being zero. Image `t` of seed `j`
    # is the identifier `j + n t` ( `diagram/Common.h` ). Absent ( `Unbound` ): no images, and nothing of it at compile time.
    periodic_shifts : RealTensor[ "num_image", "dim" ]
    # ... and their LINEAR part, when the group has rotations or symmetries: image `t` of `x` is `periodic_linear( t ) x +
    # periodic_shifts( t )`. Absent: translations only.
    periodic_linear : RealTensor[ "num_image", "dim", "dir" ]

    # A SINK ( partial transport, `sink_weight = ...` ): the cell of seed `i` is integrated inside the ball `|x - p_i|² <= w_i -
    # sink_weight` only ( `integration/InBall.h` ). Absent: the whole cell.
    sink_weight     : RealTensor

    num_start      : Axis[ "nb_starts" ]
    num_topo       : Axis[ "nb_topo" ]
    nb_starts      : ShapeVar
    nb_topo        : ShapeVar
    num_image      : Axis[ "nb_images" ]
    nb_images      : ShapeVar

    num_point      : Axis[ "nb_points" ]
    num_boundary   : Axis[ "nb_boundaries" ]
    dim            : Axis[ "nb_dims" ]
    dir            : Axis[ "nb_dims" ]

    nb_points      : ShapeVar
    nb_boundaries  : ShapeVar
    nb_dims        : CtShapeVar

    # GROUPS ( `groups = ...`, `groups.py` ): a TREE OF AGGREGATES, seeds that share one cell and split it among themselves. Per seed
    # ( storage order ): the smallest index of its root and the next member ( a cycle, storage ranks ), its aggregate ( or -1 ),
    # its virtual weight there, its unit direction in an aggregate of duplicates. Per aggregate ( a parent before its children ):
    # its parent, the centre and the scale of its frame, what its children's weights move outside ( 0: duplicates ), its own
    # virtual weight. Absent ( `Unbound` ): no group, and nothing of it at compile time.
    group_rep          : IntTensor[ "num_point", dict( size = 32 ) ]
    group_next         : IntTensor[ "num_point", dict( size = 32 ) ]
    group_parent       : IntTensor[ "num_point", dict( size = 32 ) ]
    group_vweights     : RealTensor[ "num_point" ]
    group_dirs         : RealTensor[ "num_point", "dim" ]
    group_agg_parent   : IntTensor[ "num_agg", dict( size = 32 ) ]
    group_agg_center   : RealTensor[ "num_agg", "dim" ]
    group_agg_scale    : RealTensor[ "num_agg" ]
    group_agg_outer    : RealTensor[ "num_agg" ]
    group_agg_vweights : RealTensor[ "num_agg" ]
    num_agg            : Axis[ "nb_aggs" ]
    nb_aggs            : ShapeVar

    def __new__( cls, positions = None, weights = None, *args, accelerator = None, **kwargs ):
        if cls is PowerDiagram:
            cls = diagram_class_for( positions, weights, accelerator )
        return super().__new__( cls )

    def __init__( self, positions, weights = None, boundaries = None, accelerator = None,
                  distribution = None, kernel_dtype = None, scratch_capacity = None, memory = None, periodicity = None,
                  sink_weight = None, groups = None ):
        """`positions`: `[ n, d ]`. `weights`: `[ n ]`, or nothing ( the Euclidean case ). The domain:

        - `boundaries = ( directions, offsets )` -- the half-spaces `direction . x <= offset`.
          A box is written `box_half_spaces( mi, ma )`, which is there for that;
        - nothing -- the boundary cells stay infinite.

        `accelerator`: `None` ( a BSP tree, built here ), an `AaBsp` already built on these positions
        ( what is needed to differentiate with respect to traced positions ), or `"plain"`. Without effect
        on the RESULT -- only on what it costs.

        `distribution`: WHAT to integrate against ( `Image`, `SumOfGaussians`, ... ). Absent,
        `measures` returns the volume of the cells; present, the integral of its density over them,
        NORMALIZED here once and for all. If it has a bounded SUPPORT, it is added to the domain.

        `kernel_dtype`: the float in which the geometry is cut ( `FP32` by default,
        `SDOT_KTYPE` to change the default ). `scratch_capacity`: for how many vertices per
        cell the scratch of a work-item is sized at the start -- a guess, which loom doubles
        on overflow.

        `periodicity`: the cells of the quotient of the space by a group of isometries ( `periodicity.py`: `True`, one
        boolean per axis, generators... ). The domain then bounds nothing along the periodic directions: it is the
        support of the distribution ( or `boundaries` ) that gives the period with `True`, and what the density repeats.
        The cells are those of the seeds AS GIVEN: `positions`, `cells` and `moments` are the given positions and the cells around
        them, whole ( not cut to the domain ), possibly outside it. Internally the seeds are folded onto the support, which keeps
        the images to walk few ( `unfold` takes a point of the frame of a folded seed back to that of its given position ).

        `sink_weight`: a SINK, the transport being PARTIAL. The cell of seed `i` stays its Laguerre cell, but what is integrated over it
        ( `measures`, `moments`, `hessian_rows` ) is the density inside the ball `|x - p_i|² <= w_i - sink_weight` only: the mass
        beyond goes to the sink. In 2D, with a density constant on pieces ( `Image`, `Box`, `Polygon`, or none ).

        `memory`: with the BSP tree, the diagram REMEMBERS, per seed, the neighbors of its cell
        at the last `measures` and proposes them first to the next one ( `ProviderBsp`, `MEMO` ) --
        which spares the transient cuts, a quarter to a third of the diagram in 3D, and remains
        exact whatever weights have moved in the meantime. `memory` is the capacity per
        seed ( 32 by default in 3D and above, 0 in 2D where it yields nothing measurable, and 0 in 3D on a CUDA card
        whose dedicated cells take the calls: they do not use it ); a seed that has more neighbors than that simply has
        no memory. `0` to turn it off.

        `groups`: seeds that SHARE ONE CELL and split it among themselves ( `groups.py` ) -- what makes exact duplicates, and
        seeds closer than the doubles can separate, well posed. A TREE OF AGGREGATES: a `Groups`, nested index lists ( `[ [ 0, 1 ],
        [ 2, [ 3, 4 ] ] ]` ), or `"auto"`, the separations of scales of the positions ( `groups.find_groups`: an aggregate is much
        smaller than its distance to the rest, the exact duplicates included -- on readable positions ). Seen from the other seeds, a
        member is an ordinary seed with its OUTER weight ( that of its site, the smallest index at its exact position: `weights`
        then reads these outer weights ); inside, two members are split by the bisector of their virtual diracs in the frame of
        their lowest common aggregate ( their true bisector, computed at the scale of the aggregate; for exact duplicates, unit
        directions spread by springs ), whose virtual weights come from the weights given ( `groups.virtual_weights` ) or from the
        tree itself ( a solver's ). Read back with `groups`. Not with a periodicity.
        """
        # the SUPPORT of the distribution bounds the domain, for free and without changing the
        # result: what lies beyond brings no mass. The caller's domain is INTERSECTED
        # with it, not replaced.
        if distribution is not None:
            # NORMALIZED here once and for all, with the dimension of the diagram: a distribution that does not carry one
            # ( `Box()` ) makes up its parameters from it
            pshape = positions.shape if hasattr( positions, "shape" ) else np.asarray( positions ).shape
            if len( pshape ) == 2:
                distribution = distribution.normalized_version( nb_dims = int( pshape[ 1 ] ) )
            support = distribution.bounding_half_spaces()
            if support is not None:
                if boundaries is None:
                    boundaries = support
                else:
                    boundaries = ( np.concatenate( [ np.asarray( boundaries[ 0 ], dtype = float ), support[ 0 ] ] ),
                                   np.concatenate( [ np.asarray( boundaries[ 1 ], dtype = float ), support[ 1 ] ] ) )

        # PERIODICITY: the seeds folded, the walls of the periodic directions replaced by the start box of the unfolded
        # cells, and the images ( `periodicity.py` )
        self._periodic = None
        if periodicity is not None and periodicity is not False:
            positions, boundaries = self._periodic_setup( positions, boundaries, periodicity )

        # no domain -> we do NOT name the two tensors: leaving them `Unbound` ( never
        # allocated, `NoneTensor` C++ side ) is not the same thing as passing them `None`.
        kwargs = {}
        if sink_weight is not None:
            self._check_sink( positions, distribution )
            kwargs[ "sink_weight" ] = RealTensor( getattr( sink_weight, "raw", sink_weight ) ) if loom.is_traced( getattr( sink_weight, "raw", sink_weight ) ) else float( sink_weight )
        if self._periodic is not None:
            kwargs[ "periodic_shifts" ] = self._periodic.shifts
            if self._periodic.linear is not None:
                kwargs[ "periodic_linear" ] = self._periodic.linear
        if boundaries is not None:
            # WHERE TO START FROM, read off the half-spaces themselves: those a box already expresses leave
            # the list -- they are `2d` of them and would come back on every cell ( 25 %, measured ).
            start_box = axis_aligned_box( *boundaries ) or traced_box( *boundaries )
            if start_box is None or start_box[ 2 ].any():
                # not just a box: a polytope. When it is a small simple one, THE POLYTOPE IS THE CELL the cuts start from
                # ( `StartCell.py` ); a big or degenerate one cuts the box of its vertices
                cell = start_cell( *boundaries )
                if cell is not None:
                    kwargs[ "box_min" ], kwargs[ "box_max" ] = cell[ "bounding_box" ]
                    if "vertices" in cell:
                        boundaries = ( cell[ "directions" ], cell[ "offsets" ] )
                        kwargs[ "start_vertices" ] = cell[ "vertices" ]
                        if cell[ "topology" ] is not None:
                            kwargs[ "start_topo" ] = cell[ "topology" ]
                    start_box = None
            if start_box is not None:
                mi, ma, kept = start_box
                kwargs[ "box_min" ], kwargs[ "box_max" ] = mi, ma
                boundaries = ( np.asarray( boundaries[ 0 ], dtype = float )[ kept ],
                               boundaries[ 1 ][ kept ] if loom.is_traced( boundaries[ 1 ] ) else np.asarray( boundaries[ 1 ], dtype = float )[ kept ] )
            if len( boundaries[ 1 ] ):
                kwargs[ "bnd_directions" ], kwargs[ "bnd_offsets" ] = boundaries

        pos = positions if hasattr( positions, "shape" ) else np.asarray( positions, dtype = float )
        if len( pos.shape ) != 2:
            raise ValueError( f"`positions` has to be [ n, d ] ( got { tuple( pos.shape ) } )" )
        n, d = int( pos.shape[ 0 ] ), int( pos.shape[ 1 ] )

        # GROUPS: the outer weights take the place of the weights for everything that does not look inside a group
        self._groups, group_members = None, None
        if groups is not None:
            weights, group_members = self._group_setup( groups, pos, weights, boundaries, distribution, sink_weight )

        self._kernel_dtype = kernel_dtype
        self._scratch_capacity = int( scratch_capacity or { 2: 64, 3: 128 }.get( d, 256 ) )
        self._memory = int( self._default_memory( d ) if memory is None else memory )
        self.__base_init__( nb_dims = d, nb_points = n, **self._init_seeds( pos, weights, accelerator ), **kwargs )
        if group_members is not None:
            for name, value in self._stored_group_members( group_members ).items():
                setattr( self, name, value )

        # NOT a field: the distribution is a CALL argument, normalized RIGHT HERE rather than at
        # each `measures` -- "the diagram integrates THIS measure" is a property of the object
        self.distribution = None
        if distribution is not None:
            dd = int( distribution.nb_dims.value )
            if dd != d:
                raise ValueError( f"the distribution lives in { dd }D, this diagram in { d }D" )
            self.distribution = distribution

    def _group_setup( self, groups, pos, weights, boundaries, distribution, sink_weight ):
        """`( outer weights, members in the user's order )` for `groups` ( `groups.resolve` ), and `self.groups`"""
        from . import groups as grp
        if self._periodic is not None:
            raise NotImplementedError( "groups: not with a periodicity yet" )
        if isinstance( groups, str ):
            if groups != "auto":
                raise ValueError( f"groups: a `Groups`, nested index lists, `'auto'` or `None` ( got { groups !r } )" )
            if loom.is_traced( getattr( pos, "raw", pos ) ):
                raise ValueError( "groups = 'auto': the positions must be readable here -- under a trace, give a `Groups`" )
            groups = grp.find_groups( pos )
            if groups is None:
                return weights, None
        else:
            groups = grp.as_groups( groups, pos )
        self._groups = groups
        outer, members = grp.resolve( groups, pos, weights )
        self._group_members_user = members               # ( the user's order: the oracle `cell( i )` and the display read them )
        return outer, members

    @property
    def groups( self ):
        """The `Groups` of this diagram ( `groups.py` ), in the user's order -- `None` without groups."""
        return getattr( self, "_groups", None )

    def _stored_group_members( self, members ):
        """the group members as the storage lays them out ( the user's order here; `PowerDiagram_Bsp` sorts them )"""
        return members

    def _given_virtual_weights( self ):
        """GROUPS: `( seed_v, agg_v )`, the virtual weights the diagram carries ( the seeds in the user's order ): host arrays, or
        the framework's under a trace ( a solve's outputs under `jax.jit` )"""
        from loom.tensor.host import to_host
        sv, av = self._user_order( self.group_vweights.raw ), self.group_agg_vweights.raw
        if loom.is_traced( sv ) or loom.is_traced( av ):
            return sv, av
        return np.asarray( to_host( sv ), dtype = np.float64 ).reshape( -1 ), np.asarray( to_host( av ), dtype = np.float64 ).reshape( -1 )

    def _user_order( self, stored ):
        """a value per seed in the storage's order, in the user's ( the same here; `PowerDiagram_Bsp` sorts )"""
        return stored

    def _grad_groups_expr( self ):
        """the cotangents of the group members, as `diagram::measures_bwd` takes them"""
        if not self.group_rep.is_defined:
            return "sdot::diagram::NoGroupGrads()"
        g = "grad_of_inputs.power_diagram"
        return f"sdot::diagram::group_grads( { g }.group_vweights, { g }.group_agg_vweights )"

    def _periodic_setup( self, positions, boundaries, periodicity ):
        """`( folded positions, boundaries )` for a periodic diagram, and `self._periodic` ( `periodicity.PeriodicSetup` )"""
        from .periodicity import setup_for
        raw = getattr( positions, "raw", positions )
        if not hasattr( raw, "shape" ):
            raw = np.asarray( raw, dtype = float )
        if len( raw.shape ) != 2:
            raise ValueError( f"`positions` has to be [ n, d ] ( got { tuple( raw.shape ) } )" )
        if boundaries is None:
            raise ValueError( "periodicity: a bounded domain is needed ( the support of the distribution, or `boundaries` ) -- "
                              "it is what the density repeats, and where the seeds are folded" )
        ps, folded = setup_for( raw, boundaries, periodicity )
        if ps is None:
            return positions, boundaries
        self._periodic = ps
        box = box_half_spaces( ps.start_lo, ps.start_hi )
        return folded, ( np.concatenate( [ ps.boundaries[ 0 ], box[ 0 ] ] ), np.concatenate( [ ps.boundaries[ 1 ], box[ 1 ] ] ) )

    @staticmethod
    def _check_sink( positions, distribution ):
        """what a sink ( partial transport ) can integrate today: 2D, a density constant on pieces"""
        pshape = positions.shape if hasattr( positions, "shape" ) else np.asarray( positions ).shape
        if len( pshape ) != 2 or int( pshape[ 1 ] ) != 2:
            raise NotImplementedError( "sink_weight ( partial transport ): in 2D only for now" )
        if distribution is not None and not getattr( distribution, "constant_on_pieces", False ):
            raise NotImplementedError( f"sink_weight ( partial transport ): a density constant on pieces only for now ( `Image`, `Box`, "
                                       f"`Polygon` ), not { type( distribution ).__name__ }" )

    # ---- the positions, as given ---------------------------------------------------------------------
    #
    # The storage holds the seeds FOLDED by a periodicity ( `_stored_positions` ): what `positions` reads is unfolded back to the
    # positions as given, what it is set to is folded first.

    def _given_positions( self, stored ):
        if self._periodic is None:
            return stored
        return RealTensor[ self.num_point, self.dim ]( self.unfold( stored.raw ) )

    def _folded_positions( self, positions ):
        """new positions for the storage: folded by the periodicity, whose folding maps follow ( a group that cannot bring them
        onto the region its cells were sized for -- a cone, seeds moved far -- needs a new diagram )"""
        pos = getattr( positions, "raw", positions )
        if not hasattr( pos, "shape" ):
            pos = np.asarray( pos, dtype = float )
        if self._periodic is None:
            return pos
        ps = self._periodic
        folded, M, c = ps.fold( pos )
        if not loom.is_traced( folded ) and not ps.translation_only:
            f = np.asarray( folded, dtype = float ).reshape( -1, self.dim_count )
            if ( ( f.min( axis = 0 ) < ps.region_lo - 1e-12 ) | ( f.max( axis = 0 ) > ps.region_hi + 1e-12 ) ).any():
                raise ValueError( "periodicity: these positions fold outside the region the cells of this diagram were sized for "
                                  "( a group without translations in every direction ): build a new `PowerDiagram`" )
        ps.fold_M, ps.fold_c = M, c
        return folded

    def _unfold_maps( self ):
        """`[ n, d, d + 1 ]` ( numpy ): per seed, the map from the frame of its folded seed to that of its given position"""
        ps = self._periodic
        n, d = int( self.nb_points.value ), self.dim_count
        c = np.asarray( ps.fold_c, dtype = float ).reshape( n, d )
        M = np.broadcast_to( np.eye( d ), ( n, d, d ) ) if ps.fold_M is None else np.asarray( ps.fold_M, dtype = float ).reshape( n, d, d )
        Mt = np.transpose( M, ( 0, 2, 1 ) )
        return np.concatenate( [ Mt, - np.einsum( "nij,nj->ni", Mt, c )[ :, :, None ] ], axis = 2 )

    def periodicity_rings( self, nb_rings = 1 ):
        """The elements of the group of `periodicity` up to the ring `nb_rings`, as affine maps `[ m, d, d + 1 ]` ( `[ M | V ]`, the map
        `x -> M x + V` ), the identity first: ring 1 holds those whose image of the support touches the support. What a display
        repeats the cells by ( see `add_to_viz` ); the identity alone without a periodicity."""
        d = self.dim_count
        if self._periodic is None:
            return np.concatenate( [ np.eye( d ), np.zeros( ( d, 1 ) ) ], axis = 1 )[ None ]
        return self._periodic.ring( nb_rings )

    def _draw_balls( self, viz, cells, first_color, opacity ):
        """what a sink leaves to each cell: the cell inter its ball, in the cell's color ( see `add_to_viz` )"""
        pos = np.asarray( self.positions, dtype = float ).reshape( -1, 2 )
        wt = self.weights
        w = np.asarray( wt, dtype = float ).reshape( -1 ) if getattr( wt, "is_defined", True ) else np.zeros( len( pos ) )
        r2 = w - float( np.asarray( self.sink_weight, dtype = float ).reshape( -1 )[ 0 ] )
        for i in range( len( pos ) ):
            if not r2[ i ] > 0:
                continue
            poly = _disc_in_polygon( cells.vertices( i ), pos[ i ], np.sqrt( r2[ i ] ) )
            if len( poly ) < 3:
                continue
            col = viz.color_at( first_color + i )
            viz.add_faces( poly, [ list( range( len( poly ) ) ) ], color = col, opacity = opacity )
            viz.add_edges( poly, color = "#000000", opacity = 0.6, closed = True )

    def domain_outline( self ):
        """The support of the periodicity ( the domain of a diagram without one ) as half-spaces `( directions, offsets )`: what a
        display draws dashed ( `add_to_viz` ). `None` for an unbounded diagram."""
        if self._periodic is not None:
            return self._periodic.support
        if not self.bnd_directions.is_defined and not self.box_min.is_defined:
            return None
        dirs, offs = [], []
        if self.box_min.is_defined:
            bd, bo = box_half_spaces( np.asarray( self.box_min ).reshape( -1 ), np.asarray( self.box_max ).reshape( -1 ) )
            dirs.append( bd ); offs.append( bo )
        if self.bnd_directions.is_defined:
            dirs.append( np.asarray( self.bnd_directions ).reshape( -1, self.dim_count ) )
            offs.append( np.asarray( self.bnd_offsets ).reshape( -1 ) )
        return np.concatenate( dirs ), np.concatenate( offs )

    def unfold( self, points ):
        """`points` ( `[ n, d ]`, one per seed, e.g. the barycenters of the cells ) taken from the frame of the folded seeds
        ( `positions` ) back to that of the positions as they were given. The identity without a periodicity.

        = Internals

        Each seed was folded by an element of the group, `folded = M given + c` ( `periodicity.PeriodicSetup.fold` ): this is
        `M^T ( points - c )`, differentiable."""
        if self._periodic is None:
            return points
        return self._periodic.unfold( points, self._periodic.fold_M, self._periodic.fold_c )

    def _default_memory( self, d ):
        """the neighbour memories per seed when `memory` is not given ( see `__init__` )"""
        return 0 if d <= 2 else 32

    @property
    def dim_count( self ):
        return int( self.nb_dims.value )

    def ffi_units( self, call_arg ):
        """= Internals

        The cell engine as a SHARED LIBRARY ( `sdot/engine/Cells.h` ): every call that holds this diagram links the variant
        of `sdot_engine` module (`_engine_unit`) its seeds ask for, instead of instantiating the engine in its own kernel. Called by loom for the
        calls on the CPU ( the card has its own cells ), with the resolved types of the members. Says the same as
        `engine::has_unit` ( dimensions 2 and 3 ) and as `engine_seeds()` of the storages."""
        if not call_arg.device.is_cpu or self.dim_count not in ( 2, 3 ):
            return []
        m = call_arg.attributes
        bsp = "tree" in m

        def bound( name ):
            ca = m.get( name )
            return ca is not None and getattr( ca, "io_category", None ) is not None and ca.io_category.is_bound

        positions = m[ "sorted_positions" if bsp else "seed_positions" ]
        tr = "double" if positions.itemsize == 8 else "float"
        weights = int( bound( "sorted_weights" if bsp else "weights" ) )
        if bsp:
            ti = "SI" if m[ "tree" ].attributes[ "node_begin" ].itemsize == 8 else "int"
            seeds = f"BspSeeds<{ tr },{ ti },{ self.dim_count },bool( { weights } ),bool( { int( bound( 'memo_counts' ) ) } )>"
        else:
            seeds = f"PlainSeeds<{ tr },{ self.dim_count },bool( { weights } )>"
        return [ _engine_unit( self.dim_count, seeds ) ]

    @property
    def kernel_dtype( self ):
        return self._domain_cell().kernel_dtype

    # ---- the domain, and the distribution ----------------------------------------------------------

    def _domain_cell( self ):
        """The domain as a POLYTOPE, computed ONCE -- what each cell starts from. Built with
        the SAME code as the `cell( i )` oracle, which keeps the two descriptions of the domain
        literally identical. It is the type of THIS cell that decides, C++ side, the local
        shape in which each cell is built."""
        if getattr( self, "_dom_cell", None ) is None:
            self._dom_cell = self._start_cell()
            if self.bnd_directions.is_defined:
                # the backend's VALUES, not numpy's: under a `jit` the half-spaces may be
                # traced. `stop_gradient`: the domain is a constant of the problem ( its cuts
                # carry `BOUNDARY`, "not a seed", and have nowhere to send a derivative ).
                dirs = loom.ops().stop_gradient( self.bnd_directions.raw )
                offs = loom.ops().stop_gradient( self.bnd_offsets.raw )
                for b in range( int( self.bnd_directions.shape[ 0 ] ) ):
                    self._dom_cell.cut( dirs[ b ], offs[ b ], BOUNDARY )
        return self._dom_cell

    def _start_cell( self ):
        """the starting box, or all of space"""
        d = self.dim_count
        kw = dict( kernel_dtype = self._kernel_dtype )
        if self.box_min.is_defined:
            # the backend's values ( a box known only under a trace: `traced_box` ), detached -- the domain is a constant
            mi = loom.ops().stop_gradient( self.box_min.raw ).reshape( -1 )
            ma = loom.ops().stop_gradient( self.box_max.raw ).reshape( -1 )
            return Cell.make_hypercube( d, mi, np.eye( d ) * ( ma - mi )[ None, : ], **kw )
        return Cell.make_unbounded( d, **kw )

    def _dist_for( self, moment = None ):
        """How a call names its distribution: `( C++ expression, expression of its cotangent,
        kwargs of the call )`. Without a distribution, `unit_density()` -- a value that the C++ makes
        itself -- and a cotangent `0` that `UnitDensity` ignores. `moment`: the density multiplied by a
        polynomial ( see `moment` ), as a density -- `diagram/Moments.h`."""
        if self.distribution is None:
            expr, grad, kwargs = "inputs.power_diagram.unit_density()", "0", {}
        else:
            expr, grad, kwargs = "inputs.distribution", "grad_of_inputs.distribution", { "distribution": self.distribution }
            if self._periodic is not None:
                # the density and its translates ( `Periodic.h` ): the unfolded cells go beyond its support
                expr = f"sdot::periodic_dist( { expr }, inputs.power_diagram.periodic_shifts, inputs.power_diagram.periodic_linear )"
        if moment is not None:
            expr = f"sdot::diagram::moment_dist<{ int( moment ) }>( { expr } )"
        return expr, grad, kwargs

    # ---- the memory ( `PowerDiagram_Bsp` only ) -------------------------------------------------------

    def _memo_for_call( self ):
        """`( C++ expression of the two output tensors, arguments of the call, what has to be
        taken back afterwards )` -- nothing for a storage that has no memory"""
        return "0, 0", {}, None

    def _memo_after_call( self, produced ):
        pass

    # ---- a dedicated kernel for the card ( `PowerDiagram_Bsp` only ) ---------------------------------

    def _measures_on_card( self ):
        """the measures by a kernel written for the device, or `None`: the generic path takes the call"""
        return None

    def _affine_derivable( self ):
        """the integration derives the frame and the origin of the distribution ( an `Image`, `diagram::AffineAcc` ): not through a
        periodicity ( the translates ) nor in the balls of a partial transport"""
        from .distributions.Image import Image
        return isinstance( self.distribution, Image ) and self._periodic is None and not self.sink_weight.is_defined

    def _card_laplacian_solve( self, rhs, linear_solver ):
        """`laplacian_solve` on a card, or `None` when the card does not take this diagram"""
        return None

    # ---- the scratch -------------------------------------------------------------------------------

    def _scratch_words( self, cap, nb_cells, with_grad ):
        """What a work-item holds, in words: `nb_cells` local cells of `cap` vertices, and
        for the adjoint one cotangent per vertex -- THE SAME FORMULA as `diagram::words_for`."""
        dom = self._domain_cell()
        words = nb_cells * dom.scratch_words( cap, fp_size( dom.kernel_dtype ) )
        if with_grad:
            words += -( -self.dim_count * cap * 8 // 32 ) * 8
        return words

    def _nb_work_cells( self ):
        """a distribution that CUTS the cell asks for a second local cell"""
        return 2 if self.distribution is not None and getattr( self.distribution, "cuts_pieces", False ) else 1

    # ---- what is read ------------------------------------------------------------------------------

    @property
    def measures( self ) -> Tensor:
        """The measure of each cell, `[ n ]` -- its integral against `distribution` when there is one. Differentiable with respect to the positions, the weights and the parameters of the distribution.

        = Internals

        The measure of each cell: `[ n ]`, indexed like `positions`.

        One call, one sweep: each work-item builds a cell in its scratch,
        writes its volume, and starts over with the next seed. Nothing of the diagram is kept.
        With a `distribution`, it is the INTEGRAL of its density over the cell ( same sweep ).

        DIFFERENTIABLE with respect to the seeds, `positions` as well as `weights`, and with respect to the VALUES of
        the distribution ( `diagram::measures_bwd` redoes the same sweep ). The DOMAIN is a
        constant: a cut that comes from it carries a negative identifier, so its share goes nowhere.

        On a CUDA card, in 2D, a dedicated kernel may take the call instead ( `_measures_on_card`, see
        `PowerDiagram_Bsp` ): same cells, cell in registers.
        """
        return self._integral( None )

    def moment( self, k ) -> Tensor:
        """`int_{cell_i} p( x ) rho( x ) dx`, `[ n ]`, for `p = x_k` ( `k < d` ), `|x|^2` ( `k == d` ) or `1` ( `k == d + 1` ). Differentiable like `measures`.

        = Internals

        `int_{cell_i} p( x ) rho( x ) dx`, `[ n ]`, for the polynomial `p` that `k` names: `x_k` for `k < d`, `|x|^2` for `k == d`
        and `1` for `k == d + 1` ( the mass, by the SAME quadrature as the others -- not by the closed form of `measures` ).

        DIFFERENTIABLE like `measures`, with respect to the seeds, the weights and the parameters of the density -- the
        moment is a density, and the diagram already knows how to differentiate those ( `diagram/Moments.h` ). The integral
        and its adjoint are closed forms where the density has them ( constant pieces, a `Mesh`, gaussians in 2D and 3D ), the
        adaptive quadrature of `PointwiseDensity` elsewhere. What a transport cost and its barycenters are made of ( `SdotPlanNd` )."""
        d = self.dim_count
        if not 0 <= int( k ) <= d + 1:
            raise ValueError( f"moment( { k } ): 0 .. { d - 1 } ( x_k ), { d } ( |x|^2 ) or { d + 1 } ( 1 )" )
        return self._integral( int( k ) )

    def _integral( self, moment ):
        """`measures` ( `moment = None` ) and `moment`: one sweep, the cell integrated against the density or against a moment of it"""
        if moment is None:
            res = self._measures_on_card()
            if res is not None:
                return res

        dom = self._domain_cell()
        # the budget that decides parallelism: what ONE work-item holds -- its scratch, sized
        # for the backward from the forward on ( it redoes the sweep on a scratch of the same shape )
        nb_words = self._scratch_words( self._scratch_capacity, self._nb_work_cells(), True )
        kp = loom.kernel_parameters_for( "power_diagram_moment" if moment is not None else "power_diagram_measures" )
        nt = kp.nb_threads( nb_local_bytes_per_thread = 4 * nb_words, batch_axes = [ self.num_point ] )

        # the work-item axis is a BATCH axis carried by the scratch: `thread_index` /
        # `nb_threads` are the rank of this work-item and their number, the strided loop reads off them
        num_thread = new_batch_axis( nt, prefix = "thread" )
        res = RealTensor[ self.num_point ]()
        dist_expr, grad_dist_expr, dist_kwargs = self._dist_for( moment )
        memo_expr, memo_args, memo_produced = self._memo_for_call()

        # the rank of a work-item in the sweep, read off its OWN axis: `thread_index` / `nb_threads` count every item launched, and
        # under a `vmap` those are the work-items of all the batch -- each item of which must sweep all the seeds
        rank = "SI( inputs.thread_rank( batch_index ) ), SI( inputs.thread_rank.shape( 0 ) )"
        kp.ffi_call(
            FfiCode.per_item( code = _on_item( "inputs.power_diagram.measures( outputs.res, inputs.dom_cell, scratch.pool( batch_index ), "
                           f"{ dist_expr }, { memo_expr }, " ) + rank + " );",
                # the gradients on the seeds are SHARED by all items: each work-item
                # accumulates into them ( `atomic_add` C++ side ), and the platform zeroes them before the body
                ),
            FfiCode.per_item( _on_item( "inputs.power_diagram.measures_bwd( outputs.res, inputs.dom_cell, grad_of_outputs.res, "
                           f"{ self._grad_seeds_expr() }, { self._grad_groups_expr() }, "
                           f"scratch.pool( batch_index ), { dist_expr }, { grad_dist_expr }, " ) + rank + " );",
                # what this adjoint derives: the seeds, as the storage names them, and the density's own
                # parameters, as IT knows them ( loom refuses a differentiation that needs anything else )
                computed_grads = "inputs.power_diagram.measures_computed_grads( computed_grads.from_res.power_diagram );"
                    + ( "" if self.distribution is None else
                        " inputs.distribution.computed_grads( computed_grads.from_res.distribution );" )
                    # the frame and the origin of an image ( `diagram::AffineAcc` ): not through a periodicity nor in balls
                    + ( " DECAYED_TYPE_OF( inputs.distribution )::affine_computed_grads( computed_grads.from_res.distribution );"
                        if self._affine_derivable() else "" ) ),
            power_diagram = self,
            dom_cell = dom,
            res = loom.out( res ),
            pool = CellScratch.for_call( nb_words, dom.kernel_dtype, batch_axes = [ num_thread ] ),
            thread_rank = IntTensor[ num_thread ]( np.arange( nt, dtype = np.int32 ) ),
            **dist_kwargs,
            **memo_args,
        )
        self._memo_after_call( memo_produced )
        return res

    def laplacian_solve( self, rhs, linear_solver = "auto" ) -> Tensor:
        """The solution `x` of `L x = rhs`, `L` being the derivative of the measures with respect to the weights ( `rhs` sums to zero; `x[ 0 ] == 0` ). Differentiable with respect to `rhs`.

        = Internals

        `x = L^+ rhs`: the solution, with `x[ 0 ] == 0` ( the gauge ), of the system of the LAPLACIAN of the Laguerre graph
        of this diagram -- the hessian of the dual functional of a transport, `d m_i / d w_j`, with the sum of each row
        at zero ( `rhs` must therefore sum to zero too ). `[ n ]`, indexed like `positions`.

        It is the one linear solve that the derivative of a transport asks for ( `SdotPlanNd`: the implicit function theorem ),
        and the reason it is here: DIFFERENTIABLE with respect to `rhs` -- `L` being symmetric, its adjoint is the same solve --
        and with respect to NOTHING ELSE: the diagram is a constant of the call ( the derivatives of `L` are not
        written ). The density must know how to integrate itself on a facet ( see `hessian_rows` ). On a card, the card's diagram,
        laplacian and linear solver ( `gpu/Newton2D.cuh::laplacian_solve` ), for what the card's solve takes.

        With GROUPS, the laplacian of the ROOTS: the facets between root aggregates, `rhs` summed over each root, `x` the same for
        all its members -- the outer level of the solve ( `sdotplan/Groups.h` ); the inner levels are `groups.TreeSystem`'s."""
        from .SdotPlanNd import _LIN
        if linear_solver not in _LIN:
            raise ValueError( f"unknown linear_solver: { linear_solver !r } ( { ', '.join( _LIN ) } )" )
        if getattr( loom.resolved_device(), "is_cuda_gpu", False ) or not loom.resolved_device().is_cpu:
            # on a card: the card's diagram, laplacian and linear solver ( `PowerDiagram_Bsp._card_laplacian_solve` )
            res = self._card_laplacian_solve( rhs, linear_solver )
            if res is None:
                raise NotImplementedError( "laplacian_solve on a card: the card's cells take a box domain, the BSP tree, a constant, an image, "
                                           "gaussians or a mesh, and no sink ( partial transport ) nor rotating periodicity -- the CPU for the rest" )
            return res
        dom = self._domain_cell()
        dist_expr, _, dist_kwargs = self._dist_for()
        tk = "double" if fp_size( dom.kernel_dtype ) == 64 else "float"
        x = RealTensor[ self.num_point ]()
        b = RealTensor[ self.num_point ]( getattr( rhs, "raw", rhs ) )
        code = lambda src, dst, backward = False: FfiCode.inline( includes = [ "sdot/sdotplan/Adjoint.h" ], sources = [ "sdot/sdotplan/Linear.cpp" ], code = "\n".join( [
            "auto &inputs = args.inputs; auto &outputs = args.outputs;",
            *( [ "auto &grad_of_inputs = args.grad_of_inputs; auto &grad_of_outputs = args.grad_of_outputs;" ] if backward else [] ),
            # ( `batch_axes`: a `vmap` of the right-hand side -- the diagram is the same for all, and is swept once )
            f"sdotplan::laplacian_solve<{ tk }>( queue, batch_axes, inputs.power_diagram, inputs.dom_cell, { dist_expr }, { src }, { _LIN[ linear_solver ] }, { int( self._scratch_capacity ) }, { dst } );",
            # the diagram is a constant of the call: only `rhs` is derived ( loom refuses the rest )
            *( [ "args.computed_grads.from_x.rhs = true;" ] if backward else [] ) ] ) )
        loom.ffi_call(
            "power_diagram_laplacian_solve",
            code( "inputs.rhs", "outputs.x" ),
            # the adjoint of `rhs -> x` is itself: the cotangent of `x` in, the one of `rhs` out
            code( "grad_of_outputs.x", "grad_of_inputs.rhs", backward = True ),
            power_diagram = self,
            dom_cell = dom,
            rhs = b,
            x = loom.out( x ),
            has_dynamic_capacity = False,
            **dist_kwargs,
        )
        return x

    @property
    def moments( self ):
        """`( masses, first, second )`: for each cell, `int rho`, `int x rho` ( `[ n, d ]` ) and `int |x|^2 rho`. Not differentiable ( see `moment` ).

        = Internals

        `( masses, first, second )`: for each cell, `int rho`, `int x rho` ( `[ n, d ]` ) and
        `int |x|^2 rho` -- enough to write a TRANSPORT COST, `sum_i int_{cell_i} |x - p_i|^2 rho
        = second - 2 p . first + |p|^2 mass`, and the barycenters `first / mass`. Same sweep as
        `measures`, on a piecewise-constant distribution ( `Image`, or nothing ) only.
        NOT differentiable: a transport cost is differentiated by the envelope theorem, at the
        adjusted weights -- `2 mass_i ( p_i - b_i )` -- which `SdotPlanNd` does by itself."""
        dom = self._domain_cell()
        nb_words = self._scratch_words( self._scratch_capacity, self._nb_work_cells(), False )
        kp = loom.kernel_parameters_for( "power_diagram_moments" )
        nt = kp.nb_threads( nb_local_bytes_per_thread = 4 * nb_words, batch_axes = [ self.num_point ] )
        num_thread = new_batch_axis( nt, prefix = "thread" )
        mass = RealTensor[ self.num_point ]()
        first = RealTensor[ self.num_point, self.dim ]()
        second = RealTensor[ self.num_point ]()
        dist_expr, _, dist_kwargs = self._dist_for()

        kp.ffi_call(
            FfiCode.per_item( code = "inputs.power_diagram.moments( outputs.mass, outputs.first, outputs.second, "
                           f"inputs.dom_cell, scratch.pool( batch_index ), { dist_expr }, thread_index, nb_threads );" ),
            power_diagram = self,
            dom_cell = dom,
            mass = loom.out( mass ), first = loom.out( first ), second = loom.out( second ),
            pool = CellScratch.for_call( nb_words, dom.kernel_dtype, batch_axes = [ num_thread ] ),
            **dist_kwargs,
        )
        if self._periodic is not None:
            # around the folded seeds: taken to the frame of the given positions, `x -> M^T ( x - c )`
            a = self._unfold_maps()
            m = np.asarray( mass.value, dtype = float ).reshape( -1 )
            f = np.asarray( first.value, dtype = float ).reshape( len( m ), -1 )
            q = np.asarray( second.value, dtype = float ).reshape( -1 )
            d = f.shape[ 1 ]
            f2 = np.einsum( "nij,nj->ni", a[ :, :, :d ], f ) + m[ :, None ] * a[ :, :, d ]
            q2 = q + 2 * ( f * np.einsum( "nji,nj->ni", a[ :, :, :d ], a[ :, :, d ] ) ).sum( axis = 1 ) + m * ( a[ :, :, d ] ** 2 ).sum( axis = 1 )
            first = RealTensor[ self.num_point, self.dim ]( f2 )
            second = RealTensor[ self.num_point ]( q2 )
        return mass, first, second

    def hessian_rows( self, items = None, host = True ):
        """The derivatives of the measures with respect to the weights, row by row: `( nb_nbrs, ids, vals )`, the neighbors of each cell and the coefficients ( host arrays ).

        = Internals

        `( nb_nbrs, ids, vals )`: for each cell `i`, its neighbors `j` ( `ids[ i, :nb_nbrs[ i ] ]`,
        indexed like `positions`; negative for the domain, to be ignored ) and `vals[ i, r ] =
        int_{facet ij} rho / ( 2 | p_i - p_j | )` -- enough to assemble the JACOBIAN of the measures with respect
        to the weights, `d m_i / d w_j = - vals`, `d m_i / d w_i = + sum_j vals`, which is also the
        Hessian of the dual functional of a transport ( `SdotPlanNd`, `objective = "newton"` ).
        Host arrays. A piecewise-constant distribution only. One call batched over the
        cells, like `cells` ( the number of neighbors per cell has a capacity that loom doubles ). `items`: the rows of these
        seeds only ( user indices, in that order ), the others not built. `host = False`: the backend's arrays instead, `[ n ]`,
        `[ n, cap ]`, `[ n, cap ]` -- what follows a `jax.jit` ( the entries past `nb_nbrs[ i ]` are to be ignored ).

        With GROUPS, each facet's value is the derivative with respect to ITS OWN weights ( `diagram::hessian_row` ): the outer
        weights between root aggregates, and between two members of one the weights `V` in the frame of their lowest common
        aggregate ( `vals = int_{facet} rho / ( 2 | q_i - q_j | / s )`, their virtual diracs `q` in that frame of scale `s` )."""
        d = self.dim_count
        all_ranks = np.asarray( self._ranks_of_items() ).reshape( -1 )
        sel = all_ranks if items is None else all_ranks[ np.asarray( items, dtype = np.int64 ).reshape( -1 ) ]
        n = len( sel )
        num_cell = new_batch_axis( n, prefix = "cell" )
        ranks = IntTensor[ num_cell ]( sel )

        dom = self._domain_cell()
        cap = self._scratch_capacity
        nbrs = Neighbors( batch_axes = [ num_cell ] )

        nb_words = self._scratch_words( cap, self._nb_work_cells(), False )
        nt = loom.resolved_device().nb_threads( nb_local_bytes_per_thread = 4 * nb_words, batch_axes = [ num_cell ] )
        dist_expr, _, dist_kwargs = self._dist_for()

        loom.ffi_call(
            "power_diagram_hessian_rows",
            FfiCode.per_item( code = "inputs.power_diagram.hessian_row( SI( inputs.ranks( batch_index ) ), inputs.dom_cell, "
                           f"outputs.nbrs( batch_index ), scratch.pool, thread_index, { dist_expr } );",
                max_nb_threads = "return scratch.pool.words.shape( 0 );" ),
            power_diagram = self,
            dom_cell = dom,
            ranks = ranks,
            pool = CellScratch.for_call( nb_words, dom.kernel_dtype, nb_threads = nt ),
            nbrs = loom.out( nbrs, capacities = { "nb_nbrs": 16 } ),
            **dist_kwargs,
        )
        if not host:
            # ( the raw buffers may hold more rows than the cells: a device rounds its batches up )
            raw = lambda x: x.reshape( x.shape[ 0 ], -1 )[ :n ]
            return nbrs.nb_nbrs.as_tensor().raw.reshape( -1 )[ :n ], raw( nbrs.ids.raw ), raw( nbrs.vals.raw )
        counts = np.asarray( nbrs.nb_nbrs.value ).reshape( -1 ).astype( int )
        ids = np.asarray( nbrs.ids ).reshape( n, -1 )
        vals = np.asarray( nbrs.vals ).reshape( n, -1 )
        return counts, ids, vals

    @property
    def cells( self ) -> Cell:
        """All the cells, as one `Cell` batched over the seeds ( in the order of `positions` ) -- to draw or inspect them. With a
        periodicity, each cell is the whole cell of its seed, around the seed as it was given: it can lie outside the domain,
        and away from the others ( `add_to_viz` shows its copies by the group too ).

        = Internals

        ALL the cells, in ONE call: a `Cell` batched over the seeds, in the order of
        `positions`, its `cut_ids` designating the seeds in that same order.

        The query that does not reduce a cell to a number, hence the only one whose memory is
        a function of the number of seeds -- which is what a DISPLAY is. The scratch, for its part, stays PER
        WORK-ITEM ( `max_nb_threads` ). The returned `Cell` is drawn as is.
        """
        n, d = int( self.nb_points.value ), self.dim_count
        num_cell = new_batch_axis( n, prefix = "cell" )
        ranks = IntTensor[ num_cell ]( self._ranks_of_items() )

        dom = self._domain_cell()
        cap = self._scratch_capacity
        cells = Cell( d, init_as_unbounded = False, batch_axes = [ num_cell ], kernel_dtype = dom.kernel_dtype )

        nb_words = self._scratch_words( cap, 1, False )
        nt = loom.resolved_device().nb_threads( nb_local_bytes_per_thread = 4 * nb_words, batch_axes = [ num_cell ] )
        loom.ffi_call(
            "power_diagram_cells",
            FfiCode.per_item( code = "inputs.power_diagram.build_cell( SI( inputs.ranks( batch_index ) ), inputs.dom_cell, "
                           "outputs.cells( batch_index ), scratch.pool, thread_index );",
                max_nb_threads = "return scratch.pool.words.shape( 0 );" ),
            power_diagram = self,
            dom_cell = dom,
            ranks = ranks,
            pool = CellScratch.for_call( nb_words, dom.kernel_dtype, nb_threads = nt ),
            cells = loom.out( cells, capacities = { "nb_vertices": cap, "nb_cuts": cap } ),
        )
        if self._periodic is not None:                   # built around the folded seeds: moved next to the given positions
            cells = cells.moved( self._unfold_maps() )
        return cells

    def support_pieces( self, threshold = 0.0 ):
        """= Internals

        `( pieces, owners )`: the cells seen ONLY WHERE THE DENSITY HAS MASS -- `pieces` a `Cell` batched over the pieces
        ( `None` if there is none ), piece `p` being a part of the cell of seed `owners[ p ]`.

        The distribution describes where it has mass as convex BLOCKS ( `Distribution.display_blocks`: the boxes of tiles of an
        `Image`, the simplices of a `Mesh` ) and says which of them each cell meets ( `blocks_of_cells`, host side, on the boxes
        of `cells` ): a cell that lies where the density has mass is a piece of its own, the others are cut to each block they
        meet -- in ONE call over the pieces ( `diagram::build_support_piece` ). The walls of a block carry `SEAM` when the
        region behind has mass too ( an inner seam: `cell_viz` draws neither the face nor its edges ) and `SUPPORT` otherwise.

        `threshold`: the density has mass where it is `> threshold * max`. A distribution that does not say where it has mass
        ( `display_blocks` gives `None` ) gives the cells themselves."""
        dist = self.distribution
        blocks = None if dist is None else dist.display_blocks( threshold )
        if blocks is None:
            return self.cells, np.arange( int( self.nb_points.value ) )

        owners, which = dist.blocks_of_cells( self.cells, threshold )
        if len( owners ) == 0:
            return None, owners                    # no cell meets a region with mass
        d = self.dim_count
        dirs, offs, ids = blocks
        num_piece = new_batch_axis( len( owners ), prefix = "piece" )
        num_block = Axis( ShapeVar( dirs.shape[ 0 ] ), name = "num_block" )
        num_wall = Axis( ShapeVar( dirs.shape[ 1 ] ), name = "num_wall" )
        ranks = IntTensor[ num_piece ]( np.asarray( self._ranks_of_items() ).reshape( -1 )[ owners ] )

        dom = self._domain_cell()
        cap = self._scratch_capacity
        pieces = Cell( d, init_as_unbounded = False, batch_axes = [ num_piece ], kernel_dtype = dom.kernel_dtype )

        nb_words = self._scratch_words( cap, 1, False )
        nt = loom.resolved_device().nb_threads( nb_local_bytes_per_thread = 4 * nb_words, batch_axes = [ num_piece ] )
        loom.ffi_call(
            "power_diagram_support_pieces",
            FfiCode.per_item( code = "inputs.power_diagram.build_support_piece( SI( inputs.ranks( batch_index ) ), SI( inputs.blocks( batch_index ) ), "
                           "inputs.block_dirs, inputs.block_offs, inputs.block_ids, inputs.dom_cell, outputs.pieces( batch_index ), "
                           "scratch.pool, thread_index );",
                max_nb_threads = "return scratch.pool.words.shape( 0 );" ),
            power_diagram = self,
            dom_cell = dom,
            ranks = ranks,
            blocks = IntTensor[ num_piece, dict( size = 32 ) ]( np.asarray( which, np.int32 ) ),
            block_dirs = RealTensor[ num_block, num_wall, self.dim ]( dirs ),
            block_offs = RealTensor[ num_block, num_wall ]( offs ),
            block_ids = IntTensor[ num_block, num_wall, dict( size = 32 ) ]( ids ),
            pool = CellScratch.for_call( nb_words, dom.kernel_dtype, nb_threads = nt ),
            pieces = loom.out( pieces, capacities = { "nb_vertices": cap, "nb_cuts": cap + dirs.shape[ 1 ] } ),
        )
        if self._periodic is not None:
            pieces = pieces.moved( self._unfold_maps()[ np.asarray( owners ).reshape( -1 ) ] )
        return pieces, owners

    def cell( self, i ) -> Cell:
        """= Internals

        The cell of seed `i`, built PYTHON SIDE -- one `loom.ffi_call` per cut.

        The slow path, and deliberately so: the same geometry obtained by an orchestration
        entirely different from the kernel's, hence the tests' ORACLE. It is NOT the display
        path ( `n²` round trips ), see `cells`.
        """
        if self._periodic is not None:
            raise NotImplementedError( "cell( i ) is the oracle of a diagram without periodicity: `cells` gives the periodic ones" )
        d = self.dim_count
        pos = np.asarray( self.positions ).reshape( -1, d )
        w = np.asarray( self.weights ).reshape( -1 ) if self.weights.is_defined else None

        res = self._start_cell()
        if self.bnd_directions.is_defined:
            bds = np.asarray( self.bnd_directions ).reshape( -1, d )
            bos = np.asarray( self.bnd_offsets ).reshape( -1 )
            for b in range( len( bds ) ):
                res.cut( bds[ b ], float( bos[ b ] ), BOUNDARY )

        # GROUPS: the planes inside a group, the same rules as `diagram/Common.h::group_plane`
        g = self._groups if self.group_rep.is_defined else None
        if g is not None:
            gm = self._group_members_user
            sv = _host_array( gm[ "group_vweights" ] ).reshape( -1 )
            av = _host_array( gm[ "group_agg_vweights" ] ).reshape( -1 )

        p0 = pos[ i ]
        for j in range( len( pos ) ):
            if j == i:
                continue
            if g is not None and g.rep[ j ] == g.rep[ i ]:
                direction, offset = _group_plane( g, pos, sv, av, i, j )
                res.cut( direction, offset, j )
                continue
            direction = pos[ j ] - p0
            offset = float( direction @ ( p0 + pos[ j ] ) / 2 )
            if w is not None:
                offset += float( w[ i ] - w[ j ] ) / 2
            res.cut( direction, offset, j )
        return res

    def add_to_viz( self, viz, seeds = True, seed_color = "#000000", support_only = None, threshold = 0.0,
                    nb_periodicity_rings = 1, opacity_periodicity_rings = 0.25, fit_periodicity_rings = False, domain = None,
                    domain_color = "#000000", opacity_beyond_balls = 0.2, virtual_seeds = False, virtual_color = "#888888", **kwargs ):
        """Draws itself into a `Visualizer`: all the cells, in one call ( see `cells` ), and the
        seeds as points when `seeds` ( `seed_color` ). `kwargs` are those of `Cell.add_to_viz`
        ( `opacity`, `faces`, `edges`, ... ). The quick ways to look at it -- `write_html`,
        `write_pvd`, `show` -- are those of `Displayable`.

        `support_only`: the cells only where the distribution has mass ( `support_pieces`, with
        `threshold` ) -- an image with zero pixels does not show its cells there. By default: yes without a periodicity, no with
        one ( a periodic cell is drawn whole, around its seed ).

        With a periodicity, the cells are drawn again moved by every element of the group up to the ring `nb_periodicity_rings`
        ( `periodicity_rings` ), fainter ( `opacity_periodicity_rings`; 0 draws no copy ), the same seed in the same color: the
        copies must glue without a seam. They do not count in the scale of the scene ( the zoom fits the cells themselves ) unless
        `fit_periodicity_rings`. `domain`: the outline of
        the domain ( the support that the group repeats ), dashed, and that of its images -- by default with a periodicity only.

        With a sink ( partial transport, `sink_weight` ), the cells are drawn whole, as they are computed, but fainter
        ( `opacity_beyond_balls` ): what is integrated -- the part of each cell inside its ball -- is drawn on top, in the same
        color, its arc approximated by a polygon.

        With GROUPS ( `groups` ), the cells drawn are those of the members, which split the cell of their group; `virtual_seeds`
        adds the virtual diracs of the exact duplicates ( `virtual_color` ), the points their split is the power diagram of."""
        n, d = int( self.nb_points.value ), self.dim_count
        if support_only is None:
            support_only = self._periodic is None
        if support_only:
            shown, owners = self.support_pieces( threshold )
        else:
            shown, owners = self.cells, None
        maps = self.periodicity_rings( nb_periodicity_rings if opacity_periodicity_rings > 0 else 0 )
        if domain is None:
            domain = self._periodic is not None
        outline = self.domain_outline() if domain else None
        balls = self.sink_weight.is_defined and d == 2 and shown is not None
        first_color = viz.reserve_colors( 0 )          # ( the cells reserve their colors from there: the balls take the same )
        if balls:
            kwargs = dict( kwargs )
            kwargs[ "opacity" ] = opacity_beyond_balls * kwargs.get( "opacity", 1.0 )
            kwargs[ "edge_opacity" ] = 0.5 * kwargs.get( "edge_opacity", 1.0 )
        for k, a in enumerate( maps ):
            first = k == 0
            frames = first or fit_periodicity_rings
            kw = dict( kwargs )
            if not first:
                kw[ "opacity" ] = opacity_periodicity_rings * kw.get( "opacity", 1.0 )
                kw[ "edge_opacity" ] = opacity_periodicity_rings * kw.get( "edge_opacity", 1.0 )
            if shown is not None:
                cells = shown if first else shown.moved( a )
                cells.add_to_viz( viz, owners = np.arange( n ) if owners is None else owners, nb_colors = n, frames = frames, **kw )
            if balls and first:
                self._draw_balls( viz, shown, first_color, kwargs.get( "opacity", 1.0 ) / opacity_beyond_balls )
            if seeds:
                pos = np.asarray( self.positions, dtype = float ).reshape( -1, d )
                viz.add_points( pos @ a[ :, :d ].T + a[ :, d ], color = seed_color, opacity = 1.0 if first else opacity_periodicity_rings, frames = frames )
            if outline is not None and d in ( 2, 3 ):
                _draw_outline( viz, outline, a, domain_color, 1.0 if first else opacity_periodicity_rings, frames )
        g = self._groups if self.group_rep.is_defined else None
        if virtual_seeds and g is not None:
            shared = np.flatnonzero( ( g.parent >= 0 ) & ( g.outer[ np.maximum( g.parent, 0 ) ] == 0 ) ) if g.nb_aggregates else []
            if len( shared ):
                viz.add_points( g.virtual_positions( self.positions )[ shared ], color = virtual_color )
        return viz


def _group_plane( g, pos, sv, av, i, j ):
    """`( direction, offset )` of the plane between members `i` and `j` of a group, on the side of `i` ( host, user order ): the
    oracle of `diagram/Common.h::group_plane` -- the bisector of their virtual diracs in the frame of their lowest common aggregate"""
    def chain( k ):
        res, a = [], g.parent[ k ]
        while a >= 0:
            res.append( a )
            a = g.agg_parent[ a ]
        return res
    c0 = chain( i )
    L = next( a for a in chain( j ) if a in c0 )
    def frame_weight( k ):
        V, a = sv[ k ], g.parent[ k ]
        while a != L:
            p = g.agg_parent[ a ]
            V = av[ a ] + ( g.outer[ a ] / g.scale[ p ] ) ** 2 * V
            a = p
        return V
    s, b = g.scale[ L ], g.center[ L ]
    if g.outer[ L ] == 0:                                # ( the apex follows the site: each member's virtual dirac its own position )
        b = pos[ i ] + g.apex_shift[ L ]
        q0, q1 = g.dirs[ i ], g.dirs[ j ] + ( pos[ j ] - pos[ i ] ) / s
    else:
        q0, q1 = ( pos[ i ] - b ) / s, ( pos[ j ] - b ) / s
    direction = q1 - q0
    offset = float( direction @ b + s * ( ( q1 @ q1 - q0 @ q0 ) / 2 + ( frame_weight( i ) - frame_weight( j ) ) / 2 ) )
    return direction, offset


def _host_array( x ):
    """a host copy ( float64 ) of a value of any framework"""
    from loom.tensor.host import to_host
    return np.asarray( to_host( getattr( x, "raw", x ) ), dtype = np.float64 )


def _disc_in_polygon( vertices, center, r, nb = 96 ):
    """the convex polygon `vertices` inter the disc of center `center` and radius `r` ( a regular `nb`-gon ), as a polygon"""
    t = 2 * np.pi * np.arange( nb ) / nb
    poly = center + r * np.stack( [ np.cos( t ), np.sin( t ) ], axis = 1 )
    v = np.asarray( vertices, dtype = float )
    if len( v ) < 3:
        return poly[ :0 ]
    area = np.sum( v[ :, 0 ] * np.roll( v[ :, 1 ], -1 ) - np.roll( v[ :, 0 ], -1 ) * v[ :, 1 ] )
    if area < 0:
        v = v[ ::-1 ]
    for i in range( len( v ) ):                          # Sutherland-Hodgman: one half-plane per edge
        a, b = v[ i ], v[ ( i + 1 ) % len( v ) ]
        nrm = np.array( [ b[ 1 ] - a[ 1 ], a[ 0 ] - b[ 0 ] ] )     # outward
        s = ( poly - a ) @ nrm
        out = []
        for j in range( len( poly ) ):
            p, q, sp, sq = poly[ j ], poly[ ( j + 1 ) % len( poly ) ], s[ j ], s[ ( j + 1 ) % len( poly ) ]
            if sp <= 0:
                out.append( p )
            if ( sp <= 0 ) != ( sq <= 0 ):
                out.append( p + ( q - p ) * ( sp / ( sp - sq ) ) )
        poly = np.asarray( out ).reshape( -1, 2 )
        if len( poly ) == 0:
            break
    return poly


def _draw_outline( viz, outline, a, color, opacity, frames = True ):
    """the edges of the convex polytope `outline` ( half-spaces ) moved by the affine map `a`, dashed"""
    d = a.shape[ 0 ]
    dom = Cell.make_unbounded( d, kernel_dtype = "FP64" )
    for n, o in zip( *outline ):
        dom.cut( np.asarray( n, dtype = float ), float( o ) )
    if not dom.is_bounded:
        return
    it = dom._item( 0 )
    edges = dom._edges_of( it )
    if len( edges ) == 0:
        return
    vp = it.vp @ a[ :, :d ].T + a[ :, d ]
    viz.add_edges( vp, np.asarray( edges ).reshape( -1, 2 ), color = color, opacity = opacity, dashed = True, frames = frames )


class Neighbors( Aggregate ):
    """the neighbors of ONE cell and the weight of each facet ( see `PowerDiagram.hessian_rows` ) --
    batched over the cells, `nb_nbrs` per cell"""
    ids     : IntTensor [ "num_nbr", dict( size = 32 ) ]
    vals    : RealTensor[ "num_nbr" ]
    num_nbr : Axis[ "nb_nbrs" ]
    nb_nbrs : ShapeVar


# ---- the domain, read off half-spaces ------------------------------------------------------------

def traced_box( directions, offsets ):
    """`axis_aligned_box` for a box whose OFFSETS are known only under a trace ( `SumOfGaussians.bounding_half_spaces` under
    `jax.jit` ): the directions must then be exactly `[ I, -I ]` -- readable, they are constants -- and the box is `( -offsets[ d: ],
    offsets[ :d ] )`, traced arrays. `None` otherwise."""
    if not loom.is_traced( offsets ):
        return None
    try:
        dirs = np.asarray( directions, dtype = float )
    except ( TypeError, ValueError, RuntimeError ):
        return None
    d = dirs.shape[ -1 ] if dirs.ndim == 2 else 0
    if not d or not np.array_equal( dirs, np.concatenate( [ np.eye( d ), -np.eye( d ) ] ) ):
        return None
    offsets = offsets.reshape( -1 )
    return -offsets[ d: ], offsets[ :d ], np.zeros( 2 * d, dtype = bool )


def axis_aligned_box( directions, offsets ):
    """`( mi, ma, kept )`: the box that these half-spaces bound, and which of them it does
    NOT replace. `None` if they do not bound a box.

    We do not try to recognize a box "written properly": we look, axis by axis, for the tightest
    bound that the half-spaces ALIGNED ON THAT AXIS give. A domain that is not
    a box but that contains one ( an octagon ) therefore still provides a starting point, and
    what sticks out is removed by the remaining planes. `None` as soon as an axis is not bounded on both
    sides: the starting cell must be a BOUNDED polytope.
    """
    try:
        dirs = np.asarray( directions, dtype = float )
        offs = np.asarray( offsets, dtype = float ).reshape( -1 )
    except ( TypeError, ValueError, RuntimeError ):      # RuntimeError: a torch tensor that requires grad
        return None                              # geometry not readable here ( a tracer ): too bad
    if dirs.ndim != 2 or len( dirs ) != len( offs ):
        return None

    d = dirs.shape[ 1 ]
    mi, ma = np.full( d, -np.inf ), np.full( d, np.inf )
    for k in range( len( dirs ) ):
        nz = np.flatnonzero( dirs[ k ] )
        if nz.size != 1:
            continue                             # not aligned on an axis: it will be cut, that's all
        a = int( nz[ 0 ] )
        c = dirs[ k, a ]
        if c > 0:
            ma[ a ] = min( ma[ a ], offs[ k ] / c )
        else:
            mi[ a ] = max( mi[ a ], offs[ k ] / c )

    if not np.isfinite( mi ).all() or not np.isfinite( ma ).all():
        return None
    if not ( mi < ma ).all():                    # an empty box cannot be laid down: the general path will empty it
        return None

    # WHICH planes the box already expresses: those, aligned, that reach the retained bound on their axis
    kept = np.ones( len( dirs ), dtype = bool )
    for k in range( len( dirs ) ):
        nz = np.flatnonzero( dirs[ k ] )
        if nz.size != 1:
            continue
        a = int( nz[ 0 ] )
        c = dirs[ k, a ]
        bound = ma[ a ] if c > 0 else mi[ a ]
        kept[ k ] = offs[ k ] / c != bound
    return mi, ma, kept


def box_half_spaces( mi, ma ):
    """The box `mi <= x <= ma` as `2d` half-spaces `direction . x <= offset`."""
    mi = np.asarray( mi, dtype = float ).reshape( -1 )
    ma = np.asarray( ma, dtype = float ).reshape( -1 )
    if mi.size != ma.size:
        raise ValueError( "`box_half_spaces( mi, ma )` wants two corners of the same dimension" )
    d = mi.size
    return np.concatenate( [ np.eye( d ), -np.eye( d ) ] ), np.concatenate( [ ma, -mi ] )
