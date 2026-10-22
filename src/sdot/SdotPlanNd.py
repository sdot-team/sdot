"""THE SOLUTION of a semi-discrete transport in dimension `d >= 2`: the weights of a power
diagram such that the mass of each cell against the target density equals the mass of the
matching dirac.

It is requested from an `OtProblem`, and in no other way:

    sol = OtProblem( SumOfDiracs( pos ), image ).solve()
    sol = ot_solve( SumOfDiracs( pos ), image )        # the same, for a transport solved ONCE

THE WHOLE FIT IS IN C++ ( `sdot/sdotplan/` ), in ONE `loom.ffi_call`: the starting point, the
damped Newton, its diagrams, the laplacian, the linear solver, the damping. Python only sets up
the problem and reads what comes out. That is what the `solvers_des_familles` bench concluded
( README § 3, § 7, § 9, § 10 ):

  * the DAMPED NEWTON of Kitagawa-Mérigot-Thibert wins everywhere, by 2x to 4x in diagrams as in
    time, against L-BFGS and the conjugate gradient on the dual, whatever their
    preconditioning -- the difficulty of semi-discrete transport is not the non-linearity of the
    dual, it is its NON-REGULARITY ( cells that empty out ), and a gradient step is crushed
    there just like a Newton step, only worse ( `sdotplan/Newton.h` );
  * the hessian is the laplacian of the Laguerre graph, assembled WITHOUT SORTING from the facets
    of the diagram that measured the residual -- one sweep delivers both ( `sdotplan/Sweep.h`,
    `sdotplan/Laplacian.h` ); sparse Cholesky in 2D, algebraic multigrid beyond and at large
    sizes ( `sdotplan/Linear.h` );
  * the trial step restarts from the last accepted step, never from 1 ( ten diagrams per step
    saved in the linear phase ); in 2D, the LIMITS of the cells being crushed along the
    direction replace blind trials ( `sdotplan/Bounds.h` ).

= On a CUDA card

The same solve, in ONE ffi call too, whose handler drives the Newton loop on the call's stream ( `gpu/Newton2D.cuh`, see
`_build_card` ): the tree's majorants, the cells, the laplacian, the linear solver ( the card's multigrid ) and the step all
run on the card, and the host only reads back the scalars it decides on -- so the solve runs under `jax.jit` as well. The
tree itself is one more card call just before ( `gpu/Bsp2D.cuh`, `AaBsp._init_on_card` ): under `jax.jit` it is part of
the jitted program, built at every call, from traced positions as well as from constant ones. It
takes 2D problems against a constant density ( a box, or a convex polygon the cells start from: `Polytope`, `ConvexHull` ), an
`Image` on a regular grid, isotropic gaussians or a triangle `Mesh`, with the width continuation of the CPU where there is one
( not for a polytope; for a mesh, the spreading runs its stages, see `_solve_by_spreading` ), and 3D problems against a constant density ( a box or a convex polyhedron; the step `trials`,
as on the CPU; the cells of `gpu/Cell3D.cuh` ); anything else raises on a card ( the CPU solves it ).

= The starting point

Newton needs an ADMISSIBLE start ( no empty cell ). Voronoi is one as soon as the diracs are in
the domain; otherwise, or if the given weights empty a cell, the C++ itself picks the best of
the three -- the given weights, Voronoi, the SIMILARITY that brings the cloud back into the
domain -- and reports it ( `stats[ "start" ]` ). See `sdotplan/Solve.h`.

= What the solution CARRIES, and why it is not just a weight vector

`weights` is the answer when the seeds are distinct. When two seeds are `1e-8` apart, there is
no pair of `double`s that encodes the plane separating them to within `1e-9`: the bench
measured and computed it ( README § 23.11 ), and that is why a solver whose interface is `w`
plateaus around `1e-6` on a degenerate cloud. Such seeds are AGGREGATED ( `Iterative( aggregate =
True )`, the default; `sdotplan/Aggregation.h` ): a pair is merged when one ulp of its weights moves
more than `tol / 4` of mass between them ( read on the diagram: the facet's coefficient of the
laplacian times the spacing of the doubles at the weights ), exact duplicates are merged and kept
empty, the tests read the mass of each cluster, and the merged cells are re-split by their local
problem once the aggregated problem has converged. `stats[ "status" ]` is then
`"converged (aggregated)"`, `stats[ "residual" ]` the aggregated problem's residual and
`stats[ "residual_full" ]` the full one's -- the floor of the doubles, which no weight vector beats.
`clusters` says which seeds were merged ( `None` when none was ).

What it does not cost: the TRANSPORT COST is blind to this degeneracy ( § 23.12, relative gap
`1.2e-17` ), so anything that is a sum weighted by the masses -- the cost, the total mass, a
global moment -- need know nothing about it. What stays APPROXIMATE for the members of a cluster:
their own masses and barycentres ( `cell_masses`, `barycenters`: those of the returned weights,
within the floor for near-coincident seeds ), hence their part of `cost_and_position_grad` ( exact
for the cluster as a whole, not split between its members ).

EXACT DUPLICATES are a CONVENTION ( the plan is not differentiable there ): a cluster of diracs at
the same position is ONE dirac of the summed mass. The solver leaves the cell of a duplicate empty
( its weight `w_rep - gap` ) and the merged cell to its representative; the outputs then share that
cell: `cell_masses[ i ] = m_c nu_i / nu_c`, `barycenters[ i ]` the merged barycenter, for each
member -- and the derivatives are those of the merged problem ( `_share_duplicates`,
`_attach_derivatives` ). Only for duplicates found on the host ( not under `jax.jit` ).

= A single diagram

The positions are the CONSTANTS of the fit: the diagram is built ONCE ( its tree included ), and
the solver only SETS the trial weights on it -- which, for BSP storage, recomputes the weight
upper bound of each node and nothing else. What it writes ( the sorted weights, the upper
bounds ) are OUTPUTS of the call, taken back by the diagram afterwards: the inputs of a call
are read-only.

`PowerDiagram` and `Cell` are ON-DEMAND TOOLS: `sol.power_diagram()` builds them for drawing or
for a functional that was not anticipated. Nothing in the solve goes through them on the
Python side.
"""

import copy
import warnings

import numpy as np

import loom
from loom.compilation.FfiCode import FfiCode
import loom
from loom.tensor import Axis, CtShapeVar, IntTensor, RealTensor, ShapeVar, Tensor
from loom.util import Aggregate

from .CellScratch import fp_size
from .PowerDiagram import PowerDiagram
from .viz.Displayable import Displayable


# what `stats` carries, in the order of `sdotplan/Solve.h::Stat`
_STATS = [ "status", "residual", "residual0", "nb_iter", "nb_diag", "nb_backtracks", "t_majorant", "t_diag", "t_asm", "t_lin", "t_lim", "eps",
           "domain_mass", "nb_overflowed", "nb_cell_lim", "nb_limit_rounds", "lin_nb_hierarchies", "lin_nb_iter", "lin_worst", "start", "t_total",
           "nb_continuation_steps", "min_start_mass", "it_switch", "it_double", "scratch_bytes",
           "nb_clusters", "nb_aggregated", "nb_duplicates", "residual_full", "nb_polish" ]
# one row of `history`, in the order of `sdotplan/Solve.h::Hist`
_HISTORY = [ "step", "t", "residual_l2", "min_measure", "max_abs_residual", "nb_diag", "nb_evals", "s" ]
_STATUS = { 0: "running", 1: "converged", 2: "max iterations", 3: "stagnation", 4: "linear solver failure", 5: "card capacity",
            6: "failure", 7: "converged (aggregated)" }
_START = { 0: "weights0", 1: "voronoi", 2: "similarity", 3: "balls" }
_LIN = { "auto": 0, "cholesky": 1, "amg": 2, "cg": 3, "mg": 4 }
_AMG_VARIANT = { "auto": -1, "sa_spai0": 0, "sa_gs": 1, "rs_gs": 2 }
_STEP = { "trials": 0, "limits": 1 }
_RESIDUAL = { "lin": 0, "log": 1, "power": 2 }
_CONTINUATION = { "never": 0, "auto": 1, "always": 2 }

#: THE AGGREGATION ( `sdotplan/Aggregation.h` ): a pair is merged when one ulp of its weights moves more than `tol / margin` of
#: mass between them -- a pair lands at best within half an ulp step ( 0.43 of it measured on `lines_equal` ), 4 leaves room
#: for Newton's own landing ( `Tuning( aggregation_margin = ... )` )
_AGGREGATE_MARGIN = 4.0
#: exact duplicates: `w_dup = w_rep - gap`, `gap` this fraction of the squared extent of the cloud -- any gap empties the cell of
#: a duplicate, this one is far above the rounding of the cells' cuts ( the generic 3D cell mistakes a gap under ~1e-14 |p|^2 )
_DUPLICATE_GAP = 1e-6


#: the outputs whose derivative `_attach_derivatives` attaches
_DERIVED = ( "weights", "cell_masses", "cost", "barycenters" )


def _origin_as_shift( problem ):
    """THE DERIVATIVE WITH RESPECT TO THE ORIGIN OF THE TARGET ( an `Image`, a `Box` ), by translation: the cost is invariant under a
    translation of everything, so the target moved by `s` is the diracs moved by `- s` -- same weights, masses and cost, and the
    barycenters move with the target. The derivative is therefore written with the origin DETACHED and the diracs at `p - s`,
    `s = o - stop_gradient( o )` ( worth zero, carrying the derivative of `o` ), then `s` is added to the barycenters -- no kernel
    has to differentiate the geometry of the pixels. `( problem, s )`: the problem to differentiate, and `s`
    ( a raw array ) -- or `( problem, None )` when the origin is not traced."""
    dst = problem.target
    o = getattr( dst, "origin", None )
    if not isinstance( o, Tensor ) or not o.is_defined or not ( o.storage.traces or loom.is_traced( o.raw ) ):
        return problem, None
    from loom import stop_gradient
    o0 = stop_gradient( o )
    shift = o.raw - o0.raw
    target = copy.copy( dst )
    target.origin = o0
    source = copy.copy( problem.source )
    source.positions = RealTensor[ source.num_dirac, source.dim ]( problem.source.positions.raw - shift[ None, : ] )
    moved = copy.copy( problem )
    moved._source, moved._target = source, target
    return moved, shift


def _periodicity_of( problem ):
    """the periodicity of a problem, `None` when there is none"""
    p = getattr( problem, "periodicity", None )
    return None if p is False else p


def _declared_mass( dist ):
    """the mass a distribution is normalized to: its `target_mass`, or its own mass when it has none ( `None` when it is only known
    under a trace )"""
    tm = getattr( dist, "target_mass", None )
    v = tm.raw if tm is not None and tm.is_defined else getattr( dist.mass, "raw", dist.mass )
    if loom.is_traced( v ):
        return None
    from loom.tensor.host import to_host     # a cupy array refuses `np.asarray`
    return float( np.asarray( to_host( v ) ).reshape( -1 )[ 0 ] )


def _sink_of( problem ):
    """`0.0` ( the weight of the sink ) when the transport is PARTIAL -- the source weighs less than the target ( `target_mass` ),
    and the rest of the target goes to a sink: the cells are capped by balls ( `PowerDiagram( sink_weight = ... )` ) -- `None`
    when the masses balance. Two equal `target_mass` balance exactly; otherwise up to a relative `1e-12`."""
    ms, mt = _declared_mass( problem.source ), _declared_mass( problem.target )
    if ms is None or mt is None or ms == mt or abs( ms - mt ) <= 1e-12 * max( abs( ms ), abs( mt ) ):
        return None
    if ms > mt:
        raise NotImplementedError( f"partial transport: the source ( mass { ms :g} ) weighs more than the target ( { mt :g} ) -- "
                                   "only the other way round for now ( a lighter source, the rest of the target to a sink )" )
    return 0.0


def exact_duplicates( positions ):
    """THE EXACT DUPLICATES of a cloud `[ n, d ]` ( a host array ): `( dup, rep )`, each duplicate and the smallest index of
    the seeds at the same position ( empty arrays when there is none ). A hash of the coordinates' bits, sorted: 2.8 ms for
    1e5 seeds, 22 ms for 1e6 ( M-series ); the exact grouping only runs on the seeds whose hashes collide."""
    import numpy as np
    P = np.ascontiguousarray( np.asarray( positions, dtype = np.float64 ) ) + 0.0       # ( `+ 0.0`: -0.0 and 0.0 alike )
    n = P.shape[ 0 ]
    none = ( np.zeros( 0, dtype = np.int64 ), np.zeros( 0, dtype = np.int64 ) )
    if n < 2:
        return none
    bits = P.reshape( n, -1 ).view( np.uint64 )
    h = np.zeros( n, dtype = np.uint64 )
    for k, mul in zip( range( bits.shape[ 1 ] ), ( 0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0xD6E8FEB86659FD93 ) ):
        h ^= bits[ :, k ] * np.uint64( mul )
        h ^= h >> np.uint64( 29 )
    order = np.argsort( h )
    s = h[ order ]
    eq = s[ 1: ] == s[ :-1 ]
    if not eq.any():
        return none
    cand = np.sort( order[ np.r_[ eq, False ] | np.r_[ False, eq ] ] )       # ( the colliding hashes only )
    _, inv = np.unique( P.reshape( n, -1 )[ cand ], axis = 0, return_inverse = True )
    inv = inv.reshape( -1 )
    first = np.full( int( inv.max() ) + 1, n, dtype = np.int64 )
    np.minimum.at( first, inv, cand )
    rep = first[ inv ]
    keep = rep != cand
    return cand[ keep ].astype( np.int64 ), rep[ keep ].astype( np.int64 )


def _duplicate_layout( n, dup, rep ):
    """THE MERGED CLOUD of a cloud with exact duplicates ( `exact_duplicates` ): `( keep, inv, members )` -- `keep` ( `[ k ]` ) the
    representative of each cluster ( its smallest index ), `inv` ( `[ n ]` ) the cluster of each seed, `members` ( `[ r, k ]` ) the
    seeds of each cluster by rank ( `members[ 0 ] == keep` ), `n` where a cluster has fewer than `r` of them ( a zero row, see
    `_cluster_sum` )."""
    root = np.arange( n, dtype = np.int64 )
    root[ dup ] = rep
    keep = np.nonzero( root == np.arange( n ) )[ 0 ]
    inv = np.searchsorted( keep, root )
    order = np.lexsort( ( np.arange( n ), inv ) )
    start = np.searchsorted( inv[ order ], np.arange( len( keep ) ) )
    rank = np.empty( n, dtype = np.int64 )
    rank[ order ] = np.arange( n ) - start[ inv[ order ] ]
    members = np.full( ( int( rank.max() ) + 1, len( keep ) ), n, dtype = np.int64 )
    members[ rank, inv ] = np.arange( n )
    return keep, inv, members


def _cluster_sum( raw, members ):
    """the sum over each cluster of `raw` ( `[ n, ... ]`, any framework, traced or not ) -- gathers only ( no scatter in every
    framework ), the missing members reading an appended zero row"""
    ext = loom.ops( raw ).concatenate( [ raw, raw[ :1 ] * 0 ], axis = 0 )
    out = ext[ members[ 0 ] ]
    for r in members[ 1: ]:
        out = out + ext[ r ]
    return out


#: the old arguments of `SdotPlanNd( src, dst, ... )`, and where they live now -- read by the
#: deprecated path ( see `__init__` )
_DEPRECATED_TO_TUNING = ( "accelerator", "memory", "step", "linear_solver", "mass_rtol", "t_min",
                          "max_backtracks", "restart_factor", "conv_start", "conv_ratio", "conv_min",
                          "conv_threshold" )


class _Options( Aggregate ):
    """the solver settings, as `sdotplan/Solve.h` reads them"""
    mass_tol       : RealTensor
    mass_rtol      : RealTensor
    t_min          : RealTensor
    mult_ok        : RealTensor
    factor         : RealTensor
    beta0          : RealTensor
    mult_lim       : RealTensor
    confidence     : RealTensor
    conv_s0        : RealTensor
    conv_ratio     : RealTensor
    conv_min       : RealTensor
    conv_threshold : RealTensor
    residual_power : RealTensor
    lin_tol        : RealTensor
    residual_switch : RealTensor
    max_iter       : IntTensor
    max_backtracks : IntTensor
    lin            : IntTensor
    amg_variant    : IntTensor
    mg_pack        : IntTensor
    mg_recycle     : IntTensor
    mg_rebuild     : IntTensor
    mg_stop        : IntTensor
    mg_nu          : IntTensor
    step           : IntTensor
    residual       : IntTensor
    trace          : IntTensor
    continuation   : IntTensor
    cap0           : IntTensor
    agg_margin     : RealTensor
    agg_gap        : RealTensor
    agg_nb_dups    : IntTensor
    keep_start     : IntTensor
    kernel_fp_size : CtShapeVar


#: the card solver's options, ONE real tensor ( `gpu/Newton2D.cuh::Opt`, same order )
_CARD_OPTIONS = [ "tol_abs", "tol_rel", "t_min", "mult_ok", "factor", "maxit", "max_backtracks", "step", "residual", "power", "switch",
                  "lin", "host_method", "lin_tol", "amg_variant", "mg_shift", "mg_recycle", "mg_rebuild", "mg_stop", "mg_nu", "mg_kcycle",
                  "lin_maxit", "trace", "mg_float", "mg_smoothed",
                  "continuation", "conv_threshold", "conv_s0", "conv_ratio", "conv_min", "conv_possible", "min_scale",
                  "img_x0", "img_y0", "img_hx", "img_hy", "img_nx", "img_ny", "agg_margin", "agg_gap", "agg_nb_dups", "sink", "sink_w",
                  "start_box", "start_lo0", "start_lo1", "start_lo2", "start_hi0", "start_hi1", "start_hi2" ]


def card_density_call( dens, d ):
    """= Internals

    how a handler of the card takes the density `dens` ( `PowerDiagram_Bsp._card_solve_density` ): `( C++ expression of the
    density, its tensors as kwargs of the call )` -- `DensityHost2D.cuh`"""
    import numpy as np
    def axis( m, name ):
        return Axis( ShapeVar( int( m ) ), name = name )
    if dens[ "kind" ] == "const":
        rho = dens[ "rho" ]                              # ( a number, or traced: read on the card )
        return "args.inputs.density", dict( density = RealTensor( np.float64( rho ) if isinstance( rho, ( int, float ) ) else rho ) )
    if dens[ "kind" ] == "image":
        nx, ny = int( dens[ "geom" ][ 4 ] ), int( dens[ "geom" ][ 5 ] )
        return "sdot::gpu2d::image_in( args.inputs.img_values )", dict( img_values = RealTensor[ axis( nx * ny, "num_card_pixel" ) ]( dens[ "values" ] ) )
    if dens[ "kind" ] == "mesh":
        # ( 2D: triangles, `Density2D.cuh::DensMesh`; 3D: tetrahedra, `Density3D.cuh::DensMesh3` )
        ne, nn, nb = int( dens[ "tri" ].shape[ 0 ] ), int( dens[ "nodes" ].shape[ 0 ] ), int( dens[ "lo" ].shape[ 0 ] )
        expr = ( f"sdot::gpu2d::{ 'mesh_in' if d == 2 else 'mesh3_in' }( args.inputs.m_nodes, args.inputs.m_values, args.inputs.m_tri, "
                 "args.inputs.m_grad, args.inputs.m_lo, args.inputs.m_hi, args.inputs.m_links )" )
        return expr, dict(
            m_nodes = RealTensor[ axis( nn, "num_card_mnode" ), axis( d, "num_card_mdim" ) ]( dens[ "nodes" ] ),
            m_values = RealTensor[ axis( ne, "num_card_melem_v" ), axis( d + 1, "num_card_mcorner_v" ) ]( dens[ "values" ] ),   # ( DG1 )
            m_tri = IntTensor[ axis( ne, "num_card_melem" ), axis( d + 1, "num_card_mcorner" ), dict( size = 32 ) ]( dens[ "tri" ] ),
            m_grad = RealTensor[ axis( ne, "num_card_melem_g" ), axis( d * ( d + 1 ), "num_card_mgrad" ) ]( dens[ "grad" ] ),
            m_lo = RealTensor[ axis( nb, "num_card_mbnode_l" ), axis( d, "num_card_mdim_l" ) ]( dens[ "lo" ] ),
            m_hi = RealTensor[ axis( nb, "num_card_mbnode_h" ), axis( d, "num_card_mdim_h" ) ]( dens[ "hi" ] ),
            m_links = IntTensor[ axis( nb, "num_card_mbnode_k" ), axis( 4, "num_card_mlink" ), dict( size = 32 ) ]( dens[ "links" ] ) )
    # ( 2D: `Density2D.cuh::DensGauss`; 3D: `Density3D.cuh::DensGauss3` )
    ng = int( dens[ "sigma" ].shape[ 0 ] )
    return f"sdot::gpu2d::{ 'gauss_in' if d == 2 else 'gauss3_in' }( args.inputs.g_pos, args.inputs.g_sigma, args.inputs.g_mass )", dict(
        g_pos = RealTensor[ axis( ng, "num_card_gauss" ), axis( d, "num_card_gdim" ) ]( dens[ "pos" ] ),
        g_sigma = RealTensor[ axis( ng, "num_card_gauss_s" ) ]( dens[ "sigma" ] ),
        g_mass = RealTensor[ axis( ng, "num_card_gauss_m" ) ]( dens[ "mass" ] ) )


def _card_linear_solves( row, col, val, dia, rhs, method = "mg", tol = 1e-6, smoothed = None, precision = "float", recycle = 2, rebuild = 1,
                         kcycle = -1, nu = 0 ):
    """THE CARD'S LINEAR SOLVER ALONE ( a test hook, `gpu/Newton2D.cuh::linear_solves` ): the systems `L d_j = rhs[ j ]` ( each
    `rhs[ j ]` of zero sum ) with ONE laplacian `L` given as the CSR of `Laplacian2D.cuh` ( `row`, `col`, `val > 0` the
    off-diagonals, `dia` ), solved in sequence by ONE solver, as along a Newton solve ( the recycled start: `recycle`
    solutions; a hierarchy every `rebuild` solves ). `method`: `"mg"` or `"cg"`; `smoothed`: the levels of the smoothed
    aggregation ( `None`: the default ); `precision`: of the multigrid's levels. Returns `( d, its )`: the solutions ( zero
    mean, `k x n` ) and the iterations of each solve ( -1: failed )."""
    import numpy as np
    rhs = np.atleast_2d( np.asarray( rhs, dtype = np.float64 ) )
    k, n = rhs.shape
    row = np.asarray( row, dtype = np.int64 ).reshape( -1 )[ :n + 1 ]
    nnz = int( row[ n ] )
    opts = [ n, k, 0 if method == "cg" else 1, tol, -1 if smoothed is None else smoothed, int( precision == "float" ), recycle, rebuild, kcycle, nu ]
    def axis( m, name ):
        return Axis( ShapeVar( int( m ) ), name = name )
    solution = RealTensor[ axis( k * n, "num_lin_all" ) ]()
    iterations = RealTensor[ axis( k, "num_lin_rhs" ) ]()
    loom.ffi_call(
        "sdotplan_card_linear_2d",
        FfiCode.inline( "sdot::gpu2d::linear_solves( queue, args.inputs.row, args.inputs.col, args.inputs.val, args.inputs.dia, args.inputs.rhs, "
                        "args.inputs.options, args.outputs.solution, args.outputs.iterations, args.allocator );",
                        includes = [ "sdot/gpu/Newton2D.cuh" ], sources = [ "sdot/sdotplan/Linear.cpp" ], allocator = True ),
        row = IntTensor[ axis( n + 1, "num_lin_row" ), dict( size = 64 ) ]( row ),
        col = IntTensor[ axis( max( nnz, 1 ), "num_lin_nnz" ), dict( size = 64 ) ]( np.asarray( col, dtype = np.int64 ).reshape( -1 )[ :max( nnz, 1 ) ] ),
        val = RealTensor[ axis( max( nnz, 1 ), "num_lin_val" ) ]( np.asarray( val, dtype = np.float64 ).reshape( -1 )[ :max( nnz, 1 ) ] ),
        dia = RealTensor[ axis( n, "num_lin_dia" ) ]( np.asarray( dia, dtype = np.float64 ).reshape( -1 )[ :n ] ),
        rhs = RealTensor[ axis( k * n, "num_lin_b" ) ]( rhs.reshape( -1 ) ),
        options = RealTensor[ axis( len( opts ), "num_lin_opt" ) ]( np.asarray( opts, dtype = np.float64 ) ),
        solution = loom.out( solution ),               # ( not `d`: a name loom's generated templates use )
        iterations = loom.out( iterations ),
    )
    from loom.tensor.host import to_host
    return to_host( solution.raw ).reshape( k, n ), to_host( iterations.raw ).reshape( -1 ).astype( int )


#: the precision of the card multigrid's levels by default ( the outer flexible CG is in double either way ): float, the
#: same iteration counts as double and 1.15-1.4x faster ( `calibration_lmo_today.md`, GPU step 5 )
_CARD_MG_PRECISION = "float"
#: the card multigrid's aggregates ( seeds per packet of tree ranks ) and Chebyshev degree, per dimension, when the tuning
#: does not say: 2D `calibration_lmo_today.md` ( GPU step 5 ), 3D `calibration_n22.md` ( 3D step 2 )
_CARD_MG_PACK = { 2: 4, 3: 8 }
_CARD_MG_NU = { 2: 1, 3: 1 }


def card_facet_capacity( nb_seeds, dim = 2 ):
    """the first guess of the upper facets of a diagram: in 2D a planar graph, at most `3 n - 6` edges, and a margin for the
    slivers of a float topology; in 3D no bound, 7.8 per seed on a uniform cloud ( 15.5 neighbours ), FACET_RATIO_3D with
    room for the denser clouds. Eagerly loom grows it if a diagram wants more; under a `jit` that is an error, hence the room"""
    if int( dim ) == 3:
        return int( _FACET_RATIO_3D * int( nb_seeds ) ) + 1024
    return 3 * int( nb_seeds ) + 512


#: the upper facets per seed the card's 3D solve makes room for ( `card_facet_capacity` ): measured 7.8 on uniform clouds,
#: 8.5 on the planes ( `calibration_n22.md`, 3D step 2 )
_FACET_RATIO_3D = 10.0


class _CardSolveWork( Aggregate ):
    """what the card's solve writes besides its results: the capacity that a diagram may ask loom to grow ( `nb_facets`:
    the upper facets of a diagram; the cells' fourth pass has a fixed budget, `PowerDiagram_Bsp.card_overflow_warps_for` )"""
    nb_facets : ShapeVar


class _History( Aggregate ):
    """one ACCEPTED step per row ( `step = 0`: the start ), and the weights of each step if they
    were requested ( `weights`, otherwise `Unbound` -- a `[ step, n ]` array that is not always wanted )"""
    rows      : RealTensor[ "num_step", "num_hist" ]
    weights   : RealTensor[ "num_step", "num_point" ]
    num_step  : Axis[ "nb_steps" ]
    num_hist  : Axis[ "nb_hist" ]
    num_point : Axis[ "nb_points" ]
    nb_steps  : ShapeVar
    nb_hist   : ShapeVar
    nb_points : ShapeVar


class SdotPlanNd( Displayable ):
    """A transport plan in 2D and more, from `plan`, `ot_solve` or `OtProblem.solve`.

    Each dirac receives a cell of the power diagram of `weights`, whose mass under the target density is the mass of the
    dirac: the plan sends each cell onto its dirac. `cost` is `W_2^2`, `barycenters` the barycenter of each cell,
    `cell_masses` their masses. `converged` and `stats` say how the solve went.

    = Internals

    see the module docstring"""

    #: the width continuation of a mesh ( `_solve_by_spreading` ): the largest step of the path, as a ratio of `k + 1`, and the
    #: smallest one a failed stage may be cut down to; the Newton steps of a stage
    SPREAD_RATIO = 4.0
    SPREAD_MIN_RATIO = 1.05
    SPREAD_STAGE_ITER = 60
    #: `continuation = "auto"` on a target that spreads: from this `spread_start()` on ( a density far from flat, whose max is still
    #: over 3 times its mean after so many averages ), the continuation runs at once without a warm start -- the plain Newton from
    #: scratch was measured to fail there, and trying it first was about half of the time of the solve
    SPREAD_DIRECT = 16

    # -- how to get one -----------------------------------------------------------------------

    @classmethod
    def _solve( cls, problem, settings, verbose, warm = None ):
        """THE path: `OtProblem.solve()` and only it goes through here. `warm` is the last SOLUTION
        the problem returned -- the start when the settings do not impose one ( see
        `_start_from_plan` ).

        The solve runs on the driver of its distributions, as their calls do ( `loom.ffi_call` ): Jax tracers in a Torch
        process are solved by Jax, on its own device -- what is decided here from `loom.resolved_device()` ( the card's solver
        and tree, or the CPU's ) follows them, not the default framework's card. The SOURCE decides ( its positions and masses ):
        a target is data, converted when it is another framework's -- and the detached solve of a derivative, whose positions are
        Jax arrays beside a target Torch built, stays on Jax."""
        from loom.calls import _scan
        from loom.drivers import framework_defaults
        from loom.drivers.promotion import promote
        default = framework_defaults.framework_name()
        tracing, plain = set(), set()
        _scan( problem.source, tracing, plain, set() )
        if not tracing:                                  # ( nothing traces: the positions' -- the masses may be the default's, from numpy )
            plain = set()
            _scan( problem.source.positions, tracing, plain, set() )
        framework = promote( tracing, plain, default )
        if framework is None:
            return cls._solve_here( problem, settings, verbose, warm )
        with framework_defaults.using( framework ):
            return cls._solve_here( problem, settings, verbose, warm )

    @classmethod
    def _solve_here( cls, problem, settings, verbose, warm = None ):
        """`_solve`, on the driver in force"""
        if not getattr( problem, "_is_detached", False ) and cls._wants_derivatives( problem, settings ):
            # THE SOLVE IS NOT DIFFERENTIATED: it runs on detached inputs, and its outputs get their derivative from the implicit
            # function theorem ( `_attach_derivatives` )
            plan = cls._solve( problem._detached(), settings, verbose, warm )
            solved = { name: getattr( plan, name ) for name in _DERIVED }
            moved, shift = _origin_as_shift( problem )
            plan._attach_derivatives( moved, settings )
            if shift is not None:
                plan.barycenters = RealTensor[ plan._pd.num_point, plan._pd.dim ]( plan.barycenters.raw + shift[ None, : ] )
            # THE VALUES ARE THE SOLVER'S, to the bit: what is attached carries the derivative ( its value differs by the rounding of
            # its own diagrams -- a `jit` that nobody differentiates must give what the eager solve gives )
            ops = loom.ops()
            for name, value in solved.items():
                attached = getattr( plan, name ).raw
                setattr( plan, name, type( value )( value.raw + ( attached - ops.stop_gradient( attached ) ), axes = value.axes ) )
            return plan
        spreads = cls._spreads( problem, settings )
        if spreads:
            k0 = problem.target.spread_start()
            # a warm start ( the previous solution, `weights0` ) is usually close enough for the plain solve, even on a target far
            # from flat: tried first ( a Lloyd step replayed the whole continuation without it ), the spreading if it fails
            started = warm is not None or settings.weights0 is not None
            if k0 is not None and ( settings.continuation == "always" or ( k0 >= cls.SPREAD_DIRECT and not started ) ):
                if verbose and settings.continuation != "always":
                    print( f"  the target is far from flat ( spread_start = { k0 } ): width continuation by its spreading at once" )
                return cls._solve_by_spreading( problem, settings, verbose, k0 = k0 )
        # the plain solve first; a target the C++ cannot blur ( a mesh ) that it fails on goes through its spreading
        tried = copy.copy( settings )
        tried.on_failure = "ignore" if spreads else settings.on_failure
        self = cls.__new__( cls )
        self._build( problem, tried, verbose, warm )
        if spreads and not self.converged:
            if verbose:
                print( f"  the plain solve did not converge ( { self.stats[ 'status' ] } ): width continuation by the spreading of the mesh" )
            return cls._solve_by_spreading( problem, settings, verbose, plain = self )
        self._check_converged( settings )
        return self

    @staticmethod
    def _wants_derivatives( problem, settings ):
        """`differentiable = "auto"`: when one of the inputs is traced"""
        flag = getattr( settings, "differentiable", "auto" )
        if flag != "auto":
            return bool( flag )
        from loom.tensor import Tensor
        def traced( agg ):
            for v in vars( agg ).values():
                if isinstance( v, Aggregate ):
                    if traced( v ):
                        return True
                # `storage.traces`: what an autodiff RECORDS -- a jax tracer, but also a torch tensor that `requires_grad`, which
                # `loom.is_traced` leaves out ( for torch, it only means a `torch.func` transform )
                elif isinstance( v, Tensor ) and v.is_defined and ( v.storage.traces or loom.is_traced( v.raw ) ):
                    return True
            return False
        return traced( problem.source ) or traced( problem.target )

    @staticmethod
    def _spreads( problem, settings ):
        """the TARGET spreads itself for the width continuation ( `Distribution.spread`: a mesh, which the C++ does not convolve ), and
        the settings allow a continuation"""
        if settings.continuation == "never" or getattr( settings, "ot_plan", None ) is not None:
            return False
        return callable( getattr( problem.target, "spread", None ) )

    @classmethod
    def _solve_by_spreading( cls, problem, settings, verbose, plain = None, k0 = None ):
        """THE WIDTH CONTINUATION OF A MESH: the path of its spread densities ( `Mesh.spread( k )`, the corner averages: the same mesh,
        the same pieces, wider values ), from `k = Mesh.spread_start()` down to `k = 0`, the mesh itself. Each stage starts from the
        weights of the last converged one, at a loose tolerance ( `1e-2 / n` ); the steps are in `log( k + 1 )`, at most
        `SPREAD_RATIO`, lengthened after a success and halved after a failure, down to `SPREAD_MIN_RATIO` -- between `k = 1` and
        `k = 0` too, where the spread density is a mix of the mesh and of its first average."""
        import math
        from .OtProblem import OtProblem
        target = problem.target
        k0 = target.spread_start() if k0 is None else k0
        if k0 is None:                                     # ( the mesh is not readable on the host: nothing to spread )
            if plain is None:
                plain = cls.__new__( cls )
                plain._build( problem, settings, verbose, None )
            plain._check_converged( settings )
            return plain
        n = int( problem.nb_diracs )
        s_end = math.log( k0 + 1 )
        big = math.log( cls.SPREAD_RATIO )
        s, s_prev, ds, w = 0.0, 0.0, big, None
        stages, rejected, nb_iter = [], 0, 0
        while True:
            last = s >= s_end - 1e-9                       # ( no stage at k ~ 1e-12 before the mesh )
            k = 0.0 if last else ( k0 + 1 ) * math.exp( -s ) - 1
            stage = copy.copy( settings )
            stage.weights0, stage.ot_plan, stage.on_failure, stage.continuation = w, None, "ignore", "never"
            stage.tuning = copy.copy( settings.tuning )
            stage.tuning.keep_start = w is not None
            if last:
                sol = cls.__new__( cls )
                sol._build( problem, stage, verbose, None )
            else:
                stage.tol = max( settings.tol, 1e-2 / n )
                stage.max_iter = min( settings.max_iter, cls.SPREAD_STAGE_ITER )
                try:
                    sol = OtProblem( problem.source, target.spread( k ), periodicity = _periodicity_of( problem ) ).solve( stage, verbose )
                except RuntimeError:                       # ( a stage that the solver could not run: a failed stage )
                    sol = None
            ok = sol is not None and sol.converged
            if sol is not None:
                nb_iter += int( sol.stats[ "nb_iter" ] )
            if verbose:
                print( f"  spreading, k = { 0 if last else k:.3g}: { sol.stats[ 'status' ] if sol is not None else 'failed' }" )
            if ok:
                stages.append( 0.0 if last else k )
                w = np.asarray( sol.weights ).reshape( -1 )
                if last:
                    break
                s_prev, ds = s, min( ds * 1.5, big )
            else:
                rejected += 1
                if w is None or ds / 2 < math.log( cls.SPREAD_MIN_RATIO ):
                    if sol is None or not last:            # ( the path is stuck: the mesh from the last weights, for the status )
                        sol = cls.__new__( cls )
                        final = copy.copy( settings )
                        final.weights0, final.ot_plan, final.continuation = w, None, "never"
                        final.tuning = copy.copy( settings.tuning )
                        final.tuning.keep_start = w is not None
                        final.on_failure = "ignore"
                        sol._build( problem, final, verbose, None )
                    break
                ds /= 2
            s = min( s_prev + ds, s_end )
        sol.stats.update( spread_start = k0, spread_stages = stages, spread_rejected = rejected, spread_nb_iter = nb_iter )
        sol._check_converged( settings )
        return sol

    def __init__( self, src_dist, dst_dist, verbose = False, **kwargs ):
        """DEPRECATED -- `OtProblem( src_dist, dst_dist ).solve( Iterative( ... ) )`.

        There is now a single entry point: a problem is posed ( `OtProblem` ), then a solution is
        requested from it. This constructor translates the old arguments and will be removed."""
        warnings.warn( "SdotPlanNd( src, dst, ... ) is deprecated: pose the problem then "
                       "solve it -- OtProblem( src, dst ).solve( Iterative( ... ) ). See the "
                       "docstring of `OtProblem`.", DeprecationWarning, stacklevel = 2 )
        from .OtProblem import Iterative, OtProblem, Tuning

        kw = dict( kwargs )
        if "kernel_dtype" in kw:
            kd = kw.pop( "kernel_dtype" )
            kw[ "precision" ] = { None: "auto", "FP64": "fp64", "FP32": "fp32" }.get( kd, kd )
        if "mass_tol" in kw:
            kw[ "tol" ] = kw.pop( "mass_tol" )
        tuning = Tuning( **{ k: kw.pop( k ) for k in _DEPRECATED_TO_TUNING if k in kw } )
        self._build( OtProblem( src_dist, dst_dist ), Iterative( tuning = tuning, **kw ), verbose )

    # -- what the call does -------------------------------------------------------------------

    def _build( self, problem, settings, verbose, warm = None ):
        # two solvers: the CPU's ( `sdotplan/Solve.h`, host code on the CPU queue ) and, on a CUDA device, the card's
        # ( `gpu/Newton2D.cuh`: 2D or 3D, a box -- see `_build_card` ); any other device is refused
        on_card = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
        if not loom.resolved_device().is_cpu and not on_card:
            raise NotImplementedError( f"SdotPlanNd: no solver for the device { loom.resolved_device() } ( the CPU, or a CUDA card in 2D )" )
        tun = settings.tuning
        if settings.precision == "mixed" and not on_card:
            raise ValueError( "precision = 'mixed' ( the float kernel, then the double one ) is the card's: on the CPU, 'fp64' or 'fp32'" )
        #: the problem this is the solution of
        self.problem = problem
        src_dist, dst_dist = problem.source, problem.target
        d = problem.nb_dims

        # the domain: the support of the density, which must bound it ( raises otherwise ).
        # `PowerDiagram` adds it itself from the distribution -- we ASK for it here so that the
        # refusal is reported before anything has been allocated.
        problem.domain

        # the limits-based step only exists in 2D ( `sdotplan/Bounds.h` ); elsewhere, trials. Not with a periodicity: the limits
        # rebuild the cells from the neighbours of the laplacian, which forgets through which image they are neighbours
        periodicity = _periodicity_of( problem )
        # A RADIAL integrand ( `OtProblem( radial_func = ... )` ): the masses are the diracs', the target's density a factor
        self._radial = getattr( problem, "radial_func", None )
        # PARTIAL transport ( the source is lighter than the target ): a sink, which takes what the cells' balls leave
        self._sink = _sink_of( problem ) if self._radial is None else None
        # GROUPS ( `duplicates = "split"`, `sdotplan/Groups.h` ): the tree of aggregates, found once, here, on readable positions -- the
        # Newton of the CPU and of the card ( `gpu/CardGroups.cuh` ) then runs on all the weights of that geometry; a periodicity keeps
        # the aggregation, then `_split_groups`
        self._solver_groups = None
        pos_given = getattr( src_dist.positions, "raw", src_dist.positions )
        # ( the limits' cells, `sdotplan/Bounds.h`, do not know the groups: an explicit `step = "limits"` takes that path too )
        if ( getattr( settings, "duplicates", "split" ) == "split" and settings.aggregate and periodicity is None
             and tun.step != "limits" and not loom.is_traced( pos_given ) ):
            self._solver_groups = self._groups_for( problem, settings, warm )
        if self._sink is not None and on_card:
            self._solver_groups = None                   # ( the card's sink: no tree of aggregates, no aggregation -- `_build_card` )
        if self._radial is not None:
            if on_card:
                raise NotImplementedError( "radial_func: the CPU solves it, not the card yet ( `JAX_PLATFORMS=cpu`, `LOOM_DEVICE=cpu` )" )
            if periodicity is not None:
                raise NotImplementedError( "radial_func with a periodicity: not written yet" )
            self._solver_groups = None                   # ( the tree of aggregates does not know the radial factor )
        step = { "auto": "limits" if d == 2 and periodicity is None and self._sink is None and self._radial is None and self._solver_groups is None
                 else "trials" }.get( tun.step, tun.step )
        if step == "limits" and periodicity is not None:
            raise ValueError( "step = 'limits' is not available with a periodicity: 'trials' or 'auto'" )
        if step == "limits" and self._sink is not None:
            raise ValueError( "step = 'limits' is not available for a partial transport: 'trials' or 'auto'" )
        if step == "limits" and self._radial is not None:
            raise ValueError( "step = 'limits' is not available with a radial integrand: 'trials' or 'auto'" )

        if step == "limits" and d != 2:
            raise ValueError( "step = 'limits': 2D only, on the CPU as on the card ( the area polynomials along the direction, "
                              "`sdotplan/Bounds.h`, `gpu/Newton2D.cuh`; in 3D: 'trials' or 'auto' )" )
        if step not in _STEP:
            raise ValueError( f"unknown step: { tun.step !r } ( 'auto', 'trials' or 'limits' )" )
        if tun.amg_variant not in _AMG_VARIANT:
            raise ValueError( f"unknown amg_variant: { tun.amg_variant !r } ( { ', '.join( _AMG_VARIANT ) } )" )
        if tun.residual not in _RESIDUAL:
            raise ValueError( f"unknown residual: { tun.residual !r } ( { ', '.join( _RESIDUAL ) } )" )
        if tun.linear_solver not in _LIN:
            raise ValueError( f"unknown linear_solver: { tun.linear_solver !r } ( { ', '.join( _LIN ) } )" )

        # THE START. Three sources, in this order: the given plan, the given bare weights, the last
        # plan of the problem. `_start_from_plan` says what it could draw from it, and
        # `stats[ "warm_start" ]` reports it -- a silently discarded warm start is exactly what loses
        # an afternoon.
        plan = settings.ot_plan if settings.ot_plan is not None else warm
        impose = settings.ot_plan is not None
        if settings.weights0 is not None:
            w0_given, self._warm_start = settings.weights0, "weights0"
        elif plan is not None:
            w0_given, self._warm_start = self._start_from_plan( plan, src_dist, impose )
        else:
            w0_given, self._warm_start = None, "none"

        # THE diagram, built once on the positions ( see the module docstring ); the weights it
        # carries at a given moment are the last ones set
        # Under a trace ( `jax.jit` ) the weights may be tracers, and a diagram with traced weights falls back on the
        # plain storage. The zeros of a cold start are made on the host ( a constant ); given traced weights, the TREE is
        # built on the positions alone ( it only depends on them ) and the weights enter its majorants ( a kernel ). On the
        # card, see below.
        # ON THE CARD, what the solve will take from the card, checked before anything is launched ( `CardMemory.py` )
        self._card_mg_recycle = None
        if on_card and d in ( 2, 3 ):
            traced = any( loom.is_traced( getattr( x, "raw", x ) ) for x in ( src_dist.positions, src_dist.weights )
                          if getattr( x, "is_defined", True ) )
            self._card_mg_recycle = self._check_card_memory( int( src_dist.nb_diracs.value ), settings, jitted = traced, dim = d )

        accelerator = tun.accelerator
        pos_raw = getattr( src_dist.positions, "raw", src_dist.positions )
        import numpy as np
        w_start = np.zeros( int( src_dist.nb_diracs.value ) ) if w0_given is None else w0_given
        from .AaBsp import AaBsp, builds_on_card
        if periodicity is not None and accelerator is None:
            # the tree is that of the FOLDED seeds ( the diagram folds them the same way: `periodicity.setup_for` )
            from .periodicity import setup_for
            pos_raw = setup_for( pos_raw, dst_dist.periodic_support(), periodicity )[ 1 ]
        if accelerator is None and on_card and builds_on_card( pos_raw ) and self._solver_groups is not None:
            # GROUPS: the positions are readable ( the tree of aggregates was read from them ) and the groups need the tree's
            # order on the host ( `PowerDiagram_Bsp._stored_group_members` ): the card's tree built as a CONSTANT of a trace
            with loom.concrete_eval():
                accelerator = AaBsp( pos_raw )
                if w0_given is None:
                    accelerator.set_zero_weight_majorants()
            accelerator._majorant_weights = w_start
        elif accelerator is None and on_card and builds_on_card( pos_raw ):
            # ON THE CARD the tree is built by the card ( `gpu/Bsp2D.cuh`, one call ), HERE, outside the `concrete_eval`
            # below: under a `jit` it is then a part of the jitted program -- built at every call, from traced positions as
            # well as from constant ones -- and not a constant computed while tracing. The majorants of a cold start ( zero
            # weights ) are zero: no call for them
            if w0_given is None:
                accelerator = AaBsp( pos_raw )
                accelerator.set_zero_weight_majorants()
            else:
                accelerator = AaBsp( pos_raw, getattr( w_start, "raw", w_start ) )
            accelerator._majorant_weights = w_start
        elif w0_given is not None and accelerator is None and not loom.is_traced( pos_raw ) \
                and loom.is_traced( getattr( w0_given, "raw", w0_given ) ):
            accelerator = AaBsp( pos_raw )
        # ( and what is concrete is EVALUATED under the trace: the domain, the tree, the density's values stay readable )
        with loom.concrete_eval():
            self._pd = PowerDiagram( src_dist.positions, w_start,
                                     accelerator = accelerator, kernel_dtype = settings.kernel_dtype,
                                     distribution = dst_dist, memory = tun.memory,
                                     scratch_capacity = tun.scratch_capacity, periodicity = periodicity, sink_weight = self._sink, radial_func = self._radial,
                                     groups = self._solver_groups )
        pd = self._pd
        n = int( pd.nb_points.value )

        #: the target masses, indexed like the cells
        self._masses = RealTensor[ pd.num_point ]( src_dist.weights.raw )
        # THE AGGREGATION ( `sdotplan/Aggregation.h` ): its margin, and the exact duplicates, found here
        self._aggregation_inputs( pos_raw, settings, n )

        if on_card:
            return self._build_card( pd, settings, step, verbose, self._start_box( dst_dist, d ) )

        options = _Options(
            mass_tol = float( settings.tol ), mass_rtol = float( tun.mass_rtol ), t_min = float( tun.t_min ),
            mult_ok = float( tun.restart_factor ), factor = 0.9, beta0 = 0.25, mult_lim = 2.0, confidence = 0.0,
            conv_s0 = float( tun.conv_start or 0.0 ), conv_ratio = float( tun.conv_ratio ),
            conv_min = float( tun.conv_min or 0.0 ), conv_threshold = float( tun.conv_threshold ),
            residual_power = float( tun.residual_power ), lin_tol = float( tun.linear_tol or 0.0 ), residual_switch = float( tun.residual_switch ),
            max_iter = int( settings.max_iter ), max_backtracks = int( tun.max_backtracks ), lin = _LIN[ tun.linear_solver ], amg_variant = _AMG_VARIANT[ tun.amg_variant ],
            mg_pack = int( tun.mg_pack or 0 ), mg_recycle = -1 if tun.mg_recycle is None else int( tun.mg_recycle ), mg_rebuild = int( tun.mg_rebuild or 0 ), mg_stop = int( tun.mg_stop or 0 ), mg_nu = int( tun.mg_nu or 0 ),
            step = _STEP[ step ], residual = _RESIDUAL[ tun.residual ], trace = int( bool( verbose ) ), continuation = _CONTINUATION[ settings.continuation ],
            cap0 = int( pd._scratch_capacity ),
            agg_margin = self._agg_margin, agg_gap = self._agg_gap, agg_nb_dups = self._agg_nb_dups, keep_start = int( bool( getattr( tun, "keep_start", False ) ) ),
            kernel_fp_size = fp_size( pd.kernel_dtype ),
        )

        weights = RealTensor[ pd.num_point ]()
        history = _History( nb_hist = len( _HISTORY ), nb_points = n )
        # THE AXES ARE THOSE OF THE DIAGRAM, and it is the one thing not to miss here: a `num_point`
        # axis built separately carries the same NAME without being the same, and
        # `cell_masses * ( positions - barycenters )` then becomes an outer product `[ n, n, d, d ]`
        # instead of the gradient. They are therefore declared from `pd`.
        cell_masses = RealTensor[ pd.num_point ]()
        barycenters = RealTensor[ pd.num_point, pd.dim ]()
        cost        = RealTensor()
        clusters    = IntTensor[ pd.num_point ]()
        stats = RealTensor[ Axis( ShapeVar( len( _STATS ) ), name = "num_stat" ) ]()
        # ( with groups: the outer weights -- the solver takes the virtual ones from the diagram )
        w0 = RealTensor[ pd.num_point ]( pd.weights.raw )

        dom = pd._domain_cell()
        dist_expr, _, dist_kwargs = pd._dist_for()
        pd_expr, pd_kwargs, pd_produced = pd._solver_weights_call()
        groups_produced = None
        if pd.group_rep.is_defined:                      # the virtual weights of the tree are written by the solver too
            groups_produced = ( RealTensor[ pd.num_point ](), RealTensor[ pd.num_agg ]() )
            pd_expr = f"{ pd_expr }.with_groups( outputs.group_vweights_out, outputs.group_agg_vweights_out )"
            pd_kwargs = dict( pd_kwargs, group_vweights_out = loom.out( groups_produced[ 0 ] ), group_agg_vweights_out = loom.out( groups_produced[ 1 ] ) )

        loom.ffi_call(
            "sdotplan_solve",
            # `handler` and not the default scaffolded kernel: this body IS the handler. It is
            # HOST code -- it needs the `queue`, and it drives its own parallelism ( a hundred
            # diagrams in a single call ), so there is no per-item functor nor `run_parallel`
            # to generate around it. See the docstring of `FfiCode`.
            # `inline`: the body IS that of the handler, loom only writes its wrapper
            # ( `void kernel( queue, batch_axes, args )` ). It is HOST code -- it needs the
            # `queue`, and it drives its own parallelism ( a hundred diagrams in a single call ),
            # so there is no per-item functor nor `run_parallel` to generate around it.
            FfiCode.inline( includes = [ "sdot/sdotplan/Solve.h" ],
                sources = [ "sdot/sdotplan/Linear.cpp" ],
                code = "\n".join( [
                    # the expressions that `PowerDiagram` generates name `inputs` / `outputs` ( the shape
                    # of a per-item kernel ): we FIND them again here under their name, and nothing needs to know it
                    "auto &inputs = args.inputs; auto &outputs = args.outputs;",
                    "using TK_sdotplan = std::conditional_t<CT_VALUE( inputs.options.kernel_fp_size ) == 64, double, float>;",
                    f"auto pd_sdotplan = { pd_expr };",
                    "sdotplan::SolverOptions os;",
                    "sdotplan::NewtonOptions &no = os.newton;",
                    "no.tol_abs = double( inputs.options.mass_tol ); no.tol_rel = double( inputs.options.mass_rtol ); no.t_min = double( inputs.options.t_min );",
                    "no.mult_ok = double( inputs.options.mult_ok ); no.factor = double( inputs.options.factor ); no.beta0 = double( inputs.options.beta0 );",
                    "no.mult_lim = double( inputs.options.mult_lim ); no.confidence = double( inputs.options.confidence );",
                    "no.maxit = int( SI( inputs.options.max_iter ) ); no.max_backtracks = int( SI( inputs.options.max_backtracks ) );",
                    "no.step = int( SI( inputs.options.step ) ); no.trace = SI( inputs.options.trace ) != 0;",
                    "no.residual = int( SI( inputs.options.residual ) ); no.power = double( inputs.options.residual_power ); no.switch_residual = double( inputs.options.residual_switch );",
                    "os.lin = sdotplan::Lin( int( SI( inputs.options.lin ) ) ); os.cap0 = SI( inputs.options.cap0 );",
                    "os.lin_options.tol = double( inputs.options.lin_tol ); os.lin_options.amg_variant = int( SI( inputs.options.amg_variant ) );",
                    "os.lin_options.mg_pack = int( SI( inputs.options.mg_pack ) ); os.lin_options.mg_recycle = int( SI( inputs.options.mg_recycle ) ); os.lin_options.mg_rebuild = int( SI( inputs.options.mg_rebuild ) ); os.lin_options.mg_stop = int( SI( inputs.options.mg_stop ) ); os.lin_options.mg_nu = int( SI( inputs.options.mg_nu ) );",
                    "os.continuation = int( SI( inputs.options.continuation ) ); os.continuation_threshold = double( inputs.options.conv_threshold );",
                    "os.conv_s0 = double( inputs.options.conv_s0 ); os.conv_ratio = double( inputs.options.conv_ratio ); os.conv_min = double( inputs.options.conv_min );",
                    "os.agg_margin = double( inputs.options.agg_margin ); os.agg_gap = double( inputs.options.agg_gap ); os.agg_nb_dups = SI( inputs.options.agg_nb_dups ); os.keep_start = SI( inputs.options.keep_start ) != 0;",
                    f"sdotplan::solve<TK_sdotplan>( queue, pd_sdotplan, inputs.power_diagram, inputs.dom_cell, { dist_expr }, inputs.nu, inputs.w0, inputs.dups, inputs.start_box, os, "
                    "outputs.weights, outputs.history, outputs.stats, outputs.cell_masses, outputs.barycenters, outputs.cost, outputs.clusters );",
                ] ) ),
            power_diagram = pd,
            dom_cell = dom,
            nu = self._masses,
            w0 = w0,
            dups = self._agg_dups,
            start_box = self._start_box( dst_dist, d ),
            options = options,
            weights = loom.out( weights ),
            # `nb_steps` is WRITTEN by the kernel ( the number of accepted steps ), so it must be
            # named: a `ShapeVar` that is not declared stays unbound, and the C++ only sees a null
            # view. The weights of each step, for their part, are only written if they were requested --
            # an unnamed `weights` stays observed, hence neither allocated nor read.
            history = loom.out( history, writes = ( [ "rows", "nb_steps", "weights" ] if settings.keep_weights
                                                    else [ "rows", "nb_steps" ] ),
                                capacities = { "nb_steps": int( settings.max_iter ) + 1 } ),
            stats = loom.out( stats ),
            cell_masses = loom.out( cell_masses ),
            barycenters = loom.out( barycenters ),
            cost = loom.out( cost ),
            clusters = loom.out( clusters ),
            has_dynamic_capacity = False,
            **pd_kwargs,
            **dist_kwargs,
        )
        pd._solver_weights_after( pd_produced )
        if groups_produced is not None:
            pd.group_vweights, pd.group_agg_vweights = groups_produced[ 0 ].raw, groups_produced[ 1 ].raw

        #: the weights of the power diagram of the plan, `[ n ]` ( defined up to a constant: `weights[ 0 ] == 0` )
        self.weights = weights
        self._weights_fit = weights
        #: the mass of each cell, `[ n ]`
        self.cell_masses = cell_masses
        #: the barycenter of each cell, `[ n, d ]` ( its dirac if the cell is empty ) -- with a periodicity, next to the dirac
        #: as it was given ( the cells are built around the folded diracs: the folding is undone here )
        self.barycenters = barycenters if pd._periodic is None else RealTensor[ pd.num_point, pd.dim ]( pd.unfold( barycenters.raw ) )
        #: the transport cost, `W_2^2`
        self.cost = cost
        self._clusters = clusters
        self._read_stats( stats, settings )
        self._read_history( history, settings, pd )
        self._settle_duplicates( settings )
        self._check_converged( settings )

    @staticmethod
    def _start_box( dist, d ):
        """THE BOX THE STARTING POINT PACKS THE SEEDS IN, `[ flag, lo, hi ]` ( `sdotplan/Solve.h::similarity` ): what the target says lies
        inside its support ( `Distribution.inscribed_box` ), flag 1 -- or flag 0, and the solver takes the box of the domain."""
        box = dist.inscribed_box() if hasattr( dist, "inscribed_box" ) else None
        vals = np.zeros( 2 * d + 1 ) if box is None else np.concatenate( [ [ 1.0 ], np.asarray( box[ 0 ], dtype = float ), np.asarray( box[ 1 ], dtype = float ) ] )
        return RealTensor[ Axis( ShapeVar( 2 * d + 1 ), name = "num_start_box" ) ]( vals )

    def _aggregation_inputs( self, pos_raw, settings, n ):
        """THE AGGREGATION'S INPUTS ( `sdotplan/Aggregation.h` ): `_agg_margin` ( 0: off ), and the EXACT DUPLICATES, which the
        diagram cannot see ( no plane between two equal points ): `_agg_dups` ( `2 k` values, the pairs `( dup, rep )`, one
        dummy pair when there is none ), `_agg_nb_dups`, `_agg_gap`. Found by a hash of the positions when they are readable on the host;
        traced positions ( `jax.jit` ) are not checked ( `stats[ "duplicates" ]` says so ). The near-coincident pairs are
        found by the solver itself, on its diagrams."""
        import numpy as np
        self._agg_margin, self._agg_gap, self._agg_nb_dups = 0.0, 0.0, 0
        self._dup_note = "not checked ( aggregate = False )"
        self._exact_dups = None
        dup = rep = np.zeros( 0, dtype = np.int64 )
        if getattr( self, "_solver_groups", None ) is not None:
            # GROUPS: the diagram splits what the doubles cannot separate, Newton runs on all its weights -- nothing to merge; the
            # margin is that of the criterion with which the solver finds the pairs that need to join the groups
            self._dup_note = "checked ( groups )"
            margin = getattr( settings.tuning, "aggregation_margin", None )
            self._agg_margin = float( _AGGREGATE_MARGIN if margin is None else margin )
        elif settings.aggregate:
            margin = getattr( settings.tuning, "aggregation_margin", None )
            self._agg_margin = float( _AGGREGATE_MARGIN if margin is None else margin )
            if loom.is_traced( pos_raw ):
                self._dup_note = "not checked ( traced positions )"
            else:
                from loom.tensor.host import to_host     # any framework, any device ( a cupy array refuses `np.asarray` )
                P = np.asarray( to_host( pos_raw ), dtype = np.float64 ).reshape( n, -1 )
                dup, rep = getattr( self, "_dup_hash", None ) or exact_duplicates( P )        # ( the hash of `_groups_for`, if any )
                self._dup_note = "checked"
                self._exact_dups = ( dup, rep ) if len( dup ) else None
                if len( dup ):
                    ext = P.max( axis = 0 ) - P.min( axis = 0 )
                    scale2 = max( float( ( P * P ).sum( axis = 1 ).max() ), float( ( ext * ext ).sum() ) )
                    self._agg_gap = _DUPLICATE_GAP * ( scale2 if scale2 > 0 else 1.0 )
                    self._agg_nb_dups = len( dup )
        k = max( len( dup ), 1 )
        rows = np.zeros( ( k, 2 ), dtype = np.int64 )
        rows[ :len( dup ), 0 ] = dup
        rows[ :len( dup ), 1 ] = rep
        self._agg_dups = IntTensor[ Axis( ShapeVar( 2 * k ), name = "num_dup" ), dict( size = 64 ) ]( rows.reshape( -1 ) )

    @staticmethod
    def _card_memory_kw( n, settings, dim = 2 ):
        """what `CardMemory.card_solve_bytes` needs to know of the card's solve of `n` seeds with these `settings` ( the same
        choices as `_build_card`: the kernels, the linear solver and its levels, the last pass's slots )"""
        from .AaBsp import AaBsp
        from .PowerDiagram_Bsp import PowerDiagram_Bsp, card_overflow_slot_bytes, card_overflow_warps_for, card_variant_for
        tun = settings.tuning
        nodes = AaBsp.max_nb_nodes_for( n )
        widest = card_variant_for( 32 if settings.precision == "fp32" else 64, n, nodes, dim )
        warps = card_overflow_warps_for( widest, n, PowerDiagram_Bsp.card_max_vertices, PowerDiagram_Bsp.card_overflow_warps,
                                         PowerDiagram_Bsp.card_overflow_bytes )
        warps = ( warps + 3 ) // 4 * 4                    # ( whole blocks of four warps, `Overflow::sized`, `Slots::sized` )
        overflow_bytes = warps * card_overflow_slot_bytes( widest, n, PowerDiagram_Bsp.card_max_vertices )
        lin_name = tun.linear_solver
        linear = "host" if getattr( tun, "linear_host", False ) or lin_name in ( "cholesky", "amg" ) else "cg" if lin_name == "cg" else "mg"
        return dict( nb_nodes = nodes, precision = settings.precision, linear = linear,
                     mg_float = ( getattr( tun, "mg_precision", None ) or _CARD_MG_PRECISION ) == "float",
                     recycle = 2 if tun.mg_recycle is None else int( tun.mg_recycle ),
                     shift = int( tun.mg_pack or _CARD_MG_PACK[ dim ] ).bit_length() - 1,
                     smoothed = 1 if getattr( tun, "mg_smoothed", None ) is None else int( tun.mg_smoothed ),
                     stop = min( max( int( tun.mg_stop or 64 ), 16 ), 2048 ),
                     overflow_bytes = overflow_bytes, max_iter = int( settings.max_iter ),
                     keep_weights = bool( settings.keep_weights ), dim = dim )

    @staticmethod
    def _check_card_memory( n, settings, jitted = False, dim = 2 ):
        """THE CARD'S MEMORY, before the tree and the solve ( `CardMemory.check_card_memory` ): a `MemoryError` that says what
        to do when the solve of `n` seeds does not fit in what XLA's pool has left. A VARIANT is chosen here when it helps: the
        multigrid's recycled solutions ( 16 bytes per seed each ) are given up, one then the other, before refusing -- unless
        `mg_recycle` was set. Returns the recycled solutions to ask the card for ( `None`: the default ). `jitted`: while
        tracing, the program's own buffers are counted from the shapes ( `card_solve_bytes( jitted = True )` ), which the pool
        cannot show yet -- so that a jitted solve gives up its recycled solutions, or refuses, BEFORE XLA's ten seconds."""
        from .CardMemory import check_card_memory
        kw = { **SdotPlanNd._card_memory_kw( n, settings, dim ), "jitted": bool( jitted ) }
        if kw[ "linear" ] != "mg" or settings.tuning.mg_recycle is not None:
            check_card_memory( n, **kw )
            return None
        first = None
        for recycle in ( 2, 1, 0 ):
            try:
                check_card_memory( n, **{ **kw, "recycle": recycle } )
                return None if recycle == 2 else recycle
            except MemoryError as e:
                first = first or e
        raise first

    def _build_card( self, pd, settings, step, verbose, start_box ):
        """THE CARD'S SOLVE ( `src/sdot/include/sdot/gpu/Newton2D.cuh` ): the same Newton, the same options and outputs as the CPU's, in
        ONE ffi call whose handler drives the loop on the call's stream -- so it runs under `jax.jit` too. Taken in 2D and 3D,
        with the BSP tree and a box domain ( `PowerDiagram_Bsp._card_takes` ); anything else is refused here rather than solved
        on another path.

        IN 3D ( `gpu/Cell3D.cuh`'s cells, the same solver ): a constant density, the step `trials` ( `limits` is refused in
        `_build`, as on the CPU ), the multigrid's packets of 8 tree ranks ( `_CARD_MG_PACK` ); a mixed solve that the float
        kernel finished takes its moments in float.

        The DENSITY ( `PowerDiagram_Bsp._card_solve_density`, `gpu/Density2D.cuh` ): a constant, an `Image` on a regular grid
        ( a diagonal frame, uniform knots ), or a sum of isotropic gaussians -- integrated on the cells' edges ( Green on the
        rows of the image, the polar corners and `erf`s of the gaussians ), never by cutting the cells. The WIDTH CONTINUATION
        is the CPU's ( `sdotplan/Continuation.h`: the same options, stages and targets; the image blurred on the card by the
        same filter, the gaussians widened ).

        What differs from the CPU ( said in `Newton2D.cuh` ): `step = "limits"` is the exact step of the area polynomials
        ( all the cells, from the accepted diagram's edges ) checked by the trial diagram, not the CPU's local limits with a
        first trial `beta` -- for a density that is not a constant, the frozen cells' masses bisected ( the CPU bisects exact
        cells ); the linear solver is the card's multigrid ( `"auto"`, `"mg"` ) or CG ( `"cg"` ), or the CPU's through a copy
        of the laplacian ( `"cholesky"`, `"amg"`, or `Tuning( linear_host = True )` ); the moments of a gaussian density are
        closed forms ( the CPU's a quadrature )."""
        import numpy as np
        tun = settings.tuning
        reason = None
        dens = None
        d = pd.dim_count
        if d not in ( 2, 3 ):
            reason = f"the card solves in 2D and 3D ( not { d }D )"
        elif pd._periodic is not None and pd._periodic.linear is not None:
            reason = "with a periodicity, the card solves for groups of translations ( its cells walk the images; rotations and symmetries: the CPU )"
        elif not pd._card_takes( solver = True ):
            reason = "the card's cells take a box domain, the BSP tree and no neighbour memory"
        else:
            dens = pd._card_solve_density()
            if d == 3 and dens is not None and dens[ "kind" ] not in ( "const", "mesh", "gauss" ):
                dens = None
            if self._sink is not None and dens is not None and ( d != 2 or dens[ "kind" ] not in ( "const", "mesh" ) ):
                dens = None
                reason = ( "with a sink ( partial transport ), the card integrates in the balls a constant density or a triangle "
                           "`Mesh`, in 2D -- not an image nor gaussians" )
            if pd._periodic is not None and dens is not None and dens[ "kind" ] != "const" and not pd._periodized:
                dens = None                              # ( the card's images are not repeated by a group; gaussians and meshes
                                                         # repeat themselves: `Distribution.periodized` )
                reason = ( "with a periodicity, the card integrates a constant density ( a box, a polytope, an image of equal "
                           "values ), gaussians or a mesh -- its images are not repeated by the group" )
            if dens is None and reason is None:
                reason = ( "the card integrates a constant density, an `Image` on a regular grid ( a diagonal frame, uniform knots, "
                           "covering the box ), isotropic gaussians or a triangle `Mesh` in 2D -- not this distribution" if d == 2 else
                           "in 3D the card integrates a constant density ( Lebesgue on the box, or an `Image` of equal values "
                           "covering it ), isotropic gaussians or a tetrahedral `Mesh` -- not this distribution" )
        if reason is not None:
            raise NotImplementedError( f"SdotPlanNd on a CUDA device: { reason }. Use the CPU device ( LOOM_DEVICE=cpu ) "
                                       "for this problem." )
        n = int( pd.nb_points.value )
        # THE KERNEL'S FLOAT, chosen here: `fp64` / `fp32` one kernel; `auto` / `mixed` the float kernel, then the double one
        # from the iteration where a float step stagnates ( `Newton2D.cuh`: a cut decided in float on a degenerate cloud stops
        # the merit from decreasing -- the lines with equal areas -- while elsewhere the float kernel's measures, re-solved in
        # double, reach the double's residual for a fifth of its time )
        from .PowerDiagram_Bsp import card_variant_for
        nodes = int( pd.tree.nb_bsp_nodes.value )
        nid = n * ( 1 if pd._periodic is None else len( pd._periodic.shifts ) )    # ( the cuts' identifiers: the images' too )
        if settings.precision in ( "auto", "mixed" ):
            variant = f"{ card_variant_for( 32, nid, nodes, d ) }, { card_variant_for( 64, nid, nodes, d ) }"
        else:
            variant = card_variant_for( 32 if settings.precision == "fp32" else 64, nid, nodes, d )
        # the cells' last pass: one budget for every diagram of the solve, sized on the widest kernel it runs
        overflow_warps = pd._card_overflow_warps( card_variant_for( 32 if settings.precision == "fp32" else 64, nid, nodes, d ) )

        # the linear solver: the card's ( `Linear2D.cuh` ), or a CPU one of `Linear.cpp` on a copy of the laplacian
        lin_name = tun.linear_solver
        if getattr( tun, "linear_host", False ) or lin_name in ( "cholesky", "amg" ):
            lin_kind = 2
        else:
            lin_kind = 0 if lin_name == "cg" else 1
        pack = int( tun.mg_pack or 0 )
        if pack and ( pack & ( pack - 1 ) ):
            raise ValueError( f"mg_pack must be a power of two ( got { pack } )" )
        if not pack and lin_kind != 2:                   # ( the card's multigrid: its packets per dimension; a host solver keeps its own )
            pack = _CARD_MG_PACK[ d ]
        opts = [ 0.0 ] * len( _CARD_OPTIONS )
        def put( name, v ):
            opts[ _CARD_OPTIONS.index( name ) ] = float( v )
        # the box where the similarity start packs the seeds ( `_start_box`: the target's inscribed box, flag 0 for the domain's )
        sb = np.asarray( start_box.raw, dtype = float ).reshape( -1 )
        put( "start_box", sb[ 0 ] )
        for k in range( d ):
            put( f"start_lo{ k }", sb[ 1 + k ] ); put( f"start_hi{ k }", sb[ 1 + d + k ] )
        put( "tol_abs", settings.tol ); put( "tol_rel", tun.mass_rtol ); put( "t_min", tun.t_min ); put( "mult_ok", tun.restart_factor )
        put( "factor", 0.9 ); put( "maxit", settings.max_iter ); put( "max_backtracks", tun.max_backtracks ); put( "step", _STEP[ step ] )
        put( "residual", _RESIDUAL[ tun.residual ] ); put( "power", tun.residual_power ); put( "switch", tun.residual_switch )
        put( "lin", lin_kind ); put( "host_method", _LIN[ lin_name ] ); put( "lin_tol", tun.linear_tol or 0.0 )
        put( "amg_variant", _AMG_VARIANT[ tun.amg_variant ] ); put( "mg_shift", pack.bit_length() - 1 if pack else 0 )
        recycle = tun.mg_recycle if tun.mg_recycle is not None else getattr( self, "_card_mg_recycle", None )
        put( "mg_recycle", -1 if recycle is None else recycle ); put( "mg_rebuild", tun.mg_rebuild or 0 )
        put( "mg_stop", tun.mg_stop or 0 ); put( "mg_nu", tun.mg_nu or ( _CARD_MG_NU[ d ] if lin_kind != 2 else 0 ) )
        put( "mg_kcycle", -1 if getattr( tun, "mg_kcycle", None ) is None else tun.mg_kcycle )
        put( "lin_maxit", 0 ); put( "trace", int( bool( verbose ) ) )
        mg_precision = getattr( tun, "mg_precision", None ) or _CARD_MG_PRECISION
        if mg_precision not in ( "float", "double" ):
            raise ValueError( f"mg_precision: 'float' or 'double' ( got { mg_precision !r } )" )
        put( "mg_float", int( mg_precision == "float" ) )
        put( "mg_smoothed", -1 if getattr( tun, "mg_smoothed", None ) is None else tun.mg_smoothed )
        put( "continuation", _CONTINUATION[ settings.continuation ] ); put( "conv_threshold", tun.conv_threshold )
        put( "conv_s0", tun.conv_start or 0.0 ); put( "conv_ratio", tun.conv_ratio ); put( "conv_min", tun.conv_min or 0.0 )
        put( "conv_possible", int( dens.get( "conv_possible", True ) ) ); put( "min_scale", dens.get( "min_scale", 0.0 ) )
        for name, v in zip( ( "img_x0", "img_y0", "img_hx", "img_hy", "img_nx", "img_ny" ), dens.get( "geom", [ 0.0 ] * 6 ) ):
            put( name, v )
        grouped = pd.group_rep.is_defined                # ( the tree of aggregates: no aggregation, `Newton2D.cuh` )
        sunk = self._sink is not None                    # ( a sink: no aggregation on the card, the residual `nu - a`: `Newton2D.cuh` )
        put( "agg_margin", 0.0 if grouped or sunk else self._agg_margin ); put( "agg_gap", self._agg_gap )
        put( "agg_nb_dups", 0 if grouped or sunk else self._agg_nb_dups )
        put( "sink", int( sunk ) ); put( "sink_w", float( self._sink ) if sunk else 0.0 )
        options = RealTensor[ Axis( ShapeVar( len( _CARD_OPTIONS ) ), name = "num_card_opt" ) ]( np.asarray( opts, dtype = np.float64 ) )

        weights = RealTensor[ pd.num_point ]()
        history = _History( nb_hist = len( _HISTORY ), nb_points = n )
        cell_masses = RealTensor[ pd.num_point ]()
        barycenters = RealTensor[ pd.num_point, pd.dim ]()
        cost        = RealTensor()
        clusters    = IntTensor[ pd.num_point ]()
        stats = RealTensor[ Axis( ShapeVar( len( _STATS ) ), name = "num_stat" ) ]()
        w0 = RealTensor[ pd.num_point ]( pd.weights.raw )
        work = _CardSolveWork()
        pd_expr, pd_kwargs, pd_produced = pd._solver_weights_call()
        limits = f"{ int( pd.card_max_vertices ) }, { overflow_warps }"

        # the density: its tensors, and how the handler hands them over ( `DensityHost2D.cuh` )
        dens_expr, dens_args = card_density_call( dens, d )
        name = { "const": "sdotplan_solve_card_2d", "image": "sdotplan_solve_card_2d_image", "mesh": f"sdotplan_solve_card_{ d }d_mesh",
                 "gauss": f"sdotplan_solve_card_{ d }d_gauss" }[ dens[ "kind" ] ]

        # GROUPS: the virtual weights of the tree come out too
        groups_produced, group_outs, group_kwargs = None, ", sdot::gpu2d::NoOut{}, sdot::gpu2d::NoOut{}", {}
        if grouped:
            groups_produced = ( RealTensor[ pd.num_point ](), RealTensor[ pd.num_agg ]() )
            group_outs = ", args.outputs.group_vweights_out, args.outputs.group_agg_vweights_out"
            group_kwargs = dict( group_vweights_out = loom.out( groups_produced[ 0 ] ), group_agg_vweights_out = loom.out( groups_produced[ 1 ] ) )

        loom.ffi_call(
            name,
            FfiCode.inline(
                f"sdot::gpu2d::solve<{ variant }>( queue, args.inputs.power_diagram, args.inputs.nu, args.inputs.w0, args.inputs.dups, args.inputs.options, "
                "args.outputs.weights, args.outputs.history, args.outputs.stats, args.outputs.cell_masses, args.outputs.barycenters, "
                "args.outputs.cost, args.outputs.clusters, args.outputs.sorted_weights_out, args.outputs.node_wa_out, args.outputs.node_wb_out, args.outputs.work, "
                f"args.errors, args.allocator, { dens_expr }, { limits }{ group_outs } );",
                includes = [ "sdot/gpu/Newton2D.cuh" ], sources = [ "sdot/sdotplan/Linear.cpp" ], allocator = True ),
            failures = pd._card_failures(),
            power_diagram = pd,
            nu = self._masses,
            w0 = w0,
            dups = self._agg_dups,
            options = options,
            **dens_args,
            weights = loom.out( weights ),
            history = loom.out( history, writes = ( [ "rows", "nb_steps", "weights" ] if settings.keep_weights
                                                    else [ "rows", "nb_steps" ] ),
                                capacities = { "nb_steps": int( settings.max_iter ) + 1 } ),
            stats = loom.out( stats ),
            cell_masses = loom.out( cell_masses ),
            barycenters = loom.out( barycenters ),
            cost = loom.out( cost ),
            clusters = loom.out( clusters ),
            work = loom.out( work, capacities = { "nb_facets": card_facet_capacity( n, d ) } ),
            **pd_kwargs,
            **group_kwargs,
        )
        pd._solver_weights_after( pd_produced )
        if groups_produced is not None:
            pd.group_vweights, pd.group_agg_vweights = groups_produced[ 0 ].raw, groups_produced[ 1 ].raw
        self.weights = weights
        self._weights_fit = weights
        self.cell_masses = cell_masses
        self.barycenters = barycenters if pd._periodic is None else RealTensor[ pd.num_point, pd.dim ]( pd.unfold( barycenters.raw ) )
        self.cost = cost
        self._clusters = clusters
        self._read_stats( stats, settings )
        self._read_history( history, settings, pd )
        self._settle_duplicates( settings )
        self._check_converged( settings )

    @staticmethod
    def _start_from_plan( plan, src_dist, impose ):
        """What a previous PLAN provides as a start: `( weights0, what_was_taken_back )`.

        A plan whose number of diracs no longer matches is worthless ( the common case: a stage
        of multi-scale ). It is dropped, but REPORTED -- and if the caller had imposed it
        explicitly through `ot_plan`, we raise, because they believe they are warm-starting and
        are not.

        The CLUSTERS are not taken back: each solve finds them again ( the exact duplicates by a
        hash of the positions, the near-coincident pairs on its own diagrams, from the first
        iteration when the inherited weights already put them past the criterion --
        `sdotplan/Aggregation.h` ). This is where they would go through, should the detection ever
        cost something: `plan.clusters` and the positions it was solved for."""
        n_plan = int( plan.weights.shape[ 0 ] )
        n = int( src_dist.nb_diracs.value )
        if n_plan != n:
            if impose:
                raise ValueError( f"Iterative( ot_plan = ... ): this plan carries { n_plan } weights and the "
                                  f"source has { n } -- it cannot serve as a start. Remove it, "
                                  "or keep an `OtProblem` ( it expires its solution on its own )" )
            return None, f"none ( the kept plan carries { n_plan } weights, the source has { n } )"
        return getattr( plan, "_weights_fit", plan.weights ), "ot_plan"

    def _read_stats( self, stats, settings ):
        #: how the solve went, a dict: `status`, `nb_iter`, `residual`, `nb_diag` ( the power diagrams computed ), timings...
        st = stats.raw
        if loom.is_traced( st ):
            # under a trace ( `jax.jit` ): the numbers are tracers, kept as such -- nothing is read on the host
            self.stats = { name: st[ k ] for k, name in enumerate( _STATS ) }
            self.stats[ "duplicates" ] = self._dup_note
            self.stats[ "warm_start" ] = self._warm_start
            return
        self.stats = { name: float( st[ k ] ) for k, name in enumerate( _STATS ) }
        self.stats[ "status" ] = _STATUS.get( int( self.stats[ "status" ] ), "?" )
        self.stats[ "it_switch" ] = int( self.stats[ "it_switch" ] )
        self.stats[ "it_double" ] = int( self.stats[ "it_double" ] )
        self.stats[ "start" ] = _START.get( int( self.stats[ "start" ] ), "?" )
        for name in ( "nb_iter", "nb_diag", "nb_backtracks", "nb_overflowed", "nb_cell_lim", "nb_limit_rounds",
                      "lin_nb_hierarchies", "lin_nb_iter", "nb_continuation_steps", "scratch_bytes",
                      "nb_clusters", "nb_aggregated", "nb_duplicates", "nb_polish" ):
            self.stats[ name ] = int( self.stats[ name ] )
        #: THE AGGREGATION ( `sdotplan/Aggregation.h` ), said in words: what was merged, and whether the exact duplicates were
        #: looked for ( `duplicates` ). `residual` is the aggregated problem's residual, `residual_full` the full one's.
        if not settings.aggregate:
            self.stats[ "aggregation" ] = "off"
        elif self.stats[ "nb_clusters" ] == 0:
            self.stats[ "aggregation" ] = "none"
        else:
            self.stats[ "aggregation" ] = ( f"{ self.stats[ 'nb_aggregated' ] } seeds in { self.stats[ 'nb_clusters' ] } clusters "
                                            f"( { self.stats[ 'nb_duplicates' ] } exact duplicates ), re-split in { self.stats[ 'nb_polish' ] } diagrams" )
        self.stats[ "duplicates" ] = self._dup_note
        #: what the warm start provided: `"ot_plan"`, `"weights0"`, or `"none"` ( and why )
        self.stats[ "warm_start" ] = self._warm_start

    def _read_history( self, history, settings, pd ):
        #: one dict per Newton step ( the first is the starting point ): `t` the step length, `residual_l2`,
        #: `max_abs_residual`, `min_measure`, `nb_diag`... and `weights` with `Iterative( keep_weights = True )`
        if loom.is_traced( history.rows.raw ):
            self.history = None                      # under a trace: not read ( it would be a host read of a tracer )
            return
        nb_steps = int( history.nb_steps.value )
        rows = history.rows.raw[ :nb_steps ]
        self.history = []
        for s in range( nb_steps ):
            entry = { name: float( rows[ s, k ] ) for k, name in enumerate( _HISTORY ) }
            entry[ "step" ] = int( entry[ "step" ] )
            if settings.keep_weights:
                entry[ "weights" ] = RealTensor[ pd.num_point ]( history.weights.raw[ s ] )
            self.history.append( entry )

    def _check_converged( self, settings ):
        """`OtNotConverged` when the solve ended without meeting its tolerance ( `Iterative( on_failure )` )"""
        if getattr( settings, "on_failure", "raise" ) != "raise" or loom.is_traced( self.stats[ "residual" ] ):
            return
        if not self.converged:
            from .OtProblem import OtNotConverged
            raise OtNotConverged( self )

    # -- what the solution says ----------------------------------------------------------------

    @property
    def converged( self ):
        """Whether the tolerance was reached.

        = Internals

        the test passed -- by the full problem, or by the aggregated one ( `"converged (aggregated)"`: the seeds that the
        doubles cannot separate answer as a cluster, see the module docstring )"""
        return self.stats[ "status" ] in ( "converged", "converged (aggregated)" )

    @property
    def groups( self ):
        """The groups of the plan ( `sdot.Groups`: seeds that share a cell and split it, `Iterative( duplicates = "split" )` ), else `None`.

        = Internals

        The TREE OF AGGREGATES of the solve ( `groups.Groups`: the separations of scales of the positions, or the clusters of the
        aggregation as `_split_groups` makes them ), with the precise virtual weights it found -- `PowerDiagram( positions,
        plan.weights, groups = plan.groups )` is the diagram of the plan. `None` without aggregates, with `duplicates = "merge"`,
        or where the groups could not be made ( traced positions, a periodicity )."""
        return getattr( self, "_groups", None )

    @property
    def clusters( self ):
        """Which cluster each dirac belongs to ( `[ n ]`, the smallest index of its cluster ) when diracs too close to be separated were merged, else `None`.

        = Internals

        Which CLUSTER each dirac belongs to -- `[ n ]`, the smallest index of its cluster ( itself when it is alone ) -- or
        `None` when none was merged.

        `None` is the common case and it means "the weights suffice". As soon as there are clusters, the solver's masses of
        the members are only within the floor of the doubles, an exact duplicate's cell being empty -- see the module docstring
        and README § 23.11; with `duplicates = "split"` the plan's GROUPS then split them exactly ( `groups`, which supersedes
        this attribute: kept for what reads it )."""
        if getattr( self, "_groups", None ) is not None:
            return self._clusters                        # ( the solver writes the representatives of the groups )
        nb = self.stats[ "nb_clusters" ]
        if not loom.is_traced( nb ) and int( nb ) == 0:
            return None
        return self._clusters

    @property
    def target_masses( self ) -> Tensor:
        """The mass each cell must reach ( `[ n ]` ): the masses of the diracs, normalized.

        = Internals

        `nu`: the target mass of each cell ( the masses of the diracs, normalized, then rescaled
        to what the domain contains -- see `sdotplan/Solve.h` )"""
        if getattr( self, "_sink", None ) is not None or getattr( self, "_radial", None ) is not None:
            return self._masses                          # partial or radial: the masses of the diracs themselves ( no rescaling )
        return self._masses * ( self.stats[ "domain_mass" ] / float( self._masses.sum() ) )

    def residual( self, weights = None ) -> Tensor:
        """The mass error of each cell, `[ n ]`, at the weights of the plan ( or at `weights` ). Costs one more power diagram.

        = Internals

        `measure_i( weights ) - nu_i` -- ZERO at the sought point, DIFFERENTIABLE with respect to
        `weights` ( see `PowerDiagram.measures` ). One more diagram: it is a diagnostic tool, not
        an output of the solve ( `cell_masses` is )."""
        if weights is None:
            return self.power_diagram().measures - self.target_masses
        return self.power_diagram( weights ).measures - self.target_masses

    def _groups_for( self, problem, settings, warm ):
        """THE TREE OF AGGREGATES OF A SOLVE ( `duplicates = "split"`, `groups.py` ): that of the plan it starts from when the positions
        are the same ( its precise virtual weights with it ), found otherwise ( `groups.find_groups`, `Tuning( aggregation_kappa )`:
        the separations of scales of the positions, the exact duplicates included ) -- `None` when there is none. The hash of the
        positions is kept for the aggregation's inputs ( one per solve )."""
        from . import groups as grp
        pos = getattr( problem.source.positions, "raw", problem.source.positions )
        P = grp._host( pos )
        self._tree_positions = P                         # ( what the tree is read from: a plan's problem may have new positions since )
        plan = settings.ot_plan if settings.ot_plan is not None else warm
        if plan is not None and getattr( plan, "groups", None ) is not None:
            old = getattr( plan, "_tree_positions", None )
            if old is not None and np.array_equal( old, P ):
                return plan.groups
        self._dup_hash = exact_duplicates( P )
        kappa = getattr( settings.tuning, "aggregation_kappa", grp.KAPPA )
        return grp.find_groups( P, kappa = kappa, duplicates = self._dup_hash )

    def _settle_duplicates( self, settings ):
        """THE CONVENTION FOR THE MERGED SEEDS ( `Iterative( duplicates = ... )` ): `"split"`, the groups and their own masses
        ( solved with them, `sdotplan/Groups.h`, or after the aggregation, `_split_groups` ), or `"merge"`, the merged cell shared
        ( `_share_duplicates` )"""
        self._groups, self._group_pd = None, None
        if getattr( self, "_solver_groups", None ) is not None:
            self._adopt_solver_groups()
            return
        if getattr( settings, "duplicates", "split" ) == "merge":
            self._share_duplicates()
        else:
            self._split_groups( settings )

    def _adopt_solver_groups( self ):
        """the tree the solver ran on ( `sdotplan/Groups.h` ): the plan's `groups` with the virtual weights it found, its diagram, and
        the counters of the aggregation said in terms of groups"""
        g0, pd = self._solver_groups, self._pd
        self._groups = g0.with_virtual_weights( *pd._given_virtual_weights() )
        self._group_pd = pd
        if loom.is_traced( self.stats.get( "status", 0 ) ):
            return
        g = self._groups
        members = np.flatnonzero( g.parent >= 0 )
        nb = g.nb_groups
        self.stats.update( nb_clusters = nb, nb_aggregated = len( members ), nb_duplicates = int( ( g.site != np.arange( len( g.site ) ) ).sum() ) )
        self.stats[ "groups" ] = f"{ nb } groups of { len( members ) } seeds ( { g.nb_aggregates } aggregates, depth { g.depth } ), solved with the diagram"
        self.stats[ "aggregation" ] = f"{ len( members ) } seeds in { nb } groups ( { self.stats[ 'nb_duplicates' ] } exact duplicates ), solved with their pieces"

    #: the polish of the groups ( `_split_groups` ): its Newton steps at most, and the halvings of a step that does not decrease
    GROUP_MAX_ITER = 12
    GROUP_MAX_HALVINGS = 8
    #: ... and where it stops: the members' residual relative to the largest target ( the rounding of the cells is ~1e-16 )
    GROUP_RTOL = 1e-13

    def _split_groups( self, settings ):
        """THE GROUPS OF A CONVERGED SOLVE, each member given its own mass ( `Iterative( duplicates = "split"` ) after the aggregation ).

        The solver merged what the doubles cannot separate ( `clusters`: close pairs and exact duplicates, `sdotplan/Aggregation.h` ):
        each cluster's mass is right, its members' only to the floor ( an exact duplicate's cell even empty ). Each cluster becomes
        a ROOT AGGREGATE of a tree ( `groups.Groups.from_rep`: its exact duplicates an aggregate of their own inside it ), whose cell
        its members split -- each plane in the frame of their lowest common aggregate -- and a Newton on the virtual weights gives
        every member its mass: the outer weights stay the solver's, and each step is the inner levels of `sdotplan/Groups.h`
        ( `groups.TreeSystem`: a dense block per aggregate, a level at a time ), from the derivatives of the members' masses
        ( `PowerDiagram.hessian_rows` with groups ). The fans of the duplicates are centred on the barycentre of their site's cell
        ( the solver's: the duplicates' cells are empty, the first of a site holds its cell ), inside it whatever the weights.

        After it: `weights` ( `groups.py`'s: the outer weights, and the virtual ones of the duplicates ), `groups` ( with the
        precise virtual weights: `PowerDiagram( positions, plan.weights, groups = plan.groups )` redraws the plan ), `cell_masses` and
        `barycenters` per member ( the barycentres of the pieces ), `stats[ "residual_full" ]` the full residual. The cost is the
        solver's ( that of the merged cells ). Only on readable positions, and not with a periodicity ( its images are not grouped
        yet ): the floor otherwise, as before."""
        from . import groups as grp
        if loom.is_traced( self.stats.get( "nb_clusters", 0 ) ) or not self.converged:
            return
        pos = self.problem.source.positions
        pos_raw = getattr( pos, "raw", pos )
        w_raw = self._weights_fit.raw
        if loom.is_traced( pos_raw ) or loom.is_traced( w_raw ) or self._pd._periodic is not None:
            return
        cl = self.clusters
        if cl is None:
            return
        P = grp._host( pos_raw )
        n = P.shape[ 0 ]
        rep = np.asarray( grp._host( cl ), dtype = np.int64 ).reshape( -1 )
        # the clusters as roots, the separations of scales inside them as nested aggregates ( `find_groups` )
        sets = [ np.flatnonzero( rep == r ) for r in np.unique( rep[ rep != np.arange( n ) ] ) ]
        hashed = getattr( self, "_exact_dups", None ) or ( np.zeros( 0, dtype = np.int64 ), ) * 2     # ( the aggregation's hash )
        auto = grp.find_groups( P, kappa = getattr( settings.tuning, "aggregation_kappa", grp.KAPPA ), duplicates = hashed )
        if auto is not None:
            below = [ [] for _ in range( auto.nb_aggregates ) ]
            for i in np.flatnonzero( auto.parent >= 0 ):
                a = auto.parent[ i ]
                while a >= 0:
                    below[ a ].append( int( i ) )
                    a = auto.agg_parent[ a ]
            sets += [ b for a, b in enumerate( below ) if auto.outer[ a ] > 0 and ( rep[ b ] == rep[ b[ 0 ] ] ).all() ]
        g = grp.Groups.from_sets( sets, P, apex = grp._host( self.barycenters.raw ) )
        W = grp._host( w_raw ).reshape( -1 )
        W = W[ g.site ]                                  # ( the outer weights: a duplicate's is its site's, not the solver's emptied one )
        nu = grp._host( self.target_masses.raw ).reshape( -1 )
        members = np.flatnonzero( g.parent >= 0 )

        def diagram( x, ts ):
            gx = g.with_virtual_weights( *ts.virtual_weights( x ) )
            return PowerDiagram( pos_raw, W, distribution = self.problem.target, kernel_dtype = settings.kernel_dtype, memory = 0,
                                 sink_weight = self._sink, radial_func = getattr( self, "_radial", None ), groups = gx ), gx

        pd = PowerDiagram( pos_raw, W, distribution = self.problem.target, kernel_dtype = settings.kernel_dtype, memory = 0,
                           sink_weight = self._sink, radial_func = getattr( self, "_radial", None ), groups = g )
        counts, ids, vals = pd.hessian_rows( items = members )
        ts = grp.TreeSystem( g, members, counts, ids, vals )
        x = ts.slots_of( W )                             # ( the virtual weights from the solver's: close doubles, exact differences )
        pd, gx = diagram( x, ts )
        m = grp._host( pd.measures.raw ).reshape( -1 )
        err = np.abs( m - nu )[ members ].max()
        floor = self.GROUP_RTOL * float( nu.max() )
        nb_diag = 2
        for it in range( self.GROUP_MAX_ITER ):
            if err <= floor:
                break
            if it:                                       # ( the jacobian of the current diagram )
                counts, ids, vals = pd.hessian_rows( items = members )
                ts = grp.TreeSystem( g, members, counts, ids, vals )
            D = ts.step( nu - m, np.zeros( n ) )
            # damped: the step is halved until the members' residual decreases ( a piece must not empty )
            t = 1.0
            for _ in range( self.GROUP_MAX_HALVINGS ):
                x_t = x + t * D
                pd_t, g_t = diagram( x_t, ts )
                m_t = grp._host( pd_t.measures.raw ).reshape( -1 )
                nb_diag += 1
                err_t = np.abs( m_t - nu )[ members ].max()
                if err_t < err:
                    break
                t /= 2
            else:
                break
            x, pd, gx, m, err = x_t, pd_t, g_t, m_t, err_t

        #: the GROUPS of the plan ( `groups.py` ): the clusters, as the diagram splits them -- `None` without clusters
        self._groups = gx
        self._group_pd = pd
        ax = self._pd.num_point
        self.weights = RealTensor[ ax ]( ts.weights( x ) )
        self.cell_masses = RealTensor[ ax ]( m )
        d = int( pd.dim_count )
        mass = grp._host( pd.moment( d + 1 ).raw ).reshape( -1 )
        first = np.stack( [ grp._host( pd.moment( c ).raw ).reshape( -1 ) for c in range( d ) ], axis = 1 )
        bar = np.where( mass[ :, None ] > 0, first / np.where( mass > 0, mass, 1.0 )[ :, None ], P )
        self.barycenters = RealTensor[ ax, self._pd.dim ]( bar )
        self.stats[ "residual_full" ] = float( np.abs( m - nu ).max() )
        self.stats[ "nb_group_diag" ] = nb_diag
        self.stats[ "groups" ] = f"{ g.nb_groups } groups of { len( members ) } seeds, split to { err :.1e} in { nb_diag } diagrams"

    def _share_duplicates( self ):
        """THE CONVENTION FOR EXACT DUPLICATES, on the solver's outputs: the duplicates of a dirac are ONE dirac ( the mass of the
        cluster, the merged cell ), which each member shares in proportion to its own mass -- `cell_masses[ i ] = M nu_i / sum nu`,
        `barycenters[ i ]` the barycenter of the merged cell, for every member. The solver itself left the cell of a duplicate empty
        ( `w_dup = w_rep - gap`, `sdotplan/Aggregation.h` ) and the merged cell to its representative: the weights stay those, the
        only ones a `PowerDiagram` can draw. Only the duplicates found on the host ( not under `jax.jit` ); see
        `_attach_derivatives` for the derivatives that go with these values."""
        found = getattr( self, "_exact_dups", None )
        if found is None:
            return
        keep, inv, members = _duplicate_layout( int( self.cell_masses.shape[ 0 ] ), *found )
        nu = self._masses.raw
        mk = _cluster_sum( self.cell_masses.raw, members )
        self.cell_masses = RealTensor[ self._pd.num_point ]( mk[ inv ] * nu / _cluster_sum( nu, members )[ inv ] )
        self.barycenters = RealTensor[ self._pd.num_point, self._pd.dim ]( self.barycenters.raw[ keep[ inv ] ] )

    # -- the derivative ---------------------------------------------------------------------------

    def _attach_derivatives( self, problem, settings ):
        """THE DERIVATIVE OF THE SOLVE, by the implicit function theorem -- `self` was solved on DETACHED inputs ( the
        C++ solve has no adjoint ), `problem` carries the live ones.

        At the fitted weights `w`, `F( w, p, nu, rho ) = m( w, p, rho ) - nu_eff` vanishes, `nu_eff = nu * sum( m ) / sum( nu )`
        being the target masses rescaled to what the domain holds of the density ( `sum( m )` does not depend on `w`: the
        cells tile the domain ). Its jacobian with respect to `w` is the laplacian `L` of the Laguerre graph, so
        `dw = - L^-1 ( d_p F dp + d_rho F drho - d nu_eff )`. That is written as ONE NEWTON STEP with the jacobian frozen:

            w_live = w - L^-1 F( w_live ... )          ( value: `w`, since `F( w ) = 0` )

        where `F` is built from the differentiable `PowerDiagram.measures` ( the derivative with respect to the seeds and the
        density ) and `L^-1` is `PowerDiagram.laplacian_solve` ( differentiable with respect to its right-hand side ONLY: the
        derivative of `L` multiplies `F( w ) = 0` ). Nothing in here is specific to a framework: it is all `loom.ffi_call`.

        What is then differentiable: `weights`, and `cell_masses` ( the masses of the live diagram at those weights ), and with
        them, through the moments of the density over the cells ( `PowerDiagram.moment`, differentiable like `measures` ):

          * the BARYCENTERS, `first_i / mass_i` at the live weights, so with their implicit derivative;
          * the COST, written as the DUAL functional `D = sum_i int_{cell_i} |x - p_i|^2 rho + sum_i w_i ( nu_i - m_i )`.
            At the fitted weights `D` equals the cost and `d D / d w = nu - m` vanishes: differentiating `D` with the live weights
            therefore gives the ENVELOPE derivative ( `2 m_i ( p_i - b_i )` for the seeds, `w_i` for the masses, `int psi d rho` for the
            density ) without any of it written down, and the live weights' own derivative multiplies zero.

        The moments are closed forms where the density has them -- constant pieces, a `Mesh`, gaussians in 2D and 3D, the balls
        of a partial transport -- and so are their adjoints ( `exact_moments`, `diagram/Moments.h` ): the derivatives are then
        those of the values the solver returns. Elsewhere ( a pointwise density ) they are those of the adaptive quadrature of
        `PowerDiagram.moment`.

        EXACT DUPLICATES ( diracs at the very same position ): the plan is NOT differentiable there -- moving one duplicate cuts
        the merged cell along the direction of the move, so the directional derivative is not linear in it. The CONVENTION
        ( `_share_duplicates` for the values ): a cluster of duplicates is ONE dirac, at the mean of the positions weighted by
        the masses ( which IS their position ), of mass `nu_c = sum nu_i`. The diagrams here are built on that merged cloud ( two
        equal seeds would make the cells and the laplacian degenerate ), and the results go back to the members:

          * `weights`: the cluster's, for each member ( plus the solver's constant gap of a duplicate, `w_dup = w_rep - gap` );
          * `barycenters`: the merged barycenter `b_c` for each member;
          * `cell_masses`: `m_c nu_i / nu_c`, the merged cell shared in proportion to the masses;
          * `cost`: that of the merged problem -- so `d cost / d p_i = ( nu_i / nu_c ) 2 m_c ( p_i - b_c ) = 2 m_i ( p_i - b_c )`
            ( the envelope formula of `cost_and_position_grad`, member by member ), and `d cost / d nu_i` is the cluster's.

        In other words the derivatives are those of the merged problem, composed with `p_c = sum nu_i p_i / nu_c` and
        `nu_c = sum nu_i`. The duplicates are those found on the host ( `exact_duplicates` ): positions only known under a trace
        ( `jax.jit` ) are not checked, and their duplicates keep the degenerate derivative."""
        if getattr( self, "_groups", None ) is not None:
            return self._attach_split_derivatives( problem, settings )
        src, dst = problem.source, problem.target
        fit = self._weights_fit
        ax = self._pd.num_point
        kd = settings.kernel_dtype
        per = _periodicity_of( problem )
        sk = self._sink
        rf = getattr( self, "_radial", None )
        frozen_pos, live_pos, fit_w, nu_raw = self.problem.source.positions.raw, src.positions.raw, fit.raw, src.weights.raw

        # EXACT DUPLICATES: the diagrams are built on the merged cloud ( see the docstring ), `axc` its points
        layout = None
        if not loom.is_traced( frozen_pos ):
            from loom.tensor.host import to_host
            P = np.asarray( to_host( frozen_pos ), dtype = np.float64 )
            dup, rep = exact_duplicates( P.reshape( P.shape[ 0 ], -1 ) )
            if len( dup ):
                layout = _duplicate_layout( P.shape[ 0 ], dup, rep )
        axc = ax
        if layout is not None:
            keep, inv, members = layout
            axc = Axis( ShapeVar( len( keep ) ), name = "num_point" )
            nu_raw = _cluster_sum( nu_raw, members )
            # `p_c = p_rep + sum_i ( nu_i / nu_c ) ( p_i - p_rep )`: the value of `p_rep` exactly, the derivative of the weighted mean
            p_rep = live_pos[ keep ]
            ext_p = loom.ops( live_pos ).concatenate( [ live_pos, live_pos[ :1 ] * 0 ], axis = 0 )
            ext_nu = loom.ops( src.weights.raw ).concatenate( [ src.weights.raw, src.weights.raw[ :1 ] * 0 ], axis = 0 )
            for r in members[ 1: ]:
                p_rep = p_rep + ( ext_nu[ r ] / nu_raw )[ :, None ] * ( ext_p[ r ] - live_pos[ keep ] )
            gap = fit_w - fit_w[ keep[ inv ] ]       # the solver's constant gap of the duplicates ( zero elsewhere )
            frozen_pos, live_pos, fit_w = frozen_pos[ keep ], p_rep, fit_w[ keep ]

        on_card = getattr( loom.resolved_device(), "is_cuda_gpu", False )
        if layout is None and on_card:
            # on a card, the solver's own diagram at the fitted weights: its tree was built with it -- under a `jit` too, where a
            # diagram built here on traced seeds would have none, and the card's laplacian solve needs one
            frozen = self.power_diagram()
        else:
            # ( on a card, the merged cloud's tree, built here: the weights may be traced, the card's laplacian solve needs one )
            from .AaBsp import AaBsp, builds_on_card
            acc = AaBsp( frozen_pos, fit_w ) if on_card and per is None and builds_on_card( frozen_pos ) else None
            frozen = PowerDiagram( frozen_pos, fit_w, accelerator = acc, distribution = self.problem.target, kernel_dtype = kd, memory = 0,
                                   periodicity = per, sink_weight = sk, radial_func = rf )
        live = PowerDiagram( live_pos, fit_w, distribution = dst, kernel_dtype = kd, memory = 0, periodicity = per, sink_weight = sk, radial_func = rf )
        m = RealTensor[ axc ]( live.measures.raw )
        nu = RealTensor[ axc ]( nu_raw )
        # ( partial: no rescaling -- the sink takes the rest of the density; radial: the masses are not a partition )
        residual = m - nu if sk is not None or rf is not None else m - nu * ( m.sum() / nu.sum() )
        step = RealTensor[ axc ]( frozen.laplacian_solve( residual.raw, linear_solver = settings.tuning.linear_solver ).raw )
        weights = RealTensor[ axc ]( fit_w ) - step
        at = PowerDiagram( live_pos, weights.raw, distribution = dst, kernel_dtype = kd, memory = 0, periodicity = per, sink_weight = sk, radial_func = rf )
        m = RealTensor[ axc ]( at.measures.raw )

        # the moments at the live weights: `first_c`, `second`, and the mass by the SAME quadrature
        d = int( at.dim_count )
        first = [ RealTensor[ axc ]( at.moment( c ).raw ) for c in range( d ) ]
        second = RealTensor[ axc ]( at.moment( d ).raw )
        mass = RealTensor[ axc ]( at.moment( d + 1 ).raw )
        # the moments are those of the cells around the FOLDED seeds ( periodicity ): the cost reads the same positions
        folded = live_pos if per is None else at._periodic.fold( live_pos )[ 0 ]
        pos = [ RealTensor[ axc ]( folded[ :, c ] ) for c in range( d ) ]
        pp = sum( ( p * p for p in pos[ 1: ] ), pos[ 0 ] * pos[ 0 ] )
        px = sum( ( p * f for p, f in zip( pos[ 1: ], first[ 1: ] ) ), pos[ 0 ] * first[ 0 ] )
        nu_eff = nu if sk is not None or rf is not None else nu * ( m.sum() / nu.sum() )
        self.cost = ( second - 2 * px + pp * mass ).sum() + ( weights * ( nu_eff - m ) ).sum()

        # an empty cell keeps its seed as barycenter ( there is nothing else to say )
        full = loom.ops().stack( [ loom.ops().where( mass.raw > 0, f.raw / loom.ops().where( mass.raw > 0, mass.raw, 1.0 ), p.raw ) for f, p in zip( first, pos ) ], axis = 1 )
        if per is not None:
            full = at.unfold( full )

        if layout is None:
            self.weights = weights
            self.cell_masses = m
            self.barycenters = RealTensor[ ax, self._pd.dim ]( full )
            self._live_diagram = at                      # ( `sphere_areas`, read from it when asked: the derivative goes through `weights` )
            return
        # back to the members of the clusters
        self.weights = RealTensor[ ax ]( weights.raw[ inv ] + gap )
        self.cell_masses = RealTensor[ ax ]( m.raw[ inv ] * src.weights.raw / nu_raw[ inv ] )
        self.barycenters = RealTensor[ ax, self._pd.dim ]( full[ inv ] )

    def _attach_split_derivatives( self, problem, settings ):
        """THE DERIVATIVE OF A PLAN WITH GROUPS ( `duplicates = "split"`: a tree of aggregates ), by the implicit function theorem
        level by level, as Newton solved it ( `sdotplan/Groups.h` ) -- one Newton step with the jacobian frozen, `x_live = x + J^-1 b`,
        `b = nu - m` the residual of the LIVE grouped diagram ( each member through its own position ), `x` the solver's slots:

          * THE ROOTS: `PowerDiagram.laplacian_solve` of the grouped diagram, the laplacian of the facets between roots, `b` summed
            over each root;
          * THE AGGREGATES, a level at a time: the frozen dense block of each one ( `groups.TreeSystem`, from `hessian_rows` ), on the
            sums of `b` over its children and the facets out of them at the steps already known. The virtual diracs of the
            duplicates follow their site, their directions are FROZEN: the derivative is the one along the split chosen.

        Then the outputs on the grouped diagram at the live slots: `cell_masses` and the barycentres of the PIECES, and the cost
        ( the dual functional of the SITES -- the distinct positions, `Groups.sites` -- plus the members' offsets from their site:
        see below ), whose derivative with respect to a member's position is `2 m_i ( p_i - b_i )`, the barycentre of its own piece --
        the duplicates' gradients differ, and a descent separates them. The weights' terms are written on the slots ( `TreeSystem.
        slot_sum` ): a root's residuals summed before they meet its weight, as the precise planes need. A site moved as a whole is
        differentiated exactly; one duplicate moved alone sees its neighbours' facets as the mean of the one-sided derivatives
        ( `diagram::scatter_cell_grad` )."""
        from . import groups as grp
        src, dst = problem.source, problem.target
        g = self._groups
        gpd = self._group_pd
        ax = self._pd.num_point
        kd = settings.kernel_dtype
        sk = self._sink
        rf = getattr( self, "_radial", None )
        n = len( g.parent )
        live_pos = src.positions.raw
        nu = src.weights.raw
        # ( one of them a constant: in the framework of the other -- Torch's operations take no numpy )
        if isinstance( live_pos, np.ndarray ) and not isinstance( nu, np.ndarray ):
            live_pos = grp._arr( nu, live_pos )
        elif isinstance( nu, np.ndarray ) and not isinstance( live_pos, np.ndarray ):
            nu = grp._arr( live_pos, nu )

        # the frozen levels, the solver's slots
        members = np.flatnonzero( g.parent >= 0 )
        # ( the backend's rows and gathers: the same under a `jit`, where the diagram is known only in the program )
        ts = grp.TreeSystem.from_rows( g, members, *gpd.hessian_rows( items = members, host = False ) )
        x0 = ts.slots_of( gpd.weights.raw.reshape( -1 ), g.seed_v, g.agg_v )

        def grouped( x, at_pos = live_pos ):
            gg = g.with_virtual_weights( *ts.virtual_weights( x ) )
            return PowerDiagram( at_pos, ts.outer( x ), distribution = dst, kernel_dtype = kd, memory = 0, sink_weight = sk, radial_func = rf, groups = gg )

        # ONE FROZEN NEWTON STEP: the roots, then the aggregates
        x0f = loom.array( x0 ) if not isinstance( live_pos, np.ndarray ) else x0
        m0 = grouped( x0f ).measures.raw
        b = ( nu - m0 ) if sk is not None or rf is not None else ( nu * ( m0.sum() / nu.sum() ) - m0 )
        D = gpd.laplacian_solve( b, linear_solver = settings.tuning.linear_solver ).raw * grp._arr( m0, ts.root_slot.astype( np.float64 ) )
        x1 = x0f + ts.step( b, D )

        # THE OUTPUTS, on the grouped diagram at the live slots
        at = grouped( x1 )
        m = RealTensor[ ax ]( at.measures.raw )
        d = int( at.dim_count )
        first = [ RealTensor[ ax ]( at.moment( c ).raw ) for c in range( d ) ]
        mass = RealTensor[ ax ]( at.moment( d + 1 ).raw )
        pos = [ RealTensor[ ax ]( live_pos[ :, c ] ) for c in range( d ) ]

        # THE COST, as the sites' dual functional at the mean `p_s` of their members' positions ( weighted by the masses: the
        # value of their position ), plus what the members' own positions add, `eps_i = p_i - p_s` ( zero ):
        # `int_{piece_i} |x - p_i|^2 = int_{piece_i} |x - p_s|^2 - 2 eps_i . ( first_i - m_i p_s ) + |eps_i|^2 m_i`. Its
        # derivative is the envelope one member by member, `2 m_i ( p_i - b_i )` with the barycentre of the piece -- which the
        # dual functional of the members alone would only give for a site moved as a whole ( its neighbours see a duplicate's
        # facets through the mean of the one-sided derivatives )
        keep, gs = g.sites()
        dup = np.flatnonzero( g.site != np.arange( n ) )
        keep_, inv, groups_of = _duplicate_layout( n, dup, g.site[ dup ] )
        nu_s = _cluster_sum( nu, groups_of )
        ops = loom.ops( live_pos )
        nu_ext = ops.concatenate( [ nu, nu[ :1 ] * 0 ], axis = 0 )
        p_ext = ops.concatenate( [ live_pos, live_pos[ :1 ] * 0 ], axis = 0 )
        p_site = live_pos[ keep ]
        for r in groups_of[ 1: ]:
            p_site = p_site + ( nu_ext[ r ] / nu_s )[ :, None ] * ( p_ext[ r ] - live_pos[ keep ] )
        axs = Axis( ShapeVar( len( keep ) ), name = "num_point" )
        sv1, av1 = ts.virtual_weights( x1 )
        W1 = ts.outer( x1 )
        if gs is not None:
            _, gs = g.with_virtual_weights( sv1, av1 ).sites()
        sd = PowerDiagram( p_site, W1[ keep ], distribution = dst, kernel_dtype = kd, memory = 0, sink_weight = sk, radial_func = rf, groups = gs )
        ms = RealTensor[ axs ]( sd.measures.raw )
        fs = [ RealTensor[ axs ]( sd.moment( c ).raw ) for c in range( d ) ]
        ss = RealTensor[ axs ]( sd.moment( d ).raw )
        qs = RealTensor[ axs ]( sd.moment( d + 1 ).raw )
        ps = [ RealTensor[ axs ]( p_site[ :, c ] ) for c in range( d ) ]
        nus = RealTensor[ axs ]( nu_s )
        nus_eff = nus if sk is not None or rf is not None else nus * ( ms.sum() / nus.sum() )
        pp = sum( ( p * p for p in ps[ 1: ] ), ps[ 0 ] * ps[ 0 ] )
        px = sum( ( p * f for p, f in zip( ps[ 1: ], fs[ 1: ] ) ), ps[ 0 ] * fs[ 0 ] )
        dual = ( RealTensor[ ax ]( x1 ) * RealTensor[ ax ]( ts.slot_sum( ( nus_eff - ms ).raw, rows = keep ) ) ).sum()
        cost = ( ss - 2 * px + pp * qs ).sum() + dual
        at_site = [ RealTensor[ ax ]( p_site[ inv, c ] ) for c in range( d ) ]
        for c in range( d ):
            e = pos[ c ] - at_site[ c ]
            cost = cost + ( -2 * e * ( first[ c ] - mass * at_site[ c ] ) + e * e * mass ).sum()
        self.cost = cost
        self.weights = RealTensor[ ax ]( ts.weights( x1 ) )
        self.cell_masses = m
        ops = loom.ops()
        self.barycenters = RealTensor[ ax, self._pd.dim ]( ops.stack( [ ops.where( mass.raw > 0, f.raw / ops.where( mass.raw > 0, mass.raw, 1.0 ), p.raw ) for f, p in zip( first, pos ) ], axis = 1 ) )

    @property
    def sphere_areas( self ):
        """With a sink ( partial transport ): the measure of the sphere inside each cell, where the target is not zero -- a length in
        2D, an area in 3D, `[ n ]` ( `PowerDiagram.sphere_areas` ). What a surface energy counts: differentiable with respect to the
        positions and the masses of the diracs when they are traced, through the weights ( the implicit function theorem )."""
        live = getattr( self, "_live_diagram", None )
        return ( live if live is not None else self.power_diagram() ).sphere_areas

    def power_diagram( self, weights = None ) -> PowerDiagram:
        """The `PowerDiagram` of the plan, at its weights ( or at `weights` ): to inspect or draw its cells.

        = Internals

        THE ON-DEMAND TOOL: the `PowerDiagram` for `weights` ( by default: the FITTED weights )
        -- always the SAME object, on which those weights are set. For drawing, for inspecting, for
        a functional that was not anticipated; nothing in the solve goes through it."""
        if weights is None and getattr( self, "_group_pd", None ) is not None:
            return self._group_pd                        # the groups' diagram ( `_split_groups` ): it draws their pieces
        self._pd.weights = self._weights_fit if weights is None else weights
        return self._pd

    # -- display ----------------------------------------------------------------------------------

    def add_to_viz( self, viz, weights = None, seeds = True, transport = False, target = True, **kwargs ):
        """Draws the plan into a `Visualizer`: the power diagram at the FITTED weights ( `weights`
        to see another one ), its seeds, and -- when `transport` -- the segment from each seed to
        the barycenter of its cell ( only with the fitted weights ). `target`: the target density
        adds what it knows of itself ( `Distribution.add_to_viz` ). `kwargs` are those of
        `PowerDiagram.add_to_viz`. For the quick ways to look at it -- `write_html`, `write_pvd`,
        `show` -- see `Displayable`."""
        self.power_diagram( weights ).add_to_viz( viz, seeds = seeds, **kwargs )
        if transport:
            if weights is not None:
                raise ValueError( "transport = True: the barycenters are those of the fitted weights" )
            d = self.problem.nb_dims
            # ( the cells are drawn around the given positions, as are the barycenters )
            pos = np.asarray( self._pd.positions ).reshape( -1, d )
            bar = np.asarray( self.barycenters ).reshape( -1, d )
            n = len( pos )
            viz.add_edges( np.concatenate( [ pos, bar ] ), np.stack( [ np.arange( n ), n + np.arange( n ) ], axis = 1 ),
                           color = "#d62728" )
        if target:
            self.problem.target.add_to_viz( viz )
        return viz

    # -- what the plan is WORTH -------------------------------------------------------------------

    def transport( self ):
        """`( cost, barycenters, masses )` at the FITTED weights: the cost `W_2^2 = sum_i int_{cell_i}
        |x - p_i|^2 rho` ( a scalar `Tensor` ), the barycenter of each cell ( `[ n, d ]` ) and
        its mass ( `[ n ]` ).

        All three are OUTPUTS of the call -- the solver measures them at the fitted weights, in the
        same C++ and with the same scratch. An empty cell keeps its seed as barycenter."""
        return self.cost, self.barycenters, self.cell_masses

    def cost_and_position_grad( self ):
        """`( cost, grad )`: `W_2^2` and its derivative with respect to the positions of the diracs, `2 m_i ( p_i - b_i )` ( `[ n, d ]` ).

        = Internals

        `( cost, grad )`: the cost, and its derivative with respect to the dirac POSITIONS
        ( `[ n, d ]` ) -- by the envelope theorem: at the optimal weights, the derivative of the cost
        with respect to `p_i` does not go through the cells, and equals `2 m_i ( p_i - b_i )`, `b_i` the
        barycenter of the cell and `m_i` its mass ( which is the target mass of the dirac ). It is the
        same formula as `SdotPlan1d`, and what a reconstruction consumes ( `otrec` )."""
        return self.cost, 2 * self.cell_masses * ( self._pd.positions - self.barycenters )
