"""DIRACS ON A GRID: a whole semi-discrete solve whose diracs are the centres of an N x N grid ( `SumOfDiracsOnAGrid` ),
by the paths the library has for them -- the page `docs/guide/performance-grid.md`.

    errand run --env lmo-numpy 'sdot-ffi/bench/bench_grid.py::*' --side=256,512,1024 --solver=dedicated,generic

`--solver`:
  dedicated   `Tuning( grid_solver = "dedicated" )`: the grid's own Newton ( `grid/GridNewton2D.h` ), coarse to fine; with
              `--device=default` on a CUDA card, its port ( `gpu/GridNewton2D.cuh` ). Doubles only ( `--precision` is ignored )
  cells       `Tuning( grid_solver = "cells" )`: the generic Newton on the grid's diagram ( `PowerDiagram_Grid` ); on a card, the
              card's generic Newton on the positions
  generic     the same diracs given as a `SumOfDiracs`: what a solve on a cloud does ( tree, cells, Newton ), in `--precision`
              ( fp32 alone stalls: at 512 x 512, about 1e-3 from the masses, the cells too small for its digits; `auto` is
              fp64 on a CPU, `mixed` on a card, fp32 then fp64 )

THE PROBLEM, one for every run: diracs of equal masses, against an image of 512 x 512 affine pixels on the unit square
( `Image( degree = 1 )`, the bilinear interpolation of its 513 x 513 nodes ), a field of sines with an offset,
`1 + 0.5 sin( 6 pi x ) sin( 4 pi y )`. Cold start ( zero weights ), stopped on `max |mass - target| / target <= rtol`. After a
warm-up on a 64 x 64 grid ( the first call compiles ), the smallest wall time of `reps`.
"""

import numpy
from errand import Param, bench

import benchlib

PIXELS = 512


def sines( P = PIXELS ):
    """the target's Bernstein coefficients, `[ P, P, 4 ]`: those of a pixel are its corners, `( x0, y0 ), ( x0, y1 ), ( x1, y0 ), ( x1, y1 )`"""
    u = numpy.arange( P + 1 ) / P
    v = 1 + 0.5 * numpy.sin( 6 * numpy.pi * u[ :, None ] ) * numpy.sin( 4 * numpy.pi * u[ None, : ] )
    return numpy.stack( [ v[ :-1, :-1 ], v[ :-1, 1: ], v[ 1:, :-1 ], v[ 1:, 1: ] ], axis = -1 )


if p := bench( "grid",
               side      = Param( 256, help = "N: the grid has N x N diracs ( the dedicated solver: a power of two )" ),
               solver    = Param( "dedicated", choices = [ "dedicated", "cells", "generic" ], help = "the path ( see above )" ),
               precision = Param( "auto", choices = [ "auto", "fp32", "fp64", "mixed" ], help = "the float of the generic paths' geometry ( auto: fp64 on a CPU, mixed on a card )" ),
               rtol      = Param( 1e-6, help = "stop on max |mass - target| / target <= rtol" ),
               threads   = Param( -1, help = "workers ( 0: all the cores, -1: the physical ones )" ),
               pin       = Param( "yes", choices = [ "yes", "no", "env" ], help = "pin worker w to core w" ),
               reps      = Param( 3, help = "repetitions ( the smallest wall time is kept )" ),
               device    = Param( "cpu", choices = [ "cpu", "default" ], help = "cpu: the host, even with a card" ),
               kernels   = Param( "catalogue", choices = [ "catalogue", "compile" ], help = "`compile`: compiled for this machine" ) ):
    benchlib.set_device( p.device )                # BEFORE importing loom
    benchlib.set_kernels( p.kernels )
    benchlib.set_threads( p.threads, p.pin )
    import time
    from loom.tensor.host import to_host
    from sdot import Image, Iterative, OtProblem, SumOfDiracs, Tuning
    from sdot.distributions.SumOfDiracsOnAGrid import SumOfDiracsOnAGrid

    target = Image( values = sines(), degree = 1 )
    precision = "fp64" if p.solver == "dedicated" else p.precision
    if precision == "auto":                                         # ( what `Iterative` resolves it to, recorded as such )
        precision = "fp64" if p.device == "cpu" else "mixed"

    def source( N ):
        g = SumOfDiracsOnAGrid( ( N, N ) )
        if p.solver != "generic":
            return g
        return SumOfDiracs( numpy.asarray( to_host( getattr( g.positions, "value", g.positions ) ), dtype = float ) )

    def settings():
        tuning = Tuning( mass_rtol = p.rtol, **( { "grid_solver": p.solver } if p.solver != "generic" else { } ) )
        return Iterative( tol = 0.0, precision = precision, on_failure = "ignore", tuning = tuning )

    N = p.side
    print( "  " + benchlib.env_report() + "; " + benchlib.driver_line() )
    OtProblem( source( 64 ), target ).solve( settings() )          # warm-up: the compilation

    best = None
    benchlib.start_clock()
    for _ in range( max( p.reps, 1 ) ):
        t = time.perf_counter()
        sol = OtProblem( source( N ), target ).solve( settings() )
        benchlib.block_until_ready( sol.weights )
        wall = time.perf_counter() - t
        if best is None or wall < best[ 0 ]:
            best = ( wall, sol )
    wall, sol = best
    st = sol.stats
    masses = numpy.asarray( to_host( getattr( sol.cell_masses, "value", sol.cell_masses ) ), dtype = float )
    res = float( numpy.max( numpy.abs( masses / masses.sum() * ( N * N ) - 1 ) ) )

    p.results.update( benchlib.hardware() )
    p.results.update( n = N * N, side = N, threads = benchlib.nb_threads(), solver = p.solver, precision = precision,
                      converged = int( bool( sol.converged ) ), t_total_wall = wall, ns_per_dirac = wall / ( N * N ) * 1e9,
                      residual_rel = res, iterations = st.get( "nb_iter" ), grid_solver = st.get( "grid_solver", "" ),
                      nb_continuation_steps = st.get( "nb_continuation_steps" ),
                      **{ k: st[ k ] for k in ( "t_diag", "t_lin", "t_tree" ) if isinstance( st.get( k ), ( int, float ) ) } )
    print( "  stats:", { k: ( round( v, 4 ) if isinstance( v, float ) else v ) for k, v in st.items() if not hasattr( v, "__len__" ) or isinstance( v, str ) } )
    print( benchlib.table( [ "N", "solver", "precision", "threads", "converged", "seconds", "ns / dirac", "Newton its", "residual" ],
                           [ [ f"{ N }", p.solver, precision, f"{ benchlib.nb_threads() }", "yes" if sol.converged else "NO",
                               f"{ wall :.3f}", f"{ wall / ( N * N ) * 1e9 :.0f}", f"{ st.get( 'nb_iter' ) }", f"{ res :.1e}" ] ] ) )
