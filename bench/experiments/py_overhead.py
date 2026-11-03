# What the 2D Newton spends outside C++ at n = 1e6: the solve's wall time against its C++ total, then cProfile on the same
# solve ( the heaviest Python frames, cumulative and own time ).
#   errand run --env n20-numpy --no-queue python sdot-ffi/bench/experiments/py_overhead.py 36 1000000
import cProfile, os, pstats, sys, time
sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )
import benchlib, cases

T = int( sys.argv[ 1 ] ) if len( sys.argv ) > 1 else 36
n = int( sys.argv[ 2 ] ) if len( sys.argv ) > 2 else 1_000_000
benchlib.set_device( "cpu" )
benchlib.set_kernels( "compile" )
benchlib.set_threads( T, "yes" )
import numpy
from sdot import Image, Iterative, OtNotConverged, OtProblem, SumOfDiracs, Tuning

pos, _, d = cases.case( "uniform2d", n = n, seed = 0 )
target = lambda: Image( values = numpy.ones( ( 1, ) * d ), origin = [ 0.0 ] * d, pixel_frame = numpy.eye( d ) )
settings = lambda m: Iterative( max_iter = m, continuation = "never", on_failure = "ignore", precision = "fp64", aggregate = True,
                                tuning = Tuning( step = "trials", linear_solver = "auto", mass_rtol = 1e-6 ) )
try:
    OtProblem( SumOfDiracs( pos[ :2000 ] ), target() ).solve( settings( 3 ) )
except OtNotConverged:
    pass

def solve():
    return OtProblem( SumOfDiracs( pos ), target() ).solve( settings( 100 ) )

for _ in range( 2 ):
    t = time.perf_counter()
    sol = solve()
    wall = time.perf_counter() - t
    st = sol.stats
    print( f"T={T} n={n}: wall {wall:.3f} s, C++ total {st['t_total']:.3f} s, outside {wall - st['t_total']:.3f} s "
           f"( diag {st['t_diag']:.3f}, asm {st['t_asm']:.3f}, lin {st['t_lin']:.3f}, it {st['nb_iter']} )", flush = True )

pr = cProfile.Profile()
pr.enable()
solve()
pr.disable()
ps = pstats.Stats( pr )
ps.sort_stats( "cumulative" ).print_stats( 35 )
ps.sort_stats( "tottime" ).print_stats( 25 )
