# The cost of `AaBsp( pos )` on a uniform cloud: wall time, then cProfile ( where the Python side goes ).
#   errand run --env n20-numpy --no-queue python sdot-ffi/bench/experiments/tree_cost.py 1000000 36
import cProfile, os, pstats, sys, time
sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )
import benchlib, cases

n = int( sys.argv[ 1 ] ) if len( sys.argv ) > 1 else 1_000_000
T = int( sys.argv[ 2 ] ) if len( sys.argv ) > 2 else 36
benchlib.set_device( "cpu" )
benchlib.set_kernels( "compile" )
benchlib.set_threads( T, "yes" )
from sdot import AaBsp
pos, _, d = cases.case( "uniform2d", n = n, seed = 0 )
AaBsp( pos[ :5000 ] )
for _ in range( 3 ):
    t = time.perf_counter()
    AaBsp( pos )
    print( f"T={T} n={n}: AaBsp {1e3 * ( time.perf_counter() - t ):.1f} ms", flush = True )
pr = cProfile.Profile()
pr.enable()
AaBsp( pos )
pr.disable()
pstats.Stats( pr ).sort_stats( "tottime" ).print_stats( 25 )
