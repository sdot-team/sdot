"""where the time of `AaBsp( pos )` goes on the CPU: the one ffi call against the python around it ( cProfile ).
`python tree_cost_parts.py <n> <threads>`"""
import time, cProfile, pstats, numpy as np

if __name__ == "__main__":
    import sys, os
    sys.path.insert( 0, os.path.join( os.path.dirname( __file__ ), ".." ) )
    import benchlib
    n = int( float( sys.argv[ 1 ] ) ) if len( sys.argv ) > 1 else 1000000
    benchlib.set_threads( int( sys.argv[ 2 ] ) if len( sys.argv ) > 2 else 0, "yes" )
    from sdot import AaBsp
    import loom
    print( "workers", benchlib.nb_threads() )
    pos = np.random.default_rng( 0 ).random( ( n, 2 ) )
    AaBsp( pos[ :20000 ] )
    for _ in range( 2 ):
        t = time.perf_counter(); AaBsp( pos ); print( f"tree { time.perf_counter() - t :.3f} s" )
    pr = cProfile.Profile(); pr.enable(); AaBsp( pos ); pr.disable()
    pstats.Stats( pr ).sort_stats( "tottime" ).print_stats( 12 )
