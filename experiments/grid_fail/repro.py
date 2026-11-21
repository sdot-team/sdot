"""Repro: SumOfDiracsOnAGrid's dedicated solver fails at iteration 0 on sharp images ( not committed )."""
import os, sys
sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), "..", "plan_init" ) )
import numpy as np


def main( names = "bands,star,sine", Ns = "32,64", P = "1024", trace = "1" ):
    if int( trace ):
        os.environ[ "SDOT_SOLVE_TRACE" ] = "1"
    if int( trace ) > 1:
        os.environ[ "SDOT_GRID_DEBUG" ] = "1"
    import sdot
    from exp_plan_init import image_values
    for name in names.split( "," ):
        vals = image_values( name, int( P ) )
        img = sdot.Image( vals )
        for N in map( int, Ns.split( "," ) ):
            print( f"=== { name } N={ N } P={ P }", flush = True )
            sol = sdot.ot_solve( sdot.SumOfDiracsOnAGrid( ( N, N ) ), img, on_failure = "ignore" )
            s = sol.stats
            print( { k: s[ k ] for k in [ "status", "nb_iter", "nb_diag", "nb_backtracks", "residual", "residual0", "nb_continuation_steps", "grid_solver" ] if k in s }, flush = True )


if __name__ == "__main__":
    main( *sys.argv[ 1: ] )
