"""The general path's trace ( stages, steps ) on a contrasted image, regular diracs."""
import sys
import numpy as np
from repro import image_values


def main( name = "star", N = "32" ):
    import sdot
    N = int( N )
    t = ( np.arange( N ) + 0.5 ) / N
    x, y = np.meshgrid( t, t, indexing = "ij" )
    pos = np.stack( [ x.ravel(), y.ravel() ], axis = 1 )
    r = sdot.ot_solve( sdot.SumOfDiracs( pos ), sdot.Image( image_values( name ) ), on_failure = "ignore", verbose = True )
    print( { k: v for k, v in r.stats.items() if k in ( "status", "nb_iter", "nb_continuation_steps" ) } )


if __name__ == "__main__":
    main( **dict( a.split( "=", 1 ) for a in sys.argv[ 1: ] ) )
