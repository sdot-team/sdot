"""The dedicated grid solver on contrasted images ( bands, star, sine ): status and iterations, against the general path."""
import os
import sys
import numpy as np


def image_values( name, P = 1024, floor = 1e-3 ):
    t = ( np.arange( P ) + 0.5 ) / P
    x, y = np.meshgrid( t, t, indexing = "ij" )
    if name == "bands":
        th = np.deg2rad( 30 )
        v = ( np.sin( 2 * np.pi * ( x * np.cos( th ) + y * np.sin( th ) ) / 0.25 ) > 0 ).astype( float )
    elif name == "sine":
        v = ( 0.5 + 0.5 * np.sin( 6 * np.pi * x ) * np.sin( 4 * np.pi * y ) ) ** 4
    elif name == "star":
        from matplotlib.path import Path
        k = np.arange( 10 )
        r = np.where( k % 2 == 0, 0.42, 0.17 )
        a = np.pi / 2 + k * np.pi / 5
        poly = np.stack( [ 0.5 + r * np.cos( a ), 0.5 + r * np.sin( a ) ], axis = 1 )
        v = Path( poly ).contains_points( np.stack( [ x.ravel(), y.ravel() ], axis = 1 ) ).reshape( P, P ).astype( float )
    v = v / v.max()
    return np.maximum( v, floor )


def main( names = "bands,star,sine", Ns = "32,64,128,256", general = "0", trace = "0" ):
    if trace != "0":
        os.environ[ "SDOT_SOLVE_TRACE" ] = "1"
    import sdot
    for name in names.split( "," ):
        vals = image_values( name )
        for N in map( int, Ns.split( "," ) ):
            r = sdot.ot_solve( sdot.SumOfDiracsOnAGrid( ( N, N ) ), sdot.Image( vals ), on_failure = "ignore" )
            s = r.stats
            print( f"grid    {name:6s} N {N:4d}: {s['status']} it {int( s['nb_iter'] )} res {float( s['residual'] ):.2e} levels {int( s['nb_continuation_steps'] )}", flush = True )
            if general != "0":
                t = ( np.arange( N ) + 0.5 ) / N
                x, y = np.meshgrid( t, t, indexing = "ij" )
                pos = np.stack( [ x.ravel(), y.ravel() ], axis = 1 )
                r = sdot.ot_solve( sdot.SumOfDiracs( pos ), sdot.Image( vals ), on_failure = "ignore" )
                s = r.stats
                print( f"general {name:6s} N {N:4d}: {s['status']} it {int( s['nb_iter'] )}", flush = True )


if __name__ == "__main__":
    main( **dict( a.split( "=", 1 ) for a in sys.argv[ 1: ] ) )
