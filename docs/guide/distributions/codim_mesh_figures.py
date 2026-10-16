"""The viewers of `codim-mesh.md`: the Lloyd quantization of a Klein bottle, with a constant density and with a density growing
with the height ( `docs/public/guide/codim-mesh-*.html` ).

    python docs/guide/distributions/codim_mesh_figures.py [ klein klein-rho ]
"""
import sys
import time
from pathlib import Path

import numpy as np
import sdot

OUT = Path( __file__ ).resolve().parents[ 2 ] / "public" / "guide"
NB_FRAMES = 40


def klein_bottle( nu = 48, nv = 30, neck = 0.6 ):
    """the bottle immersion of the Klein bottle ( it crosses itself where the neck enters the body ): `nodes [ n, 3 ]`,
    `triangles [ m, 3 ]`, on a grid of `nu x nv` quads in `( u, v ) in [ 0, pi ] x [ 0, 2 pi ]`.

    The usual parametrization, written as a center line plus a tube of radius `r( u ) = 2 / 15 ( 3 + 5 neck cos u sin u )`:
    `neck = 1` is the classical bottle, whose neck is 0.067 thick ( as thin as the cells of 2000 diracs ). `neck < 1` widens it,
    and keeps the seams ( `cos u sin u = 0` at `u = 0` and `pi` )"""
    u, v = np.meshgrid( np.linspace( 0, np.pi, nu + 1 ), np.linspace( 0, 2 * np.pi, nv + 1 ), indexing = "ij" )
    cu, su, cv, sv = np.cos( u ), np.sin( u ), np.cos( v ), np.sin( v )
    r = 2 / 15 * ( 3 + 5 * neck * cu * su )
    x = -2 / 15 * cu * ( - 30 * su + 90 * cu ** 4 * su - 60 * cu ** 6 * su ) - r * cu * cv
    y = -1 / 15 * su * ( - 60 * su ) - r / 2 * su * cv * ( 1 - cu ** 2 - 16 * cu ** 4 + 16 * cu ** 6 )
    z = r * sv
    nodes = np.stack( [ x, y, z ], -1 ).reshape( -1, 3 )
    I = lambda i, j: i * ( nv + 1 ) + j
    tris = np.array( [ t for i in range( nu ) for j in range( nv )
                       for t in ( [ I( i, j ), I( i + 1, j ), I( i + 1, j + 1 ) ], [ I( i, j ), I( i + 1, j + 1 ), I( i, j + 1 ) ] ) ] )

    # the seams ( v = 0 and 2 pi, u = 0 and pi ): the nodes at the same place become one, and the surface is closed
    _, first, inv = np.unique( np.round( nodes * 1e9 ).astype( np.int64 ), axis = 0, return_index = True, return_inverse = True )
    tris = inv.reshape( -1 )[ tris ]
    tris = tris[ ( tris[ :, 0 ] != tris[ :, 1 ] ) & ( tris[ :, 1 ] != tris[ :, 2 ] ) & ( tris[ :, 0 ] != tris[ :, 2 ] ) ]
    return nodes[ first ], tris.astype( np.int32 )


def quantize( name, target, n ):
    """runs `sdot.quantization` with `n` diracs, records the positions at each iteration, and writes a page whose frames are
    ( at most `NB_FRAMES` of ) these iterations ( the diracs alone: the surface drawn under them hides half of each sphere )"""
    history = []
    t0 = time.time()
    x = sdot.quantization( target, n, seed = 0, callback = lambda it, pos, plan: history.append( np.asarray( pos ) ) )
    history.append( np.asarray( x ) )
    print( f"{ name }: { len( history ) - 1 } iterations, { round( time.time() - t0, 1 ) } s" )

    kept = np.unique( np.geomspace( 1, len( history ), NB_FRAMES ).astype( int ) - 1 )
    viz = sdot.Visualizer( title = f"Lloyd quantization: { name }", frame_axis = "iteration" )
    for k, it in enumerate( kept ):
        if k:
            viz.new_frame( int( it ) )
        viz.add_points( history[ it ] )
    viz.write_html( OUT / f"codim-mesh-{ name }.html" )
    return x


def klein( n = 6000 ):
    nodes, tris = klein_bottle()
    t0 = time.time()
    target = sdot.Mesh( nodes, tris )
    print( f"klein: { len( tris ) } triangles, built in { round( time.time() - t0, 1 ) } s" )
    return quantize( "klein", target, n )


def klein_rho( n = 6000 ):
    # the density at the nodes, from 0.1 at the bottom of the bottle to 1 at the top of its neck ( `y` is the height )
    nodes, tris = klein_bottle()
    h = ( nodes[ :, 1 ] - nodes[ :, 1 ].min() ) / np.ptp( nodes[ :, 1 ] )
    target = sdot.Mesh( nodes, tris, values = 0.1 + 0.9 * h ** 2 )
    return quantize( "klein-rho", target, n )


CASES = {
    "klein"    : klein,
    "klein-rho": klein_rho,
}

if __name__ == "__main__":
    for name in sys.argv[ 1: ] or CASES:
        CASES[ name ]()
