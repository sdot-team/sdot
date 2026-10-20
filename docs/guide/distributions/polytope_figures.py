"""The pages of `polytope.md`: interactive pages of the HTML viewer ( `sdot.Visualizer`, `docs/public/guide/polytope-*.html` ).

    python docs/guide/distributions/polytope_figures.py [ plan halfspaces icosahedron cells3d ]
"""
import sys
import time
from pathlib import Path

import numpy as np
import sdot

OUT = Path( __file__ ).resolve().parents[ 2 ] / "public" / "guide"
NB_FRAMES = 30


# -- the domains ------------------------------------------------------------------------------------------------------------

def heptagon():
    """7 vertices on an ellipse, at random angles: a convex polygon with no symmetry"""
    a = np.sort( np.random.default_rng( 0 ).random( 7 ) ) * 2 * np.pi
    return np.stack( [ np.cos( a ), 0.7 * np.sin( a ) ], -1 )


def cut_nonagon():
    """a regular nonagon `directions . x <= 0.5`, cut by one more half-space"""
    t = 2 * np.pi * np.arange( 9 ) / 9
    directions = np.concatenate( [ np.stack( [ np.cos( t ), np.sin( t ) ], -1 ), [ [ 0.3, 1.0 ] ] ] )
    offsets = np.r_[ np.full( 9, 0.5 ), 0.2 ]
    return directions, offsets


def icosahedron():
    """the 12 vertices of a regular icosahedron"""
    phi = ( 1 + 5 ** 0.5 ) / 2
    v = [ [ 0, s * 1, t * phi ] for s in ( -1, 1 ) for t in ( -1, 1 ) ]
    v = [ p[ k: ] + p[ :k ] for p in v for k in range( 3 ) ]                    # ( the 3 cyclic permutations )
    return np.array( v, dtype = float )


def cut_cube():
    """the unit cube, cut by two oblique planes"""
    directions = np.concatenate( [ np.eye( 3 ), -np.eye( 3 ), [ [ 1, 1, 1 ], [ -1, 0.5, -1 ] ] ] )
    offsets = np.r_[ np.ones( 3 ), np.zeros( 3 ), 2.4, -0.25 ]
    return directions, offsets


# -- drawing ----------------------------------------------------------------------------------------------------------------

def quantize( name, title, target, n, outline = None, **viz_kwargs ):
    """runs `sdot.quantization` with `n` diracs, records the positions at each iteration, and writes a page whose frames are
    ( at most `NB_FRAMES` of ) these iterations. `outline`: `( directions, offsets )`, drawn under the diracs"""
    history = []
    t0 = time.time()
    x = sdot.quantization( target, n, seed = 0, callback = lambda it, pos, plan: history.append( np.asarray( pos ) ) )
    history.append( np.asarray( x ) )
    print( f"{ name }: { len( history ) - 1 } iterations, { round( time.time() - t0, 1 ) } s" )

    kept = np.unique( np.geomspace( 1, len( history ), NB_FRAMES ).astype( int ) - 1 )
    viz = sdot.Visualizer( title = title, frame_axis = "iteration", **viz_kwargs )
    for k, it in enumerate( kept ):
        if k:
            viz.new_frame( int( it ) )
        if outline is not None:
            viz.add_polytope( *outline, color = "#4e79a7", opacity = 0.08 )
        viz.add_points( history[ it ] )
    viz.write_html( OUT / f"polytope-{ name }.html" )


# -- the cases --------------------------------------------------------------------------------------------------------------

def plan( n = 300 ):
    """2D: diracs gathered in the middle of a heptagon, and their cells, with zero weights ( the Voronoi diagram cut by the
    polygon ) then with the weights of the transport ( each cell holds `1 / n` of the area )"""
    rng = np.random.default_rng( 1 )
    target = sdot.ConvexHull( heptagon() )
    x = 0.25 * rng.normal( size = ( n, 2 ) ) * [ 1, 0.7 ]
    t0 = time.time()
    p = sdot.plan( sdot.SumOfDiracs( x ), target )
    print( f"plan: { round( time.time() - t0, 2 ) } s, converged: { p.converged }" )
    viz = sdot.Visualizer( title = "ConvexHull: Voronoi, then transport", frame_axis = "weights", playable = False )
    p.add_to_viz( viz, weights = np.zeros( n ) )
    viz.new_frame( 1 )
    p.add_to_viz( viz )
    viz.write_html( OUT / "polytope-plan.html" )


def halfspaces( n = 2000 ):
    """2D: the Lloyd quantization of a polytope given by its half-spaces"""
    directions, offsets = cut_nonagon()
    quantize( "halfspaces", "Polytope( directions, offsets )", sdot.Polytope( directions, offsets ), n, outline = ( directions, offsets ) )


def icosahedron_lloyd( n = 4000 ):
    """3D: the Lloyd quantization of an icosahedron, the diracs colored by their depth in the view"""
    quantize( "icosahedron", "ConvexHull: an icosahedron", sdot.ConvexHull( icosahedron() ), n, depth_colors = True )


def cells3d( n = 60 ):
    """3D: the cells of the transport from random diracs to a cut cube"""
    directions, offsets = cut_cube()
    x = np.random.default_rng( 2 ).random( ( n, 3 ) ) * 0.6 + 0.2
    t0 = time.time()
    p = sdot.plan( sdot.SumOfDiracs( x ), sdot.Polytope( directions, offsets ) )
    print( f"cells3d: { round( time.time() - t0, 2 ) } s, converged: { p.converged }" )
    viz = sdot.Visualizer( title = "Polytope in 3D: the cells of a transport" )
    p.add_to_viz( viz )
    viz.write_html( OUT / "polytope-cells3d.html" )


CASES = {
    "plan"       : plan,
    "halfspaces" : halfspaces,
    "icosahedron": icosahedron_lloyd,
    "cells3d"    : cells3d,
}


if __name__ == "__main__":
    OUT.mkdir( parents = True, exist_ok = True )
    for name in sys.argv[ 1: ] or CASES:
        CASES[ name ]()
