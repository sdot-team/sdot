"""The pages and pictures of `gaussians.md`: interactive pages of the HTML viewer ( `sdot.Visualizer`,
`docs/public/guide/gaussians-*.html` ) and one explanatory picture ( matplotlib, `docs/public/guide/img/gaussians-widths.png` ).

    python docs/guide/distributions/gaussians_figures.py [ spiral trefoil widths ]
"""
import sys
import time
from pathlib import Path

import numpy as np
import sdot

HERE = Path( __file__ ).resolve().parent
OUT = HERE.parents[ 1 ] / "public" / "guide"
sys.path.insert( 0, str( HERE ) )
from image_figures import add_density                       # ( the grays of a 2D density, under the diracs )


# -- the mixtures -----------------------------------------------------------------------------------------------------------

def spiral( k = 200 ):
    """`k` gaussians along two turns of a spiral, wider and heavier outwards: `( centers, sigmas, weights )`"""
    s = np.linspace( 0, 1, k )
    r, a = 0.08 + 0.34 * s, 4 * np.pi * s
    centers = np.stack( [ 0.5 + r * np.cos( a ), 0.5 + r * np.sin( a ) ], -1 )
    return centers, 0.008 + 0.03 * s, 0.3 + s


def trefoil( k = 240 ):
    """`k` gaussians of equal widths along a trefoil knot: `( centers, sigmas )`"""
    u = 2 * np.pi * np.arange( k ) / k
    centers = np.stack( [ np.sin( u ) + 2 * np.sin( 2 * u ), np.cos( u ) - 2 * np.cos( 2 * u ), - np.sin( 3 * u ) ], -1 ) / 3
    return centers, np.full( k, 0.06 )


def density_on( centers, sigmas, weights, res = 200, box = ( ( 0, 1 ), ( 0, 1 ) ), conv = 0.0 ):
    """the 2D mixture on a `res x res` grid of `box` ( `[ x, y ]`, the centers ), convolved by a gaussian of width `conv`"""
    axes = [ lo + ( hi - lo ) * ( np.arange( res ) + 0.5 ) / res for lo, hi in box ]
    X = np.stack( np.meshgrid( *axes, indexing = "ij" ), -1 ).reshape( -1, 1, 2 )
    s2 = sigmas ** 2 + conv ** 2
    rho = ( weights / ( 2 * np.pi * s2 ) * np.exp( - ( ( X - centers ) ** 2 ).sum( -1 ) / ( 2 * s2 ) ) ).sum( -1 )
    return rho.reshape( res, res )


# -- the cases --------------------------------------------------------------------------------------------------------------

def spiral_lloyd( n = 3000 ):
    """2D: a spiral of 200 gaussians, under the diracs of its Lloyd quantization"""
    centers, sigmas, weights = spiral()
    t0 = time.time()
    x = np.asarray( sdot.quantization( sdot.SumOfGaussians( centers, sigmas, weights = weights ), n, seed = 0 ) )
    print( f"spiral: { round( time.time() - t0, 1 ) } s" )
    viz = sdot.Visualizer( title = "a spiral of 200 gaussians", face_opacity = 0.5 )
    box = ( ( -0.1, 1.1 ), ( -0.1, 1.1 ) )                 # ( the outer turn goes beyond the unit square )
    add_density( viz, density_on( centers, sigmas, weights, res = 240, box = box ), offset = ( -0.1, -0.1 ), size = ( 1.2, 1.2 ) )
    viz.add_points( x, radius = 0.0028, color = "#e8202a" )
    viz.write_html( OUT / "gaussians-spiral.html" )


def trefoil_lloyd( n = 4000 ):
    """3D: a trefoil knot of 240 gaussians, and the diracs of its Lloyd quantization, colored by their depth in the view"""
    centers, sigmas = trefoil()
    t0 = time.time()
    x = np.asarray( sdot.quantization( sdot.SumOfGaussians( centers, sigmas ), n, seed = 0 ) )
    print( f"trefoil: { round( time.time() - t0, 1 ) } s" )
    viz = sdot.Visualizer( title = "a trefoil knot of gaussians", depth_colors = True )
    viz.add_points( x, radius = 0.012 )
    viz.write_html( OUT / "gaussians-trefoil.html" )


def widths():
    """the stages of the width continuation: the spiral convolved by gaussians of decreasing widths `s`"""
    import matplotlib
    matplotlib.use( "Agg" )
    import matplotlib.pyplot as plt
    centers, sigmas, weights = spiral()
    stages = [ 0.12, 0.05, 0.02, 0.0 ]
    fig, axes = plt.subplots( 1, len( stages ), figsize = ( 2.4 * len( stages ), 2.6 ) )
    for ax, s in zip( axes, stages ):
        rho = density_on( centers, sigmas, weights, box = ( ( -0.1, 1.1 ), ( -0.1, 1.1 ) ), conv = s )
        ax.imshow( rho.T, origin = "lower", extent = ( -0.1, 1.1, -0.1, 1.1 ), cmap = "Greys", vmin = 0 )
        ax.set_title( f"s = { s }" if s else "s = 0: the target", fontsize = 10, family = "monospace" )
        ax.axis( "off" )
    fig.tight_layout()
    fig.savefig( OUT / "img" / "gaussians-widths.png", dpi = 110 )
    plt.close( fig )


CASES = {
    "spiral" : spiral_lloyd,
    "trefoil": trefoil_lloyd,
    "widths" : widths,
}


if __name__ == "__main__":
    ( OUT / "img" ).mkdir( parents = True, exist_ok = True )
    for name in sys.argv[ 1: ] or CASES:
        CASES[ name ]()
