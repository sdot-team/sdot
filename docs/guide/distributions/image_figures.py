"""The pages and pictures of `image.md`: interactive pages of the HTML viewer ( `sdot.Visualizer`, `docs/public/guide/image-*.html` )
and one explanatory picture ( matplotlib, `docs/public/guide/img/image-bernstein.png` ).

    python docs/guide/distributions/image_figures.py [ mona degrees jumps gyroid bernstein ]

The densities are drawn from their own coefficients ( the Bernstein polynomials of each pixel ), the diracs are those of a Lloyd
quantization ( `sdot.quantization` ). `data/mona-lisa.png`: Leonardo da Vinci's Mona Lisa ( public domain, the C2RMF scan on
Wikimedia Commons ), in grays, 80 x 120.
"""
import sys
import time
from pathlib import Path

import numpy as np
import sdot
from sdot.distributions._bernstein import evaluate, from_function
from sdot.distributions.Image import Image                  # ( the classes themselves: the drawing reads their fields )
from sdot.distributions.SplineGrid import SplineGrid

HERE = Path( __file__ ).resolve().parent
OUT = HERE.parents[ 1 ] / "public" / "guide"


# -- densities --------------------------------------------------------------------------------------------------------------

def coarse_image( k = 12 ):
    """`k x k` pixels: oblique waves whose amplitude varies along `y`, sampled at the centers of the pixels"""
    t = ( np.arange( k ) + 0.5 ) / k
    x, y = np.meshgrid( t, t, indexing = "ij" )
    return 0.55 + 0.45 * np.sin( 2 * np.pi * ( 1.2 * x + 0.5 * y ) ) * ( 0.4 + 0.6 * y )


def piecewise_smooth( k = 12, p = 3 ):
    """`k x k` pixels of degree `p`: four bands across `x`, each with its own smooth function ( waves, alternately dark and
    light ), so the density jumps on the borders between the bands. Returns the coefficients and the mean of each pixel"""
    tau = 2 * np.pi
    funcs = [
        lambda x: 0.15 + 0.10 * np.sin( tau * 2 * x[ :, 1 ] ),                                       # light, a slow wave
        lambda x: 1.00 + 0.80 * np.sin( tau * ( 1.5 * x[ :, 1 ] + 2 * x[ :, 0 ] ) ),                  # dark, an oblique wave
        lambda x: 0.30 + 0.25 * np.cos( tau * 3 * x[ :, 1 ] ),                                       # light, a faster wave
        lambda x: 0.90 + 0.70 * np.sin( tau * 2 * ( x[ :, 0 ] - x[ :, 1 ] ) ),                        # dark, the other way
    ]
    ends = np.array( [ 2, 6, 9, 12 ] ) / 12                                                         # ( the bands end on pixel borders )
    t = ( np.arange( k ) + 0.5 ) / k
    x, _ = np.meshgrid( t, t, indexing = "ij" )
    label = np.searchsorted( ends, x )                                                              # ( by the center of each pixel )
    c = sum( ( label == i )[ ..., None ] * from_function( f, ( k, k ), p ) for i, f in enumerate( funcs ) )
    return c, c.mean( -1 )


def mona_lisa():
    """the darkness of the Mona Lisa ( `[ x, y ]`, 80 x 120 ), its contrast stretched to `[ 0, 1 ]`"""
    from PIL import Image as PilImage
    gray = np.asarray( PilImage.open( HERE / "data" / "mona-lisa.png" ).convert( "L" ), dtype = float ) / 255
    dark = ( 1 - gray )[ ::-1 ].T
    lo, hi = np.percentile( dark, [ 2, 98 ] )
    return np.clip( ( dark - lo ) / ( hi - lo ), 0, 1 )


def density_on( target, res = 400, box = ( ( 0, 1 ), ( 0, 1 ) ) ):
    """the density of a 2D `Image` on a `res x res` grid of `box` ( `[ x, y ]`, the centers ), from its coefficients"""
    d = 2
    shape = np.asarray( target.shape.value, dtype = int ).reshape( -1 )
    frame = np.asarray( target.frame, dtype = float ).reshape( d, d )
    if target.knots.is_defined:
        kn = np.asarray( target.knots, dtype = float ).reshape( d, -1 )
        knots = [ kn[ a, : shape[ a ] + 1 ] for a in range( d ) ]
    else:
        knots = [ np.arange( shape[ a ] + 1, dtype = float ) for a in range( d ) ]
    axes = [ lo + ( hi - lo ) * ( np.arange( res ) + 0.5 ) / res for lo, hi in box ]
    X = np.stack( np.meshgrid( *axes, indexing = "ij" ), -1 ).reshape( -1, d )
    g = X @ np.linalg.inv( frame )                                  # grid coordinates ( `x = F^T g`, the origin at 0 )
    pix, loc = [], []
    for a in range( d ):
        k = knots[ a ]
        j = np.clip( np.searchsorted( k, g[ :, a ], side = "right" ) - 1, 0, len( k ) - 2 )
        pix.append( j )
        loc.append( ( g[ :, a ] - k[ j ] ) / ( k[ j + 1 ] - k[ j ] ) )
    if target.is_polynomial:
        v = evaluate( np.asarray( target.coeffs, dtype = float ), d, np.stack( pix, 1 ), np.stack( loc, 1 ) )
    else:
        v = np.asarray( target.values, dtype = float )[ pix[ 0 ], pix[ 1 ] ]
    return v.reshape( res, res )


# -- drawing in the viewer --------------------------------------------------------------------------------------------------

def add_density( viz, rho, offset = ( 0, 0 ), size = ( 1, 1 ), levels = 48, vmax = None ):
    """`rho` ( `[ nx, ny ]`, on `[ offset, offset + size ]` ) as quads in grays, one color per level: the paper's color for 0 and
    the ink's for `vmax` ( dark on the light theme, light on the dark one )"""
    nx, ny = rho.shape
    xs = offset[ 0 ] + size[ 0 ] * np.arange( nx + 1 ) / nx
    ys = offset[ 1 ] + size[ 1 ] * np.arange( ny + 1 ) / ny
    nodes = np.stack( np.meshgrid( xs, ys, indexing = "ij" ), -1 ).reshape( -1, 2 )
    lev = np.minimum( ( rho / ( vmax or rho.max() ) * levels ).astype( int ), levels - 1 ).reshape( -1 )
    i, j = np.meshgrid( np.arange( nx ), np.arange( ny ), indexing = "ij" )
    quads = np.stack( [ i * ( ny + 1 ) + j, ( i + 1 ) * ( ny + 1 ) + j, ( i + 1 ) * ( ny + 1 ) + j + 1, i * ( ny + 1 ) + j + 1 ], -1 )
    quads = quads.reshape( -1, 4 )
    for l in np.unique( lev ):
        u = 0.6 * ( l + 0.5 ) / levels
        viz.add_faces( nodes, quads[ lev == l ], color = ( ( 1 - u, ) * 3, ( 0.12 + u, ) * 3 ) )


def side_by_side( name, title, targets, n, gap = 0.08, radius = 0.008 ):
    """`targets` ( 2D, on the unit square ) side by side in one page: each density under the diracs of its Lloyd quantization"""
    viz = sdot.Visualizer( title = title, face_opacity = 0.5 )
    for k, ( label, target ) in enumerate( targets ):
        t0 = time.time()
        x = np.asarray( sdot.quantization( target, n, seed = 0 ) )
        print( f"{ label }: { round( time.time() - t0, 1 ) } s" )
        shift = np.array( [ k * ( 1 + gap ), 0.0 ] )
        add_density( viz, density_on( target, res = 120 ), offset = shift )
        viz.add_points( x + shift, radius = radius, color = "#e8202a" )
    viz.write_html( OUT / f"image-{ name }.html" )


# -- the cases --------------------------------------------------------------------------------------------------------------

def mona( n = 20000 ):
    """the Mona Lisa ( a cubic spline of 80 x 120 values, its darkness ) and the diracs of its Lloyd quantization"""
    target = SplineGrid( 0.005 + mona_lisa() ** 2.5, degree = 3, frame = [ 2 / 3, 1 ] )
    t0 = time.time()
    x = np.asarray( sdot.quantization( target, n, seed = 0 ) )
    print( f"mona: { round( time.time() - t0, 1 ) } s" )
    viz = sdot.Visualizer( title = "the Mona Lisa, 20000 diracs" )
    # ( a stippling is ink on paper: light ink on the dark theme would give its negative. A sheet under it, ivory in the dark )
    viz.add_faces( np.array( [ [ 0, 0 ], [ 2 / 3, 0 ], [ 2 / 3, 1 ], [ 0, 1 ] ] ), [ [ 0, 1, 2, 3 ] ], color = ( "#ffffff", "#ece8df" ) )
    viz.add_points( x, radius = 0.0034, color = "#222222" )
    viz.write_html( OUT / "image-mona-lisa.html" )


def degrees( n = 1500 ):
    """the same 12 x 12 samples as a constant per pixel, a linear spline and a cubic one, and their Lloyd quantizations"""
    v = coarse_image()
    side_by_side( "degrees", "constant pixels, linear spline, cubic spline",
                  [ ( "Image( v )", Image( v ) ), ( "SplineGrid( v, degree = 1 )", SplineGrid( v, degree = 1 ) ),
                    ( "SplineGrid( v, degree = 3 )", SplineGrid( v, degree = 3 ) ) ], n )


def jumps( n = 1500 ):
    """the same bands as constant pixels ( their means ) and as polynomials of degree 3, which jump on the borders of the bands
    only, and their Lloyd quantizations"""
    c, means = piecewise_smooth()
    side_by_side( "jumps", "bands of waves: constant pixels, pixels of degree 3",
                  [ ( "Image( means )", Image( means ) ), ( "Image( c, degree = 3 )", Image( c, degree = 3 ) ) ], n )


def gyroid( k = 28, n = 8000 ):
    """3D: a cubic spline of `k^3` values carried by a gyroid ( `sin x cos y + sin y cos z + sin z cos x = 0`, one period in the
    cube ), and the diracs of its Lloyd quantization, colored by their depth in the view"""
    t = ( np.arange( k ) + 0.5 ) / k * 2 * np.pi
    x, y, z = np.meshgrid( t, t, t, indexing = "ij" )
    g = np.sin( x ) * np.cos( y ) + np.sin( y ) * np.cos( z ) + np.sin( z ) * np.cos( x )
    t0 = time.time()
    p = np.asarray( sdot.quantization( SplineGrid( 1e-3 + np.exp( -g ** 2 / 0.03 ), degree = 3 ), n, seed = 0 ) )
    print( f"gyroid: { round( time.time() - t0, 1 ) } s" )
    viz = sdot.Visualizer( title = "a gyroid", depth_colors = True )
    viz.add_points( p, radius = 0.011 )
    viz.write_html( OUT / "image-gyroid.html" )


def bernstein():
    """one pixel of degree 2: its 9 Bernstein coefficients and the density they give ( coefficients >= 0: a density >= 0 )"""
    import matplotlib
    matplotlib.use( "Agg" )
    import matplotlib.pyplot as plt
    c = np.array( [ 0.2, 1.0, 0.2,
                    0.6, 0.0, 1.4,
                    0.2, 1.0, 0.2 ] )
    target = Image( c.reshape( 1, 1, 9 ), degree = 2 )
    fig, ax = plt.subplots( figsize = ( 3.6, 3.9 ) )
    ax.imshow( density_on( target ).T, origin = "lower", extent = ( 0, 1, 0, 1 ), cmap = "Greys", vmin = 0 )
    g = np.linspace( 0, 1, 3 )
    for i in range( 3 ):
        for j in range( 3 ):
            ax.plot( g[ i ], g[ j ], "o", ms = 7, color = "#4e79a7", clip_on = False, zorder = 3 )
            ax.annotate( f"{ c[ 3 * i + j ]:.1f}", ( g[ i ], g[ j ] ), xytext = ( 6, 6 ), textcoords = "offset points",
                         fontsize = 9, color = "#4e79a7", annotation_clip = False )
    ax.set_xlim( -0.08, 1.12 )
    ax.set_ylim( -0.08, 1.12 )
    ax.set_aspect( "equal" )
    ax.axis( "off" )
    ax.set_title( "Image( c, degree = 2 ), one pixel", fontsize = 10, family = "monospace", pad = 4 )
    fig.tight_layout()
    fig.savefig( OUT / "img" / "image-bernstein.png", dpi = 110 )
    plt.close( fig )


CASES = {
    "mona"     : mona,
    "degrees"  : degrees,
    "jumps"    : jumps,
    "gyroid"   : gyroid,
    "bernstein": bernstein,
}


if __name__ == "__main__":
    ( OUT / "img" ).mkdir( parents = True, exist_ok = True )
    for name in sys.argv[ 1: ] or CASES:
        CASES[ name ]()
