"""The outputs of `lot-dictionary.md`: draws of a family of rings are embedded by their transport map from the Lebesgue
measure of the unit square ( linearized optimal transport ), a PCA of these maps recovers the family, and new draws are
identified by projection. One page per figure ( `docs/public/examples/lot-*.html` ), the picture of the gallery
( `docs/public/examples/img/lot-dictionary.png` ), and the numbers of the tables of the page, printed.

    python docs/examples/lot_dictionary.py [ cells modes identify outlier sizes png ]

The 2D cases share the training set ( 200 transports of 10^4 diracs, about two minutes on a laptop ).
"""
import sys
from functools import cache
from pathlib import Path

import numpy as np
import sdot

OUT = Path( __file__ ).resolve().parents[ 1 ] / "public" / "examples"

R_IN, R_OUT = 0.25, 0.4     # the inner and outer "radius" of the rings ( circle radius, or half side of the square )
NB_TRAIN = 200              # draws to learn from
NB_TEST = 10                # draws to identify
NB_DIRACS = 10_000          # diracs per draw
NB_READ = 4096              # the maps are read at NB_READ points drawn uniformly in the unit cube, the same for all
NB_MODES = 4                # principal components kept ( see the page: s_in shares modes 2-4 with the tearing of the map )


# -- the family of shapes ---------------------------------------------------------------------------------------------

def radius( theta, s, r ):
    """the boundary in polar coordinates: a circle of radius `r` for `s = 0`, a square of half side `r` for `s = 1`, and
    the linear interpolation of the two radii in between"""
    return r * ( ( 1 - s ) + s / np.maximum( np.abs( np.cos( theta ) ), np.abs( np.sin( theta ) ) ) )


def ring( s_in, s_out, n, rng, rotation = 0.0 ):
    """`n` points drawn uniformly in the ring between the boundaries `( s_in, R_IN )` and `( s_out, R_OUT )`, centered in
    the unit square ( rejection sampling ); `rotation` turns the ring ( outside of the family )"""
    out, nb = [], 0
    while nb < n:
        p = ( 2 * rng.random( ( 4 * n, 2 ) ) - 1 ) * R_OUT * np.sqrt( 2 )
        theta = np.arctan2( p[ :, 1 ], p[ :, 0 ] ) - rotation
        r = np.linalg.norm( p, axis = 1 )
        p = p[ ( r >= radius( theta, s_in, R_IN ) ) & ( r <= radius( theta, s_out, R_OUT ) ) ]
        out.append( p )
        nb += len( p )
    return 0.5 + np.concatenate( out )[ :n ]


def boundary( s, r, rotation = 0.0, nb = 256 ):
    """`[ nb, 2 ]` points of the boundary `( s, r )`, for drawing"""
    theta = np.linspace( 0, 2 * np.pi, nb, endpoint = False )
    rho = radius( theta, s, r )
    return 0.5 + rho[ :, None ] * np.stack( [ np.cos( theta + rotation ), np.sin( theta + rotation ) ], axis = 1 )


# -- the embedding ----------------------------------------------------------------------------------------------------

@cache
def reading_points( d, nb = NB_READ, seed = 123 ):
    """the `[ nb, d ]` points where every map is read: drawn uniformly in the unit cube, once ( a Monte Carlo
    discretization of `L2( rho )`, which must be the same for all the maps to compare them )"""
    return np.random.default_rng( seed ).random( ( nb, d ) )


def transport_map( y, x ):
    """`T( x )`, `[ len( x ), d ]`: the optimal map from the Lebesgue measure of the unit cube to the diracs `y`, read at the
    points `x`. A point goes to the dirac of its power cell, `argmin_i |x - y_i|^2 - w_i`."""
    p = sdot.plan( sdot.SumOfDiracs( y ), sdot.Box() )
    w = np.asarray( p.weights )
    shift = ( y * y ).sum( axis = 1 ) - w
    owner = np.concatenate( [ ( shift[ None ] - 2 * c @ y.T ).argmin( axis = 1 ) for c in np.array_split( x, 16 ) ] )
    return y[ owner ]


def embed( draws ):
    """the maps of the draws, flattened: `[ len( draws ), NB_READ * d ]`"""
    x = reading_points( draws[ 0 ].shape[ 1 ] )
    return np.stack( [ transport_map( y, x ).ravel() for y in draws ] )


class Pca:
    """the mean map and the first principal directions of maps `[ nb, D ]`. The maps are vectors of `L2( rho )` read
    at the reading points: the inner product is the mean over these points, which `scale` puts in the coordinates."""

    def __init__( self, maps, nb_modes = NB_MODES ):
        self.mean = maps.mean( axis = 0 )
        _, sv, vt = np.linalg.svd( maps - self.mean, full_matrices = False )
        self.scale = maps.shape[ 1 ] / 2                     # reading points ( 2 = the dimension of a map's value )
        self.variances = sv ** 2 / len( maps ) / self.scale  # per mode, in L2( rho )
        self.modes = vt[ :nb_modes ]

    def coords( self, maps ):
        return ( maps - self.mean ) @ self.modes.T / np.sqrt( self.scale )

    def map( self, coords ):
        return self.mean + np.sqrt( self.scale ) * coords @ self.modes

    def residual( self, maps ):
        """the L2( rho ) distance of each map to the affine span of the modes ( an upper bound of W2 to the reconstruction )"""
        diff = maps - self.map( self.coords( maps ) )
        return np.sqrt( ( diff ** 2 ).sum( axis = -1 ) / self.scale )


def affine_fit( coords, params ):
    """the least squares affine map from the coordinates to the parameters, as a function"""
    a = np.c_[ coords, np.ones( len( coords ) ) ]
    coef, *_ = np.linalg.lstsq( a, params, rcond = None )
    return lambda c: np.c_[ c, np.ones( len( c ) ) ] @ coef


@cache
def training( n = NB_DIRACS, nb = NB_TRAIN, seed = 0 ):
    """`( params, maps )` of `nb` rings of `n` diracs, `( s_in, s_out )` uniform in [ 0, 1 ]^2"""
    rng = np.random.default_rng( seed )
    params = rng.random( ( nb, 2 ) )
    return params, embed( [ ring( a, b, n, rng ) for a, b in params ] )


@cache
def learned():
    params, maps = training()
    pca = Pca( maps )
    return params, pca, affine_fit( pca.coords( maps ), params )


@cache
def tests( n = NB_DIRACS, nb = NB_TEST, seed = 1 ):
    """`( params, draws, maps )` of the draws to identify"""
    rng = np.random.default_rng( seed )
    params = rng.random( ( nb, 2 ) )
    draws = [ ring( a, b, n, rng ) for a, b in params ]
    return params, draws, embed( draws )


# -- pages ------------------------------------------------------------------------------------------------------------

GREY = ( "#9a9a9a", "#6a6a6a" )
RAMP = [ "#2166ac", "#67a9cf", "#bdbdbd", "#ef8a62", "#b2182b" ]   # 0 -> 1, for the parameters


def add_family_outlines( viz, opacity = 0.35 ):
    """the four boundaries the family moves between: circle and square, inside and outside"""
    for s in ( 0, 1 ):
        for r in ( R_IN, R_OUT ):
            viz.add_edges( boundary( s, r ), closed = True, color = GREY, opacity = opacity, dashed = True, nb_dashes = 1 )


def add_ring_outline( viz, s_in, s_out, rotation = 0.0, color = GREY ):
    viz.add_edges( boundary( s_in, R_IN, rotation ), closed = True, color = color )
    viz.add_edges( boundary( s_out, R_OUT, rotation ), closed = True, color = color )


def cells():
    """the embedding at work: a few draws ( of 1000 diracs, to see the cells ), their power cells in the unit square"""
    rng = np.random.default_rng( 3 )
    viz = sdot.Visualizer( title = "Linearized OT: the map of a draw", frame_axis = "draw", playable = False )
    for k, ( a, b ) in enumerate( [ ( 0, 0 ), ( 1, 0 ), ( 0, 1 ), ( 1, 1 ) ] ):
        if k:
            viz.new_frame( k )
        y = ring( a, b, 1000, rng )
        sdot.plan( sdot.SumOfDiracs( y ), sdot.Box() ).add_to_viz( viz )
    viz.write_html( OUT / "lot-cells.html" )


def fold( maps, radius = 0.03 ):
    """the share of the pairs of reading points closer than `radius` that a map sends in reverse order,
    `( T( x ) - T( x' ) ) . ( x - x' ) < 0`: 0 for a transport map ( the gradient of a convex function ), more when the
    map folds onto itself"""
    x = reading_points( 2 )
    i, j = np.nonzero( np.triu( ( ( x[ :, None ] - x[ None ] ) ** 2 ).sum( axis = -1 ) < radius ** 2, k = 1 ) )
    t = maps.reshape( len( maps ), -1, 2 )
    return ( ( ( t[ :, i ] - t[ :, j ] ) * ( x[ i ] - x[ j ] ) ).sum( axis = -1 ) < 0 ).mean( axis = 1 )


def modes( nb_points = 2048, nb_steps = 9, max_fold = 2e-4 ):
    """the learned dictionary: the reconstruction ( mean + c_1 v_1 + c_2 v_2 )_# rho, on a full grid of the first two
    coordinates ( every combination is a frame: the sliders move independently ). The grid is the box of the coordinates
    of the training draws, shrunk towards the mean until the reconstruction is a transport map all over it ( checked on a
    grid twice as fine ): farther out, it extrapolates and folds ( `fold` ). `nb_points` of the reading points are
    drawn."""
    _, pca, _ = learned()
    _, maps = training()
    v = pca.variances
    print( "2d: share of the variance per mode", np.round( v[ :8 ] / v.sum(), 4 ) )
    seen = pca.coords( maps )[ :, :2 ]
    lo, hi = seen.min( axis = 0 ), seen.max( axis = 0 )

    def grid( scale, nb ):
        axes = [ scale * np.linspace( l, h, nb ) for l, h in zip( lo, hi ) ]
        return np.stack( np.meshgrid( *axes, indexing = "ij" ), axis = -1 ).reshape( -1, 2 )

    def recon( nodes ):
        return pca.map( np.c_[ nodes, np.zeros( ( len( nodes ), NB_MODES - 2 ) ) ] )

    scale = next( t for t in np.linspace( 1, 0.3, 15 ) if fold( recon( grid( t, 2 * nb_steps - 1 ) ) ).max() <= max_fold )
    print( f"modes: the box of the training coordinates, scaled by { scale:.2f}; training maps fold at most "
           f"{ fold( maps ).max():.1e}" )

    nodes = grid( scale, nb_steps )
    maps_at = recon( nodes )
    # the node closest to the mean last: the page opens on it ( the first frame takes its value from the constructor )
    order = np.argsort( -( ( nodes / np.sqrt( v[ :2 ] ) ) ** 2 ).sum( axis = 1 ) )
    viz = sdot.Visualizer( title = "Linearized OT: the learned modes", frame_axis = ( "c1", "c2" ), playable = False,
                           frame_value = tuple( nodes[ order[ 0 ] ] ) )
    for i, k in enumerate( order ):
        if i:
            viz.new_frame( tuple( nodes[ k ] ) )
        add_family_outlines( viz )
        viz.add_points( maps_at[ k ].reshape( -1, 2 )[ :nb_points ] )
    viz.write_html( OUT / "lot-modes.html" )


def identify():
    """draws that were not learned: their diracs, the reconstruction from their 4 coordinates, the true boundary"""
    _, pca, regress = learned()
    params, draws, maps = tests()
    c = pca.coords( maps )
    estimate = regress( c )
    res = pca.residual( maps )
    print( "identify:  s_in  s_in~  s_out  s_out~  residual" )
    for p, e, r in zip( params, estimate, res ):
        print( f"           { p[ 0 ]:.2f}  { e[ 0 ]:.2f}   { p[ 1 ]:.2f}   { e[ 1 ]:.2f}    { r:.4f}" )
    print( "identify: mean error", np.round( np.abs( estimate - params ).mean( axis = 0 ), 3 ) )

    viz = sdot.Visualizer( title = "Linearized OT: identifying new draws", frame_axis = "draw", playable = False )
    for k, ( p, y, ck ) in enumerate( zip( params, draws, c ) ):
        if k:
            viz.new_frame( k )
        viz.add_points( y, color = GREY, opacity = 0.5 )
        viz.add_points( pca.map( ck[ None ] )[ 0 ].reshape( -1, 2 ), color = "#d62728" )
        add_ring_outline( viz, *p )
    viz.write_html( OUT / "lot-identify.html" )


def outlier():
    """a ring outside of the family ( a square turned by 45 degrees ) next to its unturned twin, and their reconstructions"""
    _, pca, regress = learned()
    _, _, test_maps = tests()
    rng = np.random.default_rng( 2 )
    cases = [ ( "square", 0.0 ), ( "turned square", np.pi / 4 ) ]
    draws = [ ring( 0.0, 1.0, NB_DIRACS, rng, rotation = rot ) for _, rot in cases ]
    maps = embed( draws )
    c = pca.coords( maps )
    res = pca.residual( maps )
    ref = pca.residual( test_maps )
    print( f"outlier: residual of the test draws { ref.min():.4f} .. { ref.max():.4f}" )
    for ( name, _ ), r, e in zip( cases, res, regress( c ) ):
        print( f"outlier: { name }: residual { r:.4f}, ( s_in, s_out ) read as { np.round( e, 2 ) }" )

    viz = sdot.Visualizer( title = "Linearized OT: outside of the family", frame_axis = "case", playable = False )
    for k, ( ( _, rot ), y, ck ) in enumerate( zip( cases, draws, c ) ):
        if k:
            viz.new_frame( k )
        viz.add_points( y, color = GREY, opacity = 0.5 )
        viz.add_points( pca.map( ck[ None ] )[ 0 ].reshape( -1, 2 ), color = "#d62728" )
        add_ring_outline( viz, 0.0, 1.0, rot )
    viz.write_html( OUT / "lot-outlier.html" )


def sizes():
    """the error on ( s_in, s_out ) of the test draws against the number of diracs per draw"""
    print( "sizes:  n      share of modes 1-4                 error s_in  s_out" )
    for n in ( 1000, 4000, NB_DIRACS ):
        params, maps = training( n )
        pca = Pca( maps )
        regress = affine_fit( pca.coords( maps ), params )
        tp, _, tm = tests( n )
        err = np.abs( regress( pca.coords( tm ) ) - tp ).mean( axis = 0 )
        share = pca.variances[ :NB_MODES ] / pca.variances.sum()
        print( f"       { n:<6} { np.round( share, 3 ) }    { err[ 0 ]:.3f}       { err[ 1 ]:.3f}" )


def write_png():
    """the picture of the gallery: the reconstructions ( from 4 coordinates ) of the training draws closest to the four
    corners of the family"""
    import matplotlib
    matplotlib.use( "Agg" )
    import matplotlib.pyplot as plt

    params, pca, _ = learned()
    _, maps = training()
    fig, axs = plt.subplots( 2, 2, figsize = ( 8, 6 ) )
    fig.subplots_adjust( 0, 0, 1, 1, 0, 0 )
    for ax, corner in zip( axs.ravel(), [ ( 0, 1 ), ( 1, 1 ), ( 0, 0 ), ( 1, 0 ) ] ):
        i = np.argmin( ( ( params - corner ) ** 2 ).sum( axis = 1 ) )
        p = pca.map( pca.coords( maps[ i : i + 1 ] ) )[ 0 ].reshape( -1, 2 )
        ax.scatter( p[ :, 0 ], p[ :, 1 ], s = 1.5, color = RAMP[ 4 * corner[ 1 ] ], linewidths = 0 )
        ax.set_xlim( 0.0, 1.0 )
        ax.set_ylim( 0.0, 1.0 )
        ax.set_aspect( "equal" )
        ax.axis( "off" )
    fig.savefig( OUT / "img" / "lot-dictionary.png", dpi = 100, transparent = True )
    plt.close( fig )


CASES = {
    "cells"   : cells,
    "modes"   : modes,
    "identify": identify,
    "outlier" : outlier,
    "sizes"   : sizes,
    "png"     : write_png,
}

if __name__ == "__main__":
    for name in sys.argv[ 1: ] or CASES:
        CASES[ name ]()
