"""The outputs of `finite-volumes.md`: the heat equation on a torus, by finite volumes on the cells of a power diagram -- the
animated page ( `docs/public/examples/finite-volumes-heat.html` ), the card of the gallery
( `docs/public/examples/img/finite-volumes.png` ), and the decay of a Fourier mode against the exact one ( printed ).

    python docs/examples/finite_volumes.py [ heat mode png ]
"""
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import sdot

OUT = Path( __file__ ).resolve().parents[ 1 ] / "public" / "examples"
LEVELS = 64


def mesh( n, seed = 0 ):
    """`( positions, complex, cell volumes, laplacian )`: the cells of `n` seeds spread on the torus by a Lloyd quantization, and
    the matrix `L` of the finite-volume scheme, `( L u )_i = sum_f |f| / |p_i - p_j^f| ( u_i - u_j )` over the facets `f` of cell
    `i`, `p_j^f` the image of the seed across `f`"""
    positions = np.asarray( sdot.quantization( sdot.Box( nb_dims = 2 ), n, seed = seed, periodicity = True ) ) % 1.0
    pd = sdot.PowerDiagram( positions, boundaries = sdot.box_half_spaces( [ 0, 0 ], [ 1, 1 ] ), periodicity = True )
    cx = pd.cell_complex()

    a, b = cx.facets.sides.T
    keep = a != b                                        # ( a cell that meets itself exchanges nothing )
    a, b = a[ keep ], b[ keep ]
    t = cx.facets.measures[ keep ] / np.linalg.norm( cx.facets.seed_vectors[ keep ], axis = 1 )
    L = sp.coo_matrix( ( np.concatenate( [ t, t, -t, -t ] ), ( np.concatenate( [ a, b, a, b ] ), np.concatenate( [ a, b, b, a ] ) ) ),
                       shape = ( n, n ) ).tocsr()
    return positions, cx, cx.cells.measures, L


def solve( vol, L, u0, times ):
    """implicit Euler, `( M + dt L ) u^{k+1} = M u^k`, the field at each of `times` ( steps of `dt = times[ 0 ]` )"""
    dt = times[ 0 ]
    M = sp.diags( vol )
    step = spl.factorized( ( M + dt * L ).tocsc() )
    res, u, t = [], u0.copy(), 0.0
    for target in times:
        while t < target - 1e-12:
            u = step( vol * u )
            t += dt
        res.append( u.copy() )
    return res


def initial( p ):
    """a hot disc across the corner of the square ( it is one disc on the torus ) and a cold band"""
    d = p - np.array( [ 0.95, 0.9 ] )
    d -= np.round( d )
    u = np.where( ( d ** 2 ).sum( axis = 1 ) < 0.2 ** 2, 1.0, 0.0 )
    return u - 0.8 * ( np.abs( p[ :, 0 ] - 0.35 ) < 0.06 )


def heat( n = 1500 ):
    positions, cx, vol, L = mesh( n )
    u0 = initial( positions )
    times = np.concatenate( [ [ 2e-5 ], np.geomspace( 1e-4, 0.03, 23 ) ] )
    times = np.round( times / 2e-5 ) * 2e-5
    fields = [ u0 ] + solve( vol, L, u0, times )
    print( f"heat: { cx }, total heat { ( vol * fields[ 0 ] ).sum():.6f} -> { ( vol * fields[ -1 ] ).sum():.6f}" )

    polys = [ c.points for c in cx.cells ]
    viz = sdot.Visualizer( title = "Heat on a torus", frame_axis = "t" )
    for k, u in enumerate( fields ):
        if k:
            viz.new_frame( float( times[ k - 1 ] ) )
        lev = np.clip( ( ( u + 0.8 ) / 1.8 * ( LEVELS - 1 ) ).round().astype( int ), 0, LEVELS - 1 )
        for l in np.unique( lev ):
            ids = np.flatnonzero( lev == l )
            pts = np.concatenate( [ polys[ i ] for i in ids ] )
            starts = np.cumsum( [ 0 ] + [ len( polys[ i ] ) for i in ids ] )
            viz.add_faces( pts, [ list( range( starts[ j ], starts[ j + 1 ] ) ) for j in range( len( ids ) ) ], color = _color( l ) )
        viz.add_edges( np.array( [ [ 0, 0 ], [ 1, 0 ], [ 1, 1 ], [ 0, 1 ] ], float ), color = "#ffffff", closed = True, dashed = True )
    viz.write_html( OUT / "finite-volumes-heat.html" )


def mode( sizes = ( 250, 1000, 4000 ), k = 1, t = 0.01 ):
    """the decay of `cos( 2 pi k x )`, exact: `exp( -4 pi^2 k^2 t )`, against the scheme -- the error of the scheme in space
    ( a small `dt`, the implicit Euler error removed by comparing to its own exact factor )"""
    for n in sizes:
        positions, cx, vol, L = mesh( n, seed = 1 )
        u0 = np.cos( 2 * np.pi * k * positions[ :, 0 ] )
        dt = t / 200
        u = solve( vol, L, u0, [ dt * ( i + 1 ) for i in range( 200 ) ] )[ -1 ]
        lam = 4 * np.pi ** 2 * k ** 2
        exact = ( 1 / ( 1 + lam * dt ) ) ** 200            # what implicit Euler does to the exact mode
        rate = ( vol * u * u0 ).sum() / ( vol * u0 * u0 ).sum()
        print( f"mode: n = { n:5d}   decay { rate:.5f}   exact { exact:.5f}   relative error { abs( rate - exact ) / exact:.1e}" )


def _color( l ):
    import matplotlib
    r, g, b, _ = matplotlib.colormaps[ "inferno" ]( 0.08 + 0.9 * l / ( LEVELS - 1 ) )
    return "#{:02x}{:02x}{:02x}".format( int( 255 * r ), int( 255 * g ), int( 255 * b ) )


def write_png( n = 1500 ):
    """the card: the field at a middle time, the cells tiled over the torus and cut to the square"""
    import matplotlib
    matplotlib.use( "Agg" )
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    positions, cx, vol, L = mesh( n )
    u = solve( vol, L, initial( positions ), [ 2e-4, 2e-3 ] )[ -1 ]
    cmap = matplotlib.colormaps[ "inferno" ]
    fig = plt.figure( figsize = ( 8, 6 ) )
    ax = fig.add_axes( [ 0, 0, 1, 1 ] )
    polys = [ c.points + [ ox, oy ] for ox in ( -1, 0, 1 ) for oy in ( -1, 0, 1 ) for c in cx.cells ]
    cols = np.tile( cmap( 0.08 + 0.9 * np.clip( ( u + 0.8 ) / 1.8, 0, 1 ) ), ( 9, 1 ) )
    ax.add_collection( PolyCollection( polys, facecolors = cols, edgecolors = "#00000033", linewidths = 0.3 ) )
    ax.set_xlim( 0, 1 )
    ax.set_ylim( 0.125, 0.875 )
    ax.set_aspect( "equal" )
    ax.axis( "off" )
    fig.savefig( OUT / "img" / "finite-volumes.png", dpi = 100 )
    plt.close( fig )


CASES = {
    "heat": heat,
    "mode": mode,
    "png" : write_png,
}

if __name__ == "__main__":
    for name in sys.argv[ 1: ] or CASES:
        CASES[ name ]()
