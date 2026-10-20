"""The figures of `partial.md`: pictures ( `docs/public/guide/img/partial-*.png` ) and interactive pages of the HTML viewer
( `sdot.Visualizer`, `docs/public/guide/partial-*.html` ).

    python docs/guide/ground-metrics/partial_figures.py [ power masses own_masses lloyd ]

The cells are drawn as they are computed -- Laguerre cells, faded -- and on top of them, in full color, what each one transports:
the part inside its ball, of radius `sqrt( w_i )`. The rest of the target goes to the sink. `lloyd` also writes the card of the
gallery, its last iteration on a transparent background ( `docs/public/examples/img/partial-transport.png` ).
"""
import sys
from pathlib import Path

import numpy as np
import sdot

OUT = Path( __file__ ).resolve().parents[ 2 ] / "public" / "guide"
CARD = Path( __file__ ).resolve().parents[ 2 ] / "public" / "examples" / "img" / "partial-transport.png"
COLORS = [ "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1" ]
SINK = "#bbbbbb"


# -- matplotlib -------------------------------------------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use( "Agg" )
    import matplotlib.pyplot as plt
    return plt


def _save( fig, name ):
    ( OUT / "img" ).mkdir( parents = True, exist_ok = True )
    fig.tight_layout()
    fig.savefig( OUT / "img" / f"partial-{ name }.png", dpi = 110, facecolor = "#ffffff", bbox_inches = "tight" )


def power():
    """1D: the power distances to three diracs ( weights chosen by hand ), and the sink at 0. A point goes to the lowest curve
    when it is negative, to the sink otherwise: the cell of a dirac is its Laguerre interval cut by its ball"""
    plt = _plt()
    p = np.array( [ 0.2, 0.47, 0.8 ] )
    w = np.array( [ 0.016, 0.028, 0.009 ] )
    x = np.linspace( 0, 1, 2001 )
    d = ( x[ None, : ] - p[ :, None ] ) ** 2 - w[ :, None ]
    owner = np.where( d.min( 0 ) < 0, d.argmin( 0 ), -1 )

    fig, ax = plt.subplots( figsize = ( 8.5, 3.9 ) )
    for i in range( 3 ):
        ax.plot( x, d[ i ], color = COLORS[ i ], linewidth = 1.2, alpha = 0.45 )
        ax.plot( p[ i ], - w[ i ], "o", color = COLORS[ i ], markersize = 4 )
        ax.annotate( f"$-w_{ i + 1 }$", ( p[ i ], - w[ i ] ), ( 6, -12 ), textcoords = "offset points", fontsize = 9, color = COLORS[ i ] )
    ax.plot( x, np.minimum( d.min( 0 ), 0 ), color = "k", linewidth = 2 )
    ax.axhline( 0, color = SINK, linewidth = 1.5, linestyle = "--" )
    ax.annotate( "the sink: $0$", ( 1.0, 0 ), ( -4, 5 ), textcoords = "offset points", ha = "right", fontsize = 9, color = "#777777" )

    # the band under the axis: who takes each point
    y0, h = -0.052, 0.012
    for k, ( a, b ) in enumerate( _runs( owner ) ):
        o = owner[ a ]
        ax.fill_between( x[ a:b ], y0, y0 + h, color = SINK if o < 0 else COLORS[ o ], linewidth = 0 )
        if o >= 0:
            ax.annotate( "", ( x[ a ], y0 + h + 0.004 ), ( x[ b - 1 ], y0 + h + 0.004 ),
                         arrowprops = dict( arrowstyle = "<->", color = COLORS[ o ], linewidth = 0.8 ) )
    for i in range( 3 ):
        ax.plot( p[ i ], y0 + h / 2, "ko", markersize = 4 )
    ax.annotate( "who takes $x$", ( 0, y0 ), ( 0, -14 ), textcoords = "offset points", fontsize = 9 )

    ax.set_xlim( 0, 1 )
    ax.set_ylim( -0.07, 0.04 )
    ax.set_xlabel( "$x$" )
    ax.set_yticks( [] )
    ax.set_title( "power distances $|x - p_i|^2 - w_i$, their minimum, and the sink", fontsize = 10 )
    for s in ( "top", "right", "left" ):
        ax.spines[ s ].set_visible( False )
    _save( fig, "power" )


def _runs( owner ):
    """the ranges `[ a, b )` where `owner` is constant"""
    cut = np.flatnonzero( np.diff( owner ) ) + 1
    bounds = np.r_[ 0, cut, len( owner ) ]
    return list( zip( bounds[ :-1 ], bounds[ 1: ] ) )


# -- the viewer -------------------------------------------------------------------------------------------------------------

def _bands():
    """a density in vertical bands, slowly growing along y"""
    g = ( np.arange( 64 ) + 0.5 ) / 64
    return sdot.Image( 0.25 + np.sin( 3 * np.pi * g[ :, None ] ) ** 2 * ( 0.6 + 0.4 * g[ None, : ] ) )


def masses( n = 40 ):
    """the same diracs on a density in bands, one frame per mass of the source: the balls are small where the density is high
    ( the same mass on a smaller area ), and grow until they meet; at a mass of 1 the transport is balanced"""
    target = _bands()
    pos = np.random.default_rng( 2 ).uniform( 0.03, 0.97, size = ( n, 2 ) )
    mass = [ 0.15, 0.3, 0.5, 0.7, 0.85, 1.0 ]
    viz = sdot.Visualizer( title = "Partial transport: the mass of the source", frame_axis = "mass", playable = False,
                           frame_value = mass[ 0 ] )
    for k, m in enumerate( mass ):
        if k:
            viz.new_frame( m )
        p = sdot.plan( sdot.SumOfDiracs( pos, target_mass = m ), target )
        print( f"masses: { m }, converged: { p.converged }" )
        p.add_to_viz( viz )
    viz.write_html( OUT / "partial-masses.html" )


def own_masses( n = 30 ):
    """`target_mass = None`: the source weighs what its weights sum to ( 30 uneven diracs summing to 0.45 ), against a box of mass
    1 -- a heavy dirac takes a bigger ball"""
    rng = np.random.default_rng( 3 )
    pos = rng.uniform( 0.08, 0.92, size = ( n, 2 ) )
    masses = rng.uniform( 0.2, 1, size = n ) ** 2
    masses *= 0.45 / masses.sum()
    p = sdot.plan( sdot.SumOfDiracs( pos, masses, target_mass = None ), sdot.Box( nb_dims = 2 ) )
    print( f"own_masses: converged: { p.converged }" )
    viz = sdot.Visualizer( title = "30 diracs of uneven masses summing to 0.45, target_mass = None" )
    p.add_to_viz( viz )
    viz.write_html( OUT / "partial-own-masses.html" )


def lloyd():
    """Lloyd's iterations with a source of mass 0.4 on a density with two bumps: the diracs gather where the density is high"""
    g = ( np.arange( 64 ) + 0.5 ) / 64
    X, Y = np.meshgrid( g, g, indexing = "ij" )
    values = 0.05 + np.exp( - ( ( X - 0.3 ) ** 2 + ( Y - 0.35 ) ** 2 ) / 0.02 ) + 0.7 * np.exp( - ( ( X - 0.72 ) ** 2 + ( Y - 0.7 ) ** 2 ) / 0.01 )
    target = sdot.Image( values, frame = [ 1.0, 1.0 ] )
    rng = np.random.default_rng( 2 )
    x = rng.uniform( 0, 1, size = ( 40, 2 ) )
    pb = sdot.OtProblem( sdot.SumOfDiracs( x, target_mass = 0.4 ), target )
    nb_iter, kept = 40, { 0, 1, 2, 3, 5, 8, 12, 18, 26, 39 }
    viz = sdot.Visualizer( title = "Partial transport: Lloyd", frame_axis = "iteration" )
    first = True
    for it in range( nb_iter ):
        pb.source = sdot.SumOfDiracs( x, target_mass = 0.4 )
        p = pb.solve()
        if it in kept:
            if not first:
                viz.new_frame( it )
            first = False
            p.add_to_viz( viz )
        x = np.asarray( p.barycenters ).reshape( x.shape )
    viz.write_html( OUT / "partial-lloyd.html" )
    write_png( p )


def write_png( p ):
    """the picture of the gallery: the faded Laguerre cells and what each dirac transports, on a transparent background"""
    import matplotlib
    matplotlib.use( "Agg" )
    import matplotlib.pyplot as plt
    from sdot.PowerDiagram import _disc_in_polygon

    pd = p.power_diagram()
    cells = pd.cells
    pos = np.asarray( pd.positions ).reshape( -1, 2 )
    r2 = np.asarray( pd.weights ).reshape( -1 )
    colors = plt.get_cmap( "tab20" ).colors
    # 4:3 like the cards of the gallery, which crop to fill
    fig = plt.figure( figsize = ( 8, 6 ) )
    ax = fig.add_axes( ( 0, 0, 1, 1 ) )
    for i in range( len( pos ) ):
        v = cells.vertices( i )
        col = colors[ i % len( colors ) ]
        if len( v ) >= 3:
            ax.fill( v[ :, 0 ], v[ :, 1 ], color = col, alpha = 0.2, linewidth = 0.5, edgecolor = "#00000044" )
        if r2[ i ] > 0:
            b = _disc_in_polygon( v, pos[ i ], np.sqrt( r2[ i ] ) )
            if len( b ) >= 3:
                ax.fill( b[ :, 0 ], b[ :, 1 ], color = col, linewidth = 1, edgecolor = "#000000" )
        ax.plot( pos[ i, 0 ], pos[ i, 1 ], "k.", markersize = 3 )
    ax.set_xlim( -1 / 6, 7 / 6 )
    ax.set_ylim( 0, 1 )
    ax.set_aspect( "equal" )
    ax.axis( "off" )
    fig.savefig( CARD, dpi = 100, transparent = True )
    plt.close( fig )


CASES = {
    "power"     : power,
    "masses"    : masses,
    "own_masses": own_masses,
    "lloyd"     : lloyd,
}


if __name__ == "__main__":
    OUT.mkdir( parents = True, exist_ok = True )
    for name in sys.argv[ 1: ] or CASES:
        CASES[ name ]()
