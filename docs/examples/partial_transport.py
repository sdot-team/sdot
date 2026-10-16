"""The outputs of `partial-transport.md` ( `docs/public/examples/partial-*.html` ), and the picture of the gallery, the last
iteration of Lloyd on a transparent background ( `docs/public/examples/img/partial-transport.png` ).

    python docs/examples/partial_transport.py [ masses lloyd ]

The cells are drawn as they are computed -- Laguerre cells, faded -- and on top of them, in full color, what each one transports:
the part inside its ball, of radius `sqrt( w_i )`. The rest of the target goes to the sink.
"""
import sys
from pathlib import Path

import numpy as np
import sdot

OUT = Path( __file__ ).resolve().parents[ 1 ] / "public" / "examples"
BACKGROUND = "#ffffff"


def masses():
    """the same 30 diracs, a lighter and lighter source: the balls shrink, the sink takes the rest of the square"""
    rng = np.random.default_rng( 1 )
    pos = rng.uniform( 0, 1, size = ( 30, 2 ) )
    viz = sdot.Visualizer( title = "Partial transport: the mass of the source", frame_axis = "mass", playable = False, background = BACKGROUND,
                           frame_value = 0.1 )
    for k, m in enumerate( ( 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99 ) ):
        if k:
            viz.new_frame( m )
        p = sdot.plan( sdot.SumOfDiracs( pos, target_mass = m ), sdot.Box() )
        p.add_to_viz( viz )
    viz.write_html( OUT / "partial-masses.html" )


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
    viz = sdot.Visualizer( title = "Partial transport: Lloyd", frame_axis = "iteration", background = BACKGROUND )
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
    fig.savefig( OUT / "img" / "partial-transport.png", dpi = 100, transparent = True )
    plt.close( fig )


CASES = {
    "masses": masses,
    "lloyd" : lloyd,
}

if __name__ == "__main__":
    for name in sys.argv[ 1: ] or CASES:
        CASES[ name ]()
