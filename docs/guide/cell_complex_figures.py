"""The figures of `cell-complex.md` ( `docs/public/guide/cell-complex-*.html` ) and the card of the gallery
( `docs/public/guide/img/cell-complex.png` ).

    python docs/guide/cell_complex_figures.py [ walk-3d laguerre-graph torus simplices png ]
"""
import sys
from pathlib import Path

import numpy as np
import sdot

OUT = Path( __file__ ).resolve().parents[ 1 ] / "public" / "guide"


def walk_3d( n = 60 ):
    """From the cell at the center ( wireframe, black ) to its neighbors ( the other sides of its facets, wireframe, grey ), to
    its longest edge ( red ), to the facets around that edge ( its parents, solid ), to the cells around it ( their seeds, red )."""
    rng = np.random.default_rng( 1 )
    positions = rng.random( ( n, 3 ) )
    pd = sdot.PowerDiagram( positions, boundaries = sdot.box_half_spaces( [ 0, 0, 0 ], [ 1, 1, 1 ] ) )
    cx = pd.cell_complex()

    cell = cx.cells[ int( np.argmin( np.linalg.norm( positions - 0.5, axis = 1 ) ) ) ]
    neighbors = sorted( { s for f in cell.children for s in f.sides if s >= 0 and s != cell.seed } )
    edge = max( cell.faces( 1 ), key = lambda e: e.measure )

    viz = sdot.Visualizer( title = "Walking the cell complex" )
    cx.cells.add_to_viz( viz, ids = neighbors, faces = False, edge_color = "#c8c8c8" )
    cx.cells.add_to_viz( viz, ids = [ cell.seed ], faces = False, edge_color = "#000000" )
    for f, c in zip( edge.parents, ( "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd" ) ):
        f.add_to_viz( viz, color = c )
    edge.add_to_viz( viz, color = "#d62728" )
    viz.add_points( positions[ [ c.seed for c in edge.cells ] ], color = "#d62728" )
    viz.write_html( OUT / "cell-complex-walk-3d.html" )


def laguerre_graph( n = 80 ):
    """The cells, the seeds joined across each facet they share ( the Laguerre graph ), and the facets on the domain in red."""
    rng = np.random.default_rng( 0 )
    positions = rng.random( ( n, 2 ) )
    pd = sdot.PowerDiagram( positions, 0.002 * rng.random( n ), boundaries = sdot.box_half_spaces( [ 0, 0 ], [ 1, 1 ] ) )
    cx = pd.cell_complex()

    viz = sdot.Visualizer( title = "Laguerre graph" )
    viz.add( pd, opacity = 0.3 )
    sides = cx.facets.sides
    inner = sides[ :, 1 ] >= 0
    viz.add_edges( positions, sides[ inner ], color = "#000000" )
    cx.facets.add_to_viz( viz, ids = np.flatnonzero( ~inner ), color = "#d62728" )
    viz.write_html( OUT / "cell-complex-laguerre-graph.html" )


def torus( n = 40 ):
    """A torus: the cells and their copies by the group ( fainter ), the facets across the period in red -- each once, on the
    copy of one of its cells -- and the Laguerre graph of the quotient, each edge from a seed to the image of the other."""
    rng = np.random.default_rng( 2 )
    positions = rng.random( ( n, 2 ) )
    pd = sdot.PowerDiagram( positions, boundaries = sdot.box_half_spaces( [ 0, 0 ], [ 1, 1 ] ), periodicity = True )
    cx = pd.cell_complex()

    viz = sdot.Visualizer( title = "The cell complex of a torus" )
    viz.add( pd, opacity = 0.35 )
    sides, sv = cx.facets.sides, cx.facets.seed_vectors
    across = np.flatnonzero( np.abs( sv - ( positions[ sides[ :, 1 ] ] - positions[ sides[ :, 0 ] ] ) ).max( axis = 1 ) > 1e-6 )
    cx.facets.add_to_viz( viz, ids = across, color = "#d62728" )
    a = positions[ sides[ :, 0 ] ]
    pts = np.concatenate( [ a, a + sv ] )
    viz.add_edges( pts, np.stack( [ np.arange( len( a ) ), np.arange( len( a ) ) + len( a ) ], axis = 1 ), color = "#000000" )
    viz.write_html( OUT / "cell-complex-torus.html" )
    print( f"torus: { cx }, { len( across ) } facets across the period" )


def simplices( n = 14 ):
    """The cells cut into triangles ( `simplices` ), points coloured by their distance to the boundary of their cell
    ( `boundary_distances` ), and for some of them the segment to the nearest facet."""
    rng = np.random.default_rng( 4 )
    positions = rng.random( ( n, 2 ) )
    pd = sdot.PowerDiagram( positions, boundaries = sdot.box_half_spaces( [ 0, 0 ], [ 1, 1 ] ) )
    cx = pd.cell_complex()

    viz = sdot.Visualizer( title = "Simplices and distances" )
    S, owner = cx.simplices()
    tri = S.reshape( -1, 2 )
    spokes = ( 3 * np.arange( len( S ) )[ :, None, None ] + np.array( [ [ 0, 1 ], [ 0, 2 ] ] )[ None ] ).reshape( -1, 2 )
    viz.add_edges( tri, spokes, color = ( "#c0c0c0", "#5a5a5a" ) )   # ( from the centroid of each cell to its vertices )
    cx.edges.add_to_viz( viz, color = ( "#000000", "#ffffff" ) )
    X = rng.random( ( 2500, 2 ) )
    dist, facets = cx.boundary_distances( X )
    import matplotlib
    cmap = matplotlib.colormaps[ "viridis" ]
    top = float( dist.max() )
    for l in range( 12 ):
        sel = ( dist >= top * l / 12 ) & ( dist <= top * ( l + 1 ) / 12 )
        r, g, b, _ = cmap( l / 11 )
        viz.add_points( X[ sel ], color = "#{:02x}{:02x}{:02x}".format( int( 255 * r ), int( 255 * g ), int( 255 * b ) ) )
    m = 25
    cells, _ = cx.locate( X[ :m ] )
    sides, normals = cx.facets.sides[ facets[ :m ] ], cx.facets.normals[ facets[ :m ] ]
    out = np.where( ( sides[ :, 0 ] == cells )[ :, None ], normals, -normals )      # ( outward of the point's cell )
    foot = X[ :m ] + dist[ :m, None ] * out
    viz.add_edges( np.concatenate( [ X[ :m ], foot ] ), np.stack( [ np.arange( m ), np.arange( m ) + m ], axis = 1 ), color = "#d62728" )
    viz.write_html( OUT / "cell-complex-simplices.html" )


def write_png( n = 60 ):
    """The card of the gallery: the complex of a torus, tiled -- pastel cells, their edges and vertices, the Laguerre graph of the
    quotient, and the square the group repeats."""
    import matplotlib
    matplotlib.use( "Agg" )
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    rng = np.random.default_rng( 7 )
    positions = sdot.quantization( sdot.Box( nb_dims = 2 ), n, seed = 3 )
    positions = np.asarray( positions ) + 0.01 * rng.standard_normal( ( n, 2 ) )
    pd = sdot.PowerDiagram( positions % 1.0, boundaries = sdot.box_half_spaces( [ 0, 0 ], [ 1, 1 ] ), periodicity = True )
    cx = pd.cell_complex()
    p = np.asarray( pd.positions )
    hues = plt.get_cmap( "Set3" )( np.arange( n ) % 12 )

    fig = plt.figure( figsize = ( 8, 6 ) )
    ax = fig.add_axes( [ 0, 0, 1, 1 ] )
    polys, cols, segs, verts, graph = [], [], [], [], []
    for ox in ( -1, 0, 1 ):
        for oy in ( -1, 0, 1 ):
            o = np.array( [ ox, oy ] )
            for c in cx.cells:
                polys.append( c.points + o )
                cols.append( hues[ c.seed ] )
            for e in cx.edges:
                segs.append( e.points + o )
            verts.append( cx.vertex_positions + o )
            sides, sv = cx.facets.sides, cx.facets.seed_vectors
            graph += [ [ p[ a ] + o, p[ a ] + v + o ] for ( a, _ ), v in zip( sides, sv ) ]
    ax.add_collection( PolyCollection( polys, facecolors = cols, edgecolors = "none" ) )
    ax.add_collection( LineCollection( graph, colors = "#555555", linewidths = 0.6, alpha = 0.6 ) )
    ax.add_collection( LineCollection( segs, colors = "#222222", linewidths = 1.2 ) )
    v = np.concatenate( verts )
    ax.scatter( v[ :, 0 ], v[ :, 1 ], s = 7, color = "#222222", zorder = 3 )
    tile = np.concatenate( [ p + [ ox, oy ] for ox in ( -1, 0, 1 ) for oy in ( -1, 0, 1 ) ] )
    ax.scatter( tile[ :, 0 ], tile[ :, 1 ], s = 9, color = "#d62728", zorder = 4 )
    ax.plot( [ 0, 1, 1, 0, 0 ], [ 0, 0, 1, 1, 0 ], "--", color = "#000000", linewidth = 1.5, zorder = 5 )
    ax.set_xlim( -0.25, 1.25 )
    ax.set_ylim( -0.125, 1.125 )
    ax.set_aspect( "equal" )
    ax.axis( "off" )
    ( OUT / "img" ).mkdir( exist_ok = True )
    fig.savefig( OUT / "img" / "cell-complex.png", dpi = 100, transparent = True )
    plt.close( fig )


CASES = {
    "walk-3d"       : walk_3d,
    "laguerre-graph": laguerre_graph,
    "torus"         : torus,
    "simplices"     : simplices,
    "png"           : write_png,
}

if __name__ == "__main__":
    for name in sys.argv[ 1: ] or CASES:
        CASES[ name ]()
