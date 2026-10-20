"""The pages of `mesh.md`: interactive pages of the HTML viewer ( `sdot.Visualizer`, `docs/public/guide/mesh-*.html` ).

    python docs/guide/distributions/mesh_figures.py [ contours p1 bands hole tube ]

`contours` and `hole` need `pygmsh` ( `Mesh.from_contours` ).
"""
import sys
import time
from pathlib import Path

import numpy as np
import sdot
from scipy.spatial import Delaunay

OUT = Path( __file__ ).resolve().parents[ 2 ] / "public" / "guide"


# -- the meshes -------------------------------------------------------------------------------------------------------------

def gear( k = 9, r0 = 0.3, r1 = 0.42, c = ( 0.5, 0.5 ) ):
    """the border of a gear of `k` teeth ( `[ n, 2 ]`, in order ): a non-convex region"""
    t = np.linspace( 0, 2 * np.pi, 8 * k, endpoint = False )
    r = np.where( ( t * k / ( 2 * np.pi ) ) % 1 < 0.5, r1, r0 )
    return np.stack( [ c[ 0 ] + r * np.cos( t ), c[ 1 ] + r * np.sin( t ) ], -1 )


def square_mesh( n = 2000, seed = 0 ):
    """the Delaunay triangulation of the unit square: its corners and `n` random points"""
    rng = np.random.default_rng( seed )
    nodes = np.concatenate( [ [ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ], [ 1, 1 ] ], rng.random( ( n, 2 ) ) ] )
    return nodes, Delaunay( nodes ).simplices


def wavy_bands( nx = 48, rows = ( 5, 7, 4, 8, 6 ), amplitude = 0.06 ):
    """the unit square, a grid of `nx` columns and `sum( rows )` rows bent by a wave, cut into triangles: `( nodes, triangles,
    band )`, `band` the number of the band of each triangle ( `rows` rows each ). The borders of the bands are lines of the grid,
    so wavy curves made of edges of the mesh"""
    ny = sum( rows )
    u, v = np.meshgrid( np.linspace( 0, 1, nx + 1 ), np.linspace( 0, 1, ny + 1 ), indexing = "ij" )
    y = v + amplitude * np.sin( 2 * np.pi * u ) * np.sin( np.pi * v )        # ( the border of the square stays put )
    nodes = np.stack( [ u, y ], -1 ).reshape( -1, 2 )
    I = lambda i, j: i * ( ny + 1 ) + j
    tris, rws = [], []
    for i in range( nx ):
        for j in range( ny ):
            tris += [ [ I( i, j ), I( i + 1, j ), I( i + 1, j + 1 ) ], [ I( i, j ), I( i + 1, j + 1 ), I( i, j + 1 ) ] ]
            rws += [ j, j ]
    band = np.searchsorted( np.cumsum( rows ), np.array( rws ), side = "right" )
    return nodes, np.array( tris ), band


def tube( nr = 3, nt = 48, nz = 8, r0 = 0.6, r1 = 1.0, h = 1.2 ):
    """a hollow cylinder ( a thick tube, not convex ) cut into tetrahedra: a grid in `( r, theta, z )`, each hexahedron cut in 6"""
    r = np.linspace( r0, r1, nr + 1 )
    t = 2 * np.pi * np.arange( nt ) / nt
    z = np.linspace( 0, h, nz + 1 )
    R, T, Z = np.meshgrid( r, t, z, indexing = "ij" )
    nodes = np.stack( [ R * np.cos( T ), R * np.sin( T ), Z ], -1 ).reshape( -1, 3 )
    I = lambda i, j, k: ( i * nt + j % nt ) * ( nz + 1 ) + k
    tets = []
    for i in range( nr ):
        for j in range( nt ):
            for k in range( nz ):
                c = [ I( i + a, j + b, k + e ) for a in ( 0, 1 ) for b in ( 0, 1 ) for e in ( 0, 1 ) ]   # ( c[ 4 a + 2 b + e ] )
                # the 6 tetrahedra of a hexahedron around its diagonal c0 - c7
                for p in ( ( 1, 3 ), ( 3, 2 ), ( 2, 6 ), ( 6, 4 ), ( 4, 5 ), ( 5, 1 ) ):
                    tets.append( [ c[ 0 ], c[ p[ 0 ] ], c[ p[ 1 ] ], c[ 7 ] ] )
    return nodes, np.array( tets )


# -- drawing ----------------------------------------------------------------------------------------------------------------

def add_triangles( viz, nodes, triangles, values, levels = 32, vmax = None ):
    """the triangles in grays, by their mean value ( none where it is zero ): the paper's color for 0, the ink's for `vmax` ( dark on the light theme,
    light on the dark one )"""
    v = values.mean( -1 ) if values.ndim == 2 else values[ triangles ].mean( -1 )
    lev = np.minimum( ( v / ( vmax or v.max() ) * levels ).astype( int ), levels - 1 )
    lev[ v <= 0 ] = -1                                                      # ( zero: not drawn, the background shows )
    for l in np.unique( lev[ lev >= 0 ] ):
        u = 0.6 * ( l + 0.5 ) / levels
        viz.add_faces( nodes, triangles[ lev == l ], color = ( ( 1 - u, ) * 3, ( 0.12 + u, ) * 3 ) )


def quantize( name, title, target, n, under = None, **viz_kwargs ):
    """the Lloyd quantization of `target` with `n` diracs, drawn over `under( viz )`"""
    t0 = time.time()
    x = np.asarray( sdot.quantization( target, n, seed = 0 ) )
    print( f"{ name }: { round( time.time() - t0, 1 ) } s" )
    viz = sdot.Visualizer( title = title, face_opacity = 0.5 if under is not None else 1.0, **viz_kwargs )
    if under is not None:
        under( viz )
    viz.add_points( x, radius = 0.006 if x.shape[ 1 ] == 2 else 0.012, color = "#e8202a" if x.shape[ 1 ] == 2 else None )
    viz.write_html( OUT / f"mesh-{ name }.html" )


# -- the cases --------------------------------------------------------------------------------------------------------------

def contours( n = 2000 ):
    """2D: a gear, meshed by `Mesh.from_contours` ( density 1 inside the gear, 0 around ): a non-convex domain"""
    target = sdot.Mesh.from_contours( [ gear() ], mesh_size = 0.02 )
    nodes, sx, values = np.asarray( target.nodes ), np.asarray( target.simplices ), np.asarray( target.values )
    quantize( "contours", "Mesh.from_contours: a gear", target, n,
              under = lambda viz: add_triangles( viz, nodes, sx, values, levels = 4, vmax = 2 ) )


def p1( n = 2000 ):
    """2D: a continuous piecewise linear density, one value per node of a Delaunay triangulation"""
    nodes, tris = square_mesh()
    x, y = nodes.T
    values = 0.1 + np.exp( - 30 * ( ( x - 0.3 ) ** 2 + ( y - 0.6 ) ** 2 ) ) + 0.6 * ( np.sin( 6 * x + 3 * y ) > 0.5 ) * y
    target = sdot.Mesh( nodes, tris, values = values )
    quantize( "p1", "Mesh: a value per node", target, n, under = lambda viz: add_triangles( viz, nodes, tris, values ) )


def bands( n = 2000 ):
    """2D: five wavy bands, each with its own smooth density, alternately dark and light: a value per corner, so the density
    jumps on the borders of the bands, which are edges of the mesh"""
    nodes, tris, band = wavy_bands()
    tau = 2 * np.pi
    funcs = [
        lambda x, y: 0.15 + 0.10 * np.sin( tau * 2 * x ),
        lambda x, y: 1.00 + 0.70 * np.sin( tau * ( 1.5 * x + 2 * y ) ),
        lambda x, y: 0.30 + 0.25 * np.cos( tau * 3 * x ),
        lambda x, y: 0.90 + 0.60 * np.sin( tau * 2 * ( x - y ) ),
        lambda x, y: 0.20 + 0.15 * np.cos( tau * ( x + y ) ),
    ]
    P = nodes[ tris ]
    values = np.zeros( tris.shape )
    for b, f in enumerate( funcs ):
        values[ band == b ] = f( P[ band == b, :, 0 ], P[ band == b, :, 1 ] )
    target = sdot.Mesh( nodes, tris, values = values )
    quantize( "bands", "Mesh: bands, a value per corner", target, n, under = lambda viz: add_triangles( viz, nodes, tris, values ) )


def hole( n = 1500 ):
    """2D: a round hole, meshed by `Mesh.from_contours` ( 0 in the circle, 1 around ), times an affine density `1 + x`: the
    border of the hole is a polygon of 64 edges of the mesh"""
    t = 2 * np.pi * np.arange( 64 ) / 64
    circle = np.stack( [ 0.55 + 0.22 * np.cos( t ), 0.45 + 0.22 * np.sin( t ) ], -1 )
    mesh = sdot.Mesh.from_contours( [ circle ], box = ( ( 0, 0 ), ( 1, 1 ) ), mesh_size = 0.03, inside = 0, outside = 1 )
    nodes, tris = np.asarray( mesh.nodes ), np.asarray( mesh.simplices )
    values = np.asarray( mesh.values ) * ( 1 + nodes[ tris ][ ..., 0 ] )
    target = sdot.Mesh( nodes, tris, values = values )
    quantize( "hole", "Mesh: a round hole", target, n, under = lambda viz: add_triangles( viz, nodes, tris, values ) )


def tube_lloyd( n = 3000 ):
    """3D: a tube cut into tetrahedra, with a density growing with the height, and the diracs of its Lloyd quantization"""
    nodes, tets = tube()
    target = sdot.Mesh( nodes, tets, values = 0.2 + nodes[ :, 2 ] )
    quantize( "tube", "Mesh in 3D: a tube of tetrahedra", target, n, depth_colors = True )


CASES = {
    "contours": contours,
    "p1"      : p1,
    "bands"   : bands,
    "hole"    : hole,
    "tube"    : tube_lloyd,
}


if __name__ == "__main__":
    OUT.mkdir( parents = True, exist_ok = True )
    for name in sys.argv[ 1: ] or CASES:
        CASES[ name ]()
