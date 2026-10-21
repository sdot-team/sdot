"""The pages of `mesh.md`: interactive pages of the HTML viewer ( `sdot.Visualizer`, `docs/public/guide/mesh-*.html` ).

    python docs/guide/distributions/mesh_figures.py [ contours p1 bands hole tube fluid ]

`contours` and `hole` need `pygmsh` ( `Mesh.from_contours` ). `fluid` ( the card of the gallery: a fluid at rest in a knotted
tube ) takes hours: on lmo, `exp_doc_figures_lmo.py fluid`.
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


def trefoil_curve( t ):
    return np.stack( [ np.sin( t ) + 2 * np.sin( 2 * t ), np.cos( t ) - 2 * np.cos( 2 * t ), - np.sin( 3 * t ) ], -1 )


def trefoil_tube( nt = 192, ns = 8, r = 0.5, every = 4 ):
    """a tube of radius `r` around a trefoil knot ( `ns` sides, `nt` sections ) cut into tetrahedra: each slab between two sections
    is a fan of `ns` triangular prisms about the center line, 3 tetrahedra each. `( nodes, tets, wire )`, `wire` the edges of a
    wireframe of its border: the `ns` lines along the tube and a ring every `every` sections. The sections follow a frame that
    minimizes the rotation, its twist over the loop spread along"""
    t = 2 * np.pi * np.arange( nt ) / nt
    c = trefoil_curve( t )
    T = np.roll( c, -1, 0 ) - np.roll( c, 1, 0 )
    T /= np.linalg.norm( T, axis = 1 )[ :, None ]
    U = np.zeros_like( c )
    u = np.cross( T[ 0 ], [ 0, 0, 1 ] )
    U[ 0 ] = u / np.linalg.norm( u )
    for i in range( 1, nt + 1 ):                        # ( the last one: the first section seen from the end of the loop )
        u = U[ i - 1 ] - T[ i % nt ] * ( U[ i - 1 ] @ T[ i % nt ] )
        u /= np.linalg.norm( u )
        if i < nt:
            U[ i ] = u
    V0 = np.cross( T[ 0 ], U[ 0 ] )
    phi = np.arctan2( u @ V0, u @ U[ 0 ] )
    V = np.cross( T, U )
    a = - phi * np.arange( nt ) / nt
    U, V = np.cos( a )[ :, None ] * U + np.sin( a )[ :, None ] * V, - np.sin( a )[ :, None ] * U + np.cos( a )[ :, None ] * V
    th = 2 * np.pi * np.arange( ns ) / ns
    ring = c[ :, None ] + r * ( np.cos( th )[ None, :, None ] * U[ :, None ] + np.sin( th )[ None, :, None ] * V[ :, None ] )
    nodes = np.concatenate( [ c, ring.reshape( -1, 3 ) ] )
    C = lambda i: i % nt
    R = lambda i, j: nt + ( i % nt ) * ns + j % ns
    tets = []
    for i in range( nt ):
        for j in range( ns ):
            a0, a1, a2, b0, b1, b2 = C( i ), R( i, j ), R( i, j + 1 ), C( i + 1 ), R( i + 1, j ), R( i + 1, j + 1 )
            tets += [ [ a0, a1, a2, b2 ], [ a0, a1, b1, b2 ], [ a0, b0, b1, b2 ] ]
    wire = [ [ R( i, j ), R( i + 1, j ) ] for i in range( nt ) for j in range( ns ) ]
    wire += [ [ R( i, j ), R( i, j + 1 ) ] for i in range( 0, nt, every ) for j in range( ns ) ]
    return nodes, np.array( tets ), np.array( wire )


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


def fluid( a = 0.2, eps = 0.05, sigma = 0.02, damping = 2.0, duration = 6.0, nb_frames = 12, r = 0.5 ):
    """a fluid filling half of a tube knotted in a trefoil ( `Mesh( degree = 0 )`: partial transport in 3D ), released from the upper
    half of the tube and damped until it rests in the three low loops: the free-surface scheme of `free-surface-fluid.md`. The page
    draws the matter and the wireframe of the tube"""
    sys.path.insert( 0, str( Path( __file__ ).resolve().parents[ 2 ] / "examples" ) )
    from free_surface_fluid import Fluid
    from scipy.spatial import cKDTree

    nodes, tets, wire = trefoil_tube( r = r )
    P = nodes[ tets ]
    vol = float( np.abs( np.linalg.det( P[ :, 1: ] - P[ :, :1 ] ) ).sum() / 6 )
    target = sdot.Mesh( nodes, tets, target_mass = vol, degree = 0 )

    # the points of a cubic lattice inside the tube, the highest ones holding half of its volume
    g = np.arange( -3.8, 3.8, a ) + 0.5 * a
    p = np.stack( np.meshgrid( g, g, np.arange( -1.6, 1.6, a ) + 0.5 * a, indexing = "ij" ), -1 ).reshape( -1, 3 )
    d, _ = cKDTree( trefoil_curve( np.linspace( 0, 2 * np.pi, 8000, endpoint = False ) ) ).query( p )
    p = p[ d < r * np.cos( np.pi / 8 ) - 0.3 * a ]
    nb = int( round( 0.5 * vol / a ** 3 ) )
    p = p[ np.argsort( - p[ :, 2 ] )[ :nb ] ]
    p += 1e-3 * a * np.random.default_rng( 0 ).uniform( -1, 1, p.shape )
    masses = np.full( len( p ), 0.5 * vol / len( p ) )
    print( f"fluid in a trefoil: { len( tets ) } tetrahedra, { len( p ) } particles", flush = True )

    fluid = Fluid( target, masses, eps, gravity = ( 0.0, 0.0, -1.0 ), sigma = sigma )
    fluid.start_weights = np.full( len( p ), 3 * a * a )   # ( balls that cover the cells of the lattice: none empty )
    frames = fluid.run( p, dt = eps / 10, duration = duration, nb_frames = nb_frames, damping = damping )

    viz = sdot.Visualizer( title = "A fluid at rest in a knotted tube", frame_axis = "time", frame_value = 0.0 )
    for k, ( t, q, w, _ ) in enumerate( frames ):
        if k:
            viz.new_frame( round( t, 3 ) )
        viz.add_edges( nodes, wire, color = "#888888" )         # ( a frame is a whole state; `nodes`, one array: one pool )
        fluid.plan( q, weights0 = w ).add_to_viz( viz, seeds = False, opacity_beyond_balls = 0, ball_resolution = 10,
                                                  support_only = False, domain = False )
    viz.write_html( OUT / "mesh-fluid.html" )
    np.savez( OUT / "mesh-fluid.npz", **{ f"p{ k }": f[ 1 ] for k, f in enumerate( frames ) }, **{ f"w{ k }": f[ 2 ] for k, f in enumerate( frames ) },
              t = np.array( [ f[ 0 ] for f in frames ] ), energy = np.array( [ f[ 3 ] for f in frames ] ) )
    return frames


CASES = {
    "contours": contours,
    "p1"      : p1,
    "bands"   : bands,
    "hole"    : hole,
    "tube"    : tube_lloyd,
    "fluid"   : fluid,
}


if __name__ == "__main__":
    OUT.mkdir( parents = True, exist_ok = True )
    for name in sys.argv[ 1: ] or CASES:
        CASES[ name ]()
