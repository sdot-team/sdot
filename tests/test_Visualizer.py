from errand import test


if test( "shared primitives are in every frame, stored once" ):
    import numpy as np
    from sdot.viz.Visualizer import Visualizer
    square = np.array( [ [ 0, 0 ], [ 1, 0 ], [ 1, 1 ], [ 0, 1 ] ], float )
    v = Visualizer()
    v.add_points( square[ :2 ] )
    v.add_edges( square, closed = True, shared = True )                      # ( added in the first frame )
    v.new_frame()
    v.add_points( square )
    v.add_faces( square, [ [ 0, 1, 2, 3 ] ], shared = True )               # ( added in the second one )
    f0, f1 = v.frame( 0 ), v.frame( 1 )
    assert len( f0[ "points" ] ) == 2 and len( f1[ "points" ] ) == 4
    assert len( f0[ "edges" ] ) == len( f1[ "edges" ] ) == 4 and len( v.edges ) == 4
    assert len( f0[ "polygons" ] ) == len( f1[ "polygons" ] ) == 1 and len( v.polygons ) == 1
    assert "SHARED" in v.to_html()

if test( "ellipses: a matrix per point, their radius the largest half-axis" ):
    import numpy as np
    from sdot.viz.Visualizer import Visualizer
    M = np.array( [ [ [ 0.3, 0.0 ], [ 0.0, 0.1 ] ], [ [ 0.1, 0.0 ], [ 0.2, 0.1 ] ] ] )
    v = Visualizer( flat_points = True ).add_points( np.zeros( ( 2, 2 ) ), matrices = M )
    v.add_points( np.ones( ( 1, 2 ) ), radius = 0.5 )
    fr = v.frame( 0 )
    assert np.allclose( fr[ "point_matrices" ], [ M[ 0 ].ravel(), M[ 1 ].ravel(), [ 0, 0, 0, 0 ] ] )
    assert np.allclose( fr[ "point_radii" ], [ 0.3, np.linalg.norm( M[ 1 ], 2 ), 0.5 ] )
    v.to_html()

if test( "fields per point, layers, images: in the page, aligned with the points" ):
    import base64, json, re, zlib
    import numpy as np
    from sdot.viz.Visualizer import Visualizer
    v = Visualizer()
    v.add_layer( "under", visible = False )
    v.add_image( np.arange( 12.0 ).reshape( 3, 4 ), ( -1, -1 ), ( 1, 1 ), layer = "under" )
    v.add_edges( np.eye( 2 ), shared = True, layer = "outline" )
    v.add_points( np.zeros( ( 2, 2 ) ), fields = { "a": [ 1, 2 ] } )
    v.new_frame()
    v.add_points( np.ones( ( 3, 2 ) ), fields = { "b": [ 3, 4, 5 ] } )
    html = v.to_html()
    blobs = json.loads( re.search( r"const RAW = (\[.*?\]);\n", html ).group( 1 ) )
    def blob( ref, dtype ):                                         # ( "@i": inflated, its bytes shuffled back )
        e = blobs[ int( ref[ 1: ] ) ]
        b = np.frombuffer( zlib.decompress( base64.b64decode( e[ "z" ] ) ), np.uint8 )
        return b.reshape( e[ "k" ], -1 ).T.copy().view( dtype ).ravel()
    fields = json.loads( re.search( r"const FIELDS = (\[.*?\])\.map", html ).group( 1 ) )
    data = { f[ "name" ]: blob( f[ "data" ], np.float32 ) for f in fields }
    assert np.array_equal( data[ "a" ][ :2 ], [ 1, 2 ] ) and np.isnan( data[ "a" ][ 2: ] ).all()
    assert np.isnan( data[ "b" ][ :2 ] ).all() and np.array_equal( data[ "b" ][ 2: ], [ 3, 4, 5 ] )
    assert json.loads( re.search( r"const LAYERS = (.*?), layerOn", html ).group( 1 ) ) == [ [ "under", False ], [ "outline", True ] ]
    img = json.loads( re.search( r"const IMAGES = (\[.*?\])\.map", html ).group( 1 ) )[ 0 ]
    assert ( img[ "nx" ], img[ "ny" ], img[ "layer" ] ) == ( 3, 4, 1 ) and len( base64.b64decode( img[ "data" ] ) ) == 12


if test( "compact: positions and fields on 16 bits, compressed; the page smaller" ):
    import numpy as np
    from sdot.viz.Visualizer import Visualizer, _quantized
    rng = np.random.default_rng( 0 )
    pages = { }
    for compact in ( False, True ):
        v = Visualizer( compact = compact )
        for _ in range( 5 ):
            v.add_points( rng.uniform( -1, 1, ( 2000, 2 ) ), fields = { "f": rng.normal( size = 2000 ) } )
            v.new_frame()
        pages[ compact ] = len( v.to_html() )
    base = len( Visualizer().add_points( np.zeros( ( 1, 2 ) ) ).to_html() )      # ( the code of the page )
    assert pages[ True ] - base < 0.6 * ( pages[ False ] - base ), ( pages, base )
    a = np.array( [ [ 0.0, 1.0 ], [ 0.5, np.nan ], [ 1.0, 3.0 ] ] )
    q, lo, step = _quantized( a, axis = 0 )
    assert q[ 1, 1 ] == 65535 and np.allclose( lo + step * q[ [ 0, 2 ] ], a[ [ 0, 2 ] ], atol = 1e-4 )
