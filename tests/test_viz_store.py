from errand import test


def _scene( v_or_store ):
    """the same calls on a page's Visualizer and on a store: shared outline, an image, points with fields and ellipses,
    faces, edges, a polytope, over 4 frames whose number of points changes"""
    import numpy as np
    v = v_or_store
    rng = np.random.default_rng( 0 )
    v.add_layer( "fbp", visible = False )
    v.add_image( rng.random( ( 6, 5 ) ), ( -1, -1 ), ( 1, 1 ), layer = "fbp" )
    v.add_edges( np.array( [ [ 0, 0 ], [ 1, 0 ], [ 1, 1 ] ], float ), shared = True, layer = "truth" )
    for k in range( 4 ):
        if k:
            v.new_frame( 10 * k )
        n = 5 + 3 * k
        p = rng.random( ( n, 2 ) )
        v.add_points( p, radius = rng.random( n ), fields = { "a": rng.random( n ), "b": np.arange( n, dtype = float ) }, color = ( 0.2, 0.4, 0.6 ) )
        v.add_points( rng.random( ( 2, 2 ) ), matrices = rng.random( ( 2, 2, 2 ) ), layer = "ell" )
        v.add_faces( p, [ [ 0, 1, 2 ], [ 1, 2, 3, 4 ] ], opacity = 0.5 )
        v.add_edges( p[ :3 ], closed = True )
        if k == 2:
            v.add_points( np.zeros( ( 1, 2 ) ), shared = True )      # a shared primitive that comes late
        v.add_polytope( rng.random( ( 4, 2 ) ) - 0.5, rng.random( 4 ) )
    return v


def _flat( arrays, meta, colors ):
    """what a chunk draws, as coordinates and values ( no indices )"""
    import numpy as np
    pos = arrays.get( "pos", np.zeros( ( 0, 2 ) ) )
    out = { }
    if "pnt_v" in arrays:
        out.update( points = pos[ arrays[ "pnt_v" ] ], radii = arrays[ "pnt_r" ], mats = arrays[ "pnt_m" ],
                    pcol = np.array( [ colors[ c ] for c in arrays[ "pnt_c" ] ] ), play = arrays[ "pnt_l" ] )
        for k in arrays:
            if k.startswith( "field:" ):
                out[ k ] = arrays[ k ]
    if "edge_v" in arrays:
        out.update( edges = pos[ arrays[ "edge_v" ] ], ecol = np.array( [ colors[ c ] for c in arrays[ "edge_c" ] ] ), elay = arrays[ "edge_l" ] )
    if "poly_v" in arrays:
        o = arrays[ "poly_off" ]
        out.update( polys = [ pos[ arrays[ "poly_v" ][ o[ i ] : o[ i + 1 ] ] ] for i in range( len( o ) - 1 ) ],
                    qcol = np.array( [ colors[ c ] for c in arrays[ "poly_c" ] ] ) )
    if "hp_dirs" in arrays:
        out.update( hp_dirs = arrays[ "hp_dirs" ], hp_offs = arrays[ "hp_offs" ] )
    return out


def _same( a, b, atol ):
    import numpy as np
    assert set( a ) == set( b ), ( sorted( a ), sorted( b ) )
    for k in a:
        if k == "polys":
            assert len( a[ k ] ) == len( b[ k ] )
            for x, y in zip( a[ k ], b[ k ] ):
                assert np.allclose( x, y, atol = atol, equal_nan = True ), k
        else:
            assert np.shape( a[ k ] ) == np.shape( b[ k ] ), ( k, np.shape( a[ k ] ), np.shape( b[ k ] ) )
            assert np.allclose( a[ k ], b[ k ], atol = atol, equal_nan = True ), k


def _check( compact, atol ):
    import tempfile
    import numpy as np
    from sdot.viz.Visualizer import Visualizer
    from sdot.viz.store import read_scene, scene_arrays
    page = _scene( Visualizer( frame_axis = "iteration", compact = compact ) )
    path = tempfile.mkdtemp() + "/s.sviz"
    _scene( Visualizer( frame_axis = "iteration", compact = compact, store = path ) ).close()
    s = read_scene( path )
    m = s.manifest
    assert m[ "complete" ] and len( s ) == page.nb_frames == 4
    assert [ s.value( i ) for i in range( 4 ) ] == [ 0, 10, 20, 30 ]
    colors = [ tuple( c ) for c in m[ "colors" ] ]
    assert colors == page.colors
    assert m[ "layers" ] == page._layers
    base, bmeta = s.base()
    # the image: in the base, with what the page has
    assert len( bmeta[ "images" ] ) == 1 and bmeta[ "images" ][ 0 ][ "layer" ] == page._images[ 0 ][ "layer" ]
    import base64
    assert np.array_equal( base[ "image:0" ].ravel(), np.frombuffer( base64.b64decode( page._images[ 0 ][ "data" ] ), np.uint8 ) )
    pal = [ tuple( c ) for c in page.colors ]
    for i in range( 4 ):
        want = _flat( *scene_arrays( page, i )[ :2 ], pal )
        fr, meta = s.frame( i )
        got = _flat( fr, meta, colors )
        # the page draws the shared primitives in every frame, even those added late; the store draws its LAST base
        # under each frame, which is the same
        bf = _flat( base, bmeta, colors )
        for k in ( "points", "radii", "mats", "pcol", "play", "edges", "ecol", "elay" ):
            if k in bf:
                got[ k ] = np.concatenate( [ bf[ k ], got[ k ] ] ) if k in got else bf[ k ]
        for k in [ k for k in got if k.startswith( "field:" ) ]:
            got[ k ] = np.concatenate( [ np.full( len( bf.get( "points", [ ] ) ), np.nan, np.float32 ), got[ k ] ] )
        _same( want, got, atol )
        assert set( m[ "frames" ][ i ][ "fields" ] ) == { "a", "b" }


if test( "a store gives back the frames of the page ( float32 )" ):
    _check( compact = False, atol = 1e-6 )

if test( "a store gives back the frames of the page ( compact: 16 bits )" ):
    _check( compact = True, atol = 2e-4 )

if test( "a store keeps only the frame being built, and tells after each manifest" ):
    import tempfile
    import numpy as np
    from sdot.viz.Visualizer import Visualizer
    from sdot.viz.store import read_scene
    told = [ ]
    path = tempfile.mkdtemp() + "/s.sviz"
    v = Visualizer( store = path, on_frame = lambda: told.append( 1 ) )
    for k in range( 50 ):
        if k:
            v.new_frame()
        v.add_points( np.random.rand( 100, 2 ) )
        v.clear_curves().add_curve( "loss", np.arange( k + 1.0 ) )
        assert len( v._pools ) == 1 and v.nb_frames == k + 1
    v.close()
    s = read_scene( path )
    assert len( s ) == 50 and s.manifest[ "complete" ] and len( s.manifest[ "curves" ] ) == 1
    assert len( s.manifest[ "curves" ][ 0 ][ "y" ] ) == 50
    assert told and s.manifest[ "bounds" ] is not None
    try:
        v.to_html()
        assert False, "a store is not a page"
    except ValueError:
        pass

if test( "a store holds its page: the template, every placeholder read from STORE" ):
    import re
    import tempfile
    import numpy as np
    from sdot.viz.Visualizer import Visualizer
    path = tempfile.mkdtemp() + "/s.sviz"
    v = Visualizer( title = "a <b>", store = path )
    v.add_points( np.random.rand( 10, 2 ) )
    v.close()
    boot = open( path + "/index.html" ).read()
    from sdot.viz.bundle import TITLE, pinned
    pin = pinned( path )
    assert pin[ "title" ] == "a <b>" and "<title>a &lt;b&gt;</title>" in boot, "a bootstrap naming its viewer"
    page = open( path + "/viewer/" + pin[ "file" ] ).read()                  # ( a checkout's viewer is not published: copied )
    assert page.count( TITLE ) == 2, "the tab and the bar"
    page = page.replace( TITLE, "x" )
    assert not re.search( r"__[A-Z0-9_]+__", page ), "a placeholder left"
    assert "if (true) window.STORE = await storeOpen();" in page
    assert "decPos(STORE.POS)" in page and "const D      = STORE.D;" in page
    # the self-contained page: same template, its data inlined, no store
    w = Visualizer()
    w.add_points( np.random.rand( 10, 2 ) )
    page = w.to_html()
    assert not re.search( r"__[A-Z0-9_]+__", page ) and "if (false) window.STORE" in page

if test( "under errand, a store says itself a directory view at each manifest, the manifest last" ):
    import json
    import os
    import tempfile
    import numpy as np
    from errand import steps
    from sdot.viz.Visualizer import Visualizer
    d = tempfile.mkdtemp()
    journal = os.path.join( d, steps.FILE )
    was = os.environ.get( steps.ENV )
    os.environ[ steps.ENV ] = journal
    try:
        v = Visualizer( store = d + "/s.sviz" )
        v.add_points( np.random.rand( 5, 2 ) )
        v.close()
    finally:
        os.environ.pop( steps.ENV, None ) if was is None else os.environ.__setitem__( steps.ENV, was )
    views = [ e for e in map( json.loads, open( journal ).read().splitlines() ) if "view" in e ]
    assert views and all( e[ "view" ].endswith( "s.sviz" ) and e[ "last" ] == [ "manifest.json" ] for e in views ), views
    assert all( e[ "open" ] == [ { "static": "index.html" } ] for e in views ), "opened as it is: its own page"
