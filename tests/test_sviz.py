from errand import test


def _store( nb = 6, **kw ):
    import tempfile
    import numpy as np
    from sdot.viz.Visualizer import Visualizer
    path = tempfile.mkdtemp() + "/s.sviz"
    v = Visualizer( frame_axis = "it", store = path, **kw )
    v.add_edges( np.array( [ [ 0, 0 ], [ 1, 0 ], [ 1, 1 ] ], float ), shared = True )
    for k in range( nb ):
        if k:
            v.new_frame( k )
        v.add_points( np.random.rand( 20 + k, 2 ), fields = { "a": np.arange( 20.0 + k ) } )
        v.add_faces( np.random.rand( 3, 2 ), [ [ 0, 1, 2 ] ] )
        v.add_polytope( np.array( [ [ 1, 0 ], [ -1, 0 ], [ 0, 1 ], [ 0, -1 ] ], float ), np.full( 4, 0.3 ) )
    v.close()
    return path


if test( "export --html: the manifest and the chunks inlined, the frames chosen, the last one kept" ):
    import json
    import re
    import tempfile
    from sdot.viz.sviz import _select, export_html
    path = _store( 10 )
    m = json.load( open( path + "/manifest.json" ) )
    assert _select( m ) == list( range( 10 ) )
    assert _select( m, every = 4 ) == [ 0, 4, 8 ]
    assert _select( m, frames = "2:5" ) == [ 2, 3, 4 ] and _select( m, frames = "-1" ) == [ 9 ]
    one = m[ "frames" ][ 0 ][ "size" ] * 4 / 3
    assert _select( m, max_mb = 3.5 * one / 2 ** 20 ) == [ 0, 5, 9 ]                    # ( one frame out of 5, the last kept )
    assert _select( m, max_mb = 2.5 * one / 2 ** 20 ) == [ 0, 9 ]
    out = tempfile.mkdtemp() + "/p.html"
    export_html( path, out, every = 3 )
    page = open( out ).read()
    files = json.loads( re.search( r"const STORE_FILES = (\{.*\});", page ).group( 1 ) )
    mm = json.loads( files[ "manifest.json" ] )
    assert mm[ "complete" ] and [ f[ "v" ] for f in mm[ "frames" ] ] == [ [ 0 ], [ 3 ], [ 6 ], [ 9 ] ]
    assert set( files ) == { "manifest.json", m[ "base" ] } | { f[ "file" ] for f in mm[ "frames" ] }
    assert not re.search( r"__[A-Z0-9_]+__", page )

if test( "export --vtk: the series of the store, as the Visualizer writes it" ):
    import re
    import tempfile
    import numpy as np
    from pathlib import Path
    from sdot.viz.Visualizer import Visualizer
    from sdot.viz.sviz import export_vtk
    out = export_vtk( _store( 4 ), tempfile.mkdtemp() + "/s" )
    assert out.suffix == ".pvd" and len( list( out.parent.glob( "s_*.vtu" ) ) ) == 4
    counts = lambda f: tuple( map( int, re.search( r'NumberOfPoints="(\d+)" NumberOfCells="(\d+)"', Path( f ).read_text( errors = "replace" ) ).groups() ) )
    # the same scene, as a page's Visualizer writes it
    v = Visualizer( frame_axis = "it" )
    v.add_edges( np.array( [ [ 0, 0 ], [ 1, 0 ], [ 1, 1 ] ], float ), shared = True )
    for k in range( 4 ):
        if k:
            v.new_frame( k )
        v.add_points( np.random.rand( 20 + k, 2 ) )
        v.add_faces( np.random.rand( 3, 2 ), [ [ 0, 1, 2 ] ] )
        v.add_polytope( np.array( [ [ 1, 0 ], [ -1, 0 ], [ 0, 1 ], [ 0, -1 ] ], float ), np.full( 4, 0.3 ) )
    ref = v.write_vtk( tempfile.mkdtemp() + "/s" )
    for k in range( 4 ):
        assert counts( out.parent / f"s_{ k:04d}.vtu" ) == counts( ref.parent / f"s_{ k:04d}.vtu" ), k

if test( "the server: the page and the files of a registered store, behind the token; it stops when idle" ):
    import tempfile
    import threading
    import time
    import urllib.error
    import urllib.request
    from pathlib import Path
    import sdot.viz.sviz as sviz
    sviz.STATE = Path( tempfile.mkdtemp() )                                           # ( not the user's )
    path = _store( 3 )
    sid = sviz.register( path )
    th = threading.Thread( target = sviz.serve, kwargs = dict( idle = 1.5 ), daemon = True )
    th.start()
    for _ in range( 50 ):
        if ( sviz.STATE / "server.json" ).exists():
            break
        time.sleep( 0.05 )
    import json
    port, token = json.loads( ( sviz.STATE / "server.json" ).read_text() )[ "port" ], sviz._token()
    get = lambda p: urllib.request.urlopen( f"http://127.0.0.1:{ port }/{ p }", timeout = 2 ).read()

    def status( p ):
        try:
            get( p )
            return 200
        except urllib.error.HTTPError as e:
            return e.code

    assert sviz._server()[ "port" ] == port
    page = get( f"{ token }/{ sid }/" ).decode()
    assert "storeOpen" in page and "decPos(STORE.POS)" in page
    assert json.loads( get( f"{ token }/{ sid }/manifest.json" ) )[ "complete" ]
    assert get( f"{ token }/{ sid }/frames/000002.bin" ) == ( Path( path ) / "frames/000002.bin" ).read_bytes()
    assert status( f"wrong/{ sid }/manifest.json" ) == 404
    assert status( f"{ token }/nobody/manifest.json" ) == 404
    assert status( f"{ token }/{ sid }/../../etc/passwd" ) == 404
    assert status( f"{ token }/{ sid }/%2e%2e/manifest.json" ) == 404
    th.join( 6 )
    assert not th.is_alive() and not ( sviz.STATE / "server.json" ).exists(), "an idle server stops"
