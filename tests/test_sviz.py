from errand import test

import atexit as _atexit
import shutil as _shutil
import tempfile as _tempfile
_MADE = [ ]


def _tmp( ) -> str:
    """a directory of this file's, removed when it ends: a store left behind stays registered by errand as a view, and keeps
    the services' versions it asks for from being purged ( errand ADR 0028 )"""
    d = _tempfile.mkdtemp()
    _MADE.append( d )
    return d


_atexit.register( lambda: [ _shutil.rmtree( d, ignore_errors = True ) for d in _MADE ] )


def _store( nb = 6, **kw ):
    import numpy as np
    from sdot.viz.Visualizer import Visualizer
    path = _tmp() + "/s.sviz"
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
    from sdot.viz.sviz import _select, export_html
    path = _store( 10 )
    m = json.load( open( path + "/manifest.json" ) )
    assert _select( m ) == list( range( 10 ) )
    assert _select( m, every = 4 ) == [ 0, 4, 8 ]
    assert _select( m, frames = "2:5" ) == [ 2, 3, 4 ] and _select( m, frames = "-1" ) == [ 9 ]
    one = m[ "frames" ][ 0 ][ "size" ] * 4 / 3
    assert _select( m, max_mb = 3.5 * one / 2 ** 20 ) == [ 0, 5, 9 ]                    # ( one frame out of 5, the last kept )
    assert _select( m, max_mb = 2.5 * one / 2 ** 20 ) == [ 0, 9 ]
    out = _tmp() + "/p.html"
    export_html( path, out, every = 3 )
    page = open( out ).read()
    files = json.loads( re.search( r"const STORE_FILES = (\{.*\});", page ).group( 1 ) )
    mm = json.loads( files[ "manifest.json" ] )
    assert mm[ "complete" ] and [ f[ "v" ] for f in mm[ "frames" ] ] == [ [ 0 ], [ 3 ], [ 6 ], [ 9 ] ]
    assert set( files ) == { "manifest.json", m[ "base" ] } | { f[ "file" ] for f in mm[ "frames" ] }
    assert not re.search( r"__[A-Z0-9_]+__", page )

if test( "export --vtk: the series of the store, as the Visualizer writes it" ):
    import re
    import numpy as np
    from pathlib import Path
    from sdot.viz.Visualizer import Visualizer
    from sdot.viz.sviz import export_vtk
    out = export_vtk( _store( 4 ), _tmp() + "/s" )
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
    ref = v.write_vtk( _tmp() + "/s" )
    for k in range( 4 ):
        assert counts( out.parent / f"s_{ k:04d}.vtu" ) == counts( ref.parent / f"s_{ k:04d}.vtu" ), k

if test( "the server: the page and the files of a registered store, behind the token; it stops when idle" ):
    import threading
    import time
    import urllib.error
    import urllib.request
    from pathlib import Path
    import sdot.viz.sviz as sviz
    sviz.STATE = Path( _tmp() )                                           # ( not the user's )
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
    page = get( f"{ token }/{ sid }/?viewer=installed" ).decode()                   # ( the bootstrap: tested below )
    assert "storeOpen" in page and "decPos(STORE.POS)" in page
    assert json.loads( get( f"{ token }/{ sid }/manifest.json" ) )[ "complete" ]
    assert get( f"{ token }/{ sid }/frames/000002.bin" ) == ( Path( path ) / "frames/000002.bin" ).read_bytes()
    assert status( f"wrong/{ sid }/manifest.json" ) == 404
    assert status( f"{ token }/nobody/manifest.json" ) == 404
    assert status( f"{ token }/{ sid }/../../etc/passwd" ) == 404
    assert status( f"{ token }/{ sid }/%2e%2e/manifest.json" ) == 404
    th.join( 6 )
    assert not th.is_alive() and not ( sviz.STATE / "server.json" ).exists(), "an idle server stops"


if test( "the viewer is pinned: copied when unpublished, named only when published, packed, upgraded" ):
    import hashlib
    import json
    import os
    import numpy as np
    from pathlib import Path
    from sdot.viz import bundle
    from sdot.viz.Visualizer import Visualizer
    ident = bundle.viewer_id()
    assert ident[ "sha256" ] == hashlib.sha256( bundle.viewer_bytes() ).hexdigest() and ident[ "sha256" ][ :12 ] in ident[ "file" ]
    path = _store( 2 )
    pin = bundle.pinned( path )
    assert { k: pin[ k ] for k in ident } == ident and pin[ "urls" ] == bundle.URLS
    f = Path( path, "viewer", ident[ "file" ] )
    assert f.read_bytes() == bundle.viewer_bytes(), "unpublished ( a checkout ): the store holds its viewer"
    # named only ( published, or asked ), then packed from the installed viewer
    p2 = _tmp() + "/l.sviz"
    v = Visualizer( store = p2, viewer = "link" )
    v.add_points( np.random.rand( 4, 2 ) )
    v.close()
    assert not Path( p2, "viewer" ).exists()
    assert bundle.pack( p2 ).read_bytes() == bundle.viewer_bytes()
    # a store pinned to another viewer, which no URL serves: refused, then upgraded
    idx = Path( p2, "index.html" )
    idx.write_text( bundle.bootstrap_page( "old", dict( version = "0", sha256 = "0" * 64, file = "sdot-viz-0-000000000000.html" ), urls = [ ] ) )
    try:
        bundle.pack( p2 )
        assert False, "an unknown viewer cannot be packed"
    except RuntimeError as e:
        assert "--upgrade" in str( e )
    bundle.pack( p2, upgrade = True )
    assert bundle.pinned( p2 )[ "sha256" ] == ident[ "sha256" ]
    # published: listed, then only named
    out = _tmp()
    saved = bundle.PUBLISHED
    bundle.PUBLISHED = Path( out ) / "published.json"
    try:
        assert bundle.bundle( out, publish = True ).name == ident[ "file" ]
        assert ident[ "sha256" ] in bundle.published()
        p3 = _tmp() + "/p.sviz"
        v = Visualizer( store = p3 )
        v.add_points( np.random.rand( 4, 2 ) )
        v.close()
        assert not Path( p3, "viewer" ).exists(), "published: only named"
    finally:
        bundle.PUBLISHED = saved


if test( "sviz serve: the store's own bootstrap, the installed viewer on ?viewer=installed" ):
    import threading
    import urllib.request
    from sdot.viz import sviz
    from pathlib import Path
    from sdot.viz.bundle import TITLE, pinned
    sviz.STATE = Path( _tmp() )                                  # ( not the user's server )
    path = _store( 2 )
    vid = sviz.register( path )
    t = threading.Thread( target = sviz.serve, kwargs = dict( idle = 3 ), daemon = True )
    t.start()
    import time
    for _ in range( 50 ):
        s = sviz._server()
        if s:
            break
        time.sleep( 0.1 )
    base = f"http://127.0.0.1:{ s[ 'port' ] }/{ sviz._token() }/{ vid }/"
    boot = urllib.request.urlopen( base ).read().decode()
    assert 'id="sdot-viz"' in boot
    r = urllib.request.urlopen( base + "viewer/" + pinned( path )[ "file" ] )
    assert r.headers[ "Content-Type" ].startswith( "text/html" ) and TITLE in r.read().decode()
    inst = urllib.request.urlopen( base + "index.html?viewer=installed" ).read().decode()
    assert 'id="sdot-viz"' not in inst and "storeOpen" in inst
