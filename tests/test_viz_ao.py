"""Baked occlusion ( sdot ADR 0001 §4 ): the kernel, pages and stores that carry it, the `sdot.ao` service."""
from errand import test


def _box( lo, hi, base ):
    import numpy as np
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    P = np.array( [ [ x0, y0, z0 ], [ x1, y0, z0 ], [ x1, y1, z0 ], [ x0, y1, z0 ], [ x0, y0, z1 ], [ x1, y0, z1 ], [ x1, y1, z1 ], [ x0, y1, z1 ] ], float )
    F = [ [ 0, 3, 2, 1 ], [ 4, 5, 6, 7 ], [ 0, 1, 5, 4 ], [ 1, 2, 6, 5 ], [ 2, 3, 7, 6 ], [ 3, 0, 4, 7 ] ]
    return P, [ [ base + i for i in f ] for f in F ]


def _scene():
    """a unit cube resting on a slab"""
    import numpy as np
    P1, F1 = _box( ( -3, -3, -1 ), ( 3, 3, 0 ), 0 )
    P2, F2 = _box( ( 0, 0, 0 ), ( 1, 1, 1 ), 8 )
    return np.vstack( [ P1, P2 ] ), F1 + F2


if test( "a cube on a slab: the corners near the slab occluded, the open ones not" ):
    import numpy as np
    from sdot.viz.occlusion import ambient_occlusion
    pos, F = _scene()
    pv = np.array( [ i for f in F for i in f ] )
    po = np.r_[ 0, np.cumsum( [ len( f ) for f in F ] ) ]
    ao = ambient_occlusion( pos, pv, po, rays = 64, radius = 1.0 )
    side = ao[ po[ 11 ] : po[ 11 ] + 4 ]                                        # ( a side of the cube: bottom, bottom, top, top )
    assert np.all( ( side[ :2 ] > 0.4 ) & ( side[ :2 ] < 0.75 ) ) and np.all( side[ 2: ] > 0.99 ), side
    assert np.all( ao[ po[ 1 ] : po[ 2 ] ] > 0.99 ), "the slab's far corners: open"
    assert np.array_equal( ao, ambient_occlusion( pos, pv, po, rays = 64, radius = 1.0 ) ), "the same seed, the same values"


if test( "a page with ao = True carries a code per corner, a page without carries none" ):
    import base64
    import re
    import numpy as np
    from sdot.viz.Visualizer import Visualizer
    pos, F = _scene()
    pages, vs = { }, { }
    for ao in ( False, True ):
        v = vs[ ao ] = Visualizer( ao = ao )
        v.add_faces( pos, F )
        pages[ ao ] = v.to_html()
    got = lambda page: re.search( r'let POLY_AO = decU8\("([^"]*)"\);', page ).group( 1 )
    assert got( pages[ False ] ) == "" and got( pages[ True ] ).startswith( "@" ), "none, or a blob of the page"
    codes = vs[ True ]._baked_occlusion()
    assert len( codes ) == sum( len( f ) for f in F ) and codes.min() < 200 and codes.max() == 255
    assert 'id="bakeBox"' in pages[ True ]


if test( "a store with ao = True bakes each chunk; sdot.ao gives the same codes for a store without" ):
    import json
    import tempfile
    import threading
    import urllib.request
    from pathlib import Path
    from urllib.parse import quote
    import numpy as np
    from sdot.viz.Visualizer import Visualizer
    from sdot.viz.services import ao as service
    from sdot.viz.store import read_chunk, read_scene
    pos, F = _scene()
    paths = { }
    for ao in ( False, True ):
        paths[ ao ] = tempfile.mkdtemp() + "/s.sviz"
        v = Visualizer( store = paths[ ao ], ao = ao, frame_axis = "it" )
        for k in range( 2 ):
            if k:
                v.new_frame( k )
            v.add_faces( pos + [ 0, 0, 0.1 * k ], F )
        v.close()
    services = json.loads( Path( paths[ False ], "services.json" ).read_text() )
    from sdot.viz.services import version
    ao_entry = services[ "services" ][ "sdot.ao" ]                     # ( format 2, errand ADR 0027: names, no paths )
    assert services[ "format" ] == 2 and ao_entry[ "protocol" ] == 1 and ao_entry[ "version" ] == version( "sdot.ao" )
    assert ao_entry[ "described" ][ "run" ][ :3 ] == [ "{python}", "-m", "sdot.viz.services.ao" ]
    assert str( Path.home() ) not in Path( paths[ False ], "services.json" ).read_text(), "no path of the writer's"
    for i in range( 2 ):
        arr_ao, _ = read_scene( paths[ True ] ).frame( i )
        arr, _ = read_scene( paths[ False ] ).frame( i )
        assert "poly_ao" not in arr and len( arr_ao[ "poly_ao" ] ) == len( arr_ao[ "poly_v" ] )
        assert np.array_equal( service.frame_occlusion( paths[ False ], i ), arr_ao[ "poly_ao" ] )
    # over HTTP, the store said by the relay's header
    import socket
    s = socket.socket()
    s.bind( ( "127.0.0.1", 0 ) )
    port = s.getsockname()[ 1 ]
    s.close()
    threading.Thread( target = service.serve, args = ( port, ), daemon = True ).start()
    import time
    for _ in range( 100 ):
        try:
            urllib.request.urlopen( f"http://127.0.0.1:{ port }/health", timeout = 1 )
            break
        except OSError:
            time.sleep( 0.05 )
    req = urllib.request.Request( f"http://127.0.0.1:{ port }/frame/1", headers = { "X-Errand-Root": quote( paths[ False ] ) } )
    arrays, _ = read_chunk( urllib.request.urlopen( req, timeout = 60 ).read() )
    assert np.array_equal( arrays[ "poly_ao" ], read_scene( paths[ True ] ).frame( 1 )[ 0 ][ "poly_ao" ] )
    try:
        urllib.request.urlopen( f"http://127.0.0.1:{ port }/frame/0", timeout = 5 )
        assert False, "no store said"
    except urllib.error.HTTPError as e:
        assert e.code == 404
