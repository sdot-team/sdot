"""The `blender` service ( sdot ADR 0001 §5 ): a store's frame as Blender sees it, and pictures over HTTP."""
from errand import test

BLENDER = "/Applications/Blender.app/Contents/MacOS/Blender"


def _store( frames = 2 ):
    import tempfile
    import numpy as np
    from sdot.viz.Visualizer import Visualizer
    path = tempfile.mkdtemp() + "/b.sviz"
    v = Visualizer( store = path, frame_axis = "it" )
    cube = np.array( [ [ 0, 0, 0 ], [ 1, 0, 0 ], [ 1, 1, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ], [ 1, 0, 1 ], [ 1, 1, 1 ], [ 0, 1, 1 ] ], float )
    F = [ [ 0, 3, 2, 1 ], [ 4, 5, 6, 7 ], [ 0, 1, 5, 4 ], [ 1, 2, 6, 5 ], [ 2, 3, 7, 6 ], [ 3, 0, 4, 7 ] ]
    v.add_faces( cube * 3 - [ 1, 1, 3 ], F, color = "#808080", shared = True )                      # ( a floor, in the base )
    for k in range( frames ):
        if k:
            v.new_frame( k )
        v.add_faces( cube + [ 0, 0, 0.2 * k ], F, color = "#ff0000" )
        v.add_edges( cube, [ [ 0, 1 ], [ 1, 2 ] ] )
        v.add_points( np.array( [ [ 0.5, 0.5, 1.5 ] ] ) )
    v.close()
    return path


if test( "a frame as Blender gets it: the base then the frame, colours in linear light" ):
    import numpy as np
    from sdot.viz.services.blender import frame_data
    path = _store()
    d = frame_data( path, 1 )
    assert len( d[ "pos" ] ) >= 16 and len( d[ "poly_off" ] ) == 13, ( len( d[ "pos" ] ), len( d[ "poly_off" ] ) )
    assert np.allclose( d[ "pos" ][ d[ "poly_v" ][ 24: ] ].min( axis = 0 ), [ 0, 0, 0.2 ] ), "the frame's cube, moved"
    red = d[ "face_col" ][ -1 ]
    assert np.allclose( red[ :3 ], [ 1, 0, 0 ] ) and abs( d[ "face_col" ][ 0 ][ 0 ] - 0.2158 ) < 1e-3, "#808080: 0.216 in linear"
    assert d[ "edge_v" ].shape == ( 2, 2 ) and len( d[ "pnt_v" ] ) == 1 and np.allclose( d[ "bg" ], 1 )
    assert np.allclose( frame_data( path, 0, axes = ( 2, 0, 1 ) )[ "pos" ][ :, 0 ], frame_data( path, 0 )[ "pos" ][ :, 2 ] )


if test( "under Blender: the service renders the view asked for, and keeps it in the store" ):
    import json
    import os
    import socket
    import subprocess
    import time
    import urllib.request
    from pathlib import Path
    from urllib.parse import quote
    if not os.path.exists( BLENDER ):
        print( "no Blender here: skipped" )
    else:
        import sdot.viz.services.blender as B
        path = _store()
        s = socket.socket()
        s.bind( ( "127.0.0.1", 0 ) )
        port = s.getsockname()[ 1 ]
        s.close()
        proc = subprocess.Popen( [ BLENDER, "-b", "--factory-startup", "-P", B.__file__, "--", "--port", str( port ) ],
                                 stdout = subprocess.DEVNULL, stderr = subprocess.PIPE )
        try:
            for _ in range( 300 ):
                try:
                    urllib.request.urlopen( f"http://127.0.0.1:{ port }/health", timeout = 1 )
                    break
                except OSError:
                    time.sleep( 0.1 )
            cam = dict( eye = [ 4, -5, 3 ], fwd = [ -0.55, 0.69, -0.47 ], right = [ 0.78, 0.62, 0 ], up = [ 0.29, -0.37, 0.88 ],
                        fov = 0.785, ortho = False, half_h = 2.0 )
            body = json.dumps( dict( frame = 1, camera = cam, width = 96, height = 64, samples = 4, keep = True ) ).encode()
            req = urllib.request.Request( f"http://127.0.0.1:{ port }/render", data = body, method = "POST",
                                          headers = { "X-Errand-Root": quote( path ), "Content-Type": "application/json" } )
            r = urllib.request.urlopen( req, timeout = 120 )
            png = r.read()
            assert png[ :8 ] == b"\x89PNG\r\n\x1a\n" and int.from_bytes( png[ 16:20 ], "big" ) == 96, "a PNG of the size asked"
            kept = r.headers[ "X-Kept" ]
            assert kept.startswith( "renders/frame-000001-" ) and Path( path, kept ).read_bytes() == png
        finally:
            proc.terminate()
            proc.wait( 10 )
