"""Sources in a store and the `sdot.region` service ( sdot ADR 0001 §6 )."""
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


def _store( n = 300, seed = 1, box = True ):
    import numpy as np
    import sdot
    from sdot.viz.Visualizer import Visualizer
    path = _tmp() + "/r.sviz"
    x = np.random.default_rng( seed ).random( ( n, 3 ) )
    w = 0.001 * np.random.default_rng( seed + 1 ).random( n )
    v = Visualizer( store = path, frame_axis = "it" )
    for k in range( 2 ):
        if k:
            v.new_frame( k )
        bnd = sdot.box_half_spaces( np.zeros( 3 ), np.ones( 3 ) ) if box else None
        sdot.PowerDiagram( x + 0.01 * k, w, bnd ).add_to_viz( v, seeds = k == 0 )
    v.close()
    return path, x, w


if test( "a store keeps each frame's seeds, weights and domain; a page keeps none" ):
    import numpy as np
    import sdot
    from sdot.viz.Visualizer import Visualizer
    from sdot.viz.store import read_scene
    path, x, w = _store()
    sc = read_scene( path )
    for k in range( 2 ):
        arrays, meta = sc.sources( k )
        assert meta[ "kind" ] == "sdot.power_diagram" and meta[ "color0" ] == 0
        assert np.allclose( arrays[ "seeds" ], x + 0.01 * k ) and np.allclose( arrays[ "weights" ], w )
        assert arrays[ "bnd_dirs" ].shape == ( 6, 3 ) and np.allclose( np.sort( arrays[ "bnd_offs" ] ), [ 0, 0, 0, 1, 1, 1 ] )
    v = Visualizer()
    sdot.PowerDiagram( x, w ).add_to_viz( v )
    assert v._sources is None, "no store: nothing kept"


if test( "sdot.region: every cell found again; a cut, the boundary, a box choose among them" ):
    import numpy as np
    from sdot.viz.services.region import cells_of
    from sdot.viz.store import read_chunk
    path, x, w = _store()
    c = cells_of( path, 0 )
    assert set( c.cell.tolist() ) == set( range( len( x ) ) ), "each cell's faces told their seed"
    half, _ = read_chunk( c.region( "half", plane = [ 1, 0, 0, 0.5 ] ) )
    _, mh = read_chunk( c.region( "half", plane = [ 1, 0, 0, 0.5 ] ) )
    assert mh[ "cells" ] == len( { int( i ) for i in c.cell if x[ i, 0 ] <= 0.5 } ) or mh[ "cells" ] >= ( x[ :, 0 ] <= 0.5 ).sum()
    assert half[ "pos" ][ :, 0 ].min() >= -1e-6 and half[ "poly_v" ].max() < len( half[ "pos" ] )
    assert half[ "edge_v" ].max() < len( half[ "pos" ] ) and len( mh[ "colors" ] ) == len( mh[ "colors_dark" ] ) > int( half[ "poly_c" ].max() )
    _, mb = read_chunk( c.region( "boundary" ) )
    inner = np.all( ( x > 0.2 ) & ( x < 0.8 ), axis = 1 ).sum()
    assert 0 < mb[ "cells" ] < len( x ) - inner // 2, mb[ "cells" ]
    box, mx = read_chunk( c.region( "box", lo = [ 0.3 ] * 3, hi = [ 0.7 ] * 3 ) )
    assert mx[ "cells" ] >= np.all( ( x >= 0.3 ) & ( x <= 0.7 ), axis = 1 ).sum() > 0
    assert cells_of( path, 0 ) is c, "kept per sources file"


if test( "sdot.region over HTTP: the store said by the relay's header; a frame without sources refused" ):
    import socket
    import threading
    import time
    import urllib.error
    import urllib.request
    from urllib.parse import quote
    import numpy as np
    from sdot.viz.Visualizer import Visualizer
    from sdot.viz.services import region
    from sdot.viz.store import read_chunk
    path, x, w = _store( 120 )
    plain = _tmp() + "/p.sviz"
    v = Visualizer( store = plain )
    v.add_points( np.random.rand( 5, 3 ) )
    v.close()
    s = socket.socket()
    s.bind( ( "127.0.0.1", 0 ) )
    port = s.getsockname()[ 1 ]
    s.close()
    threading.Thread( target = region.serve, args = ( port, ), daemon = True ).start()
    for _ in range( 100 ):
        try:
            urllib.request.urlopen( f"http://127.0.0.1:{ port }/health", timeout = 1 )
            break
        except OSError:
            time.sleep( 0.05 )
    get = lambda store, q: urllib.request.urlopen( urllib.request.Request(
        f"http://127.0.0.1:{ port }/frame/1?{ q }", headers = { "X-Errand-Root": quote( store ) } ), timeout = 60 ).read()
    arrays, meta = read_chunk( get( path, "mode=half&plane=0,0,1,0.5" ) )
    assert 0 < meta[ "cells" ] < len( x ) and arrays[ "pos" ].min() >= -1e-6 and arrays[ "pos" ].max() <= 1 + 1e-6, "in the box"
    for store, q in ( ( plain, "mode=boundary" ), ( path, "mode=nope" ), ( path, "mode=half&plane=1,2" ) ):
        try:
            get( store, q )
            assert False, ( store, q )
        except urllib.error.HTTPError as e:
            assert e.code == 400, e.code
