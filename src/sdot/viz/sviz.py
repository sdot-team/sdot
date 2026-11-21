"""sviz: the viewer of scene stores ( `xyz.sviz`, see `store.py`; errand ADR 0017 ) as a command.

    sviz open xyz.sviz [--url]        the viewer's server ensured, the directory registered, the page opened in the
                                      browser ( --url: its URL printed, nothing opened )
    sviz export xyz.sviz --html out.html [--every k] [--frames a:b] [--max-mb N]
                                      ONE self-contained page ( the manifest and the chunks inlined ), to send to someone
    sviz export xyz.sviz --vtk out/name   the ParaView series ( `.vtu` per frame + `.pvd` )
    sviz pack xyz.sviz [--upgrade]    the viewer the store names, copied into `xyz.sviz/viewer/`: the directory then works
                                      offline, under any static server ( --upgrade: moved to the installed viewer first )
    sviz bundle DIR [--publish]       the installed viewer's bundle written into DIR ( the documentation's public/viewer );
                                      --publish: listed in `published.json`, the stores then only name it ( see `bundle.py` )
    sviz serve [--idle 10m]           the server, in the foreground
    sviz stop                         the server stopped

= The server

One per user, for every directory opened: `http://127.0.0.1:<port>/<token>/<id>/`. The token is in the path ( other
accounts of the machine reach 127.0.0.1 too ), kept in `~/.cache/sdot/sviz/token` so that a URL stays valid from one
start to the next, as the port does when it is free. Static files, Python stdlib only. The page is the store's own
`index.html`, a bootstrap that loads the viewer it is pinned to ( sdot ADR 0001, `bundle.py` ); `?viewer=installed` serves
the INSTALLED viewer instead ( `store_page` ), as for an older store, whose `index.html` is a whole viewer. The directories
opened are listed in `dirs.json`, read at each request: `sviz open` registers one without talking to the server.

Started detached by `sviz open` when it does not answer; it stops itself after `--idle` without a request ( an open
page asks for its manifest every minute: a tab kept open keeps it ).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

STATE = Path( os.environ.get( "XDG_CACHE_HOME", Path.home() / ".cache" ) ) / "sdot" / "sviz"
IDLE = 600.0


# -- state -----------------------------------------------------------------------------------------------------------------

def _write( path: Path, text: str ):
    path.parent.mkdir( parents = True, exist_ok = True )
    tmp = path.with_name( f".{ path.name }.{ os.getpid() }.tmp" )
    tmp.write_text( text )
    os.replace( tmp, path )


def _token() -> str:
    p = STATE / "token"
    if not p.exists():
        STATE.mkdir( parents = True, exist_ok = True )
        try:
            with os.fdopen( os.open( p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 ), "w" ) as f:   # ( readable by its user only )
                f.write( secrets.token_urlsafe( 18 ) )
        except FileExistsError:
            pass
    return p.read_text().strip()


def _dirs() -> dict:
    try:
        return json.loads( ( STATE / "dirs.json" ).read_text() )
    except ( OSError, ValueError ):
        return { }


def register( path ) -> str:
    """the id of a store directory in the URLs ( its name and a hash of its absolute path ), registered"""
    path = Path( path ).resolve()
    sid = f"{ path.stem }-{ hashlib.sha256( str( path ).encode() ).hexdigest()[ :8 ] }"
    dirs = _dirs()
    if dirs.get( sid ) != str( path ):
        dirs[ sid ] = str( path )
        _write( STATE / "dirs.json", json.dumps( dirs, indent = 1 ) )
    return sid


def _server() -> dict | None:
    """`{ port, pid }` of the running server, None when it does not answer"""
    try:
        info = json.loads( ( STATE / "server.json" ).read_text() )
    except ( OSError, ValueError ):
        return None
    import urllib.request
    try:
        with urllib.request.urlopen( f"http://127.0.0.1:{ info[ 'port' ] }/{ _token() }/ping", timeout = 1 ) as r:
            return info if r.read() == b"sviz" else None
    except OSError:
        return None


# -- the server ------------------------------------------------------------------------------------------------------------

def serve( port = 0, idle = IDLE ):
    """the server, in the foreground; returns after `idle` seconds without a request ( 0: never )"""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from .Visualizer import store_page
    from .bundle import pinned

    token, last = _token(), [ time.monotonic() ]

    class Handler( BaseHTTPRequestHandler ):
        def log_message( self, *args ):
            pass

        def _send( self, code, body = b"", ctype = "text/plain; charset=utf-8", head = False ):
            self.send_response( code )
            self.send_header( "Content-Type", ctype )
            self.send_header( "Content-Length", str( len( body ) ) )
            self.send_header( "Cache-Control", "no-store" )
            self.end_headers()
            if not head:
                self.wfile.write( body )

        def _route( self, head ):
            last[ 0 ] = time.monotonic()
            parts = self.path.split( "?" )[ 0 ].split( "/" )[ 1: ]      # token / id / file...
            if len( parts ) < 1 or not secrets.compare_digest( parts[ 0 ], token ):
                return self._send( 404, b"not found", head = head )
            if parts[ 1: ] == [ "ping" ]:
                return self._send( 200, b"sviz", head = head )
            if len( parts ) < 2 or parts[ 1 ] not in ( dirs := _dirs() ):
                return self._send( 404, b"unknown directory ( `sviz open` registers it )", head = head )
            root = Path( dirs[ parts[ 1 ] ] )
            if len( parts ) == 2:                                       # ( the files are read relative to the page )
                self.send_response( 301 )
                self.send_header( "Location", self.path.split( "?" )[ 0 ] + "/" )
                self.end_headers()
                return
            rel = "/".join( parts[ 2: ] ) or "index.html"
            from urllib.parse import parse_qs, urlsplit
            installed = parse_qs( urlsplit( self.path ).query ).get( "viewer" ) == [ "installed" ]
            if rel == "index.html" and ( installed or pinned( root ) is None ):     # ( an older store: its page is the whole viewer )
                try:
                    title = json.loads( ( root / "manifest.json" ).read_text() ).get( "title", root.stem )
                except ( OSError, ValueError ):
                    title = root.stem
                return self._send( 200, store_page( title ).encode(), "text/html; charset=utf-8", head )
            f = ( root / rel ).resolve()
            if root.resolve() not in f.parents or not f.is_file():
                return self._send( 404, b"not found", head = head )
            ctype = { ".json": "application/json", ".html": "text/html; charset=utf-8" }.get( f.suffix, "application/octet-stream" )
            return self._send( 200, f.read_bytes(), ctype, head )

        def do_GET( self ):
            self._route( False )

        def do_HEAD( self ):
            self._route( True )

    try:
        httpd = ThreadingHTTPServer( ( "127.0.0.1", port ), Handler )
    except OSError:
        httpd = ThreadingHTTPServer( ( "127.0.0.1", 0 ), Handler )     # ( the port of last time is taken )
    httpd.daemon_threads = True
    _write( STATE / "server.json", json.dumps( dict( port = httpd.server_address[ 1 ], pid = os.getpid() ) ) )
    if idle:
        def watch():
            while time.monotonic() - last[ 0 ] < idle:
                time.sleep( min( 5.0, idle / 4 ) )
            httpd.shutdown()
        threading.Thread( target = watch, daemon = True ).start()
    if threading.current_thread() is threading.main_thread():          # ( `sviz stop`: the state cleaned as on idle )
        import signal
        signal.signal( signal.SIGTERM, lambda *_: sys.exit( 0 ) )
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
        try:
            if json.loads( ( STATE / "server.json" ).read_text() ).get( "pid" ) == os.getpid():
                ( STATE / "server.json" ).unlink()
        except ( OSError, ValueError ):
            pass


def ensure( idle = IDLE ) -> int:
    """the port of the server, started detached when it does not answer"""
    info = _server()
    if info:
        return info[ "port" ]
    STATE.mkdir( parents = True, exist_ok = True )
    with _Lock( STATE / "start.lock" ):
        info = _server()
        if info:
            return info[ "port" ]
        try:
            port = json.loads( ( STATE / "last-port" ).read_text() )
        except ( OSError, ValueError ):
            port = 0
        log = open( STATE / "server.log", "ab" )
        kw = dict( start_new_session = True ) if os.name != "nt" else dict( creationflags = 0x00000008 | 0x00000200 )   # DETACHED, NEW_GROUP
        subprocess.Popen( [ sys.executable, "-m", "sdot.viz.sviz", "serve", "--port", str( port ), "--idle", str( idle ) ],
                          stdin = subprocess.DEVNULL, stdout = log, stderr = log, close_fds = True, **kw )
        for _ in range( 100 ):
            time.sleep( 0.1 )
            info = _server()
            if info:
                _write( STATE / "last-port", str( info[ "port" ] ) )
                return info[ "port" ]
    raise RuntimeError( f"sviz: the server did not start ( see { STATE / 'server.log' } )" )


class _Lock:
    """two `sviz open` at once start one server"""

    def __init__( self, path ):
        self.path = path

    def __enter__( self ):
        self.f = open( self.path, "a+" )
        try:
            import fcntl
            fcntl.flock( self.f, fcntl.LOCK_EX )
        except ImportError:                                             # ( Windows: msvcrt locks a byte )
            import msvcrt
            self.f.seek( 0 )
            for _ in range( 100 ):
                try:
                    msvcrt.locking( self.f.fileno(), msvcrt.LK_NBLCK, 1 )
                    break
                except OSError:
                    time.sleep( 0.1 )
        return self

    def __exit__( self, *exc ):
        self.f.close()


def stop() -> bool:
    info = _server()
    if not info:
        return False
    import signal
    os.kill( info[ "pid" ], signal.SIGTERM )
    return True


def url( path, idle = IDLE ) -> str:
    """the URL of a store directory, the server ensured"""
    path = Path( path )
    if not ( path / "manifest.json" ).is_file():
        raise FileNotFoundError( f"sviz: { path } is not a scene store ( no manifest.json )" )
    sid = register( path )
    return f"http://127.0.0.1:{ ensure( idle ) }/{ _token() }/{ sid }/"


# -- exports ---------------------------------------------------------------------------------------------------------------

def _select( m, every = 1, frames = None, max_mb = None, base_size = 0 ):
    """the indices of the frames exported; `base_size`: the bytes of the page without the frames"""
    idx = list( range( len( m[ "frames" ] ) ) )
    if frames:
        a, sep, b = frames.partition( ":" )
        idx = idx[ int( a ) if a else None : int( b ) if b else None ] if sep else [ idx[ int( a ) ] ]
    idx = idx[ :: max( 1, int( every ) ) ]
    if max_mb:
        size = lambda ii: base_size + 4 / 3 * sum( m[ "frames" ][ i ][ "size" ] for i in ii )     # ( base64 )
        pick = lambda k: idx[ ::k ] if not idx or ( len( idx ) - 1 ) % k == 0 else idx[ ::k ] + [ idx[ -1 ] ]   # ( the last frame always: the final state )
        k = 1
        while k < len( idx ) and size( pick( k ) ) > max_mb * 2 ** 20:
            k += 1
        idx = pick( k )
    return idx


def export_html( path, out, every = 1, frames = None, max_mb = None ) -> Path:
    """one self-contained page: the manifest of the frames chosen and their chunks, inlined"""
    from .Visualizer import store_page
    from .announce import write_page
    path = Path( path )
    m = json.loads( ( path / "manifest.json" ).read_text() )
    base = ( path / m[ "base" ] ).read_bytes() if m.get( "base" ) else b""
    page = len( store_page( m.get( "title", path.stem ), files = { } ).encode() )      # ( the viewer itself )
    idx = _select( m, every, frames, max_mb, len( base ) * 4 / 3 + page )
    m = dict( m, complete = True, frames = [ m[ "frames" ][ i ] for i in idx ] )
    files = { "manifest.json": json.dumps( m, separators = ( ",", ":" ) ) }
    if m.get( "base" ):
        files[ m[ "base" ] ] = base64.b64encode( base ).decode( "ascii" )
    for f in m[ "frames" ]:
        files[ f[ "file" ] ] = base64.b64encode( ( path / f[ "file" ] ).read_bytes() ).decode( "ascii" )
    return write_page( out, store_page( m.get( "title", path.stem ), files = files ) )


class _StoreScene:
    """a store seen as a `Visualizer` by `write_vtk`: `frame( i )` reads the base and frame i, `positions` is then their pool"""

    def __init__( self, path ):
        from .store import read_scene
        self.s = read_scene( path )
        m = self.s.manifest
        self.nb_dims, self.nb_frames = m[ "dim" ], len( self.s )
        self.colors = [ tuple( c ) for c in m[ "colors" ] ]
        self._base = self.s.base()
        self.positions = None

    def bounds( self, pad_ratio = 1.0 ):
        b = self.s.manifest[ "bounds" ] or [ [ -1.0, 1.0 ] ] * self.nb_dims
        pad = 0.05 * max( max( hi - lo, 1e-9 ) for lo, hi in b ) * pad_ratio
        return [ [ lo - pad, hi + pad ] for lo, hi in b ]

    def frame( self, i ):
        import numpy as np
        ( A, _ ), ( F, meta ) = self._base, self.s.frame( i )
        d, nv = self.nb_dims, len( A.get( "pos", ( ) ) )
        self.positions = np.concatenate( [ A.get( "pos", np.zeros( ( 0, d ), np.float32 ) ), F.get( "pos", np.zeros( ( 0, d ), np.float32 ) ) ] )
        cat = lambda k, shift = 0, empty = ( 0, ): np.concatenate( [ A[ k ] if k in A else np.zeros( empty ), ( F[ k ] + shift ) if k in F else np.zeros( empty ) ] )
        polys = [ ]
        for X, s in ( ( A, 0 ), ( F, nv ) ):
            if "poly_off" in X:
                o = X[ "poly_off" ]
                polys += [ X[ "poly_v" ][ o[ j ] : o[ j + 1 ] ] + s for j in range( len( o ) - 1 ) ]
        hp, at = [ ], 0
        for h in meta.get( "hpolys", [ ] ):
            hp.append( ( F[ "hp_dirs" ][ at : at + h[ "nb" ] ].reshape( -1, d ), F[ "hp_offs" ][ at : at + h[ "nb" ] ], tuple( h[ "fcol" ] ), h[ "edg" ] ) )
            at += h[ "nb" ]
        return dict( value = self.s.value( i ), polygons = polys, polygon_colors = cat( "poly_c" ).astype( int ),
                     edges = cat( "edge_v", nv, ( 0, 2 ) ).astype( int ), edge_colors = cat( "edge_c" ).astype( int ),
                     points = cat( "pnt_v", nv ).astype( int ), point_colors = cat( "pnt_c" ).astype( int ),
                     point_radii = cat( "pnt_r" ), point_matrices = cat( "pnt_m", 0, ( 0, 4 ) ), polytopes = hp )


def export_vtk( path, out, axes = ( 0, 1, 2 ) ) -> Path:
    from .vtk_writer import write_vtk
    return write_vtk( _StoreScene( path ), out, axes = axes, pvd = True )


# -- the command -----------------------------------------------------------------------------------------------------------

def _seconds( s ) -> float:
    s = str( s )
    return float( s[ :-1 ] ) * { "s": 1, "m": 60, "h": 3600 }[ s[ -1 ] ] if s[ -1 ] in "smh" else float( s )


def main( argv = None ):
    p = argparse.ArgumentParser( prog = "sviz", description = "the viewer of scene stores ( xyz.sviz )" )
    sub = p.add_subparsers( dest = "cmd", required = True )
    o = sub.add_parser( "open", help = "open a store in the browser ( the server started if needed )" )
    o.add_argument( "path" )
    o.add_argument( "--url", action = "store_true", help = "print the URL, open nothing" )
    o.add_argument( "--idle", default = "10m", help = "the server stops after this time without a request" )
    e = sub.add_parser( "export", help = "a self-contained page ( --html ) or a ParaView series ( --vtk )" )
    e.add_argument( "path" )
    g = e.add_mutually_exclusive_group( required = True )
    g.add_argument( "--html", metavar = "OUT.html" )
    g.add_argument( "--vtk", metavar = "OUT", help = "the stem: OUT_0000.vtu ..., OUT.pvd" )
    e.add_argument( "--every", type = int, default = 1, help = "one frame out of k" )
    e.add_argument( "--frames", help = "a:b, the frames a to b ( python slice )" )
    e.add_argument( "--max-mb", type = float, help = "fewer frames until the page is under N MB ( the last one kept )" )
    k = sub.add_parser( "pack", help = "copy the viewer the store names into it ( offline, any static server )" )
    k.add_argument( "path" )
    k.add_argument( "--upgrade", action = "store_true", help = "move the store to the installed viewer first" )
    b = sub.add_parser( "bundle", help = "write the installed viewer's bundle into a directory" )
    b.add_argument( "dir" )
    b.add_argument( "--publish", action = "store_true", help = "list it in published.json: stores then only name it" )
    s = sub.add_parser( "serve", help = "the server, in the foreground" )
    s.add_argument( "--port", type = int, default = 0 )
    s.add_argument( "--idle", default = "0", help = "stop after this time without a request ( 0: never )" )
    sub.add_parser( "stop", help = "stop the server" )
    a = p.parse_args( argv )

    if a.cmd == "open":
        u = url( a.path, _seconds( a.idle ) )
        print( u )
        if not a.url:
            import webbrowser
            webbrowser.open( u )
    elif a.cmd == "export" and a.html:
        export_html( a.path, a.html, a.every, a.frames, a.max_mb )
    elif a.cmd == "export":
        if a.every != 1 or a.frames or a.max_mb:
            p.error( "--every, --frames, --max-mb: with --html only" )
        print( export_vtk( a.path, a.vtk ) )
    elif a.cmd == "pack":
        from .bundle import pack
        print( pack( a.path, a.upgrade ) )
    elif a.cmd == "bundle":
        from .bundle import bundle
        print( bundle( a.dir, a.publish ) )
    elif a.cmd == "serve":
        serve( a.port, _seconds( a.idle ) )
    elif a.cmd == "stop":
        print( "stopped" if stop() else "no server" )


if __name__ == "__main__":
    main()
