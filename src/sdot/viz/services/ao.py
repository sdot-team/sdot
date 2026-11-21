"""sdot.ao: the baked occlusion of the frames of a store, on demand ( sdot ADR 0001 §4 ).

    python -m sdot.viz.services.ao --port P [--root DIR]

    GET /frame/<i>[?rays=&radius=]   a chunk ( `store.py` ) holding `poly_ao`: a code per corner ( uint8, 255 = open ) of
                                     the base's polygons then the frame's, as the page combines them
    GET /health                      `ok`

The store is `X-Errand-Root` ( percent-encoded, added by errand's relay ), else `--root`. Answers are kept in memory by
( store, frame file, base file, options ): a frame file is written once, so its occlusion never changes. CORS is answered
for `127.0.0.1` / `localhost` pages, for a service started by hand and given to a page by `?svc.sdot.ao=`.
"""
from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np

KEEP = 64


def frame_occlusion( store, i: int, **opts ) -> np.ndarray:
    """the codes of frame `i` of `store`: its polygons and the base's, baked together ( the base's first )"""
    from ..occlusion import ambient_occlusion, as_codes
    from ..store import read_scene
    sc = read_scene( store )
    ( ba, _ ), ( fa, _ ) = sc.base(), sc.frame( i )
    nb = len( ba[ "pos" ] ) if "pos" in ba else 0
    pos = np.concatenate( [ x[ "pos" ].reshape( -1, 3 ) for x in ( ba, fa ) if "pos" in x ] ) if nb or "pos" in fa else np.zeros( ( 0, 3 ) )
    pv, off = [ ], [ np.zeros( 1, np.int64 ) ]
    for x, shift in ( ( ba, 0 ), ( fa, nb ) ):
        if "poly_v" in x:
            pv.append( x[ "poly_v" ].astype( np.int64 ) + shift )
            off.append( x[ "poly_off" ][ 1: ].astype( np.int64 ) + ( off[ -1 ][ -1 ] ) )
    if not pv:
        return np.zeros( 0, np.uint8 )
    return as_codes( ambient_occlusion( pos, np.concatenate( pv ), np.concatenate( off ), **opts ) )


def serve( port: int, root: str | None = None ):
    kept = OrderedDict()

    class H( BaseHTTPRequestHandler ):
        def log_message( self, *a ):
            pass

        def _cors( self ):
            o = self.headers.get( "Origin" ) or ""
            host = urlsplit( o ).hostname
            if host in ( "127.0.0.1", "localhost" ):
                self.send_header( "Access-Control-Allow-Origin", o )

        def _send( self, code, body: bytes, kind = "text/plain; charset=utf-8" ):
            self.send_response( code )
            self.send_header( "Content-Type", kind )
            self.send_header( "Content-Length", str( len( body ) ) )
            self._cors()
            self.end_headers()
            self.wfile.write( body )

        def do_GET( self ):
            url = urlsplit( self.path )
            parts = [ p for p in url.path.split( "/" ) if p ]
            if parts == [ "health" ]:
                return self._send( 200, b"ok" )
            where = self.headers.get( "X-Errand-Root" )
            store = Path( unquote( where ) ) if where else ( Path( root ) if root else None )
            if store is None or not ( store / "manifest.json" ).is_file():
                return self._send( 404, b"no store: X-Errand-Root or --root" )
            if len( parts ) != 2 or parts[ 0 ] != "frame" or not parts[ 1 ].isdigit():
                return self._send( 404, b"GET /frame/<i>" )
            q = parse_qs( url.query )
            opts = { }
            if "rays" in q:
                opts[ "rays" ] = int( q[ "rays" ][ 0 ] )
            if "radius" in q:
                opts[ "radius" ] = float( q[ "radius" ][ 0 ] )
            i = int( parts[ 1 ] )
            try:
                m = json.loads( ( store / "manifest.json" ).read_text() )
                key = ( str( store ), m[ "frames" ][ i ][ "file" ], m.get( "base" ), tuple( sorted( opts.items() ) ) )
            except ( OSError, ValueError, IndexError, KeyError ) as err:
                return self._send( 404, f"frame { i }: { err }".encode() )
            if key not in kept:
                from ..store import chunk_bytes
                kept[ key ] = chunk_bytes( { "poly_ao": frame_occlusion( store, i, **opts ) } )
                while len( kept ) > KEEP:
                    kept.popitem( last = False )
            kept.move_to_end( key )
            return self._send( 200, kept[ key ], "application/octet-stream" )

    ThreadingHTTPServer( ( "127.0.0.1", port ), H ).serve_forever()


def main( argv = None ):
    p = argparse.ArgumentParser( prog = "sdot.ao", description = "the baked occlusion of a store's frames, over HTTP" )
    p.add_argument( "--port", type = int, required = True )
    p.add_argument( "--root", help = "the store, when no relay says it ( X-Errand-Root )" )
    a = p.parse_args( argv )
    serve( a.port, a.root )


if __name__ == "__main__":
    main()
