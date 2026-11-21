"""sdot.region: the cells of a REGION of a frame, rebuilt from the frame's sources ( sdot ADR 0001 §6 ).

    python -m sdot.viz.services.region --port P [--root DIR]

    GET /frame/<i>?mode=half&plane=nx,ny,nz,d       the cells whose seed is on the side n . x <= d ( a cut that opens the foam )
    GET /frame/<i>?mode=boundary                    the cells touching the domain ( its half-spaces, else the cells' box )
    GET /frame/<i>?mode=box&lo=x,y,z&hi=x,y,z       the cells whose seed is in the box
    GET /health                                     `ok`

-> a chunk ( `store.py` ): `pos`, `poly_v`, `poly_off`, `poly_c`, `edge_v`, `edge_c`; `meta.colors` / `meta.colors_dark`: the
table the colour ids index ( the page appends it to its own ), `meta.cells`: how many. The cells keep their colours
( `color_at( color0 + i )`, as `PowerDiagram.add_to_viz` drew them ).

A frame's sources ( kind `sdot.power_diagram`: `seeds`, `weights`, `bnd_dirs`, `bnd_offs`, `meta.color0` ) give a
`PowerDiagram`, drawn once ( all its cells ) and kept in memory per sources file; a region is then only a choice among the
cells. Each block of faces is told its cell by the power distance of its centre ( a KD-tree on the lifted seeds ).
"""
from __future__ import annotations

import argparse
import json
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np

KEEP = 4
_kept = OrderedDict()
_lock = threading.Lock()


class Cells:
    """every cell of a power diagram, drawn once: per cell its vertices, faces and edges ( as one `add_to_viz` drew them )"""

    def __init__( self, arrays: dict, meta: dict ):
        import sdot
        from scipy.spatial import cKDTree
        from ..Visualizer import Visualizer
        seeds = np.asarray( arrays[ "seeds" ], float )
        w = np.asarray( arrays.get( "weights", np.zeros( len( seeds ) ) ), float ).reshape( -1 )
        bnd = ( arrays[ "bnd_dirs" ], arrays[ "bnd_offs" ] ) if "bnd_dirs" in arrays else None
        pd = sdot.PowerDiagram( seeds, w, bnd )
        v = Visualizer()
        c0 = int( meta.get( "color0", 0 ) )
        if c0:
            v.reserve_colors( c0 )
        pd.add_to_viz( v, seeds = False )
        self.seeds, self.bnd = seeds, bnd
        self.pos = v.positions.astype( np.float64 )
        polys = v.polygons
        self.poly_c = np.asarray( v.polygon_colors )
        self.edges, self.edge_c = np.asarray( v.edges ).reshape( -1, 2 ), np.asarray( v.edge_colors )
        self.colors, self.colors_dark = [ list( c ) for c in v._colors ], [ list( c ) for c in v._colors_dark ]
        # the blocks: each cell was one `add_faces`, its own block of the pool -- a new block starts above the last's vertices
        starts, top = [ ], -1
        for k, p in enumerate( polys ):
            if min( p ) > top:
                starts.append( k )
            top = max( top, max( p ) )
        self.polys = polys
        self.blocks = [ ]                                          # ( first polygon, end polygon, first vertex, end vertex )
        for b, s in enumerate( starts ):
            e = starts[ b + 1 ] if b + 1 < len( starts ) else len( polys )
            vs = [ i for p in polys[ s : e ] for i in p ]
            self.blocks.append( ( s, e, min( vs ), max( vs ) + 1 ) )
        cen = np.array( [ self.pos[ lo : hi ].mean( axis = 0 ) for _, _, lo, hi in self.blocks ] ) if self.blocks else np.zeros( ( 0, 3 ) )
        lift = np.sqrt( np.maximum( w.max( initial = 0 ) - w, 0 ) )[ :, None ]   # ( |c - x|² - w is the distance to ( x, √( W - w ) ) )
        self.cell = cKDTree( np.hstack( [ seeds, lift ] ) ).query( np.hstack( [ cen, np.zeros( ( len( cen ), 1 ) ) ] ) )[ 1 ] \
            if len( cen ) else np.zeros( 0, int )
        # the edges of each block, by their first vertex
        ev0 = self.edges[ :, 0 ] if len( self.edges ) else np.zeros( 0, int )
        los = np.array( [ b[ 2 ] for b in self.blocks ] )
        self.edge_block = np.searchsorted( los, ev0, side = "right" ) - 1 if len( los ) else ev0

    def touching( self ) -> np.ndarray:
        """per block: whether one of its vertices lies on the domain ( its half-spaces, else the box of all the cells )"""
        span = float( np.ptp( self.pos, axis = 0 ).max() ) if len( self.pos ) else 1.0
        eps = 1e-6 * span
        if self.bnd is not None:
            d, o = np.asarray( self.bnd[ 0 ], float ), np.asarray( self.bnd[ 1 ], float ).reshape( -1 )
            on = np.any( np.abs( self.pos @ d.T - o ) < eps, axis = 1 )
        else:                                                      # ( the box of the faces' vertices, not of every edge drawn )
            used = np.concatenate( [ np.arange( a, b ) for _, _, a, b in self.blocks ] ) if self.blocks else np.zeros( 0, int )
            lo, hi = self.pos[ used ].min( axis = 0 ), self.pos[ used ].max( axis = 0 )
            on = np.any( ( self.pos - lo < eps ) | ( hi - self.pos < eps ), axis = 1 )
        return np.array( [ bool( on[ a : b ].any() ) for _, _, a, b in self.blocks ] )

    def chunk( self, keep: np.ndarray ) -> bytes:
        """the blocks `keep` ( booleans ) as a chunk, their vertices re-indexed"""
        from ..store import chunk_bytes
        pos, pv, off, pc, ev, ec, shift = [ ], [ ], [ 0 ], [ ], [ ], [ ], 0
        for b, ( s, e, lo, hi ) in enumerate( self.blocks ):
            if not keep[ b ]:
                continue
            pos.append( self.pos[ lo : hi ] )
            for k in range( s, e ):
                p = np.asarray( self.polys[ k ] ) - lo + shift
                pv.append( p )
                off.append( off[ -1 ] + len( p ) )
            pc.append( self.poly_c[ s : e ] )
            sel = self.edge_block == b
            ev.append( self.edges[ sel ] - lo + shift )
            ec.append( self.edge_c[ sel ] )
            shift += hi - lo
        cat = lambda xs, dt, shape = None: np.concatenate( xs ).astype( dt ) if xs else np.zeros( shape or 0, dt )
        arrays = dict( pos = cat( pos, np.float32, ( 0, 3 ) ), poly_v = cat( pv, np.int32 ), poly_off = np.asarray( off, np.int32 ),
                       poly_c = cat( pc, np.uint16 ), edge_v = cat( ev, np.int32, ( 0, 2 ) ), edge_c = cat( ec, np.uint16 ) )
        return chunk_bytes( arrays, dict( colors = self.colors, colors_dark = self.colors_dark, cells = int( keep.sum() ) ) )

    def region( self, mode: str, plane = None, lo = None, hi = None ) -> bytes:
        x = self.seeds[ self.cell ] if len( self.cell ) else np.zeros( ( 0, 3 ) )
        if mode == "half":
            n, d = np.asarray( plane[ :3 ], float ), float( plane[ 3 ] )
            keep = x @ n <= d
        elif mode == "boundary":
            keep = self.touching()
        elif mode == "box":
            keep = np.all( ( x >= np.asarray( lo, float ) ) & ( x <= np.asarray( hi, float ) ), axis = 1 )
        else:
            raise ValueError( f"mode: half, boundary or box ( not {mode!r} )" )
        return self.chunk( keep )


def cells_of( store, i: int ) -> Cells:
    from ..store import read_scene
    sc = read_scene( store )
    s = sc.manifest[ "frames" ][ i ].get( "sources" )
    if not s or s.get( "kind" ) != "sdot.power_diagram":
        raise LookupError( f"frame {i} has no sources of kind sdot.power_diagram" )
    key = ( str( store ), s[ "file" ] )
    with _lock:
        if key not in _kept:
            arrays, meta = sc.sources( i )
            _kept[ key ] = Cells( arrays, meta )
            while len( _kept ) > KEEP:
                _kept.popitem( last = False )
        _kept.move_to_end( key )
        return _kept[ key ]


def _floats( q, name, n ):
    v = [ float( x ) for x in q[ name ][ 0 ].split( "," ) ]
    if len( v ) != n:
        raise ValueError( f"{name}: {n} numbers" )
    return v


def serve( port: int, root: str | None = None ):
    class H( BaseHTTPRequestHandler ):
        def log_message( self, *a ):
            pass

        def _send( self, code, body: bytes, kind = "text/plain; charset=utf-8" ):
            self.send_response( code )
            self.send_header( "Content-Type", kind )
            self.send_header( "Content-Length", str( len( body ) ) )
            o = self.headers.get( "Origin" ) or ""
            if urlsplit( o ).hostname in ( "127.0.0.1", "localhost" ):
                self.send_header( "Access-Control-Allow-Origin", o )
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
                return self._send( 404, b"GET /frame/<i>?mode=half|boundary|box" )
            q = parse_qs( url.query )
            try:
                mode = q.get( "mode", [ "boundary" ] )[ 0 ]
                cells = cells_of( store, int( parts[ 1 ] ) )
                body = cells.region( mode, plane = _floats( q, "plane", 4 ) if mode == "half" else None,
                                     lo = _floats( q, "lo", 3 ) if mode == "box" else None, hi = _floats( q, "hi", 3 ) if mode == "box" else None )
            except ( LookupError, ValueError, KeyError, IndexError ) as err:
                return self._send( 400, f"region: {err}".encode() )
            return self._send( 200, body, "application/octet-stream" )

    ThreadingHTTPServer( ( "127.0.0.1", port ), H ).serve_forever()


def main( argv = None ):
    p = argparse.ArgumentParser( prog = "sdot.region", description = "the cells of a region of a store's frame, over HTTP" )
    p.add_argument( "--port", type = int, required = True )
    p.add_argument( "--root", help = "the store, when no relay says it ( X-Errand-Root )" )
    a = p.parse_args( argv )
    serve( a.port, a.root )


if __name__ == "__main__":
    main()
