"""A SCENE STORE: the frames of a `Visualizer` as files, written while the frames are made, read one at a time ( errand
ADR 0017 ). What a run writes instead of one self-contained page.

    foam.sviz/
        manifest.json        the scene: title, dim, bounds, axes, colours, layers, fields, images, curves, options, and the
                             list of the frames ( their values, file, size, counts, the range of each field ). Rewritten whole
                             and atomically; the only file that ever changes
        base-<hash>.bin      what every frame draws ( the SHARED primitives, the images ): a chunk named by its contents,
                             so a changed base is a new file
        frames/000123.bin    one frame, a chunk written once ( a temporary name, then renamed ), never changed

= A chunk

    u32 length | JSON header | blobs

The header is `{ "arrays": [ { name, dtype, shape, k, at, len, lo?, step? } ], "meta": { ... } }`. A blob is the bytes of an
array SHUFFLED ( the first byte of every item, then the second, ... ) and deflated ( zlib ): what `Visualizer._pack` does
for the self-contained page, so that the browser inflates it with its `DecompressionStream( "deflate" )`. With `lo` and
`step`, the array holds 16-bit codes of floats ( per column for a 2D array ), 65535 for NaN.

A chunk is a SCENE of its own: its primitives index its own pool of positions ( `pos` ), so a frame is drawn with nothing
but the base. Its arrays, by the names the page uses:

    pos [ n, d ]                         the pool of positions
    poly_v, poly_off, poly_c             polygons of free length ( indices, offsets [ nb + 1 ], colour ids )
    edge_v [ m, 2 ], edge_c, edge_l      edges, their colour ids, their layers
    pnt_v, pnt_c, pnt_r, pnt_m [ k, 4 ], pnt_l     points, colour ids, radii, matrices, layers
    field:<name> [ k ]                   a value per point, NaN where a point has none
    hp_dirs [ c, d ], hp_offs [ c ]      polytopes in H-representation, `meta.hpolys` giving their sizes and colours
    image:<i> [ ny, nx ] uint8           ( base only ) the grey levels of image i, `meta.images` giving the rest

The colour ids refer to the table of the manifest ( `colors`, `colors_dark` ), common to the base and all the frames.
With `compact`, every float array but the polytopes' is on 16 bits; the polytopes stay in float32 ( the page cuts them ).

`read_scene( path )` reads a store back, numpy only.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import time
import zlib
from pathlib import Path

import numpy as np

FORMAT = 1


# -- chunks ----------------------------------------------------------------------------------------------------------------

def _quantized( a, axis = None ):
    """`( codes uint16, lo, step )`: `a` on 16 bits between its min and max ( per column with `axis = 0` ), NaN as 65535"""
    a = np.asarray( a, dtype = np.float64 )
    ok = np.isfinite( a )
    if not ok.any():
        return np.full( a.shape, 65535, np.uint16 ), np.zeros( a.shape[ 1: ] if axis == 0 else () ), np.ones( a.shape[ 1: ] if axis == 0 else () )
    lo = np.nanmin( np.where( ok, a, np.nan ), axis = axis )
    hi = np.nanmax( np.where( ok, a, np.nan ), axis = axis )
    lo, hi = np.where( np.isfinite( lo ), lo, 0.0 ), np.where( np.isfinite( hi ), hi, 0.0 )
    step = np.where( hi > lo, ( hi - lo ) / 65534, 1.0 )
    q = np.where( ok, np.round( ( np.where( ok, a, lo ) - lo ) / step ), 65535 ).astype( np.uint16 )
    return q, lo, step


def _shuffled( a ):
    k = a.dtype.itemsize
    return a.view( np.uint8 ).reshape( -1, k ).T.tobytes() if k > 1 and a.size else a.tobytes()


def chunk_bytes( arrays: dict, meta = None, compact = False, exact = ( ), level = 6 ) -> bytes:
    """`arrays`: `{ name: ndarray }`. With `compact`, the float arrays not named in `exact` become 16-bit codes."""
    head, blobs, at = [ ], [ ], 0
    for name, a in arrays.items():
        a = np.ascontiguousarray( a )
        if a.dtype == np.float64:
            a = a.astype( np.float32 )
        e = dict( name = name, shape = list( a.shape ) )
        if compact and a.dtype.kind == "f" and name not in exact:
            q, lo, step = _quantized( a, axis = 0 if a.ndim == 2 else None )
            e.update( lo = np.atleast_1d( lo ).astype( float ).tolist(), step = np.atleast_1d( step ).astype( float ).tolist() )
            a = q
        e.update( dtype = a.dtype.str.lstrip( "<|=" ), k = a.dtype.itemsize )
        z = zlib.compress( _shuffled( a ), level )
        e.update( at = at, len = len( z ) )
        head.append( e )
        blobs.append( z )
        at += len( z )
    h = json.dumps( dict( arrays = head, meta = meta or { } ), separators = ( ",", ":" ) ).encode()
    return struct.pack( "<I", len( h ) ) + h + b"".join( blobs )


def read_chunk( data: bytes ) -> tuple:
    """`( arrays, meta )`: the inverse of `chunk_bytes`, the 16-bit codes back to float32"""
    n, = struct.unpack_from( "<I", data )
    head = json.loads( data[ 4 : 4 + n ] )
    base, out = 4 + n, { }
    for e in head[ "arrays" ]:
        raw = zlib.decompress( data[ base + e[ "at" ] : base + e[ "at" ] + e[ "len" ] ] )
        k = e[ "k" ]
        if k > 1 and raw:
            raw = np.frombuffer( raw, np.uint8 ).reshape( k, -1 ).T.tobytes()
        a = np.frombuffer( raw, np.dtype( e[ "dtype" ] ) ).reshape( e[ "shape" ] )
        if "lo" in e:
            lo, st = np.array( e[ "lo" ], np.float64 ), np.array( e[ "step" ], np.float64 )
            if a.ndim != 2:
                lo, st = lo[ 0 ], st[ 0 ]
            a = np.where( a == 65535, np.nan, lo + st * a ).astype( np.float32 )
        out[ e[ "name" ] ] = a
    return out, head[ "meta" ]


def _atomic( path: Path, data: bytes ):
    """written whole or not at all: a reader never sees a half-written file"""
    tmp = path.with_name( f".{ path.name }.{ os.getpid() }.tmp" )
    tmp.write_bytes( data )
    os.replace( tmp, path )


# -- a scene, as the arrays of a chunk -------------------------------------------------------------------------------------

def scene_arrays( viz, frame = 0, images = False ) -> tuple:
    """`( arrays, meta, ranges )`: frame `frame` of `viz` as the arrays of a chunk, re-indexed on a pool of its own ( only the
    positions its primitives use ), with the shared primitives first; `ranges`: `{ field: [ lo, hi ] }`. `images`: the
    images of `viz` too ( the base )."""
    fr = viz.frame( frame )
    pos = viz.positions
    d = viz.nb_dims or 2
    polys = fr[ "polygons" ]
    poly_v = np.array( [ i for p in polys for i in p ], np.int64 )
    edges, pnts = fr[ "edges" ].astype( np.int64 ), fr[ "points" ].astype( np.int64 )
    used = np.unique( np.concatenate( [ poly_v, edges.reshape( -1 ), pnts ] ) ) if len( pos ) else np.zeros( 0, np.int64 )
    local = np.full( len( pos ), -1, np.int64 )
    local[ used ] = np.arange( len( used ) )
    sv, se, sp = viz._shared[ "pnt" ], viz._shared[ "edge" ], viz._shared[ "poly" ]
    rng = lambda key: viz._frame_ranges( key )[ frame : frame + 2 ]
    v0, v1 = rng( "pnt" )
    e0, e1 = rng( "edge" )

    arrays = dict( pos = pos[ used ].reshape( -1, d ) if len( used ) else np.zeros( ( 0, d ), np.float32 ) )
    if len( polys ):
        arrays.update( poly_v = local[ poly_v ].astype( np.int32 ),
                       poly_off = np.concatenate( [ [ 0 ], np.cumsum( [ len( p ) for p in polys ] ) ] ).astype( np.int32 ),
                       poly_c = fr[ "polygon_colors" ].astype( np.uint16 ) )
    if len( edges ):
        layers = _cat_u8( viz._edge_l )
        arrays.update( edge_v = local[ edges ].astype( np.int32 ), edge_c = fr[ "edge_colors" ].astype( np.uint16 ),
                       edge_l = np.concatenate( [ layers[ : se ], layers[ e0 : e1 ] ] ) )
    ranges = { }
    if len( pnts ):
        layers = _cat_u8( viz._pnt_l )
        arrays.update( pnt_v = local[ pnts ].astype( np.int32 ), pnt_c = fr[ "point_colors" ].astype( np.uint16 ),
                       pnt_r = fr[ "point_radii" ].astype( np.float32 ), pnt_m = fr[ "point_matrices" ].astype( np.float32 ),
                       pnt_l = np.concatenate( [ layers[ : sv ], layers[ v0 : v1 ] ] ) )
        for name, v in _fields( viz ).items():
            v = np.concatenate( [ v[ : sv ], v[ v0 : v1 ] ] ).astype( np.float32 )
            ok = v[ np.isfinite( v ) ]
            if ok.size:
                arrays[ f"field:{ name }" ] = v
                ranges[ name ] = [ float( ok.min() ), float( ok.max() ) ]
    meta = { }
    if fr[ "polytopes" ]:
        hp = viz._hpolys[ rng( "hpoly" )[ 0 ] : rng( "hpoly" )[ 1 ] ]
        arrays.update( hp_dirs = np.concatenate( [ p[ 0 ] for p in fr[ "polytopes" ] ] ).astype( np.float32 ),
                       hp_offs = np.concatenate( [ p[ 1 ] for p in fr[ "polytopes" ] ] ).astype( np.float32 ) )
        meta[ "hpolys" ] = [ dict( nb = h[ "nb" ], fcol = h[ "fcol" ], ecol = h[ "ecol" ], edg = h[ "edg" ] ) for h in hp ]
    if images:
        import base64
        meta[ "images" ] = [ ]
        for i, im in enumerate( viz._images ):
            arrays[ f"image:{ i }" ] = np.frombuffer( base64.b64decode( im[ "data" ] ), np.uint8 ).reshape( im[ "ny" ], im[ "nx" ] )
            meta[ "images" ].append( { k: v for k, v in im.items() if k != "data" } )
    return arrays, meta, ranges


def _cat_u8( blocks ):
    return np.concatenate( blocks ).astype( np.uint8 ) if blocks else np.zeros( 0, np.uint8 )


def _fields( viz ) -> dict:
    """`{ name: [ all points ] }`, NaN where a chunk of points does not have the field"""
    names = [ ]
    for _, f in viz._pnt_f:
        names += [ k for k in f if k not in names ]
    return { k: np.concatenate( [ f[ k ] if k in f else np.full( n, np.nan, np.float32 ) for n, f in viz._pnt_f ] ) for k in names }


EXACT = ( "hp_dirs", "hp_offs" )


# -- the writer ------------------------------------------------------------------------------------------------------------

class StoreWriter:
    """What a `Visualizer( store = path )` writes through: the frames as they are flushed, the base when it changes, the
    manifest after each frame ( at most every `every` seconds ) and at the end."""

    def __init__( self, path, on_frame = None, every = 1.0 ):
        self.path = Path( path )
        ( self.path / "frames" ).mkdir( parents = True, exist_ok = True )
        for old in self.path.glob( "frames/*.bin" ):              # a store written again starts empty
            old.unlink()
        self.on_frame, self.every = on_frame, float( every )
        self.frames, self.base, self.bounds, self.written = [ ], None, None, 0.0

    def write_frame( self, viz, value ):
        arrays, meta, ranges = scene_arrays( viz )
        data = chunk_bytes( arrays, meta, viz.compact, EXACT )
        name = f"frames/{ len( self.frames ):06d}.bin"
        _atomic( self.path / name, data )
        self._note_bounds( viz )
        self.frames.append( dict( v = list( value ), file = name, size = len( data ), fields = ranges,
                                  n = { k: int( len( arrays[ a ] ) ) for k, a in ( ( "pnt", "pnt_v" ), ( "edge", "edge_v" ), ( "poly", "poly_c" ) ) if a in arrays } ) )
        return len( data )

    def write_base( self, base_viz ):
        arrays, meta, _ = scene_arrays( base_viz, images = True )
        data = chunk_bytes( arrays, meta, base_viz.compact, EXACT )
        name = f"base-{ hashlib.sha256( data ).hexdigest()[ :16 ] }.bin"
        if not ( self.path / name ).exists():
            _atomic( self.path / name, data )
        self._note_bounds( base_viz )
        self.base = name

    def _note_bounds( self, viz ):
        pts = viz._bb_pts or viz._pools
        if not pts:
            return
        p = np.concatenate( pts, axis = 0 )
        if not len( p ):
            return
        lo, hi = p.min( axis = 0 ), p.max( axis = 0 )
        if self.bounds is not None:
            lo, hi = np.minimum( lo, self.bounds[ 0 ] ), np.maximum( hi, self.bounds[ 1 ] )
        self.bounds = ( lo, hi )

    def write_manifest( self, viz, complete = False, force = False ):
        now = time.monotonic()
        if not ( force or complete ) and now - self.written < self.every:
            return False
        self.written = now
        from .Visualizer import _is_pair, _rgb
        bg = viz.background if _is_pair( viz.background ) else ( viz.background, viz.background )
        m = dict( format = FORMAT, title = viz.title, dim = viz.nb_dims, complete = complete,
                  axes = list( viz.frame_axes ),
                  playable = [ bool( p ) for p in ( viz.playable if isinstance( viz.playable, ( tuple, list ) ) else [ viz.playable ] * len( viz.frame_axes ) ) ],
                  bounds = None if self.bounds is None else [ [ float( a ), float( b ) ] for a, b in zip( *self.bounds ) ],
                  colors = [ list( c ) for c in viz._colors ], colors_dark = [ list( c ) for c in viz._colors_dark ],
                  layers = viz._layers, field = viz.field, compact = viz.compact, curves = viz._curves, base = self.base,
                  options = dict( fps = viz.fps, background = viz.background, depth_colors = viz.depth_colors, morph = viz.morph,
                                  loop = viz.loop, autoplay = viz.autoplay, face_opacity = viz.face_opacity, flat_points = viz.flat_points,
                                  bg = [ None if c is None else list( _rgb( c ) ) for c in bg ] ),
                  frames = self.frames )
        _atomic( self.path / "manifest.json", json.dumps( m, separators = ( ",", ":" ), default = _json_default ).encode() )
        _say_view( self.path )
        if self.on_frame is not None:
            self.on_frame()
        return True


def _say_view( path ):
    """Under errand, the store is a VIEW of the run ( errand ADR 0017 ): named again at each manifest, what changed in
    it comes back to the side that drives the run while it goes, the manifest after the files it names. Nothing
    without errand, nor with an errand that does not know directory views."""
    try:
        import errand
    except ImportError:
        return
    where = str( Path( path ).absolute() )
    try:
        if hasattr( errand, "Static" ):                 # opened as it is: its `index.html`, served beside errand's page
            errand.view( where, last = "manifest.json", open = errand.Static( "index.html" ) )
        else:
            errand.view( where, last = "manifest.json" )
    except TypeError:
        pass                                            # ( an older errand: no `last`, and no directory views )
    except Exception:
        pass                                            # a view is a courtesy: never the run's failure


def _json_default( x ):
    if isinstance( x, np.generic ):
        return x.item()
    if isinstance( x, np.ndarray ):
        return x.tolist()
    raise TypeError( f"{ type( x ).__name__ } is not JSON" )


# -- the reader ------------------------------------------------------------------------------------------------------------

class Scene:
    """A store read back: `len( s )` frames, `s.manifest`, `s.base()` and `s.frame( i )` as `( arrays, meta )`, `s.value( i )`.
    The arrays are those of a chunk ( see the module ), the 16-bit codes back to float32."""

    def __init__( self, path ):
        self.path = Path( path )
        self.manifest = json.loads( ( self.path / "manifest.json" ).read_text() )

    def __len__( self ):
        return len( self.manifest[ "frames" ] )

    def value( self, i ):
        v = self.manifest[ "frames" ][ i ][ "v" ]
        return v[ 0 ] if len( v ) == 1 else tuple( v )

    def base( self ):
        b = self.manifest.get( "base" )
        return read_chunk( ( self.path / b ).read_bytes() ) if b else ( { }, { } )

    def frame( self, i ):
        return read_chunk( ( self.path / self.manifest[ "frames" ][ i ][ "file" ] ).read_bytes() )


def read_scene( path ) -> Scene:
    return Scene( path )
