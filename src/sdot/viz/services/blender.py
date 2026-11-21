"""blender: pictures of a store's frames by Cycles, as the viewer shows them ( sdot ADR 0001 §5 ).

Runs under BLENDER's Python, which has numpy but not sdot: this file imports nothing of sdot but `store.py`, loaded by its
path ( stdlib + numpy ).

    /Applications/Blender.app/Contents/MacOS/Blender -b -P .../sdot/viz/services/blender.py -- --port P [--root DIR]
    ... -- --once STORE FRAME OUT.png [--samples N]       one picture, from a camera that frames the scene ( tests )

    GET  /health    `ok`
    POST /render    JSON { frame, camera: { eye, right, up, fwd, fov, ortho, half_h }, axes, width, height, samples,
                    theme, point_radius, edge_radius, faces, edges, points, view, region, keep } -> image/png
                    ( `region`: a chunk of `sdot.region` in base64, the cells the page shows instead of the frame's; `keep`: also
                    written as `renders/frame-<i>-<hash>.png` in the store, its path in `X-Kept`, and listed in
                    `renders/index.json` with the frame and the page's `view`, for the page's list of kept pictures )

The store is `X-Errand-Root` ( percent-encoded, errand's relay ), else `--root`. One request at a time ( bpy is not
thread safe: the server is not threaded ); the scene of the last ( store, frame, theme ) is kept loaded, a request for
the same frame only moves the camera. A progressive picture is two requests ( few samples, then many ).

The scene, from the frame's chunk and the base's ( the viewer's own data ):
    faces     one mesh, a colour per face ( attribute `col`, alpha = the colour's ), a principled material
    edges     a mesh of edges turned into tubes ( geometry nodes ), the colour of the most frequent edge colour
    points    spheres instanced on the points, radius per point ( `pnt_r`, or `point_radius` ), colour per point
    light     the world in the theme's background colour, and a sun from the upper left of the view, as the viewer's
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np

_STORE = None


def _store_module():
    """`store.py` loaded by its path ( no `import sdot`: Blender's Python has not got it )"""
    global _STORE
    if _STORE is None:
        path = Path( __file__ ).resolve().parent.parent / "store.py"
        spec = importlib.util.spec_from_file_location( "sviz_store", path )
        _STORE = importlib.util.module_from_spec( spec )
        spec.loader.exec_module( _STORE )
    return _STORE


# -- the data of a frame ( no bpy: tested without Blender ) ----------------------------------------------------------------

def frame_data( store, i: int, axes = ( 0, 1, 2 ), theme = "light", region = None ) -> dict:
    """the base and frame `i` of `store` combined as the page combines them, in 3D ( `axes` chosen ), colours resolved.
    `region`: `( arrays, meta )` of a chunk of `sdot.region`, whose faces and edges replace the frame's ( the page shows
    them so ), its colours from its own table"""
    st = _store_module()
    sc = st.read_scene( store )
    ( ba, _ ), ( fa, _ ) = sc.base(), sc.frame( i )
    table = np.asarray( sc.manifest[ "colors_dark" if theme == "dark" else "colors" ], np.float64 ).reshape( -1, 4 )
    table = np.concatenate( [ _linear( table[ :, :3 ] ), table[ :, 3: ] ], axis = 1 )   # ( the page's colours are sRGB )
    dim = int( sc.manifest.get( "dim" ) or ( fa.get( "pos", ba.get( "pos" ) ).shape[ 1 ] ) )
    take = lambda p: np.stack( [ p[ :, a ] if 0 <= a < p.shape[ 1 ] else np.zeros( len( p ) ) for a in axes ], axis = 1 )
    nb = len( ba[ "pos" ] ) if "pos" in ba else 0
    pos = [ take( x[ "pos" ].reshape( -1, dim ).astype( np.float64 ) ) for x in ( ba, fa ) if "pos" in x ]
    out = dict( pos = np.concatenate( pos ) if pos else np.zeros( ( 0, 3 ) ) )

    def cat( key, shift = False, kind = np.int64 ):
        parts = [ x[ key ].astype( kind ) + ( s if shift else 0 ) for x, s in ( ( ba, 0 ), ( fa, nb ) ) if key in x ]
        return np.concatenate( parts ) if parts else np.zeros( 0, kind )

    out[ "poly_v" ] = cat( "poly_v", True )
    offs, last = [ np.zeros( 1, np.int64 ) ], 0
    for x in ( ba, fa ):
        if "poly_off" in x:
            offs.append( x[ "poly_off" ][ 1: ].astype( np.int64 ) + last )
            last = offs[ -1 ][ -1 ]
    out[ "poly_off" ] = np.concatenate( offs )
    out[ "face_col" ] = table[ cat( "poly_c" ) ] if len( out[ "poly_v" ] ) else np.zeros( ( 0, 4 ) )
    out[ "edge_v" ] = cat( "edge_v", True ).reshape( -1, 2 )
    ec = cat( "edge_c" )
    out[ "edge_col" ] = table[ Counter( ec.tolist() ).most_common( 1 )[ 0 ][ 0 ] ] if len( ec ) else np.array( [ .1, .1, .1, 1 ] )
    out[ "pnt_v" ] = cat( "pnt_v", True )
    out[ "pnt_col" ] = table[ cat( "pnt_c" ) ] if len( out[ "pnt_v" ] ) else np.zeros( ( 0, 4 ) )
    out[ "pnt_r" ] = cat( "pnt_r", kind = np.float64 ) if len( out[ "pnt_v" ] ) else np.zeros( 0 )
    if region is not None:
        ra, rm = region
        rtab = np.asarray( rm[ "colors_dark" if theme == "dark" else "colors" ], np.float64 ).reshape( -1, 4 )
        rtab = np.concatenate( [ _linear( rtab[ :, :3 ] ), rtab[ :, 3: ] ], axis = 1 )
        shift = len( out[ "pos" ] )
        rpos = take( ra[ "pos" ].reshape( -1, dim ).astype( np.float64 ) ) if len( ra[ "pos" ] ) else np.zeros( ( 0, 3 ) )
        out[ "pos" ] = np.concatenate( [ out[ "pos" ], rpos ] )
        out[ "poly_v" ] = ra[ "poly_v" ].astype( np.int64 ) + shift
        out[ "poly_off" ] = ra[ "poly_off" ].astype( np.int64 )
        out[ "face_col" ] = rtab[ ra[ "poly_c" ].astype( np.int64 ) ] if len( ra[ "poly_c" ] ) else np.zeros( ( 0, 4 ) )
        out[ "edge_v" ] = ra[ "edge_v" ].astype( np.int64 ).reshape( -1, 2 ) + shift
        ec = ra[ "edge_c" ].astype( np.int64 )
        if len( ec ):
            out[ "edge_col" ] = rtab[ Counter( ec.tolist() ).most_common( 1 )[ 0 ][ 0 ] ]
    bg = ( sc.manifest.get( "options" ) or { } ).get( "bg" ) or [ None, None ]
    bgc = bg[ 1 if theme == "dark" else 0 ]
    out[ "bg" ] = _linear( np.asarray( bgc if bgc is not None else ( [ 0.12, 0.12, 0.12 ] if theme == "dark" else [ 1.0, 1.0, 1.0 ] ), float )[ :3 ] )
    lo, hi = out[ "pos" ].min( axis = 0 ), out[ "pos" ].max( axis = 0 )
    out[ "span" ] = float( np.max( hi - lo ) ) if len( out[ "pos" ] ) else 1.0
    out[ "center" ] = ( lo + hi ) / 2 if len( out[ "pos" ] ) else np.zeros( 3 )
    return out


def _linear( c ):
    """sRGB -> linear ( what Blender's colours are )"""
    c = np.asarray( c, np.float64 )
    return np.where( c <= 0.04045, c / 12.92, ( ( c + 0.055 ) / 1.055 ) ** 2.4 )


def default_camera( d: dict ) -> dict:
    """a three-quarter view that frames the scene ( `--once` )"""
    f = -np.array( [ 1.0, 0.8, 0.9 ] ) / np.linalg.norm( [ 1.0, 0.8, 0.9 ] )
    r = np.cross( f, [ 0, 0, 1.0 ] )
    r /= np.linalg.norm( r )
    u = np.cross( r, f )
    fov = 45 * np.pi / 180
    dist = 0.75 * d[ "span" ] / np.tan( fov / 2 )
    return dict( eye = ( d[ "center" ] - dist * f ).tolist(), right = r.tolist(), up = u.tolist(), fwd = f.tolist(), fov = fov,
                 ortho = False, half_h = dist * np.tan( fov / 2 ) )


# -- the scene in Blender ----------------------------------------------------------------------------------------------------

class Scene:
    def __init__( self ):
        self.key = None

    def load( self, d: dict ):
        import bpy
        for ob in list( bpy.data.objects ):
            bpy.data.objects.remove( ob, do_unlink = True )
        for coll in ( bpy.data.meshes, bpy.data.materials, bpy.data.node_groups, bpy.data.lights, bpy.data.cameras ):
            for x in list( coll ):
                coll.remove( x )
        sc = bpy.context.scene
        self.tube = self.points = None
        pos = d[ "pos" ]
        self.span = d[ "span" ]
        if len( d[ "poly_v" ] ):
            off = d[ "poly_off" ]
            faces = [ d[ "poly_v" ][ off[ k ] : off[ k + 1 ] ].tolist() for k in range( len( off ) - 1 ) ]
            me = bpy.data.meshes.new( "faces" )
            me.from_pydata( pos.tolist(), [ ], faces )
            col = me.color_attributes.new( "col", "FLOAT_COLOR", "FACE" )
            col.data.foreach_set( "color", d[ "face_col" ].astype( np.float32 ).reshape( -1 ) )
            ob = bpy.data.objects.new( "faces", me )
            ob.data.materials.append( self._material( "faces", attribute = "col", kind = "GEOMETRY" ) )
            sc.collection.objects.link( ob )
        if len( d[ "edge_v" ] ):
            me = bpy.data.meshes.new( "edges" )
            me.from_pydata( pos.tolist(), d[ "edge_v" ].tolist(), [ ] )
            ob = bpy.data.objects.new( "edges", me )
            sc.collection.objects.link( ob )
            self.tube = self._tubes( ob, self._material( "edges", color = d[ "edge_col" ] ) )
        if len( d[ "pnt_v" ] ):
            me = bpy.data.meshes.new( "points" )
            me.from_pydata( pos[ d[ "pnt_v" ] ].tolist(), [ ], [ ] )
            r = me.attributes.new( "radius", "FLOAT", "POINT" )
            self.pnt_r = d[ "pnt_r" ].astype( np.float32 )
            r.data.foreach_set( "value", self.pnt_r )
            c = me.color_attributes.new( "col", "FLOAT_COLOR", "POINT" )
            c.data.foreach_set( "color", d[ "pnt_col" ].astype( np.float32 ).reshape( -1 ) )
            ob = bpy.data.objects.new( "points", me )
            sc.collection.objects.link( ob )
            self.points = ob
            self._spheres( ob, self._material( "points", attribute = "col", kind = "INSTANCER" ) )
        world = bpy.data.worlds.get( "World" ) or bpy.data.worlds.new( "World" )
        sc.world = world
        world.use_nodes = True
        wn = world.node_tree.nodes                                    # ( the camera sees the theme's colour, the light is dimmer )
        bgn = wn.get( "Background" )
        bgn.inputs[ "Color" ].default_value = ( 1.0, 1.0, 1.0, 1.0 )   # ( a neutral sky, whatever the theme )
        bgn.inputs[ "Strength" ].default_value = 0.6
        seen, mix, path = wn.new( "ShaderNodeBackground" ), wn.new( "ShaderNodeMixShader" ), wn.new( "ShaderNodeLightPath" )
        seen.inputs[ "Color" ].default_value = ( *d[ "bg" ].tolist(), 1.0 )
        seen.inputs[ "Strength" ].default_value = 1.0
        wl = world.node_tree.links
        wl.new( path.outputs[ "Is Camera Ray" ], mix.inputs[ 0 ] )
        wl.new( bgn.outputs[ 0 ], mix.inputs[ 1 ] )
        wl.new( seen.outputs[ 0 ], mix.inputs[ 2 ] )
        wl.new( mix.outputs[ 0 ], wn.get( "World Output" ).inputs[ "Surface" ] )
        sun = bpy.data.objects.new( "sun", bpy.data.lights.new( "sun", "SUN" ) )
        sun.data.energy = 3.0
        sun.data.angle = 0.2
        sc.collection.objects.link( sun )
        self.sun = sun
        cam = bpy.data.objects.new( "camera", bpy.data.cameras.new( "camera" ) )
        sc.collection.objects.link( cam )
        sc.camera = cam
        self.cam = cam

    def _material( self, name, color = None, attribute = None, kind = "GEOMETRY" ):
        import bpy
        m = bpy.data.materials.new( name )
        m.use_nodes = True
        nt = m.node_tree
        bsdf = nt.nodes.get( "Principled BSDF" )
        bsdf.inputs[ "Roughness" ].default_value = 0.55
        if attribute:
            a = nt.nodes.new( "ShaderNodeAttribute" )
            a.attribute_type = kind
            a.attribute_name = attribute
            nt.links.new( a.outputs[ "Color" ], bsdf.inputs[ "Base Color" ] )
            nt.links.new( a.outputs[ "Alpha" ], bsdf.inputs[ "Alpha" ] )
        else:
            bsdf.inputs[ "Base Color" ].default_value = ( *[ float( c ) for c in color[ :3 ] ], 1.0 )
        return m

    def _tubes( self, ob, mat ):
        """edges -> tubes; their radius is the profile circle's, set per request ( -> that node )"""
        import bpy
        g = bpy.data.node_groups.new( "tubes", "GeometryNodeTree" )
        g.interface.new_socket( "Geometry", in_out = "INPUT", socket_type = "NodeSocketGeometry" )
        g.interface.new_socket( "Geometry", in_out = "OUTPUT", socket_type = "NodeSocketGeometry" )
        n = g.nodes
        gi, go = n.new( "NodeGroupInput" ), n.new( "NodeGroupOutput" )
        m2c, c2m, circ, setm = n.new( "GeometryNodeMeshToCurve" ), n.new( "GeometryNodeCurveToMesh" ), \
            n.new( "GeometryNodeCurvePrimitiveCircle" ), n.new( "GeometryNodeSetMaterial" )
        circ.inputs[ "Resolution" ].default_value = 8
        setm.inputs[ "Material" ].default_value = mat
        g.links.new( gi.outputs[ 0 ], m2c.inputs[ "Mesh" ] )
        g.links.new( m2c.outputs[ "Curve" ], c2m.inputs[ "Curve" ] )
        g.links.new( circ.outputs[ "Curve" ], c2m.inputs[ "Profile Curve" ] )
        g.links.new( c2m.outputs[ "Mesh" ], setm.inputs[ "Geometry" ] )
        g.links.new( setm.outputs[ "Geometry" ], go.inputs[ 0 ] )
        ob.modifiers.new( "tubes", "NODES" ).node_group = g
        return circ

    def _spheres( self, ob, mat ):
        """a sphere instanced on each point, scaled by its `radius` attribute"""
        import bpy
        g = bpy.data.node_groups.new( "spheres", "GeometryNodeTree" )
        g.interface.new_socket( "Geometry", in_out = "INPUT", socket_type = "NodeSocketGeometry" )
        g.interface.new_socket( "Geometry", in_out = "OUTPUT", socket_type = "NodeSocketGeometry" )
        n = g.nodes
        gi, go = n.new( "NodeGroupInput" ), n.new( "NodeGroupOutput" )
        ico, inst, rad, setm = n.new( "GeometryNodeMeshIcoSphere" ), n.new( "GeometryNodeInstanceOnPoints" ), \
            n.new( "GeometryNodeInputNamedAttribute" ), n.new( "GeometryNodeSetMaterial" )
        ico.inputs[ "Subdivisions" ].default_value = 3
        ico.inputs[ "Radius" ].default_value = 1.0
        rad.data_type = "FLOAT"
        rad.inputs[ "Name" ].default_value = "radius"
        setm.inputs[ "Material" ].default_value = mat
        g.links.new( ico.outputs[ "Mesh" ], setm.inputs[ "Geometry" ] )
        g.links.new( gi.outputs[ 0 ], inst.inputs[ "Points" ] )
        g.links.new( setm.outputs[ "Geometry" ], inst.inputs[ "Instance" ] )
        g.links.new( rad.outputs[ "Attribute" ], inst.inputs[ "Scale" ] )
        g.links.new( inst.outputs[ "Instances" ], go.inputs[ 0 ] )
        ob.modifiers.new( "spheres", "NODES" ).node_group = g

    def frame( self, cam: dict, width: int, height: int, samples: int, point_radius = None, edge_radius = None,
               shown = None ) -> bytes:
        import bpy
        from mathutils import Matrix
        sc = bpy.context.scene
        for name, on in ( shown or { } ).items():                       # ( what the viewer hides: faces, edges, points )
            ob = bpy.data.objects.get( name )
            if ob is not None:
                ob.hide_render = not on
        r, u, f = ( np.asarray( cam[ k ], float ) for k in ( "right", "up", "fwd" ) )
        m = np.eye( 4 )
        m[ :3, 0 ], m[ :3, 1 ], m[ :3, 2 ], m[ :3, 3 ] = r, u, -f, cam[ "eye" ]
        self.cam.matrix_world = Matrix( m.tolist() )
        c = self.cam.data
        c.sensor_fit = "VERTICAL"
        if cam.get( "ortho" ):
            c.type = "ORTHO"
            c.ortho_scale = 2 * float( cam[ "half_h" ] ) * max( 1.0, width / height )
        else:
            c.type = "PERSP"
            c.angle_y = float( cam[ "fov" ] )
        c.clip_start, c.clip_end = 1e-4 * self.span, 1e3 * self.span
        light = 0.25 * r + 0.45 * u - 1.0 * f                          # ( from the upper left of the view, as the viewer )
        z = light / np.linalg.norm( light )                            # ( a sun shines along its -Z: its Z points to it )
        x = np.cross( u, z )
        x /= np.linalg.norm( x )
        ms = np.eye( 4 )
        ms[ :3, 0 ], ms[ :3, 1 ], ms[ :3, 2 ] = x, np.cross( z, x ), z
        self.sun.matrix_world = Matrix( ms.tolist() )
        if getattr( self, "tube", None ) is not None:
            self.tube.inputs[ "Radius" ].default_value = float( edge_radius if edge_radius else 0.0015 * self.span )
        if self.points is not None and point_radius:
            me = self.points.data                                      # ( bare points: the viewer's radius )
            rr = np.where( self.pnt_r > 0, self.pnt_r, float( point_radius ) ).astype( np.float32 )
            me.attributes[ "radius" ].data.foreach_set( "value", rr )
            me.update()
        sc.render.engine = "CYCLES"
        sc.cycles.samples = max( 1, int( samples ) )
        sc.cycles.use_denoising = True
        _gpu( sc )
        sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = int( width ), int( height ), 100
        sc.render.film_transparent = False
        sc.view_settings.view_transform = "Standard"
        sc.render.image_settings.file_format = "PNG"
        out = Path( tempfile.mkdtemp() ) / "frame.png"
        sc.render.filepath = str( out )
        bpy.ops.render.render( write_still = True )
        data = out.read_bytes()
        out.unlink()
        return data


def _gpu( sc ):
    """Cycles on the GPU when there is one ( Metal, OptiX, CUDA, HIP ), else the CPU"""
    import bpy
    try:
        prefs = bpy.context.preferences.addons[ "cycles" ].preferences
        for kind in ( "METAL", "OPTIX", "CUDA", "HIP" ):
            try:
                prefs.compute_device_type = kind
            except TypeError:
                continue
            prefs.get_devices()
            devs = [ d for d in prefs.devices if d.type == kind ]
            if devs:
                for d in prefs.devices:
                    d.use = d.type == kind
                sc.cycles.device = "GPU"
                return
    except Exception:
        pass
    sc.cycles.device = "CPU"


SCENE = Scene()


def render( store, req: dict ) -> bytes:
    theme = req.get( "theme", "light" )
    axes = tuple( int( a ) for a in req.get( "axes", ( 0, 1, 2 ) ) )
    i = int( req.get( "frame", 0 ) )
    m = json.loads( ( Path( store ) / "manifest.json" ).read_text() )
    raw = base64.b64decode( req[ "region" ] ) if req.get( "region" ) else None
    key = ( str( store ), m[ "frames" ][ i ][ "file" ], m.get( "base" ), theme, axes, hashlib.sha1( raw ).hexdigest() if raw else None )
    if key != SCENE.key:
        d = frame_data( store, i, axes, theme, _store_module().read_chunk( raw ) if raw else None )
        SCENE.load( d )
        SCENE.key = key
        SCENE.data = d
    cam = req.get( "camera" ) or default_camera( SCENE.data )
    return SCENE.frame( cam, req.get( "width", 800 ), req.get( "height", 600 ), req.get( "samples", 64 ),
                        req.get( "point_radius" ), req.get( "edge_radius" ),
                        { k: bool( req.get( k, True ) ) for k in ( "faces", "edges", "points" ) } )


def serve( port: int, root: str | None = None ):
    class H( BaseHTTPRequestHandler ):
        def log_message( self, *a ):
            pass

        def _cors( self ):
            o = self.headers.get( "Origin" ) or ""
            if urlsplit( o ).hostname in ( "127.0.0.1", "localhost" ):
                self.send_header( "Access-Control-Allow-Origin", o )
                self.send_header( "Access-Control-Allow-Headers", "Content-Type" )
                self.send_header( "Access-Control-Expose-Headers", "X-Kept" )

        def _send( self, code, body: bytes, kind = "text/plain; charset=utf-8", extra = None ):
            self.send_response( code )
            self.send_header( "Content-Type", kind )
            self.send_header( "Content-Length", str( len( body ) ) )
            for k, v in ( extra or { } ).items():
                self.send_header( k, v )
            self._cors()
            self.end_headers()
            self.wfile.write( body )

        def do_OPTIONS( self ):
            self._send( 204, b"" )

        def do_GET( self ):
            if urlsplit( self.path ).path.strip( "/" ) == "health":
                return self._send( 200, b"ok" )
            return self._send( 404, b"POST /render" )

        def do_POST( self ):
            if urlsplit( self.path ).path.strip( "/" ) != "render":
                return self._send( 404, b"POST /render" )
            where = self.headers.get( "X-Errand-Root" )
            store = Path( unquote( where ) ) if where else ( Path( root ) if root else None )
            if store is None or not ( store / "manifest.json" ).is_file():
                return self._send( 404, b"no store: X-Errand-Root or --root" )
            try:
                req = json.loads( self.rfile.read( int( self.headers.get( "Content-Length", 0 ) ) ) or b"{}" )
                png = render( store, req )
            except Exception as err:
                import traceback
                traceback.print_exc()
                return self._send( 500, f"render: { err }".encode() )
            extra = { }
            if req.get( "keep" ):
                h = hashlib.sha1( json.dumps( req, sort_keys = True ).encode() ).hexdigest()[ :10 ]
                rel = f"renders/frame-{ int( req.get( 'frame', 0 ) ):06d}-{ h }.png"
                f = store / rel
                f.parent.mkdir( exist_ok = True )
                f.write_bytes( png )
                extra[ "X-Kept" ] = rel
                index = store / "renders" / "index.json"                 # ( what the page lists: the frame, the view )
                try:
                    kept = json.loads( index.read_text() )
                except ( OSError, ValueError ):
                    kept = [ ]
                kept = [ e for e in kept if e.get( "file" ) != rel ] + [ dict(
                    file = rel, frame = int( req.get( "frame", 0 ) ), samples = int( req.get( "samples", 0 ) ),
                    view = req.get( "view" ), theme = req.get( "theme" ), when = time.strftime( "%Y-%m-%d %H:%M:%S" ) ) ]
                tmp = index.with_suffix( ".tmp" )
                tmp.write_text( json.dumps( kept, indent = 1 ) )
                tmp.replace( index )
            return self._send( 200, png, "image/png", extra )

    HTTPServer( ( "127.0.0.1", port ), H ).serve_forever()


def main( argv ):
    p = argparse.ArgumentParser( prog = "blender service", description = "Cycles pictures of a store's frames" )
    p.add_argument( "--port", type = int )
    p.add_argument( "--root" )
    p.add_argument( "--once", nargs = 3, metavar = ( "STORE", "FRAME", "OUT" ) )
    p.add_argument( "--samples", type = int, default = 32 )
    a = p.parse_args( argv )
    if a.once:
        store, i, out = a.once
        Path( out ).write_bytes( render( Path( store ), dict( frame = int( i ), samples = a.samples, width = 480, height = 360 ) ) )
        print( "written", out, flush = True )
        return
    if not a.port:
        p.error( "--port, or --once" )
    serve( a.port, a.root )


if __name__ == "__main__":
    main( sys.argv[ sys.argv.index( "--" ) + 1 : ] if "--" in sys.argv else [ ] )
