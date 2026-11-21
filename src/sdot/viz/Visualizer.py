"""Visualizer of geometries in any dimension: it is GIVEN primitives, it STORES them, and the
choice of output comes afterwards (`write_html`, `write_vtk`).

Counterpart of `otrec.viz.points_html` (a 2D point cloud on a <canvas>), extended to what a
CELL needs: edges with HIDDEN-part handling, possibly SOLID faces, in 2D, in 3D, or beyond
(beyond 3, one looks at a SLICE).

= What it is given

Primitives in WORLD coordinates, each one referring to a set of positions:

    v = Visualizer()
    v.add_points( positions )                       # [n, d]
    v.add_edges ( positions, edges )                # [n, d] + [m, 2] (or None: polyline)
    v.add_faces ( positions, faces )                # [n, d] + polygons (lists of indices)
    v.add_polytope( cut_directions, cut_offsets )   # H-representation { x : dir . x <= off }
    v.write_html( "cut.html" )                      # or v.write_vtk( "cut.vtu" )

The DIMENSION is not declared: it is read off the size of the position vectors (last axis),
and all additions to a given scene must agree on it.

= Time (or a parameter)

`new_frame( value )` opens a FRAME: everything that follows is stored in it. A frame is a
COMPLETE and independent state -- neither the vertices nor the connectivity need to match from
one frame to the next, cells may appear, disappear, change shape. This is what is needed to
replay a descent; it is also why this is NOT the same mechanism as the slices in dimension > 3,
where one looks at the same object from another angle.

    for k, x in enumerate( iterations ):
        if k: v.new_frame( x )
        ...

The axis is named (`frame_axis`, "time" by default) and plays by itself if it makes sense to
traverse it (`playable`): the page then has a bar and a play/pause button, and ParaView receives
a time series.

Several parameters: `frame_axis` is then a tuple of names, and `new_frame` takes one value per
axis ( a tuple, or a dict that only gives the axes that change ). The frames form a grid -- a
missing combination shows the nearest frame --, and each axis gets its own row in the bar:

    v = Visualizer( frame_axis = ( "mass", "iteration" ), playable = ( False, True ) )
    for m in masses:
        for it in iterations:
            v.new_frame( { "mass": m, "iteration": it } )

An object that knows how to draw itself exposes `add_to_viz( viz )` (see `Cell.add_to_viz`) and is
added with `v.add( obj )` -- the object decides its own primitives, the visualizer knows none.

= How it is stored

Nothing is formatted at addition time, everything is kept in the form CLOSEST to what was
received, so that each output can do what it wants with it (VTK will want the polygons as they
are, HTML will triangulate them):

- ONE pool of vertices for the whole scene. Two additions that receive the SAME array of positions
  (the faces and the edges of a cell) share it -- the array's identity is enough to say so.
- a primitive is only INDICES into this pool, plus a color index into a small table. This is
  what keeps the weight of an output down: a cube vertex serves 6 triangles and 3 edges, it is
  stored only once, and its color costs nothing per vertex.
- polygons keep their free length (split by an offsets array), faces are never triangulated at
  storage time.

= What the HTML page can do

The rendering goes through **hand-written WebGL2** rather than a 2D canvas: a filled 3D cell
costs a lot in triangles/edges, and it is the GPU that must absorb that. No dependency: a
three.js-type lib would require a CDN, hence a network connection, which would break the file's
autonomy (data in base64 in the HTML, no side file, no server -- opens as `file://`).

- the THEME ( light / dark ): `?theme=`, else that of the page that embeds it ( the `dark` class of the parent document, same
  origin, followed live: the doc's iframes switch with the doc ), else the system's; [d] toggles it. A color can be given per
  theme, `( light, dark )` -- the background too;
- the start view FRAMES the tight box of the scene ( its corners projected, a small margin ), again on a resize until the
  view is moved;
- the NAVIGATION switch, top right ( a four-arrow icon, outlined when on ): whether the wheel, the pinch and the touch gestures drive the view
  or scroll the host page ( off in the doc's iframes, `?wheel=0&touch=0` ); a wheel over the view while it is off makes it
  pulse, so that one finds it without opening the menu;
- the point RADIUS slider: the display radius of bare points (Diracs), and a factor on the points that carry their own
  radius ( 1 at the start );
- FIELDS ( `add_points( fields = { name: values } )` ): the points coloured by one of them, chosen in the menu, its
  range over the displayed frame, over all of them, or typed ( robust: the 1st and 99th percentiles ); a sequential
  map, or a diverging one when the field takes both signs ( its automatic range then symmetric: white at 0 ); the
  colours always go from the start of the map at min to its end at max. In the TOP ROW, after the frame bar: the field, the
  scale with its two ends ( typed: the range stays ), a button for the range of this frame / of all of them;
- LAYERS ( `layer = name` on points, edges and images, `add_layer( name, visible )` ): the eye of the top row shows or
  hides each, and the points;
- IMAGES ( `add_image` ): grey levels under everything, in the plane of the first two axes;
- solid faces (adjustable opacity) lit on both sides;
- HIDDEN EDGES: a PURE depth pass (faces written into the z-buffer without touching the color)
  precedes the edge drawing, which gives hidden-part removal EVEN when the faces are not
  displayed. Three modes, key [e]: hidden removed, hidden as a ghost (the classic readable
  wireframe), or everything visible;
- 2D: orthographic camera locked flat, pan/zoom like `points_html`;
- 3D: free rotation (the orientation is a quaternion: no reference plane, so the same gesture
  produces the same motion whatever the starting orientation) + pan + zoom, perspective or
  orthographic projection ([o]). Zoom and rotation AIM at the scene: a ray is cast into the
  primitives already present on the CPU side (`pickWorld`), so that the zoom keeps the point
  under the cursor fixed and the rotation pivots around what the screen shows at its center, at
  the RIGHT depth -- not at the arbitrary depth of the target;
- 4D and more: one chooses the 3 displayed dimensions (X/Y/Z selectors) and fixes the OTHERS with
  a slider. Polytopes (H-representation) are then truly CUT by the chosen hyperplane -- the
  slice remains a polytope, its vertices are re-enumerated in the browser at each slider
  movement --, whereas already triangulated meshes are simply PROJECTED (cutting a mesh makes no
  sense: the intersection of an edge and a hyperplane is a point).

An unbounded polytope is displayed clipped by a bounding box, without which it would have no
vertex to show.

= The ParaView output

`write_vtk`: compressed binary XML, one `.vtu` per frame plus a `.pvd` that gathers them. The
details (and the three choices it settles) are in `vtk_writer`. It needs the vertices of a
polytope, which `polytope` enumerates in Python -- in any dimension.
"""
import base64
import colorsys
import json
import re
import zlib
from pathlib import Path

import numpy as np


#: default colors, assigned in the order of additions (mid-tones: readable on a light background
#: AS WELL AS on a dark one, the page switches from one to the other on the fly)
# THE COLOR SCALE: a hue wheel, TRAVERSED WITH THE GOLDEN RATIO.
#
# Continuous and cyclic, hence with no number of colors to announce -- what matters is not
# how many there are but the STEP between two indices. Advancing one notch must not move the hue
# by a hair (two neighboring cells would be twins): the step is therefore a golden turn, the
# known way to keep any number of indices well separated -- the first 10 as well as the first 100,
# without knowing in advance which of the two we will have.
GOLDEN_STRIDE = 0.6180339887498949      # φ - 1


def scale_color( index ):
    """The color number `index` of the scale. A FUNCTION of the index, and nothing else."""
    index = int( index )
    # two short cycles on saturation and value, on top of the hue: the golden step keeps
    # NEIGHBORING indices far from one another, but two distant indices end up getting
    # close in hue (13 notches, 21 notches...) and those are the ones these cycles separate.
    h = ( index * GOLDEN_STRIDE ) % 1.0
    s = 0.55 + 0.13 * ( index % 3 )
    v = 0.95 - 0.13 * ( index % 2 )
    return colorsys.hsv_to_rgb( h, s, v )


def _np( x, dtype = np.float32 ):
    """A `Tensor` (loom), an array, a list... -> a `np.ndarray` of the requested dtype.

    `np.asarray` suffices for all three: a `Tensor` exposes `__array__`, which already returns its
    DENSE view (capacity padding removed) -- no `.raw[ :n ]` to write here.
    """
    return np.asarray( x, dtype = dtype )


def _rgb( color ):
    """`"#rgb"` / `"#rrggbb"` / `(r, g, b)` in [0,1] -> three floats in [0,1]."""
    if isinstance( color, str ):
        s = color.lstrip( "#" )
        if len( s ) == 3:
            s = "".join( 2 * c for c in s )
        return tuple( int( s[ i : i + 2 ], 16 ) / 255 for i in ( 0, 2, 4 ) )
    return tuple( float( c ) for c in color )[ :3 ]


def _is_pair( color ):
    """a color PER THEME, `( light, dark )`: two colors, not the three components of one"""
    return isinstance( color, ( tuple, list ) ) and len( color ) == 2 and all( isinstance( c, ( str, tuple, list ) ) for c in color )


def _darker( color, factor = 0.72 ):
    if _is_pair( color ):
        return tuple( _darker( c, factor ) for c in color )
    return tuple( c * factor for c in _rgb( color ) )


def _cat( blocks, dtype ):
    """Concatenates blocks (one per addition) into a single flat array of the desired dtype."""
    if not blocks:
        return np.zeros( 0, dtype = dtype )
    return np.concatenate( [ np.asarray( b ).reshape( -1 ) for b in blocks ] ).astype( dtype )


#: while a page is written ( `to_html` ): the compressed BLOBS, which `_b64` fills and refers to as "@index"
_BLOBS = None


def _b64( arr ):
    """an array for the page: its bytes in base64; while `to_html` runs, a reference "@i" to a compressed blob instead"""
    a = np.ascontiguousarray( arr )
    if _BLOBS is None:
        return base64.b64encode( a.tobytes() ).decode( "ascii" )
    _BLOBS.append( _pack( a ) )
    return f"@{ len( _BLOBS ) - 1 }"


def _quantized( a, axis = None ):
    """`( codes uint16, lo, step )`: `a` on 16 bits between its min and max ( per column with `axis = 0` ), NaN as 65535"""
    a = np.asarray( a, dtype = np.float64 )
    ok = np.isfinite( a )
    if not ok.any():
        return np.full( a.shape, 65535, np.uint16 ), 0.0, 1.0
    lo = np.nanmin( np.where( ok, a, np.nan ), axis = axis )
    hi = np.nanmax( np.where( ok, a, np.nan ), axis = axis )
    step = np.where( hi > lo, ( hi - lo ) / 65534, 1.0 )
    q = np.where( ok, np.round( ( np.where( ok, a, lo ) - lo ) / step ), 65535 ).astype( np.uint16 )
    return q, lo, step


def _pack( a ):
    """`{ z, k }`: the bytes of `a` SHUFFLED ( the first byte of every item, then the second, ...: the bytes of floats
    that change little come together ) and deflated ( zlib ), in base64; `k` the size of an item. The page inflates them
    with the `DecompressionStream` of the browser"""
    k = a.dtype.itemsize
    b = a.view( np.uint8 ).reshape( -1, k ).T.tobytes() if k > 1 and a.size else a.tobytes()
    return dict( z = base64.b64encode( zlib.compress( b, 6 ) ).decode( "ascii" ), k = k )


class Visualizer:
    """Collects geometric primitives; the outputs (`write_html`, ...) come afterwards.

    `title` names the scene (the tab, on the HTML side). `depth_colors`: in 3D, the points are colored by their depth in the
    view ( warm in front, cool behind, recomputed as the view turns ) instead of their own color -- what a cloud needs to be
    read in depth ( `?depth=0|1` in the URL and the menu still switch it ).

    `morph = k`: the playback goes from one frame to the next in `k` eased steps, the points of the two frames matched by rank
    when they have as many -- the same diracs moving ( a transport, a descent ) without storing the steps; it holds a moment on
    each frame. `loop`: it starts again, morphing back to the first frame. `autoplay`: it plays as soon as the page opens
    ( `?play=0|1` in the URL wins ).

    The page saves the played axis as an animated GIF ( menu, "save as GIF" ): one image per frame, as the view shows it
    ( zoom, theme ), at most 800 pixels wide ( `?gif_width=` ), the last frame held a moment. `?gif=1` saves it as soon
    as the page opens.

    `frame_axis`: the name of the frame axis, or a tuple of names when the frames vary along several parameters (see
    `new_frame`). `playable`: whether the axis gets a play button -- a bool for all the axes, or one per axis. Either way
    the bar of the axes sits next to the menu button, in plain sight: it is what a reader of the doc has to touch.

    `flat_points`: in 2D, the points are flat discs ( no sphere shading ) -- what a disc of the plane is, a particle of
    a reconstruction for instance.

    `add_curve` puts CURVES beside the scene ( a loss, a metric along a descent ): a panel at the bottom of the page, one
    small chart per curve, with a cursor at the abscissa of the displayed frame; a click in a chart goes to the nearest frame.

    `store = path`: the STORE mode ( see `store.py` ): each frame is written to a file of the directory `path` when the next
    one opens ( `new_frame` ), then forgotten -- the frames of a long run never stay in memory, and a viewer reads them while
    the run goes on. `close()` writes the last one. The shared primitives and the images go to a base written once ( again
    when they change ); the curves are given again whole as they grow ( `clear_curves`, then `add_curve` ). `on_frame`: called
    after each write of the manifest ( a run says there that its view changed ). `viewer`: whether the store holds its
    viewer ( `"copy"` ), only names it ( `"link"` ), or holds it unless it is published ( `"auto"` ), see `bundle.py`.

    `ao`: in 3D, the faces get a BAKED ambient occlusion ( `occlusion.py` ): how open the space is in front of each corner,
    computed when the page or the frame is written, then used by the lighting ( the "baked occlusion" slider, `?bake=` ).
    `True`, or the options of `ambient_occlusion` ( `dict( rays = 64, radius = 0.05 )` ). Each frame is baked with the
    shared faces as occluders; the shared faces with themselves.

    `face_opacity`: the starting value of the opacity slider of the faces ( `?opacity=` in the URL wins ) -- below 1 for a
    density drawn under diracs.
    """

    def __init__( self, title = "sdot", background = None, frame_axis = "time",
                  playable = True, fps = 5.0, frame_value = 0.0, depth_colors = False, morph = 0, loop = False,
                  autoplay = False, face_opacity = 1.0, flat_points = False , field = None, compact = False,
                  store = None, on_frame = None, viewer = "auto", ao = False ):
        self.title      = title
        self.ao         = ao              # baked occlusion of the faces ( 3D ): False, True, or the options of `ambient_occlusion`
        self.face_opacity = float( face_opacity )
        self.flat_points = flat_points
        self.depth_colors = depth_colors
        self.morph      = int( morph )
        self.loop       = loop
        self.autoplay   = autoplay
        self.background = background      # None -> the theme's; a color, or one per theme `( light, dark )`
        self.nb_dims    = None            # deduced from the first addition (size of the position vectors)

        # axis of the FRAMES (see `new_frame`): its name (time, or a parameter), and whether it can
        # be played by itself -- playback only makes sense on an axis that is traversed, which
        # is the case of time and not necessarily of a parameter.
        self.frame_axis = frame_axis
        self.frame_axes = ( frame_axis, ) if isinstance( frame_axis, str ) else tuple( frame_axis )
        self.playable   = playable
        self.fps        = fps

        self._pools    = []               # vertex pools, each [n, d]
        self._bb_pts   = []               # those that FRAME the scene (see `_pool( frames = )`)
        self._keep     = []               # the ORIGINAL arrays: `id()` must stay valid
        self._base     = {}               # id( received array ) -> offset of its pool
        self._size     = {}               # offset -> number of vertices of the pool
        self._nb_verts = 0

        self._colors   = []               # table of distinct colors, as (r, g, b, a)
        self._colors_dark = []            # ... the same in the dark theme ( a color may be given per theme, `( light, dark )` )
        self._col_ids  = {}

        self._poly_v, self._poly_len, self._poly_c = [], [], []   # polygons (free length)
        self._edge_v, self._edge_c                 = [], []
        self._pnt_v,  self._pnt_c,  self._pnt_r    = [], [], []
        self._pnt_m                                = []           # a matrix [ 4 ] per point: an ELLIPSE ( 0: a disc )
        self._pnt_f                                = []           # per chunk of points: `( n, { field name: [ n ] } )`
        self._pnt_l, self._edge_l                  = [], []       # the LAYER of each point / edge ( 0: none )
        self._layers                               = []           # `[ name, visible ]`, layer i + 1
        self._images                               = []           # see `add_image`
        self.field                                 = field        # the field the points are coloured by at the start
        self.compact                               = compact      # positions and fields on 16 bits in the page
        self._hpolys                               = []           # polytopes in H-representation
        self._curves                               = []           # see `add_curve`

        # the next free index of the scale. Reset to zero at each FRAME (see `new_frame`):
        # a color must say WHAT, not HOW MANY were drawn before.
        self._next_color = 0

        # A FRAME only remembers where it starts in the lists above: everything that is
        # added goes into the LAST one, so each frame occupies a contiguous range. Nothing
        # is duplicated, and a frame has no structure of its own to carry.
        self._nb_edges  = 0
        self._nb_points = 0
        # SHARED primitives ( `shared = True` ) are drawn in every frame and stored once: at the HEAD of the lists, the
        # frames' ranges after them ( the marks count the frames' primitives only, the output shifts them )
        self._shared    = dict( poly = 0, edge = 0, pnt = 0 )     # how many primitives
        self._sh_chunks = dict( poly = 0, edge = 0, pnt = 0 )     # how many chunks in the lists of arrays
        self._sh_poly_v = 0                                       # how many indices at the head of `_poly_v`
        self._frames    = [ self._frame_mark( frame_value ) ]       # ( `frame_value`: the abscissa of the first frame )

        # STORE mode ( `store = path`, see `store.py` ): the frames are written as files as they are made, and forgotten.
        # Only the frame being built stays in the lists above; the shared primitives go to a second scene, the BASE,
        # which shares the tables of colours, layers and images with this one.
        self._writer = None
        self._sources = None
        if store is not None:
            from .store import StoreWriter
            self._writer = StoreWriter( store, on_frame = on_frame )
            b = self._base_viz = Visualizer( title = title, compact = compact, ao = ao )
            b._colors, b._colors_dark, b._col_ids, b._layers, b._images = self._colors, self._colors_dark, self._col_ids, self._layers, self._images
            self._base_dirty = False
            self._closed = False
            from .bundle import write_store_page                                 # ( a bootstrap naming a pinned viewer, sdot ADR 0001 )
            write_store_page( store, title, viewer = viewer )                    # ( read through a server: no file:// )
            from .services import declare                                        # ( what its page may ask for, errand ADR 0026 )
            declare( store )
            self._writer.write_manifest( self, force = True )

    # ---- colors -------------------------------------------------------------------------------

    def color_at( self, index ):
        """The color number `index` of the scale -- see `scale_color`.

        This is the entry point to prefer whenever an object has its OWN NUMBER: the cell of
        Dirac `i` is colored `color_at( i )`, and its color then depends only on `i` -- not on the
        order of calls, nor on what was drawn before, nor on the frame. A Dirac keeps its color
        from one descent step to the next, and a neighbor that loses its cell no longer shifts
        anyone.
        """
        return scale_color( index )

    def next_color( self ):
        """The next free color (the one an `add_*` without `color` would take), without advancing."""
        return self.color_at( self._next_color )

    def add_sources( self, kind: str, arrays: dict, meta: dict | None = None ):
        """SOURCES of the frame being built: small data its scene can be REBUILT from ( sdot ADR 0001 §6 ), for the services
        of a store ( `sdot.region` rebuilds the cells of a region from the seeds and weights of a power diagram ). `kind`
        says what they are ( `"sdot.power_diagram"` ); a viewer that does not know it ignores them.

        Kept by stores only ( `sources/<frame>.bin`, named in the frame's entry of the manifest ); a page has no use for
        them. The last call of a frame wins."""
        if self._writer is not None:
            self._sources = ( str( kind ), { k: np.asarray( v ) for k, v in arrays.items() }, dict( meta or { } ) )
        return self

    def reserve_colors( self, nb = 1 ):
        """Reserves `nb` CONSECUTIVE indices on the scale and returns the first.

        A block rather than one by one: what must number the colors of a composite object is
        the rank of its parts WITHIN the object (item `b` of a batched cell takes `base + b`),
        not the number of parts actually drawn -- otherwise an empty cell, which is skipped,
        would shift all the following ones.

        Numbering restarts from zero at each frame (`new_frame`), so that the same object
        drawn at the same rank in each frame keeps the same color there.
        """
        res = self._next_color
        self._next_color += int( nb )
        return res

    def take_color( self ):
        """Reserves ONE index and returns its color -- for an object that has no number of its own.

        To be taken when a single object yields SEVERAL primitives (a cell: its faces, its
        edges, its vertices) and they should have the same color: without this, each `add_*`
        without `color` would consume a new one and the cell would be motley.
        """
        return self.color_at( self.reserve_colors() )

    @staticmethod
    def darker( color, factor = 0.72 ):
        """The same hue, darkened -- enough to derive an edge color from a face color."""
        return _darker( color, factor )

    def _color_id( self, color, opacity ):
        """Index of `(color, opacity)` in the table -- a primitive carries only this index."""
        if color is None:
            color = self.take_color()
        light, dark = color if _is_pair( color ) else ( color, color )
        key = ( ( *_rgb( light ), float( opacity ) ), ( *_rgb( dark ), float( opacity ) ) )
        if key not in self._col_ids:
            self._col_ids[ key ] = len( self._colors )
            self._colors.append( key[ 0 ] )
            self._colors_dark.append( key[ 1 ] )
        return self._col_ids[ key ]

    # ---- vertex pool --------------------------------------------------------------------------

    def _pool( self, positions, frames = True ):
        """Registers `positions` (once) and returns the offset of its pool.

        Sharing is read off the IDENTITY of the received array: two additions that receive the same
        object (the faces and the edges of a cell) refer to the same pool, and the vertices are
        stored only once. This is also where the scene's dimension is decided -- the size
        of the position vectors, nothing else.

        `frames = False` draws without FRAMING: these vertices do not enter the scene's box
        (`bounds`). For what is drawn at a distance that means nothing -- the stub
        of an edge going off to infinity (`Cell.add_to_viz`) -- and which, if counted, would crush
        to a point what we came to look at. It is the FIRST addition of a pool that decides, the
        following ones finding the same offset without coming through here.
        """
        key = id( positions )
        if key in self._base:
            return self._base[ key ]

        pos = _np( positions )
        if pos.ndim == 1:
            pos = pos.reshape( 1, -1 )
        pos = pos.reshape( -1, pos.shape[ -1 ] )
        self._note_dims( pos.shape[ 1 ] )

        self._base[ key ] = base = self._nb_verts
        self._size[ base ] = len( pos )
        self._pools.append( pos )
        if frames:
            self._bb_pts.append( pos )
        self._keep.append( positions )
        self._nb_verts += len( pos )
        return base

    def _note_dims( self, nb_dims ):
        if nb_dims < 2:
            raise ValueError( f"Visualizer: dimension { nb_dims } (at least 2 expected)" )
        if self.nb_dims is None:
            self.nb_dims = int( nb_dims )
        elif int( nb_dims ) != self.nb_dims:
            raise ValueError( f"Visualizer: dimension { nb_dims } incompatible with the scene "
                              f"(already in { self.nb_dims }D)" )

    # ---- frames (time, or parameter) ----------------------------------------------------------

    def _frame_values( self, value ):
        """`value` as one float per frame axis. With several axes: a tuple, or a dict whose missing axes keep the value of
        the previous frame."""
        n = len( self.frame_axes )
        if isinstance( value, dict ):
            prev = self._frames[ -1 ][ "values" ] if getattr( self, "_frames", None ) else ( 0.0, ) * n
            unknown = set( value ) - set( self.frame_axes )
            if unknown:
                raise ValueError( f"Visualizer: unknown frame axes { sorted( unknown ) } ( axes: { self.frame_axes } )" )
            return tuple( float( value.get( a, p ) ) for a, p in zip( self.frame_axes, prev ) )
        vals = tuple( float( v ) for v in np.ravel( value ) )
        if len( vals ) == 1:
            vals = vals * n                                         # a scalar: the same on every axis
        if len( vals ) != n:
            raise ValueError( f"Visualizer: { len( vals ) } frame value(s) for the axes { self.frame_axes }" )
        return vals

    def _frame_mark( self, value ):
        return { "values": self._frame_values( value ),
                 "poly": len( self._poly_len ) - self._shared[ "poly" ], "edge": self._nb_edges,
                 "pnt": self._nb_points, "hpoly": len( self._hpolys ) }

    def new_frame( self, value = None ):
        """Opens a new FRAME: everything added afterwards will be stored in it.

        A frame is a COMPLETE STATE, independent of the others: neither the vertices nor the
        connectivity need to match from one frame to the next (cells appear, disappear,
        change shape -- this is what is needed to replay a descent). They share
        only the vertex pool, and only where the caller passes the SAME array again.

        `value` is the abscissa on the axis (`frame_axis`): an instant, or the value of the parameter.
        By default, the rank of the frame. With several axes: one value per axis, as a tuple, or as a
        dict that only names the axes that change ( the others keep the previous frame's ); by default,
        the rank of the frame on the first axis.

        Color numbering restarts from zero here. This is what keeps an animation from
        flickering: what is redrawn at the same rank in each frame takes its color back,
        instead of taking a new one because the previous frame consumed one.
        """
        self._next_color = 0
        if value is None:
            value = { self.frame_axes[ 0 ]: self.nb_frames } if len( self.frame_axes ) > 1 else self.nb_frames
        if self._writer is not None:
            values = self._frame_values( value )                # ( a dict reads the frame being flushed )
            self._flush()
            self._frames = [ self._frame_mark( values ) ]
            return self
        self._frames.append( self._frame_mark( value ) )
        return self

    @property
    def nb_frames( self ):
        if self._writer is not None:
            return len( self._writer.frames ) + ( 0 if self._closed else 1 )
        return len( self._frames )

    # ---- store mode ---------------------------------------------------------------------------

    def _flush( self ):
        """STORE mode: the frame being built written as a chunk, the base if it changed, the manifest; then the frame
        forgotten ( its pools and primitives ), the colours, layers, images and curves kept."""
        if self._base_dirty:
            self._writer.write_base( self._base_viz )
            self._base_dirty = False
        if self.nb_dims is not None:
            self._writer.write_frame( self, self._frames[ 0 ][ "values" ] )
        self._sources = None
        self._pools, self._bb_pts, self._keep, self._base, self._size, self._nb_verts = [ ], [ ], [ ], { }, { }, 0
        self._poly_v, self._poly_len, self._poly_c = [ ], [ ], [ ]
        self._edge_v, self._edge_c, self._edge_l = [ ], [ ], [ ]
        self._pnt_v, self._pnt_c, self._pnt_r, self._pnt_m, self._pnt_f, self._pnt_l = [ ], [ ], [ ], [ ], [ ], [ ]
        self._hpolys = [ ]
        self._nb_edges = self._nb_points = 0
        self._writer.write_manifest( self )

    def _to_base( self, method, *args, **kwargs ):
        """STORE mode: a shared primitive goes to the base scene ( drawn in every frame, written when it changes )"""
        kwargs[ "shared" ] = False
        if kwargs.get( "color" ) is None:
            kwargs[ "color" ] = self.take_color()             # ( the numbering of THIS frame, as without a store )
        getattr( self._base_viz, method )( *args, **kwargs )
        if self._base_viz.nb_dims is not None:
            self._note_dims( self._base_viz.nb_dims )
        self._base_dirty = True
        return self

    def close( self ):
        """STORE mode: the last frame written, the manifest marked `complete`. Nothing to do otherwise."""
        if self._writer is None or self._closed:
            return self
        if self._base_dirty:
            self._writer.write_base( self._base_viz )
            self._base_dirty = False
        if self.nb_dims is not None:
            self._writer.write_frame( self, self._frames[ 0 ][ "values" ] )
        self._closed = True
        self._writer.write_manifest( self, complete = True )
        return self

    def clear_curves( self ):
        """Removes every curve: in STORE mode, the curves are given again, whole, as they grow ( the manifest carries
        them, rewritten after each frame )."""
        self._curves = [ ]
        return self

    def _ao_options( self ) -> dict:
        return dict( self.ao ) if isinstance( self.ao, dict ) else { }

    def _baked_occlusion( self ) -> np.ndarray:
        """the occlusion codes of every corner of `_poly_v` ( uint8, 255 = open ): the shared faces with themselves, each
        frame's with the shared ones"""
        from .occlusion import ambient_occlusion, as_codes
        pos = self.positions
        pv = np.asarray( self._poly_v, np.int64 )
        off = np.concatenate( [ [ 0 ], np.cumsum( self._poly_len ) ] ).astype( np.int64 )
        codes = np.full( len( pv ), 255, np.uint8 )
        sp = self._shared[ "poly" ]
        opts = self._ao_options()

        def bake( polys ):                                  # ( polygon indices ) -> the occlusion of their corners, in order
            polys = np.asarray( polys, np.int64 )
            lens = off[ polys + 1 ] - off[ polys ]
            corners = np.concatenate( [ np.arange( off[ p ], off[ p + 1 ] ) for p in polys ] ) if len( polys ) else np.zeros( 0, np.int64 )
            return corners, ambient_occlusion( pos, pv[ corners ], np.concatenate( [ [ 0 ], np.cumsum( lens ) ] ), **opts )

        if sp:
            c, a = bake( np.arange( sp ) )
            codes[ c ] = as_codes( a )
        r = self._frame_ranges( "poly" )
        for i in range( len( r ) - 1 ):
            if r[ i + 1 ] > r[ i ]:
                c, a = bake( np.r_[ np.arange( sp ), np.arange( r[ i ], r[ i + 1 ] ) ] )
                own = len( c ) - int( off[ r[ i + 1 ] ] - off[ r[ i ] ] )
                codes[ c[ own: ] ] = as_codes( a[ own: ] )
        return codes

    def _frame_ranges( self, key ):
        """The bounds of the frames over one of the primitive lists: `nb_frames + 1` integers ( after the shared
        primitives, which come first )."""
        total = { "poly": len( self._poly_len ) - self._shared[ "poly" ], "edge": self._nb_edges,
                  "pnt": self._nb_points, "hpoly": len( self._hpolys ) }[ key ]
        return np.array( [ f[ key ] for f in self._frames ] + [ total ], dtype = np.int32 ) + self._shared.get( key, 0 )

    def _store( self, key, lists, items, nb, shared ):
        """Appends one chunk ( `items`, one per list of `lists` ) of `nb` primitives to the frame being built, or, `shared`,
        to the shared ones."""
        if shared:
            i = self._sh_chunks[ key ]
            for l, x in zip( lists, items ):
                l.insert( i, x )
            self._sh_chunks[ key ] += 1
            self._shared[ key ] += nb
        else:
            for l, x in zip( lists, items ):
                l.append( x )

    # ---- primitives ---------------------------------------------------------------------------

    def add( self, obj, **kwargs ):
        """Adds an object that knows how to draw itself (it exposes `add_to_viz( viz, ... )`).

        The visualizer knows no domain type: it is the object that translates its geometry into
        primitives (`Cell.add_to_viz` -> `add_faces` / `add_edges` / `add_polytope`).
        """
        obj.add_to_viz( self, **kwargs )
        return self

    def note_bounds( self, positions ):
        """Extends the scene's box to `positions`, WITHOUT drawing anything.

        For what is displayed otherwise than by vertices: a polytope given as half-spaces
        has no readable extent, whereas the object sending it often knows it (a
        cell keeps its vertices) -- and it is this box that frames the camera, gives the cut sliders
        their range and sets the clipping of unbounded polytopes.
        """
        self._pool( positions )
        return self

    def add_curve( self, name, y, x = None, log_y = False, color = None, visible = True ):
        """A curve drawn beside the scene, in the panel of the curves. `y`: its values; `x`: their abscissas ON THE FIRST
        FRAME AXIS ( an iteration, a time: the unit of `new_frame( value )` ), `0, 1, 2, ...` by default. A curve may have
        more samples than there are frames ( every iteration of a descent that keeps one frame out of ten ). The cursor of
        the chart follows the displayed frame, and shows the value of the nearest sample. `log_y`: a log scale ( the
        values `<= 0` are then left out ). Several curves with the same `name` share a chart ( `color` tells them
        apart: one per theme as a pair, the scale of the scene by default ). Each chart is a window over the scene, that
        one moves by its title and resizes by its corner; the eye of the top row shows or hides it. `visible`: at the
        start ( a chart is shown when one of its curves is )."""
        y = np.asarray( y, dtype = float ).reshape( -1 )
        x = np.arange( len( y ), dtype = float ) if x is None else np.asarray( x, dtype = float ).reshape( -1 )
        if len( x ) != len( y ):
            raise ValueError( f"add_curve: { len( x ) } abscissas for { len( y ) } values" )
        ok = np.isfinite( y ) & np.isfinite( x ) & ( ( y > 0 ) if log_y else True )
        nb = sum( c[ "name" ] == name for c in self._curves )
        col = scale_color( nb ) if color is None else color
        pair = col if _is_pair( col ) else ( col, col )
        self._curves.append( dict( name = str( name ), x = [ float( v ) for v in x[ ok ] ], y = [ float( v ) for v in y[ ok ] ],
                                   log = bool( log_y ), color = [ list( _rgb( c ) ) for c in pair ], visible = bool( visible ) ) )
        return self

    def add_points( self, positions, radius = 0.0, color = None, opacity = 1.0, frames = True, matrices = None, shared = False,
                    fields = None, layer = None ):
        """`positions`: `[n, d]`. `radius`: the WORLD radius of the points (a scalar, or one radius
        per point). `0` (default) = the point has no size of its own (a Dirac): it is the page's
        slider that sets its display radius -- same convention as `points_html`.

        `matrices`: `[n, 2, 2]`, ELLIPSES ( 2D scenes ): `M` maps the unit disc onto the ellipse, in the plane of the first two
        viewed axes -- one quad per ellipse, deformed on the card, instead of an outline of vertices. Their radius
        ( for the aim, and the slider, which scales them all ) is their largest half-axis unless `radius` is given.

        `shared`: drawn in EVERY frame, stored once ( an outline of the object under a descent, a sample of the truth ).

        `fields`: `{ name: [n] }`, VALUES per point. The page offers to colour the points by one of them ( a colour map,
        its range over all the frames, over the displayed one, or typed ); a point without the chosen field keeps its
        color. `layer`: see `add_layer`.
        """
        if shared and self._writer is not None:
            return self._to_base( "add_points", positions, radius = radius, color = color, opacity = opacity, frames = frames,
                                  matrices = matrices, fields = fields, layer = layer )
        base = self._pool( positions, frames )
        n = self._pool_len( base )
        if n == 0:
            return self
        mat = np.zeros( ( n, 4 ), np.float32 ) if matrices is None else _np( matrices ).reshape( n, 4 )
        if matrices is not None and np.isscalar( radius ) and radius == 0:
            radius = np.linalg.norm( mat.reshape( n, 2, 2 ), ord = 2, axis = ( 1, 2 ) )     # the largest half-axis
        rad = np.full( n, float( radius ), dtype = np.float32 ) \
            if np.isscalar( radius ) else _np( radius ).reshape( -1 )
        if len( rad ) != n:
            raise ValueError( f"radius: { len( rad ) } radii for { n } points" )
        ci = self._color_id( color, opacity )
        flds = { str( k ): _np( v ).reshape( -1 ) for k, v in ( fields or { } ).items() }
        for k, v in flds.items():
            if len( v ) != n:
                raise ValueError( f"fields: { len( v ) } values of { k } for { n } points" )
        self._store( "pnt", ( self._pnt_v, self._pnt_r, self._pnt_c, self._pnt_m, self._pnt_f, self._pnt_l ),
                     ( np.arange( base, base + n, dtype = np.int32 ), rad, np.full( n, ci, dtype = np.uint16 ), mat, ( n, flds ),
                       np.full( n, self._layer_id( layer ), dtype = np.uint8 ) ), n, shared )
        if not shared:
            self._nb_points += n
        return self

    def add_edges( self, positions, edges = None, color = None, opacity = 1.0, closed = False,
                   dashed = False, nb_dashes = 7, frames = True, shared = False, layer = None ):
        """`positions`: `[n, d]`. `edges`: `[m, 2]` of indices, or `None` to connect
        consecutive points (`closed` then closes the loop -- the case of a polygon given in
        cyclic order, which is what a 2D cell yields).

        `dashed` draws DOTTED lines. The dash is not a style carried all the way to the output: each
        segment is simply SPLIT here, into `nb_dashes` pieces separated by as many gaps, and
        what goes out afterwards is a sequence of ordinary edges. Two reasons: the rendering (hand-written
        WebGL) has no line pattern, and neither does VTK -- split geometry is the
        only thing BOTH outputs know how to display identically. The price is that the length of a
        dash is in WORLD units, not in pixels: it is a fixed number of dashes per edge, so
        the density does not change with zoom.

        `shared`: drawn in EVERY frame, stored once. `layer`: see `add_layer`.
        """
        if shared and self._writer is not None:
            return self._to_base( "add_edges", positions, edges, color = color, opacity = opacity, closed = closed, dashed = dashed,
                                  nb_dashes = nb_dashes, frames = frames, layer = layer )
        base = self._pool( positions, frames )
        n = self._pool_len( base )
        if edges is None:
            if n < 2:
                return self
            idx = np.stack( [ np.arange( n - 1 ), np.arange( 1, n ) ], axis = 1 )
            if closed:
                idx = np.concatenate( [ idx, [ [ n - 1, 0 ] ] ], axis = 0 )
        else:
            idx = np.asarray( edges, dtype = np.int64 ).reshape( -1, 2 )
        if len( idx ) == 0:
            return self
        if dashed:
            return self._add_dashes( positions, idx, color, opacity, nb_dashes, frames, shared, layer )
        ci = self._color_id( color, opacity )
        self._store( "edge", ( self._edge_v, self._edge_c, self._edge_l ), ( idx.astype( np.int32 ) + base, np.full( len( idx ), ci, dtype = np.uint16 ),
                                                                            np.full( len( idx ), self._layer_id( layer ), dtype = np.uint8 ) ), len( idx ), shared )
        if not shared:
            self._nb_edges += len( idx )
        return self

    def _add_dashes( self, positions, idx, color, opacity, nb_dashes, frames = True, shared = False, layer = None ):
        """The `idx` segments of `positions`, dotted: a SEPARATE vertex pool, holding the
        ends of the dashes, and ordinary edges on it (see `add_edges( dashed = True )`).
        """
        p = _np( positions )
        a, b = p[ idx[ :, 0 ] ], p[ idx[ :, 1 ] ]

        # `2 k - 1` intervals -> `k` dashes separated by `k - 1` gaps, and an edge that STARTS
        # and ENDS with a dash: its two ends thus remain visible where they matter
        # (the real vertex on one side, the vanishing direction on the other).
        nb_dashes = max( 1, int( nb_dashes ) )
        k = 2 * nb_dashes - 1
        t = np.arange( k + 1, dtype = np.float32 ) / k
        pts = a[ :, None, : ] + ( b - a )[ :, None, : ] * t[ None, :, None ]

        starts = ( np.arange( len( idx ) )[ :, None ] * ( k + 1 )
                 + np.arange( 0, k, 2 )[ None, : ] )
        seg = np.stack( [ starts, starts + 1 ], axis = -1 ).reshape( -1, 2 )
        return self.add_edges( pts.reshape( -1, p.shape[ 1 ] ), seg, color = color,
                               opacity = opacity, frames = frames, shared = shared, layer = layer )

    def add_layer( self, name, visible = True ):
        """A LAYER: what is added with `layer = name` ( points, edges, images ) is shown or hidden at once, from a box of
        the page's menu. `visible`: at the start. A layer named in an addition without this call is visible."""
        self._layer_id( name )
        self._layers[ self._layer_id( name ) - 1 ][ 1 ] = bool( visible )
        return self

    def _layer_id( self, name ):
        if name is None:
            return 0
        for i, ( n, _ ) in enumerate( self._layers ):
            if n == name:
                return i + 1
        if len( self._layers ) >= 255:
            raise ValueError( "Visualizer: 255 layers at most" )
        self._layers.append( [ str( name ), True ] )
        return len( self._layers )

    def add_image( self, values, lo, hi, layer = None, opacity = 1.0, vmin = None, vmax = None ):
        """An IMAGE in the plane of the first two viewed axes ( 2D scenes ), under everything else and in every frame:
        `values [ nx, ny ]` ( `values[ i, j ]` at the i-th column along x, the j-th row along y ) over the rectangle from
        `lo` to `hi`, in grey from `vmin` ( black ) to `vmax` ( white ) -- by default the 0.5 and 99.5 percentiles. Stored
        on 8 bits. `layer`: see `add_layer`."""
        v = np.asarray( values, dtype = np.float64 )
        if v.ndim != 2:
            raise ValueError( "add_image: values [ nx, ny ]" )
        a = np.percentile( v, 0.5 ) if vmin is None else float( vmin )
        b = np.percentile( v, 99.5 ) if vmax is None else float( vmax )
        g = np.clip( ( v - a ) / ( b - a if b > a else 1.0 ), 0, 1 )
        self._images.append( dict( nx = v.shape[ 0 ], ny = v.shape[ 1 ], lo = [ float( x ) for x in lo[ :2 ] ], hi = [ float( x ) for x in hi[ :2 ] ],
                                   layer = self._layer_id( layer ), opacity = float( opacity ), vmin = float( a ), vmax = float( b ),
                                   data = _b64( np.ascontiguousarray( np.round( 255 * g ).astype( np.uint8 ).T ) ) ) )
        if self._writer is not None:
            self._base_dirty = True                            # ( an image is in the base: drawn in every frame )
        return self

    def _point_fields( self ):
        """`[ { name, data } ]`: each field over all the points ( NaN where a chunk does not have it )"""
        names = [ ]
        for _, f in self._pnt_f:
            names += [ k for k in f if k not in names ]
        out = [ ]
        for k in names:
            v = np.concatenate( [ f[ k ] if k in f else np.full( n, np.nan, np.float32 ) for n, f in self._pnt_f ] or [ np.zeros( 0, np.float32 ) ] )
            if self.compact:                                   # ( 16 bits: what a colour needs )
                q, lo, step = _quantized( v )
                out.append( dict( name = k, data = _b64( q ), lo = float( lo ), step = float( step ) ) )
            else:
                out.append( dict( name = k, data = _b64( v.astype( np.float32 ) ) ) )
        return out

    def add_faces( self, positions, faces, color = None, opacity = 1.0, frames = True, shared = False ):
        """`positions`: `[n, d]`. `faces`: a sequence of POLYGONS (lists of indices, free lengths).
        They are kept AS IS -- it is the output that decides whether to triangulate them.
        `shared`: drawn in EVERY frame, stored once.
        """
        if shared and self._writer is not None:
            return self._to_base( "add_faces", positions, faces, color = color, opacity = opacity, frames = frames )
        base = self._pool( positions, frames )
        ci = self._color_id( color, opacity )
        verts, lens = [], []
        for f in faces:
            f = [ int( i ) + base for i in f ]
            if len( f ) < 3:
                continue
            verts += f
            lens.append( len( f ) )
        if shared:
            self._poly_v[ self._sh_poly_v : self._sh_poly_v ] = verts
            self._poly_len[ self._shared[ "poly" ] : self._shared[ "poly" ] ] = lens
            self._sh_poly_v += len( verts )
        else:
            self._poly_v += verts
            self._poly_len += lens
        self._store( "poly", ( self._poly_c, ), ( np.full( len( lens ), ci, dtype = np.uint16 ), ), 0, shared )
        if shared:
            self._shared[ "poly" ] += len( lens )
        return self

    def add_polytope( self, cut_directions, cut_offsets, color = None, opacity = 1.0,
                      edge_color = None, nb_dims = None, edges = True ):
        """Adds a polytope through its H-representation: `{ x : dir_i . x <= off_i }`.

        This is the form to give when the V-representation is NOT enough: in dimension > 3 (the
        page shows a 3D SLICE of it, and cutting half-spaces yields half-spaces again, hence
        a polytope, re-enumerated in the browser at each slider movement), or for an
        UNBOUNDED polytope (no vertex to enumerate on the Python side -- the page clips it with a
        bounding box and shows what remains).

        `edges = False` keeps only the FACES. For a caller who draws the edges itself and
        knows more than the enumeration: that of an unbounded polytope stops at the scene's
        box, so it would render an edge going off to infinity as solid all the way to the border -- where
        `Cell.add_to_viz`, for its part, knows it is truncated and makes it dotted.
        """
        dirs = _np( cut_directions )
        dirs = dirs.reshape( -1, dirs.shape[ -1 ] )
        offs = _np( cut_offsets ).reshape( -1 )
        if len( dirs ) != len( offs ):
            raise ValueError( f"polytope: { len( dirs ) } directions for { len( offs ) } offsets" )
        self._note_dims( nb_dims if nb_dims is not None else dirs.shape[ 1 ] )

        if color is None:
            color = self.take_color()
        self._hpolys.append( {
            "dirs": _b64( dirs.astype( np.float32 ) ),
            "offs": _b64( offs.astype( np.float32 ) ),
            "nb"  : int( len( dirs ) ),
            "fcol": [ *_rgb( color ), float( opacity ) ],
            "ecol": [ *_rgb( edge_color if edge_color is not None else _darker( color ) ), 1.0 ],
            "edg" : bool( edges ),
        } )
        return self

    # ---- reading the scene --------------------------------------------------------------------

    def _pool_len( self, base ):
        """The number of vertices of the pool starting at `base`."""
        return self._size[ base ]

    @property
    def positions( self ):
        """The vertex pool of the whole scene, `[nb_verts, d]` -- the primitives' indices
        refer to it."""
        if not self._pools:
            return np.zeros( ( 0, self.nb_dims or 2 ), dtype = np.float32 )
        return np.concatenate( self._pools, axis = 0 )

    @property
    def polygons( self ):
        """The faces, as a list of lists of indices (free lengths, not triangulated)."""
        res, o = [], 0
        for n in self._poly_len:
            res.append( self._poly_v[ o : o + n ] )
            o += n
        return res

    @property
    def edges( self ):
        """The edges, `[m, 2]` of indices."""
        return _cat( self._edge_v, np.int32 ).reshape( -1, 2 )

    @property
    def points( self ):
        """The points, `[k]` of indices."""
        return _cat( self._pnt_v, np.int32 )

    @property
    def point_radii( self ):
        """The world radius of each point (0 = no size of its own, cf. `add_points`)."""
        return _cat( self._pnt_r, np.float32 )

    @property
    def point_matrices( self ):
        """The matrix of each point, `[k, 4]` ( `M` row by row, mapping the unit disc onto its ellipse; 0: a disc )."""
        return _cat( self._pnt_m, np.float32 ).reshape( -1, 4 )

    @property
    def colors( self ):
        """The table of distinct colors, as `(r, g, b, a)` -- the primitives refer to it."""
        return list( self._colors )

    @property
    def polygon_colors( self ):
        """The color index of each polygon, in `colors`."""
        return _cat( self._poly_c, np.uint16 )

    @property
    def edge_colors( self ):
        """The color index of each edge, in `colors`."""
        return _cat( self._edge_c, np.uint16 )

    @property
    def point_colors( self ):
        """The color index of each point, in `colors`."""
        return _cat( self._pnt_c, np.uint16 )

    def frame( self, i ):
        """The primitives of frame `i`, already sliced -- enough to write an output without
        knowing anything about the internal slicing.

        Returns a dict: `value` ( a float, a tuple with several frame axes ), `polygons`, `polygon_colors`, `edges`, `edge_colors`, `points`,
        `point_colors`, `point_radii`, `point_matrices`, `polytopes`. The indices refer to the `positions` pool,
        common to ALL frames. The shared primitives come first.
        """
        def rng( key ):
            r = self._frame_ranges( key )
            return int( r[ i ] ), int( r[ i + 1 ] )

        p0, p1 = rng( "poly" )
        e0, e1 = rng( "edge" )
        v0, v1 = rng( "pnt" )
        h0, h1 = rng( "hpoly" )
        sp, se, sv = self._shared[ "poly" ], self._shared[ "edge" ], self._shared[ "pnt" ]
        both = lambda a, s, i0, i1: np.concatenate( [ a[ : s ], a[ i0 : i1 ] ] )
        return {
            "value"         : self._frames[ i ][ "values" ] if len( self.frame_axes ) > 1 else self._frames[ i ][ "values" ][ 0 ],
            "polygons"      : self.polygons[ : sp ] + self.polygons[ p0 : p1 ],
            "polygon_colors": both( self.polygon_colors, sp, p0, p1 ),
            "edges"         : both( self.edges, se, e0, e1 ),
            "edge_colors"   : both( self.edge_colors, se, e0, e1 ),
            "points"        : both( self.points, sv, v0, v1 ),
            "point_colors"  : both( self.point_colors, sv, v0, v1 ),
            "point_radii"   : both( self.point_radii, sv, v0, v1 ),
            "point_matrices": both( self.point_matrices, sv, v0, v1 ),
            "polytopes"     : self.polytopes[ h0 : h1 ],
        }

    @property
    def polytopes( self ):
        """The polytopes in H-representation, `[ ( dirs [c, d], offs [c], rgba, edges ), ... ]`.

        They have NO vertices: an output that wants some (VTK) must enumerate them itself,
        as the HTML page does at each slice.
        """
        res = []
        for h in self._hpolys:
            dirs = np.frombuffer( base64.b64decode( h[ "dirs" ] ), np.float32 ).reshape( -1, self.nb_dims )
            offs = np.frombuffer( base64.b64decode( h[ "offs" ] ), np.float32 )
            res.append( ( dirs, offs, tuple( h[ "fcol" ] ), h[ "edg" ] ) )
        return res

    def bounds( self, pad_ratio = 1.0 ):
        """Bounding box `[ [lo, hi] ] * d`, dimension by dimension.

        It frames the camera AND gives the cut sliders their range. A polytope has no
        vertices to measure: it is bounded by the distance of its planes to the origin, which is
        rough but of the right order of magnitude (and moot as soon as a pool is there as well).
        """
        d = self.nb_dims
        # what FRAMES, not everything that is stored (see `_pool( frames = )`). A scene where
        # everything gave up framing falls back on its vertices: better a box that is too big than
        # no box at all.
        pts = self._bb_pts or self._pools
        if pts:
            allp = np.concatenate( pts, axis = 0 )
            lo, hi = allp.min( axis = 0 ), allp.max( axis = 0 )
        else:
            r = 1.0
            for h in self._hpolys:
                dirs = np.frombuffer( base64.b64decode( h[ "dirs" ] ), dtype = np.float32 ).reshape( -1, d )
                offs = np.frombuffer( base64.b64decode( h[ "offs" ] ), dtype = np.float32 )
                nrm  = np.linalg.norm( dirs, axis = 1 )
                ok   = nrm > 1e-12
                if ok.any():
                    r = max( r, float( np.abs( offs[ ok ] / nrm[ ok ] ).max() ) )
            lo, hi = np.full( d, -r, np.float32 ), np.full( d, r, np.float32 )

        span = np.maximum( hi - lo, 1e-9 )
        pad  = 0.05 * span.max() * pad_ratio
        return [ [ float( a - pad ), float( b + pad ) ] for a, b in zip( lo, hi ) ]

    # ---- sorties ------------------------------------------------------------------------------

    def write_vtk( self, filename, axes = ( 0, 1, 2 ), pvd = False ):
        """Writes the scene for ParaView (compressed binary XML). Returns the path written.

        A single frame -> one `.vtu`. Several -> one `.vtu` per frame plus a `.pvd` that
        gathers them with their abscissa on the axis (the `.pvd` is what one opens, and it is what
        is returned). `pvd = True` asks for the `.pvd` even for a single frame. `axes` chooses the 3 dimensions written as GEOMETRY; beyond 3, the other
        coordinates go out as point data, so that ParaView does its slices itself.
        """
        from .vtk_writer import write_vtk
        return write_vtk( self, filename, axes = axes, pvd = pvd )

    def write_html( self, filename ):
        """Writes a self-contained HTML page (see the module header). Returns the path written."""
        path = Path( filename )
        html = self.to_html()

        from .announce import write_page
        return write_page( path, html )

    def to_html( self, wheel = True, touch = True ):
        """The self-contained HTML page, as a string. `wheel` / `touch` say whether the wheel and
        the touch gestures drive the view by default (`?wheel=0` / `?touch=0` in the URL still
        win) -- off when the page is embedded in a host page that must keep scrolling."""
        global _BLOBS
        if self._writer is not None:
            raise ValueError( "Visualizer: a store ( `store =` ) is not a page: its frames are files, read by `sviz open`" )
        if self.nb_dims is None:
            raise ValueError( "Visualizer: nothing to display (no primitive added)" )
        _BLOBS = [ ]
        try:
            return self._to_html( wheel, touch )
        finally:
            _BLOBS = None

    def _positions_b64( self ):
        """the pool: float32, or with `compact`, 16 bits per coordinate over the box of the scene ( "@i|lo,..|step,.." )"""
        pos = self.positions
        if not self.compact or not pos.size:
            return _b64( pos.astype( np.float32 ) )
        q, lo, step = _quantized( pos, axis = 0 )
        return _b64( q ) + "|" + ",".join( map( repr, np.atleast_1d( lo ).tolist() ) ) + "|" + ",".join( map( repr, np.atleast_1d( step ).tolist() ) )

    def _to_html( self, wheel, touch ):

        d       = self.nb_dims
        bounds  = self.bounds()
        span    = max( hi - lo for lo, hi in bounds )
        r0      = span / 300                                    # starting radius of bare points

        poly_off = np.concatenate( [ [ 0 ], np.cumsum( self._poly_len ) ] ).astype( np.int32 )

        subs = {
            "__TITLE__"   : self.title,
            "__D__"       : str( d ),
            "__BOUNDS__"  : json.dumps( bounds ),
            "__TIGHT__"   : json.dumps( self.bounds( pad_ratio = 0.0 ) ),
            "__POS__"     : self._positions_b64(),
            "__COLORS__"  : json.dumps( [ list( c ) for c in self._colors ] ),
            "__COLORS_DARK__": json.dumps( [ list( c ) for c in self._colors_dark ] ),
            "__POLY_V__"  : _b64( np.asarray( self._poly_v, dtype = np.int32 ) ),
            "__POLY_OFF__": _b64( poly_off ),
            "__POLY_C__"  : _b64( self.polygon_colors ),
            "__POLY_AO__" : _b64( self._baked_occlusion() ) if self.ao and self.nb_dims == 3 else "",
            "__EDG_V__"   : _b64( self.edges.reshape( -1 ) ),
            "__EDG_C__"   : _b64( self.edge_colors ),
            "__PNT_V__"   : _b64( self.points ),
            "__PNT_C__"   : _b64( self.point_colors ),
            "__PNT_R__"   : _b64( self.point_radii ),
            "__PNT_M__"   : _b64( self.point_matrices.reshape( -1 ) ),
            "__PNT_L__"   : _b64( _cat( self._pnt_l, np.uint8 ) ),
            "__EDG_L__"   : _b64( _cat( self._edge_l, np.uint8 ) ),
            "__LAYERS__"  : json.dumps( self._layers ),
            "__FIELDS__"  : json.dumps( self._point_fields() ),
            "__IMAGES__"  : json.dumps( self._images ),
            "__FIELD0__"  : json.dumps( self.field ),
            "__SHARED__"  : json.dumps( self._shared ),
            "__HPOLY__"   : json.dumps( self._hpolys ),
            "__CURVES__"  : json.dumps( self._curves ),
            "__FR_VAL__"  : _b64( np.array( [ f[ "values" ] for f in self._frames ], np.float32 ) ),
            "__FR_POLY__" : _b64( self._frame_ranges( "poly" ) ),
            "__FR_EDG__"  : _b64( self._frame_ranges( "edge" ) ),
            "__FR_PNT__"  : _b64( self._frame_ranges( "pnt" ) ),
            "__FR_HP__"   : _b64( self._frame_ranges( "hpoly" ) ),
            "__AXES__"    : json.dumps( list( self.frame_axes ) ),
            "__PLAYABLE__": json.dumps( [ bool( p ) for p in ( self.playable if isinstance( self.playable, ( tuple, list ) )
                                                              else [ self.playable ] * len( self.frame_axes ) ) ] ),
            "__FPS__"     : repr( float( self.fps ) ),
            "__R0__"      : repr( float( r0 ) ),
            "__RMIN__"    : repr( float( r0 / 30 ) ),
            "__RMAX__"    : repr( float( r0 * 60 ) ),
            "__BG__"      : json.dumps( [ None if c is None else list( _rgb( c ) ) for c in
                                        ( self.background if _is_pair( self.background ) else ( self.background, self.background ) ) ] ),
            "__WHEEL__"   : "true" if wheel else "false",
            "__TOUCH__"   : "true" if touch else "false",
            "__DEPTHCOL__": "true" if self.depth_colors else "false",
            "__FACE_OP__" : repr( self.face_opacity ),
            "__MORPH__"   : str( self.morph ),
            "__LOOP__"    : "true" if self.loop else "false",
            "__FLATPTS__" : "1.0" if self.flat_points else "0.0",
            "__AUTOPLAY__": "true" if self.autoplay else "false",
            "__STORE_MODE__": "false",
            "__STORE_FILES__": "null",
        }
        subs[ "__BLOBS__" ] = json.dumps( _BLOBS )                     # ( last: the substitutions above fill it )
        return re.sub( r"__[A-Z0-9_]+__", lambda m: subs.get( m.group( 0 ), m.group( 0 ) ), _HTML )


def store_page( title = "sdot", files = None ):
    """The page of a STORE ( `index.html` in its directory, see `store.py` ): the page of `to_html`, with no data in it. Its
    loader reads the manifest and the chunks beside it, and fills `STORE` with what the self-contained page has inlined:
    each placeholder of the template becomes a member of `STORE` ( `STORE.POS` for the pool... ).

    `files`: `{ name: text }`, the files of the store INLINED ( the manifest as it is, the chunks in base64 ): a page that reads
    them there instead of fetching them -- the self-contained export of a store ( `sviz export --html` )."""
    import html

    inlined = "null" if files is None else json.dumps( files, separators = ( ",", ":" ) ).replace( "</", "<\\/" )

    def sub( m ):
        name = m.group( 1 ) or m.group( 2 )
        return { "TITLE": html.escape( title ), "BLOBS": "[]", "STORE_MODE": "true", "STORE_FILES": inlined }.get( name, "STORE." + name )

    return re.sub( r'"__([A-Z0-9_]+)__"|__([A-Z0-9_]+)__', sub, _HTML )


_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  /* the page: the SCENE takes the whole window. Over it float, translucent: the frame AXES ( top left, one pill each ),
     the TOOLBAR ( top right: Render, one icon per panel ), the PANELS opened by those icons ( stacked under it ), the
     colour SCALE ( bottom left ) and the curve WINDOWS. Axes and the scale can be taken out of their place and moved
     by their grip; a double-click on the grip puts them back. Colours per theme, as variables. */
  :root {
    --bg: #ffffff; --glass: rgba(250,250,250,0.62); --pane: rgba(250,250,250,0.80); --raised: #ffffff; --well: rgba(0,0,0,0.05);
    --line: rgba(0,0,0,0.12); --soft: rgba(0,0,0,0.06); --text: #1d1d1f; --muted: #5f6368; --accent: #1f6fd1; --accent-ink: #174f96;
    --accent-soft: rgba(31,111,209,0.16); --sel: rgba(0,0,0,0.10); --ok: #2e9e57; --warn: #c27a16; --chart: rgba(255,255,255,0.55);
    --shadow: 0 2px 12px rgba(0,0,0,0.12);
  }
  body.dark {
    --bg: #1a1a1a; --glass: rgba(28,28,30,0.55); --pane: rgba(28,28,30,0.74); --raised: #242426; --well: rgba(0,0,0,0.30);
    --line: rgba(255,255,255,0.14); --soft: rgba(255,255,255,0.07); --text: #ececec; --muted: #b0b0b5; --accent: #1f6fd1;
    --accent-ink: #8ab4f8; --accent-soft: rgba(138,180,248,0.22); --sel: rgba(255,255,255,0.16); --ok: #4cc274; --warn: #f0b45a;
    --chart: rgba(28,28,30,0.45); --shadow: 0 6px 24px rgba(0,0,0,0.40);
  }
  html, body { margin: 0; padding: 0; overflow: hidden; background: var(--bg); color: var(--text); }
  body { font-family: Inter, system-ui, -apple-system, "Segoe UI", sans-serif; font-size: 13px; }
  canvas { display: block; }
  button, select, input { font: inherit; color: inherit; }
  button { cursor: pointer; }
  .hidden { display: none !important; }
  .muted { color: var(--muted); }
  .glass { background: var(--glass); backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); }
  .ibtn { width: 30px; height: 30px; padding: 0; border-radius: 7px; border: none; background: transparent; position: relative;
          display: inline-flex; align-items: center; justify-content: center; flex: none; text-decoration: none; color: inherit; }
  .ibtn svg { width: 16px; height: 16px; }
  .ibtn.on { background: var(--accent-soft); color: var(--accent-ink); }
  .ibtn .badge { position: absolute; right: 4px; top: 4px; width: 7px; height: 7px; border-radius: 50%; box-shadow: 0 0 0 2px var(--raised); display: none; }
  .ibtn .badge.ok { display: block; background: var(--ok); } .ibtn .badge.warn { display: block; background: var(--warn); }
  .btn { height: 26px; padding: 0 10px; border-radius: 6px; border: 1px solid var(--line); background: transparent; font-size: 12px;
         display: inline-flex; align-items: center; gap: 6px; text-decoration: none; color: inherit; box-sizing: border-box; }
  .btn:disabled { opacity: 0.5; cursor: default; }
  .btn.primary { border: none; background: var(--accent); color: #fff; font-weight: 500; }
  .seg { display: grid; grid-auto-flow: column; grid-auto-columns: minmax(0, 1fr); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
  .seg button { height: 24px; border: none; background: transparent; font-size: 12px; padding: 0 4px; }
  .seg button.on { background: var(--sel); font-weight: 500; }
  .grip { width: 14px; height: 18px; flex: none; cursor: grab; color: var(--muted); display: flex; align-items: center; justify-content: center; touch-action: none; }
  .grip svg { width: 10px; height: 14px; }
  .pop { position: fixed; z-index: 8; padding: 6px; border-radius: 10px; background: var(--raised); border: 1px solid var(--line); box-shadow: var(--shadow); }
  .menu { display: flex; flex-direction: column; gap: 2px; min-width: 250px; }
  .menu > button { display: grid; grid-template-columns: 22px minmax(0, 1fr); gap: 2px 10px; align-items: center; text-align: left;
                   padding: 7px 10px; border: none; border-radius: 6px; background: transparent; }
  .menu > button:hover { background: var(--soft); }
  .menu > button.on { background: var(--accent-soft); }
  .menu > button svg { width: 18px; height: 18px; }
  .menu > button small { grid-column: 2; color: var(--muted); font-size: 12px; }
  .menu hr { border: none; height: 1px; background: var(--line); margin: 4px 6px; }
  code.cmd { display: block; padding: 7px 9px; border-radius: 6px; background: var(--well); border: 1px solid var(--line);
             font: 11.5px/1.5 ui-monospace, "JetBrains Mono", Menlo, monospace; user-select: all; white-space: pre-wrap; word-break: break-all; }

  #view { position: fixed; inset: 0; overflow: hidden; }
  #renderImg { position: absolute; left: 0; top: 0; pointer-events: none; display: none; }

  /* the AXES: one pill per frame axis */
  #anim { display: flex; flex-direction: column; gap: 6px; align-items: flex-start; }
  #axesDock { position: fixed; z-index: 4; left: 12px; top: 10px; display: flex; flex-direction: column; gap: 6px; align-items: flex-start; }
  .axis { display: flex; align-items: center; gap: 8px; padding: 4px 12px 4px 2px; border-radius: 999px; }
  .axis.floating { position: fixed; z-index: 5; box-shadow: var(--shadow); }
  .axis .play { width: 26px; height: 26px; border-radius: 50%; border: none; background: var(--text); color: var(--bg); padding: 0;
                display: flex; align-items: center; justify-content: center; flex: none; }
  .axis .play svg { width: 10px; height: 10px; }
  .axis .gap { width: 26px; flex: none; }
  .axis .lab { white-space: nowrap; font-variant-numeric: tabular-nums; min-width: 92px; }
  .axis .lab b { font-weight: 600; }
  .axis .cnt { color: var(--muted); margin-left: 4px; }
  .axis input[type=range] { width: 240px; accent-color: var(--accent); }
  .axis .live { display: flex; align-items: center; gap: 5px; font-size: 11px; color: var(--ok); }
  .axis .live::before { content: ""; width: 7px; height: 7px; border-radius: 50%; background: var(--ok); }

  /* the TOOLBAR and the PANELS */
  #toolbar { position: fixed; z-index: 6; right: 12px; top: 10px; display: flex; align-items: center; gap: 3px; padding: 3px; border-radius: 10px; }
  #toolbar .sep { width: 1px; height: 18px; background: var(--line); margin: 0 2px; }
  #renderGroup { display: none; height: 30px; border-radius: 7px; background: var(--accent); overflow: hidden; color: #fff; }
  #renderGroup button { border: none; background: transparent; color: #fff; }
  #renderBtn { padding: 0 8px 0 10px; font-weight: 500; }
  #renderMode { width: 30px; border-left: 1px solid rgba(255,255,255,0.28) !important; display: flex; align-items: center; justify-content: center; }
  #renderMode svg { width: 15px; height: 15px; }
  #renderMode.open { background: rgba(0,0,0,0.2) !important; }
  #panels { position: fixed; z-index: 5; right: 12px; top: 56px; bottom: 12px; width: 320px; display: flex; flex-direction: column; gap: 8px;
            overflow-y: auto; pointer-events: none; }
  .panel { pointer-events: auto; flex: none; border-radius: 10px; background: var(--pane); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
           border: 1px solid var(--line); padding: 6px 10px 10px; display: none; flex-direction: column; gap: 10px; }
  .panel.open { display: flex; }
  .panel > .head { display: flex; align-items: center; gap: 6px; height: 26px; }
  .panel > .head b { flex: 1; font-weight: 600; }
  .panel > .head small { color: var(--muted); font-size: 12px; }
  .panel > .head button { width: 22px; height: 22px; padding: 0; border: none; background: transparent; color: var(--muted); display: flex; align-items: center; justify-content: center; }
  .panel > .head button svg { width: 12px; height: 12px; }
  .panel.folded > :not(.head) { display: none !important; }
  .panel.folded .fold svg { transform: rotate(180deg); }
  .panel section { display: flex; flex-direction: column; gap: 8px; }
  .panel h3 { margin: 0; font-size: 11px; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
  .panel .row { display: grid; grid-template-columns: 96px minmax(0, 1fr) 44px; align-items: center; gap: 8px; }
  .panel .row .val { text-align: right; color: var(--muted); font-variant-numeric: tabular-nums; font-size: 12px; }
  .panel .kv { display: grid; grid-template-columns: 60px minmax(0, 1fr); align-items: center; gap: 8px; font-size: 12px; }
  .panel .kv > span:nth-child(odd) { color: var(--muted); }
  .panel input[type=range] { width: 100%; accent-color: var(--accent); }
  .panel label.chk { display: flex; align-items: center; gap: 8px; }
  .panel .hint { color: var(--muted); font-size: 12px; }
  .panel select { height: 26px; border-radius: 6px; border: 1px solid var(--line); background: var(--raised); padding: 0 4px; }
  .panel input.numf { height: 26px; width: 100%; box-sizing: border-box; border-radius: 6px; border: 1px solid var(--line);
                      background: var(--raised); padding: 0 6px; font-variant-numeric: tabular-nums; }
  /* a row of a list ( layers, curves ): its settings' chevron, a mark, the name, what it holds, the eye */
  .item { display: grid; grid-template-columns: 18px 12px minmax(0, 1fr) auto 26px; align-items: center; gap: 8px; height: 30px; padding-left: 2px; border-radius: 6px; }
  .item.open { background: var(--soft); }
  .item.off { color: var(--muted); }
  .item .more, .item .eye { width: 22px; height: 26px; padding: 0; border: none; background: transparent; color: inherit; display: flex; align-items: center; justify-content: center; }
  .item .more { color: var(--muted); }
  .item .more svg { width: 11px; height: 11px; transition: transform 0.15s; }
  .item.open .more svg { transform: rotate(90deg); }
  .item .eye svg { width: 16px; height: 16px; }
  .item .mark { width: 12px; height: 12px; border-radius: 3px; }
  .item .mark.line { height: 3px; border-radius: 2px; }
  .item .name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .item .what { color: var(--muted); font-size: 12px; font-variant-numeric: tabular-nums; white-space: nowrap; }
  .sub { display: none; flex-direction: column; gap: 8px; padding: 4px 4px 10px 28px; }
  .sub.open { display: flex; }
  .panel .sub .row { grid-template-columns: 58px minmax(0, 1fr) 40px; }
  .panel .sub .seg button { font-size: 11px; padding: 0 2px; }
  .list { display: flex; flex-direction: column; gap: 1px; }
  .choice { display: flex; align-items: center; gap: 10px; padding: 7px 10px; border-radius: 8px; border: 1px solid var(--line); cursor: pointer; }
  .choice.on { border-color: var(--accent-ink); background: var(--accent-soft); }
  .choice b { font-weight: 500; } .choice small { display: block; color: var(--muted); font-size: 12px; }
  #kept { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
  #kept figure { margin: 0; display: flex; flex-direction: column; gap: 4px; cursor: pointer; }
  #kept img { width: 100%; aspect-ratio: 4 / 3; object-fit: cover; border-radius: 6px; border: 1px solid var(--line); box-sizing: border-box; }
  #kept figcaption { font-size: 12px; color: var(--muted); }
  #views { display: flex; flex-wrap: wrap; gap: 6px; }
  #views button { height: 24px; padding: 0 10px; border-radius: 999px; border: 1px solid var(--line); background: transparent; font-size: 12px; }
  #views button.add { border-style: dashed; color: var(--muted); }
  .svc { display: grid; grid-template-columns: 10px minmax(0, 1fr) auto; gap: 6px 8px; align-items: center; font-size: 12px; }
  .sdot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .sdot.ok { background: var(--ok); } .sdot.warn { background: var(--warn); } .sdot.idle { border: 1.5px solid var(--ok); box-sizing: border-box; }
  #panels:has(.panel[data-pane="services"].open:not(.folded)) { width: min(460px, calc(100vw - 24px)); }
  .svcCard { display: flex; flex-direction: column; gap: 6px; padding: 6px 0 8px; border-top: 1px solid var(--line); font-size: 12px; }
  .svcCard:first-child { border-top: none; padding-top: 0; }
  .svcTop { display: grid; grid-template-columns: 18px 10px minmax(0, 1fr); gap: 6px; align-items: center; }
  .svcTop .more { width: 18px; height: 20px; padding: 0; border: none; background: transparent; color: var(--muted); display: flex; align-items: center; justify-content: center; }
  .svcTop .more svg { width: 11px; height: 11px; transition: transform 0.12s; } .svcCard.open .svcTop .more svg { transform: rotate(90deg); }
  .svcName b { font-weight: 500; } .svcName code { font: 11px ui-monospace, Menlo, monospace; }
  .svcOn { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 4px 8px; align-items: center; padding-left: 24px; }
  .svcOn select { min-width: 0; } .svcOn > span:last-child { grid-column: 2; }
  .warnText { color: var(--warn); }
  .svcDet { display: none; flex-direction: column; gap: 6px; padding-left: 24px; }
  .svcCard.open .svcDet { display: flex; }
  .svcMissing { color: var(--text) !important; }
  .svcFound { margin: 0; padding-left: 16px; color: var(--muted); line-height: 1.5; overflow-wrap: anywhere; }
  pre.cmd { margin: 0; padding: 7px 9px; border-radius: 6px; background: var(--well); border: 1px solid var(--line);
            font: 11px/1.5 ui-monospace, "JetBrains Mono", Menlo, monospace; white-space: pre-wrap; word-break: break-all; user-select: text; }
  pre.cmd .cmt { color: var(--muted); } pre.cmd .cmt.bad, .hint.bad { color: #d9534f; }
  .svcCopyRow, .svcCopy, .svcPath { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .svcPath input { flex: 1; min-width: 0; }
  #renderGroup.off { background: var(--well); } #renderGroup.off button { color: var(--muted); }

  /* over the view: the picture's state and its card, the colour scale */
  #renderHint { position: fixed; z-index: 4; left: 50%; top: 12px; transform: translateX(-50%); display: none; align-items: center; gap: 8px;
                padding: 5px 12px; border-radius: 999px; font-size: 12px; }
  #renderCard { position: fixed; z-index: 4; left: 50%; bottom: 14px; transform: translateX(-50%); display: none; align-items: center;
                gap: 8px; padding: 6px 8px 6px 12px; border-radius: 10px; }
  #renderCard .info { font-variant-numeric: tabular-nums; font-size: 12px; padding-right: 4px; }
  #scalesDock { position: fixed; z-index: 4; left: 12px; bottom: 12px; display: flex; flex-direction: column; gap: 6px; align-items: flex-start; }
  #legend { display: none; align-items: center; gap: 8px; padding: 5px 10px 5px 2px; border-radius: 8px; font-size: 12px; cursor: pointer; }
  #legend.floating { position: fixed; z-index: 5; box-shadow: var(--shadow); }
  #legend canvas { width: 120px; height: 8px; border-radius: 2px; }
  #legend .name { font-weight: 500; }

  /* the CURVES: one window per chart, see-through, moved by its title, resized by its corner */
  #curves { position: fixed; inset: 0; z-index: 3; pointer-events: none; font-size: 11px; }
  #curves.off { visibility: hidden; }
  #curves .chart { position: absolute; display: flex; flex-direction: column; box-sizing: border-box; min-width: 200px; min-height: 30px;
                   pointer-events: auto; resize: both; overflow: hidden; border-radius: 10px; background: var(--chart);
                   backdrop-filter: blur(3px); -webkit-backdrop-filter: blur(3px); border: 1px solid var(--line); }
  #curves .chart.folded { resize: none; height: 30px !important; }
  #curves .chart.folded .plot, #curves .chart.folded .axes { display: none; }
  #curves .ttl { display: flex; align-items: center; gap: 7px; height: 28px; flex: none; padding: 0 4px 0 10px; cursor: grab; user-select: none; touch-action: none; font-size: 12px; }
  #curves .ttl .mk { width: 10px; height: 3px; border-radius: 2px; flex: none; }
  #curves .ttl .nm { font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  #curves .ttl .v { font-variant-numeric: tabular-nums; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; min-width: 0; }
  #curves .ttl button { height: 20px; min-width: 20px; padding: 0 5px; border-radius: 4px; border: none; background: transparent;
                        color: var(--muted); font-size: 11px; display: flex; align-items: center; justify-content: center; flex: none; }
  #curves .ttl button.log { border: 1px solid var(--line); color: var(--text); }
  #curves .ttl button.log.on { border-color: var(--accent-ink); background: var(--accent-soft); }
  #curves .ttl button svg { width: 11px; height: 11px; }
  #curves .plot { position: relative; flex: 1 1 auto; min-height: 30px; margin: 2px 10px 0 50px; }
  #curves svg { display: block; width: 100%; height: 100%; overflow: visible; cursor: crosshair; }
  #curves .ylab { position: absolute; left: -48px; width: 44px; text-align: right; font-size: 10px; color: var(--muted); font-variant-numeric: tabular-nums; }
  #curves .axes { display: flex; justify-content: space-between; margin: 2px 10px 5px 50px; font-size: 10px; color: var(--muted); font-variant-numeric: tabular-nums; }
  #curves svg .grid { stroke: var(--line); stroke-width: 1; }
  #curves svg .cur { stroke: var(--text); stroke-width: 1; }
  #curves svg .hov { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; }
  #curveTip { position: fixed; z-index: 9; display: none; padding: 5px 8px; border-radius: 6px; background: var(--text); color: var(--bg);
              font-size: 12px; font-variant-numeric: tabular-nums; pointer-events: none; white-space: nowrap; }

  #help { display: none; position: fixed; z-index: 9; top: 56px; right: 344px; font-size: 12px; padding: 12px 14px; }
  #help table { border-collapse: collapse; }
  #help td { padding: 1px 0; }
  #help td:first-child { padding-right: 12px; white-space: nowrap; }
  #help td:last-child { color: var(--muted); }
  #nav.pulse { transform: scale(1.15); background: var(--accent); color: #fff; }
  #nav { transition: transform 0.2s, background 0.2s; }

  /* EMBEDDED ( an iframe ): the toolbar shows the navigation, a page of its own and the gear; the gear shows the rest */
  body.embed #toolbar .full { display: none; }
  body.embed.tools #toolbar .full { display: inline-flex; }
  body:not(.embed) #gearBtn, body:not(.embed) #openFull { display: none; }
</style>
</head>
<body data-theme="light">
<div id="view">
  <canvas id="c"></canvas>
  <img id="renderImg" alt="">
</div>

<div id="axesDock" aria-label="Frames"><div id="anim"></div></div>

<div id="toolbar" class="glass" role="toolbar" aria-label="Tools">
  <div id="renderGroup" class="full">
    <button id="renderBtn" title="a picture of this view by Blender ( r )">Render</button>
    <button id="renderMode" aria-haspopup="menu" title="when to render"></button>
  </div>
  <button class="ibtn" id="nav" aria-label="Navigation: wheel, pinch and touch drive the view"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M10 2v16M2 10h16M7.5 4.5 10 2l2.5 2.5M7.5 15.5 10 18l2.5-2.5M4.5 7.5 2 10l2.5 2.5M15.5 7.5 18 10l-2.5 2.5"/></svg></button>
  <a class="ibtn" id="openFull" href="#" target="_blank" aria-label="Open in a page of its own" title="open in a page of its own"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M11 3h6v6M17 3l-8 8M14 12v5H3V6h5"/></svg></a>
  <span class="sep full"></span>
  <button class="ibtn full" data-panel="layers" title="layers"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><path d="M10 3 18 7l-8 4-8-4z"/><path d="M2 11l8 4 8-4"/><path d="M2 14.5l8 4 8-4"/></svg></button>
  <button class="ibtn full" data-panel="curves" title="curves ( g: all shown / hidden )"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2 15l5-6 4 3 7-8"/></svg></button>
  <button class="ibtn full" data-panel="colour" title="colour by a field"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><rect x="3" y="3" width="14" height="14" rx="2"/><path d="M3 13l4-4 3 3 3-4 4 5"/></svg></button>
  <button class="ibtn full" data-panel="display" title="display: theme, light, view"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><circle cx="10" cy="10" r="3"/><path d="M10 2.5v2M10 15.5v2M2.5 10h2M15.5 10h2M4.7 4.7l1.4 1.4M13.9 13.9l1.4 1.4M4.7 15.3l1.4-1.4M13.9 6.1l1.4-1.4"/></svg></button>
  <button class="ibtn full" data-panel="pictures" title="pictures by Blender"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><path d="M3 6h3l1.5-2h5L14 6h3v10H3z"/><circle cx="10" cy="11" r="2.6"/></svg></button>
  <button class="ibtn full" data-panel="services" title="services"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linejoin="round"><rect x="3" y="4" width="14" height="5" rx="1.5"/><rect x="3" y="11" width="14" height="5" rx="1.5"/></svg><span class="badge" id="svcBadge"></span></button>
  <button class="ibtn full" data-panel="more" title="cells, export, about, help"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="round"><path d="M4 10h.01M10 10h.01M16 10h.01"/></svg></button>
  <button class="ibtn" id="gearBtn" aria-label="All the tools" title="all the tools"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M4 6h12M4 10h12M4 14h12"/><circle cx="8" cy="6" r="1.6" fill="var(--raised)"/><circle cx="13" cy="10" r="1.6" fill="var(--raised)"/><circle cx="7" cy="14" r="1.6" fill="var(--raised)"/></svg></button>
</div>

<div id="panels">
  <section class="panel" data-pane="layers" aria-label="Layers">
    <div class="head"><b>Layers</b><small>1–9</small><button class="fold" title="fold"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M5 12l5-5 5 5"/></svg></button><button class="close" title="close"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M5 5l10 10M15 5 5 15"/></svg></button></div>
    <div class="list" id="layers"></div>
  </section>
  <section class="panel" data-pane="curves" aria-label="Curves">
    <div class="head"><b>Curves</b><small>g</small><button class="fold" title="fold"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M5 12l5-5 5 5"/></svg></button><button class="close" title="close"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M5 5l10 10M15 5 5 15"/></svg></button></div>
    <div class="list" id="curveList"></div>
  </section>
  <section class="panel" data-pane="colour" aria-label="Colour">
    <div class="head"><b>Colour</b><button class="fold" title="fold"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M5 12l5-5 5 5"/></svg></button><button class="close" title="close"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M5 5l10 10M15 5 5 15"/></svg></button></div>
    <section>
      <h3>Points by</h3>
      <div id="fieldList" style="display: flex; flex-direction: column; gap: 6px"></div>
      <select id="field" class="hidden"><option value="-1">plain colours</option></select>
    </section>
    <section id="fieldRange">
      <h3>Range</h3>
      <div class="seg" id="scaleSeg"><button data-s="frame">this frame</button><button data-s="all">all frames</button><button data-s="manual">typed</button></div>
      <div style="display: grid; grid-template-columns: 72px minmax(0, 1fr) 72px; gap: 8px; align-items: center">
        <input class="numf" id="fieldMin" inputmode="decimal" title="the start of the scale ( typed: the scale stays )">
        <canvas id="legendBar2" width="120" height="10" style="width: 100%; height: 10px; border-radius: 2px"></canvas>
        <input class="numf" id="fieldMax" inputmode="decimal" title="the end of the scale">
      </div>
      <button id="scaleBtn" class="hidden"></button>
    </section>
  </section>
  <section class="panel" data-pane="display" aria-label="Display">
    <div class="head"><b>Display</b><button class="fold" title="fold"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M5 12l5-5 5 5"/></svg></button><button class="close" title="close"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M5 5l10 10M15 5 5 15"/></svg></button></div>
    <div class="kv">
      <span>theme</span><div class="seg" id="themeSeg"><button data-t="light">light</button><button data-t="dark">dark</button><button data-t="auto">auto</button></div>
      <span id="lightLbl">light</span><div class="seg" id="lightSeg"><button data-l="flat">flat</button><button data-l="relief">relief</button><button data-l="custom">custom</button></div>
      <span>view</span><div style="display: flex; gap: 6px"><div class="seg" id="projSeg" style="flex: 1"><button data-p="persp">persp.</button><button data-p="ortho">ortho.</button></div><button class="btn" id="resetBtn" title="reset the view ( 0 )">Reset</button></div>
    </div>
    <div id="lightCustom" style="display: none; flex-direction: column; gap: 8px">
      <label class="chk"><input type="checkbox" id="cbEdl" checked> eye-dome lighting</label>
      <div class="row"><span>strength</span><input id="edlK" type="range" min="0" max="1" step="0.01" value="0.5"><span class="val" id="edlVal"></span></div>
      <label class="chk"><input type="checkbox" id="cbAo" checked> occlusion, on screen</label>
      <div class="row"><span>strength</span><input id="aoK" type="range" min="0" max="1" step="0.01" value="0.7"><span class="val" id="aoKVal"></span></div>
      <div class="row"><span>radius × points</span><input id="aoR" type="range" min="1" max="20" step="0.5" value="5"><span class="val" id="aoRVal"></span></div>
    </div>
    <div id="bakeBox" style="display: none; flex-direction: column; gap: 6px">
      <div class="row" id="bakeLbl"><span>baked occlusion</span><input id="bake" type="range" min="0" max="1" step="0.01" value="0.8"><span class="val" id="bakeVal"></span></div>
      <div><button class="btn" id="bakeBtn" style="display: none" title="asks the sdot.ao service">Bake the occlusion</button></div>
      <div class="hint" id="bakeHint" style="display: none"></div>
    </div>
    <div id="axes" style="display: none">views on <select id="axX"></select> <select id="axY"></select> <select id="axZ"></select></div>
    <div id="slices"></div>
    <label class="chk"><input type="checkbox" id="cbWheel"> the wheel moves / zooms the view</label>
    <label class="chk"><input type="checkbox" id="cbTouch"> touch drives the view</label>
  </section>
  <section class="panel" data-pane="pictures" aria-label="Pictures">
    <div class="head"><b>Pictures</b><small>r</small><button class="fold" title="fold"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M5 12l5-5 5 5"/></svg></button><button class="close" title="close"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M5 5l10 10M15 5 5 15"/></svg></button></div>
    <section id="renderBox">
      <h3>Refining</h3>
      <div class="seg" id="stepsSeg"><button data-n="8">8</button><button data-n="32">32</button><button data-n="128">128</button><button data-n="512">512</button></div>
      <div class="row"><span>auto, at rest</span><input id="restDelay" type="range" min="0.2" max="3" step="0.1" value="0.6"><span class="val" id="restVal"></span></div>
      <input type="number" id="renderSamples" class="hidden" value="128">
    </section>
    <section>
      <h3>Kept <span id="keptCount"></span></h3>
      <div id="kept"></div>
      <div class="hint" id="keptHint"></div>
    </section>
    <section>
      <h3>Views</h3>
      <div id="views"></div>
    </section>
  </section>
  <section class="panel" data-pane="services" aria-label="Services">
    <div class="head"><b>Services</b><button id="svcRefresh" title="ask again what the machine has"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 10a6 6 0 1 1-1.8-4.3"/><path d="M16 3.5v3.2h-3.2"/></svg></button><button class="fold" title="fold"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M5 12l5-5 5 5"/></svg></button><button class="close" title="close"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M5 5l10 10M15 5 5 15"/></svg></button></div>
    <div id="svcBody"></div>
  </section>
  <section class="panel" data-pane="more" aria-label="More">
    <div class="head"><b>More</b><button class="fold" title="fold"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M5 12l5-5 5 5"/></svg></button><button class="close" title="close"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M5 5l10 10M15 5 5 15"/></svg></button></div>
    <section id="cellsBox" style="display: none">
      <h3>Cells <span class="muted" style="text-transform: none; letter-spacing: 0; font-weight: 400">· rebuilt from the seeds</span></h3>
      <div id="regionChoices" style="display: flex; flex-direction: column; gap: 6px">
        <label class="choice" data-m="off"><input type="radio" name="region" value="off" checked><span><b>as stored</b><small>what the run wrote</small></span></label>
        <label class="choice" data-m="cut"><input type="radio" name="region" value="cut"><span><b>cut</b><small>the cells behind a plane across the view</small></span></label>
        <label class="choice" data-m="boundary"><input type="radio" name="region" value="boundary"><span><b>boundary</b><small>the cells touching the domain</small></span></label>
      </div>
      <select id="regionMode" class="hidden"><option value="off">as stored</option><option value="cut">cut</option><option value="boundary">boundary</option></select>
      <div style="display: flex; gap: 8px; align-items: center">
        <button class="btn" id="regionBtn" title="the cut again, through the target of the view as it is now">Cut here</button>
        <label class="chk"><input type="checkbox" id="regionFollow" checked> follow the view</label>
      </div>
      <div class="hint" id="regionHint" style="display: none"></div>
    </section>
    <section>
      <h3>Export</h3>
      <div style="display: flex; gap: 6px; flex-wrap: wrap">
        <button class="btn" id="gifBtn" title="one image per frame of the played axis, as the view shows it"><span>GIF of the frames</span></button>
        <button class="btn" id="pngBtn" title="the view as it is">PNG of the view</button>
      </div>
    </section>
    <section>
      <h3>About</h3>
      <div class="kv" id="about">
        <span>name</span><span id="storeTitle" style="overflow: hidden; text-overflow: ellipsis">__TITLE__</span>
        <span>state</span><span id="statusText">—</span>
        <span>holds</span><span id="counts"></span>
      </div>
      <div><button class="btn" id="helpBtn">Keyboard shortcuts ( ? )</button></div>
    </section>
  </section>
</div>

<div id="scalesDock"><div id="legend" class="glass" title="a click: the colour settings">
  <span class="grip" title="move it ( a double-click puts it back )"><svg viewBox="0 0 10 14" fill="currentColor"><circle cx="3" cy="3" r="1.1"/><circle cx="7" cy="3" r="1.1"/><circle cx="3" cy="7" r="1.1"/><circle cx="7" cy="7" r="1.1"/><circle cx="3" cy="11" r="1.1"/><circle cx="7" cy="11" r="1.1"/></svg></span>
  <span class="name" id="legendName"></span><span class="muted" id="legendLo"></span><canvas id="legendBar" width="120" height="8"></canvas><span class="muted" id="legendHi"></span>
</div></div>
<div id="renderHint" class="glass"><span class="sdot ok"></span><span id="renderHintText"></span></div>
<div id="renderCard" class="glass">
  <span class="info" id="renderInfo"></span>
  <button class="btn primary" id="keepBtn" title="written into the store ( renders/ )">Keep</button>
  <button class="btn" id="compareBtn" title="hold: the live view">Compare</button>
  <a class="btn" id="downloadBtn" download="render.png">Download</a>
  <button class="ibtn" id="closeRenderBtn" aria-label="Close the picture" style="width: 26px; height: 26px"><svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M5 5l10 10M15 5 5 15"/></svg></button>
</div>
<div id="curves"></div>
<div id="curveTip"></div>
<div class="pop menu" id="renderMenu" role="menu" style="display: none"></div>
<div class="pop" id="help">
  <b>Keyboard shortcuts</b>
  <table>
    <tr><td>&larr; / &rarr;</td><td id="hTime">frame -1 / +1 (an axis has the focus)</td></tr>
    <tr><td>Home / End</td><td>first / last frame</td></tr>
    <tr><td>Space</td><td>play / pause</td></tr>
    <tr><td>drag</td><td id="hDrag">orbit</td></tr>
    <tr><td>Shift + drag</td><td>move (pan)</td></tr>
    <tr><td>wheel</td><td>move · Ctrl/pinch: zoom towards the cursor</td></tr>
    <tr><td>&larr; &rarr; &uarr; &darr;</td><td id="hArrows">rotate (Shift/Ctrl: faster)</td></tr>
    <tr><td>+ / -</td><td>zoom in / out</td></tr>
    <tr><td>[ / ]</td><td>point radius - / +</td></tr>
    <tr><td>f · e · p</td><td>faces · hidden-edge mode · points</td></tr>
    <tr><td>1 – 9</td><td>a layer on / off</td></tr>
    <tr><td>c</td><td>points coloured by depth (3D)</td></tr>
    <tr><td>g</td><td>the curves</td></tr>
    <tr><td>r</td><td>a picture by Blender ( with the service )</td></tr>
    <tr><td>o</td><td>ortho / perspective projection</td></tr>
    <tr><td>0 · double-click</td><td>reset the view</td></tr>
    <tr><td>d</td><td>light / dark ( <span id="modeLabel">light</span> )</td></tr>
    <tr><td>?</td><td>show / hide this help</td></tr>
  </table>
</div>
<div class="hidden">
  <input type="checkbox" id="cbFaces" checked><input id="op" type="range" min="0.05" max="1" step="0.01" value="1"><span id="opVal"></span>
  <select id="edgeMode"><option value="hide">hidden removed</option><option value="ghost" selected>hidden as ghost</option><option value="all">all visible</option><option value="none">none</option></select>
  <input type="checkbox" id="cbPoints" checked><input type="checkbox" id="cbDepth"><input id="r" type="range" min="0" max="1" step="0.001" value="0.5"><span id="rVal"></span>
  <span id="status"></span><span id="svcBtn"></span><div id="svcPop"></div><div id="exportMenu"></div><span id="curvesLbl"></span><button id="curvesBtn"></button><span id="keptCountDummy"></span>
  <button id="themeBtn"></button><div id="exportCli"></div><span id="lightBox"></span>
</div>
<script id="main" type="text/plain">
// ============================================================================================
// data -- everything is encoded in base64 in the page (self-contained file, openable as file://); the big arrays are
// BLOBS, compressed, that the loader at the end of the page inflates before it runs this script ( "@i": blob i )
// ============================================================================================
function b64bytes(s) {
  if (typeof s !== "string") return s;                // ( a store: the arrays come decoded )
  if (s[0] === "@") return BLOBS[parseInt(s.slice(1))];
  const raw = atob(s), b = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) b[i] = raw.charCodeAt(i);
  return b;
}
function decF32(s) { if (typeof s !== "string") return s; const b = b64bytes(s); return new Float32Array(b.buffer, 0, b.length / 4); }
function decI32(s) { if (typeof s !== "string") return s; const b = b64bytes(s); return new Int32Array(b.buffer, 0, b.length / 4); }
function decU16(s) { if (typeof s !== "string") return s; const b = b64bytes(s); return new Uint16Array(b.buffer, 0, b.length / 2); }
function decU8(s) { if (typeof s !== "string") return s; return b64bytes(s); }
// the pool of positions: float32, or 16 bits per coordinate ( "codes|lo per dim|step per dim", `compact` )
function decPos(s) {
  if (typeof s !== "string") return s;
  const p = s.split("|");
  if (p.length === 1) return decF32(s);
  const q = decU16(p[0]), lo = p[1].split(",").map(Number), st = p[2].split(",").map(Number), d = lo.length, out = new Float32Array(q.length);
  for (let i = 0; i < q.length; i++) out[i] = lo[i % d] + st[i % d] * q[i];
  return out;
}
// a field: float32, or 16 bits between lo and lo + 65534 step ( 65535: none )
function decField(f) {
  if (f.step === undefined) return decF32(f.data);
  const q = decU16(f.data), out = new Float32Array(q.length);
  for (let i = 0; i < q.length; i++) out[i] = q[i] === 65535 ? NaN : f.lo + f.step * q[i];
  return out;
}

const D      = __D__;                  // dimension of the WORLD (size of the position vectors)
const BOUNDS = __BOUNDS__;             // [lo, hi] per dimension

// ONE single vertex pool for the whole scene (POS, with stride D); a primitive only refers to it by
// INDICES, and its color is only an index into COLORS. This is what keeps the weight of the
// file down: a cube vertex serves 6 triangles and 3 edges, it is stored only once, and
// a color costs only 2 bytes per primitive instead of 4 per vertex.
let POS      = decPos("__POS__");
const COLORS = __COLORS__, COLORS_DARK = __COLORS_DARK__;
let PAL = COLORS;                      // the table of the CURRENT theme             // table of distinct colors, as [r, g, b, a]

// polygons of FREE length: POLY_OFF slices POLY_V (POLY_OFF[p] .. POLY_OFF[p+1]). They are only
// triangulated here, at display time -- the file itself keeps the faces as they are.
let POLY_V = decI32("__POLY_V__"), POLY_OFF = decI32("__POLY_OFF__"), POLY_C = decU16("__POLY_C__");
// the BAKED occlusion ( `Visualizer( ao = True )`, see `occlusion.py` ): a code per corner, aligned with POLY_V, 255 = open;
// empty when none was baked
let POLY_AO = decU8("__POLY_AO__");
let EDG_V  = decI32("__EDG_V__"),  EDG_C    = decU16("__EDG_C__");
let PNT_V  = decI32("__PNT_V__"),  PNT_C    = decU16("__PNT_C__"), PNT_R = decF32("__PNT_R__");
let PNT_M  = decF32("__PNT_M__");    // a matrix per point, row by row: an ELLIPSE ( all 0: a disc )
// LAYERS ( `add_layer` ): the layer of each point and edge, 0 for none; `LAYERS[ l - 1 ] = [ name, visible ]`
let PNT_L = new Uint8Array(b64bytes("__PNT_L__")), EDG_L = new Uint8Array(b64bytes("__EDG_L__"));
const LAYERS = __LAYERS__, layerOn = [true].concat(LAYERS.map(l => l[1]));
// FIELDS ( `add_points( fields = )` ): values per point, NaN where a point does not have the field
const FIELD0 = __FIELD0__;            // the field shown at the start ( null: none )
const FIELDS = __FIELDS__.map(f => ({ name: f.name, data: decField(f), all: f.all || null }));
// IMAGES ( `add_image` ): grey levels on 8 bits, row by row from the lowest y
const IMAGES = __IMAGES__.map(m => Object.assign(m, { bytes: b64bytes(m.data) }));
// SHARED primitives: the first SHARED.poly / .edge / .pnt of their lists, drawn in every frame ( the frames' ranges follow )
let SHARED = __SHARED__;

let CURVES = __CURVES__;             // { name, x, y, log, color: [ light, dark ] }, see `add_curve`
let HPOLY  = __HPOLY__.map(h => ({                                     // polytopes in H-rep
  nb: h.nb, dirs: decF32(h.dirs), offs: decF32(h.offs), fcol: h.fcol, ecol: h.ecol, edg: h.edg,
}));
// FRAMES: each one is only a RANGE in the lists above (they share nothing else than
// the vertex pool) -- FR_* gives the bounds, FR_VAL the abscissae on the axes ( NB_AXES per frame ).
const AXES = __AXES__, PLAYABLE = __PLAYABLE__, FPS = __FPS__, NB_AXES = AXES.length;
let FR_VAL = decF32("__FR_VAL__");
let FR_POLY = decI32("__FR_POLY__"), FR_EDG = decI32("__FR_EDG__");
let FR_PNT = decI32("__FR_PNT__"), FR_HP = decI32("__FR_HP__");
let NB_FRAMES = FR_VAL.length / NB_AXES;

const RMIN = __RMIN__, RMAX = __RMAX__, R0 = __R0__;
const BG = __BG__;                      // [ light, dark ], `null`: the theme's

let NB_V = POS.length / D, NB_POLY = POLY_OFF.length - 1, NB_EDG = EDG_V.length / 2;

// the STORE mode ( `Visualizer( store = )`, `index.html` in the directory of a store ): the arrays above are those of the
// frame shown, with the base ( the shared primitives ) first -- swapped when another frame comes ( `useScene` ). The frames'
// ranges then say, for the frame shown, "the shared ones, then all the rest".
const STORED = !!window.STORE;
let curScene = null;
function useScene(sc) {
  if (sc === curScene) return;
  curScene = sc;
  ({ POS, POLY_V, POLY_OFF, POLY_C, POLY_AO, EDG_V, EDG_C, EDG_L, PNT_V, PNT_C, PNT_R, PNT_M, PNT_L, SHARED, HPOLY } = sc);
  for (const f of FIELDS) f.data = sc.fields[f.name];
  const fr = (a, b) => { const r = new Int32Array(NB_FRAMES + 1); r.fill(a, 0, frameIdx + 1); r.fill(b, frameIdx + 1); return r; };
  FR_POLY = fr(SHARED.poly, POLY_OFF.length - 1); FR_EDG = fr(SHARED.edge, EDG_V.length / 2);
  FR_PNT = fr(SHARED.pnt, PNT_V.length); FR_HP = fr(0, HPOLY.length);
  NB_V = POS.length / D; NB_POLY = POLY_OFF.length - 1; NB_EDG = EDG_V.length / 2;
  projectPool();
}
// the number of points of frame `f` ( the morph matches two frames that have as many )
function framePoints(f) {
  return STORED ? ((STORE.m.frames[f].n || {}).pnt || 0) : FR_PNT[f + 1] - FR_PNT[f];
}

// ============================================================================================
// projection world -> 3D: which dimensions we look at, and where we cut the others
// ============================================================================================
// AX = the dimensions shown in x, y, z (z = -1 in 2D: everything is flat). SLICE fixes the value
// of the NON-shown dimensions -- this is the slice setting, only polytopes truly undergo it
// (see sliceHalfspaces); an already triangulated mesh is simply projected.
let AX = [0, 1, D >= 3 ? 2 : -1];
const SLICE = new Float32Array(D);
for (let k = 0; k < D; k++) SLICE[k] = 0.5 * (BOUNDS[k][0] + BOUNDS[k][1]);

function px(src, i) { return AX[0] >= 0 ? src[i * D + AX[0]] : 0; }
function py(src, i) { return AX[1] >= 0 ? src[i * D + AX[1]] : 0; }
function pz(src, i) { return AX[2] >= 0 ? src[i * D + AX[2]] : 0; }

// ============================================================================================
// polytopes: slice then vertex enumeration (in the browser, at each change)
// ============================================================================================
// { dir . x <= off } in dimension D, restricted to the shown dimensions and the fixed slices:
// dir_seen . y <= off - sum( dir_k * SLICE[k] ) over the hidden dimensions. A slice of
// half-spaces remains a set of half-spaces -- hence a polytope, which we know how to re-enumerate.
function sliceHalfspaces(h) {
  const A = [], b = [];
  for (let i = 0; i < h.nb; i++) {
    const a0 = AX[0] >= 0 ? h.dirs[i * D + AX[0]] : 0;
    const a1 = AX[1] >= 0 ? h.dirs[i * D + AX[1]] : 0;
    const a2 = AX[2] >= 0 ? h.dirs[i * D + AX[2]] : 0;
    let rhs = h.offs[i];
    for (let k = 0; k < D; k++)
      if (k !== AX[0] && k !== AX[1] && k !== AX[2]) rhs -= h.dirs[i * D + k] * SLICE[k];
    const n = Math.hypot(a0, a1, a2);
    // plane that became constant: either the constraint is empty (the slice misses the polytope),
    // or it is always true and says nothing anymore.
    if (n < 1e-12) { if (rhs < -1e-9) return null; continue; }
    A.push([a0 / n, a1 / n, a2 / n]); b.push(rhs / n);   // NORMALIZED planes: a single eps everywhere
  }
  return { A, b };
}

// bounding box added to every polytope: without it, an unbounded polytope would have no
// vertex to enumerate, hence nothing to show. It is EXACTLY the scene's box, the one the
// camera frames -- and it is already padded on the Python side, so it trims nothing bounded.
function clipPlanes() {
  const c = [], r = [];
  for (let a = 0; a < 3; a++) {
    const k = AX[a];
    const lo = k >= 0 ? BOUNDS[k][0] : -1, hi = k >= 0 ? BOUNDS[k][1] : 1;
    const mid = 0.5 * (lo + hi), half = Math.max(1e-6, 0.5 * (hi - lo));
    const e = [0, 0, 0]; e[a] = 1;
    c.push(e.slice()); r.push(mid + half);
    c.push(e.map(v => -v)); r.push(-(mid - half));
  }
  return { A: c, b: r };
}

function feasible(A, b, x, eps) {
  for (let m = 0; m < A.length; m++)
    if (A[m][0] * x[0] + A[m][1] * x[1] + A[m][2] * x[2] > b[m] + eps) return false;
  return true;
}

// V-representation of a 3D polytope given as half-spaces: every vertex is the intersection of 3
// planes (and is feasible for all the others); a face is the set of vertices carried by
// one same plane, put back in cyclic order around their center.
function polyhedron3D(A, b) {
  const n = A.length, eps = 1e-6 * Math.max(1, Math.max(...b.map(Math.abs)));
  const V = [], key = new Map();
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) for (let k = j + 1; k < n; k++) {
    const a = A[i], c = A[j], e = A[k];
    const det = a[0] * (c[1] * e[2] - c[2] * e[1])
              - a[1] * (c[0] * e[2] - c[2] * e[0])
              + a[2] * (c[0] * e[1] - c[1] * e[0]);
    if (Math.abs(det) < 1e-9) continue;                       // (nearly) dependent planes: no vertex
    const d0 = b[i], d1 = b[j], d2 = b[k];
    const x = ( d0 * (c[1] * e[2] - c[2] * e[1]) - a[1] * (d1 * e[2] - c[2] * d2)
              + a[2] * (d1 * e[1] - c[1] * d2) ) / det;
    const y = ( a[0] * (d1 * e[2] - c[2] * d2) - d0 * (c[0] * e[2] - c[2] * e[0])
              + a[2] * (c[0] * d2 - d1 * e[0]) ) / det;
    const z = ( a[0] * (c[1] * d2 - d1 * e[1]) - a[1] * (c[0] * d2 - d1 * e[0])
              + d0 * (c[0] * e[1] - c[1] * e[0]) ) / det;
    const p = [x, y, z];
    if (!feasible(A, b, p, eps)) continue;
    const kk = p.map(v => Math.round(v / (eps * 10))).join(",");   // coincident vertices merged
    if (key.has(kk)) continue;
    key.set(kk, V.length); V.push(p);
  }
  if (V.length < 3) return null;

  // which planes each vertex lies on: this is what then lets us recognize an
  // edge carried by the CLIPPING BOX, which is not an edge of the polytope (see buildScene).
  const vpl = V.map(() => []);
  const faces = [], fpl = [];
  for (let p = 0; p < n; p++) {
    const on = [];
    for (let v = 0; v < V.length; v++) {
      const q = V[v];
      if (Math.abs(A[p][0] * q[0] + A[p][1] * q[1] + A[p][2] * q[2] - b[p]) < eps * 10) {
        on.push(v); vpl[v].push(p);
      }
    }
    if (on.length < 3) continue;
    const nz = A[p];
    let u = Math.abs(nz[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
    u = [u[0] - nz[0] * (u[0] * nz[0] + u[1] * nz[1] + u[2] * nz[2]),
         u[1] - nz[1] * (u[0] * nz[0] + u[1] * nz[1] + u[2] * nz[2]),
         u[2] - nz[2] * (u[0] * nz[0] + u[1] * nz[1] + u[2] * nz[2])];
    const ul = Math.hypot(u[0], u[1], u[2]); u = [u[0] / ul, u[1] / ul, u[2] / ul];
    const w = [nz[1] * u[2] - nz[2] * u[1], nz[2] * u[0] - nz[0] * u[2], nz[0] * u[1] - nz[1] * u[0]];
    const ctr = [0, 0, 0];
    for (const v of on) { ctr[0] += V[v][0]; ctr[1] += V[v][1]; ctr[2] += V[v][2]; }
    ctr[0] /= on.length; ctr[1] /= on.length; ctr[2] /= on.length;
    // angle around the center in the plane's (u, w) frame: cyclic order, oriented towards
    // the outside (w = normal x u, and the normal points out of the polytope).
    const ang = q => {
      const dx = q[0] - ctr[0], dy = q[1] - ctr[1], dz = q[2] - ctr[2];
      return Math.atan2(dx * w[0] + dy * w[1] + dz * w[2], dx * u[0] + dy * u[1] + dz * u[2]);
    };
    on.sort((p1, p2) => ang(V[p1]) - ang(V[p2]));
    faces.push(on); fpl.push(p);
  }
  return { V, faces, fpl, vpl };
}

// same thing flat: a vertex is the intersection of 2 lines, and there is only one face.
function polygon2D(A, b) {
  const n = A.length, eps = 1e-6 * Math.max(1, Math.max(...b.map(Math.abs)));
  const V = [], key = new Map();
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const det = A[i][0] * A[j][1] - A[i][1] * A[j][0];
    if (Math.abs(det) < 1e-9) continue;
    const x = (b[i] * A[j][1] - A[i][1] * b[j]) / det;
    const y = (A[i][0] * b[j] - b[i] * A[j][0]) / det;
    const p = [x, y, 0];
    if (!feasible(A, b, p, eps)) continue;
    const kk = p.map(v => Math.round(v / (eps * 10))).join(",");
    if (key.has(kk)) continue;
    key.set(kk, V.length); V.push(p);
  }
  if (V.length < 3) return null;
  const vpl = V.map(() => []);
  for (let p = 0; p < n; p++) for (let v = 0; v < V.length; v++)
    if (Math.abs(A[p][0] * V[v][0] + A[p][1] * V[v][1] - b[p]) < eps * 10) vpl[v].push(p);
  const ctr = V.reduce((a, q) => [a[0] + q[0] / V.length, a[1] + q[1] / V.length, 0], [0, 0, 0]);
  const idx = V.map((_, i) => i);
  idx.sort((p1, p2) => Math.atan2(V[p1][1] - ctr[1], V[p1][0] - ctr[0])
                     - Math.atan2(V[p2][1] - ctr[1], V[p2][0] - ctr[0]));
  return { V, faces: [idx], fpl: [-1], vpl };
}

// ============================================================================================
// building the rendered scene (projected 3D positions + sliced polytopes)
// ============================================================================================
let TRI = { pos: null, nrm: null, col: null, ao: null, n: 0 };
let LIN = { a: null, b: null, col: null, n: 0 };
let PNT = { pos: null, col: null, rad: null, mat: null, n: 0 };

// The pool is projected ONCE per choice of viewed dimensions, not at each frame: it is the
// only array that carries the whole scene, whereas the rest of a rebuild only costs the
// displayed frame.
let PR = new Float32Array(0);
function projectPool() {
  PR = new Float32Array(NB_V * 3);
  for (let i = 0; i < NB_V; i++) {
    PR[3 * i] = px(POS, i); PR[3 * i + 1] = py(POS, i); PR[3 * i + 2] = pz(POS, i);
  }
}

let frameIdx = 0;

function buildScene() {
  // 1. bounds of the displayed frame in each primitive list
  const p0 = FR_POLY[frameIdx], p1 = FR_POLY[frameIdx + 1];
  const e0 = FR_EDG[frameIdx],  e1 = FR_EDG[frameIdx + 1];
  const v0 = FR_PNT[frameIdx],  v1 = FR_PNT[frameIdx + 1];
  const h0 = FR_HP[frameIdx],   h1 = FR_HP[frameIdx + 1];
  // the frame's primitives, then the shared ones ( the points of the frame first: the morph moves those )
  const polys = [[p0, p1], [0, SHARED.poly]], edgs = [[e0, e1], [0, SHARED.edge]], pnts = [[v0, v1], [0, SHARED.pnt]];

  // 2. polytopes: slice -> enumeration -> triangles (fan) + edges (deduplicated)
  const hp = [], hpc = [], hl = [], hlc = [];
  for (let hi = h0; hi < h1; hi++) {
    const h = HPOLY[hi];
    const s = sliceHalfspaces(h);
    if (!s) continue;                                   // empty slice: nothing to show
    const box = clipPlanes();
    const A = s.A.concat(box.A), b = Array.from(s.b).concat(box.b);
    const poly = AX[2] >= 0 ? polyhedron3D(A, b) : polygon2D(A, b);
    if (!poly) continue;
    // The box is a means of SHOWING an open polytope, not a part of it: its planes
    // fill the face through which the polytope leaves the field (otherwise we would see inside), but
    // they yield no edge -- an edge lying on the box is the box being drawn,
    // and it would reappear on its own as soon as the faces are unchecked.
    const nReal = s.A.length;
    const onBox = (i0, i1) => poly.vpl[i0].some(q => q >= nReal && poly.vpl[i1].indexOf(q) >= 0);
    const seen = new Set();
    for (const f of poly.faces) {
      for (let k = 1; k + 1 < f.length; k++) {
        for (const v of [f[0], f[k], f[k + 1]])
          hp.push(poly.V[v][0], poly.V[v][1], poly.V[v][2]);
        hpc.push(h.fcol);
      }
      if (!h.edg) continue;                             // the caller draws its edges itself
      for (let k = 0; k < f.length; k++) {
        const i0 = f[k], i1 = f[(k + 1) % f.length];
        const kk = Math.min(i0, i1) + "," + Math.max(i0, i1);
        if (seen.has(kk)) continue;                     // an edge is carried by 2 faces
        seen.add(kk);
        if (onBox(i0, i1)) continue;
        hl.push(poly.V[i0][0], poly.V[i0][1], poly.V[i0][2],
                poly.V[i1][0], poly.V[i1][1], poly.V[i1][2]);
        hlc.push(h.ecol);
      }
    }
  }

  // 3. triangles. The normal is computed AFTER projection (it only makes sense in the viewed
  //    space) -> flat shading, hence one vertex per triangle corner: it is a display buffer,
  //    not a storage format, the expense stays in GPU memory.
  let nTri = hp.length / 9;
  for (const [a, b] of polys) for (let p = a; p < b; p++) nTri += Math.max(0, POLY_OFF[p + 1] - POLY_OFF[p] - 2);
  FLOOR = floorOn && D >= 3 && !FLAT ? floorOf(polys, edgs, pnts, hp) : null;
  if (FLOOR) nTri += 2;
  TRI = { pos: new Float32Array(nTri * 9), nrm: new Float32Array(nTri * 9), ao: new Float32Array(nTri * 3).fill(1),
          col: new Float32Array(nTri * 12), n: nTri };
  let t = 0;
  function putTri(a, b, c, col, ka, kb, kc) {
    const o = t * 9;
    if (ka !== undefined) { TRI.ao[t * 3] = ka; TRI.ao[t * 3 + 1] = kb; TRI.ao[t * 3 + 2] = kc; }
    TRI.pos[o    ] = a[0]; TRI.pos[o + 1] = a[1]; TRI.pos[o + 2] = a[2];
    TRI.pos[o + 3] = b[0]; TRI.pos[o + 4] = b[1]; TRI.pos[o + 5] = b[2];
    TRI.pos[o + 6] = c[0]; TRI.pos[o + 7] = c[1]; TRI.pos[o + 8] = c[2];
    let nx = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1]);
    let ny = (b[2] - a[2]) * (c[0] - a[0]) - (b[0] - a[0]) * (c[2] - a[2]);
    let nz = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    const nl = Math.hypot(nx, ny, nz);
    if (nl < 1e-20) { nx = 0; ny = 0; nz = 1; } else { nx /= nl; ny /= nl; nz /= nl; }
    for (let k = 0; k < 3; k++) {
      TRI.nrm[o + k * 3] = nx; TRI.nrm[o + k * 3 + 1] = ny; TRI.nrm[o + k * 3 + 2] = nz;
      for (let q = 0; q < 4; q++) TRI.col[t * 12 + k * 4 + q] = col[q];
    }
    t++;
  }
  const at = i => [PR[3 * i], PR[3 * i + 1], PR[3 * i + 2]];
  const baked = !!POLY_AO && POLY_AO.length === POLY_V.length && POLY_AO.length > 0;
  for (const [a, b] of polys) for (let p = a; p < b; p++) {
    const o0 = POLY_OFF[p], o1 = POLY_OFF[p + 1], col = PAL[POLY_C[p]];
    for (let k = o0 + 1; k + 1 < o1; k++)                 // fan from the first vertex
      if (baked) putTri(at(POLY_V[o0]), at(POLY_V[k]), at(POLY_V[k + 1]), col, POLY_AO[o0] / 255, POLY_AO[k] / 255, POLY_AO[k + 1] / 255);
      else putTri(at(POLY_V[o0]), at(POLY_V[k]), at(POLY_V[k + 1]), col);
  }
  for (let i = 0; i + 8 < hp.length; i += 9)
    putTri(hp.slice(i, i + 3), hp.slice(i + 3, i + 6), hp.slice(i + 6, i + 9), hpc[i / 9]);
  if (FLOOR) {
    const [a, b, c, d] = FLOOR.corners;
    putTri(a, b, c, FLOOR.color);
    putTri(a, c, d, FLOOR.color);
  }

  // 4. edges: one INSTANCE per edge (two ends), widened into a quadrilateral in the
  //    shader -- gl.lineWidth is capped at 1 pixel on most drivers.
  const nLin = (e1 - e0) + SHARED.edge + hl.length / 6;
  LIN = { a: new Float32Array(nLin * 3), b: new Float32Array(nLin * 3),
          col: new Float32Array(nLin * 4), n: nLin };
  let e = 0;
  function putLin(a, b, col) {
    LIN.a[e * 3] = a[0]; LIN.a[e * 3 + 1] = a[1]; LIN.a[e * 3 + 2] = a[2];
    LIN.b[e * 3] = b[0]; LIN.b[e * 3 + 1] = b[1]; LIN.b[e * 3 + 2] = b[2];
    for (let q = 0; q < 4; q++) LIN.col[e * 4 + q] = col[q];
    e++;
  }
  for (const [a, b] of edgs) for (let i = a; i < b; i++)
    if (layerOn[EDG_L[i]]) putLin(at(EDG_V[2 * i]), at(EDG_V[2 * i + 1]), PAL[EDG_C[i]]);
  for (let i = 0; i + 5 < hl.length; i += 6)
    putLin(hl.slice(i, i + 3), hl.slice(i + 3, i + 6), hlc[i / 6]);
  LIN.n = e;

  // 5. points
  const nPnt = (v1 - v0) + SHARED.pnt;
  PNT = { pos: new Float32Array(nPnt * 3), col: new Float32Array(nPnt * 4),
          rad: new Float32Array(nPnt), mat: new Float32Array(nPnt * 4), n: nPnt };
  let i = 0;
  const fld = fieldIdx >= 0 ? FIELDS[fieldIdx].data : null, rng = fld ? fieldRange(pnts) : null;
  for (const [a, b] of pnts) for (let j = a; j < b; j++) {
    if (!layerOn[PNT_L[j]]) continue;
    const v = PNT_V[j], val = fld ? fld[j] : NaN;
    const col = val === val ? fieldColor(val, rng, PAL[PNT_C[j]][3]) : PAL[PNT_C[j]];
    PNT.pos[i * 3] = PR[3 * v]; PNT.pos[i * 3 + 1] = PR[3 * v + 1]; PNT.pos[i * 3 + 2] = PR[3 * v + 2];
    for (let q = 0; q < 4; q++) { PNT.col[i * 4 + q] = col[q]; PNT.mat[i * 4 + q] = PNT_M[j * 4 + q]; }
    PNT.rad[i] = PNT_R[j];
    i++;
  }
  PNT.n = i;
  shownRange = rng;
  showLegend(rng);

  upload();
  pivotDirty = true;                                    // the scene changed: nothing aimed at anymore
  centerRays.length = 0;
  document.getElementById("counts").textContent =
    [[PNT.n, "points"], [LIN.n, "edges"], [TRI.n, "triangles"]].filter(([n]) => n > 0).map(([n, w]) => n.toLocaleString() + " " + w).join(" · ");
  sceneBuilt = true;
  sceneVersion++;                                       // ( a picture of the scene before is out of date )
}
var sceneVersion = 0;
// the FLOOR ( a layer, off at first ): a square under what is shown, three times as wide, across the world axis closest
// to the screen's vertical when it is switched on ( kept then: the floor stays where it was put ), below the lowest point
// less a gap; a picture by Blender gets its corners, and the shadows on it
var floorOn = false, FLOOR = null, floorUp = null, shownRange = null;
function floorOf(polys, edgs, pnts, hp) {
  if (!floorUp) {
    const u = camBasis().up;
    let k = 0;
    for (let q = 1; q < 3; q++) if (Math.abs(u[q]) > Math.abs(u[k])) k = q;
    floorUp = { axis: k, sign: u[k] >= 0 ? 1 : -1 };
  }
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  const add = v => { for (let k = 0; k < 3; k++) { const x = PR[3 * v + k]; if (x < lo[k]) lo[k] = x; if (x > hi[k]) hi[k] = x; } };
  for (const [a, b] of polys) for (let p = a; p < b; p++) for (let o = POLY_OFF[p]; o < POLY_OFF[p + 1]; o++) add(POLY_V[o]);
  for (const [a, b] of edgs) for (let i = a; i < b; i++) { add(EDG_V[2 * i]); add(EDG_V[2 * i + 1]); }
  for (const [a, b] of pnts) for (let j = a; j < b; j++) add(PNT_V[j]);
  for (let i = 0; i + 2 < hp.length; i += 3) for (let k = 0; k < 3; k++) { lo[k] = Math.min(lo[k], hp[i + k]); hi[k] = Math.max(hi[k], hp[i + k]); }
  if (!(lo[0] <= hi[0])) return null;
  const span = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-9), dark = document.body.classList.contains("dark");
  const k = floorUp.axis, s = floorUp.sign, i = (k + 1) % 3, j = (k + 2) % 3, h = 1.5 * span;
  const level = s > 0 ? lo[k] - 0.02 * span : hi[k] + 0.02 * span;
  const ci = (lo[i] + hi[i]) / 2, cj = (lo[j] + hi[j]) / 2;
  const pt = (di, dj) => { const p = [0, 0, 0]; p[k] = level; p[i] = ci + di * h; p[j] = cj + dj * h; return p; };
  const corners = s > 0 ? [pt(-1, -1), pt(1, -1), pt(1, 1), pt(-1, 1)] : [pt(-1, -1), pt(-1, 1), pt(1, 1), pt(1, -1)];  // ( facing up )
  return { axis: k, level, corners, color: dark ? [0.42, 0.42, 0.45, 1] : [0.80, 0.80, 0.80, 1] };
}

// ============================================================================================
// FIELDS: the points coloured by their values. The range is robust ( the 1st and 99th percentiles ), over all the
// frames, over the displayed one, or typed; a range across 0 takes a DIVERGING map ( white at 0 ), else a sequential one.
// ============================================================================================
let fieldIdx = -1;
// the scale: over the displayed frame, over all the frames, or typed ( the inputs ); the button goes from the first to
// the second, and back to the first from a typed one
let scaleMode = "frame";
const fieldUi = { sel: document.getElementById("field"), range: document.getElementById("fieldRange"),
                  btn: document.getElementById("scaleBtn"), lo: document.getElementById("fieldMin"),
                  hi: document.getElementById("fieldMax") };
// the scale's mode, in the Colour pane: this frame, all the frames, typed
function syncScaleBtn() {
  for (const b of document.querySelectorAll("#scaleSeg button")) b.classList.toggle("on", b.dataset.s === scaleMode);
}
function quantiles(vals) {                             // [ q01, q99 ] of the finite values
  const v = Float64Array.from(vals.filter(x => x === x)).sort();
  if (!v.length) return [0, 1];
  const q = t => v[Math.min(v.length - 1, Math.floor(t * (v.length - 1) + 0.5))];
  let lo = q(0.01), hi = q(0.99);
  if (!(hi > lo)) { lo = v[0]; hi = v[v.length - 1] > lo ? v[v.length - 1] : lo + 1; }
  return [lo, hi];
}
function fieldRange(pnts) {
  const f = FIELDS[fieldIdx], mode = scaleMode;
  let r;
  const num = inp => parseFloat(inp.value.replace(",", "."));              // ( a comma as a decimal point too )
  if (mode === "manual") r = [num(fieldUi.lo), num(fieldUi.hi)];
  else if (mode === "all") r = f.all || (f.all = quantiles(f.data));
  else {
    const vals = [];
    for (const [a, b] of pnts) for (let j = a; j < b; j++) vals.push(f.data[j]);
    r = quantiles(vals);
  }
  if (mode !== "manual" && isDiverging(f)) { const a = Math.max(Math.abs(r[0]), Math.abs(r[1])); r = [-a, a]; }  // ( white at 0 )
  if (!(r[1] > r[0])) r = [r[0], r[0] + 1];
  if (mode !== "manual") { fieldUi.lo.value = +r[0].toPrecision(3); fieldUi.hi.value = +r[1].toPrecision(3); }
  return r;
}
function ramp(stops, t) {
  t = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const k = Math.min(stops.length - 2, Math.floor(t)), u = t - k;
  return stops[k].map((c, q) => c + u * (stops[k + 1][q] - c));
}
const SEQ = [[0.267, 0.005, 0.329], [0.230, 0.322, 0.546], [0.128, 0.567, 0.551], [0.369, 0.789, 0.383], [0.993, 0.906, 0.144]];
const DIV = [[0.230, 0.299, 0.754], [0.552, 0.690, 0.996], [0.865, 0.865, 0.865], [0.958, 0.604, 0.482], [0.706, 0.016, 0.150]];
// the MAP of a field is chosen once, from its values over all the frames: diverging when they take both signs, else
// sequential. Whatever the range ( typed or not ), its colours go from the start of the map at `min` to its end at `max`
function isDiverging(f) {
  if (f.div === undefined) { const q = f.all || (f.all = quantiles(f.data)); f.div = q[0] < 0 && q[1] > 0; }
  return f.div;
}
function fieldColor(v, r, alpha) {
  const c = ramp(isDiverging(FIELDS[fieldIdx]) ? DIV : SEQ, (v - r[0]) / (r[1] - r[0]));
  return [c[0], c[1], c[2], alpha];
}
function showLegend(r) {
  const chip = document.getElementById("legend");
  chip.style.display = r ? "flex" : "none";
  if (!r) return;
  const div = isDiverging(FIELDS[fieldIdx]);
  for (const id of ["legendBar", "legendBar2"]) {
    const cv = document.getElementById(id), ctx = cv.getContext("2d");
    for (let x = 0; x < cv.width; x++) {
      const c = ramp(div ? DIV : SEQ, x / (cv.width - 1));
      ctx.fillStyle = "rgb(" + c.map(q => Math.round(255 * q)).join(",") + ")";
      ctx.fillRect(x, 0, 1, cv.height);
    }
    cv.title = div ? "diverging: white in the middle of the range" : "";
  }
  document.getElementById("legendName").textContent = FIELDS[fieldIdx].name;
  document.getElementById("legendLo").textContent = fmtNum(r[0]);
  document.getElementById("legendHi").textContent = fmtNum(r[1]);
}
const fmtNum = v => v === 0 ? "0" : (Math.abs(v) >= 1e4 || Math.abs(v) < 1e-3 ? v.toExponential(2) : String(parseFloat(v.toPrecision(3))));

// LAYERS, in the Display pane: one row each -- the eye, a mark, the name, what it holds; the faces and the points fold
// their own settings ( the controls of `ui`, moved there ). Keys 1-9 switch them in that order.
const EYE = '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M1.5 10S5 4 10 4s8.5 6 8.5 6-3.5 6-8.5 6-8.5-6-8.5-6z"/><circle cx="10" cy="10" r="2.6"/></svg>';
const EYE_OFF = '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M3 3l14 14M8 5.2A9 9 0 0 1 10 5c5 0 8.5 5 8.5 5a14 14 0 0 1-2.6 3M5.5 6.6C3 8.2 1.5 10 1.5 10S5 16 10 16a8 8 0 0 0 3.2-.7"/></svg>';
const CHEV = '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M8 5l5 5-5 5"/></svg>';
const layerRows = [];
// a row: `get()` / `set( on )` its state, `what()` what it holds, `mark` a css background, `sub` its folded settings
function addLayer(name, get, set, what, mark, sub, boxId, line) {
  const box = document.getElementById(boxId || "layers");
  const row = document.createElement("div"); row.className = "item";
  const eye = document.createElement("button"); eye.className = "eye";
  const mk = document.createElement("span"); mk.className = "mark" + (line ? " line" : ""); mk.style.background = mark || "var(--muted)";
  const nm = document.createElement("span"); nm.className = "name"; nm.textContent = name;
  const wh = document.createElement("span"); wh.className = "what";
  const more = document.createElement("button"); more.className = "more";
  row.append(more, mk, nm, wh, eye);
  box.appendChild(row);
  let pane = null;
  if (sub) { pane = document.createElement("div"); pane.className = "sub"; sub(pane); box.appendChild(pane); more.innerHTML = CHEV; more.title = "its settings"; }
  else more.style.visibility = "hidden";
  const sync = () => {
    const on = get();
    eye.innerHTML = on ? EYE : EYE_OFF; eye.title = (on ? "hide " : "show ") + name;
    row.classList.toggle("off", !on);
    wh.textContent = what ? what() : "";
  };
  eye.addEventListener("click", () => { set(!get()); sync(); });
  more.addEventListener("click", () => { const o = !pane.classList.contains("open"); pane.classList.toggle("open", o); row.classList.toggle("open", o); });
  const item = { name, get, set, sync, row, mk };
  if (!boxId) layerRows.push(item);
  sync();
  return item;
}
function syncLayers() { for (const l of layerRows) l.sync(); }
// a segmented control over `options` ( [ value, label ] ), `get()` the current one, `set( v )` on a click
function segmented(options, get, set) {
  const seg = document.createElement("div"); seg.className = "seg";
  const btns = options.map(([v, label]) => {
    const b = document.createElement("button"); b.textContent = label;
    b.addEventListener("click", () => { set(v); sync(); });
    seg.appendChild(b); return [v, b];
  });
  const sync = () => { for (const [v, b] of btns) b.classList.toggle("on", get() === v); };
  sync(); seg.sync = sync;
  return seg;
}
function sliderRow(label, input, valId) {
  const row = document.createElement("div"); row.className = "row";
  const l = document.createElement("span"); l.textContent = label;
  const v = document.getElementById(valId) || document.createElement("span"); v.className = "val";
  row.append(l, input, v);
  return row;
}
let edgeSeg = null;
function initTools() {                                 // ( called once `ui` exists )
  const any = arr => arr && arr.length > 0;
  const stored = k => STORED && STORE.m.frames.some(f => f.n && f.n[k]);
  if (POLY_OFF.length > 1 || HPOLY.length || stored("poly"))
    addLayer("faces", () => ui.faces.checked, on => { ui.faces.checked = on; draw(); },
             () => TRI.n ? TRI.n + " triangles" : "", "linear-gradient(135deg,#5ec962,#3b528b,#c4621a)", pane => {
      pane.appendChild(sliderRow("opacity", ui.op, "opVal"));
      edgeSeg = segmented([["hide", "hidden"], ["ghost", "ghost"], ["all", "all"], ["none", "none"]],
                          () => ui.edgeMode.value, v => { ui.edgeMode.value = v; draw(); });
      const r = document.createElement("div"); r.className = "row";
      const l = document.createElement("span"); l.textContent = "edges"; edgeSeg.style.gridColumn = "span 2";
      r.append(l, edgeSeg); pane.appendChild(r);
    });
  if (PNT_V.length || stored("pnt"))
    addLayer("points", () => ui.points.checked, on => { ui.points.checked = on; draw(); },
             () => PNT.n ? PNT.n + " points" : "", "var(--text)", pane => {
      pane.appendChild(sliderRow("radius", ui.r, "rVal"));
      if (FIELDS.length) {
        const r = document.createElement("div"); r.className = "row";
        const l = document.createElement("span"); l.textContent = "colour";
        const sel = document.createElement("select"); sel.style.gridColumn = "span 2"; sel.id = "pointField";
        sel.add(new Option("their own colours", -1));
        FIELDS.forEach((f, k) => sel.add(new Option(f.name, k)));
        sel.addEventListener("change", () => { fieldIdx = parseInt(sel.value, 10); fieldUi.sel.value = sel.value; syncFieldPane(); buildScene(); draw(); });
        r.append(l, sel); pane.appendChild(r);
      }
      if (D >= 3) { const c = document.createElement("label"); c.className = "chk"; c.append(ui.depth, document.createTextNode(" coloured by depth")); pane.appendChild(c); }
    });
  LAYERS.forEach((l, k) => addLayer(l[0], () => layerOn[k + 1], on => { layerOn[k + 1] = on; buildScene(); draw(); }, null, "var(--accent)"));
  if (D >= 3 && !FLAT)                                  // a floor under the scene ( z down ), for its shadow in pictures
    addLayer("floor", () => floorOn, on => { floorOn = on; if (on) floorUp = null; buildScene(); draw(); }, () => floorOn && FLOOR ? "xyz"[FLOOR.axis] + " = " + fmtNum(FLOOR.level) : "",
             "linear-gradient(180deg, transparent 55%, var(--muted) 55%)");
  if (!layerRows.length) document.getElementById("layers").style.display = "none";
  // the Colour pane: one choice per field, the range
  const list = document.getElementById("fieldList");
  const choice = (label, k) => {
    const lab = document.createElement("label"); lab.className = "chk";
    const rb = document.createElement("input"); rb.type = "radio"; rb.name = "field"; rb.value = k;
    rb.addEventListener("change", () => { fieldIdx = k; fieldUi.sel.value = String(k); syncFieldPane(); buildScene(); draw(); });
    lab.append(rb, document.createTextNode(" " + label)); list.appendChild(lab);
  };
  if (FIELDS.length) {
    choice("their own colours", -1);
    FIELDS.forEach((f, k) => {
      choice(f.name, k);
      const o = document.createElement("option"); o.value = k; o.textContent = f.name; fieldUi.sel.appendChild(o);
    });
    fieldIdx = FIELDS.findIndex(f => f.name === FIELD0);                 // ( -1: plain colours )
    fieldUi.sel.value = String(fieldIdx);
    const rebuild = () => { buildScene(); draw(); };
    for (const b of document.querySelectorAll("#scaleSeg button"))
      b.addEventListener("click", () => { scaleMode = b.dataset.s; syncScaleBtn(); rebuild(); });
    for (const inp of [fieldUi.lo, fieldUi.hi])
      inp.addEventListener("change", () => { scaleMode = "manual"; syncScaleBtn(); rebuild(); });
    syncScaleBtn(); syncFieldPane();
  }
}
function syncFieldPane() {
  for (const rb of document.querySelectorAll('#fieldList input')) rb.checked = parseInt(rb.value, 10) === fieldIdx;
  const pf = document.getElementById("pointField"); if (pf) pf.value = String(fieldIdx);
  document.getElementById("fieldRange").style.display = fieldIdx >= 0 ? "flex" : "none";
}
function syncTools() { syncLayers(); }
function placeTools() {}

// ============================================================================================
// linear algebra (just what the camera needs)
// ============================================================================================
function mul(a, e) {                                   // a * e, 4x4 matrices in column-major
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0;
    for (let k = 0; k < 4; k++) s += a[k * 4 + r] * e[c * 4 + k];
    o[c * 4 + r] = s;
  }
  return o;
}
function perspective(fov, asp, n, f) {
  const t = 1 / Math.tan(fov / 2), o = new Float32Array(16);
  o[0] = t / asp; o[5] = t; o[10] = (f + n) / (n - f); o[11] = -1; o[14] = 2 * f * n / (n - f);
  return o;
}
function ortho(hw, hh, n, f) {
  const o = new Float32Array(16);
  o[0] = 1 / hw; o[5] = 1 / hh; o[10] = 2 / (n - f); o[14] = (f + n) / (n - f); o[15] = 1;
  return o;
}

// ============================================================================================
// camera
// ============================================================================================
const FLAT = D <= 2;                                   // 2D: flat, no orbit
const FOV = 45 * Math.PI / 180;

// The camera's orientation is a QUATERNION, not an (azimuth, elevation) pair. There is no
// reference plane here -- no ground, no world "up": two Euler angles would
// impose one, and the same gesture would then not produce the same motion depending on the starting
// orientation (squashing near the poles, elevation stop, impossible roll). With a quaternion,
// a drag always applies the SAME rotation, expressed in the SCREEN frame.
const cam = { target: [0, 0, 0], dist: 3, rot: [0, 0, 0, 1], pivotZ: 3, ortho: FLAT };

function qMul(a, b) {                                  // Hamilton product: `b` first, then `a`
  return [a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
          a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
          a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
          a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2]];
}
function qAxis(ax, ang) {                              // rotation by angle `ang` around `ax`
  const h = 0.5 * ang, s = Math.sin(h);
  return [ax[0] * s, ax[1] * s, ax[2] * s, Math.cos(h)];
}
function qNorm(q) {                                    // rounding drifts: renormalize
  const l = Math.hypot(q[0], q[1], q[2], q[3]) || 1;
  return [q[0] / l, q[1] / l, q[2] / l, q[3] / l];
}
function qApply(q, v) {                                // v rotated by q
  const t = [2 * (q[1] * v[2] - q[2] * v[1]),
             2 * (q[2] * v[0] - q[0] * v[2]),
             2 * (q[0] * v[1] - q[1] * v[0])];
  return [v[0] + q[3] * t[0] + q[1] * t[2] - q[2] * t[1],
          v[1] + q[3] * t[1] + q[2] * t[0] - q[0] * t[2],
          v[2] + q[3] * t[2] + q[0] * t[1] - q[1] * t[0]];
}

function sceneSphere() {                               // center + radius of the VIEWED dimensions
  const c = [0, 0, 0]; let r = 0;
  for (let a = 0; a < 3; a++) {
    const k = AX[a];
    const lo = k >= 0 ? BOUNDS[k][0] : 0, hi = k >= 0 ? BOUNDS[k][1] : 0;
    c[a] = 0.5 * (lo + hi); r += 0.25 * (hi - lo) * (hi - lo);
  }
  return { c, r: Math.max(Math.sqrt(r), 1e-6) };
}
// the box the camera frames at the start: the TIGHT one of the viewed dimensions ( BOUNDS has a margin for the sliders )
const TIGHT = __TIGHT__, FIT_MARGIN = 0.04;
function resetView() {
  const s = sceneSphere();
  // three-quarter view by default -- a starting orientation like any other, with no special
  // status: nothing later takes it up as a reference.
  cam.rot = FLAT ? [0, 0, 0, 1]
                 : qNorm(qMul(qAxis([0, 1, 0], 0.6), qAxis([1, 0, 0], -0.35)));
  // FIT THE BOX, not its sphere: its corners in the screen frame of that orientation, the target at their middle, and the
  // distance at which the farthest one touches the border ( the field of view is vertical, `asp` widens it )
  const asp = canvas.width / Math.max(canvas.height, 1);
  const b = camBasis(), corners = [];
  for (let m = 0; m < 8; m++) {
    const p = [0, 1, 2].map(a => AX[a] >= 0 ? TIGHT[AX[a]][(m >> a) & 1] : 0);
    corners.push(p);
  }
  const proj = corners.map(p => [b.right, b.up, b.fwd].map(e => e[0] * p[0] + e[1] * p[1] + e[2] * p[2]));
  const lo = [0, 1, 2].map(k => Math.min(...proj.map(q => q[k]))), hi = [0, 1, 2].map(k => Math.max(...proj.map(q => q[k])));
  const mid = [0, 1, 2].map(k => 0.5 * (lo[k] + hi[k]));
  cam.target = [0, 1, 2].map(i => mid[0] * b.right[i] + mid[1] * b.up[i] + mid[2] * b.fwd[i]);
  const t = Math.tan(FOV / 2) * (1 - FIT_MARGIN);
  let dist = 0;
  for (const q of proj) {
    const x = Math.abs(q[0] - mid[0]), y = Math.abs(q[1] - mid[1]);
    const zf = FLAT ? 0 : -(q[2] - mid[2]);                 // ( how far the corner stands in front of the target )
    dist = Math.max(dist, zf + x / (t * asp), zf + y / t);
  }
  cam.dist = Math.max(dist, 1e-6 * s.r);
  cam.pivotZ = cam.dist; pivotDirty = true; centerRays.length = 0;
  draw();
}
function camEye() {
  const b = qApply(cam.rot, [0, 0, 1]);                // the axis that "comes out" of the screen
  return [cam.target[0] + cam.dist * b[0],
          cam.target[1] + cam.dist * b[1],
          cam.target[2] + cam.dist * b[2]];
}
function camBasis() {                                  // right / up of the screen, in world
  return { right: qApply(cam.rot, [1, 0, 0]),
           up:    qApply(cam.rot, [0, 1, 0]),
           fwd:   qApply(cam.rot, [0, 0, -1]) };
}
function viewMatrix() {                                // world -> camera, taken directly from `rot`
  // no `lookAt`: it would need a world "up", exactly the assumption we refuse
  // (and it would degenerate when the camera looks straight at it).
  const b = camBasis(), e = camEye(), r = b.right, u = b.up, f = b.fwd;
  return new Float32Array([
    r[0], u[0], -f[0], 0,
    r[1], u[1], -f[1], 0,
    r[2], u[2], -f[2], 0,
    -(r[0] * e[0] + r[1] * e[1] + r[2] * e[2]),
    -(u[0] * e[0] + u[1] * e[1] + u[2] * e[2]),
      f[0] * e[0] + f[1] * e[1] + f[2] * e[2], 1,
  ]);
}
function orthoHalfH() { return cam.dist * Math.tan(FOV / 2); }

// ============================================================================================
// aiming: which point of the scene lies under a pixel
// ============================================================================================
// Zooming towards the cursor and the rotation pivot need a 3D point, not a pixel. Reading it
// from the depth buffer is out of reach in WebGL2 (`readPixels` can only read back a
// color), and an extra render pass for that would be costly. So we cast a ray into the
// primitives of the displayed frame, which are ALREADY there on the CPU side (`TRI`/`LIN`/`PNT`):
// linear cost, but paid once per GESTURE, never per rendered frame.

function aimPlane(sx, sy) {                            // aimed point, in the target's plane
  const b = camBasis(), hh = orthoHalfH();
  const w = Math.max(canvas.clientWidth, 1), h = Math.max(canvas.clientHeight, 1);
  const u = (2 * sx / w - 1) * hh * (w / h), v = (1 - 2 * sy / h) * hh;
  return [cam.target[0] + u * b.right[0] + v * b.up[0],
          cam.target[1] + u * b.right[1] + v * b.up[1],
          cam.target[2] + u * b.right[2] + v * b.up[2]];
}
function screenRay(sx, sy) {
  const b = camBasis(), e = camEye(), p = aimPlane(sx, sy);
  if (cam.ortho)                                       // parallel rays: it is the origin that moves
    return { o: [p[0] - cam.dist * b.fwd[0], p[1] - cam.dist * b.fwd[1],
                 p[2] - cam.dist * b.fwd[2]], d: b.fwd };
  const d = [p[0] - e[0], p[1] - e[1], p[2] - e[2]];
  const l = Math.hypot(d[0], d[1], d[2]) || 1;
  return { o: e, d: [d[0] / l, d[1] / l, d[2] / l] };
}

const AIM_PX = 18;                                     // aiming tolerance, in pixels

function pickWorld(sx, sy) {
  const r = screenRay(sx, sy), o = r.o, d = r.d;
  const h = Math.max(canvas.clientHeight, 1);
  // width of a pixel, in world units, at distance `t` along the ray
  const wpx = cam.ortho ? () => 2 * orthoHalfH() / h
                        : t => 2 * Math.tan(FOV / 2) * t / h;

  // 1. faces first: EXACT intersection (Möller-Trumbore). This is what the eye sees.
  let best = Infinity;
  const P = TRI.pos;
  for (let i = 0; i < TRI.n; i++) {
    const j = i * 9;
    const ax = P[j], ay = P[j + 1], az = P[j + 2];
    const e1x = P[j + 3] - ax, e1y = P[j + 4] - ay, e1z = P[j + 5] - az;
    const e2x = P[j + 6] - ax, e2y = P[j + 7] - ay, e2z = P[j + 8] - az;
    const hx = d[1] * e2z - d[2] * e2y, hy = d[2] * e2x - d[0] * e2z, hz = d[0] * e2y - d[1] * e2x;
    const det = e1x * hx + e1y * hy + e1z * hz;
    if (det > -1e-12 && det < 1e-12) continue;         // ray parallel to the triangle's plane
    const f = 1 / det, wx = o[0] - ax, wy = o[1] - ay, wz = o[2] - az;
    const u = f * (wx * hx + wy * hy + wz * hz);
    if (u < 0 || u > 1) continue;
    const qx = wy * e1z - wz * e1y, qy = wz * e1x - wx * e1z, qz = wx * e1y - wy * e1x;
    const v = f * (d[0] * qx + d[1] * qy + d[2] * qz);
    if (v < 0 || u + v > 1) continue;
    const t = f * (e2x * qx + e2y * qy + e2z * qz);
    if (t > 1e-9 && t < best) best = t;
  }

  // 2. otherwise edges and points: nothing SOLID to pierce (wireframe, cloud), so we accept what
  //    passes close enough to the ray -- tolerance in pixels, so that it does not depend on zoom.
  if (best === Infinity) {
    const A = LIN.a, B = LIN.b;
    for (let i = 0; i < LIN.n; i++) {
      const k = i * 3;
      const vx = B[k] - A[k], vy = B[k + 1] - A[k + 1], vz = B[k + 2] - A[k + 2];
      const wx = o[0] - A[k], wy = o[1] - A[k + 1], wz = o[2] - A[k + 2];
      const bb = d[0] * vx + d[1] * vy + d[2] * vz, cc = vx * vx + vy * vy + vz * vz;
      const dd = d[0] * wx + d[1] * wy + d[2] * wz, ee = vx * wx + vy * wy + vz * wz;
      const den = cc - bb * bb;                        // (|d| = 1); zero <=> ray and edge parallel
      let s = Math.abs(den) < 1e-12 ? 0 : (ee - bb * dd) / den;
      s = Math.max(0, Math.min(1, s));
      const px2 = A[k] + s * vx, py2 = A[k + 1] + s * vy, pz2 = A[k + 2] + s * vz;
      const t = (px2 - o[0]) * d[0] + (py2 - o[1]) * d[1] + (pz2 - o[2]) * d[2];
      if (t <= 1e-9 || t >= best) continue;
      if (Math.hypot(px2 - o[0] - t * d[0], py2 - o[1] - t * d[1], pz2 - o[2] - t * d[2])
          < AIM_PX * wpx(t)) best = t;
    }
    const Q = PNT.pos, rad = PNT.rad;
    for (let i = 0; i < PNT.n; i++) {
      const k = i * 3;
      const t = (Q[k] - o[0]) * d[0] + (Q[k + 1] - o[1]) * d[1] + (Q[k + 2] - o[2]) * d[2];
      if (t <= 1e-9 || t >= best) continue;
      if (Math.hypot(Q[k] - o[0] - t * d[0], Q[k + 1] - o[1] - t * d[1], Q[k + 2] - o[2] - t * d[2])
          < rad[i] + AIM_PX * wpx(t)) best = t;
    }
  }
  return best === Infinity ? null
                           : [o[0] + best * d[0], o[1] + best * d[1], o[2] + best * d[2]];
}

function depthAt(sx, sy) {                             // depth of the aimed point, along fwd
  const c = pickWorld(sx, sy);
  if (c === null) return cam.dist;                     // background: for lack of better, the target's plane
  const e = camEye(), f = camBasis().fwd;
  return Math.max(1e-6, (c[0] - e[0]) * f[0] + (c[1] - e[1]) * f[1] + (c[2] - e[2]) * f[2]);
}
function aimAt(sx, sy, z) {                            // the point of the ray located at depth z
  const r = screenRay(sx, sy), f = camBasis().fwd;
  const c = r.d[0] * f[0] + r.d[1] * f[1] + r.d[2] * f[2];   // 1 in orthographic
  const e = camEye(), o = r.o;
  const t = (z - ((o[0] - e[0]) * f[0] + (o[1] - e[1]) * f[1] + (o[2] - e[2]) * f[2])) / c;
  return [o[0] + t * r.d[0], o[1] + t * r.d[1], o[2] + t * r.d[2]];
}

// Depth of the rotation pivot. The pivot always lies ON the center-of-screen ray; only its depth is
// in question. It is the best of three estimates:
//  1. a TRIANGULATION from the center rays recorded at the start of the previous rotations: the user
//     centers what interests them, rotates, re-centers it from another angle -- the rays then
//     cross at that very point, in all 3 axes, after a few rotations/moves. Stale rays (the user
//     moved on to something else) pass far from the candidate and are discounted (IRLS);
//  2. otherwise what the scene shows at the center of the screen (solid pick);
//  3. otherwise the previous depth `cam.dist`.
// It is only recomputed if the view moved OTHERWISE than by a rotation -- a rotation must above
// all not redefine its own pivot mid-gesture, it would drift under the hand.
const centerRays = [];                                 // lines { o, d } through the screen center
const MAX_RAYS = 8;
function dot3(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function perp(v, u) { const c = dot3(v, u); return [v[0] - c * u[0], v[1] - c * u[1], v[2] - c * u[2]]; }
function recordCenterRay(e, f) {
  const s = sceneSphere(), last = centerRays[centerRays.length - 1];
  const L = { o: e.slice(), d: f.slice() };
  if (last && dot3(last.d, f) > 0.99999 &&
      Math.hypot(...perp([e[0] - last.o[0], e[1] - last.o[1], e[2] - last.o[2]], last.d)) < 1e-3 * s.r)
    centerRays[centerRays.length - 1] = L;             // same line again: just refresh it
  else centerRays.push(L);
  if (centerRays.length > MAX_RAYS) centerRays.shift();
}
function triangulateDepth(e, f) {                      // depth along `f` from `e`, or null
  const n = centerRays.length;
  if (n === 0) return null;
  const sigma = 0.03 * sceneSphere().r;
  const base = centerRays.map((_, i) => Math.pow(0.8, n - 1 - i));   // recent rays count more
  let w = base.slice(), t = 0;
  for (let it = 0; it < 4; it++) {
    let num = 0, den = 0;
    for (let i = 0; i < n; i++) {
      const L = centerRays[i], pf = perp(f, L.d);
      const pw = perp([e[0] - L.o[0], e[1] - L.o[1], e[2] - L.o[2]], L.d);
      num -= w[i] * dot3(pw, pf); den += w[i] * dot3(pf, pf);
    }
    if (den < 0.03) return null;                       // rays (nearly) parallel: no depth information
    t = num / den;
    for (let i = 0; i < n; i++) {
      const L = centerRays[i];
      const q = perp([e[0] + t * f[0] - L.o[0], e[1] + t * f[1] - L.o[1], e[2] + t * f[2] - L.o[2]], L.d);
      w[i] = base[i] / (1 + dot3(q, q) / (sigma * sigma));
    }
  }
  return t > 1e-6 && t < 20 * (cam.dist + sceneSphere().r) ? t : null;
}
let pivotDirty = true;
function pivotDepth(record) {
  if (pivotDirty) {
    pivotDirty = false;
    const e = camEye(), f = camBasis().fwd;
    const zt = triangulateDepth(e, f);
    if (record) recordCenterRay(e, f);
    cam.pivotZ = zt !== null ? zt : depthAt(canvas.clientWidth / 2, canvas.clientHeight / 2);
  }
  return cam.pivotZ;
}

// ============================================================================================
// WebGL
// ============================================================================================
const canvas = document.getElementById("c");
const gl = canvas.getContext("webgl2", { antialias: true, alpha: false });
if (!gl) document.body.innerHTML = "<p style='font-family:sans-serif;padding:20px'>WebGL2 "
  + "unavailable in this browser.</p>";

function prog(vs, fs) {
  function sh(t, src) {
    const s = gl.createShader(t); gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw gl.getShaderInfoLog(s);
    return s;
  }
  const p = gl.createProgram();
  gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw gl.getProgramInfoLog(p);
  return p;
}

const pFace = prog(`#version 300 es
in vec3 aPos; in vec3 aNrm; in vec4 aCol; in float aAo;
uniform mat4 uMVP; uniform mat4 uMV;
out vec4 vCol; out vec3 vN; out float vAo;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); vN = mat3(uMV) * aNrm; vCol = aCol; vAo = aAo; }`,
`#version 300 es
precision highp float;
in vec4 vCol; in vec3 vN; in float vAo; out vec4 oCol;
uniform float uAlpha; uniform float uBake;
void main() {
  // frontal lighting, DOUBLE-SIDED (abs): the inside of an open cell stays readable. The baked occlusion ( 1 = open )
  // darkens the corners of the faces that something stands in front of, by uBake ( no backquote: this is a JS template ).
  vec3 n = normalize(vN);
  float d = abs(dot(n, normalize(vec3(0.25, 0.45, 1.0))));
  oCol = vec4(vCol.rgb * (0.45 + 0.55 * d) * mix(1.0, vAo, uBake), vCol.a * uAlpha);
}`);

const pLine = prog(`#version 300 es
in vec2 aCorner; in vec3 aA; in vec3 aB; in vec4 aCol;
uniform mat4 uMVP; uniform vec2 uHalfVP; uniform float uW;
out vec4 vCol;
void main() {
  // an edge = one INSTANCE widened into a quadrilateral on SCREEN (gl.lineWidth does not exceed 1px
  // on most drivers).
  vec4 ca = uMVP * vec4(aA, 1.0), cb = uMVP * vec4(aB, 1.0);
  vec2 sa = ca.xy / max(ca.w, 1e-5) * uHalfVP, sb = cb.xy / max(cb.w, 1e-5) * uHalfVP;
  vec2 dir = sb - sa;
  float l = length(dir);
  dir = l > 1e-6 ? dir / l : vec2(1.0, 0.0);
  vec4 c = mix(ca, cb, aCorner.x);
  c.xy += vec2(-dir.y, dir.x) * (uW * 0.5 * aCorner.y) / uHalfVP * max(c.w, 1e-5);
  gl_Position = c; vCol = aCol;
}`,
`#version 300 es
precision highp float;
in vec4 vCol; out vec4 oCol; uniform float uAlpha;
void main() { oCol = vec4(vCol.rgb, vCol.a * uAlpha); }`);

const pPoint = prog(`#version 300 es
in vec3 aPos; in vec4 aCol; in float aRad; in vec4 aMat;
uniform mat4 uMVP; uniform mat4 uMV; uniform float uSizeK; uniform float uOrtho; uniform float uR; uniform float uRScale;
uniform float uDepthCol; uniform vec2 uDepthRange; uniform vec2 uHalfVP;
out vec4 vCol; out vec3 vC; out float vR; out vec2 vD;
// a point = one INSTANCE, a square on SCREEN of its radius ( gl_PointSize is capped by the drivers, at 64 to 511 pixels:
// zooming in on a point with a radius of its own, it stopped growing )
const vec2 CORNER[6] = vec2[6](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
// the depth colors: warm in front, cool behind
vec3 depthRamp(float t) {
  const vec3 c0 = vec3(0.85, 0.22, 0.16), c1 = vec3(0.97, 0.70, 0.22), c2 = vec3(0.30, 0.68, 0.62), c3 = vec3(0.20, 0.28, 0.62);
  return t < 1.0 / 3.0 ? mix(c0, c1, 3.0 * t) : t < 2.0 / 3.0 ? mix(c1, c2, 3.0 * t - 1.0) : mix(c2, c3, 3.0 * t - 2.0);
}
void main() {
  vec4 c = uMVP * vec4(aPos, 1.0);
  // a zero radius = a point with no size of its own (a Dirac): the slider sizes it.
  // A point with its own radius keeps it at the slider's start, and the slider scales it.
  float r = aRad > 0.0 ? aRad * uRScale : uR;
  float zv = -(uMV * vec4(aPos, 1.0)).z;
  float px = max(uSizeK * r / (uOrtho > 0.5 ? 1.0 : max(zv, 1e-4)), 0.5);      // its radius on screen: half a pixel at least
  vD = CORNER[gl_VertexID];
  if (aMat != vec4(0.0)) {                             // an ELLIPSE: the quad deformed by its matrix, in the world
    vec2 o = vec2(aMat.x * vD.x + aMat.y * vD.y, aMat.z * vD.x + aMat.w * vD.y) * uRScale;
    c = uMVP * vec4(aPos + vec3(o, 0.0), 1.0);
  } else
    c.xy += vD * px / uHalfVP * max(c.w, 1e-5);
  gl_Position = c;
  vC = (uMV * vec4(aPos, 1.0)).xyz; vR = r;
  vCol = uDepthCol > 0.5 ? vec4(depthRamp(clamp((zv - uDepthRange.x) / (uDepthRange.y - uDepthRange.x), 0.0, 1.0)), aCol.a) : aCol;
}`,
`#version 300 es
precision highp float;
in vec4 vCol; in vec3 vC; in float vR; in vec2 vD; out vec4 oCol;
uniform mat4 uP; uniform float uSphere;
const float FLATPTS = ${__FLATPTS__ ? "1.0" : "0.0"};
// a light from the upper left, in front ( view space )
const vec3 L = vec3(-0.45, 0.55, 0.70);
void main() {
  vec2 d = vec2(vD.x, -vD.y);
  float q = dot(d, d);
  if (q > 1.0) discard;                                // disc, not square
  if (uSphere > 0.5) {
    // a true SPHERE ( 3D ): the depth of its surface at this pixel, so that spheres cut into one another and the
    // ambient occlusion sees relief, and a diffuse + specular light on its normal
    vec3 n = vec3(d.x, -d.y, sqrt(max(0.0, 1.0 - q)));
    vec4 c = uP * vec4(vC + vR * n, 1.0);
    gl_FragDepth = 0.5 * c.z / c.w + 0.5;
    vec3 l = normalize(L);
    float diff = max(dot(n, l), 0.0), spec = pow(max(dot(reflect(-l, n), vec3(0.0, 0.0, 1.0)), 0.0), 24.0);
    oCol = vec4(vCol.rgb * (0.32 + 0.68 * diff) + 0.28 * spec, vCol.a);
  } else {
    gl_FragDepth = gl_FragCoord.z;
    float l = FLATPTS > 0.5 ? 1.0 : 0.55 + 0.45 * sqrt(max(0.0, 1.0 - q));   // sphere shading, or a flat disc
    oCol = vec4(vCol.rgb * l, vCol.a);
  }
}`);

// EYE-DOME LIGHTING (3D): the scene is drawn into a texture, then each pixel is darkened by how far its
// neighbours stand IN FRONT of it -- the silhouettes and the steps in depth come out, with no normal and no
// light to set up. It is what point cloud viewers use (ParaView, CloudCompare, Potree): a lone dirac is a disc
// with no surface around it, which neither the sphere shading nor an ambient occlusion can give a depth to.
const pEdl = prog(`#version 300 es
void main() {                                          // one triangle that covers the screen
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`,
`#version 300 es
precision highp float;
uniform sampler2D uCol; uniform sampler2D uDep;
uniform float uNear, uFar, uOrtho, uRef, uK, uRad;
uniform float uAo, uAoRad, uSizeK;
out vec4 oCol;
float viewDepth(float d) {                             // the distance along the view axis
  float z = 2.0 * d - 1.0;
  return uOrtho > 0.5 ? z * uFar : 2.0 * uNear * uFar / (uFar + uNear - z * (uFar - uNear));
}
void main() {
  ivec2 c = ivec2(gl_FragCoord.xy), sz = textureSize(uDep, 0);
  float zc = viewDepth(texelFetch(uDep, c, 0).r);
  // in perspective, the differences are RELATIVE to the depth (a log depth): the same object looks the
  // same near and far. In orthographic, relative to the size of the scene.
  float ref = uOrtho > 0.5 ? uRef : zc;
  float s = 0.0;
  for (int i = 0; i < 8; i++) {
    float a = 0.7853982 * float(i);
    ivec2 q = clamp(c + ivec2(round(uRad * vec2(cos(a), sin(a)))), ivec2(0), sz - 1);
    s += max(0.0, zc - viewDepth(texelFetch(uDep, q, 0).r)) / ref;
  }
  vec4 col = texelFetch(uCol, c, 0);
  float shade = exp(-uK * s / 8.0);
  // AMBIENT OCCLUSION ( screen space, from the depths only ): the neighbours within uAoRad ( world ) that stand in
  // front darken the pixel, less and less as they stand farther in front -- the hollows and contacts come out
  if (uAo > 0.0 && texelFetch(uDep, c, 0).r < 0.99999) {
    float rpx = uAoRad * uSizeK / (uOrtho > 0.5 ? 1.0 : max(zc, 1e-6));
    float occ = 0.0;
    for (int i = 0; i < 24; i++) {
      float t = (float(i) + 0.5) / 24.0;
      float a = 2.3999632 * float(i);                  // the golden angle: a spiral
      ivec2 q = clamp(c + ivec2(round(rpx * sqrt(t) * vec2(cos(a), sin(a)))), ivec2(0), sz - 1);
      float dz = zc - viewDepth(texelFetch(uDep, q, 0).r);
      occ += dz > 0.05 * uAoRad ? clamp(2.0 - dz / (2.0 * uAoRad), 0.0, 1.0) : 0.0;
    }
    shade *= 1.0 - uAo * occ / 24.0;
  }
  oCol = vec4(col.rgb * shade, 1.0);
}`);
const edl = { fb: null, col: null, dep: null, w: 0, h: 0, vao: gl.createVertexArray() };
function edlTargets(w, h) {                            // (re)allocated when the canvas changes size
  if (edl.w === w && edl.h === h) return;
  if (edl.fb) { gl.deleteFramebuffer(edl.fb); gl.deleteTexture(edl.col); gl.deleteTexture(edl.dep); }
  const tex = (internal, format, type) => {
    const t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    return t;
  };
  edl.col = tex(gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE);
  edl.dep = tex(gl.DEPTH_COMPONENT24, gl.DEPTH_COMPONENT, gl.UNSIGNED_INT);
  edl.fb = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, edl.fb);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, edl.col, 0);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.TEXTURE_2D, edl.dep, 0);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  edl.w = w; edl.h = h;
}
function edlStrength() { return 20 * Math.pow(100, parseFloat(ui.edlK.value)); }
function aoRadius() { return parseFloat(ui.aoR.value) * pointRadius(); }

// IMAGES ( `add_image` ): a textured rectangle in the plane z = 0, grey levels on one channel
const pImage = prog(`#version 300 es
uniform mat4 uMVP; uniform vec2 uLo; uniform vec2 uHi;
out vec2 vUv;
const vec2 C[6] = vec2[6](vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0));
void main() { vUv = C[gl_VertexID]; gl_Position = uMVP * vec4(uLo + vUv * (uHi - uLo), 0.0, 1.0); }`,
`#version 300 es
precision highp float;
in vec2 vUv; out vec4 oCol; uniform sampler2D uTex; uniform float uAlpha;
void main() { float g = texture(uTex, vUv).r; oCol = vec4(g, g, g, uAlpha); }`);
for (const m of IMAGES) {
  m.tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, m.tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, m.nx, m.ny, 0, gl.RED, gl.UNSIGNED_BYTE, m.bytes);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
}
const vaoImage = gl.createVertexArray();

const buf = {};
["triPos", "triNrm", "triCol", "triAo", "linA", "linB", "linCol", "linCorner", "pntPos", "pntCol", "pntRad", "pntMat"]
  .forEach(k => buf[k] = gl.createBuffer());

gl.bindBuffer(gl.ARRAY_BUFFER, buf.linCorner);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, -1, 1, -1, 1, 1, 0, -1, 1, 1, 0, 1]), gl.STATIC_DRAW);

const vaoFace = gl.createVertexArray(), vaoLine = gl.createVertexArray(), vaoPoint = gl.createVertexArray();

function attr(p, name, b, size, divisor) {
  const loc = gl.getAttribLocation(p, name);
  if (loc < 0) return;
  gl.bindBuffer(gl.ARRAY_BUFFER, b);
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, size, gl.FLOAT, false, 0, 0);
  if (divisor) gl.vertexAttribDivisor(loc, divisor);
}
gl.bindVertexArray(vaoFace);
attr(pFace, "aPos", buf.triPos, 3); attr(pFace, "aNrm", buf.triNrm, 3); attr(pFace, "aCol", buf.triCol, 4);
attr(pFace, "aAo", buf.triAo, 1);
gl.bindVertexArray(vaoLine);
attr(pLine, "aCorner", buf.linCorner, 2);
attr(pLine, "aA", buf.linA, 3, 1); attr(pLine, "aB", buf.linB, 3, 1); attr(pLine, "aCol", buf.linCol, 4, 1);
gl.bindVertexArray(vaoPoint);
attr(pPoint, "aPos", buf.pntPos, 3, 1); attr(pPoint, "aCol", buf.pntCol, 4, 1); attr(pPoint, "aRad", buf.pntRad, 1, 1);
attr(pPoint, "aMat", buf.pntMat, 4, 1);
gl.bindVertexArray(null);

function up(b, data) { gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW); }
function upload() {
  up(buf.triPos, TRI.pos); up(buf.triNrm, TRI.nrm); up(buf.triCol, TRI.col); up(buf.triAo, TRI.ao);
  up(buf.linA, LIN.a); up(buf.linB, LIN.b); up(buf.linCol, LIN.col);
  up(buf.pntPos, PNT.pos); up(buf.pntCol, PNT.col); up(buf.pntRad, PNT.rad); up(buf.pntMat, PNT.mat);
}

// ============================================================================================
// drawing
// ============================================================================================
const ui = {
  faces: document.getElementById("cbFaces"), points: document.getElementById("cbPoints"),
  edgeMode: document.getElementById("edgeMode"), op: document.getElementById("op"),
  r: document.getElementById("r"),
  edl: document.getElementById("cbEdl"), edlK: document.getElementById("edlK"),
  ao: document.getElementById("cbAo"), aoK: document.getElementById("aoK"), aoR: document.getElementById("aoR"),
  depth: document.getElementById("cbDepth"), bake: document.getElementById("bake"),
};
{ const b = new URLSearchParams(location.search).get("bake"); if (b !== null) ui.bake.value = b; }   // ( before the first draw )
function pointRadius() { return RMIN * Math.pow(RMAX / RMIN, parseFloat(ui.r.value)); }
ui.r.value = Math.log(R0 / RMIN) / Math.log(RMAX / RMIN);      // ( the start: R0, i.e. the given radii as they are )
// every point has its own radius: the slider is a factor on them
const PNT_SIZED = PNT_R.length > 0 && PNT_R.every(r => r > 0);

function draw() {
  if (typeof renderStale === "function") renderStale();   // ( a Blender picture over an older view: hidden )
  const w = canvas.width, h = canvas.height;
  const dark = document.body.classList.contains("dark");
  const useEdl = D >= 3 && (ui.edl.checked || ui.ao.checked);   // the post pass: eye-dome lighting and / or occlusion
  if (useEdl) { edlTargets(w, h); gl.bindFramebuffer(gl.FRAMEBUFFER, edl.fb); }
  gl.viewport(0, 0, w, h);
  const bg = BG[dark ? 1 : 0] !== null ? BG[dark ? 1 : 0] : (dark ? [0.10, 0.10, 0.10] : [1, 1, 1]);
  gl.clearColor(bg[0], bg[1], bg[2], 1);
  gl.clearDepth(1);
  gl.enable(gl.DEPTH_TEST);
  gl.depthMask(true);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const s = sceneSphere();
  // The clipping planes follow the DEPTH OF THE SCENE, not `cam.dist`: after an orbit (re-pivoting)
  // or a pan, the target no longer sits at the scene's center, and planes centered on it would cut the scene.
  const eye = camEye(), fw = camBasis().fwd;
  const dc = (s.c[0] - eye[0]) * fw[0] + (s.c[1] - eye[1]) * fw[1] + (s.c[2] - eye[2]) * fw[2];
  const near = Math.max(dc - 1.5 * s.r, 1e-3 * s.r), far = Math.max(dc, 0) + 1.5 * s.r;
  const asp = w / Math.max(h, 1);
  const P = cam.ortho ? ortho(orthoHalfH() * asp, orthoHalfH(), -far, far)
                      : perspective(FOV, asp, near, far);
  const V = viewMatrix();
  const MVP = mul(P, V);
  const alpha = parseFloat(ui.op.value);
  const showFaces = ui.faces.checked, mode = ui.edgeMode.value;

  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  // (0) the images, under everything: no depth
  if (IMAGES.some(m => layerOn[m.layer])) {
    gl.useProgram(pImage); gl.bindVertexArray(vaoImage);
    gl.disable(gl.DEPTH_TEST); gl.depthMask(false);
    gl.uniformMatrix4fv(gl.getUniformLocation(pImage, "uMVP"), false, MVP);
    gl.uniform1i(gl.getUniformLocation(pImage, "uTex"), 0);
    gl.activeTexture(gl.TEXTURE0);
    for (const m of IMAGES) if (layerOn[m.layer]) {
      gl.bindTexture(gl.TEXTURE_2D, m.tex);
      gl.uniform2f(gl.getUniformLocation(pImage, "uLo"), m.lo[0], m.lo[1]);
      gl.uniform2f(gl.getUniformLocation(pImage, "uHi"), m.hi[0], m.hi[1]);
      gl.uniform1f(gl.getUniformLocation(pImage, "uAlpha"), m.opacity);
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
    gl.enable(gl.DEPTH_TEST); gl.depthMask(true);
  }

  // (1) PURE DEPTH pass on the faces: this is what makes hidden-edge removal possible,
  // including when the faces are not displayed or are transparent.
  const needDepth = TRI.n > 0 && (mode === "hide" || mode === "ghost" || showFaces);
  gl.enable(gl.POLYGON_OFFSET_FILL);
  gl.polygonOffset(1.0, 1.0);
  if (needDepth) {
    gl.useProgram(pFace); gl.bindVertexArray(vaoFace);
    gl.uniformMatrix4fv(gl.getUniformLocation(pFace, "uMVP"), false, MVP);
    gl.uniformMatrix4fv(gl.getUniformLocation(pFace, "uMV"), false, V);
    gl.uniform1f(gl.getUniformLocation(pFace, "uAlpha"), 1);
    gl.colorMask(false, false, false, false);
    gl.depthMask(true);
    gl.drawArrays(gl.TRIANGLES, 0, TRI.n * 3);
    gl.colorMask(true, true, true, true);
  }

  // (2) visible faces -- the depth is already written, we only color what is in front
  if (showFaces && TRI.n > 0) {
    gl.useProgram(pFace); gl.bindVertexArray(vaoFace);
    gl.uniformMatrix4fv(gl.getUniformLocation(pFace, "uMVP"), false, MVP);
    gl.uniformMatrix4fv(gl.getUniformLocation(pFace, "uMV"), false, V);
    gl.uniform1f(gl.getUniformLocation(pFace, "uAlpha"), alpha);
    gl.uniform1f(gl.getUniformLocation(pFace, "uBake"), POLY_AO && POLY_AO.length ? parseFloat(ui.bake.value) : 0);
    gl.depthFunc(gl.LEQUAL);
    gl.depthMask(!needDepth);
    gl.drawArrays(gl.TRIANGLES, 0, TRI.n * 3);
    gl.depthMask(true);
  }

  gl.disable(gl.POLYGON_OFFSET_FILL);

  // (3) edges. "ghost" = two passes: first ALL of them, with no depth test and very pale
  // (the hidden ones), then only the visible ones in full color on top.
  if (mode !== "none" && LIN.n > 0) {
    gl.useProgram(pLine); gl.bindVertexArray(vaoLine);
    gl.uniformMatrix4fv(gl.getUniformLocation(pLine, "uMVP"), false, MVP);
    gl.uniform2f(gl.getUniformLocation(pLine, "uHalfVP"), w / 2, h / 2);
    gl.uniform1f(gl.getUniformLocation(pLine, "uW"), 1.6 * (window.devicePixelRatio || 1));
    gl.depthMask(false);
    if (mode === "ghost" || mode === "all") {
      gl.disable(gl.DEPTH_TEST);
      gl.uniform1f(gl.getUniformLocation(pLine, "uAlpha"), mode === "all" ? 1 : 0.22);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, LIN.n);
      gl.enable(gl.DEPTH_TEST);
    }
    if (mode !== "all") {
      gl.depthFunc(gl.LEQUAL);
      gl.uniform1f(gl.getUniformLocation(pLine, "uAlpha"), 1);
      gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, LIN.n);
    }
    gl.depthMask(true);
  }

  // (4) points
  if (ui.points.checked && PNT.n > 0) {
    gl.useProgram(pPoint); gl.bindVertexArray(vaoPoint);
    gl.uniformMatrix4fv(gl.getUniformLocation(pPoint, "uMVP"), false, MVP);
    gl.uniformMatrix4fv(gl.getUniformLocation(pPoint, "uMV"), false, V);
    gl.uniform1f(gl.getUniformLocation(pPoint, "uOrtho"), cam.ortho ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(pPoint, "uSizeK"),
      cam.ortho ? (h / 2) / orthoHalfH() : (h / 2) / Math.tan(FOV / 2));
    gl.uniform2f(gl.getUniformLocation(pPoint, "uHalfVP"), w / 2, h / 2);
    gl.uniform1f(gl.getUniformLocation(pPoint, "uR"), pointRadius());
    gl.uniform1f(gl.getUniformLocation(pPoint, "uRScale"), pointRadius() / R0);
    gl.uniform1f(gl.getUniformLocation(pPoint, "uDepthCol"), D >= 3 && ui.depth.checked ? 1 : 0);
    gl.uniformMatrix4fv(gl.getUniformLocation(pPoint, "uP"), false, P);
    gl.uniform1f(gl.getUniformLocation(pPoint, "uSphere"), D >= 3 ? 1 : 0);
    gl.uniform2f(gl.getUniformLocation(pPoint, "uDepthRange"), dc - s.r, dc + s.r);   // ( the scene's sphere, along the view )
    gl.depthFunc(gl.LEQUAL);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, PNT.n);
  }

  // (5) eye-dome lighting: the texture to the screen, darkened where the neighbours are in front
  if (useEdl) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.disable(gl.DEPTH_TEST); gl.disable(gl.BLEND);
    gl.useProgram(pEdl); gl.bindVertexArray(edl.vao);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, edl.col);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, edl.dep);
    gl.uniform1i(gl.getUniformLocation(pEdl, "uCol"), 0);
    gl.uniform1i(gl.getUniformLocation(pEdl, "uDep"), 1);
    gl.uniform1f(gl.getUniformLocation(pEdl, "uNear"), near);
    gl.uniform1f(gl.getUniformLocation(pEdl, "uFar"), far);
    gl.uniform1f(gl.getUniformLocation(pEdl, "uOrtho"), cam.ortho ? 1 : 0);
    gl.uniform1f(gl.getUniformLocation(pEdl, "uRef"), s.r);
    gl.uniform1f(gl.getUniformLocation(pEdl, "uK"), ui.edl.checked ? edlStrength() : 0);
    gl.uniform1f(gl.getUniformLocation(pEdl, "uAo"), ui.ao.checked ? parseFloat(ui.aoK.value) : 0);
    gl.uniform1f(gl.getUniformLocation(pEdl, "uAoRad"), aoRadius());
    gl.uniform1f(gl.getUniformLocation(pEdl, "uSizeK"), cam.ortho ? (h / 2) / orthoHalfH() : (h / 2) / Math.tan(FOV / 2));
    gl.uniform1f(gl.getUniformLocation(pEdl, "uRad"), 1.5 * (window.devicePixelRatio || 1));
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.activeTexture(gl.TEXTURE0);
    gl.enable(gl.DEPTH_TEST);
  }
  gl.bindVertexArray(null);
}

function resize() {
  const dpr = window.devicePixelRatio || 1;
  if (typeof layoutTimeline === "function") layoutTimeline();
  const view = document.getElementById("view");
  const w = Math.max(50, view.clientWidth), h = Math.max(50, view.clientHeight);   // ( the curves float over the view )
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  if (viewUntouched && typeof resetView === "function") resetView();   // ( the page resized before anyone moved the view: refit )
  else draw();
}

// ============================================================================================
// interactions
// ============================================================================================
// The wheel comes and goes: we do not want to re-aim at each notch. The captured depth is only
// taken again if the cursor moved, or after a pause -- in the meantime the gesture keeps its own.
let wheelAim = { x: 1e9, y: 1e9, z: 0, t: -1e9 };
function wheelDepth(sx, sy, now) {
  if (Math.hypot(sx - wheelAim.x, sy - wheelAim.y) > 8 || now - wheelAim.t > 250)
    wheelAim.z = depthAt(sx, sy);
  wheelAim.x = sx; wheelAim.y = sy; wheelAim.t = now;
  return wheelAim.z;
}
function panBy(dxPix, dyPix, z) {
  const b = camBasis();
  // pixels -> world at the depth `z` of what we MOVE (that of the grabbed point), and not at
  // that of the target: in perspective the two differ, and it is the former that makes the
  // scenery follow the cursor exactly. In orthographic the scale does not depend on depth.
  const d = cam.ortho ? cam.dist : (z === undefined ? pivotDepth() : z);
  const k = 2 * d * Math.tan(FOV / 2) / Math.max(canvas.clientHeight, 1);
  for (let i = 0; i < 3; i++)
    cam.target[i] += -dxPix * k * b.right[i] + dyPix * k * b.up[i];
  pivotDirty = true;
}
function zoomBy(f, sx, sy, z) {
  // The aiming comes BEFORE the change of `dist`: the eye depends on it, so aiming afterwards would cast
  // the ray from a camera that has already moved.
  const p = sx === undefined ? null : aimAt(sx, sy, z === undefined ? depthAt(sx, sy) : z);
  const d0 = cam.dist;
  cam.dist = Math.max(1e-6, Math.min(cam.dist * f, 1e9));
  const g = cam.dist / d0;                             // factor ACTUALLY applied (bounds)
  // The aimed point stays still: eye and target move closer to it by the same factor `g`.
  // This is exact in BOTH projections, because the on-screen position of a point depends on
  // the target only through the gap `p - target`, which here is multiplied by `g` just like
  // the half-height of the view. Nothing to aim at (background): `depthAt` returns the target's plane, and
  // the zoom then falls back on the usual behavior, centered on the cursor.
  if (p) for (let i = 0; i < 3; i++) cam.target[i] = p[i] + g * (cam.target[i] - p[i]);
  // everything that remained visible moved closer by the same factor; the pivot, however, must be redone --
  // a zoom towards the cursor makes the scene SLIDE, hence changes what occupies the center.
  wheelAim.z *= g;
  pivotDirty = true;
  return g;
}
function zoomAtCenter(f) {                             // keyboard: no cursor, we aim at the center
  zoomBy(f, canvas.clientWidth / 2, canvas.clientHeight / 2, pivotDepth());
}
function orbitBy(dx, dy) {
  if (FLAT) { panBy(dx, dy); return; }                 // in 2D there is nothing to rotate
  // The pivot is at the CENTER OF THE SCREEN, at the depth of what is there -- not at that of the
  // target, which means nothing anymore after a lateral move. `cam.dist` is NOT touched:
  // it carries the zoom level in orthographic, and the re-pivoting must stay invisible.
  const z = pivotDepth(true), e = camEye(), f = camBasis().fwd;
  const pv = [e[0] + z * f[0], e[1] + z * f[1], e[2] + z * f[2]];

  // The rotation is expressed in the SCREEN frame, hence POST-multiplied: `dx` turns around
  // the screen's up axis, `dy` around its right axis -- whatever the current orientation. Hence
  // the absence of a stop: there is no pole, and roll appears naturally along a curved
  // gesture (two rotations about different axes do not commute), as on a real trackball.
  const k = 0.008;
  cam.rot = qNorm(qMul(cam.rot,
                       qMul(qAxis([0, 1, 0], -dx * k), qAxis([1, 0, 0], -dy * k))));

  // the target is repositioned so that the PIVOT itself does not move: the eye ends up at distance `z` from
  // `pv`, still on its axis -- so `pv` stays at the center of the screen, at the same depth.
  const g = camBasis().fwd, s = cam.dist - z;
  cam.target = [pv[0] + s * g[0], pv[1] + s * g[1], pv[2] + s * g[2]];
}

// Coordinates of an event WITHIN the canvas -- it fills the window, but we do not assume it.
function evPos(e) { const r = canvas.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }

let drag = null;
canvas.style.cursor = FLAT ? "grab" : "move";
canvas.addEventListener("mousedown", e => {
  const pan = e.shiftKey || e.button === 1 || e.button === 2;
  pivotDirty = true;                                   // new gesture: the pivot must be redone
  // The depth of the move is that of the GRABBED point, frozen for the whole gesture:
  // recomputing it at each movement would make the speed jump whenever the cursor crosses an edge.
  drag = { x: e.clientX, y: e.clientY, pan, z: pan ? depthAt(...evPos(e)) : 0 };
});
window.addEventListener("mousemove", e => {
  if (!drag) return;
  const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
  drag.x = e.clientX; drag.y = e.clientY;
  if (drag.pan) panBy(dx, dy, drag.z); else orbitBy(dx, dy);
  draw();
});
window.addEventListener("mouseup", () => { drag = null; });
canvas.addEventListener("contextmenu", e => e.preventDefault());
canvas.addEventListener("dblclick", resetView);

// wheel: a touchpad pinch arrives as a `wheel` with ctrlKey (convention
// shared with a mouse's Ctrl+wheel) -> zoom; otherwise scrolling = move.
// `?wheel=0` in the URL (e.g. when embedded in an iframe) leaves the wheel to the host page.
// `?touch=0` does the same for touch gestures. Both can be switched back on from the menu.
// Without the parameter, the default comes from Python (`to_html( wheel, touch )`).
const URL_OPTS = new URLSearchParams(location.search);
let wheelEnabled = URL_OPTS.has("wheel") ? URL_OPTS.get("wheel") !== "0" : __WHEEL__;
let touchEnabled = URL_OPTS.has("touch") ? URL_OPTS.get("touch") !== "0" : __TOUCH__;
canvas.addEventListener("wheel", e => {
  if (!wheelEnabled) return;
  e.preventDefault();
  const [sx, sy] = evPos(e);
  if (e.ctrlKey || e.metaKey) zoomBy(Math.exp(e.deltaY * 0.01), sx, sy);
  else panBy(-e.deltaX, -e.deltaY, wheelDepth(sx, sy, e.timeStamp));
  draw();
}, { passive: false });

let touch = null;
canvas.addEventListener("touchstart", e => {
  if (!touchEnabled) return;
  e.preventDefault();
  pivotDirty = true;                                   // new gesture: the pivot must be redone
  if (e.touches.length === 1) touch = { mode: "rot", x: e.touches[0].clientX, y: e.touches[0].clientY };
  else if (e.touches.length === 2) {
    const [a, b] = e.touches;
    const cx = (a.clientX + b.clientX) / 2, cy = (a.clientY + b.clientY) / 2;
    const r = canvas.getBoundingClientRect();
    // depth of the pinched point, frozen for the gesture -- as for the mouse move
    touch = { mode: "pinch", d: Math.hypot(b.clientX - a.clientX, b.clientY - a.clientY),
              x: cx, y: cy, z: depthAt(cx - r.left, cy - r.top) };
  }
}, { passive: false });
canvas.addEventListener("touchmove", e => {
  if (!touchEnabled) return;
  e.preventDefault();
  if (!touch) return;
  if (touch.mode === "rot" && e.touches.length === 1) {
    orbitBy(e.touches[0].clientX - touch.x, e.touches[0].clientY - touch.y);
    touch.x = e.touches[0].clientX; touch.y = e.touches[0].clientY;
  } else if (touch.mode === "pinch" && e.touches.length === 2) {
    const [a, b] = e.touches;
    const d = Math.hypot(b.clientX - a.clientX, b.clientY - a.clientY);
    const mx = (a.clientX + b.clientX) / 2, my = (a.clientY + b.clientY) / 2;
    const r = canvas.getBoundingClientRect();
    // the pinch zooms on what it HOLDS, then moves at the depth of that same point --
    // which we moved closer to by the factor `zoomBy` actually applied.
    touch.z *= zoomBy(touch.d / Math.max(d, 1), mx - r.left, my - r.top, touch.z);
    panBy(mx - touch.x, my - touch.y, touch.z);
    touch.d = d; touch.x = mx; touch.y = my;
  }
  draw();
}, { passive: false });
canvas.addEventListener("touchend", e => { if (e.touches.length === 0) touch = null; });

function pick(e, a, b, c) { return (e.ctrlKey || e.metaKey) ? c : (e.shiftKey ? b : a); }
const helpPanel = document.getElementById("help");

// EMBEDDED ( an iframe, `?embed=1`; `?embed=0` never ): no bar, a few buttons over the view, the inspector behind the gear
const EMBED = URL_OPTS.has("embed") ? URL_OPTS.get("embed") === "1" : window.self !== window.top;
document.body.classList.toggle("embed", EMBED);
document.getElementById("gearBtn").addEventListener("click", () => {
  const on = !document.body.classList.contains("tools");
  document.body.classList.toggle("tools", on);
  if (!on) closePanel();
});
document.getElementById("helpBtn").addEventListener("click", e => { e.stopPropagation(); helpPanel.style.display = helpPanel.style.display === "block" ? "none" : "block"; });
const cbWheel = document.getElementById("cbWheel"), cbTouch = document.getElementById("cbTouch");
cbWheel.checked = wheelEnabled; cbTouch.checked = touchEnabled;
const navBtn = document.getElementById("nav");
function syncNav() {
  cbWheel.checked = wheelEnabled; cbTouch.checked = touchEnabled;
  const on = wheelEnabled || touchEnabled;
  navBtn.classList.toggle("on", on);
  navBtn.title = on ? "navigation on: wheel, pinch and touch move / zoom the view (click: give them back to the page)"
                    : "navigation off: click so that the wheel, pinch and touch move / zoom the view";
}
cbWheel.addEventListener("change", () => { wheelEnabled = cbWheel.checked; syncNav(); });
cbTouch.addEventListener("change", () => { touchEnabled = cbTouch.checked; syncNav(); });
navBtn.addEventListener("click", () => {
  const on = !(wheelEnabled || touchEnabled);
  wheelEnabled = on; touchEnabled = on; syncNav();
});
syncNav();
// the switch is off and someone scrolls over the view: show where it is
let navPulse = 0;
canvas.addEventListener("wheel", () => {
  if (wheelEnabled) return;
  navBtn.classList.add("pulse");
  clearTimeout(navPulse); navPulse = setTimeout(() => navBtn.classList.remove("pulse"), 700);
}, { passive: true });
// embedded: offer the standalone page -- unless there is none (`srcdoc` iframe of a notebook)
if (window.self !== window.top && location.href !== "about:srcdoc") {
  const a = document.getElementById("openFull");
  a.href = location.pathname + "?embed=0"; a.style.display = "inline-flex";
}
window.addEventListener("keydown", e => {
  if (e.key === "?") { e.preventDefault(); helpPanel.style.display = helpPanel.style.display === "block" ? "none" : "block"; return; }
  if (e.key === "Escape") { helpPanel.style.display = "none"; return; }
  // Space comes BEFORE the early return below: the time bar keeps the focus (so that the
  // arrows drive it natively), and playback must remain accessible in that state.
  if (e.key === " " && playableAxes.length) { e.preventDefault(); togglePlay(); return; }
  if (e.target.tagName === "SELECT" || e.target.tagName === "INPUT") return;
  const s = pick(e, 12, 40, 90);
  switch (e.key) {
    case "ArrowLeft":  e.preventDefault(); orbitBy(-s, 0); break;
    case "ArrowRight": e.preventDefault(); orbitBy( s, 0); break;
    case "ArrowUp":    e.preventDefault(); orbitBy(0, -s); break;
    case "ArrowDown":  e.preventDefault(); orbitBy(0,  s); break;
    case "+": case "=": e.preventDefault(); zoomAtCenter(1 / Math.pow(1.2, pick(e, 1, 2, 5))); break;
    case "-": case "_": e.preventDefault(); zoomAtCenter(Math.pow(1.2, pick(e, 1, 2, 5))); break;
    case "0": e.preventDefault(); resetView(); return;
    case "[": case "]":
      e.preventDefault();
      ui.r.value = Math.max(0, Math.min(1, parseFloat(ui.r.value) + (e.key === "]" ? 0.03 : -0.03)));
      syncLabels(); break;
    case "f": case "F": ui.faces.checked = !ui.faces.checked; break;
    case "p": case "P": ui.points.checked = !ui.points.checked; break;
    case "c": case "C": ui.depth.checked = !ui.depth.checked; break;
    case "e": case "E": {
      const opts = ["hide", "ghost", "all", "none"];
      ui.edgeMode.value = opts[(opts.indexOf(ui.edgeMode.value) + 1) % opts.length];
      break;
    }
    case "o": case "O": cam.ortho = !cam.ortho; syncProj(); break;
    case "r": case "R": if (SVC.blender) { e.preventDefault(); renderView(false); } return;
    default:
      if (e.key >= "1" && e.key <= "9" && !e.ctrlKey && !e.metaKey && !e.altKey) {
        const l = layerRows[parseInt(e.key, 10) - 1];
        if (l) { l.set(!l.get()); l.sync(); }
        return;
      }
      return;
  }
  syncLayers(); if (edgeSeg) edgeSeg.sync();
  draw();
});

// ============================================================================================
// time / parameter bar
// ============================================================================================
// The frames form a grid over the axes: on each axis, the RANK of a frame among the distinct values. With one axis the
// rank is the frame's own index ( frames in the order given, equal values allowed ). A combination that no frame
// has shows the NEAREST frame, the axis just moved weighing the most -- the bars then snap to what is shown.
let AX_VALS = [], FR_RANK = null, FRAME_AT = new Map();
function frameGrid() {                                 // ( again when frames come: the store mode )
  AX_VALS = []; FR_RANK = new Int32Array(NB_FRAMES * NB_AXES);
  for (let a = 0; a < NB_AXES; a++) {
    const col = Array.from({ length: NB_FRAMES }, (_, f) => FR_VAL[f * NB_AXES + a]);
    if (NB_AXES === 1) { AX_VALS.push(col); col.forEach((_, f) => { FR_RANK[f] = f; }); continue; }
    const vals = Array.from(new Set(col)).sort((x, y) => x - y);
    AX_VALS.push(vals);
    col.forEach((v, f) => { FR_RANK[f * NB_AXES + a] = vals.indexOf(v); });
  }
  FRAME_AT = new Map();
  for (let f = NB_FRAMES - 1; f >= 0; f--) FRAME_AT.set(Array.from(FR_RANK.subarray(f * NB_AXES, (f + 1) * NB_AXES)).join(","), f);
}
frameGrid();
function frameFor(ranks, moved) {
  const exact = FRAME_AT.get(ranks.join(","));
  if (exact !== undefined) return exact;
  let best = 0, bestCost = Infinity;
  for (let f = 0; f < NB_FRAMES; f++) {
    let c = 0;
    for (let a = 0; a < NB_AXES; a++) c += Math.abs(FR_RANK[f * NB_AXES + a] - ranks[a]) * (a === moved ? 1000 : 1);
    if (c < bestCost) { bestCost = c; best = f; }
  }
  return best;
}
const fmtVal = v => Number.isInteger(v) ? String(v) : String(parseFloat(v.toPrecision(4)));

// the bar: one row per axis that takes more than one value -- [ play ] [ slider ] name = value  k/n
// ( no play button anywhere: no column for it either )
const animBox = document.getElementById("anim");
let axRows = [], shownAxes = [], playableAxes = [], focusAxis = -1, barKey = null;
// the rows are made again only when the set of the shown axes changes; frames that come only stretch the bars
function buildBar() {
  const key = AX_VALS.map(v => v.length > 1 ? 1 : 0).join("");
  if (key === barKey) { for (const a of shownAxes) axRows[a].inp.max = AX_VALS[a].length - 1; return; }
  const again = barKey !== null;
  barKey = key; animBox.innerHTML = ""; axRows = [];
  const anyPlay = PLAYABLE.some((p, a) => p && AX_VALS[a].length > 1);
  for (let a = 0; a < NB_AXES; a++) {
    if (AX_VALS[a].length < 2) { axRows.push(null); continue; }
    const row = document.createElement("div"); row.className = "axis glass";
    const grip = document.createElement("span"); grip.className = "grip"; grip.innerHTML = GRIP; grip.title = "move it ( a double-click puts it back )";
    const btn = document.createElement(PLAYABLE[a] ? "button" : "span");
    btn.className = PLAYABLE[a] ? "play" : "gap";
    if (!anyPlay) btn.style.display = "none";
    if (PLAYABLE[a]) { btn.innerHTML = PLAY_ICON; btn.title = "play / pause " + AXES[a] + " (Space)"; btn.setAttribute("aria-label", "play " + AXES[a]); }
    const inp = document.createElement("input");
    inp.type = "range"; inp.min = 0; inp.max = AX_VALS[a].length - 1; inp.step = 1; inp.title = AXES[a];
    const lab = document.createElement("span"); lab.className = "lab";
    row.append(grip, btn, lab, inp);
    if (a === 0) { const live = document.createElement("span"); live.className = "live hidden"; live.id = "liveTag"; live.textContent = "live"; row.appendChild(live); }
    movable(row, grip);
    animBox.appendChild(row);
    inp.addEventListener("input", () => {
      stopPlaying();
      const r = curRanks(); r[a] = parseInt(inp.value, 10);
      setFrame(frameFor(r, a));
    });
    inp.addEventListener("focus", () => { focusAxis = a; });
    if (PLAYABLE[a]) btn.addEventListener("click", () => togglePlay(a));
    axRows.push({ btn, inp, lab });
  }
  shownAxes = axRows.map((r, a) => r ? a : -1).filter(a => a >= 0);
  playableAxes = shownAxes.filter(a => PLAYABLE[a]);
  if (!axRows[focusAxis]) focusAxis = playableAxes.length ? playableAxes[0] : (shownAxes.length ? shownAxes[0] : -1);
  animBox.style.display = shownAxes.length ? "flex" : "none";
  layoutTimeline();
}
const GRIP = '<svg viewBox="0 0 10 14" fill="currentColor"><circle cx="3" cy="3" r="1.1"/><circle cx="7" cy="3" r="1.1"/><circle cx="3" cy="7" r="1.1"/><circle cx="7" cy="7" r="1.1"/><circle cx="3" cy="11" r="1.1"/><circle cx="7" cy="11" r="1.1"/></svg>';
// an axis or a scale taken out of its place by its grip: it floats where it is dropped; a double-click puts it back
function movable(el, grip) {
  grip.addEventListener("pointerdown", e => {
    e.preventDefault(); e.stopPropagation();
    const r = el.getBoundingClientRect(), dx = e.clientX - r.left, dy = e.clientY - r.top;
    grip.setPointerCapture(e.pointerId);
    el.classList.add("floating"); el.style.left = r.left + "px"; el.style.top = r.top + "px";
    const mv = ev => {
      el.style.left = Math.max(0, Math.min(window.innerWidth - 60, ev.clientX - dx)) + "px";
      el.style.top = Math.max(0, Math.min(window.innerHeight - 30, ev.clientY - dy)) + "px";
    };
    const up = () => { grip.removeEventListener("pointermove", mv); grip.removeEventListener("pointerup", up); };
    grip.addEventListener("pointermove", mv); grip.addEventListener("pointerup", up);
  });
  grip.addEventListener("dblclick", () => { el.classList.remove("floating"); el.style.left = el.style.top = ""; });
}
const PLAY_ICON = '<svg viewBox="0 0 20 20" fill="currentColor"><path d="M5 3l12 7-12 7z"/></svg>';
const PAUSE_ICON = '<svg viewBox="0 0 20 20" fill="currentColor"><rect x="4.5" y="3" width="4" height="14" rx="1"/><rect x="11.5" y="3" width="4" height="14" rx="1"/></svg>';
// the timeline is there when there are frames to go through, or curves; the view and the rail end above it
function layoutTimeline() {
  document.getElementById("axesDock").style.display = shownAxes.length ? "flex" : "none";
}
buildBar();
initTools();

let playing = -1, playTimer = null;          // `playing`: the axis being played, -1 when paused
const curRanks = () => Array.from(FR_RANK.subarray(frameIdx * NB_AXES, (frameIdx + 1) * NB_AXES));

function syncBar() {
  for (const a of shownAxes) {
    const r = FR_RANK[frameIdx * NB_AXES + a], row = axRows[a];
    row.inp.value = r;
    row.lab.innerHTML = "<b>" + AXES[a] + " " + fmtVal(FR_VAL[frameIdx * NB_AXES + a]) + "</b>" +
      '<span class="cnt">' + (r + 1) + " / " + AX_VALS[a].length + "</span>";
  }
}
function setFrame(i) {
  frameIdx = Math.max(0, Math.min(NB_FRAMES - 1, i));
  syncBar();
  curveCursors();
  if (STORED) {                                       // its chunk: at once from the cache, else drawn when it arrives
    STORE.want(frameIdx, f => {
      if (f !== frameIdx) return;
      useScene(STORE.get(f)); buildScene(); draw(); bakeFrame(f);
      if (regionMode !== "off") showRegion(f);
    });
    return;
  }
  buildScene();                                       // only the displayed frame is rebuilt
  draw();
}

// ============================================================================================
// the CURVES ( `add_curve` ): one chart per name, the abscissa on the first frame axis. Each chart: its curves, the
// extreme values of both axes, a cursor at the displayed frame with the value of the nearest sample of each curve.
const curveBox = document.getElementById("curves"), curveTip = document.getElementById("curveTip");
var charts = [];                                       // ( var: the timeline asks for it before this line )
const CW = 300, CH = 80;                               // ( the svg's units: it is stretched to its window )
const fmtY = v => v === 0 ? "0" : (Math.abs(v) >= 1e4 || Math.abs(v) < 1e-3 ? v.toExponential(2) : String(parseFloat(v.toPrecision(4))));
const logOf = {};                                      // a chart's log scale, as switched by its button ( name -> bool )
const cssCol = c => "rgb(" + c.map(v => Math.round(255 * v)).join(",") + ")";
function curveNames() {
  const names = [];
  for (const c of CURVES) if (c.x.length && !names.includes(c.name)) names.push(c.name);
  return names;
}
// the nearest sample of curve `c` at `x` ( abscissas in increasing order )
function nearest(c, x) {
  let lo = 0, hi = c.x.length - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (c.x[m] <= x) lo = m; else hi = m; }
  return Math.abs(c.x[hi] - x) < Math.abs(c.x[lo] - x) ? hi : lo;
}
function makeChart(name) {
    const NS = "http://www.w3.org/2000/svg";
    const cs = CURVES.filter(c => c.name === name && c.x.length);
    const canLog = cs.every(c => c.y.every(v => v > 0 || v !== v));
    const log = canLog && (logOf[name] !== undefined ? logOf[name] : cs.some(c => c.log));
    const ty = v => log ? Math.log10(v) : v;
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const c of cs) for (let i = 0; i < c.x.length; i++) {
      x0 = Math.min(x0, c.x[i]); x1 = Math.max(x1, c.x[i]);
      const y = ty(c.y[i]); if (y === y && isFinite(y)) { y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    }
    if (x1 <= x0) x1 = x0 + 1;
    if (!(y1 > y0)) { const m = Math.abs(y0) * 0.05 || 1; y0 -= m; y1 += m; }
    const pad = 0.06 * (y1 - y0); y0 -= pad; y1 += pad;
    const sx = x => (x - x0) / (x1 - x0) * CW, sy = y => CH - (ty(y) - y0) / (y1 - y0) * CH;
    const div = document.createElement("div"); div.className = "chart";
    // the title bar: a mark per curve, the name, the values at the frame shown, log, fold, close
    const ttl = document.createElement("div"); ttl.className = "ttl";
    const marks = cs.map(() => { const m = document.createElement("span"); m.className = "mk"; ttl.appendChild(m); return m; });
    const nm = document.createElement("span"); nm.className = "nm"; nm.textContent = name; nm.title = name;
    const val = document.createElement("span"); val.className = "v";
    const logB = document.createElement("button"); logB.className = "log" + (log ? " on" : ""); logB.textContent = "log";
    logB.title = canLog ? "logarithmic scale" : "values not all positive: no log scale"; logB.disabled = !canLog;
    const fold = document.createElement("button"); fold.title = "fold"; fold.innerHTML = '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M5 12l5-5 5 5"/></svg>';
    const close = document.createElement("button"); close.title = "hide"; close.innerHTML = '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M5 5l10 10M15 5 5 15"/></svg>';
    ttl.append(nm, val, logB, fold, close);
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", "0 0 " + CW + " " + CH); svg.setAttribute("preserveAspectRatio", "none");
    const mk = (tag, at) => { const e = document.createElementNS(NS, tag); for (const k in at) e.setAttribute(k, at[k]); svg.appendChild(e); return e; };
    const plot = document.createElement("div"); plot.className = "plot";
    plot.appendChild(svg);
    const ylab = (y, txt) => { const s = document.createElement("span"); s.className = "ylab"; s.style.top = "calc(" + (100 * y / CH) + "% - 6px)"; s.textContent = txt; plot.appendChild(s); };
    const inv = v => log ? Math.pow(10, v) : v;
    // the grid: the decades of a log scale, else the quarters
    if (log && Math.floor(y1) - Math.ceil(y0) >= 1) {
      for (let k = Math.ceil(y0 - 1e-9); k <= Math.floor(y1 + 1e-9); k++) {
        const y = CH - (k - y0) / (y1 - y0) * CH;
        mk("line", { class: "grid", x1: 0, x2: CW, y1: y, y2: y, "vector-effect": "non-scaling-stroke" });
        ylab(y, "1e" + k);
      }
    } else {
      for (const q of [0, 0.5, 1]) {
        const y = CH * (1 - q);
        mk("line", { class: "grid", x1: 0, x2: CW, y1: y, y2: y, "vector-effect": "non-scaling-stroke" });
        ylab(y, fmtY(inv(y0 + q * (y1 - y0))));
      }
    }
    const pts = c => c.x.map((x, i) => sx(x).toFixed(2) + "," + sy(c.y[i]).toFixed(2)).join(" ");
    const areas = cs.length === 1 ? [mk("polygon", { points: "0," + CH + " " + pts(cs[0]) + " " + CW + "," + CH, "fill-opacity": 0.09, stroke: "none" })] : [];
    const paths = cs.map(c => mk("polyline", { fill: "none", "stroke-width": 1.6, "vector-effect": "non-scaling-stroke", points: pts(c) }));
    const hov = mk("line", { class: "hov", y1: 0, y2: CH, "vector-effect": "non-scaling-stroke", visibility: "hidden" });
    const cur = mk("line", { class: "cur", y1: 0, y2: CH, "vector-effect": "non-scaling-stroke" });
    // the dots at the frame shown: html, round whatever the stretch of the svg
    const dots = cs.map(() => { const d = document.createElement("span"); d.style.cssText = "position:absolute;width:7px;height:7px;margin:-4.5px 0 0 -4.5px;border-radius:50%;background:var(--raised);border:1.6px solid;pointer-events:none"; plot.appendChild(d); return d; });
    const axes = document.createElement("div"); axes.className = "axes";
    axes.innerHTML = "<span>" + fmtVal(x0) + "</span><span>" + AXES[0] + "</span><span>" + fmtVal(x1) + "</span>";
    div.append(ttl, plot, axes);
    curveBox.appendChild(div);
    // a click ( or a drag ) goes to the frame whose value on the first axis is the nearest; hovering says the values there
    const xAt = e => { const r = svg.getBoundingClientRect(); return x0 + (e.clientX - r.left) / r.width * (x1 - x0); };
    const go = e => {
      const x = xAt(e);
      let best = 0, bd = Infinity;
      AX_VALS[0].forEach((v, k) => { const d = Math.abs(v - x); if (d < bd) { bd = d; best = k; } });
      stopPlaying();
      const rk = curRanks(); rk[0] = best; setFrame(frameFor(rk, 0));
    };
    svg.addEventListener("pointerdown", e => { svg.setPointerCapture(e.pointerId); go(e); });
    svg.addEventListener("pointermove", e => {
      if (e.buttons & 1) go(e);
      const x = xAt(e);
      hov.setAttribute("x1", sx(x)); hov.setAttribute("x2", sx(x)); hov.setAttribute("visibility", "visible");
      curveTip.textContent = AXES[0] + " " + fmtVal(x) + " · " + cs.map(c => fmtY(c.y[nearest(c, x)])).join(" · ");
      curveTip.style.display = "block";
      curveTip.style.left = (e.clientX + 12) + "px"; curveTip.style.top = (e.clientY - 30) + "px";
    });
    svg.addEventListener("pointerleave", () => { hov.setAttribute("visibility", "hidden"); curveTip.style.display = "none"; });
    // a window: moved by its title bar, resized by its corner ( css ), folded, hidden by its cross
    div.style.display = cs.some(c => c.visible !== false) ? "flex" : "none";
    ttl.addEventListener("pointerdown", e => {
      if (e.target.closest("button")) return;
      const r = div.getBoundingClientRect(), dx = e.clientX - r.left, dy = e.clientY - r.top;
      ttl.setPointerCapture(e.pointerId); ttl.style.cursor = "grabbing";
      const mv = ev => {
        div.style.left = Math.max(0, Math.min(window.innerWidth - 60, ev.clientX - dx)) + "px";
        div.style.top = Math.max(0, Math.min(window.innerHeight - 34, ev.clientY - dy)) + "px";
      };
      const up = () => { ttl.style.cursor = ""; ttl.removeEventListener("pointermove", mv); ttl.removeEventListener("pointerup", up); };
      ttl.addEventListener("pointermove", mv); ttl.addEventListener("pointerup", up);
    });
    const ch = { cs, paths, areas, marks, dots, cur, val, sx, sy, div, name, close, log };
    fold.addEventListener("click", () => {
      const f = !div.classList.contains("folded");
      div.classList.toggle("folded", f);
      fold.firstChild.style.transform = f ? "rotate(180deg)" : "";
    });
    logB.addEventListener("click", () => { logOf[name] = !log; refreshCurves(); });
    return ch;
}
function chartItem(ch) {
  ch.close.addEventListener("click", () => { ch.div.style.display = "none"; if (ch.item) ch.item.sync(); });
}
// the start: the windows stacked at the top left of the view
function placeCharts() {
  const dock = document.getElementById("axesDock");
  let y = (dock.style.display !== "none" ? dock.getBoundingClientRect().bottom : 0) + 10;
  for (const ch of charts) {
    ch.div.style.width = "300px"; ch.div.style.height = "150px"; ch.div.style.left = "12px"; ch.div.style.top = y + "px";
    if (ch.div.style.display !== "none") y += 160;
  }
}
// the store mode: the curves given again ( longer ), or a scale switched; each window made again where it was
function refreshCurves() {
  for (const name of curveNames()) {
    const ch = charts.find(c => c.name === name), n = makeChart(name);
    if (!ch) {
      charts.push(n); chartItem(n);
      const prev = charts.length > 1 ? charts[charts.length - 2].div.getBoundingClientRect() : null;
      n.div.style.width = "300px"; n.div.style.height = "150px"; n.div.style.left = "12px";
      n.div.style.top = (prev ? prev.bottom + 10 : document.getElementById("axesDock").getBoundingClientRect().bottom + 10) + "px";
      continue;
    }
    for (const k of ["left", "top", "width", "height", "display"]) n.div.style[k] = ch.div.style[k];
    n.div.classList.toggle("folded", ch.div.classList.contains("folded"));
    ch.div.replaceWith(n.div);
    Object.assign(ch, n);
    chartItem(ch);
  }
  syncCurvesBtn(); layoutTimeline();
  curveCursors();
}
// the timeline's button: how many, shown or not; a click hides them all, or brings them all back
// the Curves panel: one row per chart -- its settings ( the scale, folded ), the value at the frame shown, its eye
function syncCurvesBtn() {
  tabAvailable("curves", charts.length > 0);
  const box = document.getElementById("curveList"), dark = document.body.classList.contains("dark");
  box.innerHTML = "";
  for (const ch of charts) {
    ch.item = addLayer(ch.name, () => ch.div.style.display !== "none" && !curveBox.classList.contains("off"),
      on => { if (on) curveBox.classList.remove("off"); ch.div.style.display = on ? "flex" : "none"; },
      () => ch.val.textContent, cssCol(ch.cs[0].color[dark ? 1 : 0]), pane => {
        const r = document.createElement("div"); r.className = "row";
        const l = document.createElement("span"); l.textContent = "scale";
        const seg = segmented([[false, "linear"], [true, "log"]], () => !!ch.log, v => { logOf[ch.name] = v; refreshCurves(); });
        seg.style.gridColumn = "span 2"; r.append(l, seg); pane.appendChild(r);
        const f = document.createElement("label"); f.className = "chk";
        const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = ch.div.classList.contains("folded");
        cb.addEventListener("change", () => ch.div.classList.toggle("folded", cb.checked));
        f.append(cb, document.createTextNode(" window folded")); pane.appendChild(f);
      }, "curveList", true);
  }
}
function toggleCurves() {
  const off = curveBox.classList.contains("off") || charts.every(c => c.div.style.display === "none");
  curveBox.classList.toggle("off", !off);
  if (off) for (const ch of charts) ch.div.style.display = "flex";
  for (const ch of charts) if (ch.item) ch.item.sync();
}
function buildCurves() {
  for (const name of curveNames()) { const ch = makeChart(name); charts.push(ch); chartItem(ch); }
  placeCharts();
  syncCurvesBtn();
  if (charts.length)
    document.addEventListener("keydown", e => {                        // [g]: all the curves out of the way, or back
      if ((e.key === "g" || e.key === "G") && e.target.tagName !== "SELECT" && !(e.target.tagName === "INPUT" && e.target.type !== "range"))
        toggleCurves();
    });
  layoutTimeline();
}
function curveCursors() {
  if (!charts.length) return;
  const dark = document.body.classList.contains("dark"), x = FR_VAL[frameIdx * NB_AXES];
  for (const ch of charts) {
    ch.cur.setAttribute("x1", ch.sx(x)); ch.cur.setAttribute("x2", ch.sx(x));
    const vals = [];
    ch.cs.forEach((c, k) => {
      const col = cssCol(c.color[dark ? 1 : 0]);
      ch.paths[k].setAttribute("stroke", col); ch.marks[k].style.background = col;
      if (ch.areas[k]) ch.areas[k].setAttribute("fill", col);
      const i = nearest(c, x);
      vals.push(fmtY(c.y[i]));
      const d = ch.dots[k];
      d.style.left = (100 * ch.sx(c.x[i]) / CW) + "%"; d.style.top = (100 * ch.sy(c.y[i]) / CH) + "%"; d.style.borderColor = col;
      d.style.display = c.y[i] === c.y[i] ? "" : "none";
    });
    ch.val.textContent = vals.join(" · ");
    if (ch.item) ch.item.sync();
  }
}
function stopPlaying() {
  if (playing >= 0 && axRows[playing]) { axRows[playing].btn.innerHTML = PLAY_ICON; axRows[playing].btn.classList.remove("on"); }
  playing = -1;
  clearInterval(playTimer);
}
// the frame after the current one along the played axis, the other axes held ( -1: the end, no loop )
function nextFrame() {
  const r = curRanks(), n = AX_VALS[playing].length;
  if (r[playing] + 1 < n) r[playing]++; else if (LOOP) r[playing] = 0; else return -1;
  return frameFor(r, playing);
}
// MORPH: from one frame to the next in `MORPH` eased steps, the points matched by rank ( same count ); only the point
// positions move, the rest of the scene is the frame's. A hold of a moment on each frame.
const MORPH = __MORPH__, LOOP = __LOOP__, HOLD = MORPH > 0 ? Math.round(0.8 * FPS) : 0;
function pointPositions(f) {                           // the projected positions of the points of frame `f`
  let src = POS, V = PNT_V, v0 = FR_PNT[f], v1 = FR_PNT[f + 1];
  if (STORED) { const sc = STORE.get(f); src = sc.POS; V = sc.PNT_V; v0 = sc.SHARED.pnt; v1 = V.length; }
  const out = new Float32Array((v1 - v0) * 3);
  for (let i = 0; i < v1 - v0; i++) {
    const v = V[v0 + i];
    out[3 * i] = px(src, v); out[3 * i + 1] = py(src, v); out[3 * i + 2] = pz(src, v);
  }
  return out;
}
let sub = 0, hold = 0, morphA = null, morphB = null;
function step() {
  if (hold > 0) { hold--; return; }
  const next = nextFrame();
  if (next < 0) { stopPlaying(); return; }
  if (STORED && !STORE.has(next)) { STORE.want(next, () => {}); return; }   // ( its chunk is on its way: this tick waits )
  const n0 = framePoints(frameIdx), n1 = framePoints(next);
  if (MORPH > 0 && n0 === n1 && n0 > 0) {
    if (sub === 0) { morphA = PNT.pos.slice(); morphB = pointPositions(next); }
    if (++sub < MORPH) {
      const t = 0.5 - 0.5 * Math.cos(Math.PI * sub / MORPH);
      for (let k = 0; k < morphB.length; k++) PNT.pos[k] = morphA[k] + t * (morphB[k] - morphA[k]);
      up(buf.pntPos, PNT.pos);
      draw();
      return;
    }
    sub = 0;
  }
  setFrame(next);
  hold = HOLD;
}
// plays axis `a` ( by default the one with the focus, else the first playable ); again on the axis being played: pause
function togglePlay(a) {
  if (a === undefined) a = PLAYABLE[focusAxis] && axRows[focusAxis] ? focusAxis : (playing >= 0 ? playing : playableAxes[0]);
  if (a === undefined || a < 0 || !axRows[a] || !PLAYABLE[a]) return;
  const was = playing;
  stopPlaying();
  if (was === a) return;
  playing = a; focusAxis = a;
  axRows[a].btn.innerHTML = PAUSE_ICON; axRows[a].btn.classList.add("on");
  if (curRanks()[a] >= AX_VALS[a].length - 1 && !LOOP) { const r = curRanks(); r[a] = 0; setFrame(frameFor(r, a)); }
  sub = 0; hold = 0;
  playTimer = setInterval(step, 1000 / FPS);
}
if (playableAxes.length && (URL_OPTS.has("play") ? URL_OPTS.get("play") !== "0" : __AUTOPLAY__)) setTimeout(() => togglePlay(playableAxes[0]), 400);

// ============================================================================================
// GIF export: the frames of the played axis ( the other axes held ), each drawn as the view shows it, then read back
// at once ( no `preserveDrawingBuffer`: the copy happens in the same task as the draw ). Two passes over the frames,
// so that no image is kept: the first counts the colors ( 15 bits ) to pick the 256 most frequent as the palette, the
// second maps each pixel to the nearest entry and encodes. A frame only stores the box where it differs from the
// previous one. The scenes have few colors ( a background, a few plain colors and their opacities ): no dithering.
function gifLzw(idx, out) {                            // the LZW of GIF, 8-bit indices, codes up to 12 bits
  const CLEAR = 256, EOI = 257;
  let size = 9, next = EOI + 1, cur = 0, nb = 0, dict = new Map(), bytes = [];
  const emit = c => { cur |= c << nb; nb += size; while (nb >= 8) { bytes.push(cur & 255); cur >>>= 8; nb -= 8; } };
  emit(CLEAR);
  let prefix = idx[0];
  for (let i = 1; i < idx.length; i++) {
    const k = idx[i], key = (prefix << 8) | k, v = dict.get(key);
    if (v !== undefined) { prefix = v; continue; }
    emit(prefix);
    if (next > (1 << size) - 1 && size < 12) size++;
    if (next < 4096) dict.set(key, next++);
    else { emit(CLEAR); dict.clear(); size = 9; next = EOI + 1; }
    prefix = k;
  }
  emit(prefix);
  if (next > (1 << size) - 1 && size < 12) size++;
  emit(EOI);
  if (nb > 0) bytes.push(cur & 255);
  out.push(8);                                         // the minimum code size, then the data in sub-blocks of 255 bytes
  for (let i = 0; i < bytes.length; i += 255) { const b = bytes.slice(i, i + 255); out.push(b.length, ...b); }
  out.push(0);
}
async function exportGif() {
  const a = playing >= 0 ? playing : (playableAxes.length ? playableAxes[0] : focusAxis);
  if (a === undefined || a < 0 || !axRows[a] || gifBtn.disabled) return;
  stopPlaying();
  const keep = frameIdx, label = gifLabel.textContent, ranks = curRanks(), n = AX_VALS[a].length;
  const frames = Array.from({ length: n }, (_, k) => { ranks[a] = k; return frameFor(ranks, a); });
  const maxW = parseInt(URL_OPTS.get("gif_width") || "800", 10);
  const sc = Math.min(1, maxW / canvas.width), W = Math.max(1, Math.round(canvas.width * sc)), H = Math.max(1, Math.round(canvas.height * sc));
  const c2 = document.createElement("canvas"); c2.width = W; c2.height = H;
  const g2 = c2.getContext("2d", { willReadFrequently: true });
  g2.imageSmoothingQuality = "high";
  const pause = () => new Promise(r => setTimeout(r, 0));
  const grab = async f => { if (STORED) await STORE.load(f); setFrame(f); g2.drawImage(canvas, 0, 0, W, H); return g2.getImageData(0, 0, W, H).data; };
  gifBtn.disabled = true;
  try {
    // (1) the palette: the 256 most frequent colors ( 5 bits per channel ), each the mean of its bucket
    const cnt = new Float64Array(32768), sum = new Float64Array(3 * 32768);
    for (let k = 0; k < n; k++) {
      gifLabel.textContent = "colors " + (k + 1) + "/" + n; await pause();
      const px = await grab(frames[k]);
      for (let i = 0; i < px.length; i += 4) {
        const key = ((px[i] >> 3) << 10) | ((px[i + 1] >> 3) << 5) | (px[i + 2] >> 3);
        cnt[key]++; sum[3 * key] += px[i]; sum[3 * key + 1] += px[i + 1]; sum[3 * key + 2] += px[i + 2];
      }
    }
    const used = []; for (let key = 0; key < 32768; key++) if (cnt[key]) used.push(key);
    used.sort((x, y) => cnt[y] - cnt[x]);
    const pal = used.slice(0, 256).map(key => [0, 1, 2].map(j => Math.round(sum[3 * key + j] / cnt[key])));
    while (pal.length < 256) pal.push([0, 0, 0]);
    const lut = new Uint8Array(32768);                 // bucket -> nearest entry of the palette
    for (const key of used) {
      const r = sum[3 * key] / cnt[key], g = sum[3 * key + 1] / cnt[key], b = sum[3 * key + 2] / cnt[key];
      let best = 0, bd = Infinity;
      for (let e = 0; e < 256; e++) { const d = (r - pal[e][0]) ** 2 + (g - pal[e][1]) ** 2 + (b - pal[e][2]) ** 2; if (d < bd) { bd = d; best = e; } }
      lut[key] = best;
    }
    // (2) the file: header, global palette, endless loop, then one image per frame
    const out = [...Array.from("GIF89a", ch => ch.charCodeAt(0)), W & 255, W >> 8, H & 255, H >> 8, 0xF7, 0, 0];
    for (const c of pal) out.push(c[0], c[1], c[2]);
    out.push(0x21, 0xFF, 11, ...Array.from("NETSCAPE2.0", ch => ch.charCodeAt(0)), 3, 1, 0, 0, 0);
    let prev = null;
    for (let k = 0; k < n; k++) {
      gifLabel.textContent = "GIF " + (k + 1) + "/" + n; await pause();
      const px = await grab(frames[k]), idx = new Uint8Array(W * H);
      for (let i = 0, j = 0; j < idx.length; i += 4, j++) idx[j] = lut[((px[i] >> 3) << 10) | ((px[i + 1] >> 3) << 5) | (px[i + 2] >> 3)];
      let x0 = 0, y0 = 0, x1 = W - 1, y1 = H - 1;     // the box that changed since the previous frame
      if (prev) {
        x0 = W; y0 = H; x1 = -1; y1 = -1;
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
          if (idx[y * W + x] !== prev[y * W + x]) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
        if (x1 < 0) { x0 = y0 = x1 = y1 = 0; }       // nothing changed: one pixel, the delay still counts
      }
      const bw = x1 - x0 + 1, bh = y1 - y0 + 1, sub = new Uint8Array(bw * bh);
      for (let y = 0; y < bh; y++) sub.set(idx.subarray((y0 + y) * W + x0, (y0 + y) * W + x0 + bw), y * bw);
      const delay = Math.round((k === n - 1 ? 1.5 : 1 / FPS) * 100);    // in 1/100 s; the last frame held
      out.push(0x21, 0xF9, 4, 0x04, delay & 255, delay >> 8, 0, 0);     // disposal: keep, under the next box
      out.push(0x2C, x0 & 255, x0 >> 8, y0 & 255, y0 >> 8, bw & 255, bw >> 8, bh & 255, bh >> 8, 0);
      gifLzw(sub, out);
      prev = idx;
    }
    out.push(0x3B);
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([new Uint8Array(out)], { type: "image/gif" }));
    link.download = (document.title.replace(/[^\w\-]+/g, "-").replace(/^-+|-+$/g, "") || "frames") + ".gif";
    document.body.appendChild(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 10000);
  } finally {
    gifLabel.textContent = label; gifBtn.disabled = false;
    setFrame(keep);
  }
}
const gifBtn = document.getElementById("gifBtn"), gifLabel = gifBtn.querySelector("span");
if (shownAxes.length) gifBtn.addEventListener("click", () => { closePops(); exportGif(); });
else gifBtn.style.display = "none";
document.getElementById("pngBtn").addEventListener("click", () => {     // ( read in the task that draws: no preserved buffer )
  closePops(); draw();
  const link = document.createElement("a");
  link.href = canvas.toDataURL("image/png");
  link.download = (document.title.replace(/[^\w\-]+/g, "-").replace(/^-+|-+$/g, "") || "view") + ".png";
  document.body.appendChild(link); link.click(); link.remove();
});

// ============================================================================================
// the PANELS: one per icon of the toolbar, the open ones stacked under it ( a click on its icon again closes it ); each
// folds or closes from its head. Which are open is kept in this browser.
let currentTab = null;
const panelOf = name => document.querySelector('#panels .panel[data-pane="' + name + '"]');
const iconOf = name => document.querySelector('#toolbar [data-panel="' + name + '"]');
function savePanels() {
  try { localStorage.setItem("sviz.panels", JSON.stringify(Array.from(document.querySelectorAll("#panels .panel.open")).map(p => p.dataset.pane))); } catch (e) {}
}
function tabAvailable(name, on) {
  const i = iconOf(name); if (i) i.style.display = on ? "" : "none";
  if (!on) { const p = panelOf(name); if (p) p.classList.remove("open"); if (i) i.classList.remove("on"); }
}
function openTab(name) {
  const p = panelOf(name); if (!p) return;
  currentTab = name; p.classList.add("open"); p.classList.remove("folded");
  const i = iconOf(name); if (i) i.classList.add("on");
  savePanels();
}
function closeTab(name) {
  const p = panelOf(name); if (p) p.classList.remove("open");
  const i = iconOf(name); if (i) i.classList.remove("on");
  savePanels();
}
function closePanel() { for (const p of document.querySelectorAll("#panels .panel")) closeTab(p.dataset.pane); }
for (const b of document.querySelectorAll("#toolbar [data-panel]"))
  b.addEventListener("click", () => { const n = b.dataset.panel; if (panelOf(n).classList.contains("open")) closeTab(n); else openTab(n); });
for (const p of document.querySelectorAll("#panels .panel")) {
  p.querySelector(".head .close").addEventListener("click", () => closeTab(p.dataset.pane));
  p.querySelector(".head .fold").addEventListener("click", () => p.classList.toggle("folded"));
}
tabAvailable("colour", FIELDS.length > 0);
tabAvailable("pictures", false);                        // ( shown when the services say so, `findServices` )
tabAvailable("services", false);
// the light: flat ( no shading of the scene ), relief ( eye-dome lighting and occlusion ), or custom ( the sliders )
let lightMode = "relief";
function setLight(m) {
  lightMode = m;
  if (m === "flat") { ui.edl.checked = false; ui.ao.checked = false; }
  if (m === "relief") { ui.edl.checked = true; ui.ao.checked = true; ui.edlK.value = 0.5; ui.aoK.value = 0.7; }
  for (const b of document.querySelectorAll("#lightSeg button")) b.classList.toggle("on", b.dataset.l === m);
  document.getElementById("lightCustom").style.display = m === "custom" ? "flex" : "none";
  syncLabels();
}
for (const b of document.querySelectorAll("#lightSeg button")) b.addEventListener("click", () => setLight(b.dataset.l));
if (D < 3) { document.getElementById("lightLbl").style.display = "none"; document.getElementById("lightSeg").style.display = "none"; }
function syncProj() { for (const b of document.querySelectorAll("#projSeg button")) b.classList.toggle("on", (b.dataset.p === "ortho") === !!cam.ortho); }
for (const b of document.querySelectorAll("#projSeg button")) b.addEventListener("click", () => { cam.ortho = b.dataset.p === "ortho"; syncProj(); draw(); });
if (FLAT) document.getElementById("projSeg").style.display = "none";
document.getElementById("resetBtn").addEventListener("click", resetView);
// popovers ( menus, the services ): one at a time, closed by a click elsewhere or Escape
const POPS = ["renderMenu"];
function closePops(except) {
  for (const id of POPS) if (id !== except) document.getElementById(id).style.display = "none";
  document.getElementById("renderMode").classList.remove("open");
}
function openPop(id, anchor) {
  const el = document.getElementById(id), was = el.style.display !== "none";
  closePops();
  if (was) return false;
  el.style.display = id === "svcPop" ? "block" : "flex";
  const r = anchor.getBoundingClientRect();
  el.style.top = (r.bottom + 6) + "px";
  el.style.left = Math.max(8, Math.min(window.innerWidth - el.offsetWidth - 8, r.right - el.offsetWidth)) + "px";
  return true;
}
document.addEventListener("click", e => {
  if (!e.target.closest(".pop") && !e.target.closest("#renderMode")) closePops();
  if (!e.target.closest("#help, #helpBtn")) helpPanel.style.display = "none";
});
document.addEventListener("keydown", e => { if (e.key === "Escape") closePops(); });
for (const b of document.querySelectorAll("#themeSeg button")) b.addEventListener("click", () => window.setThemeMode(b.dataset.t));
// the colour scale: a click opens the colour panel; its grip moves it
{ const lg = document.getElementById("legend");
  movable(lg, lg.querySelector(".grip"));
  lg.addEventListener("click", e => { if (!e.target.closest(".grip")) openTab("colour"); }); }
// the state of a store ( live while its run writes it ): in About, and beside the first axis
function syncStatus() {
  const txt = document.getElementById("statusText"), tag = document.getElementById("liveTag");
  if (!STORED) { txt.textContent = NB_FRAMES > 1 ? NB_FRAMES + " frames" : "one frame"; return; }
  const live = !STORE.m.complete;
  txt.textContent = (live ? "live · " : "complete · ") + NB_FRAMES + (NB_FRAMES > 1 ? " frames" : " frame");
  if (tag) tag.classList.toggle("hidden", !live);
}

// ============================================================================================
// panel: opacity, radius, viewed dimensions, slices
// ============================================================================================
function syncLabels() {
  document.getElementById("opVal").textContent = parseFloat(ui.op.value).toFixed(2);
  document.getElementById("rVal").textContent = PNT_SIZED ? "\u00d7 " + (pointRadius() / R0).toPrecision(3) : pointRadius().toPrecision(3);
  document.getElementById("edlVal").textContent = edlStrength().toFixed(0);
  document.getElementById("aoKVal").textContent = parseFloat(ui.aoK.value).toFixed(2);
  document.getElementById("aoRVal").textContent = parseFloat(ui.aoR.value).toFixed(1);
  document.getElementById("bakeVal").textContent = parseFloat(ui.bake.value).toFixed(2);
  document.getElementById("restVal").textContent = parseFloat(document.getElementById("restDelay").value).toFixed(1) + " s";
  draw();
}
document.getElementById("restDelay").addEventListener("input", syncLabels);
ui.op.addEventListener("input", syncLabels);
ui.r.addEventListener("input", syncLabels);
ui.faces.addEventListener("change", draw);
ui.points.addEventListener("change", draw);
ui.edgeMode.addEventListener("change", draw);
ui.edl.addEventListener("change", draw);
ui.depth.addEventListener("change", draw);
ui.depth.checked = URL_OPTS.has("depth") ? URL_OPTS.get("depth") !== "0" : __DEPTHCOL__;
ui.op.value = URL_OPTS.has("opacity") ? URL_OPTS.get("opacity") : __FACE_OP__;
ui.edlK.addEventListener("input", syncLabels);
ui.ao.addEventListener("change", draw);
ui.aoK.addEventListener("input", syncLabels);
ui.aoR.addEventListener("input", syncLabels);
ui.bake.addEventListener("input", syncLabels);
if (POLY_AO && POLY_AO.length) document.getElementById("bakeBox").style.display = "flex";

// SERVICES ( errand ADR 0026; sdot ADR 0001 ): what errand serves this page beside its files ( `.errand/services` ), or a
// server given by hand ( `?svc.sdot.ao=http://127.0.0.1:PORT` ). `sdot.ao` bakes the occlusion of a frame stored without.
const SVC = { ao: null, blender: null, region: null };
let serviceBake = false;
const aoAsked = new Map();
async function serviceStates() {                     // ( null: not served by errand -- the files alone )
  try { const r = await fetch(".errand/services", { cache: "no-store" }); return r.ok ? await r.json() : null; }
  catch (e) { return null; }
}
async function getJson(url, opts) {
  try { const r = await fetch(url, Object.assign({ cache: "no-store" }, opts || {})); return r.ok ? await r.json() : null; }
  catch (e) { return null; }
}
const postJson = (url, body) => getJson(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
// only `ready` and `running` are reached through the relay ( ADR errand 0027 ); `declared` / `stopped`: an older errand
const REACHABLE = st => st === "ready" || st === "running" || st === "declared" || st === "stopped";
// the base URL of service `name`: given by hand ( `?svc.<name>=` ), else through errand's relay to the machine chosen
function serviceUrl(name, states) {
  const byHand = URL_OPTS.get("svc." + name);
  if (byHand) return byHand.replace(/\/?$/, "/");
  const s = states && states[name];
  return s && REACHABLE(s.state) ? ".errand/svc/" + name + "/" : null;
}
// one line: why a service cannot be used now, on the machine chosen
function svcReason(s) {
  if (!s) return "not offered here";
  if (s.needs_copy) return s.on + " has no copy of this store";
  return (s.on ? s.on + ": " : "") + (s.state === "installable" ? "not declared" : s.state) + (s.why ? " — " + s.why : "");
}
function reasonHint(el, what, s) {
  el.textContent = what + " — " + svcReason(s) + " ( see Services )";
  el.style.display = "block";
}
// THE SERVICES PANEL ( ADR errand 0027 ): per service, the machine it runs on, its state, what that machine's agent
// found, the lines that would make it ready there ( to type: the page installs nothing ), a place to look for a tool,
// and the copy of the store when that machine lacks it.
let svcStates = null, svcMachines = [], svcTools = {}, svcOpen = {};
const SVC_DOT = st => REACHABLE(st) ? (st === "running" ? "ok" : "idle") : "warn";
const SVC_WHAT = { "blender": "pictures", "sdot.ao": "occlusion", "sdot.region": "cells of a region" };
function svcEl(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
function syncSvcChip(states) {
  const body = document.getElementById("svcBody"), badge = document.getElementById("svcBadge");
  const names = Object.keys(states || {}).filter(k => k[0] !== ".");
  tabAvailable("services", names.length > 0);
  if (!names.length) return;
  const bad = names.filter(n => !REACHABLE(states[n].state)).length;
  badge.className = "badge " + (bad ? "warn" : "ok");
  iconOf("services").title = "services: " + (names.length - bad) + " ready" + (bad ? ", " + bad + " not" : "");
  body.innerHTML = "";
  if (states[".error"]) body.appendChild(svcEl("div", "hint bad", states[".error"]));
  for (const n of names) body.appendChild(svcCard(n, states[n]));
}
function svcCard(n, s) {
  const card = svcEl("div", "svcCard"), ok = REACHABLE(s.state) && !s.needs_copy;
  const top = svcEl("div", "svcTop");
  const fold = svcEl("button", "more"); fold.innerHTML = CHEV; fold.title = "what the machine has";
  const open = svcOpen[n] != null ? svcOpen[n] : !ok;
  card.classList.toggle("open", open);
  fold.addEventListener("click", () => { svcOpen[n] = !card.classList.contains("open"); card.classList.toggle("open", svcOpen[n]); });
  const nm = svcEl("span", "svcName"); nm.append(svcEl("b", null, n));
  if (s.version) nm.append(svcEl("code", "muted", " #" + s.version));
  if (SVC_WHAT[n]) nm.append(svcEl("span", "muted", " · " + SVC_WHAT[n]));
  const dot = svcEl("span", "sdot " + SVC_DOT(s.state));
  top.append(fold, dot, nm);
  card.appendChild(top);
  // the machine, and the state there
  const row = svcEl("div", "svcOn");
  row.append(svcEl("span", "muted", "on"));
  const sel = document.createElement("select"); sel.title = "the machine " + n + " runs on";
  const ms = svcMachines.length ? svcMachines : [{ name: s.on, here: true }];
  for (const m of ms) {
    const ms_ = m.services && m.services[n];
    const label = m.name + (m.here ? " · here" : "") + (m.origin ? " · where the run wrote it" : "") + (ms_ ? " — " + ms_.state : "");
    const o = new Option(label, m.name); o.selected = m.name === s.on;
    if (ms_ && ms_.state === "unreachable") { o.disabled = m.name !== s.on; o.title = ms_.why || ""; }
    sel.add(o);
  }
  sel.addEventListener("change", async () => {
    sel.disabled = true;
    await postJson(".errand/services/" + encodeURIComponent(n), { on: sel.value });
    await refreshServices(false);
  });
  const st = svcEl("span", ok ? "muted" : "warnText", ok ? (s.state === "running" ? "running" : "ready") : svcReason(s).replace(s.on + ": ", ""));
  row.append(sel, st);
  card.appendChild(row);
  // the details: what the agent found, the lines, a path, the copy
  const det = svcEl("div", "svcDet");
  if (s.needs_copy || (s.store && s.store.copying)) {
    const c = s.store && s.store.copying, box = svcEl("div", "svcCopy");
    if (c && c.state === "copying") box.append(svcEl("span", "muted", "copying the store to " + s.on + "… " + Math.round(100 * (c.done || 0)) + " %"));
    else {
      const sz = s.store && s.store.size ? " ( " + (s.store.size / 1e6).toFixed(s.store.size > 1e7 ? 0 : 1) + " MB )" : "";
      const b = svcEl("button", "btn", "Copy the store to " + s.on);
      b.title = "its files into errand's cache on " + s.on + sz + ": data, nothing run";
      b.addEventListener("click", async () => { b.disabled = true; await postJson(".errand/copy", { on: s.on }); setTimeout(() => refreshServices(false), 800); });
      box.append(b, svcEl("span", "muted", s.on + " has no copy of this store" + sz));
    }
    det.appendChild(box);
  }
  if (s.missing && s.missing.length) {
    det.appendChild(svcEl("h3", "warnText", "Missing on " + s.on));
    const ul = svcEl("ul", "svcFound svcMissing");
    for (const f of s.missing) ul.appendChild(svcEl("li", null, f.replace(/(^|[\s(])\/(?:Users|home)\/[^/\s]+/g, "$1~")));
    det.appendChild(ul);
  }
  if (s.found && s.found.length) {
    det.appendChild(svcEl("h3", null, "What " + s.on + "'s agent found"));
    const ul = svcEl("ul", "svcFound");
    for (const f of s.found) ul.appendChild(svcEl("li", null, f.replace(/(^|[\s(])\/(?:Users|home)\/[^/\s]+/g, "$1~")));
    det.appendChild(ul);
  }
  if (s.lines && s.lines.length) {
    det.appendChild(svcEl("h3", null, "To type, in this order"));
    const pre = svcEl("pre", "cmd");
    let text = "";
    for (const l of s.lines) {
      const c = svcEl("span", "cmt" + (l.unknown_source ? " bad" : ""),
                      "# " + l.comment + (l.unknown_source ? " — a source outside your list" : "")
                      + (l.described_by === "store" ? " ( as the store describes it )" : "") + "\n");
      pre.append(c, document.createTextNode(l.command + "\n"));
      text += l.command + "\n";
    }
    det.appendChild(pre);
    const cp = svcEl("div", "svcCopyRow");
    const b = svcEl("button", "btn", "Copy the lines");
    b.addEventListener("click", () => { navigator.clipboard && navigator.clipboard.writeText(text); b.textContent = "copied"; setTimeout(() => b.textContent = "Copy the lines", 1200); });
    cp.append(b, svcEl("span", "muted", "the page runs none of them"));
    det.appendChild(cp);
  }
  // a place to look, per tool the description names ( never run: the agent reads its files )
  for (const tool of svcTools[n] || []) {
    const r = svcEl("div", "svcPath");
    const inp = document.createElement("input"); inp.type = "text"; inp.placeholder = "another " + tool + " on " + s.on + ": its path";
    const b = svcEl("button", "btn", "Look there");
    b.addEventListener("click", async () => {
      if (!inp.value.trim()) return;
      b.disabled = true;
      await postJson(".errand/machines/" + encodeURIComponent(s.on) + "/paths", { tool, path: inp.value.trim() });
      await refreshServices(false);
    });
    r.append(inp, b);
    det.appendChild(r);
  }
  if (!det.childNodes.length && !ok) det.appendChild(svcEl("div", "hint", s.text || ""));
  card.appendChild(det);
  return card;
}
// the tools each service's description names ( for the path fields ): the store's own services.json
async function readTools() {
  const j = await getJson("services.json");
  const out = {};
  if (j && j.format === 2 && j.services)
    for (const [n, v] of Object.entries(j.services)) out[n] = Object.keys((v && v.described && v.described.tools) || {});
  return out;
}
async function refreshServices(again) {
  const b = document.getElementById("svcRefresh"); b.disabled = true;
  if (again && svcStates) {            // ( each machine chosen looks again )
    const ms = new Set(Object.keys(svcStates).filter(k => k[0] !== ".").map(k => svcStates[k].on).filter(Boolean));
    await Promise.all(Array.from(ms).map(m => getJson(".errand/machines/" + encodeURIComponent(m) + "?refresh=1")));
  }
  const [states, machines] = await Promise.all([serviceStates(), getJson(".errand/machines")]);
  svcMachines = machines || [];
  svcStates = states;
  syncSvcChip(states);
  applyServices(states);
  b.disabled = false;
  // ( a copy under way: asked again until it is done )
  if (states && Object.values(states).some(s => s && s.store && s.store.copying && s.store.copying.state === "copying"))
    setTimeout(() => refreshServices(false), 1500);
}
document.getElementById("svcRefresh").addEventListener("click", () => { if (STORED) refreshServices(true); });
// the controls of each service, enabled when its machine says ready or running ( run again after a refresh, a choice )
let svcWired = false, keptLoaded = false;
function applyServices(states) {
  if (D !== 3) return;
  if (!svcWired) {
    svcWired = true;
    document.getElementById("bakeBtn").addEventListener("click", e => {
      const btn = e.currentTarget; serviceBake = true; btn.disabled = true; btn.textContent = "baking…"; bakeFrame(frameIdx);
    });
    for (const rb of document.querySelectorAll('#regionChoices input'))
      rb.addEventListener("change", () => { regionMode = rb.value; regionPlane = null; syncChoices(); showRegion(frameIdx); });
    document.getElementById("regionBtn").addEventListener("click", () => {
      regionMode = "cut"; regionPlane = null; syncChoices(); showRegion(frameIdx);
    });
  }
  // sdot.ao: the occlusion of a frame stored without
  const box = document.getElementById("bakeBox"), btn = document.getElementById("bakeBtn"), hint = document.getElementById("bakeHint");
  const aoAsked = states && states["sdot.ao"];
  SVC.ao = serviceUrl("sdot.ao", states);
  if (!(POLY_AO && POLY_AO.length) && (SVC.ao || aoAsked)) {
    document.getElementById("bakeLbl").style.display = "none";
    box.style.display = "flex";
    btn.style.display = "inline-flex";
    if (!serviceBake) btn.disabled = !SVC.ao;
    if (SVC.ao) hint.style.display = "none"; else reasonHint(hint, "occlusion on demand", aoAsked);
  }
  // sdot.region: the cells of a region, rebuilt from the frame's seeds ( the frames that kept their sources )
  const sources = STORE.m.frames.some(fr => fr.sources), rAsked = states && states["sdot.region"];
  SVC.region = sources ? serviceUrl("sdot.region", states) : null;
  if (sources && (SVC.region || rAsked)) {
    document.getElementById("cellsBox").style.display = "flex";
    for (const rb of document.querySelectorAll('#regionChoices input')) rb.disabled = !SVC.region;
    document.getElementById("regionBtn").disabled = !SVC.region;
    const rh = document.getElementById("regionHint");
    if (SVC.region) { if (rh.dataset.why) { rh.style.display = "none"; delete rh.dataset.why; } }
    else { reasonHint(rh, "cells of a region", rAsked); rh.dataset.why = "1"; }
    syncChoices();
  }
  // blender: pictures of this view
  const bAsked = states && states.blender;
  SVC.blender = serviceUrl("blender", states);
  if (SVC.blender || bAsked) {
    document.getElementById("renderGroup").style.display = "flex";
    tabAvailable("pictures", true);
    const rb = document.getElementById("renderBtn");
    rb.disabled = !SVC.blender;
    rb.title = SVC.blender ? "a picture of this view by Blender ( r )" : "pictures by Blender — " + svcReason(bAsked);
    document.getElementById("renderGroup").classList.toggle("off", !SVC.blender);
    if (SVC.blender) {
      if (!keptLoaded) { keptLoaded = true; loadKept(); }
      lastViewKey = null; renderStale();               // ( a mode that renders: the view as it is now )
    }
  }
}
async function findServices() {
  if (STORED) {
    svcTools = await readTools();
    await refreshServices(false);
  }
  try { if (!EMBED) for (const n of JSON.parse(localStorage.getItem("sviz.panels") || "[]")) { const i = iconOf(n); if (i && i.style.display !== "none") openTab(n); } } catch (e) {}
}
function syncChoices() {
  for (const l of document.querySelectorAll("#regionChoices .choice")) {
    const on = l.dataset.m === regionMode;
    l.classList.toggle("on", on); l.querySelector("input").checked = on;
  }
}
// REGIONS ( sdot.region ): the frame's cells replaced by those of a region, rebuilt by the service from the seeds. The cut
// is the plane through the view's target, across the view: the cells whose seed is beyond it stay. Its plane is taken
// when asked ( `cut here` ), then kept from frame to frame.
var regionMode = "off", regionPlane = null, regionShown = null;   // ( regionShown: the bytes of the region on display )
const regionCache = new Map();
function regionQuery() {
  if (regionMode === "boundary") return "mode=boundary";
  if (!regionPlane) {
    const f = camBasis().fwd, n = new Array(D).fill(0);
    let d = 0;
    for (let k = 0; k < 3; k++) if (AX[k] >= 0) { n[AX[k]] = -f[k]; d -= f[k] * cam.target[k]; }
    regionPlane = n.concat([d]);
  }
  return "mode=half&plane=" + regionPlane.map(v => v.toPrecision(7)).join(",");
}
async function showRegion(f) {
  const hint = document.getElementById("regionHint");
  if (!SVC.region || !STORE.has(f)) return;
  if (regionMode === "off" || !STORE.m.frames[f].sources) {
    hint.style.display = "none"; regionShown = null;
    useScene(STORE.get(f)); buildScene(); draw();
    return;
  }
  const q = regionQuery(), key = f + "?" + q;
  let ch = regionCache.get(key);
  if (!ch) {
    hint.textContent = "asking sdot.region…"; hint.style.display = "block";
    let r;
    try { r = await fetch(SVC.region + "frame/" + f + "?" + q, { cache: "no-store" }); }
    catch (e) { hint.textContent = "sdot.region: " + e; return; }
    if (!r.ok) { hint.textContent = "sdot.region: " + r.status + " " + (await r.text()); return; }
    const raw = await r.arrayBuffer();
    ch = await readChunk(raw);
    ch.raw = raw;
    ch.colBase = COLORS.length;                          // ( its colours appended to the tables, light and dark )
    for (const c of ch.meta.colors) COLORS.push(c);
    for (const c of ch.meta.colors_dark) COLORS_DARK.push(c);
    regionCache.set(key, ch);
    if (regionCache.size > 8) regionCache.delete(regionCache.keys().next().value);
  }
  if (f !== frameIdx) return;
  if (regionShown === ch.raw && curScene && curScene.REGION === ch) { hint.textContent = ch.meta.cells + " cells"; return; }   // ( already shown )
  const sc = STORE.get(f), A = ch.arrays, nv = sc.POS.length / D;
  const plus = (x, d, T) => { const o = new T(x.length); for (let i = 0; i < x.length; i++) o[i] = x[i] + d; return o; };
  const pos = new Float32Array(sc.POS.length + A.pos.length);
  pos.set(sc.POS); pos.set(A.pos, sc.POS.length);
  useScene(Object.assign({}, sc, {
    POS: pos, POLY_V: plus(A.poly_v, nv, Int32Array), POLY_OFF: A.poly_off, POLY_C: plus(A.poly_c, ch.colBase, Uint16Array),
    POLY_AO: new Uint8Array(0), EDG_V: plus(A.edge_v, nv, Int32Array), EDG_C: plus(A.edge_c, ch.colBase, Uint16Array),
    EDG_L: new Uint8Array(A.edge_c.length), SHARED: { poly: 0, edge: 0, pnt: sc.SHARED.pnt }, REGION: ch,
  }));
  regionShown = ch.raw;
  buildScene(); draw();
  hint.textContent = ch.meta.cells + " cells"; hint.style.display = "block";
}
// PICTURES by Blender. The Render button renders the view; its MODE ( the icon beside it ): once, when asked; when the
// view rests ( `restDelay` without a move ); also while playing ( each frame, the first step only ). A picture is
// refined by the steps of the Pictures pane ( 8, 32, 128 samples... ), each replacing the last; it stays over the view
// until the view, the frame or the size change.
var renderShownKey = null, renderSeq = 0, lastViewKey = null, lastCamKey = null, restTimer = 0, cutTimer = 0, renderBusy = false;   // ( var: draw() runs before )
let renderMode = "once";
try { renderMode = localStorage.getItem("sviz.renderMode") || "once"; } catch (e) {}
const MODE_ICON = {
  once: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h3l1.5-2h5L14 6h3v10H3z"/><circle cx="10" cy="11" r="2.6"/></svg>',
  rest: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M16 10a6 6 0 1 1-1.8-4.3"/><path d="M16 3.5v3.2h-3.2"/><circle cx="10" cy="10" r="1.6" fill="currentColor" stroke="none"/></svg>',
  play: '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M16 10a6 6 0 1 1-1.8-4.3"/><path d="M16 3.5v3.2h-3.2"/><path d="M8.5 7.5v5l4-2.5z" fill="currentColor"/></svg>',
};
const MODES = [["once", "Once", "a picture of this view when you press Render"],
               ["rest", "When the view rests", "rendered once the view stays still, then refined"],
               ["play", "Also while playing", "each frame, the first step only"]];
function syncRenderMode() {
  const b = document.getElementById("renderMode");
  b.innerHTML = MODE_ICON[renderMode];
  b.title = "when to render: " + MODES.find(m => m[0] === renderMode)[1].toLowerCase();
  const menu = document.getElementById("renderMenu");
  menu.innerHTML = "";
  for (const [m, label, what] of MODES) {
    const it = document.createElement("button"); it.setAttribute("role", "menuitemradio");
    it.setAttribute("aria-checked", String(m === renderMode)); it.classList.toggle("on", m === renderMode);
    it.innerHTML = MODE_ICON[m] + "<span></span><small></small>";
    it.children[1].textContent = label; it.children[2].textContent = what;
    it.addEventListener("click", () => {
      renderMode = m; try { localStorage.setItem("sviz.renderMode", m); } catch (e) {}
      syncRenderMode(); closePops(); lastViewKey = null;
      if (m !== "once") renderStale();               // ( a mode that renders: now )
    });
    menu.appendChild(it);
  }
  const hr = document.createElement("hr"); menu.appendChild(hr);
  const note = document.createElement("small"); note.className = "muted"; note.style.cssText = "padding: 2px 10px 6px; font-size: 12px";
  note.textContent = "the steps and the delay: the Pictures tab"; menu.appendChild(note);
}
document.getElementById("renderMode").addEventListener("click", e => {
  if (openPop("renderMenu", e.currentTarget)) e.currentTarget.classList.add("open");
});
document.getElementById("renderBtn").addEventListener("click", () => renderView(false));
syncRenderMode();
// the steps: the samples of each picture of a series
for (const b of document.querySelectorAll("#stepsSeg button")) {
  b.classList.toggle("on", ["8", "32", "128"].includes(b.dataset.n));
  b.addEventListener("click", () => {
    b.classList.toggle("on");
    if (!document.querySelector("#stepsSeg button.on")) b.classList.add("on");
  });
}
const renderSteps = () => Array.from(document.querySelectorAll("#stepsSeg button.on")).map(b => parseInt(b.dataset.n, 10)).sort((a, b) => a - b);
function camKey() { return JSON.stringify([camEye(), cam.rot, cam.ortho, canvas.clientWidth, canvas.clientHeight]); }
function renderKey() { return camKey() + frameIdx + "/" + sceneVersion; }
function renderRequest(samples, scale, keep) {
  const b = camBasis(), H = canvas.clientHeight;
  return { frame: frameIdx, axes: AX, theme: document.body.classList.contains("dark") ? "dark" : "light",
           camera: { eye: camEye(), right: b.right, up: b.up, fwd: b.fwd, fov: FOV, ortho: !!cam.ortho, half_h: orthoHalfH() },
           view: { target: cam.target, dist: cam.dist, rot: cam.rot, ortho: !!cam.ortho },
           width: Math.max(16, Math.round(canvas.clientWidth * scale)), height: Math.max(16, Math.round(H * scale)), samples,
           point_radius: pointRadius(), edge_radius: 0.8 * 2 * orthoHalfH() / H,
           faces: ui.faces.checked, edges: ui.edgeMode.value !== "none", points: ui.points.checked, keep: !!keep,
           ...(FLOOR ? { floor: FLOOR } : {}),
           ...(fieldIdx >= 0 && shownRange ? { field: { name: FIELDS[fieldIdx].name, lo: shownRange[0], hi: shownRange[1],
                                                         diverging: isDiverging(FIELDS[fieldIdx]) } } : {}),
           ...(regionShown ? { region: b64Of(regionShown) } : {}) };          // ( the cells on display, not the stored ones )
}
function b64Of(buf) {
  const u = new Uint8Array(buf); let s = "";
  for (let i = 0; i < u.length; i += 0x8000) s += String.fromCharCode.apply(null, u.subarray(i, i + 0x8000));
  return btoa(s);
}
const renderUi = { img: document.getElementById("renderImg"), hint: document.getElementById("renderHint"),
                   hintText: document.getElementById("renderHintText"), card: document.getElementById("renderCard"),
                   info: document.getElementById("renderInfo") };
function say(text) { renderUi.hintText.textContent = text; renderUi.hint.style.display = text ? "flex" : "none"; }
let lastSamples = 0;
async function renderView(keep, only) {
  if (!SVC.blender) return;
  const seq = ++renderSeq, key = renderKey();
  const all = renderSteps(), steps = keep ? [lastSamples || all[all.length - 1]] : (only ? all.slice(0, 1) : all);
  renderBusy = true;
  try {
    for (const samples of steps) {
      say((keep ? "keeping, " : "rendering, ") + samples + " samples…");
      let r;
      try {
        r = await fetch(SVC.blender + "render", { method: "POST", headers: { "Content-Type": "application/json" },
                                                  body: JSON.stringify(renderRequest(samples, samples <= 8 && !keep ? 0.5 : 1, keep)) });
      } catch (e) { say("render failed: " + e); return; }
      if (!r.ok) { say("render failed: " + r.status + " " + (await r.text())); return; }
      const blob = await r.blob();
      if (seq !== renderSeq || key !== renderKey()) return;              // ( the view moved: that picture is for nothing )
      if (renderUi.img.src) URL.revokeObjectURL(renderUi.img.src);
      renderUi.img.src = URL.createObjectURL(blob);
      renderUi.img.style.width = canvas.clientWidth + "px"; renderUi.img.style.height = canvas.clientHeight + "px";
      renderUi.img.style.display = "block";
      document.getElementById("downloadBtn").href = renderUi.img.src;
      renderShownKey = key; lastSamples = samples;
      renderUi.info.textContent = "Cycles · " + samples + " samples" + (NB_FRAMES > 1 ? " · " + AXES[0] + " " + fmtVal(FR_VAL[frameIdx * NB_AXES]) : "");
      renderUi.card.style.display = "flex";
      const kept = r.headers.get("X-Kept");
      if (kept) { say("kept: " + kept); loadKept(); setTimeout(() => { if (renderShownKey === key) say(""); }, 2500); }
    }
    if (!keep) say("");
  } finally { renderBusy = false; }
}
// at each draw: a picture over an older view goes; a mode that renders asks again once the view rests ( or at once,
// playing, when nothing is on its way )
function renderStale() {
  const k = renderKey();
  if (k === lastViewKey) return;
  lastViewKey = k;
  if (renderShownKey !== null && renderShownKey !== k) {
    renderUi.img.style.display = "none"; renderUi.card.style.display = "none"; renderShownKey = null; say("");
  }
  const delay = 1000 * parseFloat(document.getElementById("restDelay").value);
  // a cut that follows the view: made again through the new target once the CAMERA rests ( its own change of scene
  // does not ask for another )
  const ck = camKey();
  if (ck !== lastCamKey) {
    lastCamKey = ck;
    if (SVC.region && regionMode === "cut" && document.getElementById("regionFollow").checked && playing < 0) {
      clearTimeout(cutTimer);
      cutTimer = setTimeout(() => { if (!drag && !touch && camKey() === ck) { regionPlane = null; showRegion(frameIdx); } }, delay);
    }
  }
  if (!SVC.blender || renderMode === "once") return;
  clearTimeout(restTimer);
  if (playing >= 0) { if (renderMode === "play" && !renderBusy) renderView(false, true); return; }
  restTimer = setTimeout(() => { if (!drag && !touch && renderKey() === k) renderView(false); }, delay);
}
document.getElementById("keepBtn").addEventListener("click", () => renderView(true));
document.getElementById("closeRenderBtn").addEventListener("click", () => {
  renderSeq++; renderUi.img.style.display = "none"; renderUi.card.style.display = "none"; renderShownKey = null; say("");
});
{ const cmp = document.getElementById("compareBtn");               // ( held: the live view under the picture )
  cmp.addEventListener("pointerdown", () => { renderUi.img.style.visibility = "hidden"; });
  for (const ev of ["pointerup", "pointerleave"]) cmp.addEventListener(ev, () => { renderUi.img.style.visibility = ""; }); }
// what was kept ( `renders/index.json`, written by the service ): a click goes back to that frame and that view
async function loadKept() {
  let list = [];
  try { const r = await fetch("renders/index.json", { cache: "no-store" }); if (r.ok) list = await r.json(); } catch (e) {}
  const box = document.getElementById("kept");
  box.innerHTML = "";
  document.getElementById("keptCount").textContent = list.length ? "· " + list.length : "";
  document.getElementById("keptHint").textContent = list.length ? "a click: back to that frame and that view" : "Keep, over a picture, writes it here";
  for (const e of list.slice().reverse()) {
    const fig = document.createElement("figure"), img = document.createElement("img"), cap = document.createElement("figcaption");
    img.src = e.file; img.alt = "kept picture"; img.loading = "lazy";
    cap.textContent = (e.frame < NB_FRAMES && NB_FRAMES > 1 ? AXES[0] + " " + fmtVal(FR_VAL[e.frame * NB_AXES]) + " · " : "") + e.samples + " samples";
    fig.append(img, cap); box.appendChild(fig);
    fig.addEventListener("click", () => {
      if (e.view) { cam.target = e.view.target.slice(); cam.dist = e.view.dist; cam.rot = e.view.rot.slice(); cam.ortho = e.view.ortho; syncProj(); viewUntouched = false; }
      if (e.frame < NB_FRAMES) setFrame(e.frame); else draw();
    });
  }
}
// VIEWS: named cameras, kept in this browser ( per store )
const VIEWS_KEY = "sviz.views:" + location.pathname;
function readViews() { try { return JSON.parse(localStorage.getItem(VIEWS_KEY) || "[]"); } catch (e) { return []; } }
function syncViews() {
  const box = document.getElementById("views"), list = readViews();
  box.innerHTML = "";
  for (const v of list) {
    const b = document.createElement("button"); b.textContent = v.name; b.title = "go to this view ( alt-click: forget it )";
    b.addEventListener("click", e => {
      if (e.altKey) { try { localStorage.setItem(VIEWS_KEY, JSON.stringify(readViews().filter(w => w.name !== v.name))); } catch (err) {} syncViews(); return; }
      cam.target = v.target.slice(); cam.dist = v.dist; cam.rot = v.rot.slice(); cam.ortho = v.ortho; syncProj(); viewUntouched = false; draw();
    });
    box.appendChild(b);
  }
  const add = document.createElement("button"); add.className = "add"; add.textContent = "+ this view";
  add.addEventListener("click", () => {
    const name = window.prompt("a name for this view", "view " + (list.length + 1));
    if (!name) return;
    const all = readViews().filter(w => w.name !== name);
    all.push({ name, target: cam.target.slice(), dist: cam.dist, rot: cam.rot.slice(), ortho: !!cam.ortho });
    try { localStorage.setItem(VIEWS_KEY, JSON.stringify(all)); } catch (e) {}
    syncViews();
  });
  box.appendChild(add);
}
syncViews();
function bakeFrame(f) {
  if (!serviceBake || !SVC.ao || !STORE.has(f)) return;
  const sc = STORE.get(f);
  if (!sc.POLY_V.length || (sc.POLY_AO && sc.POLY_AO.length) || aoAsked.has(f)) return;
  aoAsked.set(f, fetch(SVC.ao + "frame/" + f, { cache: "no-store" })
    .then(r => r.ok ? r.arrayBuffer() : Promise.reject(new Error("sdot.ao: " + r.status)))
    .then(readChunk)
    .then(ch => {
      sc.POLY_AO = ch.arrays.poly_ao;
      const btn = document.getElementById("bakeBtn");
      btn.style.display = "none"; document.getElementById("bakeLbl").style.display = "";
      if (f === frameIdx && curScene === sc) { POLY_AO = sc.POLY_AO; buildScene(); draw(); }
    }, e => { aoAsked.delete(f); console.error(e); document.getElementById("bakeBtn").textContent = "bake failed"; }));
}
findServices();

// beyond 3 dimensions: we choose the 3 we look at, and fix the others (the SLICE).
if (D > 3) {
  document.getElementById("axes").style.display = "block";
  const sels = [document.getElementById("axX"), document.getElementById("axY"), document.getElementById("axZ")];
  sels.forEach((sel, a) => {
    for (let k = 0; k < D; k++) sel.add(new Option("x" + k, k));
    sel.value = AX[a];
    sel.addEventListener("change", () => {
      const v = parseInt(sel.value, 10);
      const other = sels.findIndex((s, i) => i !== a && parseInt(s.value, 10) === v);
      if (other >= 0) { sels[other].value = AX[a]; AX[other] = AX[a]; }   // we SWAP
      AX[a] = v;
      projectPool(); buildSlices(); buildScene(); resetView();
    });
  });
  buildSlices();
}
function buildSlices() {
  const box = document.getElementById("slices");
  box.innerHTML = "";
  const hidden = [];
  for (let k = 0; k < D; k++) if (k !== AX[0] && k !== AX[1] && k !== AX[2]) hidden.push(k);
  if (!hidden.length) return;
  box.style.cssText = "display: flex; flex-direction: column; gap: 6px";
  const t = document.createElement("h3");
  t.textContent = "slice";
  box.appendChild(t);
  for (const k of hidden) {
    const row = document.createElement("div");
    row.className = "row";
    const lab = document.createElement("span");
    const inp = document.createElement("input");
    inp.type = "range"; inp.min = BOUNDS[k][0]; inp.max = BOUNDS[k][1];
    inp.step = (BOUNDS[k][1] - BOUNDS[k][0]) / 400 || 0.01;
    inp.value = SLICE[k];
    const upd = () => { lab.textContent = "x" + k + " = " + parseFloat(inp.value).toPrecision(3); };
    upd();
    inp.addEventListener("input", () => { SLICE[k] = parseFloat(inp.value); upd(); buildScene(); draw(); });
    const val = document.createElement("span"); val.className = "val";
    row.appendChild(lab); row.appendChild(inp); row.appendChild(val);
    box.appendChild(row);
  }
}

if (FLAT) {
  document.getElementById("hDrag").textContent = "move (pan)";
  document.getElementById("hArrows").textContent = "move (Shift/Ctrl: faster)";
}

// theme: `?theme=dark|light`, else the viewer's choice ( Display: light, dark, auto -- kept in this browser ), auto
// following the page that embeds this one ( its `dark` class, followed live: a VitePress page, same origin ), else the
// system preference; [d] flips light and dark. Colors given per theme switch with it.
let sceneBuilt = false;
(function () {
  const ml = window.matchMedia("(prefers-color-scheme: dark)");
  const asked = URL_OPTS.get("theme");
  let host = null;
  try { if (window.parent !== window) host = window.parent.document.documentElement; } catch (e) { host = null; }
  let mode = asked || "auto", cur = false;
  if (!asked && !EMBED) try { mode = localStorage.getItem("sviz.theme") || "auto"; } catch (e) {}
  const autoDark = () => host ? host.classList.contains("dark") : ml.matches;
  function setTheme(on) {
    cur = on;
    document.body.classList.toggle("dark", on);
    document.body.dataset.theme = on ? "dark" : "light";
    document.getElementById("modeLabel").textContent = on ? "dark" : "light";
    const pal = on ? COLORS_DARK : COLORS;
    if (pal !== PAL) { PAL = pal; if (sceneBuilt) buildScene(); }
    draw();
    curveCursors();
    if (typeof syncCurvesBtn === "function" && typeof charts !== "undefined" && charts.length) syncCurvesBtn();
  }
  window.setThemeMode = m => {
    mode = m;
    if (!asked && !EMBED) try { localStorage.setItem("sviz.theme", m); } catch (e) {}
    for (const b of document.querySelectorAll("#themeSeg button")) b.classList.toggle("on", b.dataset.t === m);
    setTheme(m === "auto" ? autoDark() : m === "dark");
  };
  window.toggleTheme = () => window.setThemeMode(cur ? "light" : "dark");
  window.setThemeMode(mode);
  if (host) new MutationObserver(() => { if (mode === "auto") setTheme(autoDark()); }).observe(host, { attributes: true, attributeFilter: ["class"] });
  ml.addEventListener("change", () => { if (mode === "auto") setTheme(autoDark()); });
  document.addEventListener("keydown", e => {
    if ((e.key === "d" || e.key === "D") && e.target.tagName !== "SELECT" && e.target.tagName !== "INPUT")
      { e.preventDefault(); window.toggleTheme(); }
  });
})();

// the view refits the box on a resize until the user moves it
let viewUntouched = true;
canvas.addEventListener("pointerdown", () => { viewUntouched = false; }, { passive: true });
canvas.addEventListener("wheel", () => { if (wheelEnabled) viewUntouched = false; }, { passive: true });
canvas.addEventListener("touchstart", () => { if (touchEnabled) viewUntouched = false; }, { passive: true });
window.addEventListener("keydown", () => { viewUntouched = false; });
window.addEventListener("resize", resize);
projectPool();
buildCurves();
setFrame(NB_FRAMES - 1);          // the final state first -- it is what we want to see first
// a store whose run goes on: its manifest asked again every second; new frames stretch the bar ( the view follows them
// when it shows the last one and does not play ), a new base or new curves are taken as they come
if (STORED && !STORE.m.complete) (function follow() {
  STORE.poll().then(ch => {
    if (ch.frames) {
      const last = frameIdx === NB_FRAMES - 1 && playing < 0;
      FR_VAL = STORE.FR_VAL; NB_FRAMES = FR_VAL.length / NB_AXES;
      for (const f of FIELDS) { f.all = STORE.fieldAll(f.name); f.div = undefined; }
      frameGrid(); buildBar();
      if (playing >= 0 && axRows[playing]) { axRows[playing].btn.innerHTML = PAUSE_ICON; axRows[playing].btn.classList.add("on"); }
      if (last) setFrame(NB_FRAMES - 1); else syncBar();
      syncStatus();
    }
    if (ch.base) { curScene = null; setFrame(frameIdx); }
    if (ch.curves) { CURVES = STORE.m.curves; refreshCurves(); }
    if (!STORE.m.complete) setTimeout(follow, 1000); else syncStatus();
  }, () => setTimeout(follow, 2000));
})();
// a store served: the page says it is open ( a server that stops when nobody asks, `sviz serve --idle`, keeps going )
if (STORED && !STORE_FILES) setInterval(() => fetch("manifest.json", { method: "HEAD", cache: "no-store" }).catch(() => {}), 60000);
setLight(ui.edl.checked || ui.ao.checked ? "relief" : "flat");
syncProj(); syncStatus(); syncLayers();
resize();                         // BEFORE framing: the latter depends on the window's aspect ratio
resetView();
// The bar of an axis takes the focus: the arrows drive it NATIVELY (and Home/End too),
// without taking anything away from the camera -- a click on the view gives the arrows back to the orbit.
if (focusAxis >= 0) axRows[focusAxis].inp.focus({ preventScroll: true });
if (shownAxes.length && URL_OPTS.get("gif") === "1") setTimeout(exportGif, 400);
</script>
<script>
// the LOADER. A self-contained page: its blobs inflated ( the browser's DecompressionStream; their bytes shuffled back,
// see `_pack` ), then the main script run. A STORE page ( `index.html` in the directory of a store, see `store.py` ): the
// manifest, the base and the last frame read first, into `STORE` ( see `storeOpen` ), then the main script run.
async function inflate(bytes, k) {
  const out = new Uint8Array(await new Response(new Blob([bytes]).stream().pipeThrough(new DecompressionStream("deflate"))).arrayBuffer());
  if (k <= 1) return out;
  const n = out.length / k, u = new Uint8Array(out.length);
  for (let j = 0; j < k; j++) for (let i = 0, o = j * n; i < n; i++) u[i * k + j] = out[o + i];
  return u;
}
// a CHUNK: u32 length | JSON header | blobs ( see `store.py` ); the 16-bit codes back to float32
const DTYPES = { f4: Float32Array, f8: Float64Array, i4: Int32Array, u4: Uint32Array, i2: Int16Array, u2: Uint16Array, i1: Int8Array, u1: Uint8Array };
async function readChunk(buf) {
  const n = new DataView(buf).getUint32(0, true);
  const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, n))), arrays = {};
  await Promise.all(head.arrays.map(async e => {
    const raw = await inflate(new Uint8Array(buf, 4 + n + e.at, e.len), e.k);
    let a = new DTYPES[e.dtype](raw.buffer, 0, raw.length / e.k);
    if (e.lo) {
      const c = e.shape.length === 2 ? e.shape[1] : 1, f = new Float32Array(a.length);
      for (let i = 0; i < a.length; i++) f[i] = a[i] === 65535 ? NaN : e.lo[i % c] + e.step[i % c] * a[i];
      a = f;
    }
    arrays[e.name] = a;
  }));
  return { arrays, meta: head.meta };
}
// the STORE: the manifest, the base, and a cache of frames ( 256 MB, the least recently shown dropped first ), each frame
// joined to the base as one scene in the arrays of the self-contained page ( `combine` )
const STORE_FILES = __STORE_FILES__;                  // ( the files inlined: `sviz export --html` )
const bytes = s => { const r = atob(s), b = new Uint8Array(r.length); for (let i = 0; i < r.length; i++) b[i] = r.charCodeAt(i); return b; };
async function storeOpen() {
  const get = async (path, json) => {
    if (STORE_FILES) {
      if (!(path in STORE_FILES)) throw new Error(path + ": not in the page");
      return json ? JSON.parse(STORE_FILES[path]) : bytes(STORE_FILES[path]).buffer;
    }
    if (location.protocol === "file:") {
      document.body.innerHTML = '<p style="font:14px sans-serif;margin:2em">This page reads the files beside it: open the directory with <code>sviz open</code>.</p>';
      throw new Error("a store read as file://");
    }
    const r = await fetch(path, { cache: "no-store" });
    if (!r.ok) throw new Error(path + ": HTTP " + r.status);
    return json ? r.json() : r.arrayBuffer();
  };
  let m = await get("manifest.json", true);
  if (!m.frames.length || !m.dim) {                    // a run that has not written its first frame yet
    const note = document.createElement("div");
    note.style.cssText = "position:fixed;inset:0;display:flex;align-items:center;justify-content:center;font:14px sans-serif;color:#888";
    note.textContent = "waiting for the first frame of " + (m.title || "the run") + " ...";
    document.body.appendChild(note);
    while (!m.frames.length || !m.dim) { await new Promise(r => setTimeout(r, 1000)); m = await get("manifest.json", true); }
    note.remove();
  }
  const D = m.dim, S = { m, cache: new Map(), loading: new Map(), bytes: 0, budget: 256 * 2 ** 20, base: null, baseName: null };
  const fieldNames = () => {
    const out = [];
    for (const f of S.m.frames) for (const k in f.fields || {}) if (!out.includes(k)) out.push(k);
    return out;
  };
  S.fieldAll = name => {                               // the range of a field over all the frames
    let lo = Infinity, hi = -Infinity;
    for (const f of S.m.frames) { const r = (f.fields || {})[name]; if (r) { lo = Math.min(lo, r[0]); hi = Math.max(hi, r[1]); } }
    return lo <= hi ? [lo, hi] : null;
  };
  const cat = (T, x, y) => {
    const o = new T((x ? x.length : 0) + (y ? y.length : 0));
    if (x) o.set(x); if (y) o.set(y, x ? x.length : 0);
    return o;
  };
  const shift = (x, d) => { if (!x || !d) return x; const o = new Int32Array(x.length); for (let i = 0; i < x.length; i++) o[i] = x[i] + d; return o; };
  const nans = n => new Float32Array(n).fill(NaN);
  function combine(b, fr) {                            // the base first ( the shared primitives ), then the frame
    const A = b.arrays, F = fr.arrays, nv = A.pos ? A.pos.length / D : 0;
    const nb = { poly: A.poly_c ? A.poly_c.length : 0, edge: A.edge_c ? A.edge_c.length : 0, pnt: A.pnt_c ? A.pnt_c.length : 0 };
    const nf = F.pnt_c ? F.pnt_c.length : 0, offB = A.poly_off ? A.poly_off[A.poly_off.length - 1] : 0;
    const sc = {
      POS: cat(Float32Array, A.pos, F.pos),
      POLY_V: cat(Int32Array, A.poly_v, shift(F.poly_v, nv)),
      POLY_OFF: cat(Int32Array, A.poly_off || new Int32Array(1), F.poly_off ? shift(F.poly_off.subarray(1), offB) : null),
      POLY_C: cat(Uint16Array, A.poly_c, F.poly_c),
      POLY_AO: A.poly_ao || F.poly_ao ? cat(Uint8Array, A.poly_v ? A.poly_ao || new Uint8Array(A.poly_v.length).fill(255) : null,
                                         F.poly_v ? F.poly_ao || new Uint8Array(F.poly_v.length).fill(255) : null) : new Uint8Array(0),
      EDG_V: cat(Int32Array, A.edge_v, shift(F.edge_v, nv)), EDG_C: cat(Uint16Array, A.edge_c, F.edge_c), EDG_L: cat(Uint8Array, A.edge_l, F.edge_l),
      PNT_V: cat(Int32Array, A.pnt_v, shift(F.pnt_v, nv)), PNT_C: cat(Uint16Array, A.pnt_c, F.pnt_c),
      PNT_R: cat(Float32Array, A.pnt_r, F.pnt_r), PNT_M: cat(Float32Array, A.pnt_m, F.pnt_m), PNT_L: cat(Uint8Array, A.pnt_l, F.pnt_l),
      SHARED: nb, HPOLY: [], fields: {},
    };
    for (const name of fieldNames()) sc.fields[name] = cat(Float32Array, nans(nb.pnt), F["field:" + name] || nans(nf));
    let at = 0;
    for (const h of fr.meta.hpolys || []) {
      sc.HPOLY.push({ nb: h.nb, dirs: F.hp_dirs.subarray(at * D, (at + h.nb) * D), offs: F.hp_offs.subarray(at, at + h.nb),
                      fcol: h.fcol, ecol: h.ecol, edg: h.edg });
      at += h.nb;
    }
    sc.bytes = 0;
    for (const k in sc) if (sc[k] && sc[k].byteLength) sc.bytes += sc[k].byteLength;
    for (const k in sc.fields) sc.bytes += sc.fields[k].byteLength;
    return sc;
  }
  S.setBase = async name => {
    S.base = name ? await readChunk(await get(name)) : { arrays: {}, meta: {} };
    S.baseName = name; S.cache.clear(); S.bytes = 0;
  };
  S.has = f => S.cache.has(f);
  S.get = f => {                                       // ( the most recently shown last: the order of the drops )
    const sc = S.cache.get(f);
    if (sc) { S.cache.delete(f); S.cache.set(f, sc); }
    return sc;
  };
  S.load = f => {
    if (S.cache.has(f)) return Promise.resolve(S.cache.get(f));
    if (S.loading.has(f)) return S.loading.get(f);
    const base = S.base;
    const p = get(S.m.frames[f].file).then(readChunk).then(fr => {
      S.loading.delete(f);
      if (base !== S.base) return S.load(f);           // the base changed meanwhile
      const sc = combine(base, fr);
      S.cache.set(f, sc); S.bytes += sc.bytes;
      for (const [k, v] of S.cache) {
        if (S.bytes <= S.budget || S.cache.size <= 2) break;
        if (k !== f) { S.cache.delete(k); S.bytes -= v.bytes; }
      }
      return sc;
    }, e => { S.loading.delete(f); throw e; });
    S.loading.set(f, p);
    return p;
  };
  // `cb( f )` once frame `f` is there, and the next ones asked ahead
  S.want = (f, cb) => {
    if (S.cache.has(f)) cb(f); else S.load(f).then(() => cb(f), e => console.error(e));
    for (const g of [f + 1, f + 2, f + 3, f - 1])
      if (g >= 0 && g < S.m.frames.length && !S.cache.has(g)) S.load(g).catch(() => {});
  };
  const values = () => new Float32Array(S.m.frames.flatMap(f => f.v));
  // the manifest again: what changed ( frames, base, curves )
  S.poll = async () => {
    const m = await get("manifest.json", true);
    const ch = { frames: m.frames.length !== S.m.frames.length, curves: JSON.stringify(m.curves) !== JSON.stringify(S.m.curves) };
    if (m.base !== S.baseName) { await S.setBase(m.base); ch.base = true; }
    S.m = m; S.FR_VAL = values();
    return ch;
  };

  // what the main script reads where the self-contained page has its data inlined ( see `store_page` )
  await S.setBase(m.base);
  const last = m.frames.length - 1, sc = await S.load(last), o = m.options || {};
  const tight = m.bounds || Array.from({ length: D }, () => [-1, 1]);
  const span = Math.max(...tight.map(([a, b]) => Math.max(b - a, 1e-9))), pad = 0.05 * span;
  const bounds = tight.map(([a, b]) => [a - pad, b + pad]), r0 = (span + 2 * pad) / 300;
  const fr = (a, b) => { const r = new Int32Array(m.frames.length + 1); r.fill(a, 0, last + 1); r.fill(b, last + 1); return r; };
  const images = (S.base.meta.images || []).map((im, i) => Object.assign({}, im, { data: S.base.arrays["image:" + i] }));
  Object.assign(S, sc, {
    D, BOUNDS: bounds, TIGHT: tight, COLORS: m.colors, COLORS_DARK: m.colors_dark, LAYERS: m.layers, FIELD0: m.field,
    FIELDS: fieldNames().map(name => ({ name, data: sc.fields[name], all: S.fieldAll(name) })), IMAGES: images,
    CURVES: m.curves, AXES: m.axes, PLAYABLE: m.playable, FPS: o.fps || 10, FR_VAL: values(),
    FR_POLY: fr(sc.SHARED.poly, sc.POLY_OFF.length - 1), FR_EDG: fr(sc.SHARED.edge, sc.EDG_V.length / 2),
    FR_PNT: fr(sc.SHARED.pnt, sc.PNT_V.length), FR_HP: fr(0, sc.HPOLY.length),
    R0: r0, RMIN: r0 / 30, RMAX: r0 * 60, BG: o.bg || [null, null], WHEEL: true, TOUCH: true,
    DEPTHCOL: !!o.depth_colors, FACE_OP: o.face_opacity === undefined ? 1 : o.face_opacity, MORPH: o.morph || 0,
    LOOP: !!o.loop, FLATPTS: !!o.flat_points, AUTOPLAY: !!o.autoplay,
  });
  return S;
}
(async () => {
  const RAW = __BLOBS__;
  if (__STORE_MODE__) window.STORE = await storeOpen();
  else window.BLOBS = await Promise.all(RAW.map(e => inflate(bytes(e.z), e.k)));
  const main = document.createElement("script");
  main.textContent = document.getElementById("main").textContent;
  document.body.appendChild(main);
})();
</script>
</body>
</html>
"""
