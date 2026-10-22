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
    return tuple( c * factor for c in _rgb( color ) )


def _cat( blocks, dtype ):
    """Concatenates blocks (one per addition) into a single flat array of the desired dtype."""
    if not blocks:
        return np.zeros( 0, dtype = dtype )
    return np.concatenate( [ np.asarray( b ).reshape( -1 ) for b in blocks ] ).astype( dtype )


def _b64( arr ):
    return base64.b64encode( np.ascontiguousarray( arr ).tobytes() ).decode( "ascii" )


class Visualizer:
    """Collects geometric primitives; the outputs (`write_html`, ...) come afterwards.

    `title` names the scene (the tab, on the HTML side). `depth_colors`: in 3D, the points are colored by their depth in the
    view ( warm in front, cool behind, recomputed as the view turns ) instead of their own color -- what a cloud needs to be
    read in depth ( `?depth=0|1` in the URL and the menu still switch it ).

    `morph = k`: the playback goes from one frame to the next in `k` eased steps, the points of the two frames matched by rank
    when they have as many -- the same diracs moving ( a transport, a descent ) without storing the steps; it holds a moment on
    each frame. `loop`: it starts again, morphing back to the first frame. `autoplay`: it plays as soon as the page opens
    ( `?play=0|1` in the URL wins ).

    `frame_axis`: the name of the frame axis, or a tuple of names when the frames vary along several parameters (see
    `new_frame`). `playable`: whether the axis gets a play button -- a bool for all the axes, or one per axis. Either way
    the bar of the axes sits next to the menu button, in plain sight: it is what a reader of the doc has to touch.

    `flat_points`: in 2D, the points are flat discs ( no sphere shading ) -- what a disc of the plane is, a particle of
    a reconstruction for instance.

    `face_opacity`: the starting value of the opacity slider of the faces ( `?opacity=` in the URL wins ) -- below 1 for a
    density drawn under diracs.
    """

    def __init__( self, title = "sdot", background = None, frame_axis = "time",
                  playable = True, fps = 5.0, frame_value = 0.0, depth_colors = False, morph = 0, loop = False,
                  autoplay = False, face_opacity = 1.0, flat_points = False ):
        self.title      = title
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
        self._hpolys                               = []           # polytopes in H-representation

        # the next free index of the scale. Reset to zero at each FRAME (see `new_frame`):
        # a color must say WHAT, not HOW MANY were drawn before.
        self._next_color = 0

        # A FRAME only remembers where it starts in the lists above: everything that is
        # added goes into the LAST one, so each frame occupies a contiguous range. Nothing
        # is duplicated, and a frame has no structure of its own to carry.
        self._nb_edges  = 0
        self._nb_points = 0
        self._frames    = [ self._frame_mark( frame_value ) ]       # ( `frame_value`: the abscissa of the first frame )

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
                 "poly": len( self._poly_len ), "edge": self._nb_edges,
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
            value = { self.frame_axes[ 0 ]: len( self._frames ) } if len( self.frame_axes ) > 1 else len( self._frames )
        self._frames.append( self._frame_mark( value ) )
        return self

    @property
    def nb_frames( self ):
        return len( self._frames )

    def _frame_ranges( self, key ):
        """The bounds of the frames over one of the primitive lists: `nb_frames + 1` integers."""
        total = { "poly": len( self._poly_len ), "edge": self._nb_edges,
                  "pnt": self._nb_points, "hpoly": len( self._hpolys ) }[ key ]
        return np.array( [ f[ key ] for f in self._frames ] + [ total ], dtype = np.int32 )

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

    def add_points( self, positions, radius = 0.0, color = None, opacity = 1.0, frames = True ):
        """`positions`: `[n, d]`. `radius`: the WORLD radius of the points (a scalar, or one radius
        per point). `0` (default) = the point has no size of its own (a Dirac): it is the page's
        slider that sets its display radius -- same convention as `points_html`.
        """
        base = self._pool( positions, frames )
        n = self._pool_len( base )
        if n == 0:
            return self
        rad = np.full( n, float( radius ), dtype = np.float32 ) \
            if np.isscalar( radius ) else _np( radius ).reshape( -1 )
        if len( rad ) != n:
            raise ValueError( f"radius: { len( rad ) } radii for { n } points" )
        ci = self._color_id( color, opacity )
        self._pnt_v.append( np.arange( base, base + n, dtype = np.int32 ) )
        self._pnt_r.append( rad )
        self._pnt_c.append( np.full( n, ci, dtype = np.uint16 ) )
        self._nb_points += n
        return self

    def add_edges( self, positions, edges = None, color = None, opacity = 1.0, closed = False,
                   dashed = False, nb_dashes = 7, frames = True ):
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
        """
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
            return self._add_dashes( positions, idx, color, opacity, nb_dashes, frames )
        ci = self._color_id( color, opacity )
        self._edge_v.append( idx.astype( np.int32 ) + base )
        self._edge_c.append( np.full( len( idx ), ci, dtype = np.uint16 ) )
        self._nb_edges += len( idx )
        return self

    def _add_dashes( self, positions, idx, color, opacity, nb_dashes, frames = True ):
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
                               opacity = opacity, frames = frames )

    def add_faces( self, positions, faces, color = None, opacity = 1.0, frames = True ):
        """`positions`: `[n, d]`. `faces`: a sequence of POLYGONS (lists of indices, free lengths).
        They are kept AS IS -- it is the output that decides whether to triangulate them.
        """
        base = self._pool( positions, frames )
        ci = self._color_id( color, opacity )
        nb = 0
        for f in faces:
            f = [ int( i ) + base for i in f ]
            if len( f ) < 3:
                continue
            self._poly_v += f
            self._poly_len.append( len( f ) )
            nb += 1
        self._poly_c.append( np.full( nb, ci, dtype = np.uint16 ) )
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
        `point_colors`, `point_radii`, `polytopes`. The indices refer to the `positions` pool,
        common to ALL frames.
        """
        def rng( key ):
            r = self._frame_ranges( key )
            return int( r[ i ] ), int( r[ i + 1 ] )

        p0, p1 = rng( "poly" )
        e0, e1 = rng( "edge" )
        v0, v1 = rng( "pnt" )
        h0, h1 = rng( "hpoly" )
        return {
            "value"         : self._frames[ i ][ "values" ] if len( self.frame_axes ) > 1 else self._frames[ i ][ "values" ][ 0 ],
            "polygons"      : self.polygons[ p0 : p1 ],
            "polygon_colors": self.polygon_colors[ p0 : p1 ],
            "edges"         : self.edges[ e0 : e1 ],
            "edge_colors"   : self.edge_colors[ e0 : e1 ],
            "points"        : self.points[ v0 : v1 ],
            "point_colors"  : self.point_colors[ v0 : v1 ],
            "point_radii"   : self.point_radii[ v0 : v1 ],
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

        path.parent.mkdir( parents = True, exist_ok = True )
        path.write_text( html )
        print( f"OUTPUT: file://{ path.absolute() }" )
        return path

    def to_html( self, wheel = True, touch = True ):
        """The self-contained HTML page, as a string. `wheel` / `touch` say whether the wheel and
        the touch gestures drive the view by default (`?wheel=0` / `?touch=0` in the URL still
        win) -- off when the page is embedded in a host page that must keep scrolling."""
        if self.nb_dims is None:
            raise ValueError( "Visualizer: nothing to display (no primitive added)" )

        d       = self.nb_dims
        bounds  = self.bounds()
        span    = max( hi - lo for lo, hi in bounds )
        r0      = span / 150                                    # starting radius of bare points

        poly_off = np.concatenate( [ [ 0 ], np.cumsum( self._poly_len ) ] ).astype( np.int32 )

        subs = {
            "__TITLE__"   : self.title,
            "__D__"       : str( d ),
            "__BOUNDS__"  : json.dumps( bounds ),
            "__TIGHT__"   : json.dumps( self.bounds( pad_ratio = 0.0 ) ),
            "__POS__"     : _b64( self.positions.astype( np.float32 ) ),
            "__COLORS__"  : json.dumps( [ list( c ) for c in self._colors ] ),
            "__COLORS_DARK__": json.dumps( [ list( c ) for c in self._colors_dark ] ),
            "__POLY_V__"  : _b64( np.asarray( self._poly_v, dtype = np.int32 ) ),
            "__POLY_OFF__": _b64( poly_off ),
            "__POLY_C__"  : _b64( self.polygon_colors ),
            "__EDG_V__"   : _b64( self.edges.reshape( -1 ) ),
            "__EDG_C__"   : _b64( self.edge_colors ),
            "__PNT_V__"   : _b64( self.points ),
            "__PNT_C__"   : _b64( self.point_colors ),
            "__PNT_R__"   : _b64( self.point_radii ),
            "__HPOLY__"   : json.dumps( self._hpolys ),
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
        }
        return re.sub( r"__[A-Z0-9_]+__", lambda m: subs.get( m.group( 0 ), m.group( 0 ) ), _HTML )


_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  html, body { margin: 0; padding: 0; overflow: hidden; background: #ffffff; color: #111; }
  canvas { display: block; }
  #controls {
    display: none; position: absolute; top: 30px; left: 0;
    max-height: calc(100vh - 50px); overflow-y: auto;
    background: rgba(255,255,255,0.88); padding: 8px 12px; border-radius: 6px;
    font-family: sans-serif; font-size: 13px; box-shadow: 0 1px 4px rgba(0,0,0,0.3);
    width: 244px; box-sizing: border-box;
  }
  #controls *, #controls { box-sizing: border-box; }
  #menu { position: fixed; top: 10px; left: 10px; z-index: 2; }
  #menuBtn {
    display: block; width: 30px; height: 30px; padding: 0; cursor: pointer; border: none;
    border-radius: 6px; background: rgba(255,255,255,0.88); color: #333; font-size: 18px; line-height: 1;
    box-shadow: 0 1px 4px rgba(0,0,0,0.3);
  }
  #menu:hover #controls, #menu.open #controls { display: block; }
  #controls a { color: inherit; }
  body.dark #menuBtn { background: rgba(30,30,30,0.92); color: #ddd; }
  #controls label { display: block; margin: 2px 0; }
  #controls input[type=range] { vertical-align: middle; width: 100%; }
  #controls select { font-size: 12px; }
  #controls .hint { color: #666; margin-top: 6px; }
  #controls .sec { margin-top: 8px; padding-top: 8px; border-top: 1px solid #ddd; }
  #controls .val { color: #666; font-variant-numeric: tabular-nums; }
  /* the frame axes: in plain sight, right of the menu button -- one row per axis */
  #anim {
    display: none; position: fixed; top: 10px; left: 48px; z-index: 1;
    width: min(460px, calc(100vw - 96px)); box-sizing: border-box; padding: 2px 8px;
    background: rgba(255,255,255,0.88); border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,0.3);
    font-family: sans-serif; font-size: 13px;
    grid-template-columns: auto minmax(40px, 1fr) auto; align-items: center; column-gap: 8px;
  }
  #anim .row { display: contents; }
  #anim .row > * { margin: 2px 0; }
  #anim input[type=range] { width: 100%; accent-color: #2a7de1; }
  #anim .play {
    cursor: pointer; border: none; border-radius: 50%; background: #2a7de1; color: #fff; padding: 0;
    width: 22px; height: 22px; font-size: 10px; line-height: 22px; flex: none;
  }
  #anim .play.on { background: #444; }
  #anim .gap { width: 22px; flex: none; }
  #anim .lab { flex: none; white-space: nowrap; font-variant-numeric: tabular-nums; }
  #anim .cnt { color: #888; font-size: 11px; margin-left: 4px; }
  body.dark #anim { background: rgba(30,30,30,0.92); color: #ddd; }
  body.dark #anim input[type=range] { accent-color: #8ab4f8; }
  body.dark #anim .play { background: #8ab4f8; color: #172033; }
  body.dark #anim .play.on { background: #ddd; }
  body.dark #anim .cnt { color: #999; }
  #slices .row { display: flex; align-items: center; gap: 6px; }
  #slices .row span { flex: none; width: 62px; }
  #axes select { width: 58px; }
  #nav {
    position: fixed; top: 10px; right: 10px; z-index: 1; width: 30px; height: 30px; padding: 0;
    display: flex; align-items: center; justify-content: center; cursor: pointer; border: none; border-radius: 6px;
    background: rgba(255,255,255,0.88); color: #888; box-shadow: 0 1px 4px rgba(0,0,0,0.3);
    transition: background 0.2s, color 0.2s, transform 0.2s;
  }
  #nav svg { width: 20px; height: 20px; }
  #nav.on { color: #2a7de1; box-shadow: 0 0 0 2px #2a7de1; }
  #nav.pulse { transform: scale(1.15); background: #2a7de1; color: #fff; }
  body.dark #nav { background: rgba(30,30,30,0.92); color: #999; }
  body.dark #nav.on { color: #8ab4f8; box-shadow: 0 0 0 2px #8ab4f8; }
  body.dark #nav.pulse { background: #8ab4f8; color: #172033; }
  #help {
    display: none; position: fixed; top: 48px; right: 10px; z-index: 2;
    background: rgba(255,255,255,0.95); padding: 10px 14px; border-radius: 6px;
    font-family: sans-serif; font-size: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.3);
  }
  #help table { border-collapse: collapse; }
  #help td { padding: 1px 0; }
  #help td:first-child { padding-right: 10px; color: #333; white-space: nowrap; }
  #help td:last-child { color: #666; }
  body.dark { background: #1a1a1a; color: #ddd; }
  body.dark #controls { background: rgba(30,30,30,0.92); color: #ddd; }
  body.dark #controls .hint, body.dark #controls .val { color: #aaa; }
  body.dark #controls .sec { border-top-color: #555; }
  body.dark #controls input[type=range] { accent-color: #8ab4f8; }
  body.dark #help { background: rgba(30,30,30,0.95); color: #ccc; }
  body.dark #help td:first-child { color: #ddd; }
  body.dark #help td:last-child { color: #aaa; }
</style>
</head>
<body data-theme="light">
<div id="menu">
<button id="menuBtn" title="menu">&#9776;</button>
<div id="controls">
  <div id="counts"></div>
  <div class="sec">
    <label><input type="checkbox" id="cbFaces" checked> faces</label>
    <label>opacity <span class="val" id="opVal"></span>
      <input id="op" type="range" min="0.05" max="1" step="0.01" value="1"></label>
    <label>edges: <select id="edgeMode">
      <option value="hide">hidden removed</option>
      <option value="ghost" selected>hidden as ghost</option>
      <option value="all">all visible</option>
      <option value="none">none</option>
    </select></label>
  </div>
  <div class="sec" id="ptBox">
    <label><input type="checkbox" id="cbPoints" checked> points</label>
    <label id="depthLbl" style="display:none"><input type="checkbox" id="cbDepth"> colored by depth</label>
    <label>radius <span class="val" id="rVal"></span>
      <input id="r" type="range" min="0" max="1" step="0.001" value="0.5"></label>
  </div>
  <div class="sec" id="edlBox" style="display:none">
    <label><input type="checkbox" id="cbEdl" checked> eye-dome lighting</label>
    <label>strength <span class="val" id="edlVal"></span>
      <input id="edlK" type="range" min="0" max="1" step="0.01" value="0.5"></label>
    <label><input type="checkbox" id="cbAo" checked> ambient occlusion</label>
    <label>strength <span class="val" id="aoKVal"></span>
      <input id="aoK" type="range" min="0" max="1" step="0.01" value="0.7"></label>
    <label>radius ( &times; points ) <span class="val" id="aoRVal"></span>
      <input id="aoR" type="range" min="1" max="20" step="0.5" value="5"></label>
  </div>
  <div class="sec" id="axes" style="display:none">
    views on <select id="axX"></select><select id="axY"></select><select id="axZ"></select>
  </div>
  <div id="slices"></div>
  <div class="sec">
    <label><input type="checkbox" id="cbWheel"> wheel moves/zooms the view</label>
    <label><input type="checkbox" id="cbTouch"> touch drives the view</label>
    <a id="openFull" href="#" target="_blank" style="display:none">open in a separate page &#8599;</a>
  </div>
  <div class="hint"><b>?</b>: help · <b>d</b>: <span id="modeLabel">light</span></div>
</div>
</div>
<div id="anim"></div>
<button id="nav" title="navigation: wheel, pinch and touch move / zoom the view">
  <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">
    <path d="M10 2v16M2 10h16M7.5 4.5 10 2l2.5 2.5M7.5 15.5 10 18l2.5-2.5M4.5 7.5 2 10l2.5 2.5M15.5 7.5 18 10l-2.5 2.5"/></svg>
</button>
<div id="help">
  <b>Keyboard shortcuts</b>
  <table>
    <tr><td>&larr; / &rarr;</td><td id="hTime">frame -1 / +1 (a bar of the top row has the focus)</td></tr>
    <tr><td>Home / End</td><td>first / last frame</td></tr>
    <tr><td>Space</td><td>play / pause</td></tr>
    <tr><td>drag</td><td id="hDrag">orbit</td></tr>
    <tr><td>Shift + drag</td><td>move (pan)</td></tr>
    <tr><td>wheel</td><td>move · Ctrl/pinch: zoom towards the cursor</td></tr>
    <tr><td>&larr; &rarr; &uarr; &darr;</td><td id="hArrows">rotate (Shift/Ctrl: faster)</td></tr>
    <tr><td>+ / -</td><td>zoom in / out</td></tr>
    <tr><td>[ / ]</td><td>point radius - / +</td></tr>
    <tr><td>f</td><td>faces</td></tr>
    <tr><td>e</td><td>hidden-edge mode</td></tr>
    <tr><td>p</td><td>points</td></tr>
    <tr><td>c</td><td>points colored by depth (3D)</td></tr>
    <tr><td>o</td><td>ortho / perspective projection</td></tr>
    <tr><td>0 · double-click</td><td>reset the view</td></tr>
    <tr><td>d</td><td>light / dark</td></tr>
    <tr><td>?</td><td>show / hide this help</td></tr>
  </table>
</div>
<canvas id="c"></canvas>
<script>
// ============================================================================================
// data -- everything is encoded in base64 in the page (self-contained file, openable as file://)
// ============================================================================================
function b64bytes(s) {
  const raw = atob(s), b = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i++) b[i] = raw.charCodeAt(i);
  return b;
}
function decF32(s) { const b = b64bytes(s); return new Float32Array(b.buffer, 0, b.length / 4); }
function decI32(s) { const b = b64bytes(s); return new Int32Array(b.buffer, 0, b.length / 4); }
function decU16(s) { const b = b64bytes(s); return new Uint16Array(b.buffer, 0, b.length / 2); }

const D      = __D__;                  // dimension of the WORLD (size of the position vectors)
const BOUNDS = __BOUNDS__;             // [lo, hi] per dimension

// ONE single vertex pool for the whole scene (POS, with stride D); a primitive only refers to it by
// INDICES, and its color is only an index into COLORS. This is what keeps the weight of the
// file down: a cube vertex serves 6 triangles and 3 edges, it is stored only once, and
// a color costs only 2 bytes per primitive instead of 4 per vertex.
const POS    = decF32("__POS__");
const COLORS = __COLORS__, COLORS_DARK = __COLORS_DARK__;
let PAL = COLORS;                      // the table of the CURRENT theme             // table of distinct colors, as [r, g, b, a]

// polygons of FREE length: POLY_OFF slices POLY_V (POLY_OFF[p] .. POLY_OFF[p+1]). They are only
// triangulated here, at display time -- the file itself keeps the faces as they are.
const POLY_V = decI32("__POLY_V__"), POLY_OFF = decI32("__POLY_OFF__"), POLY_C = decU16("__POLY_C__");
const EDG_V  = decI32("__EDG_V__"),  EDG_C    = decU16("__EDG_C__");
const PNT_V  = decI32("__PNT_V__"),  PNT_C    = decU16("__PNT_C__"), PNT_R = decF32("__PNT_R__");

const HPOLY  = __HPOLY__.map(h => ({                                     // polytopes in H-rep
  nb: h.nb, dirs: decF32(h.dirs), offs: decF32(h.offs), fcol: h.fcol, ecol: h.ecol, edg: h.edg,
}));
// FRAMES: each one is only a RANGE in the lists above (they share nothing else than
// the vertex pool) -- FR_* gives the bounds, FR_VAL the abscissae on the axes ( NB_AXES per frame ).
const AXES = __AXES__, PLAYABLE = __PLAYABLE__, FPS = __FPS__, NB_AXES = AXES.length;
const FR_VAL = decF32("__FR_VAL__");
const FR_POLY = decI32("__FR_POLY__"), FR_EDG = decI32("__FR_EDG__");
const FR_PNT = decI32("__FR_PNT__"), FR_HP = decI32("__FR_HP__");
const NB_FRAMES = FR_VAL.length / NB_AXES;

const RMIN = __RMIN__, RMAX = __RMAX__, R0 = __R0__;
const BG = __BG__;                      // [ light, dark ], `null`: the theme's

const NB_V = POS.length / D, NB_POLY = POLY_OFF.length - 1, NB_EDG = EDG_V.length / 2;

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
let TRI = { pos: null, nrm: null, col: null, n: 0 };
let LIN = { a: null, b: null, col: null, n: 0 };
let PNT = { pos: null, col: null, rad: null, n: 0 };

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
  for (let p = p0; p < p1; p++) nTri += Math.max(0, POLY_OFF[p + 1] - POLY_OFF[p] - 2);
  TRI = { pos: new Float32Array(nTri * 9), nrm: new Float32Array(nTri * 9),
          col: new Float32Array(nTri * 12), n: nTri };
  let t = 0;
  function putTri(a, b, c, col) {
    const o = t * 9;
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
  for (let p = p0; p < p1; p++) {
    const o0 = POLY_OFF[p], o1 = POLY_OFF[p + 1], col = PAL[POLY_C[p]];
    for (let k = o0 + 1; k + 1 < o1; k++)                 // fan from the first vertex
      putTri(at(POLY_V[o0]), at(POLY_V[k]), at(POLY_V[k + 1]), col);
  }
  for (let i = 0; i + 8 < hp.length; i += 9)
    putTri(hp.slice(i, i + 3), hp.slice(i + 3, i + 6), hp.slice(i + 6, i + 9), hpc[i / 9]);

  // 4. edges: one INSTANCE per edge (two ends), widened into a quadrilateral in the
  //    shader -- gl.lineWidth is capped at 1 pixel on most drivers.
  const nLin = (e1 - e0) + hl.length / 6;
  LIN = { a: new Float32Array(nLin * 3), b: new Float32Array(nLin * 3),
          col: new Float32Array(nLin * 4), n: nLin };
  let e = 0;
  function putLin(a, b, col) {
    LIN.a[e * 3] = a[0]; LIN.a[e * 3 + 1] = a[1]; LIN.a[e * 3 + 2] = a[2];
    LIN.b[e * 3] = b[0]; LIN.b[e * 3 + 1] = b[1]; LIN.b[e * 3 + 2] = b[2];
    for (let q = 0; q < 4; q++) LIN.col[e * 4 + q] = col[q];
    e++;
  }
  for (let i = e0; i < e1; i++)
    putLin(at(EDG_V[2 * i]), at(EDG_V[2 * i + 1]), PAL[EDG_C[i]]);
  for (let i = 0; i + 5 < hl.length; i += 6)
    putLin(hl.slice(i, i + 3), hl.slice(i + 3, i + 6), hlc[i / 6]);

  // 5. points
  const nPnt = v1 - v0;
  PNT = { pos: new Float32Array(nPnt * 3), col: new Float32Array(nPnt * 4),
          rad: PNT_R.subarray(v0, v1), n: nPnt };
  for (let i = 0; i < nPnt; i++) {
    const v = PNT_V[v0 + i], col = PAL[PNT_C[v0 + i]];
    PNT.pos[i * 3] = PR[3 * v]; PNT.pos[i * 3 + 1] = PR[3 * v + 1]; PNT.pos[i * 3 + 2] = PR[3 * v + 2];
    for (let q = 0; q < 4; q++) PNT.col[i * 4 + q] = col[q];
  }

  upload();
  pivotDirty = true;                                    // the scene changed: nothing aimed at anymore
  centerRays.length = 0;
  document.getElementById("counts").textContent =
    PNT.n + " points · " + LIN.n + " edges · " + TRI.n + " triangles";
  sceneBuilt = true;
}

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
in vec3 aPos; in vec3 aNrm; in vec4 aCol;
uniform mat4 uMVP; uniform mat4 uMV;
out vec4 vCol; out vec3 vN;
void main() { gl_Position = uMVP * vec4(aPos, 1.0); vN = mat3(uMV) * aNrm; vCol = aCol; }`,
`#version 300 es
precision highp float;
in vec4 vCol; in vec3 vN; out vec4 oCol;
uniform float uAlpha;
void main() {
  // frontal lighting, DOUBLE-SIDED (abs): the inside of an open cell stays readable.
  vec3 n = normalize(vN);
  float d = abs(dot(n, normalize(vec3(0.25, 0.45, 1.0))));
  oCol = vec4(vCol.rgb * (0.45 + 0.55 * d), vCol.a * uAlpha);
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
in vec3 aPos; in vec4 aCol; in float aRad;
uniform mat4 uMVP; uniform mat4 uMV; uniform float uSizeK; uniform float uOrtho; uniform float uR; uniform float uRScale;
uniform float uDepthCol; uniform vec2 uDepthRange;
out vec4 vCol; out vec3 vC; out float vR;
// the depth colors: warm in front, cool behind
vec3 depthRamp(float t) {
  const vec3 c0 = vec3(0.85, 0.22, 0.16), c1 = vec3(0.97, 0.70, 0.22), c2 = vec3(0.30, 0.68, 0.62), c3 = vec3(0.20, 0.28, 0.62);
  return t < 1.0 / 3.0 ? mix(c0, c1, 3.0 * t) : t < 2.0 / 3.0 ? mix(c1, c2, 3.0 * t - 1.0) : mix(c2, c3, 3.0 * t - 2.0);
}
void main() {
  gl_Position = uMVP * vec4(aPos, 1.0);
  // a zero radius = a point with no size of its own (a Dirac): the slider sizes it.
  // A point with its own radius keeps it at the slider's start, and the slider scales it.
  float r = aRad > 0.0 ? aRad * uRScale : uR;
  float zv = -(uMV * vec4(aPos, 1.0)).z;
  gl_PointSize = clamp(uSizeK * r / (uOrtho > 0.5 ? 1.0 : max(zv, 1e-4)), 1.0, 1024.0);
  vC = (uMV * vec4(aPos, 1.0)).xyz; vR = r;
  vCol = uDepthCol > 0.5 ? vec4(depthRamp(clamp((zv - uDepthRange.x) / (uDepthRange.y - uDepthRange.x), 0.0, 1.0)), aCol.a) : aCol;
}`,
`#version 300 es
precision highp float;
in vec4 vCol; in vec3 vC; in float vR; out vec4 oCol;
uniform mat4 uP; uniform float uSphere;
const float FLATPTS = __FLATPTS__;
// a light from the upper left, in front ( view space )
const vec3 L = vec3(-0.45, 0.55, 0.70);
void main() {
  vec2 d = gl_PointCoord * 2.0 - 1.0;
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

const buf = {};
["triPos", "triNrm", "triCol", "linA", "linB", "linCol", "linCorner", "pntPos", "pntCol", "pntRad"]
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
gl.bindVertexArray(vaoLine);
attr(pLine, "aCorner", buf.linCorner, 2);
attr(pLine, "aA", buf.linA, 3, 1); attr(pLine, "aB", buf.linB, 3, 1); attr(pLine, "aCol", buf.linCol, 4, 1);
gl.bindVertexArray(vaoPoint);
attr(pPoint, "aPos", buf.pntPos, 3); attr(pPoint, "aCol", buf.pntCol, 4); attr(pPoint, "aRad", buf.pntRad, 1);
gl.bindVertexArray(null);

function up(b, data) { gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW); }
function upload() {
  up(buf.triPos, TRI.pos); up(buf.triNrm, TRI.nrm); up(buf.triCol, TRI.col);
  up(buf.linA, LIN.a); up(buf.linB, LIN.b); up(buf.linCol, LIN.col);
  up(buf.pntPos, PNT.pos); up(buf.pntCol, PNT.col); up(buf.pntRad, PNT.rad);
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
  depth: document.getElementById("cbDepth"),
};
function pointRadius() { return RMIN * Math.pow(RMAX / RMIN, parseFloat(ui.r.value)); }
ui.r.value = Math.log(R0 / RMIN) / Math.log(RMAX / RMIN);      // ( the start: R0, i.e. the given radii as they are )
// every point has its own radius: the slider is a factor on them
const PNT_SIZED = PNT_R.length > 0 && PNT_R.every(r => r > 0);

function draw() {
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
    gl.uniform1f(gl.getUniformLocation(pPoint, "uR"), pointRadius());
    gl.uniform1f(gl.getUniformLocation(pPoint, "uRScale"), pointRadius() / R0);
    gl.uniform1f(gl.getUniformLocation(pPoint, "uDepthCol"), D >= 3 && ui.depth.checked ? 1 : 0);
    gl.uniformMatrix4fv(gl.getUniformLocation(pPoint, "uP"), false, P);
    gl.uniform1f(gl.getUniformLocation(pPoint, "uSphere"), D >= 3 ? 1 : 0);
    gl.uniform2f(gl.getUniformLocation(pPoint, "uDepthRange"), dc - s.r, dc + s.r);   // ( the scene's sphere, along the view )
    gl.depthFunc(gl.LEQUAL);
    gl.drawArrays(gl.POINTS, 0, PNT.n);
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
  canvas.width = Math.round(window.innerWidth * dpr);
  canvas.height = Math.round(window.innerHeight * dpr);
  canvas.style.width = window.innerWidth + "px";
  canvas.style.height = window.innerHeight + "px";
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

// menu: shown on hover, or toggled by a click (touch screens have no hover)
const menu = document.getElementById("menu");
document.getElementById("menuBtn").addEventListener("click", () => menu.classList.toggle("open"));
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
  a.href = location.pathname; a.style.display = "block";
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
    case "o": case "O": cam.ortho = !cam.ortho; break;
    default: return;
  }
  draw();
});

// ============================================================================================
// time / parameter bar
// ============================================================================================
// The frames form a grid over the axes: on each axis, the RANK of a frame among the distinct values. With one axis the
// rank is the frame's own index ( frames in the order given, equal values allowed ). A combination that no frame
// has shows the NEAREST frame, the axis just moved weighing the most -- the bars then snap to what is shown.
const AX_VALS = [], FR_RANK = new Int32Array(NB_FRAMES * NB_AXES);
for (let a = 0; a < NB_AXES; a++) {
  const col = Array.from({ length: NB_FRAMES }, (_, f) => FR_VAL[f * NB_AXES + a]);
  if (NB_AXES === 1) { AX_VALS.push(col); col.forEach((_, f) => { FR_RANK[f] = f; }); continue; }
  const vals = Array.from(new Set(col)).sort((x, y) => x - y);
  AX_VALS.push(vals);
  col.forEach((v, f) => { FR_RANK[f * NB_AXES + a] = vals.indexOf(v); });
}
const FRAME_AT = new Map();
for (let f = NB_FRAMES - 1; f >= 0; f--) FRAME_AT.set(Array.from(FR_RANK.subarray(f * NB_AXES, (f + 1) * NB_AXES)).join(","), f);
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
const animBox = document.getElementById("anim"), axRows = [];
const anyPlay = PLAYABLE.some((p, a) => p && AX_VALS[a].length > 1);
if (!anyPlay) animBox.style.gridTemplateColumns = "minmax(40px, 1fr) auto";
for (let a = 0; a < NB_AXES; a++) {
  if (AX_VALS[a].length < 2) { axRows.push(null); continue; }
  const row = document.createElement("div"); row.className = "row";
  const btn = document.createElement(PLAYABLE[a] ? "button" : "span");
  btn.className = PLAYABLE[a] ? "play" : "gap";
  if (!anyPlay) btn.style.display = "none";
  if (PLAYABLE[a]) { btn.innerHTML = "&#9654;"; btn.title = "play / pause " + AXES[a] + " (Space)"; }
  const inp = document.createElement("input");
  inp.type = "range"; inp.min = 0; inp.max = AX_VALS[a].length - 1; inp.step = 1; inp.title = AXES[a];
  const lab = document.createElement("span"); lab.className = "lab";
  row.append(btn, inp, lab);
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
const shownAxes = axRows.map((r, a) => r ? a : -1).filter(a => a >= 0);
const playableAxes = shownAxes.filter(a => PLAYABLE[a]);
let focusAxis = playableAxes.length ? playableAxes[0] : (shownAxes.length ? shownAxes[0] : -1);
if (shownAxes.length) animBox.style.display = "grid";

let playing = -1, playTimer = null;          // `playing`: the axis being played, -1 when paused
const curRanks = () => Array.from(FR_RANK.subarray(frameIdx * NB_AXES, (frameIdx + 1) * NB_AXES));

function setFrame(i) {
  frameIdx = Math.max(0, Math.min(NB_FRAMES - 1, i));
  for (const a of shownAxes) {
    const r = FR_RANK[frameIdx * NB_AXES + a], row = axRows[a];
    row.inp.value = r;
    row.lab.innerHTML = AXES[a] + " = " + fmtVal(FR_VAL[frameIdx * NB_AXES + a]) +
      '<span class="cnt">' + (r + 1) + "/" + AX_VALS[a].length + "</span>";
  }
  buildScene();                                       // only the displayed frame is rebuilt
  draw();
}
function stopPlaying() {
  if (playing >= 0) { axRows[playing].btn.innerHTML = "&#9654;"; axRows[playing].btn.classList.remove("on"); }
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
  const v0 = FR_PNT[f], v1 = FR_PNT[f + 1], out = new Float32Array((v1 - v0) * 3);
  for (let i = 0; i < v1 - v0; i++) {
    const v = PNT_V[v0 + i];
    out[3 * i] = PR[3 * v]; out[3 * i + 1] = PR[3 * v + 1]; out[3 * i + 2] = PR[3 * v + 2];
  }
  return out;
}
let sub = 0, hold = 0, morphA = null, morphB = null;
function step() {
  if (hold > 0) { hold--; return; }
  const next = nextFrame();
  if (next < 0) { stopPlaying(); return; }
  const n0 = FR_PNT[frameIdx + 1] - FR_PNT[frameIdx], n1 = FR_PNT[next + 1] - FR_PNT[next];
  if (MORPH > 0 && n0 === n1 && n0 > 0) {
    if (sub === 0) { morphA = PNT.pos.slice(); morphB = pointPositions(next); }
    if (++sub < MORPH) {
      const t = 0.5 - 0.5 * Math.cos(Math.PI * sub / MORPH);
      for (let k = 0; k < morphA.length; k++) PNT.pos[k] = morphA[k] + t * (morphB[k] - morphA[k]);
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
  axRows[a].btn.innerHTML = "&#10074;&#10074;"; axRows[a].btn.classList.add("on");
  if (curRanks()[a] >= AX_VALS[a].length - 1 && !LOOP) { const r = curRanks(); r[a] = 0; setFrame(frameFor(r, a)); }
  sub = 0; hold = 0;
  playTimer = setInterval(step, 1000 / FPS);
}
if (playableAxes.length && (URL_OPTS.has("play") ? URL_OPTS.get("play") !== "0" : __AUTOPLAY__)) setTimeout(() => togglePlay(playableAxes[0]), 400);

// ============================================================================================
// panel: opacity, radius, viewed dimensions, slices
// ============================================================================================
function syncLabels() {
  document.getElementById("opVal").textContent = parseFloat(ui.op.value).toFixed(2);
  document.getElementById("rVal").textContent = PNT_SIZED ? "\u00d7 " + (pointRadius() / R0).toPrecision(3) : pointRadius().toPrecision(3);
  document.getElementById("edlVal").textContent = edlStrength().toFixed(0);
  document.getElementById("aoKVal").textContent = parseFloat(ui.aoK.value).toFixed(2);
  document.getElementById("aoRVal").textContent = parseFloat(ui.aoR.value).toFixed(1);
  draw();
}
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
if (D >= 3) document.getElementById("edlBox").style.display = "block";
if (D >= 3) document.getElementById("depthLbl").style.display = "block";
if (PNT_V.length === 0) document.getElementById("ptBox").style.display = "none";

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
  box.className = "sec";
  const t = document.createElement("div");
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
    row.appendChild(lab); row.appendChild(inp);
    box.appendChild(row);
  }
}

if (FLAT) {
  document.getElementById("hDrag").textContent = "move (pan)";
  document.getElementById("hArrows").textContent = "move (Shift/Ctrl: faster)";
}

// dark mode: `?theme=dark|light`, else the theme of the page that embeds this one ( its `dark` class, followed live: a
// VitePress page, same origin ), else the system preference; then toggled with [d]. Colors given per theme switch with it.
let sceneBuilt = false;
(function () {
  const ml = window.matchMedia("(prefers-color-scheme: dark)");
  const asked = URL_OPTS.get("theme");
  let host = null;
  try { if (window.parent !== window) host = window.parent.document.documentElement; } catch (e) { host = null; }
  let cur = asked ? asked === "dark" : host ? host.classList.contains("dark") : ml.matches;
  function setTheme(on) {
    cur = on;
    document.body.classList.toggle("dark", on);
    document.body.dataset.theme = on ? "dark" : "light";
    document.getElementById("modeLabel").textContent = on ? "dark" : "light";
    const pal = on ? COLORS_DARK : COLORS;
    if (pal !== PAL) { PAL = pal; if (sceneBuilt) buildScene(); }
    draw();
  }
  setTheme(cur);
  if (!asked && host)
    new MutationObserver(() => setTheme(host.classList.contains("dark"))).observe(host, { attributes: true, attributeFilter: ["class"] });
  else if (!asked)
    ml.addEventListener("change", e => setTheme(e.matches));
  document.addEventListener("keydown", e => {
    if ((e.key === "d" || e.key === "D") && e.target.tagName !== "SELECT" && e.target.tagName !== "INPUT")
      { e.preventDefault(); setTheme(!cur); }
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
setFrame(NB_FRAMES - 1);          // the final state first -- it is what we want to see first
syncLabels();
resize();                         // BEFORE framing: the latter depends on the window's aspect ratio
resetView();
// The bar of an axis takes the focus: the arrows drive it NATIVELY (and Home/End too),
// without taking anything away from the camera -- a click on the view gives the arrows back to the orbit.
if (focusAxis >= 0) axRows[focusAxis].inp.focus({ preventScroll: true });
</script>
</body>
</html>
"""
