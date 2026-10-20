<!-- written by scripts/api_docs.py from the docstrings: do not edit -->

# API — Visualization

Pages to look at what was computed ( `show`, `write_html` of the objects use them ).

::: v-pre

## `Visualizer`

```python
Visualizer(title='sdot', background=None, frame_axis='time', playable=True, fps=5.0, frame_value=0.0, depth_colors=False, morph=0, loop=False, autoplay=False, face_opacity=1.0)
```

Collects geometric primitives; the outputs (`write_html`, ...) come afterwards.

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

`face_opacity`: the starting value of the opacity slider of the faces ( `?opacity=` in the URL wins ) -- below 1 for a
density drawn under diracs.

| Attribute | |
|---|---|
| `colors` | The table of distinct colors, as `(r, g, b, a)` -- the primitives refer to it. |
| `edge_colors` | The color index of each edge, in `colors`. |
| `edges` | The edges, `[m, 2]` of indices. |
| `point_colors` | The color index of each point, in `colors`. |
| `point_radii` | The world radius of each point (0 = no size of its own, cf. `add_points`). |
| `points` | The points, `[k]` of indices. |
| `polygon_colors` | The color index of each polygon, in `colors`. |
| `polygons` | The faces, as a list of lists of indices (free lengths, not triangulated). |
| `polytopes` | The polytopes in H-representation, `[ ( dirs [c, d], offs [c], rgba, edges ), ... ]`. |
| `positions` | The vertex pool of the whole scene, `[nb_verts, d]` -- the primitives' indices refer to it. |

### `.add(obj, **kwargs)`

Adds an object that knows how to draw itself (it exposes `add_to_viz( viz, ... )`).

The visualizer knows no domain type: it is the object that translates its geometry into
primitives (`Cell.add_to_viz` -> `add_faces` / `add_edges` / `add_polytope`).

### `.add_edges(positions, edges=None, color=None, opacity=1.0, closed=False, dashed=False, nb_dashes=7, frames=True)`

`positions`: `[n, d]`. `edges`: `[m, 2]` of indices, or `None` to connect
consecutive points (`closed` then closes the loop -- the case of a polygon given in
cyclic order, which is what a 2D cell yields).

`dashed` draws DOTTED lines. The dash is not a style carried all the way to the output: each
segment is simply SPLIT here, into `nb_dashes` pieces separated by as many gaps, and
what goes out afterwards is a sequence of ordinary edges. Two reasons: the rendering (hand-written
WebGL) has no line pattern, and neither does VTK -- split geometry is the
only thing BOTH outputs know how to display identically. The price is that the length of a
dash is in WORLD units, not in pixels: it is a fixed number of dashes per edge, so
the density does not change with zoom.

### `.add_faces(positions, faces, color=None, opacity=1.0, frames=True)`

`positions`: `[n, d]`. `faces`: a sequence of POLYGONS (lists of indices, free lengths).
They are kept AS IS -- it is the output that decides whether to triangulate them.

### `.add_points(positions, radius=0.0, color=None, opacity=1.0, frames=True)`

`positions`: `[n, d]`. `radius`: the WORLD radius of the points (a scalar, or one radius
per point). `0` (default) = the point has no size of its own (a Dirac): it is the page's
slider that sets its display radius -- same convention as `points_html`.

### `.add_polytope(cut_directions, cut_offsets, color=None, opacity=1.0, edge_color=None, nb_dims=None, edges=True)`

Adds a polytope through its H-representation: `{ x : dir_i . x <= off_i }`.

This is the form to give when the V-representation is NOT enough: in dimension > 3 (the
page shows a 3D SLICE of it, and cutting half-spaces yields half-spaces again, hence
a polytope, re-enumerated in the browser at each slider movement), or for an
UNBOUNDED polytope (no vertex to enumerate on the Python side -- the page clips it with a
bounding box and shows what remains).

`edges = False` keeps only the FACES. For a caller who draws the edges itself and
knows more than the enumeration: that of an unbounded polytope stops at the scene's
box, so it would render an edge going off to infinity as solid all the way to the border -- where
`Cell.add_to_viz`, for its part, knows it is truncated and makes it dotted.

### `.bounds(pad_ratio=1.0)`

Bounding box `[ [lo, hi] ] * d`, dimension by dimension.

It frames the camera AND gives the cut sliders their range. A polytope has no
vertices to measure: it is bounded by the distance of its planes to the origin, which is
rough but of the right order of magnitude (and moot as soon as a pool is there as well).

### `.color_at(index)`

The color number `index` of the scale -- see `scale_color`.

This is the entry point to prefer whenever an object has its OWN NUMBER: the cell of
Dirac `i` is colored `color_at( i )`, and its color then depends only on `i` -- not on the
order of calls, nor on what was drawn before, nor on the frame. A Dirac keeps its color
from one descent step to the next, and a neighbor that loses its cell no longer shifts
anyone.

### `.darker(color, factor=0.72)`

The same hue, darkened -- enough to derive an edge color from a face color.

### `.frame(i)`

The primitives of frame `i`, already sliced -- enough to write an output without
knowing anything about the internal slicing.

Returns a dict: `value` ( a float, a tuple with several frame axes ), `polygons`, `polygon_colors`, `edges`, `edge_colors`, `points`,
`point_colors`, `point_radii`, `polytopes`. The indices refer to the `positions` pool,
common to ALL frames.

### `.new_frame(value=None)`

Opens a new FRAME: everything added afterwards will be stored in it.

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

### `.next_color()`

The next free color (the one an `add_*` without `color` would take), without advancing.

### `.note_bounds(positions)`

Extends the scene's box to `positions`, WITHOUT drawing anything.

For what is displayed otherwise than by vertices: a polytope given as half-spaces
has no readable extent, whereas the object sending it often knows it (a
cell keeps its vertices) -- and it is this box that frames the camera, gives the cut sliders
their range and sets the clipping of unbounded polytopes.

### `.reserve_colors(nb=1)`

Reserves `nb` CONSECUTIVE indices on the scale and returns the first.

A block rather than one by one: what must number the colors of a composite object is
the rank of its parts WITHIN the object (item `b` of a batched cell takes `base + b`),
not the number of parts actually drawn -- otherwise an empty cell, which is skipped,
would shift all the following ones.

Numbering restarts from zero at each frame (`new_frame`), so that the same object
drawn at the same rank in each frame keeps the same color there.

### `.take_color()`

Reserves ONE index and returns its color -- for an object that has no number of its own.

To be taken when a single object yields SEVERAL primitives (a cell: its faces, its
edges, its vertices) and they should have the same color: without this, each `add_*`
without `color` would consume a new one and the cell would be motley.

### `.to_html(wheel=True, touch=True)`

The self-contained HTML page, as a string. `wheel` / `touch` say whether the wheel and
the touch gestures drive the view by default (`?wheel=0` / `?touch=0` in the URL still
win) -- off when the page is embedded in a host page that must keep scrolling.

### `.write_html(filename)`

Writes a self-contained HTML page (see the module header). Returns the path written.

### `.write_vtk(filename, axes=(0, 1, 2), pvd=False)`

Writes the scene for ParaView (compressed binary XML). Returns the path written.

A single frame -> one `.vtu`. Several -> one `.vtu` per frame plus a `.pvd` that
gathers them with their abscissa on the axis (the `.pvd` is what one opens, and it is what
is returned). `pvd = True` asks for the `.pvd` even for a single frame. `axes` chooses the 3 dimensions written as GEOMETRY; beyond 3, the other
coordinates go out as point data, so that ParaView does its slices itself.

## `write_convergence_html`

```python
write_convergence_html(series, out_path, title='convergence', xlabel='iteration', ylabel='residual', log_y=True)
```

`series`: `{ name: [ y0, y1, ... ] }` (an implicit abscissa `0, 1, 2, ...`), or
`{ name: [ ( x0, y0 ), ( x1, y1 ), ... ] }` for an explicit abscissa (non-consecutive
steps, for instance). Returns the path written.

`log_y`: log scale on the y axis (the default -- a convergence curve says nothing on a
linear scale, it is squashed against the axis from the first two steps on). Values `<= 0`
then have no point: `0` IS the goal, but cannot be placed on a log axis -- only its
approach can be read, in the slope.

:::
