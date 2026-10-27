<script setup>
import { withBase } from 'vitepress'
</script>

# The Visualizer

Plans, power diagrams, cells, cell complexes and some distributions can draw themselves. `sdot.Visualizer` collects what they draw, with your own points, edges, faces and polytopes, in one **scene**. A scene goes out as a **self-contained HTML page** (WebGL, no server, no network: it opens as `file://`, in a notebook or in an iframe of a web page), or as **ParaView** files.

This page goes through the usual cases. The figures are written by [`visualizer_figures.py`](https://github.com/sdot-team/sdot/blob/main/docs/guide/visualizer_figures.py), the way the page tells it. The full list of arguments is in the [API reference](../api/visualization).

## Looking at one object

Every object that draws itself has the same four methods:

```python
p = sdot.plan( f, g )

p.show()                       # inline in a notebook, otherwise a temporary page in the browser
p.write_html( "plan.html" )    # a self-contained page
p.write_pvd( "plan.pvd" )      # ParaView: a .pvd and its .vtu
viz = p.to_viz()               # the Visualizer itself, to add more to it
```

Their keyword arguments go to the object's `add_to_viz`, which says what it can draw. For a plan: `transport = True` adds a segment from each seed to the barycenter of its cell, `seeds = False` hides the seeds, `opacity` sets that of the cells, `weights` draws the diagram at other weights than the fitted ones:

```python
p.write_html( "plan.html", title = "A plan and its transport", transport = True, opacity = 0.5 )
```

<iframe
  :src="withBase('/guide/visualizer-plan.html') + '?wheel=0&touch=0'"
  title="A plan and its transport"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/visualizer-plan.html')" target="_blank">Open in a separate page ↗</a></p>

What each object accepts:

| object | its options |
|---|---|
| `SdotPlanNd` (`sdot.plan`) | `transport`, `seeds`, `weights`, `target`, and those of `PowerDiagram` |
| `PowerDiagram` | `seeds`, `seed_color`, `support_only`, `domain`, the copies of a periodicity (`nb_periodicity_rings`, ...), and those of `Cell` |
| `Cell` (`pd.cells`) | `color`, `opacity`, `faces`, `edges`, `points`, `edge_opacity` |
| `Faces` of a `CellComplex` | `ids` (which faces), `color`, `edge_color`, `faces`, `edges` |
| `SumOfDiracs`, `SumOfGaussians` | `color` (their positions, their centers) |

## Several objects in one scene

To compare, highlight or annotate, build the `Visualizer` and add things to it one by one. `viz.add( obj, ... )` takes the same options as above, and the `add_*` methods take your own geometry in world coordinates:

| | |
|---|---|
| `add_points( x, radius = 0 )` | points `[ n, d ]`; a radius `0` is a Dirac, its size set by the slider of the page |
| `add_edges( x, edges )` | segments `[ m, 2 ]` of indices into `x`, or `None` for a polyline (`closed` for a loop); `dashed` |
| `add_faces( x, faces )` | polygons, as lists of indices of any length |
| `add_polytope( dirs, offs )` | the polytope $\{ x : \mathrm{dirs}_i \cdot x \le \mathrm{offs}_i \}$, bounded or not, in any dimension |

Below, the cells of a plan are faded, one cell and its neighbors (from the [cell complex](./cell-complex)) are highlighted, and the 8 seeds that are the farthest from the barycenter of their cell are drawn in red, with a dashed segment to it:

```python
viz = sdot.Visualizer( title = "One scene, several objects" )
viz.add( p, opacity = 0.15, seeds = False )                     # the cells, faded
cx = p.power_diagram().cell_complex()
cx.cells.add_to_viz( viz, ids = neighbors, color = ( "#c8c8c8", "#606060" ) )  # the neighbors of cell i, one color per theme
cx.cells.add_to_viz( viz, ids = [ i ], color = "#4e79a7" )      # cell i
viz.add_points( x[ ~far ], color = "#000000" )
viz.add_points( x[ far ], color = "#d62728" )
viz.add_edges( np.concatenate( [ x[ far ], bar[ far ] ] ), segments, color = "#d62728", dashed = True )
viz.write_html( "scene.html" )
```

<iframe
  :src="withBase('/guide/visualizer-scene.html') + '?wheel=0&touch=0'"
  title="One scene, several objects"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/visualizer-scene.html')" target="_blank">Open in a separate page ↗</a></p>

The **dimension** is not declared: it is read from the positions, and everything in a scene must agree on it.

**Colors** are `"#rrggbb"` strings or `( r, g, b )` tuples in $[ 0, 1 ]$. A pair `( light, dark )` gives one color per theme: the page follows the theme of the site that embeds it, or of the system, and `d` switches it. Without a color, a primitive takes the next one on the scale of the scene. The cells of a diagram take the color of the rank of their seed: cell $i$ has `viz.color_at( i )` whatever was drawn before it, so the same seed keeps the same color in a comparison or an animation. For one object that draws several primitives with the same color, take it once with `viz.take_color()`.

`background` sets the background color (also per theme), `face_opacity` the starting value of the opacity slider of the faces.

## Replaying a descent

`viz.new_frame( value )` opens a **frame**: what is added afterwards belongs to it. Each frame is a complete state, independent of the others, so cells can appear, vanish or change shape from one to the next. The page gets a bar with a play button, next to the menu; the arrows of the keyboard step through it and `Space` plays.

To replay an algorithm, record its states in a callback, then draw one frame per state. Here, the power diagrams of the Lloyd iterations of [`sdot.quantization`](../examples/lloyd-quantization):

```python
frames = []
sdot.quantization( target, 200, callback = lambda it, pos, plan: frames.append( plan.power_diagram() ) )

viz = sdot.Visualizer( title = "Lloyd, iteration by iteration", frame_axis = "iteration", fps = 4 )
for it, pd in enumerate( frames ):
    if it:
        viz.new_frame( it )    # the first frame exists from the start
    viz.add( pd, opacity = 0.35 )
viz.write_html( "descent.html" )
```

<iframe
  :src="withBase('/guide/visualizer-descent.html') + '?wheel=0&touch=0'"
  title="Lloyd, iteration by iteration"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/visualizer-descent.html')" target="_blank">Open in a separate page ↗</a></p>

Some options for animations:

- `frame_axis` names the axis ("time" by default), and `value` is the abscissa of the frame on it: an iteration, an instant, a parameter. `fps` sets the speed, `loop` starts again at the end, `autoplay` plays as soon as the page opens.
- `morph = k` goes from one frame to the next in `k` eased steps, matching the points of two frames by rank when they have as many. The Diracs of a transport or of a descent then glide instead of jumping, without storing the steps.
- Points are shared between frames only when the **same array** is passed again. A page with many frames weighs what its frames hold: keep one frame out of several for a long descent (`np.geomspace` spacing suits Lloyd, which moves a lot at first).
- Colors are numbered again from zero at each frame, so what is drawn at the same rank in each frame keeps its color.

## A sweep over several parameters

With a tuple for `frame_axis`, the frames form a grid over several parameters, and each axis gets its own row in the bar. `new_frame` takes one value per axis, or a dict that names only the axes that change. `playable` says which axes have a play button: a sweep over a parameter is read by moving its slider, a time by playing it.

```python
viz = sdot.Visualizer( title = "Two parameters", frame_axis = ( "sigma", "center" ), playable = ( False, True ) )
for s in ( 0.08, 0.12, 0.18, 0.25 ):
    for c in np.linspace( 0.3, 0.7, 5 ):
        viz.new_frame( ( s, c ) )        # ( except for the first one )
        g = sdot.SumOfGaussians( np.array( [ [ c, 0.5 ] ] ), sigmas = np.array( [ s ] ) )
        viz.add( sdot.plan( sdot.SumOfDiracs( x ), g ), opacity = 0.4 )
```

<iframe
  :src="withBase('/guide/visualizer-sweep.html') + '?wheel=0&touch=0'"
  title="Two parameters"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/visualizer-sweep.html')" target="_blank">Open in a separate page ↗</a></p>

A combination that is missing from the grid shows the nearest frame.

## Point clouds and cells in 3D

In 3D the view turns freely around what is at the center of the screen, and zooms on the point under the cursor. A few things help to read a scene in depth:

- `depth_colors = True` colors the points by their depth in the view (warm in front, cool behind), recomputed as the view turns. It is what a cloud of Diracs needs: without it, it is a flat blot. `c` switches it.
- **Hidden edges**: `e` cycles through hidden edges removed, drawn as a ghost, all visible, or none. It works even when the faces are not displayed, so a wireframe (`faces = False`) still reads as a solid.
- `o` switches between the perspective and the orthographic views, `f` and `p` show or hide the faces and the points, `[` and `]` change the radius of the points.

Below, the Diracs of a Lloyd quantization of three gaussians, colored by depth, and the cells of the six nearest to their center, as wireframes:

```python
viz = sdot.Visualizer( title = "A cloud in 3D", depth_colors = True )
viz.add_points( x )
cx = p.power_diagram().cell_complex()
cx.cells.add_to_viz( viz, ids = near, faces = False, edge_color = ( "#000000", "#ffffff" ) )
```

<iframe
  :src="withBase('/guide/visualizer-cloud-3d.html') + '?wheel=0&touch=0'"
  title="A cloud in 3D"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/visualizer-cloud-3d.html')" target="_blank">Open in a separate page ↗</a></p>

In 2D, `flat_points = True` draws the points as flat discs rather than shaded spheres: what a particle of the plane is.

## Beyond 3D: slices

In dimension 4 and more, the menu chooses the 3 displayed coordinates and fixes the others with sliders. Polytopes are truly **cut** by this affine 3-space: the slice of a cell is again a polytope, whose vertices are enumerated in the browser at each move of a slider. Points and meshes are projected.

The cells of a diagram are sent as polytopes, so nothing changes on the Python side:

```python
pd = sdot.PowerDiagram( rng.random( ( 16, 4 ) ), boundaries = sdot.box_half_spaces( [ 0 ] * 4, [ 1 ] * 4 ) )
pd.write_html( "slice-4d.html", opacity = 0.3 )
```

<iframe
  :src="withBase('/guide/visualizer-slice-4d.html') + '?wheel=0&touch=0'"
  title="A slice of a 4D diagram"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/visualizer-slice-4d.html')" target="_blank">Open in a separate page ↗</a></p>

`add_polytope( dirs, offs )` is also the way to draw an **unbounded** polytope, in any dimension: the page clips it with the box of the scene. That box is computed from the vertices drawn; `viz.note_bounds( x )` extends it without drawing anything.

## Convergence curves

`sdot.write_convergence_html` writes a page of curves (log scale by default), one per series, with an implicit abscissa or explicit `( x, y )` pairs:

```python
history = []
sdot.quantization( target, 300, tol = 3e-3, callback = lambda it, pos, plan: history.append( np.asarray( pos ) ) )
moves = [ np.sqrt( ( ( b - a ) ** 2 ).sum( axis = 1 ).mean() ) for a, b in zip( history, history[ 1: ] ) ]
sdot.write_convergence_html( { "300 diracs": moves }, "convergence.html", ylabel = "root mean square move" )
```

<iframe
  :src="withBase('/guide/visualizer-convergence.html')"
  title="Lloyd: how much the diracs move"
  loading="lazy"
  style="width: 100%; height: 50vh; min-height: 320px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/visualizer-convergence.html')" target="_blank">Open in a separate page ↗</a></p>

## In a notebook, a web page, ParaView

**Notebooks** (Jupyter, VS Code, Colab): `obj.show( height = 500 )` displays the page inline. The wheel and the touch gestures are then left to the notebook, so that it keeps scrolling; the **navigation switch** at the top right of the view (the four-arrow icon) gives them back to the view.

**Web pages**: the HTML file can be served as it is, and embedded in an iframe, as in this documentation. Options in the URL take precedence over those of the file:

| | |
|---|---|
| `?wheel=0`, `?touch=0` | the wheel and the touch gestures scroll the host page (switch them back on with the navigation switch) |
| `?theme=light`, `?theme=dark` | the theme; by default that of the host page (same origin), or of the system |
| `?play=0`, `?play=1` | play on opening, or not |
| `?depth=0`, `?depth=1` | colors by depth |
| `?opacity=0.4` | the opacity of the faces |

```html
<iframe src="plan.html?wheel=0&touch=0" style="width: 100%; height: 60vh; border: 0"></iframe>
```

`viz.to_html( wheel = False, touch = False )` gives the page as a string, to embed it some other way.

**ParaView**: `write_vtk( "scene.vtu" )` writes one `.vtu` (compressed binary XML), or one per frame and a `.pvd` that gathers them with their abscissa: open the `.pvd`. `axes` chooses the 3 coordinates written as geometry; in dimension 4 and more, the others go out as point data, for ParaView to slice on.

## The keyboard

`?` in the page shows the help.

| key | |
|---|---|
| arrows | turn the view (3D); with a frame bar focused, step through the frames |
| `+`, `-`, `0` | zoom in, zoom out, reset the view |
| `Space` | play or pause |
| `[`, `]` | radius of the points |
| `f`, `p` | show or hide the faces, the points |
| `e` | the hidden edges: removed, ghost, visible, none |
| `c` | colors by depth |
| `o` | perspective or orthographic |
| `d` | light or dark theme |
