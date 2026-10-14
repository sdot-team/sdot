<!-- written by scripts/api_docs.py from the docstrings: do not edit -->

# API — Power Diagrams

The geometry under the plans: power diagrams, their cells, and what speeds them up.

::: v-pre

## `PowerDiagram`

```python
PowerDiagram(positions=None, weights=None, *args, accelerator=None, **kwargs)
```

Power diagram ( Laguerre ) -- THE VIEW, and what all the storages have in common.

The cell of seed `i` is where its POWER DISTANCE wins:

    |x - d_i|² - w_i  <=  |x - d_j|² - w_j   for every other j

Expanded, the inequality loses its `|x|²` on both sides and becomes a half-space: a power
diagram costs exactly what a Voronoi diagram costs, one plane per rival and the same cut. Only the
DIFFERENCES of weights reach the planes: "all equal" and "no weights at all" are the same
object, and the Euclidean case is called `Voronoi` ( see `Voronoi.py` ).

`PowerDiagram( positions, weights, ... )` does not carry a diagram: it carries its SEEDS and the
convex domain that bounds them, and rebuilds what it is asked for, cell by cell, in the
scratch of a work-item ( `diagram/Ops.h` ). This file is the CONTRACT -- what a user reads and
writes: `positions`, `weights`, `measures`, `cells`, `cell( i )` -- and the common trunk: the domain,
the distribution, the scratch, the three kernels. HOW the seeds are laid out is up to a
specialization, chosen at construction:

  * `PowerDiagram_Plain` -- the seeds as they came, each cell cut by the `n - 1`
    bisectors. The floor, and what remains when the positions are a tracer;
  * `PowerDiagram_Bsp`   -- the seeds in the order of a BSP tree ( `AaBsp` ), a leaf being
    read in one piece, and each cell cut only by the seeds the tree could not
    rule out. The default as soon as the positions are concrete.

The neighborhood is ACCELERABLE, not the result: an accelerator can only silence cuts that
would have removed nothing, so the cells are the SAME, up to rounding errors.

| Attribute | |
|---|---|
| `cells` | All the cells, as one `Cell` batched over the seeds ( in the order of `positions` ) -- to draw or inspect them. |
| `measures` | The measure of each cell, `[ n ]` -- its integral against `distribution` when there is one. Differentiable with respect to the positions, the weights and the parameters of the distribution. |
| `moments` | `( masses, first, second )`: for each cell, `int rho`, `int x rho` ( `[ n, d ]` ) and `int \|x\|^2 rho`. Not differentiable ( see `moment` ). |

### `.add_to_viz(viz, seeds=True, seed_color='#000000', support_only=True, threshold=0.0, **kwargs)`

Draws itself into a `Visualizer`: all the cells, in one call ( see `cells` ), and the
seeds as points when `seeds` ( `seed_color` ). `kwargs` are those of `Cell.add_to_viz`
( `opacity`, `faces`, `edges`, ... ). The quick ways to look at it -- `write_html`,
`write_pvd`, `show` -- are those of `Displayable`.

`support_only`: the cells only where the distribution has mass ( `support_pieces`, with
`threshold` ) -- an image with zero pixels does not show its cells there.

### `.hessian_rows()`

The derivatives of the measures with respect to the weights, row by row: `( nb_nbrs, ids, vals )`, the neighbors of each cell and the coefficients ( host arrays ).

### `.laplacian_solve(rhs, linear_solver='auto')`

The solution `x` of `L x = rhs`, `L` being the derivative of the measures with respect to the weights ( `rhs` sums to zero; `x[ 0 ] == 0` ). Differentiable with respect to `rhs`.

### `.moment(k)`

`int_{cell_i} p( x ) rho( x ) dx`, `[ n ]`, for `p = x_k` ( `k < d` ), `|x|^2` ( `k == d` ) or `1` ( `k == d + 1` ). Differentiable like `measures`.

### `.show(filename=None, title=None, height=500, **kwargs)`

In a notebook ( Jupyter, VS Code, Colab, ... ): displays the page inline, in an iframe
`height` pixels high -- the wheel and touch gestures are then left to the notebook, they
can be switched back on from the menu. Elsewhere, or when `filename` is given: writes the
HTML page ( `filename`, or a temporary file ) and opens it in the browser. Returns the
path written ( None when nothing was written ).

### `.to_viz(viz=None, title=None, **kwargs)`

A `Visualizer` holding this object ( `viz`, when given: it is then completed ). `title`
names a NEW scene; `kwargs` are those of `add_to_viz`.

### `.write_html(filename, title=None, **kwargs)`

Writes the self-contained HTML page of this object. Returns the path written.

### `.write_pvd(filename, title=None, axes=(0, 1, 2), **kwargs)`

Writes this object for ParaView as a `.pvd` collection ( with its `.vtu` beside it ),
the entry point one opens. Returns the `.pvd` path.

### `.write_vtk(filename, title=None, axes=(0, 1, 2), **kwargs)`

Writes this object for ParaView, as a `.vtu`. Returns the path written.

## `Voronoi`

```python
Voronoi(positions, **kwargs)
```

The Voronoi diagram of `positions`: a `PowerDiagram` without weights ( same arguments, `weights` excepted ).

## `Cell`

```python
Cell(nb_dims=None, *args, **kwargs)
```

A convex polytope, possibly unbounded, possibly batched ( `PowerDiagram.cells` is a batch of cells ). Start from
`Cell.make_hypercube( d, origin, axes )` or `Cell.make_unbounded( d )`, then `cut` it by half-spaces.

| Attribute | |
|---|---|
| `cut_planes` | `( directions[ nb_cuts, d ], offsets[ nb_cuts ] )`, re-read off the geometry -- the UNIT outward normal of each face and its offset. A cut without a vertex has a zero direction. |
| `edges` | `[ nb_edges, 2 ]` : the edges, as vertex indices. |
| `faces` | A list of cycles of vertex indices, one face per cut that carries one ( 2D and 3D ). |
| `is_bounded` | Whether the cell is bounded. |
| `measure` | The measure of the cell: length, area, volume... Infinite ( the largest float ) for an unbounded cell. Differentiable with respect to the vertices. |

### `.add_to_viz(viz, color=None, opacity=1.0, faces=True, edges=True, points=False, owners=None, nb_colors=None)`

Draws itself into a `Visualizer` ( see `sdot.viz.Visualizer`, and `cell_viz` for what an
unbounded cell drops ). Each item takes its color from its RANK IN THE BATCH,
on a block reserved in advance : the color of a cell of a diagram says WHICH seed.
`owners`: the rank to take the color from, per item, on a block of `nb_colors` -- the
pieces of one cell ( `PowerDiagram.support_pieces` ) have its color.

### `.cut(direction, offset, cut_id=-1)`

Intersects the cell with the half-space `direction . x <= offset`, IN PLACE.

`direction` need not be normalized : `offset` is the dot product it is
compared against as is. Since the inputs and outputs of a `loom.ffi_call` are disjoint, the
kernel writes into a NEW cell and the in-place update is only a rebinding. The
room the cut requires is BOUNDED in advance ( `_cut_capacities` ) : no second pass.

### `.make_hypercube(nb_dims, origin=None, axes=None, cut_id=-1, batch_axes=None, **kwargs)`

A parallelotope: `origin + sum_j t_j axes[ j ]`, `t` in `[ 0, 1 ]^d` ( the unit cube by default ).

### `.make_unbounded(nb_dims, batch_axes=None, **kwargs)`

The whole space, in dimension `nb_dims`, to cut from.

### `.show(filename=None, title=None, height=500, **kwargs)`

In a notebook ( Jupyter, VS Code, Colab, ... ): displays the page inline, in an iframe
`height` pixels high -- the wheel and touch gestures are then left to the notebook, they
can be switched back on from the menu. Elsewhere, or when `filename` is given: writes the
HTML page ( `filename`, or a temporary file ) and opens it in the browser. Returns the
path written ( None when nothing was written ).

### `.to_viz(viz=None, title=None, **kwargs)`

A `Visualizer` holding this object ( `viz`, when given: it is then completed ). `title`
names a NEW scene; `kwargs` are those of `add_to_viz`.

### `.vertices(item=0)`

`vertex_positions` of item `item`, in numpy `[ nb_vertices, d ]` -- without the padding
of a batch.

### `.write_html(filename, title=None, **kwargs)`

Writes the self-contained HTML page of this object. Returns the path written.

### `.write_pvd(filename, title=None, axes=(0, 1, 2), **kwargs)`

Writes this object for ParaView as a `.pvd` collection ( with its `.vtu` beside it ),
the entry point one opens. Returns the `.pvd` path.

### `.write_vtk(filename, title=None, axes=(0, 1, 2), **kwargs)`

Writes this object for ParaView, as a `.vtu`. Returns the path written.

## `box_half_spaces`

```python
box_half_spaces(mi, ma)
```

The box `mi <= x <= ma` as `2d` half-spaces `direction . x <= offset`.

## `AaBsp`

```python
AaBsp(positions, weights=None, max_seeds_per_leaf=10)
```

A tree of boxes over the seeds of a power diagram, which spares each cell the seeds that cannot cut it. The diagrams
build one by themselves ( the default as soon as the positions are not traced ); `AaBsp.of( power_diagram )` builds
one explicitly, to give as `accelerator`.

### `.of(power_diagram, max_seeds_per_leaf=10)`

The accelerator of the seeds of `power_diagram` -- its positions AND its weights.

The shortcut one almost always wants: a BSP built on other weights than those
of the diagram would remain CORRECT (the majorant would only prune less well) but
would have no reason to be good.

## `SpatialAccelerator`

```python
SpatialAccelerator(**kwargs)
```

The base of the accelerators of power diagrams ( `AaBsp` ): what proposes, for each cell, the seeds worth trying.

## `set_kernel_dtype`

```python
set_kernel_dtype(dtype)
```

The kernel float for cells ( and diagrams ) built FROM NOW ON without an explicit
`kernel_dtype` : `"FP32"` ( the default ) or `"FP64"`. Returns the previous setting.

## `kernel_dtype`

```python
kernel_dtype()
```

The kernel float of the cells ( and diagrams ) built without saying one: `"FP64"` or `"FP32"` ( see
`set_kernel_dtype` )

:::
