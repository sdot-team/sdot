<!-- written by scripts/api_docs.py from the docstrings: do not edit -->

# API — Power Diagrams

The geometry under the plans: power diagrams, their cells, and what speeds them up. For seeds at the same position, see [Guide: Duplicates and Close Seeds](/guide/duplicates).

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
| `cells` | All the cells, as one `Cell` batched over the seeds ( in the order of `positions` ) -- to draw or inspect them. With a periodicity, each cell is the whole cell of its seed, around the seed as it was given: it can lie outside the domain, and away from the others ( `add_to_viz` shows its copies by the group too ). |
| `groups` | The `Groups` of this diagram ( `groups.py` ), in the user's order -- `None` without groups. |
| `measures` | The measure of each cell, `[ n ]` -- its integral against `distribution` when there is one. Differentiable with respect to the positions, the weights and the parameters of the distribution. |
| `moments` | `( masses, first, second )`: for each cell, `int rho`, `int x rho` ( `[ n, d ]` ) and `int \|x\|^2 rho`. Not differentiable ( see `moment` ). |

### `.add_to_viz(viz, seeds=True, seed_color='#000000', support_only=None, threshold=0.0, nb_periodicity_rings=1, opacity_periodicity_rings=0.25, fit_periodicity_rings=False, domain=None, domain_color='#000000', opacity_beyond_balls=0.2, virtual_seeds=False, virtual_color='#888888', **kwargs)`

Draws itself into a `Visualizer`: all the cells, in one call ( see `cells` ), and the
seeds as points when `seeds` ( `seed_color` ). `kwargs` are those of `Cell.add_to_viz`
( `opacity`, `faces`, `edges`, ... ). The quick ways to look at it -- `write_html`,
`write_pvd`, `show` -- are those of `Displayable`.

`support_only`: the cells only where the distribution has mass ( `support_pieces`, with
`threshold` ) -- an image with zero pixels does not show its cells there. By default: yes without a periodicity, no with
one ( a periodic cell is drawn whole, around its seed ).

With a periodicity, the cells are drawn again moved by every element of the group up to the ring `nb_periodicity_rings`
( `periodicity_rings` ), fainter ( `opacity_periodicity_rings`; 0 draws no copy ), the same seed in the same color: the
copies must glue without a seam. They do not count in the scale of the scene ( the zoom fits the cells themselves ) unless
`fit_periodicity_rings`. `domain`: the outline of
the domain ( the support that the group repeats ), dashed, and that of its images -- by default with a periodicity only.

With a sink ( partial transport, `sink_weight` ), the cells are drawn whole, as they are computed, but fainter
( `opacity_beyond_balls` ): what is integrated -- the part of each cell inside its ball -- is drawn on top, in the same
color, its arc approximated by a polygon.

With GROUPS ( `groups` ), the cells drawn are those of the members, which split the cell of their group; `virtual_seeds`
adds the virtual diracs of the exact duplicates ( `virtual_color` ), the points their split is the power diagram of.

### `.domain_outline()`

The support of the periodicity ( the domain of a diagram without one ) as half-spaces `( directions, offsets )`: what a
display draws dashed ( `add_to_viz` ). `None` for an unbounded diagram.

### `.hessian_rows(items=None, host=True)`

The derivatives of the measures with respect to the weights, row by row: `( nb_nbrs, ids, vals )`, the neighbors of each cell and the coefficients ( host arrays ).

### `.laplacian_solve(rhs, linear_solver='auto')`

The solution `x` of `L x = rhs`, `L` being the derivative of the measures with respect to the weights ( `rhs` sums to zero; `x[ 0 ] == 0` ). Differentiable with respect to `rhs`.

### `.moment(k)`

`int_{cell_i} p( x ) rho( x ) dx`, `[ n ]`, for `p = x_k` ( `k < d` ), `|x|^2` ( `k == d` ) or `1` ( `k == d + 1` ). Differentiable like `measures`.

### `.periodicity_rings(nb_rings=1)`

The elements of the group of `periodicity` up to the ring `nb_rings`, as affine maps `[ m, d, d + 1 ]` ( `[ M | V ]`, the map
`x -> M x + V` ), the identity first: ring 1 holds those whose image of the support touches the support. What a display
repeats the cells by ( see `add_to_viz` ); the identity alone without a periodicity.

### `.show(filename=None, title=None, height=500, **kwargs)`

In a notebook ( Jupyter, VS Code, Colab, ... ): displays the page inline, in an iframe
`height` pixels high -- the wheel and touch gestures are then left to the notebook, they
can be switched back on from the menu. Elsewhere, or when `filename` is given: writes the
HTML page ( `filename`, or a temporary file ) and opens it in the browser. Returns the
path written ( None when nothing was written ).

### `.to_viz(viz=None, title=None, **kwargs)`

A `Visualizer` holding this object ( `viz`, when given: it is then completed ). `title`
names a NEW scene; `kwargs` are those of `add_to_viz`.

### `.unfold(points)`

`points` ( `[ n, d ]`, one per seed, e.g. the barycenters of the cells ) taken from the frame of the folded seeds
( `positions` ) back to that of the positions as they were given. The identity without a periodicity.

### `.write_html(filename, title=None, **kwargs)`

Writes the self-contained HTML page of this object. Returns the path written.

### `.write_pvd(filename, title=None, axes=(0, 1, 2), **kwargs)`

Writes this object for ParaView as a `.pvd` collection ( with its `.vtu` beside it ),
the entry point one opens. Returns the `.pvd` path.

### `.write_vtk(filename, title=None, axes=(0, 1, 2), **kwargs)`

Writes this object for ParaView, as a `.vtu`. Returns the path written.

## `Groups`

```python
Groups(parent, agg_parent, center, scale, outer, dirs=None, seed_v=None, agg_v=None, positions=None, apex_shift=None)
```

THE TREE OF AGGREGATES of a diagram, in the user's order. Aggregates are numbered top-down ( a parent before its children ).

Per seed: `parent` ( `[ n ]` ints, the aggregate the seed is a child of, `-1` when it is alone ), `dirs` ( `[ n, d ]`, the unit
direction of a seed in an aggregate of exact duplicates, zero elsewhere ). Per aggregate: `agg_parent` ( `[ m ]`, `-1` for a root ),
`center` ( `[ m, d ]`, the origin of its frame -- the barycentre of its diracs, or the apex of a fan of duplicates, which
follows its site: `apex_shift`, read from `positions` ), `scale`
( `[ m ]`, the scale of its frame -- `r` for duplicates ), `outer` ( `[ m ]`, what its children's weights move outside: `scale`,
or 0 for duplicates ). Optional precise virtual weights `seed_v` ( `[ n ]` ) and `agg_v` ( `[ m ]` ), each child's `v` in its
parent's frame ( a solver's ): without them, they are read from the weights given to the diagram ( `virtual_weights` ).

Host arrays: the tree is a constant of the positions it was built on ( a derivative goes through the positions of the
seeds, not through the tree ). Built by `find_groups( positions )` ( `groups = "auto"`, the solver ), from nested index lists
( `Groups.nested`, or the lists themselves as `PowerDiagram( groups = ... )` ), or one level from representatives
( `Groups.from_rep` ).

| Attribute | |
|---|---|
| `apex_shift` | per aggregate of duplicates: its apex minus the position of its site ( the fan follows the site ), zero for the others |
| `depth` | the largest number of nested aggregates above a seed |
| `nb_groups` | the ROOT aggregates ( the groups of seeds that share one outer cell ) |
| `rep` | per seed: the smallest index of its root aggregate ( itself when it is alone ) |
| `site` | per seed: the smallest index at its exact position ( the first seed of its aggregate of duplicates, or itself ) |

### `.as_lists()`

the tree as nested index lists ( what `Groups.nested` takes ): one list per root aggregate

### `.from_rep(rep, positions, radius=None, dirs=None, apex=None)`

ONE LEVEL: `rep` ( `[ n ]` ) the smallest index of the group of each seed ( itself when alone ) -- a tree of depth one,
the exact duplicates of a group in an aggregate of their own

### `.from_sets(sets, positions, radius=None, dirs=None, apex=None)`

THE TREE OF A LAMINAR FAMILY of index sets ( two sets nested or disjoint; the ones that cross another are left out ): each
set an aggregate inside the smallest set holding it

### `.members()`

`{ rep: [ members ] }` for the root aggregates ( `rep` the smallest index )

### `.nested(lists, positions, radius=None, dirs=None, apex=None)`

THE TREE OF NESTED INDEX LISTS: each list an aggregate, an int a seed, a list in a list a nested aggregate --
`[ [ 0, 1 ], [ 2, [ 3, 4 ] ] ]` makes two roots, the second holding seed 2 and the aggregate `[ 3, 4 ]`. Seeds at the same
exact position must be in one aggregate of their own ( a list of duplicates only ), or they are put in one ( the
duplicates among an aggregate's direct seeds ). `radius` / `dirs` ( `[ n, d ]` ): the radius and the unit directions of the
virtual diracs of duplicates, springs and `VIRTUAL_RADIUS` local spacings by default; `apex` ( `[ n, d ]`, read at each
site's first seed ): the centres of their fans, when a point inside each site's cell is known ( the barycentre of the cell,
which a solver has: at the weights of a transport a seed need not lie in its own cell ).

### `.sites()`

`( keep, tree )`: the SITES ( the distinct positions: each aggregate of duplicates replaced by its first seed, which takes its
place and its virtual weight ) and their tree ( `None` when no aggregate is left ) -- the slots of the solver are the same
( `x[ keep ]` )

### `.virtual_positions(positions)`

`[ n, d ]` host: the virtual dirac of each member of an aggregate of exact duplicates, `center + scale u` ( the points
their split is the power diagram of ), the position of the others

### `.with_virtual_weights(seed_v, agg_v)`

the same tree with these precise virtual weights

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

### `.add_to_viz(viz, color=None, opacity=1.0, faces=True, edges=True, points=False, owners=None, nb_colors=None, frames=True, edge_opacity=1.0)`

Draws itself into a `Visualizer` ( see `sdot.viz.Visualizer`, and `cell_viz` for what an
unbounded cell drops ). Each item takes its color from its RANK IN THE BATCH,
on a block reserved in advance : the color of a cell of a diagram says WHICH seed.
`owners`: the rank to take the color from, per item, on a block of `nb_colors` -- the
pieces of one cell ( `PowerDiagram.support_pieces` ) have its color. `frames = False`: drawn without counting in the
scale of the scene ( the zoom fits what does count ). `edge_opacity`: that of the edges ( `opacity` is the faces' ).

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

### `.moved(maps)`

A copy of the cell moved by the affine map `x -> M x + V` -- `maps`: `[ d, d + 1 ]` ( `[ M | V ]` ), or one per item
`[ nb_items, d, d + 1 ]`. `M` orthogonal: the cell stays a cell, its topology and cut identifiers unchanged.

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
