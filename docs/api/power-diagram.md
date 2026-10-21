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
| `sphere_areas` | With a sink ( partial transport ): the measure of the SPHERE inside each cell, where the density is not zero -- a length in 2D, an area in 3D, `[ n ]`; zeros without a sink. What a surface energy counts. Differentiable like `measures`. |

### `.add_to_viz(viz, seeds=True, seed_color='#000000', support_only=None, threshold=0.0, nb_periodicity_rings=1, opacity_periodicity_rings=0.25, fit_periodicity_rings=False, domain=None, domain_color='#000000', opacity_beyond_balls=0.2, ball_resolution=96, virtual_seeds=False, virtual_color='#888888', **kwargs)`

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
( `opacity_beyond_balls`, 0 to leave them out ): what is integrated -- the part of each cell inside its ball -- is
drawn on top, in the same color, its arc approximated by a polygon of `ball_resolution` sides ( in 3D: only the boundary of
the matter, its spheres by an icosphere of about as many points on a great circle ). A smaller one makes lighter pages,
which matters with many cells or many frames.

With GROUPS ( `groups` ), the cells drawn are those of the members, which split the cell of their group; `virtual_seeds`
adds the virtual diracs of the exact duplicates ( `virtual_color` ), the points their split is the power diagram of.

### `.cell_complex(tolerance=None)`

The cells glued along their faces ( `CellComplex` ): each vertex, edge, ..., facet once, numbered per dimension, with
its children ( the faces on its boundary ) and its parents ( the faces it bounds ) -- to walk the geometry of the diagram.
The facets know their two seeds, or their seed and the wall of the domain ( `walls`, the half-spaces of `domain_outline` ).
A host-side snapshot: a new one after moving the seeds.

With a periodicity, the complex of the QUOTIENT ( a torus, a cylinder... ): a facet across the period is there once, between a
seed and an image of another -- or of itself --, and the geometry of each face is read off one of its copies, next to the
given position of a seed of one of its cells ( `walls`: those that the group leaves, in the frame of the folded seeds ).

The cells are cut in doubles for it, whatever the `kernel_dtype` of the diagram: in floats, two cells can disagree on the
facet they share. `tolerance`: the distance under which two vertices are one where the diagram is not in general position
( a grid of seeds ) -- `1e-12` times the size of the diagram by default ( the cells are exact to about `1e-15` ).

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

## `CellComplex`

```python
CellComplex(cells, walls=None, tolerance=None, seeds=None, weights=None, power=True, group=None, maps=None, values=None, fold=None)
```

The cells of a diagram glued along their faces: every face once, numbered per dimension, with its incidences. Built by
`PowerDiagram.cell_complex()` -- or from any batch of cells, `CellComplex( cells, walls )`, the items being the seeds their
cut identifiers designate.

`faces( k )` the `k`-faces, `vertices` / `edges` / `facets` / `cells` those of dimension `0`, `1`, `d - 1`, `d`. `walls`: the
half-spaces `( directions, offsets )` of the domain, which the negative sides of the facets designate ( `-1 - wall` ).
`tolerance`: the distance under which two vertices are one, where the diagram is not in general position ( a grid of seeds ).

| Attribute | |
|---|---|
| `cells` | The `dim`-faces, numbered like the seeds ( an empty cell has no face ). |
| `diagnostics` | What the check of the gluing found ( see `_check` ): `contracted_facets`, `one_sided_facets`, `open_links`, and `facet_mismatch` ( computed when read: the measures of the copies of every facet ). |
| `dim` | The dimension of the space ( and of the cells ). |
| `edges` | The `1`-faces ( the cells themselves in 1D ). |
| `facets` | The `( dim - 1 )`-faces: what two cells share, or a cell and a wall of the domain ( `Faces.sides` ). |
| `is_consistent` | Whether every facet between two seeds is in both their cells and the facets close around every face of dimension `d - 2` ( see `diagnostics`: `one_sided_facets`, `open_links`, `contracted_facets`, `facet_mismatch` ). |
| `vertex_positions` | `[ nb_vertices, d ]`: the position of each vertex ( with a periodicity, of one of its copies ) -- a tensor of the framework of the positions, differentiable ( `geometry` ). |
| `vertices` | The `0`-faces. |

### `.boundary_distances(points, cells=None)`

`( distances [ m ], facets [ m ] )`: for each point, its distance to the boundary of its cell -- positive inside, the
distance to the plane of the nearest facet -- and that facet. `cells`: the cell of each point, the points then being in the
frame of the copies of these cells ( `locate`'s `moved` ); by default, `locate` finds them. A point outside the domain has
NaN and `-1`; a point given outside its cell, minus the largest overshoot of a plane ( a lower bound of its distance ).
`distances`: a tensor of the framework of the points, differentiable with respect to them and to the seeds.

### `.faces(k)`

The `k`-faces, `0 <= k <= dim`: a sequence of `Face`, and their arrays ( `Faces` ).

### `.geometry(positions=None, weights=None)`

The numbers of this complex for given seeds ( `CellGeometry` ): measures, centroids, normals, seed vectors, vertex positions,
simplices, distances -- tensors of the framework of `positions`, differentiable with respect to them and to `weights`. By
default the positions and weights the complex was built from ( what `measures`, `centroids`... read ); others keep THIS
topology -- the way to differentiate, or to evaluate under a `jax.jit`, with the complex built outside.

### `.incident(k, j)`

The `j`-faces of each `k`-face, as a CSR `( offsets, ids )`: those of face `i` are `ids[ offsets[ i ] : offsets[ i + 1 ] ]`
-- the faces it contains for `j < k`, the faces that contain it for `j > k`, sorted, except the vertices of a `2`-face, in
the order of its boundary ( counterclockwise for a 2D cell, around `normal` for a 3D facet; with a periodicity, a vertex
can come back there, two of its copies bounding the face ). `j == k`: each face itself.

### `.locate(points)`

`( cells [ m ], moved [ m, d ] )`: the cell that holds each point ( `-1` outside the domain ), and the point moved by the
group to the copy of that cell ( `Face.points` ) -- the point itself without a periodicity, a tensor of its framework. Exact:
the cell of the smallest power distance, found by a nearest-neighbour search among the seeds lifted in one more dimension
( `|x - p|^2 - w = |( x, 0 ) - ( p, sqrt( W - w ) )|^2 - W` ), and among their images with a periodicity.

### `.nb_faces(k)`

The number of `k`-faces.

### `.simplices(k=None, ids=None)`

A decomposition of the `k`-faces ( the cells by default; `ids`: those faces only ) into `k`-simplices: `( points [ m,
k + 1, d ], faces [ m ] )`, the corners of each simplex and the face it belongs to. Each face is fanned from the mean of its
vertices over its faces one dimension below, themselves fanned from theirs, down to the edges: a simplex per chain
`face > ... > edge`. Exact: the volumes of the simplices of a face add up to its measure. Unbounded faces have none; with a
periodicity, the simplices of a face are those of the copy its geometry is read off ( `Face.points` ). `points`: a tensor of
the framework of the positions, differentiable.

## `Faces`

```python
Faces(complex, k)
```

The `k`-faces of a `CellComplex`: a sequence of `Face` ( `len`, `[ i ]`, iteration ), and their values as arrays.

For the facets: `sides` and `normals`. For the cells: `seeds` ( their index, which is the seed's ).

| Attribute | |
|---|---|
| `centroids` | `[ m, d ]`: the centers of mass ( NaN for an unbounded face ). |
| `is_bounded` | `[ m ]` booleans: whether each face is bounded ( a diagram without a domain has unbounded cells ). |
| `measures` | `[ m ]`: length, area, volume... ( `1` for a vertex; infinite for an unbounded face ). Like all the numbers of the complex, a tensor of the framework of the positions, differentiable with respect to the positions and the weights ( `CellComplex.geometry` ). |
| `normals` | For the facets: `[ m, d ]`, the unit normal, from the first side to the second ( outward on the boundary ). |
| `seed_vectors` | For the facets: `[ m, d ]`, from the seed `sides[ 0 ]` to the seed `sides[ 1 ]` -- with a periodicity, to the image of `sides[ 1 ]` across the facet ( a cell can meet another, or itself, across the period ). Zero on the boundary of the domain. What a finite-volume scheme divides the area of a facet by: `measures / norm( seed_vectors )`. |
| `seeds` | For the cells: `[ n ]`, the seed of each ( its index ). |
| `sides` | For the facets: `[ m, 2 ]`, the seeds on both sides, the smaller first; the second is `-1 - wall` on the boundary of the domain ( `CellComplex.walls` ). |

### `.add_to_viz(viz, ids=None, color=None, opacity=1.0, faces=True, edges=True, edge_color=None, frames=True)`

Draws these faces ( or those of `ids` ), the bounded ones: the vertices as points, the edges as segments, the
`2`-faces as polygons, the `3`-faces by their `2`-faces. `color`: one for all, by default one per face ( on a block
reserved for all the faces of this dimension: the cell of a seed always has the same ). `faces = False`: the edges of
the polygons only ( a wireframe ), in `edge_color` ( by default a darker `color` ).

### `.children()`

`incident( dim - 1 )`: the faces of each one, one dimension below.

### `.incident(j)`

The `j`-faces of each of these faces, as a CSR `( offsets, ids )` ( see `CellComplex.incident` ).

### `.parents()`

`incident( dim + 1 )`: the faces that contain each one, one dimension above.

### `.show(filename=None, title=None, height=500, **kwargs)`

In a notebook ( Jupyter, VS Code, Colab, ... ): displays the page inline, in an iframe
`height` pixels high -- the wheel and touch gestures are then left to the notebook, they
can be switched back on from the menu. Elsewhere, or when `filename` is given: writes the
HTML page ( `filename`, or a temporary file ) and opens it in the browser. Returns the
path written ( None when nothing was written ).

### `.simplices(ids=None)`

`( points [ m, dim + 1, d ], faces [ m ] )`: these faces ( or those of `ids` ) cut into simplices ( see
`CellComplex.simplices` ).

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

## `Face`

```python
Face(complex, k, i)
```

ONE face of a `CellComplex`: its dimension `dim`, its number `index` among the faces of that dimension, and what it
is attached to -- `children`, `parents`, `faces( j )` -- as `Face`s again, to walk the complex.

| Attribute | |
|---|---|
| `cells` | The cells that contain it, as `Face`s ( their `index` is the seed ). |
| `centroid` | Its center of mass, `[ d ]`. |
| `children` | Its `dim - 1`-faces ( none for a vertex ). |
| `measure` | Its length, area, volume... ( `1` for a vertex, infinite when unbounded ): a scalar of the framework, differentiable. |
| `normal` | For a facet: its unit normal, from `sides[ 0 ]` to `sides[ 1 ]`. |
| `parents` | The `dim + 1`-faces that contain it ( none for a cell ). |
| `points` | `[ nb_vertices, d ]`: the positions of its vertices, in the order of `vertices` -- read off ONE copy of the face ( with a periodicity: the copy of one of its cells, next to the given position of its seed ). |
| `seed` | For a cell: its seed. |
| `seed_vector` | For a facet: from the seed `sides[ 0 ]` to ( the image of ) the seed `sides[ 1 ]` across it, zero on a wall. |
| `sides` | For a facet: `( a, b )`, the seeds on both sides, `b = -1 - wall` on the boundary of the domain. |
| `vertices` | Its vertices, as `Face`s -- in the order of its boundary for a `2`-face. |

### `.add_to_viz(viz, **kwargs)`

Draws this face ( see `Faces.add_to_viz` ).

### `.faces(j)`

The `j`-faces of this one, as `Face`s ( contained for `j < dim`, containing for `j > dim` ).

### `.ids(j)`

The numbers of the `j`-faces of this one ( contained for `j < dim`, containing for `j > dim` ), an int array.

### `.show(filename=None, title=None, height=500, **kwargs)`

In a notebook ( Jupyter, VS Code, Colab, ... ): displays the page inline, in an iframe
`height` pixels high -- the wheel and touch gestures are then left to the notebook, they
can be switched back on from the menu. Elsewhere, or when `filename` is given: writes the
HTML page ( `filename`, or a temporary file ) and opens it in the browser. Returns the
path written ( None when nothing was written ).

### `.simplices()`

`[ m, dim + 1, d ]`: a decomposition of this face into simplices ( see `CellComplex.simplices` ).

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

## `CellGeometry`

```python
CellGeometry(complex, positions, weights=None)
```

The numbers of a `CellComplex` for given seeds ( `CellComplex.geometry` ): tensors of the framework of the positions,
differentiable with respect to the positions and the weights. The faces are those of the complex; their geometry is that of
these seeds.

| Attribute | |
|---|---|
| `normals` | `[ nb_facets, d ]`: the unit normals of the facets, from their first side to their second. |
| `seed_vectors` | `[ nb_facets, d ]`: from the seed `sides[ 0 ]` to ( the image of ) the seed `sides[ 1 ]` across each facet, zero on a wall. |
| `vertex_positions` | `[ nb_vertices, d ]`: the position of each vertex ( with a periodicity, of one of its copies ). |

### `.boundary_distances(points, cells=None)`

`( distances [ m ], facets [ m ] )`: the distance from each point to the boundary of its cell ( positive inside ) and the
nearest facet, differentiable with respect to the points and the seeds ( the facet chosen on the values, a constant of the
derivative ). `cells` / `points` as in `CellComplex.boundary_distances`.

### `.centroids(k)`

`[ nb_faces( k ), d ]`: the centers of mass ( NaN for an unbounded face ).

### `.measures(k)`

`[ nb_faces( k ) ]`: length, area, volume... ( `1` for a vertex, infinite for an unbounded face ).

### `.simplices(k=None)`

`( points [ m, k + 1, d ], faces [ m ] )`: the faces of dimension `k` ( the cells by default ) cut into simplices ( see
`CellComplex.simplices` ), the points a tensor of the framework of the positions.

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
