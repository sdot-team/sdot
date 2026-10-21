---
gallery: feature
description: "Walk the geometry of a power diagram: vertices, edges, facets and cells, each once, with their parents and children, simplices and distances — also on a torus."
tags: [geometry, cells, periodic, 2d, 3d]
image: /guide/img/cell-complex.png
order: 8
---

<script setup>
import { withBase } from 'vitepress'
</script>

# The Cell Complex

`PowerDiagram.cells` gives each cell on its own: the facet between two seeds is stored twice, and a vertex is stored once in every cell that meets there. To work on the geometry of the diagram itself — its facets, its graph of neighbors, the edges where three cells meet — `cell_complex()` glues the cells along the faces they share:

```python
cx = pd.cell_complex()
```

In the **cell complex**, each face is counted once and numbered within its dimension $k$:

| $k$ | | in 2D | in 3D |
|---|---|---|---|
| $0$ | `cx.vertices` | vertices | vertices |
| $1$ | `cx.edges` | edges (= facets) | edges |
| $d - 1$ | `cx.facets` | edges | polygons |
| $d$ | `cx.cells` | cells, numbered like the seeds | cells, numbered like the seeds |

Faces are linked by **incidence**. The *children* of a $k$-face are the $(k-1)$-faces on its boundary, and its *parents* are the $(k+1)$-faces that contain it. More generally, `faces( j )` gives the $j$-faces of a face for any $j$: the vertices of a cell, or the cells around a vertex.

<<< @/snippets/cell_complex.py

## One face at a time

`cx.faces( k )[ i ]` — or `cx.vertices[ i ]`, `cx.facets[ i ]`, ... — is a `Face`. Walking from a face gives more `Face`s:

| | |
|---|---|
| `children`, `parents` | the faces one dimension below, one dimension above |
| `faces( j )` | the $j$-faces it contains ($j <$ `dim`) or that contain it ($j >$ `dim`) |
| `vertices`, `cells` | `faces( 0 )` and `faces( d )`; the vertices of a 2-face come in the order of its boundary |
| `points` | the positions of its vertices, `[ nb_vertices, d ]` |
| `measure`, `centroid` | its length, area or volume, and its center of mass |
| `sides`, `normal`, `seed_vector` | for a facet: the two seeds it separates (the smaller first), its unit normal from the first to the second, and the vector from the first seed to the second |
| `seed` | for a cell: its seed |
| `simplices()` | the face cut into simplices, `[ m, dim + 1, d ]` |

A facet on the boundary of the domain has a single seed. Its second side is negative, `-1 - w`, where `w` is the index of the wall it lies on among the half-spaces `cx.walls`: those of the domain, or of the support of the distribution.

Below, starting from the cell at the center (black), its neighbors are the other sides of its facets (grey). Its longest edge (red) is a child of three facets, its `parents` (solid). Those facets are shared by the three cells around the edge, whose seeds are drawn in red.

<iframe
  :src="withBase('/guide/cell-complex-walk-3d.html') + '?wheel=0&touch=0'"
  title="Walking the cell complex"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/cell-complex-walk-3d.html')" target="_blank">Open in a separate page ↗</a></p>

## All the faces at once

To work on many faces at once, the same quantities exist as arrays over all the faces of one dimension (a `Faces`): `measures`, `centroids`, `is_bounded`, and for the facets `sides`, `normals` and `seed_vectors`.

The incidences come as CSR arrays: `off, ids = cx.faces( k ).incident( j )` gives the $j$-faces of face `i` as `ids[ off[ i ] : off[ i + 1 ] ]`. `children()` and `parents()` are `incident( k - 1 )` and `incident( k + 1 )`.

For example, the **Laguerre graph** — two seeds are linked when their cells share a facet — is `cx.facets.sides` without its negative entries. The area of each facet divided by the distance between its two seeds gives the transmissibilities of a finite-volume scheme on the cells. The [heat equation on a torus](../examples/finite-volumes) is built from them:

```python
sides = cx.facets.sides
inner = sides[ :, 1 ] >= 0                       # the facets between two seeds
trans = cx.facets.measures[ inner ] / np.linalg.norm( cx.facets.seed_vectors[ inner ], axis = 1 )
```

<iframe
  :src="withBase('/guide/cell-complex-laguerre-graph.html') + '?wheel=0&touch=0'"
  title="The Laguerre graph of a power diagram"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/cell-complex-laguerre-graph.html')" target="_blank">Open in a separate page ↗</a></p>

The facets on the domain are drawn in red, with `cx.facets.add_to_viz( viz, ids = ..., color = ... )`. Each `Face` and each `Faces` can draw itself (`add_to_viz`, `show`, `write_html`); `faces = False` draws only the edges.

## Simplices, point location and distances

`cx.simplices( k )` cuts the $k$-faces into $k$-simplices — the cells by default. It returns `( points, faces )`: the corners of each simplex, `[ m, k + 1, d ]`, and the face it belongs to, `[ m ]`. Each face is fanned from its centroid over its faces one dimension below, and those are fanned from theirs, down to the edges. The volumes of the simplices of a face add up to its measure exactly. This is what is needed to integrate over the cells with any quadrature rule, or to mesh them.

`cx.locate( points )` gives the cell that holds each point. It is exact: the smallest power distance, found by a nearest-neighbour search on the seeds lifted in one more dimension, $|x - p_i|^2 - w_i = |( x, 0 ) - ( p_i, \sqrt{W - w_i} )|^2 - W$. Points outside the domain get `-1`. With a periodicity, the point is also moved by the group next to the copy of its cell.

`cx.boundary_distances( points )` gives the distance from each point to the boundary of its cell, and the facet that is nearest. The cells are convex, so this is the distance to the nearest plane of their facets. Points outside their cell get a negative value, a lower bound of their distance.

```python
S, owner = cx.simplices()                        # triangles in 2D, tetrahedra in 3D
cells, moved = cx.locate( X )                    # the cell of each point of X
dist, facets = cx.boundary_distances( X )        # its distance to the boundary of that cell, and the nearest facet
```

Below, the cells are cut into triangles (grey) and points are coloured by their distance to the boundary of their cell. For some of them, a red segment goes to the nearest facet.

<iframe
  :src="withBase('/guide/cell-complex-simplices.html') + '?wheel=0&touch=0'"
  title="Simplices and distances"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/cell-complex-simplices.html')" target="_blank">Open in a separate page ↗</a></p>

## With a periodicity

With a [periodicity](./ground-metrics/periodicity), the complex is that of the **quotient**: a torus, a Klein bottle, a cylinder, a cone... A facet across the period is counted once. It lies between a seed and an image of another seed, or of the same seed: with few seeds, a cell can meet itself. Its `sides` are the two seeds, and `seed_vector` points from the first seed to the image of the second that the facet faces.

The geometry of each face (`points`, `centroid`, `normal`) is read from one of its copies, next to the given position of a seed of one of its cells. A vertex can therefore appear twice in the `points` of a face, when two of its images bound that face.

The complex of a torus has Euler characteristic $V - E + F = 0$, and a cone has $1$. Below, each facet across the period is drawn in red once, on the copy of one of its cells. The black segments link each seed to the image of its neighbor across each facet.

<iframe
  :src="withBase('/guide/cell-complex-torus.html') + '?wheel=0&touch=0'"
  title="The cell complex of a torus"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/cell-complex-torus.html')" target="_blank">Open in a separate page ↗</a></p>

## Differentiable numbers

The numbers of the complex — `measures`, `centroids`, `normals`, `seed_vectors`, `vertex_positions`, the points of `simplices`, `boundary_distances` — are **tensors of the framework of the positions** (NumPy, JAX, PyTorch…), differentiable with respect to the positions and the weights of the seeds:

```python
pd = sdot.PowerDiagram( positions, weights, boundaries = box )       # torch tensors with requires_grad, or jax arrays
cx = pd.cell_complex()
loss = ( cx.facets.measures ** 2 ).sum() + cx.cells.centroids.sum()
loss.backward()                                                       # torch: positions.grad, weights.grad
```

The **topology** — which faces exist and what contains what — is computed from the values on the host. It is a constant of the derivative: the derivative is that of the geometry of this topology. The geometry itself is computed from the seeds by a kernel with its adjoint (`sdot/complex/Geometry.h`):

- each vertex is the solution of the $d$ equations of its generators, $2 ( q - p_s ) \cdot x = |q|^2 - |p_s|^2 - w_q + w_s$ for each bisector, or $n \cdot x = o$ for a wall;
- each face is integrated over the simplices of its fan, whose corners are these vertices or the means of the vertices of its sub-faces.

Both sides of a facet read the same numbers, by construction.

`cx.geometry( positions, weights )` evaluates the same topology for other seeds. This is the way to differentiate inside a `jax.jit`, where the topology cannot be built (it needs values): build the complex outside, evaluate the geometry inside.

```python
cx = sdot.PowerDiagram( positions, weights, boundaries = box ).cell_complex()         # outside the jit
step = jax.jit( jax.grad( lambda p: cx.geometry( p, weights ).measures( 1 ).sum() ) )  # inside: the same topology
```

The topology stays valid as long as the seeds move little, and the derivatives are exact as long as it is. Rebuild the complex when the seeds have moved enough to change which cells are neighbours. For $10^4$ cells in 3D, the kernels take about 30 ms and building the topology about 2.5 s.

## A consistent connectivity

Each cell of a power diagram is computed on its own, by cutting a polytope with planes. In floating point, two neighbouring cells can disagree on what they share: a facet seen by one cell and not by the other, or an edge of slightly different length on each side. Gluing such cells gives a "complex" whose numbers of faces are wrong, with facets that bound a single cell and edges around which the facets do not close.

How often does it happen? Below, the facets of raw cells compared side by side, sizes relative to the facet of a typical cell:

| cells cut in | facets on one side only | the largest of them | largest difference between the two sides |
|---|---|---|---|
| FP32, random seeds | 0–1 | $10^{-5}$ | $2 \cdot 10^{-3}$ |
| FP32, grid nudged by $10^{-6}$ | 20 000 out of 70 000 | $5 \cdot 10^{-5}$ | $10^{-4}$ |
| FP64, all of the above | 0 | — | $2 \cdot 10^{-12}$ |

The cell complex is built so that its connectivity can be relied on:

1. **The cells are cut in FP64**, whatever the `kernel_dtype` of the diagram (which stays the float of `measures` and of the transport).
2. **Every face is named exactly by its generators**: the seeds whose bisectors carry it, or the walls. A vertex is the corner of $d$ cuts in the cell of a seed, so $d + 1$ generators. With a periodicity, each generator is a seed together with its image, and the name is written in the frame of the smallest seed among them, so every cell that has the face gives the same name. No distance is compared for that.
3. **Where the diagram is degenerate** — more than $d + 1$ cells at a point, as on a grid — one point has several names, which differ from cell to cell. There, `tolerance` acts as a resolution: points closer than that are one vertex, in every cell alike, so that a feature smaller than the tolerance disappears everywhere at once. The default is $10^{-12}$ times the size of the diagram, a thousand times above what FP64 cells are exact to. A point that is a corner of no cell (a cut through the middle of an edge) is not a vertex: its edge stays one edge.
4. **The faces of each cell come from its vertices, not from its cuts.** In 4D and above, a vertex of a grid is shared by $2^d$ cells and a cell is no longer a simple polytope: a cut can hold a single point, and $d$ cuts through a vertex do not make it a corner. So, for each cell (in C++, in parallel), a cut is a facet when the vertices on it span a $(d-1)$-plane, which is judged on the exact normals of the cuts, not on the positions; a vertex is a corner when the facets through it have normals of rank $d$; and the faces of dimension $k$ are the intersections of facets whose normals have rank $d - k$. This is the true face lattice of the cell, in any dimension, and the faces of all the cells are then glued by the names of their corners.
5. **A check pass** looks at the result: `cx.diagnostics` and `cx.is_consistent`.
   - A facet between two seeds that only one of them has is contracted to a point on both sides when it is smaller than 100 times the tolerance, and the faces are glued again (`contracted_facets`).
   - A larger one is reported (`one_sided_facets`), as are the faces of dimension $d - 2$ around which the facets do not close (`open_links`). Both come with a warning.
   - `facet_mismatch` is the largest difference between the measures of the two copies of a facet.

On grids nudged by 0 to $10^{-3}$ of a cell, in 2D and 3D, in a box and on a torus, the complex has the right Euler characteristic and nothing to report. On exact grids, up to 5D, it has the exact number of faces of each dimension, and random seeds give the right Euler characteristic in 4D and 5D too. A cell that the diagram itself gets wrong is reported, not repaired.

One case is beyond the resolution: grids nudged by about $10^{-9}$, where the planes through a vertex are nearly dependent. The copies of a vertex computed in two cells can then differ by $10^{-7}$, far more than the tolerance. Some faces are then seen by one side only; the check reports them (`is_consistent` is false, with a warning) instead of guessing. Raising `tolerance` above that gap (`cell_complex( tolerance = 1e-6 )`) glues them.

## What to know

- **A snapshot of the topology.** The topology is computed on the host from the seeds at the time of the call, and the numbers are recomputed from it (see [Differentiable numbers](#differentiable-numbers)). Build a new complex when the seeds have moved enough to change the neighbours.
- **Degenerate diagrams.** See [A consistent connectivity](#a-consistent-connectivity) above.
- **Cost.** For $10^4$ random cells in 3D, the topology takes about 1 s on one laptop, and the first numbers about 0.7 s more (the tables they are computed from are kept).
- **Unbounded cells.** Without a domain, faces that extend to infinity are still in the complex, with `is_bounded` false and an infinite measure. Only their finite vertices are listed.
- **Empty cells.** A seed whose cell is empty (its weight is too small) keeps its number among the cells, but has no face.
- **Any batch of cells.** `sdot.CellComplex( cells )` glues any batch of `Cell`s whose cut identifiers name the other items. The walls of the domain are then read from the faces themselves.
