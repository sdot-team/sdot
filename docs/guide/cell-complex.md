---
gallery: feature
description: "Walk the geometry of a power diagram: vertices, edges, facets and cells, each once, with their parents and children, simplices and distances — also on a torus."
tags: [geometry, cells, periodic, 2d, 3d]
image: /examples/img/cell-complex.png
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

The topology stays valid as long as the seeds move little, and the derivatives are exact as long as it is. Rebuild the complex when the seeds have moved enough to change which cells are neighbours. For $10^4$ cells in 3D, the kernels take about 30 ms and building the topology about 1 s.

## A consistent connectivity

Each cell of a power diagram is computed on its own, by cutting a polytope with planes. In floating point, two neighbouring cells can disagree on what they share: a facet seen by one cell and not by the other, or an edge of slightly different length on each side. Gluing such cells gives a "complex" whose numbers of faces are wrong, with facets that bound a single cell and edges around which the facets do not close.

How often does it happen? Below, the facets of raw cells compared side by side, sizes relative to the facet of a typical cell:

| cells cut in | facets on one side only | the largest of them | largest difference between the two sides |
|---|---|---|---|
| FP32, random seeds | 0–1 | $10^{-5}$ | $2 \cdot 10^{-3}$ |
| FP32, grid nudged by $10^{-6}$ | 20 000 out of 70 000 | $5 \cdot 10^{-5}$ | $10^{-4}$ |
| FP64, all of the above | 0 | — | $2 \cdot 10^{-12}$ |
| canonical predicates (what the complex cuts), all of the above | 0 | — | $10^{-14}$ |

The cell complex does not try to reconcile cells after the fact. It makes them agree from the start, so that gluing them is pure connectivity, with no distance compared and no tolerance:

1. **Every vertex is named by its generators.** A vertex is the corner of $d$ cuts in the cell of a seed, so it has $d + 1$ generators: seeds (with their image under a periodicity) or walls of the domain. Sorted, and written in the frame of the smallest seed among them, they are its key: the same in every cell that has the vertex.
2. **Every decision about a vertex depends on its generators alone.** The cells of the complex are cut with **canonical predicates** (`PowerDiagram( predicates = "canonical" )`, always used by `cell_complex()`):
   - the position of a vertex is solved from its own $d$ equations, in the frame of its smallest seed and in a fixed order, never interpolated along an edge;
   - whether a plane removes it is the sign of the power distance to the plane's seed minus that to its smallest seed (or of the distance to a wall). This sign is computed in double-double, about 106 bits, by the same code in every cell, so it is the sign for the doubles given, as good as exact;
   - an exact zero (a **tie**: a plane through a vertex, as on a grid of seeds) is broken by **simulation of simplicity**. The weights are perturbed by infinitesimals ordered by seed. The predicate is linear in them, so the first nonzero coefficient decides. Images of one seed share its weight, and four of them can sit on one sphere whatever the seeds (the corners of a period); there, a smaller deformation of the lattice of the periodicity decides.

   Two cells that share a vertex therefore take the same decisions about it. The perturbed diagram is in general position, the same in every cell: each vertex is in exactly $d + 1$ cells, and each facet is seen identically from both sides. Where planes meet at angles of $10^{-9}$, no copy of a vertex is computed differently from another.
3. **The faces of zero measure are contracted.** The tie-breaking blows a degenerate point up into vertices joined by edges of zero length, and so does rounding of doubles meant to be degenerate (a grid of thirds). The cells mark these edges: both ends of the edge lie, within $2^{-48}$ of the size of their terms, on the plane of the other's extra generator. That is a function of the two vertices alone, the same in every cell. The complex contracts them, and every face of zero measure disappears in every cell at once: an exact grid has the exact number of faces, its vertices shared by $2^d$ cells. `cell_complex( contract = False )` keeps the complex of the perturbed diagram instead, in general position.
4. **The faces of each cell come from its points**, once the zero edges are contracted, and not from its cuts. A cut is a facet when its points are in no other cut's. A point is a corner when it is alone in the smallest face containing it. The faces of a face are the maximal intersections of its corners with the other facets. This is the face lattice of the cell as it is, degenerate or not, in any dimension, and the faces of all the cells are then glued by the names of their corners.
5. **A check pass** looks at the result: `cx.diagnostics` and `cx.is_consistent`. It counts the facets between two seeds that only one of them has (`one_sided_facets`) and the faces of dimension $d - 2$ around which the facets do not close (`open_links`), and it measures the largest difference between the two copies of a facet (`facet_mismatch`). With canonical cells it finds nothing. The one exception is a contraction that would glue inconsistently: an **elongated sliver**, some of whose edges are below the resolution and others not, collapses into a lower face in one of its cells and not in the other. The zero edges of the faces in conflict are then kept, by their keys, in every cell alike, and counted in `kept_zero_edges`.

On grids nudged by 0, $10^{-12}$, $10^{-9}$, $10^{-6}$ and $10^{-3}$ of a cell, in 2D, 3D and 4D, in a box and on a torus, the complex has the right Euler characteristic, both copies of each facet agree to $10^{-14}$, and the check reports nothing. On exact grids up to 5D it has the exact number of faces of each dimension, and random seeds give the right Euler characteristic in 4D and 5D too.

What remains is the resolution of the contraction. A genuinely tiny feature, below $2^{-48}$ of the coordinates, is contracted like a degenerate one. In rare configurations (a 4D grid nudged by $10^{-12}$) contracting such a feature changes the topology near it without any local inconsistency. `contract = False` never contracts anything. At a point fixed by the group of a periodicity (the apex of a cone), no perturbation can separate the images of a seed: the contraction joins them, and only the contracted complex is meaningful there.

The canonical cut costs about four times the usual one (0.08 s against 0.02 s for $10^4$ cells in 3D), little next to the gluing. It is used for the complex; `PowerDiagram( predicates = "canonical" )` uses it for every computation of a diagram, `measures` included.

## What to know

- **A snapshot of the topology.** The topology is computed on the host from the seeds at the time of the call, and the numbers are recomputed from it (see [Differentiable numbers](#differentiable-numbers)). Build a new complex when the seeds have moved enough to change the neighbours.
- **Degenerate diagrams.** See [A consistent connectivity](#a-consistent-connectivity) above.
- **Cost.** For $10^4$ random cells in 3D, the topology takes about 1 s on one laptop (2 s on a grid, where the degenerate points are blown up and contracted), and the first numbers about 0.7 s more (the tables they are computed from are kept).
- **Unbounded cells.** Without a domain, faces that extend to infinity are still in the complex, with `is_bounded` false and an infinite measure. Only their finite vertices are listed.
- **Empty cells.** A seed whose cell is empty (its weight is too small) keeps its number among the cells, but has no face.
- **Any batch of cells.** `sdot.CellComplex( cells )` glues any batch of `Cell`s whose cut identifiers name the other items. The walls of the domain are then read from the faces themselves.
