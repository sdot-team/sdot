---
gallery: feature
description: "The Lebesgue measure on a convex polytope: a box, half-spaces, a convex hull, in 2D and 3D."
tags: [cells, 2d, 3d]
image: /examples/img/polytope.png
order: 1
---

<script setup>
import { withBase } from 'vitepress'
</script>

# Convex Polytopes

The simplest target: the **Lebesgue measure** on a convex polytope, a constant density inside and zero outside. Three ways to
give the polytope:

```python
target = sdot.ConvexHull( points )                      # points [ n, d ]: the convex hull of points, in any dimension
target = sdot.Polytope( directions, offsets )           # the half-spaces directions . x <= offsets ( [ m, d ], [ m ] )
target = sdot.Box( origin = None, frame = None )        # a box: the unit square ( cube ) by default
```

The polytope is also the **domain** of the transport: the cells start from it, and are cut by the planes of the power
diagram. Nothing lies outside it, so no cell is unbounded.

## ConvexHull

`ConvexHull( points )` is the convex hull of `points` ( `[ n, d ]`, in any order ): a polygon in 2D, a polyhedron in 3D. The
points inside the hull are ignored, and a flat hull ( all the points in a hyperplane ) raises a `ValueError`.

300 diracs gathered in the middle of a heptagon, and their cells: first with zero weights ( the Voronoi diagram cut by the
polygon ), then with the weights of the transport, where each cell holds $1 / 300$ of the area ( the slider at the top
switches between them ):

```python
target = sdot.ConvexHull( vertices )                    # 7 vertices on an ellipse
p = sdot.plan( sdot.SumOfDiracs( x ), target )
```

<iframe
  :src="withBase('/guide/polytope-plan.html') + '?wheel=0&touch=0'"
  title="The cells of 300 diracs in a heptagon: Voronoi, then transport"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/polytope-plan.html')" target="_blank">Open in a separate page ↗</a></p>

## Polytope

`Polytope( directions, offsets, density = 1 )` is the intersection of the half-spaces $a_c \cdot x \le b_c$. The directions
need not be unit vectors: each row is normalized, with its offset. The polytope has to be bounded and non-empty ( it raises
otherwise ), but a half-space may be redundant.

A regular nonagon cut by one more half-space, and the Lloyd quantization of its Lebesgue measure ( 2000 diracs, see
[Lloyd quantization](/examples/lloyd-quantization) ):

```python
t = 2 * np.pi * np.arange( 9 ) / 9
directions = np.concatenate( [ np.stack( [ np.cos( t ), np.sin( t ) ], -1 ), [ [ 0.3, 1.0 ] ] ] )
offsets = np.r_[ np.full( 9, 0.5 ), 0.2 ]

x = sdot.quantization( sdot.Polytope( directions, offsets ), 2000, seed = 0 )
```

<iframe
  :src="withBase('/guide/polytope-halfspaces.html') + '?wheel=0&touch=0'"
  title="Lloyd quantization of a polytope given by its half-spaces"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/polytope-halfspaces.html')" target="_blank">Open in a separate page ↗</a></p>

## In 3D

Nothing changes but the dimension. An icosahedron from its 12 vertices, and 4000 diracs colored by their depth in the view
( warm in front; `c` switches it ):

```python
phi = ( 1 + 5 ** 0.5 ) / 2
v = [ [ 0, s, t * phi ] for s in ( -1, 1 ) for t in ( -1, 1 ) ]
vertices = np.array( [ p[ k: ] + p[ :k ] for p in v for k in range( 3 ) ], dtype = float )

x = sdot.quantization( sdot.ConvexHull( vertices ), 4000, seed = 0 )
```

<iframe
  :src="withBase('/guide/polytope-icosahedron.html') + '?wheel=0&touch=0'"
  title="Lloyd quantization of an icosahedron"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/polytope-icosahedron.html')" target="_blank">Open in a separate page ↗</a></p>

The cells of a transport from 60 random diracs to the unit cube cut by two oblique planes:

```python
directions = np.concatenate( [ np.eye( 3 ), -np.eye( 3 ), [ [ 1, 1, 1 ], [ -1, 0.5, -1 ] ] ] )
offsets = np.r_[ np.ones( 3 ), np.zeros( 3 ), 2.4, -0.25 ]

p = sdot.plan( sdot.SumOfDiracs( x ), sdot.Polytope( directions, offsets ) )
```

<iframe
  :src="withBase('/guide/polytope-cells3d.html') + '?wheel=0&touch=0'"
  title="The cells of a transport to a cut cube"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/polytope-cells3d.html')" target="_blank">Open in a separate page ↗</a></p>

## Box

`Box( origin = None, frame = None )` is the box `origin + frame^T [ 0, 1 ]^d`: the rows of `frame` are its edges, or `frame`
is their lengths for an axis-aligned box. By default it is the unit square ( cube... ) in the dimension of the source:
`sdot.w2( f, sdot.Box() )` works for any `f`.

A box is an [`Image`](./image) of one pixel. It is differentiable with respect to `origin` and `frame`.

## What is computed

The density is constant, so every integral over a cell is a polynomial integral over a convex polytope: the masses, the
barycenters, the transport cost and the facets of the Newton system are **exact**, with no quadrature. There is no width
continuation either: the density is already as smooth as it gets, and the solver goes straight to Newton.

On the CUDA card, a polygon or a polyhedron with few enough facets, all of whose vertices are simple, is the cell every seed
starts from; anything else starts from the box of its vertices, cut by its facets. The result is the same.

A constant density on a polytope also works with a [periodicity](../ground-metrics/periodicity) ( 2D and 3D, on the card
too for translations ) and with [partial transport](../ground-metrics/partial) ( 2D, on the CPU ).

## Reproducing the pages

They are written by [`docs/guide/distributions/polytope_figures.py`](https://github.com/sdot-team/sdot/blob/main/docs/guide/distributions/polytope_figures.py):

```bash
python docs/guide/distributions/polytope_figures.py                     # all of them
python docs/guide/distributions/polytope_figures.py plan icosahedron    # or some of them
```

## Limits

- Convex polytopes only. A non-convex domain is a union of convex ones, or an `Image` ( a mask ), or a `Mesh` of simplices.
- The geometry ( `points`, `directions`, `offsets` ) is a constant of the transport: nothing differentiates through it. The
  `density` may be traced, but the target is normalized, so its derivative is zero. A `Box` is differentiable with respect to
  its `origin` and `frame`.
