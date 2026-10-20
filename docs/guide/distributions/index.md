# Distributions

A transport goes from a **source**, a sum of diracs, to a **target**, a continuous density. Both are normalized to a
total mass of 1: only the ratios between masses ( or between pixel values ) count.

<<< @/snippets/distributions.py

## Sources

| Class | |
|---|---|
| `SumOfDiracs( positions, weights = None )` | diracs at `positions` ( `[ n, d ]` ), of masses `weights` ( `[ n ]`, all equal by default ) |
| `ProjectedSumOfDiracs( points, normal, weights = None )` | 1D diracs, the projections of shared points on a normal per item ( e.g. one per angle of a tomography ), never materialized |

## Targets

The target also gives the **domain** of the transport: the cells of the power diagram are cut by the support it declares.

| Class | density |
|---|---|
| `Box( origin = None, frame = None )` | uniform on a box ( the unit cube by default, in the dimension of the source ): see [Convex Polytopes](./polytope#box) |
| `ConvexHull( points )` | uniform on the convex hull of `points` ( `[ n, d ]` ): see [Convex Polytopes](./polytope) |
| `Polytope( directions, offsets )` | uniform on the convex polytope `directions . x <= offsets` |
| `Image( values, frame = None, pixel_frame = None, origin = None )` | piecewise constant on a grid of pixels |
| `Image( coeffs, degree = p )` | a polynomial per pixel: see [Images and Splines](./image) |
| `SplineGrid( values, degree = 3 )` | a B-spline on a grid of pixels: see [Images and Splines](./image) |
| `SumOfGaussians( positions, sigmas, weights = None )` | a mixture of isotropic gaussians: see [Sums of Gaussians](./gaussians) |
| `Mesh( nodes, simplices, values = None )` | piecewise linear on a mesh of simplices ( triangles in 2D, tetrahedra in 3D ): see [Meshes](./mesh) |
| `Mesh( nodes, segments_or_triangles, values = None )` | carried by lines ( 2D, 3D ) or surfaces ( 3D ): see [Lines and Surfaces](./codim-mesh) |

### `Image`

`values` is indexed as `[ x, y, ... ]`. By default the image fits in the unit square ( cube... ): its longest side has
length 1, and the proportions are kept. `origin`, `frame` ( the whole image ) or `pixel_frame` ( one pixel ) place it
elsewhere. Polynomial pixels and splines: see [Images and Splines](./image).

### `SumOfGaussians`

`weights` are the masses of the gaussians, not their heights. Its support is the box that holds every gaussian up to
`support_sigmas = 6` standard deviations, or `support_box = ( lo, hi )`. See [Sums of Gaussians](./gaussians).

### `Mesh`

`values` gives the density at the corners: `[ nb_nodes ]` for a continuous density ( one value per node ), or
`[ nb_simplices, d + 1 ]` for a density discontinuous between simplices. Constant by default. See [Meshes](./mesh).

## Dimensions

The dimension comes from the source: `positions` of shape `[ n, d ]`. In 1D, the plan is computed by a sort, exactly
( see [plans and settings](../ot-plan#ot-solve-the-settings-of-the-solver) ).
