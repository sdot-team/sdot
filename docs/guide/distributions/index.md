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
| `Box( origin = None, frame = None )` | uniform on a box ( the unit cube by default, in the dimension of the source ) |
| `Polygon( points )`, `Polyhedron( points )` | uniform on the convex hull of `points` ( every point must be a vertex ) |
| `Polytope( directions, offsets )` | uniform on the convex polytope `directions . x <= offsets` |
| `Image( values, frame = None, pixel_frame = None, origin = None )` | piecewise constant on a grid of pixels |
| `SumOfGaussians( positions, sigmas, weights = None )` | a mixture of isotropic gaussians |
| `Mesh( nodes, simplices, values = None )` | piecewise linear on a mesh of simplices ( triangles in 2D, tetrahedra in 3D ) |
| `Mesh( nodes, segments_or_triangles, values = None )` | carried by lines ( 2D, 3D ) or surfaces ( 3D ): see [Lines and Surfaces](./codim-mesh) |

### `Image`

`values` is indexed as `[ x, y, ... ]`. By default the image fits in the unit square ( cube... ) at `origin`: its longest
side has length 1, and the proportions are kept. To place it elsewhere:

- `origin`: the corner of the image;
- `frame`: the edges of the whole image, as rows ( `[ d, d ]` ), or their lengths for an axis-aligned image ( `[ d ]` );
- `pixel_frame`: the edges of one pixel instead.

### `SumOfGaussians`

`weights` are the masses of the gaussians, not their heights. Its support is the box that holds every gaussian up to
`support_sigmas = 6` standard deviations, or `support_box = ( lo, hi )`.

### `Mesh`

`values` gives the density at the corners: `[ nb_nodes ]` for a continuous density ( one value per node ), or
`[ nb_simplices, d + 1 ]` for a density discontinuous between simplices. Constant by default.

## Dimensions

The dimension comes from the source: `positions` of shape `[ n, d ]`. In 1D, the plan is computed by a sort, exactly
( see [W2, plans and settings](../ot-plan#ot-solve-the-settings-of-the-solver) ).
