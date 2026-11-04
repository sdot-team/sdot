# 1D Transport

On a line, transport needs no Newton loop. The monotone map is optimal: sort the diracs, accumulate their masses, and
invert the cumulative distribution function of the density at those masses. sdot uses this whenever the problem is 1D:

```python
plan = sdot.ot_solve( sdot.SumOfDiracs( x[ :, None ] ), sdot.Image( values = density ) )   # d = 1: a `SdotPlan1d`, exact
```

In 1D, `ot_solve` returns a `SdotPlan1d`, whose settings are a `Direct`: the result is exact, and the cost, the barycenters
and the gradients with respect to the positions come from a single pass. The image may have constant pixels or affine
ones ( `Image( values = coeffs, degree = 1 )`, `coeffs [ pixels, 2 ]` the density at both ends of each pixel: a piecewise
linear density ); the gradients then go to its coefficients.

Between two images, `ContinuousOtPlan1d( Image, Image )` needs no sort: it merges the pixels of both ( degree 0 or 1 on
each side ), O( pixels ). It computes the cost only.

## Many transports at once

The 1D case matters most when there are many transports between the projections of the same objects, for example the
projections of a cloud along hundreds of directions in tomography or in sliced transport. The plans take a batch of
directions. The projected source is never stored: with 1e7 points and 600 directions, the `[ directions, n ]` array of
projected positions would be 48 GB. The kernels compute the projections on the fly and send the gradients back to the
shared points ( [Projected Sources](./distributions/projected) ):

| Plan | source | |
|---|---|---|
| `SdotPlan1d( ProjectedSumOfDiracs, Image )` | diracs | exact, by a sort per direction |
| `ContinuousOtPlan1d( ProjectedSumOfTents, Image )` | tents ( a half-width per point ) | exact: the breakpoints sorted, the map quadratic on each piece |
| `ContinuousOtPlan1d( ..., method = "grid", scale = 8 )` | tents | the tents deposited on a fine grid, no sort: `O( directions × cells )` |
| `projected1d.ProjectedLoss1d` | domes or tents | the same grid transport, value and gradient in one call |

## Performance

The timings, against POT and scipy, are on [Performance: 1D Transport](./performance-1d).
