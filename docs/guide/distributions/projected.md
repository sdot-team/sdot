---
gallery: feature
description: "Clouds seen along lines: projected diracs, tents and domes, transported to one profile per direction."
tags: [projection, tomography, 2d, jax]
image: /examples/img/tomography-tents.png
order: 6
---

<script setup>
import { withBase } from 'vitepress'
</script>

# Projected Sources

A cloud of points $x_i$ of $\mathbb{R}^d$, seen along a **batch of directions**: for each unit vector $n_a$, the 1D measure

$$
\mu_a = \tfrac{1}{n} \sum_i f_i( \,\cdot - x_i \cdot n_a )
$$

on the line, where $f_i$ is the profile a point leaves there. $\mu_a$ is transported to a 1D target $\nu_a$, one per direction,
and the cost is $W_2^2( \mu_a, \nu_a )$, differentiable with respect to the points. It is the right tool wherever the
data are 1D marginals of a cloud: the projections of an object in tomography, for example.

The cost is the same as for any other source, only the source is never built: with $10^7$ points and $10^3$ directions the
`[ directions, n ]` array of the projected positions is 80 GB. The classes below compute $x_i \cdot n_a$ on the fly, in the
kernels, and scatter the gradients back onto the shared points.

| Class | $f_i$ | |
|---|---|---|
| `ProjectedSumOfDiracs( points, normal )` | a dirac at $x_i \cdot n$ | |
| `ProjectedSumOfTents( points, half_widths, normal )` | $( 1 - \lvert s \rvert / w_i )_+ / w_i$, a tent of half-width $w_i$ | continuous |
| `ProjectedSumOfAnisotropicTents( points, matrices, normal )` | the same, with the half-width $w_i = \lvert M_i^T n \rvert$ that depends on the direction | the shadow of an ellipse |

`points` `[ n, d ]`, `half_widths` `[ n ]` and the masses are **shared** by all the directions. `normal` has one row per
direction, declared as a batch axis, and so does the target: a 1D `Image`, one profile per direction.

## Diracs

```python
from loom.tensor import Axis, ShapeVar

num_angle = Axis( ShapeVar( len( normals ) ), name = "num_angle" )           # the batch: one transport per direction
src = sdot.ProjectedSumOfDiracs( points = x, normal = normals, batch_axes = [ num_angle ] )
dst = sdot.Image( values = profiles, origin = [ s_min ], pixel_frame = [ [ dw ] ], batch_axes = [ num_angle ] )

plan = sdot.SdotPlan1d( src, dst )
plan.cost                                        # [ directions ]: W2^2 per direction, both sides normalized to mass 1
```

`normals` is `[ directions, d ]`, `profiles` `[ directions, bins ]`: the mean density on the bins of width `dw` from `s_min`.
In 1D the plan is a sort, exact, and `jax.grad` of `plan.cost.sum()` with respect to `x` is what you expect.

The cost is only piecewise smooth, though: when two diracs cross on the line they exchange their ranks, and the gradient jumps.
A descent on the positions of many diracs, whose projections cross all the time, feels it.

## Tents

A tent is the simplest object with a width. Two tents that cross just overlap, and the cost stays smooth.

```python
src = sdot.ProjectedSumOfTents( points = x, half_widths = w, normal = normals, batch_axes = [ num_angle ] )
plan = sdot.ContinuousOtPlan1d( src, dst )
plan.cost
```

`ContinuousOtPlan1d` is the transport between two **continuous** densities on a line, here piecewise linear against piecewise
constant. It is exact: per direction the $3n$ breakpoints of the tents are sorted, and on each piece the transport map is
quadratic, integrated by a 3-point Gauss-Legendre rule. It differentiates with respect to **both** `points` and `half_widths`:

```python
def cost( x, w ):
    src = sdot.ProjectedSumOfTents( points = x, half_widths = w, normal = normals, batch_axes = [ num_angle ] )
    return sdot.ContinuousOtPlan1d( src, dst ).cost.sum()

grad_x, grad_w = jax.grad( cost, argnums = ( 0, 1 ) )( x, w )
```

Options of `ContinuousOtPlan1d`:

- `with_barycenters = True`: the barycenter of the transport of each tent ( `plan.barycenters` );
- `with_gradient = True`: the gradient of `plan.cost.sum()` comes out of the **same pass** ( `plan.points_grad`,
  `plan.half_widths_grad` ), half the work of `jax.value_and_grad`, whose adjoint runs the transport again. The call is then
  not differentiated: it goes inside a `jax.custom_vjp` that returns these gradients. Optional: `jax.grad` of the plain cost
  works as is ( see [the loss](#the-loss) of the tomography example ), and this is about twice as fast with many
  tents;
- `method = "grid"`, `scale = 8`: each tent taken as uniform inside the cells of a grid `scale` times finer than the bins of the
  target: an exact deposit of the tents into the cells, then a transport with no sort at all, in
  $O( n \times \text{cells per tent} )$ per direction. The error vanishes as `scale` grows. This is the version for a card and
  many tents ( the target's bins must be uniform ).

## Anisotropic tents

A round object projects to the same tent at every direction. An **ellipse** does not: the image of a disk by a matrix $M_i$ has,
along the direction $n$, a shadow of half-width $w_i( n ) = \lvert M_i^T n \rvert$ ( the support function of the ellipse ).
`ProjectedSumOfAnisotropicTents( points, matrices, normal )` is the tent of that half-width, with `matrices` `[ n, d, d ]`; round
tents are `matrices = w * identity`.

```python
src = sdot.ProjectedSumOfAnisotropicTents( points = x, matrices = M, normal = normals, batch_axes = [ num_angle ] )
plan = sdot.ContinuousOtPlan1d( src, dst )
```

It is differentiated with respect to `points` and `matrices` ( `jax.grad` of `plan.cost.sum()` ), in the exact and in the grid
methods. Not yet: `with_gradient`, and the card ( its kernels read the half-widths from the first direction, and refuse this type at
compile time ). `matrices` must keep $\lvert M_i^T n \rvert > 0$ at every direction; `M = L L^T + \varepsilon I` with `L` free is a
parametrization that does. **Keep `ProjectedSumOfTents` for round tents**: it computes one half-width per tent where this
class computes one per tent and direction, and it has the one-pass gradient and the card.

## Domes: the projection of a radial density

An object that has a size in the plane projects to the same profile at **every** direction, if it is radial. The density
$( 1 - \lvert x \rvert^2 / r^2 )_+^{\,k - 1/2}$ gives, by Abel transform, $c_k ( 1 - s^2 / r^2 )_+^k$. For $k = 1$ it is the
closest thing to a disk ( the disk itself would be $k = 1/2$, whose profile has a vertical tangent at the edge ). A **dome** is
such an object: its profile has a closed-form primitive, so its exact mass in each cell of a fine grid is known.
`sdot.projected1d.ProjectedLoss1d` computes the loss for domes ( or tents ) in three steps, none of which sorts or builds an
`[ directions, n ]` array:

1. **deposit**: each object puts into every fine cell it covers its exact mass, from the primitive of $f$;
2. **grid transport**, per direction: $W_2^2$ between two densities piecewise constant on the same grid, exact, in
   $O( \text{directions} \times \text{cells} )$;
3. **gather**: $\partial L / \partial x_i$ from the same primitives.

```python
from sdot.projected1d import ProjectedLoss1d

L = ProjectedLoss1d( angles, profiles, s_min, dw, shape = "dome", k = 1, max_radius = 0.05 )
value = L( points, radii )                          # differentiable in points [ n, 2 ] and radii [ n ]
```

`angles` are in the plane ( normals $( \cos, \sin )$ ). The only approximation is the grid ( `scale` fine cells per bin, 8 by
default ); fused kernels, CPU and CUDA, run steps 1 and 3, and `backend = "jax"` is the reference. `L.sharded( mesh )` splits
the directions between the devices of a `jax.sharding.Mesh`, and a batch axis solves $K$ problems of the same geometry at once.
On a 2080 Ti, one value and gradient takes 615 ms for $10^7$ objects, 600 directions and 2048 bins.

## Which one?

| | diracs | tents | domes |
|---|---|---|---|
| smooth in the points | no | yes | yes |
| transport | exact, a sort | exact, a sort ( or a grid ) | grid |
| size of the objects | none | `half_widths`, differentiable | `radii`, up to a static bound |

Growing or stretching the tents during a descent is not available yet: see the [roadmap](../roadmap).

## Optimizing the positions

The cost is a plain jax function of the points, so any optimizer does. `sdot.device_optim.lbfgs( value_and_grad, x0 )` is an
L-BFGS whose loop is **one compiled `lax.while_loop`**, with no round trip to the host between the steps; it returns
`( x, info )`, `info[ "losses" ]` holding the loss after each step, and can call back every few steps
( `snapshot_every`, `on_snapshot` ) to watch the descent. `gradient_descent` is its plain counterpart.

## Example: tomography with tents

A tomograph sees an unknown density $\rho$ of the plane only through its **projections**: for each angle $\theta_a$, the
profile $\nu_a$ of the mass of $\rho$ along the lines orthogonal to $n_a = ( \cos \theta_a, \sin \theta_a )$, measured on a
detector. To reconstruct $\rho$, we look for a cloud of $n$ tents, each carrying a mass $1/n$, whose projections reproduce the
measured profiles in the sense of the cost above, $L( x ) = \sum_a W_2^2( \mu_a( x ), \nu_a )$.

### The loss

The tents are a source, the profiles of all the angles a batched 1D `Image`, `ContinuousOtPlan1d` the transport between them,
and `jax.grad` differentiates it with respect to the points. The half-width is a parameter, since it changes during the
reconstruction:

<<< @/examples/tomography_tents.py#loss{python}

### Coarse to fine

L-BFGS on a large cloud of random tents ends in a poor local minimum. The remedy is a multiscale: converge on a few tents, split
each one into several children a little apart, converge again. The tents **narrow** along the way, with the spacing of the cloud,
down to half a bin of the detector: wide tents see the large scales only, and they smooth the loss where the points are still far
from their place. `sdot.device_optim.lbfgs` runs each descent on the device and calls back every few iterations.

A descent stops when the **points** have settled: `drift_tol` bounds the root mean square of their displacement over the last
`snapshot_every` iterations. It is the points that must converge, not the loss: the loss falls by 99 % in the first 30 iterations and
then creeps, while the tents keep sliding along the walls, where it is nearly flat, for hundreds of iterations more. Stopping on the
loss leaves them a few bins from their place. Neither more memory for L-BFGS, nor more levels, nor a ramp-filter preconditioner ( the
inverse of the smoothing that projections apply to a displacement, as in filtered back projection ) makes it faster: the slow
directions are local to the walls, where a filter that is the same everywhere does not reach them.

<<< @/examples/tomography_tents.py#multiscale{python}

### Stretching the tents

Once the points have settled, the shape of the tents is the next unknown: one half-width for all is a guess, and a round tent cannot
follow a wall that is not round. The [anisotropic tents](#anisotropic-tents) give each one a matrix, with
$M = L L^T / w + ( dw / 4 ) I$: symmetric and positive definite, the three entries of the lower triangular $L$ free ( divided by the
half-width $w$ of the last level so that $L$ is a length, like the points ). The points stay where they are: L-BFGS runs on the
matrices alone, starting from the round tents of the last level.

<<< @/examples/tomography_tents.py#matrices{python}

The problem is nearly convex in the matrices: the descent settles by itself in about 500 iterations, some twenty seconds. On this
object the loss goes from $2.0 \cdot 10^{-6}$ ( round tents of half-width 6.7 bins ) to $6.0 \cdot 10^{-7}$, a factor 3.4. The
median semi-axes are 5.0 and 7.4 bins: a typical tent is elongated by a factor 1.5, along the walls.

### A cylinder pierced by bubbles

The object is a cylinder of material pierced by 63 bubbles of various sizes, packed as tight as can be, with $1/120$ of the
diameter of the cylinder between two neighbours. It is the kind of data where the method pays: three quarters of the section are
empty, and the material left is a network of thin walls that a good reconstruction must follow without filling the bubbles. The data
are 600 profiles on a detector of 1024 bins.

The page shows the iterations recorded by the callback ( one in twenty ), from the 1024 tents thrown at random in the cylinder to the
4096 of the last level, then the matrices at fixed points ( play, or drag the slider ); each stage stops by itself when it has settled.
Each tent is drawn as a disk of its half-width, which shrinks from level to level, then as the outline of its ellipse in the last
stage.

<iframe
  :src="withBase('/examples/tomography-tents.html') + '?wheel=0&touch=0'"
  title="Tents, coarse to fine"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 70vh; min-height: 420px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/examples/tomography-tents.html')" target="_blank">Open in a separate page ↗</a></p>

The whole takes a little over a minute on a laptop once the kernels are compiled. The card and 3D from 2D projections are on the
[roadmap](../roadmap). The complete script is
[`docs/examples/tomography_tents.py`](https://github.com/sdot-team/sdot/blob/main/docs/examples/tomography_tents.py).
