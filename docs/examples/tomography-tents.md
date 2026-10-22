---
title: Tomography with Tents
description: Reconstruct a density from its projections with a cloud of tents, coarse to fine, with L-BFGS — on a cylinder pierced by packed bubbles.
tags: [tomography, 2d, jax]
image: /examples/img/reconstruction.png
order: 30
---

<script setup>
import { withBase } from 'vitepress'
</script>

# Tomography with Tents

## Reconstructing from projections

A tomograph sees an unknown density $\rho$ of the plane only through its **projections**: for each angle $\theta_a$, the
profile $\nu_a$ of the mass of $\rho$ along the lines orthogonal to $n_a = ( \cos \theta_a, \sin \theta_a )$, measured on a
detector. To reconstruct $\rho$, we look for a cloud of $n$ points $x_1, \dots, x_n$ of the plane, each carrying a mass
$1/n$, whose projections reproduce the measured profiles. A point $x_i$ lands at $x_i \cdot n_a$ on the detector of angle $a$,
and what "reproduce" means is a transport: the $W_2$ distance on the line,

$$
L( x ) = \sum_a W_2^2\big( \mu_a( x ), \nu_a \big), \qquad \mu_a( x ) = \tfrac{1}{n} \sum_i f( \,\cdot - x_i \cdot n_a ).
$$

## What a point projects to

If $f$ is a dirac, $\mu_a$ is a sum of diracs on the line, and $L$ has a kink wherever two projected points cross. We take
instead a **tent** of half-width $w$, $f( s ) = ( 1 - \lvert s \rvert / w )_+ / w$: two tents that cross only overlap, and $L$ is
smooth in $x$. The transport between the tents and the profile of one angle is exact; the transports of all the angles are one
batch. This is `sdot.ProjectedSumOfTents` against a 1D `sdot.Image`, transported by `sdot.ContinuousOtPlan1d`
( see [Projected Sources](/guide/distributions/projected) ). The projected positions $x_i \cdot n_a$ are computed on the fly:
the `[ angles, n ]` array is never built, which is what lets $n$ grow to millions.

## The loss

The loss is plain jax: the tents are a source, the profiles of all the angles a batched 1D `Image`, `ContinuousOtPlan1d` the
transport between them, and `jax.grad` differentiates it with respect to the points. The half-width is a parameter, since it
changes during the reconstruction:

<<< @/examples/tomography_tents.py#loss{python}

## Optimizing: coarse to fine

L-BFGS on a large cloud of random tents ends in a poor local minimum. The remedy is a multiscale: converge on a few tents, split
each one into several children a little apart, converge again. The tents **narrow** along the way, with the spacing of the cloud,
down to half a bin of the detector: wide tents see the large scales only, and they smooth the loss where the points are still far
from their place. `sdot.device_optim.lbfgs` runs a whole descent on the device, as one compiled loop, and calls back every few
iterations. A descent stops when the **points** have settled: `drift_tol` bounds the root mean square of their displacement over the last
`snapshot_every` iterations. It is the points that must converge, not the loss: the loss falls by 99 % in the first 30 iterations and
then creeps, while the tents keep sliding along the walls, where it is nearly flat, for hundreds of iterations more. Stopping on the
loss leaves them a few bins from their place. Neither more memory for L-BFGS, nor more levels, nor a ramp-filter preconditioner ( the
inverse of the smoothing that projections apply to a displacement, as in filtered back projection ) makes it faster: the slow
directions are local to the walls, where a filter that is the same everywhere does not reach them.

<<< @/examples/tomography_tents.py#multiscale{python}

## Stretching the tents

Once the points have settled, the shape of the tents is the next unknown: one half-width for all is a guess, and a round tent cannot
follow a wall that is not round. Let each tent have a **matrix**: `sdot.ProjectedSumOfAnisotropicTents` gives it, at the direction
$n$, the half-width $\lvert M^T n \rvert$, the shadow of the ellipse $M ( \text{disk} )$ ( see
[Projected Sources](/guide/distributions/projected#anisotropic-tents) ). $M = L L^T / w + ( dw / 4 ) I$ keeps it symmetric and
positive definite, with the three entries of the lower triangular $L$ free ( divided by the half-width $w$ of the last level so that
$L$ is a length, like the points ). The points stay where they are: L-BFGS runs on the matrices alone, starting from the round tents
of the last level.

<<< @/examples/tomography_tents.py#matrices{python}

The problem is nearly convex in the matrices: the descent settles by itself in about 500 iterations, some twenty seconds. On this
object the loss goes from $2.0 \cdot 10^{-6}$ ( round tents of half-width 6.7 bins ) to $6.0 \cdot 10^{-7}$, a factor 3.4. The
median semi-axes are 5.0 and 7.4 bins: a typical tent is elongated by a factor 1.5, along the walls.

## A cylinder pierced by bubbles

The object is a cylinder of material pierced by 63 bubbles of various sizes, packed as tight as can be, with $1/120$ of the
diameter of the cylinder between two neighbours. It is the kind of data where the method pays: three quarters of the section are
empty, and the material left is a network of thin walls that a good reconstruction must follow without filling the bubbles. The data
are 600 profiles on a detector of 1024 bins.

The page shows the iterations recorded by the callback ( one in twenty ), from the 1024 tents thrown at random in the cylinder to the
4096 of the last level, then the matrices at fixed points ( play, or drag the slider ); each stage stops by itself when it has settled. Each tent is drawn as a disk of its half-width, which shrinks from level to level, then as the outline of its ellipse in the last stage.

<iframe
  :src="withBase('/examples/tomography-tents.html') + '?wheel=0&touch=0'"
  title="Tents, coarse to fine"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 70vh; min-height: 420px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/examples/tomography-tents.html')" target="_blank">Open in a separate page ↗</a></p>

The whole takes a little over a minute on a laptop once the kernels are compiled. The card and 3D from 2D projections are on the [roadmap](/guide/roadmap).

The complete script is
[`docs/examples/tomography_tents.py`](https://github.com/sdot-team/sdot/blob/main/docs/examples/tomography_tents.py).
