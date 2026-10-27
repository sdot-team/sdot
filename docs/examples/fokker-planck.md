---
title: Fokker-Planck Diffusion
description: Linear diffusion with particles, the entropy being replaced by its Moreau-Yosida regularization (Leclerc-Mérigot-Santambrogio-Stra scheme).
tags: [pde, diffusion, gradient-flow, 2d]
image: /examples/img/fokker-planck.png
order: 9
---

<script setup>
import { withBase } from 'vitepress'
</script>

# Fokker-Planck Diffusion

The Fokker-Planck equation $\partial_t \rho = \Delta \rho + \operatorname{div}( \rho \nabla V )$ is the Wasserstein gradient flow
of the free energy

$$
E( \rho ) = \int V \, d\rho + \int \rho \log \rho .
$$

A Lagrangian discretization moves $n$ particles $x_i$ of mass $1 / n$. The entropy of a sum of Diracs is $+\infty$, so the
scheme of L., Mérigot, Santambrogio and Stra (*Lagrangian discretization of crowd motion and linear diffusion*, SIAM J.
Numer. Anal., 2020, [arXiv:1905.08507](https://arxiv.org/abs/1905.08507)) replaces it by its Moreau-Yosida regularization:

$$
H_\varepsilon( \mu ) = \min_\sigma \frac{1}{2 \varepsilon} W_2^2( \sigma, \mu ) + \int \sigma \log \sigma .
$$

The optimal $\sigma$ is a power diagram whose cells carry a gaussian centered on their own particle:

$$
\sigma( x ) = e^{ ( \psi_i - | x - x_i |^2 ) / ( 2 \varepsilon ) } \quad \text{on the Laguerre cell } L_i( \psi ),
\qquad \text{where the weights } \psi \text{ give } \int_{L_i} \sigma = \frac1n \text{ for every } i .
$$

This is a semi-discrete transport where each cell integrates a **radial function of its seed**,
`radial_func = sdot.RadialExp( eps )`. The gradient of $H_\varepsilon$ with respect to $x_i$ is $( x_i - b_i ) / ( n \varepsilon )$, where
$b_i$ is the barycenter of $\sigma$ on $L_i$. The ODE $\dot x_i = ( b_i - x_i ) / \varepsilon - \nabla V( x_i )$ is integrated
with explicit Euler steps.

```python
problem = sdot.OtProblem( sdot.SumOfDiracs( x ), domain, radial_func = sdot.RadialExp( eps ) )
for step in range( nb_steps ):
    problem.source = sdot.SumOfDiracs( x )           # every cell gets the mass 1 / n
    b = np.asarray( problem.solve().barycenters )    # the barycenters of sigma
    x += tau * ( ( b - x ) / eps - grad_V( x ) )
```

With a radial function the masses no longer sum to a constant: raising every weight by $c$ multiplies $\sigma$ by
$e^{ c / ( 2 \varepsilon ) }$. There is no gauge to fix, and $\partial m_i / \partial \psi_i$ has one more term,
$m_i / ( 2 \varepsilon )$, beyond the facets. The solver's Newton sees it as a sink of fixed weight: a ground node linked to each
cell by that coefficient (`sdotplan/Sweep.h`). The start, the Voronoi diagram or the weights of the previous step, is raised so
that $\sigma$ weighs 1. The plan is differentiable, as any plan: positions, masses, values of the density. The cells and their
integrals alone are `sdot.PowerDiagram( x, psi, distribution = domain, radial_func = sdot.RadialExp( eps ) )`.

## Heat equation

$V = 0$. The particles start on $h \mathbb{Z}^2$ inside the disk of radius $1/2$, in the box $[ -1.5, 1.5 ]^2$, with
$h = 1/30$ (one of the two discretizations of figure 6 of the paper), $\varepsilon = 8 h^2$ and $\tau = \varepsilon / 4$. The color is $\sigma$, scaled by its
maximum at each time.

<iframe
  :src="withBase('/examples/fokker-planck-heat.html') + '?wheel=0&touch=0'"
  title="Heat equation"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/examples/fokker-planck-heat.html')" target="_blank">Open in a separate page ↗</a></p>

Inside the disk the cells are squares and $\sigma$ is almost flat: the particles do not move. The cells on the edge of the disk
reach the walls, and $\sigma$ goes out along them, as far as $\sqrt{\varepsilon}$ from the particles. Their barycenters lie
outside, which pulls the edge outward. Each particle then has room behind it, and the motion spreads inward.

The heat flow from a uniform disk is known: $\rho_t = \rho_0 * \mathcal{N}( 0, 2 t )$. Its second moment grows as $r^2 / 2 + 4 t$.
On the left, the law of the distance to the center, particles against the exact flow; on the right, the second moment, for
$h = 1/30$ and $h = 1/80$:

<img :src="withBase('/examples/img/fokker-planck-heat.png')" alt="Particles against the exact heat flow" style="width: 100%; margin: 1rem 0;" />

## The choice of ε

$\varepsilon$ is compared to the square of the spacing $a$ of the particles. Two errors compete:

- if $\varepsilon$ is of the order of $a^2$ or smaller, the gaussian of a cell is not flat over it, and the motion is slowed;
- if $\varepsilon$ is not small compared to the square of the scale of $\rho$, $\sigma$ is a blurred version of $\rho$.

The slope of the second moment ($4$ for the heat flow) measures them. Between $t = 0.025$ and $t = 0.05$ (`study` in the script):

| | $\varepsilon = 2 h^2$ | $\varepsilon = 4 h^2$ | $\varepsilon = 8 h^2$ | $\varepsilon = 16 h^2$ |
|---|---|---|---|---|
| $h = 1/30$ (697 particles) | 3.39 | 3.56 | 3.60 | 3.48 |
| $h = 1/50$ (1941 particles) | 3.43 | 3.63 | 3.73 | 3.73 |
| $h = 1/80$ (5013 particles) | | 3.66 | 3.78 | 3.83 |

The slope approaches 4 as $h$ decreases: the gap is roughly proportional to $h$. The spacing that matters is the local one, which grows
as $\rho$ spreads: at $\varepsilon = 2 h^2$ the slope drops over time (3.56 in the first fifth of the run, 3.39 at the end for
$h = 1/30$), while at $\varepsilon = 16 h^2$ it rises (the blur fades as $\rho$ widens).

The time step plays almost no part: $\tau = \varepsilon / 10$ and $\tau = \varepsilon$ give the same slopes as $\tau = \varepsilon / 4$.
Each step takes about three Newton iterations ( to an absolute tolerance of $10^{-12}$ on the masses ), started from the weights of
the previous step.

## With a potential

$V( x ) = | x |^2 / ( 2 s^2 )$ with $s = 1/4$: the equilibrium is the gaussian $e^{ - V }$, and the flow is that of an
Ornstein-Uhlenbeck process, whose mean and second moment are known. The particles start on a disk of radius $0.3$ centered at
$( 0.45, 0 )$, with $h = 1/40$, $\varepsilon = 8 h^2$ and $\tau = \varepsilon / 4$.

<iframe
  :src="withBase('/examples/fokker-planck-drift.html') + '?wheel=0&touch=0'"
  title="Fokker-Planck with a quadratic potential"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/examples/fokker-planck-drift.html')" target="_blank">Open in a separate page ↗</a></p>

<img :src="withBase('/examples/img/fokker-planck-drift.png')" alt="Mean and second moment against the Ornstein-Uhlenbeck process" style="width: 100%; margin: 1rem 0;" />

The mean follows the exact decay $e^{ - t / s^2 }$. The second moment settles about 13 % below $2 s^2$: the variance at
equilibrium is proportional to the diffusion coefficient, and the diffusion is slowed as for the heat equation. At equilibrium the
437 particles are about $1/30$ apart, as in the first row of the table above.

## Limits

`RadialExp` works in 2D, with a density constant on pieces (`Box`, `Image` of degree 0, `ConvexHull`), on the CPU, without a
periodicity or a sink. See the [roadmap](/guide/roadmap).

The pages are written by [`docs/examples/fokker_planck.py`](https://github.com/sdot-team/sdot/blob/main/docs/examples/fokker_planck.py).
