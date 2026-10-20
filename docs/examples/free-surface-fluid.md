---
title: Free-Surface Fluid
description: An incompressible fluid with a free surface, its particles being cells of a partial transport (Gallouët-Mérigot scheme).
tags: [pde, physics, partial, 2d, 3d]
order: 8
---

<script setup>
import { withBase } from 'vitepress'
</script>

# Free-Surface Fluid

A fluid in 2D is cut into $n$ particles of fixed areas $m_i$. The matter held by particle $i$ is a cell of a
[partial transport](/guide/ground-metrics/partial) from the Diracs $p_i$ to the domain the fluid may fill. That cell is
the Laguerre cell of $p_i$ cut by the ball of radius $\sqrt{w_i}$. Every cell keeps its area $m_i$, so the fluid
is incompressible. Where the cells are crowded they tile the domain. At the edge of the fluid, they end on their
balls: this is the free surface.

The scheme is the one of Gallouët and Mérigot (*A Lagrangian scheme à la Brenier for the incompressible Euler equations*,
2018), with partial transport for the free surface, as in B. Lévy's free-surface version (2022). The Diracs carry
the matter and its velocity. The pressure is a spring that pulls each Dirac towards the barycenter $b_i$ of its cell, with
a stiffness $1 / \varepsilon^2$:

$$
\ddot p_i = g + \frac{b_i( p ) - p_i}{\varepsilon^2}.
$$

The spring derives from the potential $\sum_i m_i | p_i - b_i( p ) |^2 / ( 2 \varepsilon^2 )$: half the transport
cost, up to the factor $1 / \varepsilon^2$. Velocity Verlet (symplectic) integrates it, so the energy

$$
E = \sum_i m_i \Big( \tfrac12 | \dot p_i |^2 - g \cdot p_i + \frac{| p_i - b_i |^2}{2 \varepsilon^2} \Big)
$$

oscillates, but does not drift. Each step solves one transport, started from the weights of the previous step:

```python
problem = sdot.OtProblem( sdot.SumOfDiracs( p, masses, target_mass = None ), domain )
for step in range( nb_steps ):
    v += 0.5 * dt * a
    p += dt * v
    problem.source = sdot.SumOfDiracs( p, masses, target_mass = None )   # target_mass = None: the masses as given
    b = np.asarray( problem.solve().barycenters )
    a = g + ( b - p ) / eps**2
    v += 0.5 * dt * a
```

With `target_mass = None`, the source weighs the sum of its masses, which is less than the domain: the transport is
partial. The walls need nothing more. A barycenter stays inside the domain, so a Dirac that leaves it is pulled back in.
## The surface

As it is, nothing in this energy costs a VOID. With fixed masses and the air at weight 0, the partial transport is the
projection onto the densities $\le 1$ of crowd motion (Leclerc, Mérigot, Santambrogio, Stra, 2020): pressing costs,
parting does not. The particles bead up into balls, which leave gaps between them and against the walls, like a granular
material. A SURFACE ENERGY closes them: $\sigma \sum_i | S_i \cap V_i |$, the spheres inside the cells, where the free surface
is. A void has some, so the particles close it, and the free surface tightens. Its force, $- \sigma \nabla$, goes through the
transport: the masses stay as they are when a particle moves, so the weights move too. `sphere_areas`, an output of the plan,
is differentiable like the others:

```python
def area( p ):
    plan = sdot.OtProblem( sdot.SumOfDiracs( p, masses, target_mass = None ), domain ).solve()
    return jnp.sum( plan.sphere_areas ), plan.barycenters      # the length of the arcs in 2D, the area of the caps in 3D

( A, b ), grad_A = jax.value_and_grad( area, has_aux = True )( p )
a = g + ( b - p ) / eps**2 - sigma * grad_A / masses[ :, None ]
```

The force derives from an energy: $\sigma A$ is added to $E$. In 3D, with $\sigma = 0.02$, the faces against the walls stay
full, and the energy holds to about 1 %.

In 2D the length of the arcs does not do: where a circle touches a facet, the length of the arc grows like a square root
($dA / dh = 2 r / \rho$, against $2 \pi r$ for a cap in 3D), and these kicks feed the energy (+40 % on the dam break). The 2D
example counts instead the EMPTY part of each cell within a slightly bigger ball,
$\sum_i \big( | L_i \cap \Omega \cap B( p_i, \sqrt{ r_i^2 + c } ) | - m_i \big)$, $c = ( \bar r / 2 )^2$: zero for a cell full of
matter, positive next to a void or the free surface, and an area, smooth where a circle touches a facet. The walls do not
count: the integration stops at the domain. It is the same diagram with the sink at $- c$:

```python
wider = sdot.PowerDiagram( p, plan.weights, distribution = domain, sink_weight = - c )
A = jnp.sum( wider.measures ) - jnp.sum( masses )
```

With $\sigma = 0.6$ the energy of the dam break holds to 1.5 %, and in a column at rest the gaps against the walls go from
2 % of the surface to none.

The fluid starts AT REST, where the springs hold the forces: $p_i = b_i( p ) + \varepsilon^2 f_i$. A lattice is not such a state
(the particles of the free surface are not balls, and nothing holds the weight above), so the starting lattice is first
settled behind a gate (a box as wide as the column) by iterating $p \leftarrow b( p ) + \varepsilon^2 f$, $f$ the forces: Lloyd's
iterations, shifted by them. At $t = 0$ the gate is gone.

$\varepsilon$ sets how compressible the fluid is: the smaller it is, the stiffer the spring, and the time step, a tenth of
$\varepsilon$ here, follows it.

## Dam break

A column of 740 particles, 0.5 wide and 0.8 high, against the left wall of a $2 \times 1$ box, released at $t = 0$ (with
$g = 1$ and $\varepsilon = 0.02$).

<iframe
  :src="withBase('/examples/free-surface-dam-break.html') + '?wheel=0&touch=0'"
  title="Dam break"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/examples/free-surface-dam-break.html')" target="_blank">Open in a separate page ↗</a></p>

Each colored piece is what a particle holds: the part of its Laguerre cell inside its ball.

## In 3D

The same scheme, with a 3D partial transport: the cells are polyhedra cut by balls. The volumes, barycenters and
derivatives of these intersections are exact, as in 2D. A column of 200 particles, 0.5 wide, 0.8 high and 0.5 deep, in a
$2 \times 1 \times 0.5$ box as deep as the column, over the first 1.5 time units. Only the boundary of the matter is drawn: the spheres of the free
surface, and the faces against the walls.

<iframe
  :src="withBase('/examples/free-surface-dam-break-3d.html') + '?wheel=0&touch=0'"
  title="Dam break in 3D"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/examples/free-surface-dam-break-3d.html')" target="_blank">Open in a separate page ↗</a></p>

The pages are written by [`docs/examples/free_surface_fluid.py`](https://github.com/sdot-team/sdot/blob/main/docs/examples/free_surface_fluid.py).
