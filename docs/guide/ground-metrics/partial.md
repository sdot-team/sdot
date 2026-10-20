---
gallery: feature
description: "A source lighter than the target, the rest of the target going to a sink."
tags: [partial, lloyd, 2d]
image: /examples/img/partial-transport.png
order: 7
---

<script setup>
import { withBase } from 'vitepress'
</script>

# Partial Transport

When the source weighs less than the target, only part of the target is transported, and the rest goes to a **sink**. The masses
decide: each distribution is normalized to its `target_mass` ( 1 by default ).

```python
p = sdot.plan( sdot.SumOfDiracs( positions, target_mass = 0.4 ), sdot.Box() )     # 40 % of the box is transported
```

Two equal `target_mass` values balance exactly; otherwise the masses are compared up to a relative $10^{-12}$. With
`target_mass = None`, a distribution weighs what it holds: the sum of the weights of the diracs, or the integral of the density of
an image. For instance, 30 diracs of uneven masses summing to $0.45$ against the unit square take $45 \, \%$ of it, a heavy dirac
taking a bigger ball than a light one:

```python
masses = rng.uniform( 0.2, 1, size = 30 ) ** 2
masses *= 0.45 / masses.sum()
p = sdot.plan( sdot.SumOfDiracs( positions, masses, target_mass = None ), sdot.Box() )
```

<iframe
  :src="withBase('/guide/partial-own-masses.html') + '?wheel=0&touch=0'"
  title="30 diracs of uneven masses against the unit square"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/partial-own-masses.html')" target="_blank">Open in a separate page ↗</a></p>

## What each dirac takes

The sink has weight $0$: a point $x$ goes to dirac $i$ when $|x - p_i|^2 - w_i$ is the smallest power distance **and** is negative.
In 1D, with three diracs ( weights chosen by hand ): each point goes to the lowest of the parabolas, or to the sink when they are all
above $0$. The interval of a dirac stops either where another parabola gets lower ( a Laguerre facet ) or where its own parabola
crosses $0$, at a distance $\sqrt{w_i}$ from it:

![The power distances to three diracs, their minimum, the sink at 0, and who takes each point](/guide/img/partial-power.png)

The cell of $i$ is its Laguerre cell intersected with the ball $B_i$ of radius $\sqrt{w_i}$, and the cost is
$\sum_i \int_{L_i \cap B_i} |x - p_i|^2 \rho$.

## Visualization

The Laguerre cells are computed as usual. Only the integration is restricted to the balls, so `add_to_viz` draws the cells whole but
faded ( `opacity_beyond_balls` ), with what each dirac transports in full color on top:

```python
viz = sdot.Visualizer()
p.add_to_viz( viz )
viz.write_html( "partial.html" )
```

## The mass of the source

40 diracs on a density in bands, the same positions, a source whose mass goes from 0.15 to 1 ( the slider at the top of the
viewer ). At low mass the balls are disjoint, smaller where the density is high ( the same mass on a smaller area ). As the mass
grows they meet, and the Laguerre facets split them; at a mass of 1 the transport is balanced, and the cells fill the square.

<iframe
  :src="withBase('/guide/partial-masses.html') + '?wheel=0&touch=0'"
  title="Partial transport: the mass of the source, from 0.15 to 1"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/partial-masses.html')" target="_blank">Open in a separate page ↗</a></p>
<!-- 
## Lloyd

The plan reads as a balanced one: `p.weights` gives the radii of the balls, $\sqrt{w_i}$; `p.barycenters` are those of what each
dirac transports; the cost and its derivatives ( with respect to the positions, the masses of the diracs and the parameters of the
target ) are those of the partial transport, and `cost_and_position_grad` is still $2 m_i ( p_i - b_i )$. So the loops written for
the balanced transport work as they are.

[Lloyd's iterations](/examples/lloyd-quantization) with a source of mass 0.4 on an image with two bumps, of peak densities 1 and 0.7.
Each dirac moves to the barycenter of what it transports. The diracs gather where the density is high and pack there, while those
left in the low-density background keep wide, isolated balls.

```python
pb = sdot.OtProblem( sdot.SumOfDiracs( x, target_mass = 0.4 ), target )
for it in range( 40 ):
    pb.source = sdot.SumOfDiracs( x, target_mass = 0.4 )
    x = np.asarray( pb.solve().barycenters )
```

<iframe
  :src="withBase('/guide/partial-lloyd.html') + '?wheel=0&touch=0'"
  title="Partial transport: Lloyd"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/partial-lloyd.html')" target="_blank">Open in a separate page ↗</a></p> -->

## Under the hood

The cells are not changed: they stay the Laguerre cells, and it is the **integration** that is restricted to the balls, in closed
form ( a polygon inter a disc, for a density constant or affine on it: an `Image`, a `Box`, a `Mesh` ), with the derivatives. In 3D,
a polyhedron inter a ball, for a density constant on it: by the divergence theorem, the faces inter their discs ( the 2D closed
form, in the plane of each face ) and the solid angles of the sphere inside the cell ( `integration/InBall3.h` ). For Newton, the
sink is **one more seed**, whose target mass is what the
diracs leave of the target and whose weight, $0$, is the gauge: the solver and its linear solvers are those of the balanced
transport. The diagram takes it as `PowerDiagram( ..., sink_weight = 0 )`.

## What is not supported yet

* a source **heavier** than the target ( an explicit error ), and the penalized variant ( a sink of fixed weight );
* targets that are neither constant nor affine on pieces ( `SumOfGaussians`, a polynomial `Image` ), and in 3D the affine ones
  too ( `Mesh` ); the CUDA solve, the `"limits"` step.

The figures are written by [`docs/guide/ground-metrics/partial_figures.py`](https://github.com/sdot-team/sdot/blob/main/docs/guide/ground-metrics/partial_figures.py).
