---
title: Partial Transport
description: A source lighter than the target, the rest of the target going to a sink.
tags: [partial, 2d]
order: 7
---

<script setup>
import { withBase } from 'vitepress'
</script>

# Partial Transport

When the source weighs less than the target, only part of the target is transported, and the rest goes to a **sink**. The masses
decide: `target_mass` on the distributions ( 1 by default ).

```python
p = sdot.plan( sdot.SumOfDiracs( positions, target_mass = 0.4 ), sdot.Image( values ) )
```

The sink has weight $0$: a point $x$ goes to dirac $i$ when $|x - p_i|^2 - w_i$ is the smallest power distance **and** is negative.
The cell of $i$ is its Laguerre cell intersected with the ball of radius $\sqrt{w_i}$, and the cost is
$\sum_i \int_{L_i \cap B_i} |x - p_i|^2 \rho$. Details in [Guide: Ground Metrics](/guide/ground_metrics#partial-transport).

## Visualization

The Laguerre cells are computed as usual. Only the integration is restricted to the balls, so `add_to_viz` draws the cells whole but
faded ( `opacity_beyond_balls` ), with what each dirac transports in full color on top:

```python
viz = sdot.Visualizer()
p.add_to_viz( viz )
viz.write_html( "partial.html" )
```

## The mass of the source

30 diracs on the unit square, the same positions, a source whose mass goes from 0.1 to 0.99 ( the slider at the bottom of the
viewer ). At low mass the balls are disjoint discs of area $\nu_i$, each dirac at the center of its own. As the mass grows they meet,
and the Laguerre facets split them.

<iframe
  :src="withBase('/examples/partial-masses.html') + '?wheel=0&touch=0'"
  title="Partial transport: the mass of the source"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/examples/partial-masses.html')" target="_blank">Open in a separate page ↗</a></p>

## Lloyd

[Lloyd's iterations](./lloyd-quantization) with a source of mass 0.4 on an image with two bumps, of peak densities 1 and 0.7. Each
dirac moves to the barycenter of what it transports. The diracs gather where the density is high and pack there, while those left
in the low-density background keep wide, isolated balls.

```python
pb = sdot.OtProblem( sdot.SumOfDiracs( x, target_mass = 0.4 ), target )
for it in range( 40 ):
    pb.source = sdot.SumOfDiracs( x, target_mass = 0.4 )
    x = np.asarray( pb.solve().barycenters )
```

<iframe
  :src="withBase('/examples/partial-lloyd.html') + '?wheel=0&touch=0'"
  title="Partial transport: Lloyd"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/examples/partial-lloyd.html')" target="_blank">Open in a separate page ↗</a></p>

The pages are written by `docs/examples/partial_transport.py`.
