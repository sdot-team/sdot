<script setup>
import { withBase } from 'vitepress'
</script>

# Ground Metrics

SDOT transports with the cost $c( x, y ) = |x - y|^2$. Three independent choices change it, each in its own place:

| | what changes | where it is stated |
|---|---|---|
| **The space** | $\mathbb{R}^d$, or its quotient by a group $G$ of isometries: a torus, a Klein bottle, a cone, a crystal... The cost becomes $\min_{g \in G} |x - g( y )|^2$ | `periodicity = ...`: see [Periodicity](./periodicity) |
| **The cost of a distance** | $\varphi( \| x - y \| )$, for all the diracs or per dirac ( radial functions around the diracs ) | [roadmap](../roadmap) |
| **The mass balance** | equal masses, or not: what is not transported goes to a global sink ( partial transport, cells capped by balls of radius $\sqrt{w_i}$ ) | `target_mass` of the distributions: see [Partial Transport](./partial) |

They compose: each one is stated on its own, and they apply together. A partial transport on a torus, where 30 diracs share
half of the unit square:

```python
f = sdot.SumOfDiracs( positions, target_mass = 0.5 )        # the mass balance: half of the square is transported
g = sdot.Box()

p = sdot.plan( f, g, periodicity = True )                   # the space: a torus
p.add_to_viz( viz )
```

Each dirac takes the part of its cell that lies in a ball around it, in full color; the rest of the cell, faded, goes to the sink.
The space is a torus, so the cells are drawn whole around their diracs, and again, fainter, moved by the translations of the
torus: a ball that crosses a side of the square ( dashed ) comes back on the other one.

<iframe
  :src="withBase('/guide/ground-metrics-axes.html') + '?wheel=0&touch=0'"
  title="A partial transport on a torus"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/ground-metrics-axes.html')" target="_blank">Open in a separate page ↗</a></p>

| Page | |
|---|---|
| [Periodicity](./periodicity) | quotients of $\mathbb{R}^d$ by a group of isometries: how to give the group, where the cells are, how to look at them |
| [Partial Transport](./partial) | a source lighter than the target: how to state it, what each dirac takes, how to use the result |

## Reproducing the pages

The figures of these pages are written by the scripts next to them, in
[`docs/guide/ground-metrics/`](https://github.com/sdot-team/sdot/blob/main/docs/guide/ground-metrics):

```bash
python docs/guide/ground-metrics/index_figures.py
python docs/guide/ground-metrics/periodicity_figures.py                 # all of them
python docs/guide/ground-metrics/partial_figures.py power masses        # or some of them
```
