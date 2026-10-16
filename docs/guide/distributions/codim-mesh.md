<script setup>
import { withBase } from 'vitepress'
</script>

# Lines and Surfaces

A target can be carried by a mesh $M$ of **lower dimension** than the space: segments in 2D or 3D, triangles in 3D. The
measure is $\rho$ times the length ( or the area ) on $M$, with $\rho$ piecewise linear on its nodes. `Mesh` gives one as soon
as its simplices have fewer than $d + 1$ nodes:

```python
target = sdot.Mesh( nodes, triangles )                  # nodes [ n, 3 ], triangles [ m, 3 ]: a surface in 3D
target = sdot.Mesh( nodes, segments, values = rho )     # segments [ m, 2 ]: lines in 2D or 3D, rho at the nodes
```

$M$ is any simplicial complex: open or closed, with boundaries, junctions, and even self-intersections. Nothing is assumed about
its topology or its orientation.

## What is computed

The cells of the transport are integrated **exactly** on $M$: the mass of a cell $L_i$ is $\int_{L_i \cap M} \rho$, and its
barycenter, its derivatives and its Hessian are exact too.

A measure on $M$ is hard to start from, though. A cell that does not meet $M$ has no mass, and its weight gets no gradient, so
Newton cannot move it. The solver goes through smoothed densities first,

$$
\rho( p( x ) ) \, e^{ -d( x )^2 / s^2 },
$$

with $d$ the distance to $M$ and $p( x )$ the closest point of $M$. It starts from a width $s$ of the size of the domain, then
narrows it stage by stage down to `width`, and ends with the measure on $M$ itself. Each stage starts from the weights of the
previous one.

These densities are integrated in closed form, on the **envelope** of $M$: the generalized Voronoi diagram of the elements of
$M$ ( its simplices, their edges, their nodes ). It is cut into simplices on which $d$ and $\rho$ are affine. Around the edges and
the nodes, where $d$ is not affine, the envelope uses fans of sectors at most `alpha` wide, and over-estimates $d$ by at most
$1 / \cos( \alpha / 2 )$. Along $M$ itself, $d$ is exact, so the smoothed densities do tend to the measure on $M$.

When the starting point already gives a mass to every cell ( diracs on $M$ ), the solver skips the smoothed stages.

## Parameters

```python
sdot.Mesh( nodes, simplices, values = None, width = None, alpha = 15, margin = 0.25, smoothed = False )
```

- `values`: $\rho$ at the nodes, ones by default;
- `width`: the narrowest smoothed density, before the measure itself. By default, $10^{-5}$ times the diagonal of $M$. Wider,
  some cells of the last smoothed stage stay in layers along $M$ without crossing it, and they would be empty for the measure;
- `alpha`: the widest angle of the fans of the envelope, in degrees;
- `margin`: the domain of the smoothed densities is the box of $M$, widened by `margin` times its diagonal;
- `smoothed = True`: the target is the smoothed density at `width` rather than the measure on $M$.

The envelope is built once, with the target. In 3D it is the largest cost: on a laptop, 3 s for a torus of 1000 triangles,
25 s for 4000, and 45 s for the 2880 triangles of the bottle below, whose sheets come close to each other.

## Quantization of a Klein bottle

`sdot.quantization` works on these targets as on any other one ( see [Lloyd quantization](/examples/lloyd-quantization) ).
Given a number of diracs, it draws them on $M$, by mass: the first transport then goes directly to the measure on $M$.

The surface is the bottle immersion of the Klein bottle in $\mathbb{R}^3$, with a neck a little wider than the classical one. It
crosses itself where the neck enters the body, and it has no inside nor outside. 6000 diracs:

```python
nodes, triangles = klein_bottle()                       # 2880 triangles, see the script below
target = sdot.Mesh( nodes, triangles )

x = sdot.quantization( target, 6000, seed = 0 )
```

<iframe
  :src="withBase('/guide/codim-mesh-klein.html') + '?wheel=0&touch=0'"
  title="Lloyd quantization of a Klein bottle"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/codim-mesh-klein.html')" target="_blank">Open in a separate page ↗</a></p>

The diracs end at the barycenters of their cells: slightly inside the curved parts of the surface, by the sagitta of a cell.

## A density on the surface

The same bottle, with $\rho$ growing from 0.1 at the bottom to 1 at the top of the neck: the diracs follow the density.

```python
h = ( nodes[ :, 1 ] - nodes[ :, 1 ].min() ) / np.ptp( nodes[ :, 1 ] )     # y is the height
target = sdot.Mesh( nodes, triangles, values = 0.1 + 0.9 * h ** 2 )

x = sdot.quantization( target, 6000, seed = 0 )
```

<iframe
  :src="withBase('/guide/codim-mesh-klein-rho.html') + '?wheel=0&touch=0'"
  title="Lloyd quantization of a Klein bottle with a varying density"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/codim-mesh-klein-rho.html')" target="_blank">Open in a separate page ↗</a></p>

## Reproducing the pages

The viewers are written by [`docs/guide/distributions/codim_mesh_figures.py`](https://github.com/sdot-team/sdot/blob/main/docs/guide/distributions/codim_mesh_figures.py),
which also builds the bottle:

```bash
python docs/guide/distributions/codim_mesh_figures.py                  # both
python docs/guide/distributions/codim_mesh_figures.py klein-rho        # or one of them
```

## Limits

- CPU only for now.
- The smoothed stages are much slower than the measure on $M$: a first solve from diracs far from $M$ can take minutes on a few
  thousand triangles. Starting on $M$ avoids them.
- Points ( a codimension equal to the dimension ) are not meshes: use `SumOfGaussians` for them.
