---
title: Linearized OT
description: Embed point clouds by their transport map from the Lebesgue measure of the unit square, learn a family of shapes by PCA of these maps, and identify new draws by projection.
tags: [learning, ot-plan, 2d, wasserstein]
order: 3
---

<script setup>
import { withBase } from 'vitepress'
</script>

# Linearized OT

We have many point clouds, each drawn from a member of a family of shapes. The aims are to **find the family back** from the clouds alone, and then to **say which member a new cloud comes from**, in the Wasserstein sense.

Point clouds are not vectors. Draw twice from the same shape, and the $i$-th point of the first cloud has nothing to do with the $i$-th point of the second: a PCA of the coordinates of the points, or of their displacements, means nothing. What the clouds do share is a **reference measure**, here the Lebesgue measure $\rho$ of the unit square. Each cloud $\mu_k = \frac{1}{n} \sum_i \delta_{y^k_i}$ is the image of $\rho$ by its optimal transport map $T_k$, and these maps all live in the same space, $L^2( \rho )$:

$$
\mu_k = ( T_k )_\# \rho, \qquad \| T_k - T_l \|_{L^2( \rho )} \;\ge\; W_2( \mu_k, \mu_l ).
$$

This is **linearized optimal transport** (Wang et al., 2013). With a semi-discrete transport, $T_k$ is constant on each power cell: it sends the cell of $y^k_i$ to $y^k_i$. Close to a reference measure that is smooth enough, the distance between the maps is close to $W_2$ (Mérigot, Delalande and Chazal, *Quantitative stability of optimal transport maps and linearization of the 2-Wasserstein space*, 2020). Every linear tool then works on the maps: the mean, the PCA, least squares.

## The map of a cloud

`sdot.plan` gives the weights $w_i$ of the power diagram, and a point $x$ of the square goes to the dirac of its cell, $\arg\min_i |x - y_i|^2 - w_i$. The map is read at 4096 points drawn uniformly in the square, once and for all: the same points for every cloud, so that the maps can be compared. That gives one vector of $\mathbb{R}^{4096 \times 2}$ per cloud, and the mean over these points stands for the integral against $\rho$.

```python
import numpy as np
import sdot

x = np.random.default_rng( 123 ).random( ( 4096, 2 ) )   # where every map is read

def transport_map( y ):
    """T( x ): the dirac of the power cell of each reading point"""
    w = np.asarray( sdot.plan( sdot.SumOfDiracs( y ), sdot.Box() ).weights )
    return y[ ( ( y * y ).sum( axis = 1 ) - w - 2 * x @ y.T ).argmin( axis = 1 ) ]
```

The family of the example is a **ring**, centered in the square. Its inner and outer boundaries each go from a circle ($s = 0$) to a square ($s = 1$), by interpolating the polar radius: two parameters, $s_\text{in}$ and $s_\text{out}$ in $[ 0, 1 ]$. Below are the power cells of four draws (of 1000 diracs), at the four corners of the family. Use the slider at the top of the viewer to go from one draw to the next.

<iframe
  :src="withBase('/examples/lot-cells.html') + '?wheel=0&touch=0'"
  title="The power cells of four draws"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/examples/lot-cells.html')" target="_blank">Open in a separate page ↗</a></p>

The star in the middle is where the map **tears**. The square has no hole and the ring has one, so no continuous map goes from one to the other. The center of the square is split between all the diracs of the inner boundary. This tear decides most of what follows.

## Learning the family

200 draws of $10^4$ diracs, with $( s_\text{in}, s_\text{out} )$ uniform in $[ 0, 1 ]^2$, give 200 maps. The parameters are not used to learn anything. The PCA of the maps:

```python
maps = np.stack( [ transport_map( y ).ravel() for y in draws ] )   # [ 200, 4096 * 2 ]
mean = maps.mean( axis = 0 )
_, sv, modes = np.linalg.svd( maps - mean, full_matrices = False )
coords = ( maps - mean ) @ modes[ :4 ].T
```

| mode                  | 1     | 2     | 3     | 4     | 5     | 6     |
| --------------------- | ----- | ----- | ----- | ----- | ----- | ----- |
| share of the variance | 0.616 | 0.081 | 0.066 | 0.058 | 0.016 | 0.012 |

The first mode is the outer boundary, $s_\text{out}$: it moves the most mass, the farthest. The inner boundary moves much less mass, and shares modes 2 to 4 with the tear, which the sampling noise moves from one draw to the next.

A point of the learned space is a measure: $\big( \bar T + \sum_k c_k v_k \big)_\# \rho$, the image of the reading points by the reconstructed map. Below, the first two coordinates, the others at 0. Far from the training draws, the reconstruction extrapolates and folds onto itself: it is no longer a transport map. The sliders therefore span the range of the training draws, shrunk towards the mean until the reconstruction folds nowhere. The dashed lines are the circles and squares the family moves between. $c_1$ turns the outside from a circle into a square, while $c_2$ mostly pushes the hole aside.

<iframe
  :src="withBase('/examples/lot-modes.html') + '?wheel=0&touch=0'"
  title="The learned modes"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/examples/lot-modes.html')" target="_blank">Open in a separate page ↗</a></p>

## Identifying a new draw

A new cloud is identified by its map. Its 4 coordinates $c = V^T ( T - \bar T )$ give the closest member of the learned space, $( \bar T + V c )_\# \rho$. The distance from $T$ to its projection is an upper bound of $W_2$ between the cloud and this member. To check the result against the truth, and only for that, an affine least squares fit of the training coordinates onto $( s_\text{in}, s_\text{out} )$ reads the coordinates as parameters.

On 10 new draws, the mean error is **0.039** on $s_\text{in}$ and **0.007** on $s_\text{out}$, and the distance to the learned space ranges from 0.0079 to 0.0104. Each frame shows the diracs of a draw (grey), its reconstruction (red), and its true boundaries:

<iframe
  :src="withBase('/examples/lot-identify.html') + '?wheel=0&touch=0'"
  title="Identifying new draws"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/examples/lot-identify.html')" target="_blank">Open in a separate page ↗</a></p>

<!-- ### How many diracs?

The tear makes the noise of the sampling large: $n$ must be large enough for $s_\text{in}$ to rise above it. Same experiment, with $n$ diracs per draw:

| $n$      | share of modes 1 to 4      | error on $s_\text{in}$ | error on $s_\text{out}$ |
| -------- | -------------------------- | ---------------------- | ----------------------- |
| 1000     | 0.270, 0.173, 0.125, 0.038 | 0.253                  | 0.039                   |
| 4000     | 0.495, 0.094, 0.080, 0.058 | 0.072                  | 0.018                   |
| $10^4$   | 0.616, 0.081, 0.066, 0.058 | 0.039                  | 0.007                   |

At $n = 1000$, $s_\text{in}$ is not found at all: 0.25 is the error of a guess at random. The outer boundary is found at every size. -->

## Outside of the family

A ring whose outer square is **turned by 45°** is not a member of the family. Its distance to the learned space, 0.057, is more than five times that of the test draws (at most 0.0104), while its unturned twin is at 0.0097. Reading its coordinates as parameters would give $( 0.68, -0.03 )$, a meaningless answer that the distance flags. Use the slider to compare the twin and the turned square:

<iframe
  :src="withBase('/examples/lot-outlier.html') + '?wheel=0&touch=0'"
  title="A ring outside of the family"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/examples/lot-outlier.html')" target="_blank">Open in a separate page ↗</a></p>

## The script

[`docs/examples/lot_dictionary.py`](https://github.com/sdot-team/sdot/blob/main/docs/examples/lot_dictionary.py) computes everything on this page. It is NumPy only, with one `sdot.plan` per draw, and takes about three minutes on a laptop.

```bash
python docs/examples/lot_dictionary.py           # every page and table
python docs/examples/lot_dictionary.py 1d cells  # the quick ones
```
