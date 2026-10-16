# Ground Metrics

SDOT transports with the cost $c( x, y ) = |x - y|^2$. Three independent choices change it, each in its own place:

| | what changes | where it is stated |
|---|---|---|
| **The space** | $\mathbb{R}^d$, or its quotient by a group $G$ of isometries: a torus, a Klein bottle, a cone, a crystal... The cost becomes $\min_{g \in G} |x - g( y )|^2$ | `periodicity = ...` |
| **The cost of a distance** | $\varphi( |x - y| )$, for all the diracs or per dirac ( radial functions around the diracs ) | [roadmap](./roadmap) |
| **The mass balance** | equal masses, or not: what is not transported goes to a global sink ( partial transport, cells capped by balls of radius $\sqrt{w_i}$ ) | `target_mass` of the distributions |

They compose. A periodic space keeps whatever cost and mass balance come with it.

## Periodicity

`periodicity` describes a **group of isometries** $G$, and the transport takes place on the quotient space $\mathbb{R}^d / G$:
two points are the same when an element of $G$ sends one onto the other, and the cost is the squared distance to the nearest image,
$c( x, y ) = \min_{g \in G} | x - g( y ) |^2$. It is accepted by `plan`, `w2`, `ot_solve`, `OtProblem` and `PowerDiagram`.

The group is given by its **generators**, each an isometry $x \mapsto M x + V$ with $M$ orthogonal ( the cost would not be
invariant otherwise ). Their inverses and their compositions are added: give the generators, not the group.

| a generator | the map |
|---|---|
| a vector `V` | the translation $x \mapsto x + V$ |
| a pair `( M, V )` | $x \mapsto M x + V$: a rotation, a symmetry, a glide, a screw... |
| a $( d + 1 )^2$ homogeneous matrix | the same, as one matrix |

```python
import numpy as np
import sdot

flip, quarter = np.diag( [ 1, -1 ] ), np.array( [ [ 0, -1 ], [ 1, 0 ] ] )
f, g = sdot.SumOfDiracs( positions ), sdot.Box()

sdot.plan( f, g, periodicity = [ [ 1, 0 ], [ 0, 1 ] ] )                 # a torus
sdot.plan( f, g, periodicity = [ ( flip, [ 1, 1 ] ), [ 0, 1 ] ] )       # a Klein bottle: across x, the square comes back upside down
sdot.plan( f, g, periodicity = [ ( flip, [ 1, 1 ] ) ] )                 # a Moebius strip ( its other two sides stay walls )
sdot.plan( f, g, periodicity = [ ( quarter, [ 0, 0 ] ) ] )              # a cone: a quarter turn about a corner
sdot.plan( f, g, periodicity = [ [ 1, 1 ] ] )                           # a cylinder along any direction
```

Two shortcuts read the support of the target:

* `True`, or one boolean per axis ( `[ True, False ]`: a cylinder along $x$ ): one translation per side of the support. This needs an
  **axis-aligned box**. For any other domain, or a cylinder along any other direction, give the generators: translations, or
  `( M, V )` pairs and matrices, glue any domain;
* `target.domain`: the half-spaces of the support, whose opposite parallel facets are paired by translations.

The **target** is repeated by the group: its support must lie in a fundamental domain, so that its images do not overlap ( checked
for a lattice of translations ). Its sides that the group does not glue stay walls: those of a cylinder, a Moebius strip, a cone. The
**diracs** can lie anywhere, and each cell is the whole cell of its dirac **as given**: not cut at the sides of the domain, possibly
outside it, away from the others. The barycenters and the moments are those of these cells, so `2 m ( p - b )`
( `cost_and_position_grad` ) stays continuous when a dirac crosses a side. `PowerDiagram.add_to_viz( viz )` draws the
cells with their copies by the first ring of the group ( `nb_periodicity_rings`, `periodicity_rings` ), and the domain dashed: see the [example](/examples/periodic-metrics#looking-at-a-quotient).

Everything is differentiable as without periodicity: with respect to the positions, the masses of the diracs and the parameters of the
target. The [example](/examples/periodic-metrics) shows the cells on a torus, a Klein bottle, a Moebius strip, a cone, a wallpaper
group, a hexagonal lattice and a turned cylinder.

### How it works

There is one cell per dirac. It is built around its dirac, in the plane, and is not cut at the sides of the support. A cell cannot
leave the Voronoi cell of its dirac among its own images, since they all have the same weight.

Internally, the diracs are first **folded** onto the support by the group ( `sdot.wrap` does the same thing by hand ). This changes
nothing in what is computed, since a dirac and its images are the same point of the quotient, and the cells come back next to the
diracs as given ( an isometry ). What it buys is compactness. The differences between the diracs stay within a period, so the images
that can cut a cell are a short, fixed list for all the cells: 9 translations on a 2D torus, however far apart the given diracs are.
Each cell then starts from one box, the support widened by half a period. Without folding, the list of images would have to cover the
spread of the diracs, and so would that box. For the translations of the group, a
lattice, this bounds the cells, whatever the weights: the cells start from the support widened by half a period ( exactly half a
period for a rectangular lattice ).

The planes come from the diracs and from their images by the elements of the group that come close to the support. The tree of the
diracs is traversed once per element, with its boxes moved. The density is integrated as the sum of its images, each cutting the cell
into its own pieces. Cost: in 2D, a torus traverses the tree 9 times instead of once, but the pruning drops most of the traversals right away.

### On a GPU

The diagrams ( `PowerDiagram`: measures, moments, their derivatives ) work on a CUDA device for any group. The **solve** on a CUDA device
runs on the card's own cells, which walk the images of a group of **translations** ( a torus, a cylinder, a sheared lattice ) with
a constant density ( a `Box`, a `Polytope`, an `Image` of equal values ), in 2D and 3D. A group with rotations or symmetries, or a
density that is not a constant, is solved on the CPU ( `LOOM_DEVICE=cpu` ).

### What is not supported yet

* targets other than `Image`, `Box` and `Polytope` ( `SumOfGaussians`, `Mesh` );
* 1D ( the circle is another algorithm ), the `"limits"` Newton step ( `"trials"` is used ), the width continuation of an image
  ( the solve starts from the sharp image ).

## Partial transport

The **masses of the distributions** decide whether the transport is partial: each one is normalized to its `target_mass` ( 1 by
default, `None` for its own mass ). When the source is lighter than the target, the diracs take what is nearest to them, and the rest
of the target goes to a **sink**:

```python
sdot.plan( sdot.SumOfDiracs( positions, target_mass = 0.4 ), sdot.Box() )     # 40 % of the box is transported
```

The sink is a seed of weight $0$ that is at the same power distance from every point, so a point $x$ goes to dirac $i$ when
$|x - p_i|^2 - w_i$ is the smallest power distance and is negative. The cell of $i$ is then its Laguerre cell intersected with the
ball $B_i$ of radius $\sqrt{w_i}$, and the cost is $\sum_i \int_{L_i \cap B_i} |x - p_i|^2 \rho$. Two equal `target_mass` values
balance exactly; otherwise the masses are compared up to a relative $10^{-12}$.

The barycenters, the cost, the weights and their derivatives ( with respect to the positions, the masses of the diracs and the
parameters of the target ) are those of the partial transport. `cost_and_position_grad` is still $2 m_i ( p_i - b_i )$. The
[example](/examples/partial-transport) shows the balls as the mass of the source grows, and Lloyd's iterations on two bumps.

### How it works

The cells are not changed: they stay the Laguerre cells, and it is the **integration** that is restricted to the balls. A
distribution still cuts a cell into its pieces of constant density. Each piece crossing the sphere is then integrated against
$\rho \, 1_{B_i}$ in closed form ( a polygon inter a disc: triangles and circular sectors ), with its facets, its moments and their
adjoints. The diagram takes it as `PowerDiagram( ..., sink_weight = 0 )` ( `integration/InBall.h` ). Other integrands, such as the
radial costs of the roadmap, take the same place.

For Newton, the sink is **one more seed**, whose target mass is what the diracs leave of the target and whose weight, $0$, is the
gauge. Its facet with dirac $i$ is the arc of the sphere inside the cell, $\int_{\partial B_i \cap L_i} \rho / ( 2 \sqrt{w_i} )$.
The Hessian stays a graph Laplacian, and the solver and its linear solvers are those of the balanced transport. The start is a ball
per dirac, of half its target mass.

### What is not supported yet

* a source **heavier** than the target ( an explicit error ), and the penalized variant ( a sink of fixed weight );
* 3D, targets that are not constant on pieces ( `SumOfGaussians`, `Mesh` ), a periodicity, the CUDA solve, the `"limits"` step.
