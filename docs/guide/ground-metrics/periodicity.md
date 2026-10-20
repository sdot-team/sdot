---
gallery: feature
description: "OT on quotient spaces — a torus, a Klein bottle, a cone... — from any group of isometries."
tags: [geometry, periodic, lloyd, 2d]
image: /examples/img/periodic-metrics.png
order: 6
---

<script setup>
import { withBase } from 'vitepress'
</script>

# Periodicity

Given $G$ a group of isometries, the `periodicity` parameter takes the cost $c( x, y ) = \min_{g \in G} | x - g( y ) |^2$: the
transport takes place on the quotient space $\mathbb{R}^d / G$. It is accepted by `plan`, `w2`, `ot_solve`, `OtProblem` and
`PowerDiagram`.

For instance, to compute a plan inside a Klein bottle:

```python
flip = np.diag( [ 1, -1 ] )
p = sdot.plan( sdot.SumOfDiracs( positions ), sdot.Box(),
    periodicity = [
        ( flip, [ 1, 1 ] ),
        [ 0, 1 ]
    ]
)
```

## The group

`periodicity` lists the **generators** of the group: their inverses and compositions are added. Each one is an isometry
$x \mapsto M x + V$, $M$ orthogonal:

| a generator | the map |
|---|---|
| a vector `V` | the translation $x \mapsto x + V$ |
| a pair `( M, V )` | $x \mapsto M x + V$: a rotation, a symmetry, a glide, a screw... |
| a $( d + 1 )^2$ homogeneous matrix | the same, as one matrix |

```python
flip, quarter = np.diag( [ 1, -1 ] ), np.array( [ [ 0, -1 ], [ 1, 0 ] ] )
f, g = sdot.SumOfDiracs( positions ), sdot.Box()

sdot.plan( f, g, periodicity = [ [ 1, 0 ], [ 0, 1 ] ] )                 # a torus
sdot.plan( f, g, periodicity = [ ( flip, [ 1, 1 ] ), [ 0, 1 ] ] )       # a Klein bottle
sdot.plan( f, g, periodicity = [ ( flip, [ 1, 1 ] ) ] )                 # a Moebius strip ( its other two sides stay walls )
sdot.plan( f, g, periodicity = [ ( quarter, [ 0, 0 ] ) ] )              # a cone: a quarter turn about a corner
sdot.plan( f, g, periodicity = [ True, False ] )                        # a cylinder along x
sdot.plan( f, g, periodicity = [ [ 1, 1 ] ] )                           # a cylinder along any direction
```

Two sides with the same arrows are glued, the arrows matching; a grey side is a wall:

![The sides of the unit square glued by a torus, a Klein bottle, a Moebius strip, a cone and a cylinder](/guide/img/periodicity-gluings.png)

The target is the content of a **fundamental domain**: its images by the group must not overlap ( checked for a lattice of
translations ). Its sides that the group does not glue stay walls.

Two shortcuts read the support of the target:

* `True`, or one boolean per axis ( `[ True, False ]`: a cylinder along $x$ ): one translation per side of an **axis-aligned box**;
* `target.domain`: the opposite parallel facets of the support, paired by translations.

## Visualization

A cell is computed whole, around its dirac as given. To check a quotient by eye, draw the cells, then their copies by the group
around the identity, fainter: the copies must glue without a seam. The domain is dashed.

```python
p = sdot.plan( sdot.SumOfDiracs( positions ), sdot.Box(), periodicity = True )

viz = sdot.Visualizer()
p.add_to_viz( viz )    # the cells, then their copies by the first ring of the group, fainter

# more rings, more visible, and counted in the zoom of the page
p.add_to_viz( viz, nb_periodicity_rings = 2, opacity_periodicity_rings = 0.5, fit_periodicity_rings = True )
viz.write_html( "torus.html" )
```

The same by hand, with the pieces it is made of:

```python
pd = p.power_diagram()
for k, a in enumerate( pd.periodicity_rings( 1 ) ):   # the elements of the first ring, as affine maps [ M | V ], the identity first
    pd.cells.moved( a ).add_to_viz( viz, opacity = 1.0 if k == 0 else 0.25, frames = k == 0 )   # ( the copies out of the zoom )
```

`periodicity_rings( k )` gives the elements of the group up to the ring `k`: ring 1 holds the elements whose image of the domain
touches the domain ( the 8 neighbours of a square on a torus ), ring 2 those touching ring 1, and so on. `Cell.moved` moves a cell by
one of them, or each cell by its own.

## Walls, then a torus

24 diracs with random masses on the unit square. Move the slider at the top of the viewer: `periodic = 0` is the transport with
walls, its cells moved by hand by the elements of the torus's first ring; `periodic = 1` is the transport on the torus.

<iframe
  :src="withBase('/guide/periodicity-walls.html') + '?wheel=0&touch=0'"
  title="Walls, then a torus"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/periodicity-walls.html')" target="_blank">Open in a separate page ↗</a></p>

With walls, the copies do not match along the sides: the cells are cut by the box, and a dirac near a side has no neighbour beyond it.
On the torus the copies glue without a seam, and every cell is surrounded on all sides: there is no boundary effect left.

## Cells are where their diracs are

The diracs can lie anywhere. Below, three of them are given outside the domain, one period away or more, and one has a large mass.
With `ring = 0` ( `nb_periodicity_rings = 0` ) the cells are shown alone: those of the three diracs lie outside the domain, away from
the others, and the big one crosses a side. With `ring = 1` their copies fill the gaps, and the quotient appears.

<iframe
  :src="withBase('/guide/periodicity-unfolded.html') + '?wheel=0&touch=0'"
  title="Periodic cells, as computed"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/periodicity-unfolded.html')" target="_blank">Open in a separate page ↗</a></p>

The **barycenters** and the moments are those of these cells, next to the diracs as they were given. A
[Lloyd step](/examples/lloyd-quantization) $x_i \leftarrow b_i$ is then a short move, never a jump across the domain, and the gradient
$2 m_i ( x_i - b_i )$ of the cost ( `cost_and_position_grad` ) stays continuous when a dirac crosses a side, whatever the group does
to it there.

One dirac moving to the right on a torus ( the slider: its abscissa `x` ). Its cell follows it whole, out of the domain, while its
copy by the translation $( -1, 0 )$ comes in on the left side. Nothing jumps when $x$ passes $1$:

<iframe
  :src="withBase('/guide/periodicity-crossing.html') + '?wheel=0&touch=0'"
  title="A dirac crossing a side of a torus: its cell follows it"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/periodicity-crossing.html')" target="_blank">Open in a separate page ↗</a></p>

Everything is differentiable as without periodicity: with respect to the positions, the masses of the diracs and the parameters of
the target.

## A gallery of quotients

The rest of the page runs Lloyd's iterations on several quotients. Each dirac moves to the barycenter of its cell until the cells
are as regular as the space allows, so the result shows the **geometry of the quotient** itself. The slider at the top of each
viewer goes through the iterations.

```python
pb = sdot.OtProblem( sdot.SumOfDiracs( x ), target, periodicity = generators )
for it in range( 50 ):
    pb.source = sdot.SumOfDiracs( x )
    x = np.asarray( pb.solve().barycenters )      # next to the given diracs, whatever the side they crossed
```

### A torus

`periodicity = True`: one translation per side of the box.

<iframe
  :src="withBase('/guide/periodicity-torus.html') + '?wheel=0&touch=0'"
  title="Lloyd on a torus"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/periodicity-torus.html')" target="_blank">Open in a separate page ↗</a></p>

### A Klein bottle

`periodicity = [ ( flip, [ 1, 1 ] ), [ 0, 1 ] ]`. Leaving the square on the right brings you back on the left **upside down**, and the
top is glued to the bottom as on a torus. In the copies, the columns on either side of the square are its mirror images.

<iframe
  :src="withBase('/guide/periodicity-klein.html') + '?wheel=0&touch=0'"
  title="Lloyd on a Klein bottle"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/periodicity-klein.html')" target="_blank">Open in a separate page ↗</a></p>

### A Moebius strip

`periodicity = [ ( flip, [ 2, 1 ] ) ]` on a $2 \times 1$ box: the same flipped glue, alone. The top and bottom sides are not glued and
**stay walls**. SDOT finds them by itself: a side of the support that every image of the support stays behind is a wall of the quotient.

<iframe
  :src="withBase('/guide/periodicity-moebius.html') + '?wheel=0&touch=0'"
  title="Lloyd on a Moebius strip"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/periodicity-moebius.html')" target="_blank">Open in a separate page ↗</a></p>

### A cone

`periodicity = [ ( quarter, [ 0, 0 ] ) ]`, with `quarter` the rotation by 90° about the origin. The group is **finite** ( four
elements ) and has no translation at all. The quotient is a cone whose apex is the corner of the square. The target is an `Image`
whose density peaks at the apex, and the cells there **wrap around it**, each one spread over the four quarters.

<iframe
  :src="withBase('/guide/periodicity-cone.html') + '?wheel=0&touch=0'"
  title="Lloyd on a cone"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/periodicity-cone.html')" target="_blank">Open in a separate page ↗</a></p>

### A wallpaper group

`periodicity = [ ( quarter, [ 0, 0 ] ), [ 1, 0 ] ]`: quarter turns and translations generate the wallpaper group **p4**, a pattern
that repeats with a fourfold rotational symmetry. Its fundamental domain is a square of side $1/2$, and its copies around it are
**turned** by multiples of 90°.

<iframe
  :src="withBase('/guide/periodicity-p4.html') + '?wheel=0&touch=0'"
  title="Lloyd on the wallpaper group p4"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/periodicity-p4.html')" target="_blank">Open in a separate page ↗</a></p>

### A hexagonal lattice

`periodicity = [ [ 1, 0 ], [ 1/2, √3/2 ] ]`: translations again, but the second one is not orthogonal to the first. A box of height
$\sqrt{3}/2$ is glued to itself with a shift of half its width, like bricks. This is the lattice of the hexagonal packing, and 16
diracs settle into **regular hexagons**, which a square torus could not hold.

<iframe
  :src="withBase('/guide/periodicity-hexagonal.html') + '?wheel=0&touch=0'"
  title="Lloyd on a hexagonal torus"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/periodicity-hexagonal.html')" target="_blank">Open in a separate page ↗</a></p>

### A hexagon glued by `target.domain`

The same lattice, from another fundamental domain: a regular hexagon whose opposite sides are glued, read from the support by
`periodicity = target.domain`. The cells end as regular hexagons themselves.

```python
t = np.arange( 6 ) * np.pi / 3
target = sdot.ConvexHull( np.stack( [ np.cos( t ), np.sin( t ) ], -1 ) )
pb = sdot.OtProblem( sdot.SumOfDiracs( x ), target, periodicity = target.domain )
```

<iframe
  :src="withBase('/guide/periodicity-domain.html') + '?wheel=0&touch=0'"
  title="Lloyd on a hexagon whose opposite sides are glued"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/periodicity-domain.html')" target="_blank">Open in a separate page ↗</a></p>

### A cylinder along any direction

`periodicity = [ True, False ]` gives a cylinder along $x$, but the booleans only read the sides of an **axis-aligned box**. Any other
cylinder is a translation: here the support is a `ConvexHull`, a $2 \times 1$ rectangle turned by 30°, and `periodicity = [ 2 u ]` glues
it along its long side, $u = ( \cos 30°, \sin 30° )$. Its two other sides stay walls.

<iframe
  :src="withBase('/guide/periodicity-cylinder.html') + '?wheel=0&touch=0'"
  title="Lloyd on a turned cylinder"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/periodicity-cylinder.html')" target="_blank">Open in a separate page ↗</a></p>

## In 3D

Nothing above is specific to the plane. A **screw**, for example, moves up one step while turning by a quarter about the vertical axis:

```python
quarter_z = np.array( [ [ 0, -1, 0 ], [ 1, 0, 0 ], [ 0, 0, 1 ] ] )
p = sdot.plan( sdot.SumOfDiracs( x ), sdot.Box(), periodicity = [ ( quarter_z, [ 1, 0, 1 ] ), [ 1, 0, 0 ], [ 0, 1, 0 ] ] )
```

## Under the hood

There is one cell per dirac. It is built around its dirac, in the plane, and is not cut at the sides of the support. A cell cannot
leave the Voronoi cell of its dirac among its own images, since they all have the same weight.

Internally, the diracs are first **folded** onto the support by the group ( `sdot.wrap` does the same thing by hand ). This changes
nothing in what is computed, since a dirac and its images are the same point of the quotient, and the cells come back next to the
diracs as given ( an isometry ). What it buys is compactness: the differences between the diracs stay within a period, so the images
that can cut a cell are a short, fixed list for all the cells ( 9 translations on a 2D torus, however far apart the given diracs
are ), and each cell starts from one box, the support widened by half a period.

The planes come from the diracs and from their images by the elements of the group that come close to the support. The tree of the
diracs is traversed once per element, with its boxes moved. The density is integrated as the sum of its images, each cutting the cell
into its own pieces. In 2D, a torus traverses the tree 9 times instead of once, but the pruning drops most of the traversals right away.

## On a GPU

The diagrams ( `PowerDiagram`: measures, moments, their derivatives ) work on a CUDA device for any group. The **solve** on a CUDA
device handles a group of **translations** ( a torus, a cylinder, a sheared lattice ) with a constant density ( a `Box`, a
`Polytope`, an `Image` of equal values ), in 2D and 3D. A group with rotations or symmetries, or a density that is not a constant, is
solved on the CPU ( `LOOM_DEVICE=cpu` ).

## Gaussians and meshes

A `SumOfGaussians` is **wrapped**: each gaussian is summed over its images, its tails included, and its `support_box` is the
fundamental domain ( see [Gaussians](../distributions/gaussians#periodicity) ). A `Mesh` repeats the box of its nodes, without the
margin of its support ( see [Mesh](../distributions/mesh#periodicity-and-partial-transport) ). Both copy themselves by the elements
of the group that come near the cells, instead of being moved image by image in the kernels as an `Image` or a `Polytope` are.

## What is not supported yet

* 1D ( the circle is another algorithm ), the `"limits"` Newton step ( `"trials"` is used ), the width continuation of an image
  ( the solve starts from the sharp image ).

See the [roadmap](/guide/roadmap). The figures are written by
[`docs/guide/ground-metrics/periodicity_figures.py`](https://github.com/sdot-team/sdot/blob/main/docs/guide/ground-metrics/periodicity_figures.py).
