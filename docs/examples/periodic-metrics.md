---
title: Periodic Metrics
description: OT on quotient spaces — a torus, a cone... — from any group of isometries.
tags: [geometry, periodic, 2d]
order: 6
---

<script setup>
import { withBase } from 'vitepress'
</script>

# Periodic Metrics

Given $G$ a group of isometries, one can take the cost $c( x, y ) = \min_{g \in G} | x - g( y ) |^2$ using the `periodicity` parameter.

`periodicity` can be a vector (a translation), a pair `( M, V )` ( $x \mapsto M x + V$ ) or a homogeneous
matrix; the target is the content of a fundamental domain. Details in [Guide: Ground Metrics](/guide/ground_metrics#periodicity).

For instance, to compute a plan inside a Klein bottle :

```python
flip = np.diag( [ 1, -1 ] )
p = sdot.plan( sdot.SumOfDiracs( positions ), sdot.Box(), 
    periodicity = [ 
        ( flip, [ 1, 1 ] ), 
        [ 0, 1 ] 
    ] 
)
```

Alternatively, if `g` is axis aligned, one can use booleans (a single one or a list of booleans for cylinders) to use the domain bounds of `g` to specify the periodicity.

## Visualization

A cell is computed whole, around its dirac as given. To check a quotient by eye, draw the cells, then their copies by the group around the identity, fainter: the
copies must glue without a seam. The domain is dashed. The copies do not count in the zoom of the page ( `fit_periodicity_rings` ).

```python
p = sdot.plan( sdot.SumOfDiracs( positions ), sdot.Box(), periodicity = True )

viz = sdot.Visualizer()
p.add_to_viz( viz )    # the cells, then their copies by the first ring of the group, fainter

# more rings, more visible
p.add_to_viz( viz, nb_periodicity_rings = 2, opacity_periodicity_rings = 0.5 )
viz.write_html( "torus.html" )
```

The same by hand, with the pieces it is made of:

```python
pd = p.power_diagram()
for k, a in enumerate( pd.periodicity_rings( 1 ) ):   # the elements of the first ring, as affine maps [ M | V ], the identity first
    pd.cells.moved( a ).add_to_viz( viz, opacity = 1.0 if k == 0 else 0.25, frames = k == 0 )   # ( the copies out of the zoom )
```

`periodicity_rings( k )` gives the elements of the group up to the ring `k`: ring 1 holds the elements whose image of the domain touches the
domain ( the 8 neighbours of a square on a torus ), ring 2 those touching ring 1, and so on. `Cell.moved` moves a cell by one of them,
or each cell by its own. The pages below are written by `docs/examples/periodic_metrics.py` with these calls.

## Walls, then a torus

24 diracs with random masses on the unit square. Move the slider at the bottom of the viewer: `periodic = 0` is the transport with
walls, its cells moved by hand by the elements of the torus's first ring; `periodic = 1` is the transport on the torus.

<iframe
  :src="withBase('/examples/periodic-walls.html') + '?wheel=0&touch=0'"
  title="Walls, then a torus"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/examples/periodic-walls.html')" target="_blank">Open in a separate page ↗</a></p>

With walls, the copies do not match along the sides: the cells are cut by the box, and a dirac near a side has no neighbour beyond it.
On the torus the copies glue without a seam, and every cell is surrounded on all sides: there is no boundary effect left.

## Cells are where their diracs are

Below, three diracs are given outside the domain, one period away or more, and one has a large mass. With `ring = 0` ( `nb_periodicity_rings = 0` ) the cells are
shown alone: those of the three diracs lie outside the domain, away from the others, and the big one crosses a side. With `ring = 1`
their copies fill the gaps, and the quotient appears.

<iframe
  :src="withBase('/examples/periodic-unfolded.html') + '?wheel=0&touch=0'"
  title="Periodic cells, as computed"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/examples/periodic-unfolded.html')" target="_blank">Open in a separate page ↗</a></p>

The **barycenters** and the moments are those of these cells, next to the diracs as they were given. A
[Lloyd step](./lloyd-quantization) $x_i \leftarrow b_i$ is then a short move, never a jump across the domain, and the gradient
$2 m_i ( x_i - b_i )$ of the cost stays continuous when a dirac crosses a side, whatever the group does to it there.

## A gallery of quotients

The rest of the page runs Lloyd's iterations on several quotients. Each dirac moves to the barycenter of its cell until the cells
are as regular as the space allows, so the result shows the **geometry of the quotient** itself. The slider at the bottom of each
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
  :src="withBase('/examples/periodic-torus.html') + '?wheel=0&touch=0'"
  title="Lloyd on a torus"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/examples/periodic-torus.html')" target="_blank">Open in a separate page ↗</a></p>

### A Klein bottle

`periodicity = [ ( flip, [ 1, 1 ] ), [ 0, 1 ] ]`. Leaving the square on the right brings you back on the left **upside down**, and the
top is glued to the bottom as on a torus. In the copies, the columns on either side of the square are its mirror images.

<iframe
  :src="withBase('/examples/periodic-klein.html') + '?wheel=0&touch=0'"
  title="Lloyd on a Klein bottle"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/examples/periodic-klein.html')" target="_blank">Open in a separate page ↗</a></p>

### A Moebius strip

`periodicity = [ ( flip, [ 2, 1 ] ) ]` on a $2 \times 1$ box: the same flipped glue, alone. The top and bottom sides are not glued and
**stay walls**. SDOT finds them by itself: a side of the support that every image of the support stays behind is a wall of the quotient.

<iframe
  :src="withBase('/examples/periodic-moebius.html') + '?wheel=0&touch=0'"
  title="Lloyd on a Moebius strip"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/examples/periodic-moebius.html')" target="_blank">Open in a separate page ↗</a></p>

### A cone

`periodicity = [ ( quarter, [ 0, 0 ] ) ]`, with `quarter` the rotation by 90° about the origin. The group is **finite** ( four
elements ) and has no translation at all. The quotient is a cone whose apex is the corner of the square. The target is an `Image`
whose density peaks at the apex, and the cells there **wrap around it**, each one spread over the four quarters.

<iframe
  :src="withBase('/examples/periodic-cone.html') + '?wheel=0&touch=0'"
  title="Lloyd on a cone"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/examples/periodic-cone.html')" target="_blank">Open in a separate page ↗</a></p>

### A wallpaper group

`periodicity = [ ( quarter, [ 0, 0 ] ), [ 1, 0 ] ]`: quarter turns and translations generate the wallpaper group **p4**, a pattern
that repeats with a fourfold rotational symmetry. Its fundamental domain is a square of side $1/2$, and its copies around it are
**turned** by multiples of 90°.

<iframe
  :src="withBase('/examples/periodic-p4.html') + '?wheel=0&touch=0'"
  title="Lloyd on the wallpaper group p4"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/examples/periodic-p4.html')" target="_blank">Open in a separate page ↗</a></p>

### A hexagonal lattice

`periodicity = [ [ 1, 0 ], [ 1/2, √3/2 ] ]`: translations again, but the second one is not orthogonal to the first. A box of height
$\sqrt{3}/2$ is glued to itself with a shift of half its width, like bricks. This is the lattice of the hexagonal packing, and 16
diracs settle into **regular hexagons**, which a square torus could not hold.

<iframe
  :src="withBase('/examples/periodic-hexagonal.html') + '?wheel=0&touch=0'"
  title="Lloyd on a hexagonal torus"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/examples/periodic-hexagonal.html')" target="_blank">Open in a separate page ↗</a></p>

### A cylinder along any direction

`periodicity = [ True, False ]` gives a cylinder along $x$, but the booleans only read the sides of an **axis-aligned box**. Any other
cylinder is a translation: here the support is a `Polygon`, a $2 \times 1$ rectangle turned by 30°, and `periodicity = [ 2 u ]` glues
it along its long side, $u = ( \cos 30°, \sin 30° )$. Its two other sides stay walls.

<iframe
  :src="withBase('/examples/periodic-cylinder.html') + '?wheel=0&touch=0'"
  title="Lloyd on a turned cylinder"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/examples/periodic-cylinder.html')" target="_blank">Open in a separate page ↗</a></p>

## In 3D

Nothing above is specific to the plane. A **screw**, for example, moves up one step while turning by a quarter about the vertical axis:

```python
quarter_z = np.array( [ [ 0, -1, 0 ], [ 1, 0, 0 ], [ 0, 0, 1 ] ] )
p = sdot.plan( sdot.SumOfDiracs( x ), sdot.Box(), periodicity = [ ( quarter_z, [ 1, 0, 1 ] ), [ 1, 0, 0 ], [ 0, 1, 0 ] ] )
```

Gaussian and mesh targets, the GPU solver and the circle ( 1D ) are on the [roadmap](/guide/roadmap).
