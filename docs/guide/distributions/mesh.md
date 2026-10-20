---
gallery: feature
description: "Piecewise linear densities on triangles or tetrahedra: domains of any shape."
tags: [mesh, 2d, 3d]
order: 4
---

<script setup>
import { withBase } from 'vitepress'
</script>

# Meshes

A density **piecewise linear** on a mesh of simplices: triangles in 2D, tetrahedra in 3D. Zero outside the mesh, so the mesh
need not be convex, nor connected: it is the way to give a domain of any shape.

```python
target = sdot.Mesh( nodes, simplices )                  # nodes [ n, d ], simplices [ m, d + 1 ]: a constant density
target = sdot.Mesh( nodes, simplices, values = rho )    # rho [ n ]: a value per node, continuous
target = sdot.Mesh( nodes, simplices, values = rho )    # rho [ m, d + 1 ]: a value per corner of each simplex
```

With fewer nodes per simplex ( segments, or triangles in 3D ), the measure is carried by lines or surfaces: see
[Lines and Surfaces](./codim-mesh).

## A domain of any shape

`Mesh.from_contours( contours )` meshes 2D regions given by their borders ( it needs `pygmsh` ): `contours` is a list of
`[ n, 2 ]` arrays, the points of each border in order. The density is 1 inside the contours and 0 on a box around them, whose
triangles follow the contours.

A gear, which is not convex, and the diracs of its Lloyd quantization ( 2000 diracs, see
[Lloyd quantization](/examples/lloyd-quantization) ):

```python
target = sdot.Mesh.from_contours( [ gear ], mesh_size = 0.02 )   # gear [ 72, 2 ]: the border, in order

x = sdot.quantization( target, 2000, seed = 0 )
```

<iframe
  :src="withBase('/guide/mesh-contours.html') + '?wheel=0&touch=0'"
  title="Lloyd quantization of a gear meshed from its contour"
  loading="lazy"
  allowfullscreen
  style="width: 100%; aspect-ratio: 1 / 1; max-height: 75vh; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/mesh-contours.html')" target="_blank">Open in a separate page ↗</a></p>

`box` ( `( lo, hi )` ), `margin`, `inside` and `outside` ( the two values ) change the defaults.

## A value per node

`values` of shape `[ n ]`: the density is continuous, linear on each simplex between the values of its nodes ( P1 ). The
Delaunay triangulation of 2000 random points of the unit square, a gaussian bump and oblique bands:

```python
nodes = np.concatenate( [ corners, rng.random( ( 2000, 2 ) ) ] )
triangles = scipy.spatial.Delaunay( nodes ).simplices

target = sdot.Mesh( nodes, triangles, values = f( nodes ) )
```

<iframe
  :src="withBase('/guide/mesh-p1.html') + '?wheel=0&touch=0'"
  title="Lloyd quantization of a piecewise linear density"
  loading="lazy"
  allowfullscreen
  style="width: 100%; aspect-ratio: 1 / 1; max-height: 75vh; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/mesh-p1.html')" target="_blank">Open in a separate page ↗</a></p>

## A value per corner

`values` of shape `[ m, d + 1 ]`: one value per corner of each simplex, so the density may jump from one simplex to the next
( DG1 ). The edges of the mesh go where one wants them: a density smooth by pieces is then written with its jumps exactly on
the curves that separate the pieces, however they wind.

Five wavy bands, each with its own smooth function, alternately dark and light. The mesh is a grid bent by a wave, and the
borders of the bands are lines of the grid:

```python
nodes, triangles, band = wavy_bands()                  # band: the band of each triangle, see the script below
values = np.zeros( triangles.shape )                     # [ m, 3 ]
for b, f in enumerate( funcs ):
    values[ band == b ] = f( *nodes[ triangles[ band == b ] ].transpose( 2, 0, 1 ) )

target = sdot.Mesh( nodes, triangles, values = values )
```

<iframe
  :src="withBase('/guide/mesh-bands.html') + '?wheel=0&touch=0'"
  title="Lloyd quantization of five wavy bands"
  loading="lazy"
  allowfullscreen
  style="width: 100%; aspect-ratio: 1 / 1; max-height: 75vh; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/mesh-bands.html')" target="_blank">Open in a separate page ↗</a></p>

The diracs see the jumps, and nothing else of the mesh: inside a band they follow its function.

Zeros on some simplices make a hole. `from_contours( [ circle ], inside = 0, outside = 1 )` meshes a square around a round
hole, whose border is 64 edges of the mesh; times $1 + x$:

```python
mesh = sdot.Mesh.from_contours( [ circle ], box = ( ( 0, 0 ), ( 1, 1 ) ), inside = 0, outside = 1 )
nodes, triangles = np.asarray( mesh.nodes ), np.asarray( mesh.simplices )
values = np.asarray( mesh.values ) * ( 1 + nodes[ triangles ][ ..., 0 ] )

target = sdot.Mesh( nodes, triangles, values = values )
```

<iframe
  :src="withBase('/guide/mesh-hole.html') + '?wheel=0&touch=0'"
  title="Lloyd quantization of a mesh with a round hole"
  loading="lazy"
  allowfullscreen
  style="width: 100%; aspect-ratio: 1 / 1; max-height: 75vh; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/mesh-hole.html')" target="_blank">Open in a separate page ↗</a></p>

## In 3D

The same with tetrahedra. A tube ( a hollow cylinder ) cut into tetrahedra, with a density growing with the height, and 3000
diracs colored by their depth in the view ( warm in front; `c` switches it ):

```python
nodes, tetrahedra = tube()                              # see the script below
target = sdot.Mesh( nodes, tetrahedra, values = 0.2 + nodes[ :, 2 ] )

x = sdot.quantization( target, 3000, seed = 0 )
```

<iframe
  :src="withBase('/guide/mesh-tube.html') + '?wheel=0&touch=0'"
  title="Lloyd quantization of a tube of tetrahedra"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/guide/mesh-tube.html')" target="_blank">Open in a separate page ↗</a></p>

## What is computed

A cell is cut by each simplex it meets, and the density is affine on each piece: the masses, the barycenters, the transport
cost and the facets of the Newton system are **exact**. The simplices a cell meets are found in a bounding volume hierarchy,
built once with the target. Simplices of zero volume are dropped.

The domain of the transport is the box of the mesh, widened by a tenth of its largest side: the cells must be free to cross the
border of the mesh. A cell that stays outside the mesh has no mass.

## Periodicity and partial transport

With a [periodicity](../ground-metrics/periodicity), the fundamental domain is the box of the mesh itself, without the margin:
`periodicity = True` glues its opposite sides. The simplices are copied by the elements of the group that bring them near the
cells, and the values keep their derivatives.

A [partial transport](../ground-metrics/partial) ( a source lighter than the target ) integrates each cell inside its ball, in 2D:
the density is affine on each piece, and a polygon inter a disc still has its integrals in closed form, as do the facets, the
sphere and all the derivatives. The two combine.

## The continuation

A density with empty regions ( a hole, a domain that is not convex, far apart bumps ) is a hard start: a cell with no mass gives
Newton nothing to move it. The solver goes through **spread** versions of the mesh first ( `Iterative( continuation = "auto" )`,
the default ): the same mesh, its values averaged with those of the neighboring corners, again and again. The mass and the
positivity are kept, the holes fill up, and the stages go back down to the mesh itself.

## Derivatives

`sdot.w2` and `sdot.plan` are differentiable with respect to `values`, with JAX or PyTorch. The nodes are a constant of the
transport.

## Reproducing the pages

They are written by [`docs/guide/distributions/mesh_figures.py`](https://github.com/sdot-team/sdot/blob/main/docs/guide/distributions/mesh_figures.py):

```bash
python docs/guide/distributions/mesh_figures.py                         # all of them
python docs/guide/distributions/mesh_figures.py bands hole             # or some of them
```

## Limits

- 2D and 3D only.
- No derivative with respect to the `nodes`.
- [Partial transport](../ground-metrics/partial) in 2D only.
- With a periodicity or a sink, the solve runs on the CPU.
