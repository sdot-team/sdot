<!-- written by scripts/api_docs.py from the docstrings: do not edit -->

# API — Distributions

The sources ( diracs ) and the targets ( densities ). See [Guide: Distributions](/guide/distributions).

::: v-pre

## `SumOfDiracs`

```python
SumOfDiracs(positions, weights=None, target_mass=1.0, **kwargs)
```

Sum of weighted Dirac point masses.

| Attribute | |
|---|---|
| `mass` | The total mass of the distribution ( 1 once normalized: the solvers normalize the distributions they are given ). |

### `.add_to_viz(viz, color='#d62728', **kwargs)`

The diracs, as points.

### `.bounding_half_spaces()`

The support of the distribution as half-spaces `( directions, offsets )`, or `None` when it is not bounded.

## `SumOfDiracs1d`

```python
SumOfDiracs1d(positions, weights=None, **kwargs)
```

Diracs on a line: `positions` `[ n ]` ( and not `[ n, 1 ]` ), of masses `weights` `[ n ]` ( all equal by
default ). `SumOfDiracs( positions[ :, None ] )` describes the same measure.

| Attribute | |
|---|---|
| `mass` | The total mass of the distribution ( 1 once normalized: the solvers normalize the distributions they are given ). |

### `.add_to_viz(viz, **kwargs)`

What this distribution shows of itself in a `Visualizer` -- NOTHING by default; a
subclass draws what is meaningful for it ( the centres of the gaussians, ... ) and grows
with the needs. Called by `SdotPlanNd.add_to_viz` for the target density.

### `.bounding_half_spaces()`

The support of the distribution as half-spaces `( directions, offsets )`, or `None` when it is not bounded.

## `ProjectedSumOfDiracs`

```python
ProjectedSumOfDiracs(points, normal, weights=None, target_mass=1.0, proj_dims=2, **kwargs)
```

1D diracs that are the projections of shared points on a direction per item of a batch -- e.g. one per angle of a
tomography: `points` `[ n, proj_dim ]`, `normal` `[ proj_dim ]` per item, `weights` `[ n ]` the masses. The projected
positions are computed on the fly, never stored.

| Attribute | |
|---|---|
| `mass` | The total mass of the distribution ( 1 once normalized: the solvers normalize the distributions they are given ). |

### `.add_to_viz(viz, **kwargs)`

What this distribution shows of itself in a `Visualizer` -- NOTHING by default; a
subclass draws what is meaningful for it ( the centres of the gaussians, ... ) and grows
with the needs. Called by `SdotPlanNd.add_to_viz` for the target density.

### `.bounding_half_spaces()`

The support of the distribution as half-spaces `( directions, offsets )`, or `None` when it is not bounded.

## `Box`

```python
Box(origin=None, frame=None, density=1.0, nb_dims=None)
```

A uniform density on a box: `origin + frame^T [ 0, 1 ]^d`. The rows of `frame` are the edges of the box, or `frame`
is their lengths ( an axis-aligned box ). By default the unit cube, in the dimension of the source it is used with.

## `Polygon`

```python
Polygon(points, density=1.0, **kwargs)
```

The `Polytope` of the CONVEX polygon with these `points` ( `[ n, 2 ]`, in any order ). Raises if a point is
not a vertex of the convex hull -- a non convex polygon is not a polytope.

## `Polyhedron`

```python
Polyhedron(points, density=1.0, **kwargs)
```

The `Polytope` of the CONVEX polyhedron with these `points` ( `[ n, 3 ]` ): the 3D sibling of `Polygon`

## `Polytope`

```python
Polytope(directions, offsets, density=1.0, target_mass=1.0, volume=None, **kwargs)
```

A uniform density on the convex polytope `directions . x <= offsets` ( `[ nb_cuts, d ]`, `[ nb_cuts ]` ), zero
elsewhere. `Polygon( points )` and `Polyhedron( points )` build one from its vertices.

| Attribute | |
|---|---|
| `mass` | The total mass of the distribution ( 1 once normalized: the solvers normalize the distributions they are given ). |
| `volume` | the volume of the polytope ( a host float ) |

### `.add_to_viz(viz, **kwargs)`

What this distribution shows of itself in a `Visualizer` -- NOTHING by default; a
subclass draws what is meaningful for it ( the centres of the gaussians, ... ) and grows
with the needs. Called by `SdotPlanNd.add_to_viz` for the target density.

### `.bounding_half_spaces()`

the polytope itself

## `Image`

```python
Image(values, frame=None, pixel_frame=None, **kwargs)
```

Piecewise constant function on a grid.

Each square/cube/hypercube is defined by `origin` and `frame( dir ) * knots( ... )`: the FIELD `frame` is the
frame of ONE pixel ( its rows: the edges of a pixel ).

By default, knots is equal to 0, 1, ... for each dim.

What the constructor takes is said for the WHOLE image: `Image( values )` fits the image in the unit square
( cube... ) at `origin`, its longest side of length 1, the proportions kept. `frame` gives the edges of the whole
image ( rows, or their lengths for an axis-aligned image ), `pixel_frame` those of one pixel ( the field ).
With `knots`, the pixels are where the knots say: the default frame is the identity.

| Attribute | |
|---|---|
| `mass` | The total mass of the distribution ( 1 once normalized: the solvers normalize the distributions they are given ). |

### `.add_to_viz(viz, **kwargs)`

What this distribution shows of itself in a `Visualizer` -- NOTHING by default; a
subclass draws what is meaningful for it ( the centres of the gaussians, ... ) and grows
with the needs. Called by `SdotPlanNd.add_to_viz` for the target density.

### `.bounding_half_spaces()`

The support of the distribution as half-spaces `( directions, offsets )`, or `None` when it is not bounded.

## `SumOfGaussians`

```python
SumOfGaussians(positions, sigmas, weights=None, target_mass=1.0, support_sigmas=6.0, support_box=None, **kwargs)
```

A mixture of isotropic gaussians: `positions` `[ k, d ]` their centers, `sigmas` `[ k ]` their standard deviations,
`weights` `[ k ]` their masses ( not their heights; equal by default ). Differentiable with respect to all three.

Its support, which bounds the domain of a transport, is the box holding every gaussian up to `support_sigmas`
standard deviations ( 6 by default ), or the box `support_box = ( lo, hi )`.

| Attribute | |
|---|---|
| `mass` | The total mass of the distribution ( 1 once normalized: the solvers normalize the distributions they are given ). |

### `.add_to_viz(viz, color='#d62728', **kwargs)`

The centres of the gaussians, as points.

### `.bounding_half_spaces()`

the box `[ c_i - k s_i, c_i + k s_i ]` united over the gaussians ( see the class docstring ) -- `None` without
`support_sigmas`. The DOMAIN is a constant of the transport ( nothing differentiates through it ): it is computed on
DETACHED parameters. Readable on the host, it is a numpy box; under a trace ( `jax.jit`, `jax.grad` ), its offsets are
traced arrays of the framework -- a `PowerDiagram` cuts its domain with them where they are ( `_domain_cell` ).

## `Mesh`

```python
Mesh(nodes, simplices, values=None, target_mass=1.0, **kwargs)
```

A piecewise linear density on a mesh of simplices ( triangles in 2D, tetrahedra in 3D ), zero outside.

`nodes` `[ nb_nodes, d ]`, `simplices` `[ nb_simplices, d + 1 ]` ( indices of nodes ). `values`: `[ nb_nodes ]` for a
continuous density ( one value per node ), or `[ nb_simplices, d + 1 ]` for a density discontinuous between simplices
( one value per corner of each simplex: zeros make a hole ). Constant by default. Differentiable with respect to
`values`.

| Attribute | |
|---|---|
| `mass` | The total mass of the distribution ( 1 once normalized: the solvers normalize the distributions they are given ). |

### `.add_to_viz(viz, **kwargs)`

What this distribution shows of itself in a `Visualizer` -- NOTHING by default; a
subclass draws what is meaningful for it ( the centres of the gaussians, ... ) and grows
with the needs. Called by `SdotPlanNd.add_to_viz` for the target density.

### `.bounding_half_spaces()`

the box of the mesh, widened by a margin ( `MARGIN` times its largest side ): the cells must be free to move
past the border of the mesh, otherwise the border would look fixed to the derivatives while it is not

### `.from_contours(contours, box=None, mesh_size=0.05, inside=1.0, outside=0.0, margin=0.2)`

a 2D mesh generated by `pygmsh` ( needs it installed ) from closed CONTOURS: `contours` is a list of `[ n, 2 ]` arrays of
points ( in order, not repeated at the end ), each one the border of a region. The regions are paved with triangles, together
with a box around them, whose edges follow the contours ( no triangle straddles one ). The density is `inside` ( 1 ) on the
triangles within a contour and `outside` ( 0 ) on those of the box around.

`box` is `( lo, hi )`, or `None` for the box of the contours widened by `margin` times its largest side. `mesh_size` is the
target length of the edges ( a number, handed to `pygmsh` for the points of the contours and the box ).

### `.with_values(values, current_mass=None)`

the same mesh ( geometry, hierarchy ) with other DG1 values `[ nb_elems, d + 1 ]`, in the order of `self.simplices`

:::
