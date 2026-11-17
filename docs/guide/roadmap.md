# Roadmap

What we are working on. Nothing on this page is available yet: names and interfaces may still change.

## Ground metrics

- **Periodic metrics**, beyond [Ground Metrics](./ground-metrics/periodicity): the CUDA solve with rotations and symmetries or an
  image that is not a constant, 1D ( the circle ), the `"limits"` Newton step, the width continuation of an image.
- **Partial transport**, beyond [Ground Metrics](./ground-metrics/partial): a source heavier than the target, the
  penalized variant ( a sink of fixed weight ), gaussian targets and polynomial images, densities affine on pieces in 3D
  ( `Mesh( degree = 1 )` ), an image on the card, the card's own linear solver for the bordered system.
- **Radial kernels**: costs built from radial functions of $\|x - y\|$, e.g. polynomials, piecewise functions... Beyond
  `RadialExp` ( [Fokker-Planck](../examples/fokker-planck) ): 3D, the card, other densities than constant pieces, with a periodicity
  or a sink.

## Derivatives

- **Aggregates with a periodicity**: [duplicates and close seeds](./duplicates) in a periodic domain.
- **The frame of a target** through a periodicity or in partial transport, and the nodes of a mesh.

## Distributions

- **`PolynomialCells`**: piecewise polynomial densities on a polyhedral cell decomposition, the most general target.

## Tomography

- **Anisotropic tents on the card**, with `with_gradient`, and in 3D from 2D projections ( [Projected Sources](./distributions/projected) ).

## Parallel runs

Beyond [Running in Parallel](./parallel):

- **Derivatives with spread positions**: the measures are differentiable when the positions are given whole ( `nb_boxes` ), not yet
  when they are split between ranks.
- **The boxes made by the ranks**: the seeds redistributed by exchanges instead of on the host, more than 52 boxes, boxes of
  different sizes on JAX devices.
- **More than the measures**: moments, the transport itself ( a distributed Newton ), 3D, densities split like the boxes.
- **Out of core** on several ranks ( `max_resident` works on one ), boxes kept on disk, dask or zarr arrays as inputs.
- **Speed**: `loom.route` and `loom.scatter_add` in parallel, irregular exchanges on GPUs, runs on several GPUs and nodes.

## Norms

- $W_1$, ...

## Examples in progress

- **Crowd motion**: congested crowds as a Wasserstein gradient flow.
- **Incompressible Euler**: geodesics in the space of measure-preserving maps ( Gallouët-Mérigot scheme ).
