# Roadmap

What we are working on. Nothing on this page is available yet: names and interfaces may still change.

## Ground metrics

- **Periodic metrics**, beyond [Ground Metrics](./ground-metrics/periodicity): the CUDA solve with rotations and symmetries or an
  image that is not a constant, 1D ( the circle ).
- **Partial transport**, beyond [Ground Metrics](./ground-metrics/partial): a source heavier than the target, the
  penalized variant ( a sink of fixed weight ), 3D, gaussian targets, an image on the card, the card's own linear solver for the
  bordered system.
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

- **Derivatives** of the measures, the cells travelling again in the backward pass.
- **The boxes made by the ranks**: the seeds redistributed by exchanges instead of on the host, any number of seeds and of ranks.
- **More than the measures**: moments, the transport itself ( a distributed Newton ), 3D, densities split like the boxes.
- **Out of core**: more boxes than ranks, the boxes that do not fit kept in host memory or on disk ( dask or zarr arrays as inputs ).
- **Speed**: `loom.route` and `loom.scatter_add` in parallel, irregular exchanges on GPUs, runs on several GPUs and nodes.

## Norms

- $W_1$, ...

## Examples in progress

- **Crowd motion**: congested crowds as a Wasserstein gradient flow.
- **Incompressible Euler**: geodesics in the space of measure-preserving maps ( Gallouët-Mérigot scheme ).
- **Fokker-Planck**: Wasserstein gradient flow of entropy plus potential energy.
