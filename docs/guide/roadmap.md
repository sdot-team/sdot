# Roadmap

What we are working on. Nothing on this page is available yet: names and interfaces may still change.

## Ground metrics

- **Periodic metrics**, beyond [Ground Metrics](./ground-metrics/periodicity): the CUDA solve with rotations and symmetries or a
  density that is not a constant, 1D ( the circle ).
- **Partial transport**, beyond [Ground Metrics](./ground-metrics/partial): a source heavier than the target, the
  penalized variant ( a sink of fixed weight ), 3D, gaussian targets, the CUDA solve.
- **Radial kernels**: costs built from radial functions of $\|x - y\|$, e.g. polynomials, piecewise functions... Beyond
  `RadialExp` ( [Fokker-Planck](../examples/fokker-planck) ): 3D, the card, other densities than constant pieces, with a periodicity
  or a sink.

## Derivatives

- **Aggregates with a periodicity**: [duplicates and close seeds](./duplicates) in a periodic domain.
- **The frame of a target** through a periodicity or in partial transport, and the nodes of a mesh.

## Distributions

- **`PolynomialCells`**: piecewise polynomial densities on a polyhedral cell decomposition, the most general target.

## Transport maps

- **Point location**: the power cell of given points ( hence the transport map read at these points ), for [Linearized OT](../examples/lot-dictionary), which finds it by hand with an `argmin` over all the diracs.

## Frameworks

- **Dask**: problems larger than the memory of a GPU, with the power diagram distributed across workers.

## Norms

- $W_1$, ...

## Examples in progress

- **Power diagram geometry**: inspect the Laguerre cells ( facets, simplices, distance to the boundaries ).
- **Crowd motion**: congested crowds as a Wasserstein gradient flow.
- **Incompressible Euler**: geodesics in the space of measure-preserving maps ( Gallouët-Mérigot scheme ).
- **Fokker-Planck**: Wasserstein gradient flow of entropy plus potential energy.
