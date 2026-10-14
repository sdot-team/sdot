# Roadmap

What we are working on. Nothing on this page is available yet: names and interfaces may still change.

## Ground metrics

- **Periodic metrics**: transport on a torus ( textures, angles... ).
- **Entropic regularization**: an entropy penalty on the transport cost, for a smoother plan.
- **Radial kernels**: costs built from radial functions of $\|x - y\|$, e.g. polynomials.

## Distributions

- **`PolynomialCells`**: piecewise polynomial densities on a polyhedral cell decomposition, the most general target.

## Frameworks

- **Dask**: problems larger than the memory of a GPU, with the power diagram distributed across workers.

## Performance

- 1D benchmarks.

## Examples in progress

- **Power diagram geometry**: inspect the Laguerre cells ( facets, simplices, distance to the boundaries ).
- **Periodic metrics**: OT and power diagrams on a torus.
- **Crowd motion**: congested crowds as a Wasserstein gradient flow.
- **Incompressible Euler**: geodesics in the space of measure-preserving maps ( Gallouët-Mérigot scheme ).
- **Fokker-Planck**: Wasserstein gradient flow of entropy plus potential energy.
