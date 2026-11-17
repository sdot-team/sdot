# Coming from pysdot

sdot replaces [pysdot](https://github.com/sd-ot/pysdot). Code written for pysdot keeps running on sdot alone, through the
compatibility layer `sdot.pysdot`: same classes, same methods, same conventions, computed by sdot. pysdot does not need to be
installed.

## Two ways to use it

Change the imports:

```python
from sdot.pysdot import OptimalTransport, PowerDiagram          # was: from pysdot import ...
from sdot.pysdot.domain_types import ConvexPolyhedraAssembly    # was: from pysdot.domain_types import ...
from sdot.pysdot.radial_funcs import RadialFuncInBall           # was: from pysdot.radial_funcs import ...
```

or leave them as they are and add two lines before them, so that `import pysdot` gives the compatibility layer:

<<< @/snippets/pysdot_compat.py

## What is the same

- `PowerDiagram`: `integrals`, `centroids`, `second_order_moments`, `der_integrals_wrt_weights` (the same CSR matrix),
  `der_centroids_and_integrals_wrt_weight_and_positions`, `cell_polyhedra`, `vtk_mesh_data`, `display_vtk`,
  `add_replication` (periodicity), in 2D and 3D; `distances_from_boundaries` and `display_asy` in 2D. The weights have the same meaning: the cell of seed $i$ is where
  $|x - p_i|^2 - w_i$ is smallest. The integrals are not normalized, as in pysdot.
- `OptimalTransport`: `adjust_weights`, `get_centroids`, `set_positions`, `set_masses`, `set_stopping_criterion`...
- `TransportMap` and `find_transport_map`.
- The domains: `ConvexPolyhedraAssembly` (boxes, `sub_box`, simplices, convex polyhedra, `add_img`) and `ScaledImage`, with
  its polynomial pixels (6 coefficients in 2D).
- `pysdot.util.FastMarching` (it needs `scikit-fmm`), and the older call form `OptimalTransport( domain, radial_func )`.
- The radial functions `RadialFuncUnit`, `RadialFuncInBall` (partial transport) and `RadialFuncEntropy`.

The results agree with those of pysdot to $10^{-12}$ on the cases of `tests/test_pysdot_compat.py`, and the examples of pysdot
(`examples/`) run, including those that pysdot itself no longer runs (the older call form).

## What differs

- `OptimalTransport.adjust_weights` runs the solver of sdot, which starts from the previous solution. The weights are
  defined up to a constant, which sdot fixes differently: the diagram is the same, the weights are shifted. `linear_solver`,
  `relax` and `ret_if_err` are ignored, and a start where cells are empty does not raise `BadInitialGuess`: sdot gets out of it.
  `obj_max_dw`, a criterion on the weights in pysdot, becomes sdot's criterion on the normalized masses.
- With `add_replication`, the cells are those of the torus that the translations generate: pysdot lost part of the mass there.
  When the domain is larger than a period (it overlaps its translates), the translated seeds are added as seeds, as pysdot did.
- `der_centroids_and_integrals_wrt_weight_and_positions` goes through the automatic differentiation of sdot: JAX or PyTorch
  must be installed. Its derivatives with respect to the positions match finite differences, which those of pysdot did not
  always do.
- Several boxes, simplices or a single convex polyhedron are supported. A mix of several pieces with a convex polyhedron is not,
  and simplices must not overlap.
- `display_asy` ignores `avoid_bounds` and the shading of a radial function (`min_rf`, `max_rf`).
- Each call goes through sdot, which costs a few milliseconds whatever the number of cells: what is read several times from
  the same diagram is computed once, but a loop over many tiny diagrams is slower than with pysdot.
- Not available: `image_integrals` and the radial functions other than the three above. `display_jupyter` opens the
  [viewer](/guide/visualizer) of sdot.

## Going further

`PowerDiagram.sdot_diagram()` gives the `sdot.PowerDiagram` underneath, and `OptimalTransport.plan` the last
[plan](/guide/ot-plan) of sdot: what sdot does beyond pysdot (densities, GPUs, derivatives) starts from there.
