# SDOT — Semi-Discrete Optimal Transport

**Documentation: <https://sdot-team.github.io/sdot/>**

Fast, differentiable, N-dimensional semi-discrete optimal transport, between a sum of diracs and a continuous density
( an image, a mesh, a gaussian mixture, a polytope... ).

- **Fast**: a Newton method on power diagrams, O(n log n). SIMD on the CPU, CUDA on the GPU.
- **Differentiable**: a layer inside `jax.jit` / `jax.grad` or a PyTorch autograd graph.
- **Any dimension**: 2D, 3D, 6D... with the same API.
- **Framework independent**: NumPy, JAX, PyTorch or CuPy arrays, on the CPU or the GPU. The inputs decide.
- **Ground metrics**: periodic spaces and partial transport.

```bash
pip install sdot
```

```python
import numpy as np
import sdot

rng = np.random.default_rng( 0 )
f = sdot.SumOfDiracs( rng.random( ( 100, 2 ) ) )  # 100 diracs of equal mass
g = sdot.Image( rng.random( ( 10, 10 ) ) )  # a 10 x 10 image, on the unit square

print( sdot.w2( f, g ) )  # W2^2, as a numpy array
```

See the [getting started guide](https://sdot-team.github.io/sdot/guide/getting-started) and the
[examples](https://sdot-team.github.io/sdot/examples/).

H. Leclerc, Q. Mérigot, T. Gallouët · LMO / INRIA PARMA
