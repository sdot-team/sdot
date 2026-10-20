# Frameworks and Devices

sdot works with NumPy, JAX, PyTorch and CuPy arrays. There is nothing to configure: **the inputs decide**. A transport
computed from JAX arrays runs where they live and returns JAX arrays, one computed from PyTorch tensors returns PyTorch
tensors, and so on:

::: code-group

<<< @/snippets/hello_numpy.py [NumPy]

<<< @/snippets/hello_jax.py [JAX]

<<< @/snippets/hello_torch.py [PyTorch]

<<< @/snippets/hello_cupy.py [CuPy]

:::

- **Objects**: a plan, a distribution... hand back arrays of the same framework: `p.barycenters` is a JAX array when the
  inputs were JAX arrays.
- **Devices**: the computation runs on the device of the inputs. On a CUDA GPU, the whole solve runs on the card for 2D
  problems ( against a box, a polygon, an image, gaussians or a triangle mesh ) and for 3D problems against a uniform
  density, isotropic gaussians or a tetrahedral mesh; anything else raises there.
- **Gradients**: `jax.grad` and `torch.autograd` go through `w2` and the outputs of a plan, with respect to the diracs and to
  the parameters of the target ( an image's values, frame and origin, gaussians, a mesh's values... ), on the CPU and on a CUDA
  GPU, under `jax.jit` and for batches of cotangents ( `jax.jacrev` ). In 1D, the barycenters derive when the plan stores them
  ( `Direct( with_barycenters = True )` ).
- **`jax.jit`**: the whole solve is one call inside the compiled program.
- **Precision**: JAX works in float32 unless `jax_enable_x64` is set; float64 is recommended for gradients.

## What decides, when nothing does

A value built from nothing ( a list, a number ) takes the default framework: the one already imported, or
`loom.default_framework`. sdot relies on [loom](https://hleclerc.github.io/loom/) for this; see its page on
[frameworks and devices](https://hleclerc.github.io/loom/guide/frameworks).

## Compiled kernels

The C++ / CUDA kernels come precompiled with the package for the common cases. Another combination ( a dimension, a
distribution, a device ) is compiled on first use and cached, which needs a C++ compiler ( and `nvcc` for NVIDIA GPUs ).
