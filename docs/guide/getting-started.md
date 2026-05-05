# Getting Started

## Installation

```bash
pip install sdot
```

SDOT requires at least one backend. Install the one you prefer (or both):

```bash
pip install jax jaxlib          # JAX backend
pip install torch               # PyTorch backend
```

> **GPU support** — follow the standard JAX/PyTorch GPU installation instructions for your platform (CUDA, Metal, …). SDOT will automatically use the device your tensors are on.

---

## Hello, World

```python
from sdot import SplineGrid, SumOfDiracs, distance
import numpy as np

f = SumOfDiracs( np.random.rand( 1000, 2 ) )   # 1000 equal-weight diracs, 2D
g = SplineGrid( np.random.rand( 10, 10 ) )      # 10×10 spline density (auto-normalized)

print( distance( f, g ) )
```

That's it. SDOT infers the backend, dimension, dtype, and device automatically.

---

## The Driver

SDOT abstracts the underlying framework (JAX or PyTorch) behind a single `driver` object. You never import JAX or PyTorch directly in your code — everything goes through `sdot.driver`.

### Auto-detection

SDOT looks for an already-loaded backend in this order:

1. Modules already imported in the current session (`jax`, `torch`)
2. The `SDOT_FRAMEWORK` environment variable
3. First available library (`jax`, then `torch`)

### Manual configuration

```python
import sdot

sdot.driver.framework = "jax"      # or "torch"
sdot.driver.dtype     = "FP64"     # FP32, FP64
sdot.driver.device    = "cuda:0"   # or "cpu", "mps", …
```

Or via environment variables before launching Python:

```bash
export SDOT_FRAMEWORK=torch
export SDOT_DTYPE=FP32
export SDOT_DEVICE=cuda:0
```

All arrays returned by SDOT live in the same framework, on the same device, with the same dtype — so they slot directly into your JAX/PyTorch computation graph.

---

## Distributions

A distribution is either a **source** (a discrete measure, a sum of Dirac masses) or a **target** (a continuous density on some domain).

### Source distributions

| Class | Description |
|---|---|
| `SumOfDiracs( positions )` | Equal-weight Dirac masses at given positions |
| `SumOfWeightedDiracs( positions, weights )` | Dirac masses with explicit weights |

```python
import numpy as np
from sdot import SumOfDiracs, SumOfWeightedDiracs

# 500 uniform diracs in 2D
f = SumOfDiracs( np.random.rand( 500, 2 ) )

# 50 diracs with explicit masses (must sum to 1 for full transport)
positions = np.random.rand( 50, 2 )
weights   = np.ones( 50 ) / 50
f2 = SumOfWeightedDiracs( positions, weights )
```

### Target distributions

All target distributions are **normalized to mass 1 by default**. Passing a different total mass activates partial transport.

| Class | Description |
|---|---|
| `SplineGrid( values, continuity=1 )` | C¹ (or higher) spline density on a regular grid |
| `PolynomialGrid( values )` | Piecewise polynomial density on a regular grid |
| `SumOfGaussians( centers, covariances, weights )` | Mixture of Gaussians _(coming soon)_ |
| `Mesh( vertices, faces )` | Density on a surface or volume mesh _(coming soon)_ |

```python
import numpy as np
from sdot import SplineGrid, PolynomialGrid

# Uniform density on [0,1]²
uniform = PolynomialGrid( values = [[[1]]] )

# Natural cubic spline from sampled values
spline = SplineGrid( np.array( [ 1, 4, 9, 4, 1 ] ), continuity = 1 )

# Density from an image (normalize automatically)
import imageio
img = imageio.imread( "cat.png" ).mean( axis = -1 ).astype( float )
density = PolynomialGrid( values = img )
```

---

## Computing Distances and Plans

### Wasserstein distance

```python
from sdot import SplineGrid, SumOfDiracs, distance
import numpy as np

f = SumOfDiracs( np.random.rand( 200, 2 ) )
g = SplineGrid( np.random.rand( 8, 8 ) )

d = distance( f, g )
print( d )   # scalar — the W₂² transport cost
```

### Full OT plan

```python
from sdot import SplineGrid, SumOfDiracs, optimal_transport_plan
import numpy as np

f = SumOfDiracs( np.random.rand( 200, 2 ) )
g = SplineGrid( np.random.rand( 8, 8 ) )

plan = optimal_transport_plan( f, g )

plan.distance          # W₂² transport cost (scalar)
plan.barycenters       # centroid of each Laguerre cell  — shape (n, d)
plan.cell_masses       # mass of each cell               — shape (n,)
plan.brenier_potential # dual potential ψ                — shape (n,)
```

---

## Differentiability

Both `distance` and `optimal_transport_plan` are differentiable with respect to the Dirac positions and weights (and the density parameters, if supported by the target type).

### With JAX

```python
import jax
import jax.numpy as jnp
from sdot import PolynomialGrid, SumOfDiracs, distance

target = PolynomialGrid( values = jnp.ones( ( 10, 10 ) ) )

@jax.jit
def loss( positions ):
    return distance( SumOfDiracs( positions ), target )

positions = jnp.array( np.random.rand( 100, 2 ) )
grad = jax.grad( loss )( positions )   # shape (100, 2)
```

### With PyTorch

```python
import torch
import sdot

sdot.driver.framework = "torch"

from sdot import PolynomialGrid, SumOfWeightedDiracs, distance

target    = PolynomialGrid( values = torch.ones( 10, 10 ) )
positions = torch.rand( 100, 2, requires_grad = True )
weights   = torch.ones( 100 ) / 100

d = distance( SumOfWeightedDiracs( positions, weights ), target )
d.backward()

print( positions.grad )   # shape (100, 2)
```

---

## Batch Variants

For GPU-parallel workloads, all distributions have `BatchOf` and `1d` variants:

```python
from sdot import BatchOfSumOfDiracs, BatchOfSplineGrid, distance

# Compute 64 distances in parallel
f = BatchOfSumOfDiracs( positions )   # shape (64, n, d)
g = BatchOfSplineGrid( values )       # shape (64, ...)

d = distance( f, g )                  # shape (64,)
```

---

## What's Next

- [Distributions in depth →](/guide/distributions)
- [Ground metrics →](/guide/ground-metrics)
- [Backend configuration →](/guide/backends)
- [Tutorial: OT plans explained →](/tutorials/ot-plan-intro)
- [Examples gallery →](/examples/)
