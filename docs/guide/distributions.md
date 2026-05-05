# Distributions

SDOT works with two kinds of distributions: **sources** (discrete, sum of Dirac masses) and **targets** (continuous densities). Any source can be transported to any target — the library generates the appropriate C++ kernel at compile time.

---

## Sources

Sources represent the discrete side of the semi-discrete OT problem.

### `SumOfDiracs`

Equal-weight Dirac masses. The simplest source — masses are set to `1/n` automatically.

```python
from sdot import SumOfDiracs
import numpy as np

f = SumOfDiracs( np.random.rand( 500, 2 ) )   # 500 points, 2D
```

| Argument | Type | Description |
|---|---|---|
| `positions` | array `(n, d)` | Dirac locations |

### `SumOfWeightedDiracs`

Dirac masses with explicit weights. Useful for partial transport (masses don't have to sum to 1).

```python
from sdot import SumOfWeightedDiracs
import numpy as np

positions = np.random.rand( 50, 2 )
weights   = np.ones( 50 ) / 50

f = SumOfWeightedDiracs( positions, weights )
```

| Argument | Type | Description |
|---|---|---|
| `positions` | array `(n, d)` | Dirac locations |
| `weights` | array `(n,)` | Masses (positive, typically summing to 1) |

---

## Targets

Target distributions are **continuous densities** on some domain. They are **normalized to mass 1 by default** — pass `mass=` to trigger partial transport.

### `PolynomialGrid`

Piecewise polynomial density on a regular axis-aligned grid. Efficient for images and simple analytic densities.

```python
from sdot import PolynomialGrid
import numpy as np

# Uniform density on [0,1]²  (1×1 cell, constant value 1)
uniform = PolynomialGrid( values = [[[1]]] )

# From a grayscale image
img = np.load( "cat.npy" ).astype( float )
g   = PolynomialGrid( values = img )
```

| Argument | Type | Description |
|---|---|---|
| `values` | array `(n₁, …, nₐ)` | Cell values — one scalar per cell (Q₀ basis) |
| `domain` | `[[min…], [max…]]` | Bounding box (default: `[0,1]ᵈ`) |
| `mass` | float | Total mass (default: 1 — normalized automatically) |

### `SplineGrid`

C¹ or higher continuity spline density on a regular grid. More regular than `PolynomialGrid`, supports gradients of the density with respect to its values.

```python
from sdot import SplineGrid
import numpy as np

# 1D natural cubic spline from sampled values
s = SplineGrid( [ 1, 4, 9, 4, 1 ], continuity = 1 )

# 2D spline from random values
g = SplineGrid( np.random.rand( 10, 10 ), continuity = 1 )
```

| Argument | Type | Description |
|---|---|---|
| `values` | array | Spline control values on the grid |
| `continuity` | int | Smoothness order (1 = C¹, 2 = C², …) |
| `domain` | `[[min…], [max…]]` | Bounding box (default: `[0,1]ᵈ`) |
| `mass` | float | Total mass (default: 1) |

### `SumOfGaussians` _(coming soon)_

::: tip Coming soon
Mixture of Gaussians target. Enables analytic integrals over Laguerre cells and is particularly useful for learning problems.

```python
from sdot import SumOfGaussians

g = SumOfGaussians(
    centers     = centers,      # (k, d)
    covariances = covs,         # (k, d, d)
    weights     = weights,      # (k,)
)
```
:::

### `Mesh` _(coming soon)_

::: tip Coming soon
Density defined on a surface or volume mesh (triangles, tetrahedra, …). Supports non-uniform density fields on unstructured grids.
:::

### `PolynomialCells` _(coming soon)_

::: tip Coming soon
Piecewise polynomial density on a polyhedral cell decomposition — the most general target type.
:::

---

## Batch Variants

Every distribution class has a `BatchOf` prefix variant for running multiple independent problems in parallel (GPU-friendly).

```python
from sdot import BatchOfSumOfDiracs, BatchOfSplineGrid, distance

B, n, d = 64, 500, 2

positions = np.random.rand( B, n, d )
values    = np.random.rand( B, 10, 10 )

f = BatchOfSumOfDiracs( positions )   # batch of 64 source distributions
g = BatchOfSplineGrid( values )       # batch of 64 targets

d = distance( f, g )                  # shape (64,) — 64 distances in one call
```

And a `1d` suffix for simplified 1D construction (no need to add an extra axis):

```python
from sdot import SumOfDiracs1d

f = SumOfDiracs1d( [ 0.1, 0.5, 0.9 ] )   # equivalent to SumOfDiracs( [[0.1],[0.5],[0.9]] )
```

---

## Normalization and Partial Transport

By default, all targets are normalized so their total mass equals 1. If the source and target have different total masses, SDOT automatically performs **partial transport** — only the overlapping mass fraction is transported.

```python
# Partial transport: target has mass 0.5, source has mass 1
g = SplineGrid( np.random.rand( 8, 8 ), mass = 0.5 )
```
