---
title: Gaussian Mixture Fit
description: Fit the parameters of Gaussians to a point cloud by minimizing W2, by gradient descent then L-BFGS. Demonstrates the use of gradients.
tags: [gradient, jax, torch, learning]
image: /examples/img/gaussian-mixture-fit.png
order: 2
---

<script setup>
import { withBase } from 'vitepress'
</script>

# Gaussian Mixture Fit

1000 points are drawn from a mixture of 4 Gaussians. We look for the mixture back: the `SumOfGaussians` $\rho_\theta$ that is closest to the point cloud $\mu = \frac{1}{n} \sum_i \delta_{x_i}$ in the Wasserstein sense,

$$
\min_\theta \; W_2^2\big( \mu, \; \rho_\theta \big), \qquad \rho_\theta = \frac{1}{\sum_k a_k} \sum_k a_k \, \mathcal{N}( c_k, s_k^2 I ).
$$

The parameters $\theta$ are the centers $c_k$, the sigmas $s_k$ and the masses $a_k$ of the Gaussians. `sdot.w2` is differentiable with respect to all of them, with JAX or PyTorch: a plain gradient descent already finds them.

Unlike a maximum likelihood fit, nothing here can collapse. A Gaussian that shrinks onto a single point does not make the loss go to $-\infty$: it leaves the other points without mass nearby, which W2 charges for.

## The loss

```python
def loss( centers, sigmas, masses ):
    return sdot.w2( sdot.SumOfDiracs( points ), sdot.SumOfGaussians( centers, sigmas, weights = masses ) )
```

The parameters go in as they are. The masses need not sum to 1: `SumOfGaussians` normalizes them. The domain of the transport is the box that holds the Gaussians up to `support_sigmas = 6` standard deviations, recomputed at each evaluation.

## Gradient descent

The gradient comes from the autodiff of the framework. Starting from 4 wide Gaussians of equal masses in the middle, a fixed step of 0.5 is enough:

::: code-group

```python [JAX]
import jax, jax.numpy as jnp
jax.config.update( "jax_enable_x64", True )

x = jnp.asarray( points )

def loss( centers, sigmas, masses ):
    return sdot.w2( sdot.SumOfDiracs( x ), sdot.SumOfGaussians( centers, sigmas, weights = masses ) )

value_and_grad = jax.jit( jax.value_and_grad( loss, argnums = ( 0, 1, 2 ) ) )

for step in range( 300 ):
    value, ( d_centers, d_sigmas, d_masses ) = value_and_grad( centers, sigmas, masses )
    centers = centers - 0.5 * d_centers
    sigmas  = sigmas  - 0.5 * d_sigmas
    masses  = masses  - 0.5 * d_masses
```

```python [PyTorch]
import torch

x = torch.as_tensor( points )
params = [ torch.tensor( p, requires_grad = True ) for p in ( centers, sigmas, masses ) ]
optimizer = torch.optim.SGD( params, lr = 0.5 )

for step in range( 300 ):
    optimizer.zero_grad()
    centers, sigmas, masses = params
    value = sdot.w2( sdot.SumOfDiracs( x ), sdot.SumOfGaussians( centers, sigmas, weights = masses ) )
    value.backward()
    optimizer.step()
```

:::

W2 goes from 0.195 to 0.0063, and the Gaussians are close to those of the table below.

::: tip The Lloyd step
When the parameters are only the positions $y_i$ of diracs (`SumOfDiracs( y )` against a fixed density), the gradient of $W_2^2$ with respect to $y_i$ is $2 \, m_i \, ( y_i - b_i )$, where $m_i$ is the mass of the dirac and $b_i$ the barycenter of its cell. A step of $1 / ( 2 m_i )$ moves each dirac to its barycenter: that is the Lloyd algorithm of the [Lloyd quantization](./lloyd-quantization) example.
:::

## Going faster: L-BFGS

W2 is smooth in $\theta$: a quasi-Newton method needs far fewer iterations, 40 instead of 300, for a lower W2 (0.0044). The optimizer is the L-BFGS of SciPy: it only asks for the value and the gradient, which JAX or PyTorch computes, on one flat vector of parameters. Its `bounds` keep the sigmas and the masses positive: a negative mass would make W2 unbounded below.

```python
from scipy.optimize import minimize

bounds = [ ( None, None ) ] * 8 + [ ( 1e-3, None ) ] * 4 + [ ( 1e-6, None ) ] * 4   # 8 center coordinates, 4 sigmas, 4 masses
```

::: code-group

```python [JAX]
import jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
jax.config.update( "jax_enable_x64", True )

x = jnp.asarray( points )

def loss( params ):
    centers, sigmas, masses = params
    return sdot.w2( sdot.SumOfDiracs( x ), sdot.SumOfGaussians( centers, sigmas, weights = masses ) )

theta0, unravel = ravel_pytree( ( centers, sigmas, masses ) )   # the starting point, as one vector
value_and_grad = jax.jit( jax.value_and_grad( lambda theta: loss( unravel( theta ) ) ) )

def fun( theta ):
    value, grad = value_and_grad( theta )
    return float( value ), np.asarray( grad )

res = minimize( fun, theta0, jac = True, method = "L-BFGS-B", bounds = bounds )
centers, sigmas, masses = unravel( res.x )
```

```python [PyTorch]
import torch

x = torch.as_tensor( points )

def fun( theta ):
    theta = torch.tensor( theta, requires_grad = True )
    centers, sigmas, masses = theta.split( [ 8, 4, 4 ] )
    value = sdot.w2( sdot.SumOfDiracs( x ), sdot.SumOfGaussians( centers.reshape( 4, 2 ), sigmas, weights = masses ) )
    value.backward()
    return float( value ), theta.grad.numpy()

theta0 = np.concatenate( [ centers.ravel(), sigmas, masses ] )   # the starting point, as one vector
res = minimize( fun, theta0, jac = True, method = "L-BFGS-B", bounds = bounds )
```

:::

`callback = lambda r: ...` gives the iterations ( `r.x`, `r.fun` ): that is how the viewer below was recorded.

Starting from 4 wide Gaussians of equal masses in the middle, both find the mixture the points were drawn from, up to the sampling noise of 1000 points, in 40 iterations:

| | center | sigma | mass |
|---|---|---|---|
| drawn | ( -1.00, -0.60 ) | 0.35 | 0.35 |
| fitted | ( -1.00, -0.60 ) | 0.34 | 0.31 |
| drawn | ( 1.10, -0.40 ) | 0.25 | 0.25 |
| fitted | ( 1.09, -0.43 ) | 0.25 | 0.27 |
| drawn | ( 0.20, 1.00 ) | 0.30 | 0.25 |
| fitted | ( 0.17, 0.99 ) | 0.30 | 0.27 |
| drawn | ( -0.90, 0.90 ) | 0.20 | 0.15 |
| fitted | ( -0.90, 0.92 ) | 0.20 | 0.15 |

<iframe
  :src="withBase('/examples/gaussian-mixture-fit-torch.html') + '?wheel=0&touch=0'"
  title="Gaussian mixture fit by L-BFGS on W2"
  loading="lazy"
  allowfullscreen
  style="width: 100%; height: 60vh; min-height: 380px; border: 0; display: block; margin: 1.5rem 0 0.5rem;"
></iframe>
<p style="text-align: right; margin: 0"><a :href="withBase('/examples/gaussian-mixture-fit-torch.html')" target="_blank">Open in a separate page ↗</a></p>

The points in gray, and for each Gaussian its center, its circle at one sigma (solid) and at two sigmas (dashed). The opacity follows the mass. Use the slider at the top of the viewer (or the play button) to go through the L-BFGS iterations.

## Reproducing the pages

The viewer above is written by [`docs/examples/gaussian_mixture_fit.py`](https://github.com/sdot-team/sdot/blob/main/docs/examples/gaussian_mixture_fit.py), one page per framework, with the picture of the gallery:

```bash
python docs/examples/gaussian_mixture_fit.py              # JAX
python docs/examples/gaussian_mixture_fit.py jax torch    # or both
```
