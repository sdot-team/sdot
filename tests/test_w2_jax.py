import jax.numpy as jnp
import pytest
import jax
import sys
import os

# Add build and python/jax to path
sys.path.append(os.path.join(os.getcwd(), 'python/jax'))
sys.path.append(os.path.join(os.getcwd(), 'build'))

# from sdot_jax import sdot_w2

# def test_w2_jax_forward():
#     Xf = jnp.array([0.25, 0.75])
#     Wf = jnp.array([0.5, 0.5])
#     Xg = jnp.array([0.0, 1.0])
#     Yg = jnp.array([1.0, 1.0])

#     # Test distance only
#     dist = sdot_w2(Xf, Wf, Xg, Yg)
#     assert abs(dist) < 1e-5

#     # Test distance and barycenters
#     dist, bary = sdot_w2(Xf, Wf, Xg, Yg, return_barycenters=True)
#     assert jnp.allclose(bary, jnp.array([0.25, 0.75]))

# def test_w2_jax_grad():
#     Xf = jnp.array([0.25, 0.75])
#     Wf = jnp.array([0.5, 0.5])
#     Xg = jnp.array([0.0, 1.0])
#     Yg = jnp.array([1.0, 1.0])

#     grad_fn = jax.grad(sdot_w2)
#     grad_Xf = grad_fn(Xf, Wf, Xg, Yg)

#     # Stub returns zeros
#     assert jnp.allclose(grad_Xf, jnp.zeros_like(Xf))
