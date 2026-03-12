import jax.numpy as jnp
import pytest
import jax
import sys
import os

# Add build and python/jax to path
sys.path.append(os.path.join(os.getcwd(), 'python/jax'))
sys.path.append(os.path.join(os.getcwd(), 'build'))

from sdot_jax import sdot_l2

# def test_sdot_l2_jax_forward():
#     f = jnp.array([1.0, 2.0, 3.0])
#     g = jnp.array([1.0, 1.0, 1.0])

#     loss = sdot_l2(f, g)

#     expected_loss = (0.0**2 + 1.0**2 + 2.0**2) / 3.0
#     assert abs(loss - expected_loss) < 1e-6

# def test_sdot_l2_jax_backward():
#     f = jnp.array([1.0, 2.0, 3.0])
#     g = jnp.array([1.0, 1.0, 1.0])

#     grad_f = jax.grad(sdot_l2, argnums=0)(f, g)
#     grad_g = jax.grad(sdot_l2, argnums=1)(f, g)

#     expected_grad_f = jnp.array([0.0, 2/3, 4/3])
#     expected_grad_g = jnp.array([0.0, -2/3, -4/3])

#     assert jnp.allclose(grad_f, expected_grad_f)
#     assert jnp.allclose(grad_g, expected_grad_g)
