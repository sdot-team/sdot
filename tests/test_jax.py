import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

# Add python/jax to path
sys.path.append(os.path.join(os.getcwd(), 'python/jax'))

from sdot_jax import sdot_l2

def test_sdot_l2_jax():
    print("Testing JAX...")
    f = jnp.array([1.0, 2.0, 3.0])
    g = jnp.array([1.0, 1.0, 1.0])
    
    # Test forward
    loss = sdot_l2(f, g)
    print(f"Loss: {loss}")
    
    expected_loss = (0.0**2 + 1.0**2 + 2.0**2) / 3.0
    assert abs(loss - expected_loss) < 1e-6
    
    # Test backward
    grad_f = jax.grad(sdot_l2, argnums=0)(f, g)
    grad_g = jax.grad(sdot_l2, argnums=1)(f, g)
    
    print(f"Grad F: {grad_f}")
    print(f"Grad G: {grad_g}")
    
    expected_grad_f = jnp.array([0.0, 2/3, 4/3])
    expected_grad_g = jnp.array([0.0, -2/3, -4/3])
    
    assert jnp.allclose(grad_f, expected_grad_f)
    assert jnp.allclose(grad_g, expected_grad_g)
    print("JAX test passed!")

if __name__ == "__main__":
    test_sdot_l2_jax()
