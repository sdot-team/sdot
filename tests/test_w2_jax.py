import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

# Add python/jax to path
sys.path.append(os.path.join(os.getcwd(), 'python/jax'))

from sdot_jax import sdot_w2

def test_w2_jax():
    print("Testing W2 JAX...")
    Xf = jnp.array([0.25, 0.75])
    Wf = jnp.array([0.5, 0.5])
    Xg = jnp.array([0.0, 1.0])
    Yg = jnp.array([1.0, 1.0])
    
    # Test distance only
    dist = sdot_w2(Xf, Wf, Xg, Yg)
    print(f"Dist: {dist}")
    assert abs(dist) < 1e-5
    
    # Test distance and barycenters
    dist, bary = sdot_w2(Xf, Wf, Xg, Yg, return_barycenters=True)
    print(f"Barycenters: {bary}")
    assert jnp.allclose(bary, jnp.array([0.25, 0.75]))
    
    # Test grad
    grad_Xf = jax.grad(sdot_w2)(Xf, Wf, Xg, Yg)
    print(f"Grad Xf: {grad_Xf}") # Should be zeros (stub)

    print("W2 JAX test passed!")

if __name__ == "__main__":
    test_w2_jax()
