import sdot
import jax.numpy as jnp
import jax

# Use JAX driver
sdot.driver.framework = "jax"

# Setup test data
# For SDOT 1D, positions must be [batch, diracs, 1]
# Two diracs at 0.2 and 0.8
Xf = jnp.array([[[0.1], [0.8]]], dtype=jnp.float32)
Wf = jnp.array([[0.3, 0.7]], dtype=jnp.float32)

# Target: uniform on [0, 1]
Xg = jnp.array([[0.0, 1.0]], dtype=jnp.float32)
Yg = jnp.array([[1.0, 1.0]], dtype=jnp.float32)

# Test grad of distance w.r.t weights
def loss_weights(wf):
    # Ensure weights are normalized (optional for W2 but good practice)
    # wf = wf / jnp.sum(wf)
    f = sdot.BatchOfSumOfWeightedDiracs(Xf, wf)
    g = sdot.BatchOfPiecewise1dAffineFunctions(Xg, Yg)
    p = sdot.plan(f, g)
    return jnp.sum(p.distances)

# Gradient computation
grad_wf = jax.grad(loss_weights)(Wf)
print("Weights:", Wf)
print("Grad Weights (Potentials):", grad_wf)

# Verification with finite differences
def get_fd_grad(idx):
    eps = 1e-4
    l_plus = loss_weights(Wf.at[0, idx].add(eps))
    l_minus = loss_weights(Wf.at[0, idx].add(-eps))
    return (l_plus - l_minus) / (2 * eps)

print(f"Finite Diff Grad diff (1-0): {get_fd_grad(1) - get_fd_grad(0):.6f}")
print(f"Analytic Grad diff (1-0):    {grad_wf[0, 1] - grad_wf[0, 0]:.6f}")

# Test grad of distance w.r.t target function values Yg
def loss_yg(yg):
    f = sdot.BatchOfSumOfWeightedDiracs(Xf, Wf)
    g = sdot.BatchOfPiecewise1dAffineFunctions(Xg, yg)
    p = sdot.plan(f, g)
    return jnp.sum(p.distances)

grad_yg = jax.grad(loss_yg)(Yg)
print("\nYg:", Yg)
print("Grad Yg:", grad_yg)

# Verification with finite differences
def get_fd_grad_yg(idx):
    eps = 1e-4
    l_plus = loss_yg(Yg.at[0, idx].add(eps))
    l_minus = loss_yg(Yg.at[0, idx].add(-eps))
    return (l_plus - l_minus) / (2 * eps)

# Verification with mass-preserving finite differences
def get_fd_grad_yg_mass_preserving():
    eps = 1e-4
    # Increase Y0, decrease Y1 to keep mass constant in this simple [0, 1] case
    l_plus = loss_yg(Yg.at[0, 0].add(eps).at[0, 1].add(-eps))
    l_minus = loss_yg(Yg.at[0, 0].add(-eps).at[0, 1].add(eps))
    return (l_plus - l_minus) / (2 * eps)

print(f"Finite Diff mass-preserving: {get_fd_grad_yg_mass_preserving():.6f}")
print(f"Analytic mass-preserving:    {grad_yg[0, 0] - grad_yg[0, 1]:.6f}")
