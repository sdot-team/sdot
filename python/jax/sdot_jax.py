import jax
import jax.numpy as jnp
import numpy as np
import sdot_jax_cpp

@jax.custom_jvp
def sdot_w2_all(Xf, Wf, Xg, Yg):
    if Xf.ndim == 1:
        res_shape = ()
    else:
        res_shape = (Xf.shape[0],)
        
    res = np.zeros(res_shape, dtype=np.float32)
    barycenters = np.zeros(Xf.shape, dtype=np.float32)
    
    sdot_jax_cpp.w2_cpu(np.array(Xf), np.array(Wf), np.array(Xg), np.array(Yg), res, barycenters)
    return jnp.array(res), jnp.array(barycenters)

@sdot_w2_all.defjvp
def sdot_w2_jvp(primals, tangents):
    Xf, Wf, Xg, Yg = primals
    primal_out = sdot_w2_all(Xf, Wf, Xg, Yg)
    return primal_out, (jnp.zeros_like(primal_out[0]), jnp.zeros_like(primal_out[1]))

def sdot_w2(Xf, Wf, Xg, Yg, return_barycenters=False):
    dist, bary = sdot_w2_all(Xf, Wf, Xg, Yg)
    if return_barycenters:
        return dist, bary
    return dist

@jax.custom_jvp
def sdot_l2(f, g):
    res = np.zeros((), dtype=np.float32)
    sdot_jax_cpp.l2_cpu(np.array(f), np.array(g), res)
    return jnp.array(res)

@sdot_l2.defjvp
def sdot_l2_jvp(primals, tangents):
    f, g = primals
    df, dg = tangents
    primal_out = sdot_l2(f, g)
    
    grad_f = np.zeros(f.shape, dtype=np.float32)
    grad_g = np.zeros(g.shape, dtype=np.float32)
    sdot_jax_cpp.l2_backward_cpu(np.array(f), np.array(g), grad_f, grad_g)
    
    # JVP is grad_f * df + grad_g * dg
    tangent_out = jnp.sum(jnp.array(grad_f) * df) + jnp.sum(jnp.array(grad_g) * dg)
    return primal_out, tangent_out
