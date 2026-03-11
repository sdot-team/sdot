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
