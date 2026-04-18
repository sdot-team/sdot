import sdot

def test_driver():
    # will force jax
    import jax.numpy as jnp

    assert sdot.driver.normalized_framework == "jax"
    assert sdot.driver.normalized_dtype == "FP64"
    assert sdot.driver.user_dtype is None
    assert sdot.driver.dtype == jnp.float64

    sdot.driver.dtype = "FP32"
    assert sdot.driver.normalized_dtype == "FP32"
    assert sdot.driver.user_dtype == "FP32"
    assert sdot.driver._instance is None
    assert sdot.driver.dtype == jnp.float32

    sdot.driver.framework = "torch"
    assert sdot.driver.user_dtype == "FP32"
    assert sdot.driver._instance is None

    import torch
    assert sdot.driver.dtype == torch.float32


if __name__ == "__main__":
    import jax

    mod = sdot.driver.call( "test_alac", "sdot/cell/test_alac.h", sdot.Return( sdot.Tensor, [] ), 18 )
    ic( mod )
    # def loss( x ):
    #     return sdot.driver.forward( mod.forward, mod.backward, [ [ 1 ] ], x - 3 )[ 0 ].sum()

    # x = sdot.driver.t1( [ 10 ] )
    # r = sdot.driver.optimize_using_lbfgs( loss, x )
    # ic( r )
    # ic( x )
