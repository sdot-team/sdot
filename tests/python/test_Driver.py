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

    ic( sdot.driver.call( "test_alac", "sdot/cell/test_alac.h", sdot.Return( sdot.Tensor, [] ), sdot.driver.t0( 10 ), sdot.Return( sdot.Tensor, [] ) ) )

    # def loss( x ):
    #     return sdot.driver.call( "test_alac", "sdot/cell/test_alac.h", sdot.Return( sdot.Tensor, [] ), x - 2  )

    # x = sdot.driver.t0( 10 )
    # r = sdot.driver.optimize_using_lbfgs( loss, x )
    # ic( r )
    # # ic( x )
