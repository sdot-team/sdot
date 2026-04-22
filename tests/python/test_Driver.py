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

def test_alac_grad():
    from jax._src import test_util as jtu

    def f( x ):
        res = sdot.driver.call( "test_alac", "sdot/cell/test_alac.h", o0 = sdot.Return( sdot.Tensor, [] ), i0 = x, o1 = sdot.Return( sdot.Tensor, [] ), i1 = x )
        return res[ 0 ]

    x = sdot.driver.t0( 3.0 )
    jtu.check_grads( f, ( x, ), order=1, modes=[ "rev" ] )


# import jax

# def loss( x ):
#     res = sdot.driver.call( "test_alac", "sdot/cell/test_alac.h", sdot.Return( sdot.Tensor, [] ), x - 2  )
#     ic( res )
#     return res.item()

# x = sdot.driver.t0( 10 )
# # ic( sdot.driver.call( "test_alac", "sdot/cell/test_alac.h", sdot.Return( sdot.Tensor, [] ), x - 2 ) )
# r = sdot.driver.optimize_using_lbfgs( loss, x )
# ic( x )
# ic( r )
if __name__ == "__main__":
    test_alac_grad()

    # x = sdot.driver.t0( 3.0 )
    # ic( sdot.driver.call( "test_alac", "sdot/cell/test_alac.h", ret = sdot.Return( sdot.Tensor, [] ), inp = x ) )
