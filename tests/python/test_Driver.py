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

    def src():
        return sdot.driver.cpp_src( {}, """
            #include <sdot/nanobind_wrappers.h>
            #include <sdot/support/Tensor.h>
            #include <nanobind/stl/tuple.h>

            namespace nb = nanobind;
            using namespace sdot;

            using TF = SDOT_SCALAR_TYPE;

            using AF = nb::ndarray<const TF,nanobind::device::cpu>;
            using MF = nb::ndarray<TF,nanobind::device::cpu>;

            NB_MODULE( SDOT_BINDING_NAME, m ) {
                m.def( "forward", []( MF _res, AF _x ) {
                    auto res = tensor_view_1( _res );
                    auto x = tensor_view_1( _x );
                    for( PI i = 0; i < x.size(); ++i )
                        res[ i ] = x[ i ] * x[ i ];
                } );
                m.def( "backward", []( MF _res, AF _forward, AF _grad, AF _x ) {
                    auto grad = tensor_view_1( _grad );
                    auto res = tensor_view_1( _res );
                    auto x = tensor_view_1( _x );
                    for( PI i = 0; i < x.size(); ++i )
                        res[ i ] = 2 * grad[ i ] * x[ i ];
                } );
            }
        """ )

    mod = sdot.driver.import_bindings( "pouet2000", src )
    def loss( x ):
        return sdot.driver.forward( mod.forward, mod.backward, [ [ 1 ] ], x - 3 )[ 0 ].sum()

    x = sdot.driver.t1( [ 10 ] )
    r = sdot.driver.optimize_using_lbfgs( loss, x )
    ic( r )
    ic( x )
