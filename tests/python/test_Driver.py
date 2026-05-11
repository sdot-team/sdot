import sdot

# def test_driver():
#     # will force jax
#     import jax.numpy as jnp

#     assert sdot.driver.normalized_framework == "jax"
#     assert sdot.driver.normalized_dtype == "FP64"
#     assert sdot.driver.user_dtype is None
#     assert sdot.driver.dtype == jnp.float64

#     sdot.driver.dtype = "FP32"
#     assert sdot.driver.normalized_dtype == "FP32"
#     assert sdot.driver.user_dtype == "FP32"
#     assert sdot.driver._instance is None
#     assert sdot.driver.dtype == jnp.float32

#     sdot.driver.framework = "torch"
#     assert sdot.driver.user_dtype == "FP32"
#     assert sdot.driver._instance is None

#     import torch
#     assert sdot.driver.dtype == torch.float32

# def test_alac_grad():
#     from jax._src import test_util as jtu
#     import numpy

#     def f( x ):
#         res = sdot.driver.call( "test_alac", "sdot/cell/test_alac.h",
#             o0 = sdot.Return( sdot.Tensor, [] ),
#             i0 = x,
#             o1 = sdot.Return( sdot.Tensor, [] ),
#             i1 = numpy.array( 34. ) # sdot.UndefinedTensor( [] )
#         )
#         return res[ 0 ]

#     x = sdot.driver.t0( 3.0 )
#     jtu.check_grads( f, ( x, ), order=1, modes=[ "rev" ] )

# def test_codegen():
#     from jax._src import test_util as jtu
#     import numpy

#     class Pouet:
#         t = sdot.driver.t1( [ 3., 4 ] )
#         s = sdot.driver.t1( [ 3., 4 ] )
#         def __init__( self, t = None, s = None ):
#             self.t = sdot.driver.t1( t if t is not None else [ 1., 2 ] )
#             self.s = sdot.driver.t1( s if s is not None else [ 1., 2 ] )

#     # pouet = Pouet(),
#     # cell = sdot.Cell( 2, lambda x: 0 ),
#     # o1 = sdot.Return( sdot.Tensor, [], int ),

#     x = numpy.array( 34 )
#     pouet = Pouet()
#     res = sdot.driver.call( "test_alac", "sdot/cell/test_alac.h",
#         o0 = sdot.Return( sdot.Tensor, [] ),
#         pouet = sdot.Mutable( pouet ),
#         i0 = x,
#     )
#     info( pouet )
#     info( res )


# def test_fields():
#     from sdot.aggregate.aggregate import aggregate

#     @aggregate
#     class Yo:
#         a : sdot.Tensor( "nb_points[ smurf, dim ]", "smurf", "dim", ct_axes = [ "dim" ] )
#         b : sdot.Tensor( "smurf", "dim" )

#     yo = Yo( b = [ [ 1 ] ] )
#     res = sdot.driver.call( "yo", "sdot/test/yo.h", args = { "ret": sdot.Return( sdot.Tensor( "nb_elems[]", "dim", ct_axes = [ "dim" ] ), max_of_nb_elems = 5, dim = 3 ), "inp": yo }, axes = { "dim": 2 }, grad = False )
#     info( res )

def test_ffi_basic():
    input = sdot.driver.array( [ 3. ] )
    info( sdot.driver.call( "info( p.input.size() ); info( p.output.size() ); for( PI i = 0; i < p.input.size(); ++i ) { p.output[ 2 * i + 0 ] = 2 * p.input[ i ]; p.output[ 2 * i + 1 ] = 3 * p.input[ i ]; }", output = sdot.Return( sdot.Tensor( "2 * dim", ct_axes = [ "dim" ] ), dim = input.size ), input = input ) )

    input = sdot.driver.array( [ 3., 4. ] )
    info( sdot.driver.call( "info( p.input.size() ); info( p.output.size() ); for( PI i = 0; i < p.input.size(); ++i ) { p.output[ 2 * i + 0 ] = 2 * p.input[ i ]; p.output[ 2 * i + 1 ] = 3 * p.input[ i ]; }", output = sdot.Return( sdot.Tensor( "2 * dim", ct_axes = [ "dim" ] ), dim = input.size ), input = input ) )

def test_mlir_basic():
    info( sdot.driver.call( "p.output = p.input[ 0 ] + p.input[ 1 ];", mlir = True, output = sdot.Return( sdot.Tensor() ), input = sdot.driver.array( [ 3., 4. ] ) ) )


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
    # test_alac_grad()
    test_ffi_basic()
    test_mlir_basic()
    # test_codegen()

    # x = sdot.driver.t0( 3.0 )
    # ic( sdot.driver.call( "test_alac", "sdot/cell/test_alac.h", ret = sdot.Return( sdot.Tensor, [] ), inp = x ) )
