from sdot.util import all_equal
import sdot

def test_choices():
    sdot.driver.ftype = "FP64"
    assert sdot.driver.ftype.floating_point == True
    assert sdot.driver.ftype == "FP64"
    assert sdot.driver.ftype != "FP32"

    sdot.driver.ftype = "FP32"
    assert sdot.driver.ftype == "FP32"
    assert sdot.driver.ftype != "FP64"

    sdot.driver.itype = "SI64"
    assert sdot.driver.itype.floating_point == False
    assert sdot.driver.itype == "SI64"
    assert sdot.driver.itype != "SI32"

    sdot.driver.itype = "SI32"
    assert sdot.driver.itype == "SI32"
    assert sdot.driver.itype != "SI64"

    sdot.driver.device = "gpu"
    assert( sdot.driver.device == "CudaGpu" )
    assert( sdot.driver.device == "CudaGpu:0" )

    sdot.driver.framework = "jax"
    assert( sdot.driver.framework == "Jax" )
    assert( sdot.driver.framework != "Torch" )

    sdot.driver.framework = "pytorch"
    assert( sdot.driver.framework == "Torch" )
    assert( sdot.driver.framework != "Jax" )

def test_ffi_basic():
    input = sdot.driver.array( [ 3. ] )
    assert all_equal( sdot.driver.call(
        """
        p.output[ 0 ] = DECAYED_TYPE_OF( p.output.size() )::value;
        p.output[ 1 ] = p.dim;
        for( PI i = 0; i < p.input.size(); ++i ) {
            p.output[ 2 * i + 2 ] = 2 * p.input[ i ];
            p.output[ 2 * i + 3 ] = 3 * p.input[ i ];
        }
        """,
        output = sdot.Return( sdot.Tensor( "2 + 2 * dim", ct_variables = [ "dim" ] ), dim = input.size ),
        input = input
    ), [ 4, 1, 6, 9 ] )


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
#         a : sdot.Tensor( "nb_points[ smurf, dim ]", "smurf", "dim", ct_variables = [ "dim" ] )
#         b : sdot.Tensor( "smurf", "dim" )

#     yo = Yo( b = [ [ 1 ] ] )
#     res = sdot.driver.call( "yo", "sdot/test/yo.h", args = { "ret": sdot.Return( sdot.Tensor( "nb_elems[]", "dim", ct_variables = [ "dim" ] ), max_of_nb_elems = 5, dim = 3 ), "inp": yo }, axes = { "dim": 2 }, grad = False )
#     info( res )

def test_mlir_basic():
    info( sdot.driver.call( "p.output = p.input[ 0 ] + p.input[ 1 ];", output = sdot.Return( sdot.Tensor() ), input = sdot.driver.array( [ 3., 4. ] ) ) )

def test_growing_capacity():
    import jax

    # Eager mode: capacity auto-grows (2 → 4 → 8), returns exact-sized result
    res = sdot.driver.call(
        "for( PI i = 0; i < 8; ++i ) p.output[ p.n++ ] = i;",
        mlir = True,
        output = sdot.Return( sdot.Tensor( "n[]" ), max_of_n = 2 )
    )
    info( res )
    assert list( res ) == list( range( 8 ) )

    # JIT mode with sufficient capacity: no overflow, returns full pre-allocated buffer
    @jax.jit
    def f( input ):
        return sdot.driver.call(
            "for( PI i = 0; i < p.input.size(); ++i ) p.output[ p.n++ ] = i;",
            mlir = True,
            output = sdot.Return( sdot.Tensor( "n[]" ), max_of_n = 8 ),
            input = input
        )

    info( f( sdot.driver.ones( 8 ) ) )

    # # JIT mode with overflow: raises CapacityOverflow (possibly wrapped in JaxRuntimeError)
    # @jax.jit
    # def f_small( input ):
    #     return sdot.driver.call(
    #         "for( PI i = 0; i < p.input.size(); ++i ) p.output[ p.n++ ] = i;",
    #         mlir = True,
    #         output = sdot.Return( sdot.Tensor( "n[]" ), max_of_n = 2 ),
    #         input = input
    #     )

    # try:
    #     f_small( sdot.driver.ones( 16 ) )
    #     raise AssertionError( "expected CapacityOverflow" )
    # except AssertionError:
    #     raise
    # except Exception as e:
    #     txt = sdot.driver.is_capacity_overflow( e )
    #     assert txt, f"unexpected exception type: { type( e ) }: { e }"
    #     print( "----------->",txt )


def test_grad():
    def f( i0, i1 ):
        return sdot.driver.call(
            "info( p.o0.is_valid() ); info( p.o1.is_valid() ); if ( p.o0.is_valid() ) p.o0 = 2 * p.i0; if ( p.o1.is_valid() ) p.o1 = 3 * p.i1;",
            "info( p.output_grad_for_i0.is_valid(), p.input_grad_for_o0.not_surely_null(), p.output_grad_for_i1.is_valid(), p.input_grad_for_o1.not_surely_null() ); p.output_grad_for_i0 = 2 * p.input_grad_for_o0; p.output_grad_for_i1 = 3 * p.input_grad_for_o1;", # info( p.o0.is_valid() ); info( p.o1.is_valid() ); if ( p.o0.is_valid() )
            o0 = sdot.Return( sdot.Tensor() ),
            o1 = sdot.Return( sdot.Tensor() ),
            i0 = i0,
            i1 = i1,
        )
    i0 = sdot.driver.array( 10. )
    i1 = sdot.driver.array( 20. )
    import jax
    primals, f_vjp = jax.vjp( f, i0, i1 )
    info( primals )  # (20., 60.)
    cotangents = [ jax.numpy.ones_like( p ) for p in primals ]
    info( f_vjp( cotangents ) )  # (2., 3.)


def test_grad_symbolic_zero():
    """When only o0 is used in the loss, JAX sends a SymbolicZero for o1's cotangent.
    C++ should see input_grad_for_o1.not_surely_null() == false."""
    import jax

    def f( i0, i1 ):
        return sdot.driver.call(
            "if ( p.o0.is_valid() ) p.o0 = 2 * p.i0; if ( p.o1.is_valid() ) p.o1 = 3 * p.i1;",
            # grad only if cotangent is non-null; returns 0 otherwise (JAX gets uninitialized buf, but won't use it)
            "info( p.input_grad_for_o0.not_surely_null() );"
            "info( p.input_grad_for_o1.not_surely_null() );"
            "if ( p.input_grad_for_o0.not_surely_null() ) p.output_grad_for_i0 = 2 * p.input_grad_for_o0;"
            "if ( p.input_grad_for_o1.not_surely_null() ) p.output_grad_for_i1 = 3 * p.input_grad_for_o1;",
            o0 = sdot.Return( sdot.Tensor() ),
            o1 = sdot.Return( sdot.Tensor() ),
            i0 = i0,
            i1 = i1,
        )

    i0 = sdot.driver.array( 10. )
    i1 = sdot.driver.array( 20. )

    # only o0 used → o1's cotangent is SymbolicZero → input_grad_for_o1 invalid in C++
    gi0, gi1 = jax.grad( lambda i0, i1: f( i0, i1 )[ 0 ], argnums = ( 0, 1 ) )( i0, i1 )
    info( gi0, gi1 )
    assert float( gi0 ) == 2.0
    assert float( gi1 ) == 0.0  # i1 doesn't affect o0, gradient is 0


def test_grad_perturbed():
    """When differentiating w.r.t. only i0, JAX sets perturbed=False for i1.
    C++ should see output_grad_for_i1.is_valid() == false."""
    import jax

    def f( i0, i1 ):
        return sdot.driver.call(
            "if ( p.o0.is_valid() ) p.o0 = 2 * p.i0 + 3 * p.i1;",
            # skip grad computation for non-requested inputs
            "info( p.output_grad_for_i0.is_valid() );"
            "info( p.output_grad_for_i1.is_valid() );"
            "info( p.input_grad_for_o0.not_surely_null() );"
            "info( p.input_grad_for_o0.not_surely_null() );"
            "if ( p.output_grad_for_i0.is_valid() ) p.output_grad_for_i0 = 2 * p.input_grad_for_o0;"
            "if ( p.output_grad_for_i1.is_valid() ) p.output_grad_for_i1 = 3 * p.input_grad_for_o0;",
            o0 = sdot.Return( sdot.Tensor() ),
            i0 = i0,
            i1 = i1,
        )

    i0 = sdot.driver.array( 10. )
    i1 = sdot.driver.array( 20. )

    # only i0 perturbed → output_grad_for_i1 is invalid in C++
    gi0 = jax.grad( f, argnums = 0 )( i0, i1 )
    info( gi0 )
    assert float( gi0 ) == 2.0


def test_gpu_basic():
    if sdot.driver.available_gpus:
        sdot.driver.device = "gpu"
    info( sdot.driver.device )
    #info( sdot.driver.call( "arch.run_single( [p] HD () { p.output.item() = double( p.input[ 0 ] + p.input[ 1 ] ); } );", output = sdot.Return( sdot.Tensor() ), input = sdot.driver.array( [ 3., 4. ] ) ) )
    # info( sdot.driver.call( "cudaMemcpyAsync( p.output.data(), p.input.data(), sizeof( FP64 ), cudaMemcpyDeviceToDevice, stream );", output = sdot.Return( sdot.Tensor() ), input = sdot.driver.array( [ 3., 4. ] ) ) )
    # info( sdot.driver.call( "using P = DECAYED_TYPE_OF( p ); run_sequential( Range( 1 ), [] HD ( int, P p ) mutable { p.output[ 0 ] = p.input[ 1 ]; }, p );", output = sdot.Return( sdot.Tensor( "dim", ct_variables = [ "dim" ] ), dim = 2 ), input = sdot.driver.array( [ 3., 4. ] ) ) )
    info( sdot.driver.call( """
        using P = DECAYED_TYPE_OF( p );
        run_sequential( Range( 1 ), [] HD ( int, P p ) mutable {
            p.output[ 0 ] = 10;
            p.output[ 0 ] += p.input[ 1 ];
        }, p );
    """, output = sdot.Return( sdot.Tensor( "dim", ct_variables = [ "dim" ] ), dim = 2 ), input = sdot.driver.array( [ 3., 4. ] ) ) )

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
    # test_choices()
    # test_ffi_basic()
    # test_alac_grad()
    # test_mlir_basic()
    # test_growing_capacity()
    # test_grad()
    # test_grad_symbolic_zero()
    # test_grad_perturbed()
    test_gpu_basic()
    # test_codegen()

    # x = sdot.driver.t0( 3.0 )
    # ic( sdot.driver.call( "test_alac", "sdot/cell/test_alac.h", ret = sdot.Return( sdot.Tensor, [] ), inp = x ) )
