from sdot.drivers.compilation.FfiArgInfo import FfiArgInfo
from jax._src.custom_derivatives import CustomVJPPrimal
from jax._src import ad_util
import jax.core as jax_core
import jax.numpy as jnp
import importlib
import numpy
import jax
import re

map_of_plan_methods = {}

class JaxDriver:
    """
    JAX implementation for sdot centralization.
    """
    def __init__( self, normalized_dtype : str, normalized_device: str ):
        self.device = JaxDriver.find_device( normalized_device )
        self.dtype = JaxDriver.find_dtype( normalized_dtype )
        self.itype = jnp.int64
        if normalized_dtype == "FP64":
            jax.config.update( "jax_enable_x64", True )

        self.map_of_plan_methods = map_of_plan_methods

    @property
    def name( self ) -> str:
        return "jax"

    @staticmethod
    def default_normalized_device_for( user_normalized_dtype ):
        platforms = { d.platform for d in jax.devices() }
        if "gpu" in platforms:
            return "cuda:0"
        # Metal (jax-metal) only supports FP32
        if "METAL" in platforms and user_normalized_dtype == "FP32":
            return "metal"
        return "cpu"

    @staticmethod
    def find_device( normalized_device: str ):
        """ find the jax device from a normalized name like cpu, cuda:1, metal """
        if normalized_device.startswith( "cpu" ):
            return jax.devices( "cpu" )[ 0 ]
        if normalized_device.startswith( "cuda" ):
            idx = int( normalized_device.split( ":" )[ 1 ] ) if ":" in normalized_device else 0
            return jax.devices( "gpu" )[ idx ]
        if normalized_device.startswith( "metal" ):
            return jax.devices( "METAL" )[ 0 ]
        raise RuntimeError( f"Unknown device { normalized_device }" )

    @staticmethod
    def find_dtype( normalized_dtype: str ):
        if normalized_dtype == "FP32":
            return jnp.float32
        if normalized_dtype == "FP64":
            return jnp.float64
        raise RuntimeError( f"Unknown dtype { normalized_dtype }" )

    @property
    def array_type( self ):
        return ( jax.Array, jax_core.Tracer )

    @property
    def normalized_dtype( self ):
        return self.normalized_type_for( self.dtype )

    @property
    def normalized_itype( self ) -> str:
        return self.normalized_type_for( self.itype )

    def is_int_dtype( self, dtype ):
        return jnp.issubdtype( dtype, jnp.integer )

    def array( self, data, dtype = None ):
        return jnp.asarray( data, dtype = dtype or self.dtype, device = self.device )

    def t3( self, tensor, dtype = None ):
        """ make a rank 3 tensor """
        return self.tn( tensor, 3, dtype = dtype )

    def t2( self, tensor, dtype = None ):
        """ make a rank 2 tensor """
        return self.tn( tensor, 2, dtype = dtype )

    def t1( self, tensor, dtype = None ):
        """ make a rank 1 tensor """
        return self.tn( tensor, 1, dtype = dtype )

    def t0( self, tensor, dtype = None ):
        """ make a rank 0 tensor """
        return self.tn( tensor, 0, dtype = dtype )

    def tn( self, tensor, ndim, name = None, dtype = None ):
        """ make a rank ndim tensor """
        if tensor is None:
            return tensor

        if dtype is None or dtype is float:
            dtype = self.dtype
        elif dtype is int:
            dtype = self.itype

        if isinstance( tensor, jax.ShapeDtypeStruct ):
            return jnp.empty( [ s or 0 for s in tensor.shape ], dtype = tensor.dtype )

        res = jnp.asarray( tensor, dtype = dtype, device = self.device )

        if ndim is not None and res.ndim != ndim:
            if name is not None:
                raise IndexError( f"expecting for field '{ name }' a { ndim }d tensor, but { res.ndim }d was provided." )
            raise IndexError( f"expecting a { ndim }d tensor, but { res.ndim }d was provided." )

        return res

    def i0( self, tensor ):
        """ make a rank 0 int tensor """
        return self.tn( tensor, 0, dtype = self.itype )

    def i1( self, tensor ):
        """ make a rank 1 int tensor """
        return self.tn( tensor, 1, dtype = self.itype )

    def i2( self, tensor ):
        """ make a rank 2 int tensor """
        return self.tn( tensor, 2, dtype = self.itype )

    def ones( self, shape, dtype = None ):
        return jnp.ones( shape, dtype = dtype or self.dtype, device = self.device )

    def zeros( self, shape, dtype = None ):
        return jnp.zeros( shape, dtype = dtype or self.dtype, device = self.device )

    def linspace( self, a, b, n ):
        return jnp.linspace( a, b, n, dtype = self.dtype, device = self.device )

    def empty( self, shape, dtype = None ):
        return jnp.zeros( shape, dtype = dtype or self.dtype, device = self.device )

    def expand_dims( self, tensor, index ):
        return jnp.expand_dims( tensor, index )

    def repeat( self, tensor, shape ):
        return jnp.tile( tensor, shape )

    def stack( self, tensors, axis ):
        return jnp.stack( tensors, axis=axis )

    def linalg_solve( self, A, b ):
        return jnp.linalg.solve( A, b )

    def moveaxis( self, tensor, source, destination ):
        return jnp.moveaxis( tensor, source, destination )

    def hstack( self, lst ):
        return jnp.hstack( lst )

    def to_numpy( self, t ):
        return numpy.array( t )

    def normalized_type_for( self, dtype ):
        if dtype is None or dtype is float:
            return self.normalized_type_for( self.dtype )

        if dtype is int:
            return self.normalized_type_for( self.itype )

        if dtype == jnp.float32:
            return "FP32"
        if dtype == jnp.float64:
            return "FP64"

        if dtype == jnp.int32:
            return "SI32"
        if dtype == jnp.int64:
            return "SI64"

        raise NotImplementedError( f"for dtype { dtype }" )

    @property
    def normalized_device_type( self ):
        if self.device.platform == "gpu":
            return "cuda"
        return "cpu"

    @property
    def normalized_framework( self ):
        return "jax"

    def is_a_tensor( self, value ):
        return isinstance( value, ( jax.core.Tracer, jax.Array, numpy.ndarray, jax.ShapeDtypeStruct, ad_util.SymbolicZero ) )

    def differentiable_type( self, dtype ):
        if dtype is None or dtype is float:
            return True
        if dtype is int:
            return False
        return not jax.numpy.issubdtype( dtype, jax.numpy.integer )

    def call( self, func_name: str, includes: str | list[ str ], *, no_grad = False, **args ):
        """Call a C++ function via JAX XLA FFI.

        Args may be:
          - Mutable(obj)               — read+write; obj arrays reassigned after call
          - Return(Type, **kwargs)     — produces a new object or tensor
          - plain JAX array            — read-only input
          - int / float / str / ...    — scalar XLA attribute
        """
        if isinstance( includes, str ):
            includes = [ includes ]

        # get a jax compatible set of arg lists
        fai = FfiArgInfo( args, self )

        # check ffi function is registered
        module_name = self._module_name_for( func_name, includes, fai )
        self._register_ffi_target( module_name, func_name, includes, fai, make_backward_binding = not no_grad )

        # forward helper
        def _call_ffi( differentiable_input_values ):
            # update fai content with the actua values
            fai.update_differentiable_input_values_with( differentiable_input_values )

            # make the call
            func = jax.ffi.ffi_call( module_name, fai.output_specs )

            # normalize the output
            ret = func( *fai.input_values )

            # always return a tuple
            if isinstance( ret, jax.Array ):
                return ( ret, )
            if isinstance( ret, tuple ):
                return ret
            return tuple( ret )

        if no_grad:
            outputs = _call_ffi( tuple( input.python_value for input in fai.differentiable_ffi_inputs ) )
        else:
            @jax.custom_vjp
            def my_ffi_op( differentiable_inputs ):
                return _call_ffi( differentiable_inputs )

            def my_ffi_op_fwd( _differentiable_inputs ):
                # With symbolic_zeros = True, JAX wraps each input in CustomVJPPrimal( value, perturbed )
                differentiable_inputs = tuple( v.value if isinstance( v, CustomVJPPrimal ) else v for v in _differentiable_inputs )
                outputs = _call_ffi( differentiable_inputs )

                return outputs, ( differentiable_inputs, outputs )

            def my_ffi_op_bwd( residuals, grads_of_the_outputs ):
                differentiable_inputs, outputs = residuals
                if not isinstance( grads_of_the_outputs, ( tuple, list ) ):
                    grads_of_the_outputs = ( grads_of_the_outputs, )

                bfai = fai.backward_version( self, outputs, grads_of_the_outputs )
                func = jax.ffi.ffi_call( module_name + "_backward", bfai.output_specs )
                ret = func( *bfai.input_values )
                if isinstance( ret, jax.Array ):
                    ret = ( ret, )

                return ( tuple( ret ), )


            my_ffi_op.defvjp( my_ffi_op_fwd, my_ffi_op_bwd, symbolic_zeros = True )

            # --- appel ---
            outputs = my_ffi_op( tuple( input.python_value for input in fai.differentiable_ffi_inputs ) )

        # ret assembly
        res = []
        for call_arg in fai.call_args:
            if call_arg.io_category == 1: # Mutable
                call_arg.update( fai, outputs )
            if call_arg.io_category == 2: # Return
                res.append( call_arg.construct( fai, outputs ) )

        # item or list
        if len( res ) == 0:
            return None
        if len( res ) == 1:
            return res[ 0 ]
        return res

    def ffi_tensor_input_bind_code( self, ndim, dtype ) -> str:
        return f"Arg<xla::ffi::Buffer<{ self.ffi_tensor_type_code( dtype ) }>>"

    def ffi_tensor_input_arg_code( self, ndim, dtype ) -> str:
        return f"xla::ffi::Buffer<{ self.ffi_tensor_type_code( dtype ) }>"

    def ffi_tensor_output_bind_code( self, ndim, dtype ) -> str:
        return f"Ret<xla::ffi::Buffer<{ self.ffi_tensor_type_code( dtype ) }>>"

    def ffi_tensor_output_arg_code( self, ndim, dtype ) -> str:
        return f"xla::ffi::ResultBuffer<{ self.ffi_tensor_type_code( dtype ) }>"

    def ffi_tensor_output_spec( self, shape, dtype ):
        return jax.ShapeDtypeStruct( shape, dtype )

    def ffi_tensor_type_code( self, dtype ) -> str:
        """ C++ jax name for dtype """

        if dtype is None or dtype is float:
            return self.ffi_tensor_type_code( self.dtype )
        if dtype is int:
            return self.ffi_tensor_type_code( self.itype )

        if dtype == jax.numpy.float32:
            return "xla::ffi::F32"
        if dtype == jax.numpy.float64:
            return "xla::ffi::F64"
        if dtype == jax.numpy.int32:
            return "xla::ffi::S32"
        if dtype == jax.numpy.int64:
            return "xla::ffi::S64"
        if dtype == jax.numpy.uint32:
            return "xla::ffi::U32"
        if dtype == jax.numpy.uint64:
            return "xla::ffi::U64"

        raise NotImplementedError( f"for dtype { dtype }" )

    def is_zero_tensor( self, value ):
        return isinstance( value, ad_util.SymbolicZero )

    def _module_name_for( self, func_name: str, includes: list[ str ], args: FfiArgInfo ):
        # get signature
        signature_items = [ func_name ]
        for arg in args.call_args:
            signature_items.append( arg.signature )
        signature_items += includes

        # module name
        from .compilation.encode_base_62 import encode_base_62
        res = re.sub( r'[^\w]', '_', str.join( "__", signature_items ) )
        if len( res ) > 40:
            res = res[ : 40 - 11 ] + encode_base_62( res[ 40 - 11: ] )

        return res

    _registered_ffi_targets = set()

    def _register_ffi_target( self, module_name: str, func_name: str, includes: list[ str ], args: FfiArgInfo, make_backward_binding = True ):
        # already registered ?
        if module_name in JaxDriver._registered_ffi_targets:
            return
        JaxDriver._registered_ffi_targets.add( module_name )

        # else, try to load it from the disk
        from .compilation.force_build import force_build
        if not force_build():
            try:
                self._try_to_import_and_register_ffi_target( module_name, func_name, make_backward_binding )
                return
            except ( ImportError, SystemError ):
                return None

        # else, make the dylib
        self._make_dylib( func_name, includes, args, module_name, make_backward_binding )

        # and try again to import it
        self._try_to_import_and_register_ffi_target( module_name, func_name, make_backward_binding )


    def _try_to_import_and_register_ffi_target( self, module_name: str, func_name: str, make_backward_binding: bool ):
        module = importlib.import_module( "sdot.generated_files." + module_name )
        jax.ffi.register_ffi_target( module_name, getattr( module, func_name )(), platform = "cpu" )
        if make_backward_binding:
            jax.ffi.register_ffi_target( module_name + "_backward", getattr( module, func_name + "_backward" )(), platform = "cpu" )


    def _make_dylib( self, func_name: str, includes: list[ str ], fai: FfiArgInfo, module_name: str, make_backward_binding: bool ):
        # include list
        std_includes = [
            "sdot/jax_ffi_wrappers.h",
            "nanobind/nanobind.h",
        ]

        # declaration
        lines = []
        for include in std_includes + includes:
            lines.append( f"#include <{ include }>" )
        lines.append( "" )
        lines.append( "namespace nb = nanobind;" )
        lines.append( "using namespace sdot;" )
        lines.append( "" )
        lines.append( f"using TF = { self.normalized_dtype };" )
        lines.append( "" )

        # --- forward handler ---
        lines += self._handler_source( func_name, fai )
        lines.append( "" )

        # --- backward handler ---
        if make_backward_binding:
            lines += self._handler_source( func_name + "_backward", fai.backward_version( self ) )
            lines.append( "" )

        lines.append( "" )
        lines.append( "template<typename T>" )
        lines.append( "nb::capsule EncapsulateFfiCall( T *fn ) {" )
        lines.append( "  static_assert( std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>, \"Encapsulated function must be and XLA FFI handler\");" )
        lines.append( "  return nb::capsule( reinterpret_cast<void *>( fn ), \"xla._CUSTOM_CALL_TARGET\" );" )
        lines.append( "}" )
        lines.append( "" )
        lines.append( f"NB_MODULE( { module_name }, m ) {{" )
        lines.append( f"  m.def( \"{ func_name }\", []() {{ return EncapsulateFfiCall( binding_{ func_name } ); }} );" )
        if make_backward_binding:
            lines.append( f"  m.def( \"{ func_name }_backward\", []() {{ return EncapsulateFfiCall( binding_{ func_name }_backward ); }} );" )
        lines.append( "}" )

        #
        from .compilation.make_dylib_from_source import make_dylib_from_source
        return make_dylib_from_source( str.join( "\n", lines ), module_name, [], "yo" )

    def _handler_source( self, func_name: str, fai: FfiArgInfo ) -> list[ str ]:
        #
        lines = []

        # make a structure with the names
        lines.append( f"template<{ str.join( ",", [ f"typename T{ n }" for n in range( len( fai.call_args ) ) ] ) }>" )
        lines.append( f"struct Parameters_{ func_name } {{" )
        for n, call_arg in enumerate( fai.call_args ):
            lines.append( f"    T{ n } { call_arg.attribute_name };" )
        lines.append( "};" )

        # start impl: declaration
        arg_decls = []
        for name, arg in fai.named_ffi_args:
            arg_decls.append( f"{ arg.cpp_type } { name }" )
        lines.append( f"xla::ffi::Error impl_{ func_name }( { str.join( ", ", arg_decls ) } ) {{" )

        # read the validity_mask
        lines.append( "    const PI64 *validity_mask = validity_mask_buffer.typed_data();" )

        # call the function
        lines.append( f"    { func_name }( Parameters_{ func_name }{{" )
        for call_arg in fai.call_args:
            lines.append( f"        .{ call_arg.attribute_name } = { call_arg.assembled_code() }," )
        lines.append( "    } );" )

        # end impl
        lines.append( "    return xla::ffi::Error::Success();" )
        lines.append( "}" )

        # XLA_FFI_DEFINE_HANDLER_SYMBOL
        bind_chain = [ "xla::ffi::Ffi::Bind()" ] + fai.bind_chain
        lines.append( f"XLA_FFI_DEFINE_HANDLER_SYMBOL( binding_{ func_name }, impl_{ func_name }, { str.join( ".", bind_chain ) } );" )

        return lines


    def optimize_using_lbfgs( self, loss, params, max_iter=50, tol_grad=1e-7, on_iter=None ):
        """ small helper to optimize `loss` wrt `params` using L-BFGS (via scipy.optimize).
            - `params`  : JAX array or list of JAX arrays
            - `on_iter` : optional callback( params, iter, grad_norm ) called each iteration
            Returns the optimized params (same type as input).
        """
        import scipy.optimize

        # support single array or list of arrays
        is_list  = isinstance( params, ( list, tuple ) )
        p_list   = list( params ) if is_list else [ params ]
        shapes   = [ p.shape for p in p_list ]
        sizes    = [ int( np.prod( s ) ) for s in shapes ]

        def pack( arrays ):
            return np.concatenate( [ np.array( a ).flatten() for a in arrays ] ).astype( np.float64 )

        def unpack( x_flat ):
            parts, offset = [], 0
            for shape, size in zip( shapes, sizes ):
                parts.append( jnp.asarray( x_flat[ offset : offset + size ].reshape( shape ), dtype = self.dtype, device = self.device ) )
                offset += size
            return parts

        val_and_grad = jax.value_and_grad( lambda *ps: loss( list( ps ) if is_list else ps[ 0 ] ) )

        iter_ref = [ 0 ]

        def f_and_g( x_flat ):
            ps        = unpack( x_flat )
            val, grad = val_and_grad( *ps )
            g_list    = grad if isinstance( grad, ( list, tuple ) ) else [ grad ]
            return float( np.array( val ) ), pack( g_list )

        def callback( x_flat ):
            ps = unpack( x_flat )
            if on_iter:
                _, grad = val_and_grad( *ps )
                g_list   = grad if isinstance( grad, ( list, tuple ) ) else [ grad ]
                grad_norm = float( np.linalg.norm( pack( g_list ) ) )
                on_iter( ps if is_list else ps[ 0 ], iter_ref[ 0 ], grad_norm )
            iter_ref[ 0 ] += 1

        result = scipy.optimize.minimize(
            f_and_g, pack( p_list ),
            method  = 'L-BFGS-B',
            jac     = True,
            callback = callback,
            options  = { 'maxiter': max_iter, 'gtol': tol_grad },
        )

        final = unpack( result.x )
        return final if is_list else final[ 0 ]

