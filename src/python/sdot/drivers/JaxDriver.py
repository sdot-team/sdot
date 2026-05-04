from jax._src.custom_derivatives import CustomVJPPrimal
from jax._src import ad_util
import jax.core as jax_core
import jax.numpy as jnp

from sdot.compilation.CallArgs import CallArgs

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
    def find_dtype( dtype ):
        if dtype == "FP32":
            return jnp.float32
        if dtype == "FP64":
            return jnp.float64
        if dtype == "PI8":
            return jnp.uint8
        if dtype == "PI32":
            return jnp.uint32
        if dtype == "PI64":
            return jnp.uint64

        return numpy.dtype( dtype )

    @property
    def array_type( self ):
        return ( jax.Array, jax_core.Tracer )

    @property
    def normalized_dtype( self ):
        return self.normalized_type_for( self.dtype )

    @property
    def normalized_itype( self ) -> str:
        return self.normalized_type_for( self.itype )

    @property
    def uint64( self ):
        return jnp.uint64

    def is_int_dtype( self, dtype ):
        return jnp.issubdtype( dtype, jnp.integer )

    def array( self, data, dtype = None ):
        if data is None:
            return None
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
        return jnp.zeros( shape, dtype = self.find_dtype( dtype ) or self.dtype, device = self.device )

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

        if dtype == jnp.uint32:
            return "PI32"
        if dtype == jnp.uint64:
            return "PI64"

        if dtype in [ "FP32", "FP64", "PI32", "PI64", "SI32", "SI64" ]:
            return dtype

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
        if isinstance( dtype, str ):
            if dtype.startswith( "FP" ):
                return True
            if dtype.startswith( "PI" ) or dtype.startswith( "SI" ):
                return False
        return not jax.numpy.issubdtype( dtype, jax.numpy.integer )

    def call( self, func_name: str, includes: str | list[ str ], grad = True, parameters_struct = None, **args ):
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
        fai = CallArgs.factory( args = args )

        # check ffi function is registered
        module_name = self._module_name_for( func_name, includes, fai )
        self._register_ffi_target( module_name, func_name, includes, fai, make_backward_binding = grad )

        # forward helper
        def _call_ffi( differentiable_input_values ):
            # update fai content with the actual values
            fai.update_differentiable_input_values_with( differentiable_input_values )

            # loop until capacities are large enough
            while True:
                # make the call
                func = jax.ffi.ffi_call( module_name, fai.ffi_outputs )

                # normalize the output
                ret = func( *fai.ffi_inputs, **fai.ffi_attributes )

                # break if ok
                u64_output = ret[ fai.index_u64_output ]
                if u64_output[ fai.index_dynamic_size_exception ] == 0:
                    break

                # else delete outputs
                del ret

                # get faulty data
                da = fai.dynamix_axes[ u64_output[ fai.index_dynamic_size_exception + 0 ] - 1 ]
                needed_size = int( u64_output[ fai.index_dynamic_size_exception + 1 ] )
                faulty_axis_name = "max_of_" + da.name_in_parent


                # update output shape
                made_a_change = False
                for tensor_output in fai.tensor_outputs:
                    if tensor_output.parent() != da.parent():
                        continue
                    if faulty_axis_name in tensor_output.ctor_kwargs:
                        old_value = int( tensor_output.ctor_kwargs[ faulty_axis_name ] )
                        new_value = max( needed_size, 2 * old_value )

                        print( "resize", da.name_in_parent, "to", new_value )

                        tensor_output.ctor_kwargs[ faulty_axis_name ] = new_value
                        made_a_change = True

                assert( made_a_change )

            # always return a tuple
            if isinstance( ret, jax.Array ):
                return ( ret, )
            if isinstance( ret, tuple ):
                return ret
            return tuple( ret )

        if not grad:
            outputs = _call_ffi( tuple( fai.differentiable_ffi_inputs ) )
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
                ret = func( *bfai.input_values, **bfai.attributes )
                if isinstance( ret, jax.Array ):
                    ret = ( ret, )

                return ( tuple( ret ), )


            my_ffi_op.defvjp( my_ffi_op_fwd, my_ffi_op_bwd, symbolic_zeros = True )

            # --- appel ---
            outputs = my_ffi_op( tuple( input.python_value for input in fai.differentiable_ffi_inputs ) )

        # ret assembly
        fai.update_objects( outputs )
        res = fai.assemble_returns()

        # item or list
        if len( res ) == 0:
            return None
        if len( res ) == 1:
            return res[ 0 ]
        return res

    def ffi_tensor_input_bind_code( self, ndim, dtype ) -> str:
        return f"Arg<xla::ffi::Buffer<{ self.ffi_tensor_type_code( dtype ) }>>()"

    def ffi_tensor_input_arg_code( self, ndim, dtype ) -> str:
        return f"xla::ffi::Buffer<{ self.ffi_tensor_type_code( dtype ) }>"

    def ffi_tensor_output_bind_code( self, ndim, dtype ) -> str:
        return f"Ret<xla::ffi::Buffer<{ self.ffi_tensor_type_code( dtype ) }>>()"

    def ffi_tensor_output_arg_code( self, ndim, dtype ) -> str:
        return f"xla::ffi::ResultBuffer<{ self.ffi_tensor_type_code( dtype ) }>"

    def ffi_tensor_output_spec( self, shape, dtype ):
        return jax.ShapeDtypeStruct( shape, self.find_dtype( dtype ) )

    def ffi_parameter_bind_code( self, dtype, name: str ) -> str:
        return f"Attr<{ self.normalized_type_for( dtype ) }>( \"{ name }\" )"

    def ffi_tensor_type_code( self, dtype ) -> str:
        """ C++ jax name for dtype """

        if dtype is None or dtype is float:
            return self.ffi_tensor_type_code( self.dtype )
        if dtype is int:
            return self.ffi_tensor_type_code( self.itype )

        if isinstance( dtype, str ):
            return self.ffi_tensor_type_code( self.find_dtype( dtype ) )

        if dtype == jax.numpy.float32 or dtype == "FP32":
            return "xla::ffi::F32"
        if dtype == jax.numpy.float64 or dtype == "FP64":
            return "xla::ffi::F64"
        if dtype == jax.numpy.int32 or dtype == "SI32":
            return "xla::ffi::S32"
        if dtype == jax.numpy.int64 or dtype == "SI64":
            return "xla::ffi::S64"
        if dtype == jax.numpy.uint8 or dtype == "PI8":
            return "xla::ffi::U8"
        if dtype == jax.numpy.uint32 or dtype == "PI32":
            return "xla::ffi::U32"
        if dtype == jax.numpy.uint64 or dtype == "PI64":
            return "xla::ffi::U64"

        raise NotImplementedError( f"for dtype { dtype }" )

    def is_zero_tensor( self, value ):
        return isinstance( value, ad_util.SymbolicZero )

    def _module_name_for( self, func_name: str, includes: list[ str ], main_list: CallArgs ):
        # get signature
        base_signature = [ func_name ] + [ name + "_" + arg.signature() for name, arg in main_list.sub_dict.items() ] + includes

        # module name
        from sdot.util.encode_base_62 import encode_base_62
        res = re.sub( r'[^\w]', '_', str.join( "__", base_signature ) )
        if len( res ) > 40:
            res = res[ : 40 - 11 ] + encode_base_62( res[ 40 - 11: ] )

        return res

    _registered_ffi_targets = set()

    def _register_ffi_target( self, module_name: str, func_name: str, includes: list[ str ], args: CallArgs, make_backward_binding = True ):
        # already registered ?
        if module_name in JaxDriver._registered_ffi_targets:
            return
        JaxDriver._registered_ffi_targets.add( module_name )

        # else, try to load it from the disk
        from ..compilation.force_build import force_build
        if not force_build():
            try:
                self._try_to_import_and_register_ffi_target( module_name, func_name, make_backward_binding )
                return
            except ( ImportError, SystemError ):
                return None

        # else, make the dylib
        includes_set = set( includes )
        self._make_dylib( func_name, includes_set, args, module_name, make_backward_binding )

        # and try again to import it
        self._try_to_import_and_register_ffi_target( module_name, func_name, make_backward_binding )


    def _try_to_import_and_register_ffi_target( self, module_name: str, func_name: str, make_backward_binding: bool ):
        module = importlib.import_module( "sdot.generated_files." + module_name )
        jax.ffi.register_ffi_target( module_name, getattr( module, func_name )(), platform = "cpu" )
        if make_backward_binding:
            jax.ffi.register_ffi_target( module_name + "_backward", getattr( module, func_name + "_backward" )(), platform = "cpu" )


    def _make_dylib( self, func_name: str, includes: set, fai: CallArgs, module_name: str, make_backward_binding: bool ):
        # generate structs
        already_visited = set()
        fai.generate_structures( already_visited )

        # include list
        includes.add( "sdot/jax_ffi_wrappers.h" )
        includes.add( "nanobind/nanobind.h" )

        # declaration
        lines = []
        lines.append( "" )
        lines.append( "namespace nb = nanobind;" )
        lines.append( "using namespace sdot;" )
        lines.append( "" )
        lines.append( "using Arch = Cpu;" )
        lines.append( f"using TF = { self.normalized_dtype };" )
        lines.append( "" )

        # --- forward handler ---
        self._handler_source( includes, lines, func_name, fai )
        lines.append( "" )

        # --- backward handler ---
        if make_backward_binding:
            self._handler_source( includes, lines, func_name + "_backward", fai.backward_version( self ) )
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

        include_lines = [ f"#include <{ include }>" for include in sorted( includes, key = lambda s: ( -len( s ), s ) ) ]

        #
        from ..compilation.make_dylib_from_source import make_dylib_from_source
        return make_dylib_from_source( str.join( "\n", include_lines + lines ), module_name, [], "yo" )

    def _handler_source( self, includes, lines, func_name: str, fai: CallArgs ) -> list[ str ]:
        # make a structure with the names
        if fai.parameters_struct is None:
            parameters_struct = f"Parameters_{ func_name }"
            fai.make_parameters_struct( includes, lines, parameters_struct )
        else:
            parameters_struct = fai.parameters_struct
        lines.append( "" )

        # start impl: declaration
        lines.append( f"xla::ffi::Error impl_{ func_name }( { fai.arg_decl() } ) {{" )

        # u8_...
        if len( fai.u8_input_values ):
            lines.append( "    const PI8 *u8_input = u8_input_buffer.typed_data();" )

        if fai.u64_output_size:
            lines.append( "    PI64 *u64_output = u64_output_buffer->typed_data();" )
            lines.append( "    std::memset( u64_output, 0, u64_output_buffer->element_count() * sizeof( PI64 ) );" )

        # conversions
        fai.tensor_conversions( lines )

        # beg try block
        lines.append( "    try {" )

        # call the function
        lines.append( f"        { func_name }( { fai.assembled_code( parameters_struct, "        " )  } );" )

        # end try block

        lines.append( '    } catch ( DynamicSizeException de ) {' )
        lines.append( f'        u64_output[ { fai.index_dynamic_size_exception + 0 } ] = 1 + de.num_dynamic_axis;' )
        lines.append( f'        u64_output[ { fai.index_dynamic_size_exception + 1 } ] = de.needed_size;' )
        lines.append( '    }' )

        # end impl
        lines.append( "    return xla::ffi::Error::Success();" )
        lines.append( "}" )

        # XLA_FFI_DEFINE_HANDLER_SYMBOL
        bind_chain = [ "xla::ffi::Ffi::Bind()" ] + fai.bind_chain()
        lines.append( f"XLA_FFI_DEFINE_HANDLER_SYMBOL( binding_{ func_name }, impl_{ func_name }, { str.join( ".", bind_chain ) } );" )


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

