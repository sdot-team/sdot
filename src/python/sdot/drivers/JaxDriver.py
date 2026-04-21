import jax.core as jax_core
import jax.numpy as jnp
from pathlib import Path
import numpy as np
import importlib
import ctypes
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
    def int_type( self ): # TODO replace by itype
        return jnp.int64

    @property
    def normalized_itype( self ) -> str:
        return "PI64"

    @property
    def itype( self ) -> str:
        return jnp.int64

    def is_int_dtype( self, dtype ):
        return jnp.issubdtype( dtype, jnp.integer )

    def any_requires_grad( self, tensors ) -> bool:
        return True

    def t3( self, tensor ):
        """ make a rank 3 tensor """
        return self.tn( tensor, 3 )

    def t2( self, tensor ):
        """ make a rank 2 tensor """
        return self.tn( tensor, 2 )

    def t1( self, tensor ):
        """ make a rank 1 tensor """
        return self.tn( tensor, 1 )

    def t0( self, tensor ):
        """ make a rank 0 tensor """
        return self.tn( tensor, 0 )

    def tn( self, tensor, ndim, name = None, dtype = None ):
        """ make a rank ndim tensor """
        if tensor is None:
            return tensor

        if dtype is None or dtype is float:
            dtype = self.dtype
        elif dtype is int:
            dtype = self.int_type
        else:
            raise NotImplementedError( f"for { dtype }" )

        res = jnp.asarray( tensor, dtype = dtype, device = self.device )

        if ndim is not None and res.ndim != ndim:
            if name is not None:
                raise IndexError( f"expecting for field '{ name }' a { ndim }d tensor, but { res.ndim }d was provided." )
            raise IndexError( f"expecting a { ndim }d tensor, but { res.ndim }d was provided." )

        return res

    def i1( self, tensor ):
        """ make a rank 1 int tensor """
        if tensor is None:
            return tensor
        res = jnp.asarray( tensor, dtype = jnp.int64, device = self.device )
        assert res.ndim == 1
        return res

    def ones( self, shape ):
        return jnp.ones( shape, dtype = self.dtype, device = self.device )

    def zeros( self, shape ):
        return jnp.zeros( shape, dtype = self.dtype, device = self.device )

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
        return np.array( t )
        # if isinstance( t, list ):
        # return t.to_numpy()

    def to_nanobind_compatible_objects( self, obj ):
        if isinstance( obj, jax.Array ):
            if self.is_int_dtype( obj.dtype ):
                return [ ( obj, "MI" ) ]
            return [ ( obj, "MF" ) ]
        # if isinstance( obj, jax_core.Tracer ):
        #     if self.is_int_dtype( obj.dtype ):
        #         return [ ( obj., "MI" ) ]
        #     return [ ( obj, "MF" ) ]
        return None

    # def forward( self, forward_func, backward_func, fargs, input_tensors, output_args, output_tensors ):
    #     """Differentiable wrapper using pure_callback (safe for C++/nanobind).
    #         forward_func ( *fargs ) -> None  (fills output_tensors in-place via C++)
    #         backward_func( *fargs, *grad_inputs, *grad_outputs ) -> None
    #         New JAX arrays are written back into output_args via write_back_diffentiable_tensors.
    #     """
    #     np_dtype = np.dtype( self.dtype )
    #     n_out = len( output_tensors )

    #     # id-based mapping: which farg belongs to output_tensors / input_tensors / other JAX array
    #     out_id = { id( t ): i for i, t in enumerate( output_tensors ) }
    #     in_id = { id( t ): i for i, t in enumerate( input_tensors ) }

    #     def build_np_fargs( np_outputs, np_inputs ):
    #         result = []
    #         for a in fargs:
    #             if isinstance( a, ( jax.Array, jax_core.Tracer ) ) or id( a ) in out_id or id( a ) in in_id:
    #                 i_out = out_id.get( id( a ) )
    #                 i_in = in_id.get( id( a ) )
    #                 if i_out is not None:
    #                     result.append( np_outputs[ i_out ] )
    #                 elif i_in is not None:
    #                     result.append( np_inputs[ i_in ] )
    #                 else:
    #                     result.append( np.asarray( a ) )  # int / static tensors
    #             elif hasattr( a, "shape" ): # UndefinedTensor
    #                 shape = [ ( s if s is not None else 0 ) for s in a.shape ]
    #                 result.append( np.zeros( shape, dtype = a.dtype or np_dtype ) )
    #             else:
    #                 result.append( a )
    #         return result

    #     out_shapes = tuple( jax.ShapeDtypeStruct( t.shape, getattr( t, "dtype", self.dtype ) or self.dtype ) for t in output_tensors )
    #     inp_shapes = tuple( jax.ShapeDtypeStruct( t.shape, getattr( t, "dtype", self.dtype ) or self.dtype ) for t in input_tensors )

    #     @jax.custom_vjp
    #     def op( *jax_inputs ):
    #         def fwd_cb( *np_inputs ):
    #             from ..object_with_tensors.UndefinedTensor import UndefinedTensor
    #             np_outputs = [ np.empty( t.shape, dtype = t.dtype ) for t in out_shapes ]
    #             # We need to map UndefinedTensor to None for the nanobind call
    #             call_args = []
    #             for a in build_np_fargs( np_outputs, list( np_inputs ) ):
    #                 if isinstance( a, UndefinedTensor ):
    #                     call_args.append( None )
    #                 else:
    #                     call_args.append( np.asarray( a ) )
    #             forward_func( *call_args )
    #             return tuple( np_outputs )
    #         return jax.pure_callback( fwd_cb, out_shapes, *jax_inputs )

    #     def op_fwd( *jax_inputs ):
    #         outputs = op( *jax_inputs )
    #         return outputs, ( *outputs, *jax_inputs )

    #     def op_bwd( residuals, grad_outputs ):
    #         jax_outputs = residuals[ :n_out ]
    #         jax_inputs  = residuals[ n_out: ]

    #         def bwd_cb( *cb_args ):
    #             from ..object_with_tensors.UndefinedTensor import UndefinedTensor
    #             np_outputs      = list( cb_args[ :n_out ] )
    #             np_grad_outputs = list( cb_args[ n_out : 2 * n_out ] )
    #             np_inputs       = list( cb_args[ 2 * n_out: ] )
    #             np_grad_inputs  = [ np.zeros( t.shape, dtype = t.dtype ) for t in inp_shapes ]

    #             # We need to map UndefinedTensor to None for the nanobind call
    #             call_args = []
    #             for a in build_np_fargs( np_outputs, np_inputs ):
    #                 if isinstance( a, UndefinedTensor ):
    #                     call_args.append( None )
    #                 else:
    #                     call_args.append( np.asarray( a ) )

    #             backward_func( *call_args, *np_grad_inputs, *np_grad_outputs )
    #             return tuple( np_grad_inputs )

    #         return jax.pure_callback( bwd_cb, inp_shapes,
    #                                   *jax_outputs, *grad_outputs, *jax_inputs )

    #     op.defvjp( op_fwd, op_bwd )

    #     jax_inputs  = [ jnp.asarray( t, dtype = getattr( t, "dtype", self.dtype ) or self.dtype, device = self.device ) for t in input_tensors ]
    #     new_outputs = op( *jax_inputs )

    #     # write new JAX arrays back into Output/Return objects (JAX arrays are immutable)
    #     new_tuple   = new_outputs if isinstance( new_outputs, tuple ) else ( new_outputs, )
    #     tracked_iter = iter( new_tuple )
    #     for arg in output_args:
    #         arg.write_back_diffentiable_tensors( tracked_iter )

    #     if n_out == 0:
    #         return None
    #     return new_tuple[ 0 ] if n_out == 1 else new_tuple

    @property
    def normalized_dtype( self ):
        return "double" if self.dtype == jnp.float64 else "float"

    @property
    def normalized_device_type( self ):
        if self.device.platform == "gpu":
            return "cuda"
        return "cpu"

    @property
    def normalized_framework( self ):
        return "jax"

    def array( self, data ):
        return jnp.asarray( data, dtype = self.dtype, device = self.device )

    def call( self, func_name: str, includes: str | list[ str ], *args ):
        """Call a C++ function via JAX XLA FFI.

        Args may be:
          - Mutable(obj)               — read+write; obj arrays reassigned after call
          - Return(Type, **kwargs)     — produces a new object or tensor
          - plain JAX array            — read-only input
          - int / float / str / ...    — scalar XLA attribute
        """
        if isinstance( includes, str ):
            includes = [ includes ]

        # check ffi function is registered
        module_name = self._module_name_for( func_name, includes, args )
        self._register_ffi_target( module_name, func_name, includes, args )

        # arg list
        from .compilation.as_jax_ffi_compatible_args import as_jax_ffi_compatible_args
        from .compilation.Return import Return
        jax_args = []
        for arg in args:
            if isinstance( arg, Return ):
                continue
            for jax_arg, _, _ in as_jax_ffi_compatible_args( arg ):
                jax_args.append( jax_arg )

        func = jax.ffi.ffi_call( module_name, jax.ShapeDtypeStruct( [], self.dtype ) )
        return func( *jax_args )

    def cpp_ffi_type_name( self, dtype ):
        if dtype is None or dtype is float:
            return self.cpp_ffi_type_name( self.dtype )

        if dtype is int:
            return self.cpp_ffi_type_name( self.itype )

        if dtype == jnp.float32:
            return "xla::ffi::F32"
        if dtype == jnp.float64:
            return "xla::ffi::F64"
        if dtype == jnp.int32:
            return "xla::ffi::I32"
        if dtype == jnp.int64:
            return "xla::ffi::I64"

        raise NotImplementedError( f"for dtype { dtype }" )

    def normalized_type_for( self, dtype ):
        if dtype is None:
            return self.normalized_type_for( self.dtype )

        if dtype == jnp.float32:
            return "FP32"
        if dtype == jnp.float64:
            return "FP64"

        if dtype == jnp.int32:
            return "PI32"
        if dtype == jnp.int64:
            return "PI64"

        raise NotImplementedError( f"for dtype { dtype }" )

    def cpp_class_name( self, obj ):
        """ TODO: clarify the fact that for jax, cpp_class_name is used only to get the name of the function """
        if isinstance( obj, ( jax_core.Tracer, jax.Array ) ):
            return f"T{ obj.ndim }{ self.normalized_type_for( obj.dtype ) }>>"
        return None

    def as_jax_ffi_compatible_rets( self, obj ):
        if isinstance( obj, ( jax_core.Tracer, jax.Array ) ):
            dtype = self.cpp_ffi_type_name( obj.dtype )
            return [ ( obj, f"Ret<xla::ffi::Buffer<{ dtype }>>", f"xla::ffi::ResultBuffer<{ dtype }>" ) ]
        return None

    def as_jax_ffi_compatible_args( self, obj ):
        if isinstance( obj, ( jax_core.Tracer, jax.Array ) ):
            dtype = self.cpp_ffi_type_name( obj.dtype )
            return [ ( obj, f"Arg<xla::ffi::Buffer<{ dtype }>>", f"xla::ffi::Buffer<{ dtype }>" ) ]
        return None

    def from_jax_ffi_compatible_args( self, obj, flat_arg_iterator ):
        if isinstance( obj, ( jax_core.Tracer, jax.Array ) ):
            return f"tensor_view( CtInt<{ obj.ndim }>(), { next( flat_arg_iterator ) } )"
        return None

    def _module_name_for( self, func_name: str, includes: list[ str ], args ):
        # get signature
        from .compilation.cpp_class_name import cpp_class_name
        signature_items = [ func_name ]
        for arg in args:
            signature_items.append( cpp_class_name( arg ) )
        signature_items += includes

        # module name
        from .compilation.encode_base_62 import encode_base_62
        res = re.sub( r'[^\w]', '_', str.join( "__", signature_items ) )
        if len( res ) > 40:
            res = res[ : 40 - 11 ] + encode_base_62( res[ 40 - 11: ] )

        return res

    _registered_ffi_targets = set()

    def _register_ffi_target( self, module_name: str, func_name: str, includes: list[ str ], args, make_backward_binding = True ):
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


    def _make_dylib( self, func_name: str, includes: list[ str ], args, module_name: str, make_backward_binding: bool ):
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
        lines += self._handler_source( module_name, func_name, args )
        lines.append( "" )

        # --- backward handler ---
        if make_backward_binding:
            lines += self._handler_source( module_name, func_name + "_backward", self._make_instance_of_backwarg_args( args ) )
            lines.append( "" )

        lines.append( "" )
        lines.append( "template<typename T>" )
        lines.append( "nb::capsule EncapsulateFfiCall( T *fn ) {" )
        lines.append( "  static_assert( std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>, \"Encapsulated function must be and XLA FFI handler\");" )
        lines.append( "  return nb::capsule( reinterpret_cast<void *>( fn ), \"xla._CUSTOM_CALL_TARGET\" );" )
        lines.append( "}" )
        lines.append( "" )
        lines.append( f"NB_MODULE( { module_name } , m ) {{" )
        lines.append( f"  m.def( \"{ func_name }\", []() {{ return EncapsulateFfiCall( binding_{ func_name } ); }} );" )
        if make_backward_binding:
            lines.append( f"  m.def( \"{ func_name }_backward\", []() {{ return EncapsulateFfiCall( binding_{ func_name }_backward ); }} );" )
        lines.append( "}" )

        #
        from .compilation.make_dylib_from_source import make_dylib_from_source
        return make_dylib_from_source( str.join( "\n", lines ), module_name, [], "yo" )

    def _make_instance_of_backwarg_args( self, args ):
        """ make a list of arg that can be used to compile the backward version """
        from .compilation.as_jax_ffi_compatible_args import as_jax_ffi_compatible_args
        from .compilation.Mutable import Mutable
        from .compilation.Return import Return

        # transformation of the forward args ()
        res = []
        for arg in args:
            # output => send an instance ()
            if isinstance( arg, Return ):
                res.append( arg.make_fake_instance() )
                continue

            # else, simply add the argument as input
            res.append( arg )

        # additionnal args (gradients)
        for arg in args:
            # output => get the gradients as a set of input tensors
            if isinstance( arg, Return ):
                for obj, _, _ in as_jax_ffi_compatible_args( arg.make_fake_instance() ):
                    if isinstance( obj, ( jax_core.Tracer, jax.Array ) ):
                        if self.is_int_dtype( obj.dtype ):
                            continue
                        res.append( obj )
                continue

            # mutable ?
            if isinstance( arg, Mutable ):
                raise NotImplementedError

            # input => get the gradients
            # On veut en effet prendre la liste des tenseurs, mais il faut faire un ReturnInstance
            for obj, _, _ in as_jax_ffi_compatible_args( arg.make_fake_instance() ):
                if isinstance( obj, ( jax_core.Tracer, jax.Array ) ):
                    if self.is_int_dtype( obj.dtype ):
                        continue
                    res.append( ReturnAs( obj ) )

        return res

    def _handler_source( self, module_name: str, func_name: str, args: list ) -> list[ str ]:
        from sdot.drivers.compilation.from_jax_ffi_compatible_args import from_jax_ffi_compatible_args
        from .compilation.as_jax_ffi_compatible_args import as_jax_ffi_compatible_args

        #
        lines = []

        # start impl: declaration
        num_arg = 0
        impl_args = []
        for arg in args:
            for _, _, cpp_type in as_jax_ffi_compatible_args( arg ):
                impl_args.append( f"{ cpp_type } arg_{ num_arg }" )
                num_arg += 1
        lines.append( f"xla::ffi::Error impl_{ func_name }( { str.join( ", ", impl_args ) } ) {{" )

        # asembly to C++ objects
        cpp_args = [ f"arg_{ num_arg }" for num_arg in range( len( impl_args ) ) ]
        cpp_args_iter = iter( cpp_args )
        for num_cpp_obj, arg in enumerate( args ):
            lines.append( f"    auto obj_{ num_cpp_obj } = { from_jax_ffi_compatible_args( arg, cpp_args_iter ) };" )

        # call the function
        lines.append( f"    { func_name }( { str.join( ", ", [ f"obj_{ num_cpp_obj }" for num_cpp_obj, _ in enumerate( args ) ] ) } );" )

        # end impl
        lines.append( "    return xla::ffi::Error::Success();" )
        lines.append( "}" )

        # XLA_FFI_DEFINE_HANDLER_SYMBOL
        bind_chain = [ "xla::ffi::Ffi::Bind()" ]
        for arg in args:
            for _, bind_type, _ in as_jax_ffi_compatible_args( arg ):
                bind_chain.append( bind_type + "()" )
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

    # def plan( self, bindings, f, g ):
    #     np_dtype = np.dtype( self.dtype )

    #     input_tensors = flat_tensor_list( f ) + flat_tensor_list( g )
    #     dirac_xs = input_tensors[ 0 ]

    #     batch_size = dirac_xs.shape[ 0 ]
    #     nb_diracs  = dirac_xs.shape[ 1 ]
    #     dim        = dirac_xs.shape[ 2 ]

    #     fwd_shapes = (
    #         jax.ShapeDtypeStruct( ( batch_size, ),                self.dtype ),
    #         jax.ShapeDtypeStruct( ( batch_size, nb_diracs, dim ), self.dtype ),
    #         jax.ShapeDtypeStruct( ( batch_size, nb_diracs ),      self.dtype ),
    #         jax.ShapeDtypeStruct( ( batch_size, nb_diracs, 2 ),   self.dtype ),
    #     )
    #     bwd_shapes = tuple(
    #         jax.ShapeDtypeStruct( t.shape, t.dtype ) for t in input_tensors
    #     )

    #     @jax.custom_vjp
    #     def sdot_op( *inputs ):
    #         def fwd_cb( *jax_inputs ):
    #             np_inputs = [ np.asarray( x ) for x in jax_inputs ]

    #             distances   = np.empty( ( batch_size, ),                dtype = np_dtype )
    #             barycenters = np.empty( ( batch_size, nb_diracs, dim ), dtype = np_dtype )
    #             potentials  = np.empty( ( batch_size, nb_diracs ),      dtype = np_dtype )
    #             cuts        = np.empty( ( batch_size, nb_diracs, 2 ),   dtype = np_dtype )

    #             binding_inputs = unflatten_args( f, g, np_inputs )
    #             bindings.forward( *binding_inputs, distances, barycenters, potentials, cuts )
    #             return distances, barycenters, potentials, cuts
    #         return jax.pure_callback( fwd_cb, fwd_shapes, *inputs )

    #     def sdot_op_fwd( *inputs ):
    #         outputs = sdot_op( *inputs )
    #         residuals = ( outputs[ 0 ], outputs[ 1 ], outputs[ 2 ], outputs[ 3 ], *inputs )
    #         return outputs, residuals

    #     def sdot_op_bwd( residuals, grads ):
    #         distances, barycenters, potentials, cuts = residuals[ 0 ], residuals[ 1 ], residuals[ 2 ], residuals[ 3 ]
    #         saved_inputs = residuals[ 4: ]
    #         grad_distances, grad_barycenters, grad_potentials, grad_cuts = grads

    #         def bwd_cb( *jax_args ):
    #             np_args = [ np.asarray( x ) for x in jax_args ]

    #             # Order in pure_callback call below: distances, barycenters, potentials, cuts, *saved_inputs, grad_distances, ...
    #             np_dist, np_bary, np_pot, np_cuts = np_args[ 0 ], np_args[ 1 ], np_args[ 2 ], np_args[ 3 ]
    #             n_in = len( input_tensors )
    #             np_inputs = np_args[ 4 : 4 + n_in ]
    #             np_grad_out = np_args[ 4 + n_in : ] # grad_distances, grad_barycenters, grad_potentials, grad_cuts

    #             flat_grad_inputs = [ np.zeros( t.shape, dtype = np_dtype ) for t in input_tensors ]

    #             binding_inputs = unflatten_args( f, g, np_inputs )
    #             binding_grad_inputs = unflatten_args( f, g, flat_grad_inputs )

    #             # bindings.backward expects: inputs..., distances, barycenters, potentials, cuts, grad_outputs..., grad_inputs...
    #             bindings.backward( *binding_inputs, np_dist, np_bary, np_pot, np_cuts, *np_grad_out, *binding_grad_inputs )

    #             return tuple( flat_grad_inputs )

    #         return jax.pure_callback(
    #             bwd_cb, bwd_shapes,
    #             distances, barycenters, potentials, cuts, *saved_inputs,
    #             grad_distances, grad_barycenters, grad_potentials, grad_cuts
    #         )

    #     sdot_op.defvjp( sdot_op_fwd, sdot_op_bwd )

    #     return BatchOfOtPlans( *sdot_op( *input_tensors ) )
