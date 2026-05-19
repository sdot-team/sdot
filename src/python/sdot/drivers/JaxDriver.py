from typing_extensions import Optional, overload
from typing import TYPE_CHECKING, cast

from jax._src.custom_derivatives import CustomVJPPrimal
from jax._src import ad_util
import jax.core as jax_core
import jax.numpy as jnp

from ..compilation.CallArgsAnalysis import CallArgsAnalysis

from .JaxFramework import JaxFramework
from .Device import Device
from .Dtype import Dtype

import importlib
import numpy
import jax
import re


class JaxDriver:
    """
    JAX implementation for sdot centralization.
    """

    if TYPE_CHECKING:
        device: any

    def __init__( self, framework: JaxFramework, device: Device | None, ftype: Dtype | None, itype: Dtype | None ):
        if device is None:
            device = JaxDriver.default_device_for( ftype )
        if itype is None:
            itype = JaxDriver.default_itype_for( device )
        if ftype is None:
            ftype = JaxDriver.default_ftype_for( device )

        #
        self.framework = framework
        self.device = device
        self.ftype  = ftype
        self.itype  = itype

        # fill device_type for ftype and itype
        itype._driver_version = self.driver_dtype_version( itype.floating_point, itype.signed, itype.size )
        ftype._driver_version = self.driver_dtype_version( ftype.floating_point, ftype.signed, ftype.size )
        assert itype.floating_point == False
        assert ftype.floating_point == True
        if itype.size == 64:
            jax.config.update( "jax_enable_x64", True )

        #
        device.driver_version = device.driver_version_for_jax( jax.devices )

    def driver_dtype_version( self, floating_point, signed, size ):
        if floating_point:
            if size == 16:
                return jnp.float16
            if size == 32:
                return jnp.float32
            if size == 64:
                return jnp.float64
            if size is None:
                return self.ftype._driver_version
            raise ValueError( f"unsupported ftype size: { size }" )

        if signed:
            if size == 8:
                return jnp.int8
            if size == 16:
                return jnp.int16
            if size == 32:
                return jnp.int32
            if size == 64:
                return jnp.int64
            if size is None:
                return self.itype._driver_version
            raise ValueError( f"unsupported itype size: { size }" )

        if size == 8:
            return jnp.uint8
        if size == 16:
            return jnp.uint16
        if size == 32:
            return jnp.uint32
        if size == 64:
            return jnp.uint64

        raise ValueError( f"unsupported itype size: { size }" )

    @staticmethod
    def default_device_for( ftype ):
        platforms = { d.platform for d in jax.devices() }
        if "gpu" in platforms:
            from .CudaGpu import CudaGpu
            return CudaGpu( 0 )

        # Metal (jax-metal) only supports FP32
        if "METAL" in platforms and ftype == "FP32":
            from .AppleGpu import AppleGpu
            return AppleGpu()

        from .Cpu import Cpu
        return Cpu()

    @property
    def available_gpus( self ):
        res = 0
        for device in jax.devices():
            res += "gpu" in device.platform
        return res

    @staticmethod
    def default_ftype_for( device: Device ):
        if device.is_apple_gpu:
            return Dtype.fp( 32 )
        return Dtype.fp( 64 )

    @staticmethod
    def default_itype_for( device: Device ):
        if device.is_apple_gpu:
            return Dtype.si( 32 )
        return Dtype.si( 64 )




    class CapacityOverflow( RuntimeError ):
        """Raised when a dynamic-capacity tensor overflows inside jax.jit.

        Increase max_of_<axis> or call outside jax.jit to enable the automatic retry loop.
        """
        pass


    @staticmethod
    def is_capacity_overflow( e: BaseException ) -> str:
        """True if e is, or wraps (via __context__), a CapacityOverflow.

        Needed because jax.debug.callback exceptions are wrapped in jax.errors.JaxRuntimeError.
        """
        if isinstance( e, jax.errors.JaxRuntimeError ) and len( e.args ):
            for arg in e.args:
                txt = cast( str, arg )
                pos = txt.find( "CapacityOverflow" )
                if pos >= 0:
                    return txt[ pos: ]
        return ""

    @property
    def name( self ) -> str:
        return "jax"

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

    def find_dtype( self, dtype ):
        if self is not None and dtype is None:
            return self.dtype

        if dtype is float:
            return self.dtype

        if dtype is int:
            return self.itype

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

        return jnp.dtype( dtype )

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

    def nb_threads( self, **kwargs ):
        return self.device.nb_threads( **kwargs )


    if TYPE_CHECKING:
        @overload
        def array( self, data: None, dtype = None ) -> None: ...
        @overload
        def array( self, data, dtype = None ) -> jax.Array: ...

    def array( self, data, dtype = None ):
        if data is None:
            return None
        return jnp.asarray( data, dtype = Dtype.factory( dtype or self.ftype ).driver_version, device = self.device.driver_version )

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

        dtype = Dtype.factory( dtype or self.ftype )

        if isinstance( tensor, jax.ShapeDtypeStruct ):
            return jnp.empty( [ s or 0 for s in tensor.shape ], dtype = dtype.driver_version )

        res = jnp.asarray( tensor, dtype = dtype.driver_version, device = self.device.driver_version )

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
        return jnp.ones( shape, dtype = Dtype.factory( dtype or self.ftype ).driver_version, device = self.device.driver_version )

    def zeros( self, shape, dtype = None ):
        return jnp.zeros( shape, dtype = Dtype.factory( dtype or self.ftype ).driver_version, device = self.device.driver_version )

    def linspace( self, a, b, n, dtype = None ):
        return jnp.linspace( a, b, n, dtype = Dtype.factory( dtype or self.ftype ).driver_version, device = self.device.driver_version )

    def empty( self, shape, dtype = None ):
        return jnp.zeros( shape, dtype = Dtype.factory( dtype or self.ftype ).driver_version, device = self.device.driver_version )

    def expand_dims( self, tensor, index ):
        return jnp.expand_dims( tensor, index )

    def repeat( self, tensor, shape ):
        return jnp.tile( tensor, shape )

    def stack( self, tensors, axis ):
        return jnp.stack( tensors, axis=axis )

    def concatenate( self, tensors, axis ):
        return jnp.concatenate( tensors, axis=axis )

    def linalg_solve( self, A, b ):
        return jnp.linalg.solve( A, b )

    def moveaxis( self, tensor, source, destination ):
        return jnp.moveaxis( tensor, source, destination )

    def hstack( self, lst ):
        return jnp.hstack( lst )

    def to_numpy( self, t ):
        return numpy.array( t )

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

    def call( self, code: str, grad_code: str = "", includes: Optional[ list[ str ] ] = None, grad = True, mlir = True, **args ):
        """Call a C++ function via JAX XLA FFI.

        Args may be:
          - Mutable(obj)               — read+write; obj arrays reassigned after call
          - Return(Type, **kwargs)     — produces a new object or tensor
          - plain JAX array            — read-only input
          - int / float / str / ...    — scalar XLA attribute
        """

        if includes is None:
            includes = []

        # argument analysis (get a jax compatible set of arg lists, ...)
        fai = CallArgsAnalysis( args, "Parameters" )

        # check ffi function is registered
        module_name = self._module_name_for( code, grad_code, includes, fai )
        self._register_ffi_target( module_name, code, grad_code, includes, fai )

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
                    assert tensor_output.parent is not None
                    if tensor_output.parent() != da.parent():
                        continue
                    if tensor_output.ctor_kwargs is not None and faulty_axis_name in tensor_output.ctor_kwargs:
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

        if mlir and grad_code:
            from .JaxMlirPrimitive import get_or_create

            def _call_prim( differentiable_inputs ):
                fai.update_differentiable_input_values_with( differentiable_inputs )
                prim = get_or_create( module_name, fai.ffi_outputs, fai.ffi_attributes )
                ret = jax.jit( lambda *args: prim.bind( *args ) )( *fai.ffi_inputs )
                if isinstance( ret, jax.Array ):
                    return ( ret, )
                return tuple( ret )

            @jax.custom_vjp
            def my_mlir_op( differentiable_inputs ):
                return _call_prim( differentiable_inputs )

            def my_mlir_op_fwd( _differentiable_inputs ):
                perturbed_flags = tuple( v.perturbed if isinstance( v, CustomVJPPrimal ) else True for v in _differentiable_inputs )
                differentiable_inputs = tuple( v.value if isinstance( v, CustomVJPPrimal ) else v for v in _differentiable_inputs )
                outputs = _call_prim( differentiable_inputs )
                return outputs, ( differentiable_inputs, outputs, perturbed_flags )

            def my_mlir_op_bwd( residuals, grads_of_the_outputs ):
                differentiable_inputs, outputs, perturbed_flags = residuals
                if not isinstance( grads_of_the_outputs, ( tuple, list ) ):
                    grads_of_the_outputs = ( grads_of_the_outputs, )
                bfai = fai.backward_version( self, outputs, grads_of_the_outputs, "GradParameters", differentiable_inputs, perturbed_flags )
                func = jax.ffi.ffi_call( module_name + "_backward", bfai.ffi_outputs )
                ret = func( *bfai.ffi_inputs, **bfai.ffi_attributes )
                if isinstance( ret, jax.Array ):
                    ret = ( ret, )
                return ( tuple( ret[ ct.num_in_outputs ] for ct in bfai.tensor_outputs ), )

            my_mlir_op.defvjp( my_mlir_op_fwd, my_mlir_op_bwd, symbolic_zeros = True )
            outputs = my_mlir_op( tuple( fai.differentiable_ffi_inputs ) )

        elif mlir:
            outputs = self._call_via_primitive( fai, module_name )

        elif not grad:
            outputs = _call_ffi( tuple( fai.differentiable_ffi_inputs ) )
        else:
            @jax.custom_vjp
            def my_ffi_op( differentiable_inputs ):
                return _call_ffi( differentiable_inputs )

            def my_ffi_op_fwd( _differentiable_inputs ):
                # With symbolic_zeros = True, JAX wraps each input in CustomVJPPrimal( value, perturbed )
                perturbed_flags = tuple( v.perturbed if isinstance( v, CustomVJPPrimal ) else True for v in _differentiable_inputs )
                differentiable_inputs = tuple( v.value if isinstance( v, CustomVJPPrimal ) else v for v in _differentiable_inputs )
                outputs = _call_ffi( differentiable_inputs )

                return outputs, ( differentiable_inputs, outputs, perturbed_flags )

            def my_ffi_op_bwd( residuals, grads_of_the_outputs ):
                differentiable_inputs, outputs, perturbed_flags = residuals
                if not isinstance( grads_of_the_outputs, ( tuple, list ) ):
                    grads_of_the_outputs = ( grads_of_the_outputs, )

                bfai = fai.backward_version( self, outputs, grads_of_the_outputs, "GradParameters", differentiable_inputs, perturbed_flags )
                func = jax.ffi.ffi_call( module_name + "_backward", bfai.ffi_outputs )
                ret = func( *bfai.ffi_inputs, **bfai.ffi_attributes )
                if isinstance( ret, jax.Array ):
                    ret = ( ret, )

                return ( tuple( ret[ ct.num_in_outputs ] for ct in bfai.tensor_outputs ), )

            my_ffi_op.defvjp( my_ffi_op_fwd, my_ffi_op_bwd, symbolic_zeros = True )

            # --- appel ---
            outputs = my_ffi_op( tuple( fai.differentiable_ffi_inputs ) )

        # ret assembly
        fai.update_objects( outputs )
        res = fai.assemble_returns()

        # item or list
        if len( res ) == 0:
            return None
        if len( res ) == 1:
            return res[ 0 ]
        return res

    def ffi_tensor_input_bind_code( self, ndim, dtype: Dtype ) -> str:
        return f"Arg<xla::ffi::Buffer<{ dtype.jax_ffi_tensor_type() }>>()"

    def ffi_tensor_input_arg_code( self, ndim, dtype: Dtype ) -> str:
        return f"xla::ffi::Buffer<{ dtype.jax_ffi_tensor_type() }>"

    def ffi_tensor_output_bind_code( self, ndim, dtype: Dtype ) -> str:
        return f"Ret<xla::ffi::Buffer<{ dtype.jax_ffi_tensor_type() }>>()"

    def ffi_tensor_output_arg_code( self, ndim, dtype: Dtype ) -> str:
        return f"xla::ffi::ResultBuffer<{ dtype.jax_ffi_tensor_type() }>"

    def ffi_tensor_output_spec( self, shape, dtype: Dtype ):
        return jax.ShapeDtypeStruct( shape, dtype.driver_version )

    def ffi_parameter_bind_code( self, dtype, name: str ) -> str:
        return f"Attr<{ self.normalized_type_for( dtype ) }>( \"{ name }\" )"

    def is_zero_tensor( self, value ):
        return isinstance( value, ad_util.SymbolicZero )

    def _call_via_primitive( self, fai: CallArgsAnalysis, module_name: str ):
        from .JaxMlirPrimitive import get_or_create

        fai.update_differentiable_input_values_with(
            tuple( fai.differentiable_ffi_inputs )
        )

        # Detect if we're inside an outer jax.jit trace.
        # Inside a trace, any JAX op produces a Tracer instead of a concrete array.
        inside_jit = isinstance( jax.numpy.zeros( () ), jax_core.Tracer )

        if inside_jit and fai.dynamix_axes:
            # Try to run the retry loop concretely at trace time using
            # ensure_compile_time_eval. This works when all FFI inputs are concrete
            # (e.g. captured constants). If an input is abstract (JAX traced arg),
            # we fall back to a runtime callback that raises CapacityOverflow.
            # Note: JaxRuntimeError wraps the callback exception — use
            # sdot.is_capacity_overflow(e) or catch jax.errors.JaxRuntimeError.
            concrete_retry_done = False
            with jax.ensure_compile_time_eval():
                try:
                    while True:
                        func = jax.ffi.ffi_call( module_name, fai.ffi_outputs )
                        eager_ret = func( *fai.ffi_inputs, **fai.ffi_attributes )
                        if isinstance( eager_ret, jax.Array ):
                            eager_ret = ( eager_ret, )
                        else:
                            eager_ret = tuple( eager_ret )
                        u64 = eager_ret[ fai.index_u64_output ]
                        if int( u64[ fai.index_dynamic_size_exception ] ) == 0:
                            break
                        da = fai.dynamix_axes[ int( u64[ fai.index_dynamic_size_exception ] ) - 1 ]
                        need = int( u64[ fai.index_dynamic_size_exception + 1 ] )
                        name = "max_of_" + da.name_in_parent
                        for t in fai.tensor_outputs:
                            assert t.parent is not None
                            if t.parent() == da.parent() and t.ctor_kwargs is not None and name in t.ctor_kwargs:
                                t.ctor_kwargs[ name ] = max( need, 2 * int( t.ctor_kwargs[ name ] ) )
                    concrete_retry_done = True
                except Exception:
                    pass  # abstract inputs — will use runtime callback below

            prim = get_or_create( module_name, fai.ffi_outputs, fai.ffi_attributes )
            ret = jax.jit( lambda *args: prim.bind( *args ) )( *fai.ffi_inputs )
            if isinstance( ret, jax.Array ):
                ret = ( ret, )
            else:
                ret = tuple( ret )

            if not concrete_retry_done:
                def _raise_if_overflow( u64 ):
                    if u64[ fai.index_dynamic_size_exception ] != 0:
                        idx = int( u64[ fai.index_dynamic_size_exception ] ) - 1
                        da = fai.dynamix_axes[ idx ]
                        raise JaxDriver.CapacityOverflow(
                            f"Overflow on dynamic axis '{ da.name_in_parent }' inside jax.jit. "
                            f"Increase max_of_{ da.name_in_parent } or call outside jax.jit. "
                            f"Note: JAX wraps this in JaxRuntimeError — use sdot.is_capacity_overflow(e)."
                        )
                jax.debug.callback( _raise_if_overflow, ret[ fai.index_u64_output ], ordered = True )

        elif inside_jit:
            # No dynamic axes: plain JIT call.
            prim = get_or_create( module_name, fai.ffi_outputs, fai.ffi_attributes )
            ret  = jax.jit( lambda *args: prim.bind( *args ) )( *fai.ffi_inputs )
            if isinstance( ret, jax.Array ):
                ret = ( ret, )
            else:
                ret = tuple( ret )

        else:
            # Eager mode: retry loop until capacity is sufficient.
            while True:
                prim = get_or_create( module_name, fai.ffi_outputs, fai.ffi_attributes )
                ret  = jax.jit( lambda *args: prim.bind( *args ) )( *fai.ffi_inputs )

                if isinstance( ret, jax.Array ):
                    ret = ( ret, )
                else:
                    ret = tuple( ret )

                u64_output = ret[ fai.index_u64_output ]
                if u64_output[ fai.index_dynamic_size_exception ] == 0:
                    break

                da = fai.dynamix_axes[ int( u64_output[ fai.index_dynamic_size_exception + 0 ] ) - 1 ]
                needed_size = int( u64_output[ fai.index_dynamic_size_exception + 1 ] )
                faulty_axis_name = "max_of_" + da.name_in_parent

                made_a_change = False
                for tensor_output in fai.tensor_outputs:
                    assert tensor_output.parent is not None
                    if tensor_output.parent() != da.parent():
                        continue
                    if tensor_output.ctor_kwargs is not None and faulty_axis_name in tensor_output.ctor_kwargs:
                        old_value = int( tensor_output.ctor_kwargs[ faulty_axis_name ] )
                        new_value = max( needed_size, 2 * old_value )
                        tensor_output.ctor_kwargs[ faulty_axis_name ] = new_value
                        made_a_change = True

                assert made_a_change

        return ret

    def _module_name_for( self, code: str, grad_code: str, includes: list[ str ], main_list: CallArgsAnalysis ):
        # get signature — include device type to avoid CPU/GPU cache collision
        base_signature = [ code, grad_code, main_list.arguments.signature(), self.device.signature ] + includes

        # module name
        from sdot.util.encode_base_62 import encode_base_62
        res = re.sub( r'[^\w]', '_', str.join( "_", base_signature ) )
        while "__" in res:
            res = res.replace( "__", "_" )
        if len( res ) > 50:
            res = res[ : 50 - 11 ] + encode_base_62( res[ 50 - 11: ] )

        return res

    _registered_ffi_targets = set()

    def _register_ffi_target( self, module_name: str, code: str, grad_code: str, includes: list[ str ], args: CallArgsAnalysis ):
        # already registered ?
        if module_name in JaxDriver._registered_ffi_targets:
            return
        JaxDriver._registered_ffi_targets.add( module_name )

        # else, try to load it from the disk
        from ..compilation.force_build import force_build
        if not force_build():
            try:
                self._try_to_import_and_register_ffi_target( module_name, code, bool( grad_code ) )
                return
            except ( ImportError, SystemError ):
                pass

        # else, make the dylib
        includes_set = set( includes )
        self._make_dylib( code, grad_code, includes_set, args, module_name )

        # and try again to import it
        self._try_to_import_and_register_ffi_target( module_name, code, bool( grad_code ) )


    def _try_to_import_and_register_ffi_target( self, module_name: str, func_name: str, make_backward_binding: bool ):
        platform = "gpu" if self.device == "gpu" else "cpu"
        module = importlib.import_module( "sdot.generated_files." + module_name )
        jax.ffi.register_ffi_target( module_name, getattr( module, module_name )(), platform = platform )
        if make_backward_binding:
            jax.ffi.register_ffi_target( module_name + "_backward", getattr( module, module_name + "_backward" )(), platform = platform )


    def _make_dylib( self, code: str, grad_code: str, includes: set, fai: CallArgsAnalysis, module_name: str ):
        # generate structs
        already_visited = set()
        fai.arguments.generate_structures( already_visited )

        # include list
        includes.add( "sdot/jax_ffi_wrappers.h" )
        includes.add( "nanobind/nanobind.h" )

        # declaration
        lines = []
        lines.append( "" )
        lines.append( "namespace nb = nanobind;" )
        lines.append( "using namespace sdot;" )
        lines.append( "" )
        lines.append( f"using Arch = { self.device.cpp_type };" )
        lines.append( "" )

        # --- forward handler ---
        self._handler_source( includes, lines, code, fai, module_name, "Parameters" )
        lines.append( "" )

        # --- backward handler ---
        if grad_code:
            gfai = fai.backward_version( self, [], [], "GradParameters" )
            self._handler_source( includes, lines, grad_code, gfai, module_name, "GradParameters", "_backward" )
            lines.append( "" )

        lines.append( "" )
        lines.append( "template<typename T>" )
        lines.append( "nb::capsule EncapsulateFfiCall( T *fn ) {" )
        lines.append( "  static_assert( std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>, \"Encapsulated function must be and XLA FFI handler\");" )
        lines.append( "  return nb::capsule( reinterpret_cast<void *>( fn ), \"xla._CUSTOM_CALL_TARGET\" );" )
        lines.append( "}" )
        lines.append( "" )
        lines.append( f"NB_MODULE( { module_name }, m ) {{" )
        lines.append( f"  m.def( \"{ module_name }\", []() {{ return EncapsulateFfiCall( binding_{ module_name } ); }} );" )
        if grad_code:
            lines.append( f"  m.def( \"{ module_name }_backward\", []() {{ return EncapsulateFfiCall( binding_{ module_name }_backward ); }} );" )
        lines.append( "}" )

        include_lines = [ f"#include <{ include }>" for include in sorted( includes, key = lambda s: ( -len( s ), s ) ) ]

        #
        from ..compilation.make_dylib_from_source import make_dylib_from_source
        return make_dylib_from_source( str.join( "\n", include_lines + lines ), module_name, [], self.device )

    def _handler_source( self, includes, lines, code: str, fai: CallArgsAnalysis, module_name: str, struct_name: str, suffix = "" ):
        is_gpu = self.device.is_cuda_gpu

        fai.make_parameters_struct( includes, lines, struct_name )
        lines.append( "" )

        # handler signature — append stream for GPU (PlatformStream<T> decodes to T directly)
        arg_decl = fai.arg_decl()
        if is_gpu:
            stream_arg = "cudaStream_t stream"
            arg_decl = f"{ arg_decl }, { stream_arg }" if arg_decl else stream_arg
        lines.append( f"xla::ffi::Error impl_{ module_name }{ suffix }( { arg_decl } ) {{" )

        lines.append( f"    using TI = SI64;" ) # TODO: 32 bit systems

        # arch instance
        if is_gpu:
            lines.append( "    Arch arch( stream );" )
        else:
            lines.append( "    Arch arch;" )

        # u8_...
        if len( fai.u8_input_values ):
            if is_gpu:
                lines.append( f"    std::vector<PI8> _u8_host( { len( fai.u8_input_values ) } );" )
                lines.append( "    cudaMemcpy( _u8_host.data(), u8_input_buffer.typed_data(), _u8_host.size() * sizeof( PI8 ), cudaMemcpyDeviceToHost );" )
                lines.append( "    const PI8 *u8_input = _u8_host.data();" )
            else:
                lines.append( "    const PI8 *u8_input = u8_input_buffer.typed_data();" )

        if fai.u64_output_size:
            lines.append( "    PI64 *u64_output = u64_output_buffer->typed_data();" )
            if is_gpu:
                lines.append( "    cudaMemsetAsync( u64_output, 0, u64_output_buffer->element_count() * sizeof( PI64 ), stream );" )
            else:
                lines.append( "    std::memset( u64_output, 0, u64_output_buffer->element_count() * sizeof( PI64 ) );" )

        # conversions
        fai.tensor_conversions( lines )

        # beg try block
        lines.append( "    try {" )

        # call the function
        lines.append( f"        auto p = { fai.arguments.assembled_code( '        ' ) };" )
        lines.append( "" )
        lines.append( f"        { code }" )

        # end try block
        lines.append( '    } catch ( DynamicSizeException de ) {' )
        if is_gpu:
            # DynamicSizeException from device code is not supported yet; signal via cudaMemcpy
            lines.append( f'        PI64 _de_vals[2] = {{ 1 + de.num_dynamic_axis, de.needed_size }};' )
            lines.append( f'        cudaMemcpyAsync( u64_output + { fai.index_dynamic_size_exception }, _de_vals, 2 * sizeof( PI64 ), cudaMemcpyHostToDevice, stream );' )
        else:
            lines.append( f'        u64_output[ { fai.index_dynamic_size_exception + 0 } ] = 1 + de.num_dynamic_axis;' )
            lines.append( f'        u64_output[ { fai.index_dynamic_size_exception + 1 } ] = de.needed_size;' )
        lines.append( '    }' )

        # end impl
        lines.append( "    return xla::ffi::Error::Success();" )
        lines.append( "}" )

        # XLA_FFI_DEFINE_HANDLER_SYMBOL — append PlatformStream bind for GPU
        bind_chain = [ "xla::ffi::Ffi::Bind()" ] + fai.bind_chain()
        if is_gpu:
            bind_chain.append( "Ctx<xla::ffi::PlatformStream<cudaStream_t>>()" )
        lines.append( f"XLA_FFI_DEFINE_HANDLER_SYMBOL( binding_{ module_name }{ suffix }, impl_{ module_name }{ suffix }, { str.join( '.', bind_chain ) } );" )


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
