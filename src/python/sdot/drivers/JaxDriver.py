from .compilation.collect_attributes import collect_attributes
from .compilation.cpp_class_name import cpp_class_name
from .compilation.Mutable import Mutable
from .compilation.Return import Return
from .compilation.Tensor import Tensor
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

    def array( self, data ):
        return jnp.asarray( data, dtype = self.dtype, device = self.device )

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
        if dtype is None:
            return self.normalized_type_for( self.dtype )

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

    def call( self, func_name: str, includes: str | list[ str ], **args ):
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

        # Séparation niveau JAX / niveau Python :
        #   ffi_op travaille uniquement sur des flat tuples d'arrays JAX
        #   L'assemblage des objets Python se fait en dehors

        # flat args for fwd
        validity_bits = []
        flat_inputs = []
        flat_specs = []
        for arg_name, arg_data, has_input, has_output in JaxDriver._with_hio( args ):
            if has_input:
                for obj, _, valid, _, _ in self.as_jax_ffi_compatible_args( arg_data, f"inp_{ arg_name }" ):
                    validity_bits.append( valid )
                    flat_inputs.append( obj )
            if has_output:
                for spec, _, valid, _, _ in self.as_jax_ffi_compatible_rets( arg_data, f"out_{ arg_name }" ):
                    validity_bits.append( valid )
                    flat_specs.append( spec )

        # add validity_tensor in flat_inputs
        validity_tensor = numpy.array( [ ( len( validity_bits ) + 63 ) // 64 ] )
        for n, v in enumerate( validity_bits ):
            if v:
                validity_tensor[ n // 64 ] |= ( 1 << ( n % 64 ) )
        flat_inputs.append( self.t1( validity_tensor, dtype = numpy.uint64 ) )

        # --- niveau JAX : ffi_op prend UN tuple de flat arrays ---
        # (custom_vjp exige que la fonction et son bwd travaillent sur des pytrees cohérents)
        @jax.custom_vjp
        def ffi_op( flat_inputs ):
            func = jax.ffi.ffi_call( module_name, flat_specs )
            ret = func( *flat_inputs )
            if isinstance( ret, ( tuple, list ) ):
                return tuple( ret )
            return ( ret, )

        def ffi_op_fwd( flat_inputs ):
            flat_outputs = ffi_op( flat_inputs )
            return flat_outputs, ( flat_inputs, flat_outputs )

        def ffi_op_bwd( residuals, grad_outs ):
            from jax._src.ad_util import Zero

            flat_inputs, flat_outputs = residuals
            if not isinstance( grad_outs, tuple ):
                grad_outs = ( grad_outs, )

            # l'idée de base : on prend tous les outputs

            # backward output specs => shape/dtype for each differentiable flat input
            bwd_output_specs = []
            for a in flat_inputs:
                if not self.is_int_dtype( a.dtype ):
                    bwd_output_specs.append( jax.ShapeDtypeStruct( a.shape, a.dtype ) )

            # matérialiser les Zero en vrais arrays avant de les passer au C++
            grad_outs = tuple(
                jnp.zeros_like( flat_outputs[ i ] ) if isinstance( g, Zero ) else g
                for i, g in enumerate( grad_outs )
            )

            bwd_func = jax.ffi.ffi_call( module_name + "_backward", bwd_output_specs )
            raw_grad_ins = bwd_func( *( flat_outputs + flat_inputs + grad_outs ), mask=mask )
            if isinstance( raw_grad_ins, ( tuple, list ) ):
                raw_grad_ins = tuple( raw_grad_ins )
            else:
                raw_grad_ins = ( raw_grad_ins, )

            grad_iter = iter( raw_grad_ins )
            grad_flat_ins = tuple(
                None if self.is_int_dtype( a.dtype ) else next( grad_iter )
                for a in flat_inputs
            )
            return ( grad_flat_ins, )  # un gradient par argument de ffi_op (le tuple unique)

        ffi_op.defvjp( ffi_op_fwd, ffi_op_bwd )

        # --- appel ---
        flat_outs = ffi_op( flat_inputs )

        # --- réassemblage Python ---
        results = []
        out_iter = iter( flat_outs )
        for arg_name, arg_data, has_input, has_output in JaxDriver._with_hio( args ):
            if has_output:
                if has_input:
                    self.python_update_from_jax_ffi_compatible_args( arg_data, out_iter )
                else:
                    results.append( self.python_assembly_from_jax_ffi_compatible_args( arg_data, out_iter ) )

        if len( results ) == 0:
            return None
        if len( results ) == 1:
            return results[ 0 ]
        return results


    @staticmethod
    def _with_hio( args: dict ) -> list[ tuple[ str, any, bool, bool ] ]:
        """ list of arguments with as tuples with ( arg, is_input, is_output ) """
        res = []
        for arg_name, arg_data in args.items():
            if isinstance( arg_data, Mutable ):
                res.append( ( arg_name, arg_data, True, True ) )
            elif isinstance( arg_data, Return ):
                res.append( ( arg_name, arg_data, False, True ) )
            else:
                res.append( ( arg_name, arg_data, True, False ) )
        return res


    def _cpp_ffi_type_name( self, dtype ):
        """ C++ jax name for dtype """

        if dtype is None or dtype is float:
            return self._cpp_ffi_type_name( self.dtype )

        if dtype is int:
            return self._cpp_ffi_type_name( self.itype )

        if dtype == jnp.float32:
            return "xla::ffi::F32"
        if dtype == jnp.float64:
            return "xla::ffi::F64"
        if dtype == jnp.int32:
            return "xla::ffi::S32"
        if dtype == jnp.int64:
            return "xla::ffi::S64"

        raise NotImplementedError( f"for dtype { dtype }" )

    def cpp_class_name( self, obj ):
        """ TODO: clarify the fact that for jax, cpp_class_name is used only to get the name of the function """
        if isinstance( obj, ( jax_core.Tracer, jax.Array, numpy.ndarray, numpy.ndarray ) ):
            return f"T{ obj.ndim }{ self.normalized_type_for( obj.dtype ) }>>"

        return None

    def as_jax_ffi_compatible_args( self, obj, name: str ) -> list[ tuple[ any, str, bool, str, str ] ]:
        """Decompose obj into a flat list of jax-compatible Python objects that represent it on the binding boundary.

        Return a list of tuples with
        - the compatible python value (most probably a tensor)
        - a boolean to say if it's valid
        - the binding name (like Attr<xla::ffi::Buffer<FP32>>)
        - the argument name (like xla::ffi::Buffer<FP32>)
        """

        # method
        if callable( getattr( obj, "as_jax_ffi_compatible_args", None ) ):
            return obj.as_jax_ffi_compatible_args( self, str )

        # jax arrays
        if isinstance( obj, ( jax_core.Tracer, jax.Array, numpy.ndarray ) ):
            dtype = self._cpp_ffi_type_name( obj.dtype )
            return [ ( obj, name, True, f"Arg<xla::ffi::Buffer<{ dtype }>>", f"xla::ffi::Buffer<{ dtype }>" ) ]

        # std objects
        if isinstance( obj, float ):
            return self.as_jax_ffi_compatible_args( self.tn( obj, ndim = 0, dtype = self.dtype ), name )

        if isinstance( obj, int ):
            return self.as_jax_ffi_compatible_args( self.tn( obj, ndim = 0, dtype = self.itype ), name )

        if isinstance( obj, ( list, tuple ) ):
            res = []
            for num, value in enumerate( obj ):
                res += self.as_jax_ffi_compatible_args( value, f"{ name }_{ num }" )
            return res

        if obj is None:
            return []

        # else, get attributes
        out = []
        for attr, _ in collect_attributes( obj ):
            out += self.as_jax_ffi_compatible_args( getattr( obj, attr ), f"{ name }_{ attr }" )
        return out

    def _jax_shape_out( self, shape, dtype ):
        return jax.ShapeDtypeStruct( shape, dtype or self.dtype )

    def as_jax_ffi_compatible_rets( self, obj, name: str ) -> list[ tuple[ any, str, bool, str, str ] ]:
        """Decompose obj into a flat list of jax-compatible return objects that represent it on the binding boundary.

        return a list of tuples with
        - the compatibie argument value
        , a boolean to say if it's valid
        - the binding name (like Ret<xla::ffi::Buffer<FP32>>)
        - the argument name (like xla::ffi::ResultBuffer<FP32>)
        """

        # method
        if callable( getattr( obj, "as_jax_ffi_compatible_rets", None ) ):
            return obj.as_jax_ffi_compatible_rets( self, name )

        # arrays
        if isinstance( obj, ( jax_core.Tracer, jax.Array, numpy.ndarray ) ):
            dtype = self._cpp_ffi_type_name( obj.dtype )
            return [ ( obj, name, True, f"Ret<xla::ffi::Buffer<{ dtype }>>", f"xla::ffi::ResultBuffer<{ dtype }>" ) ]

        # std objects
        if isinstance( obj, float ):
            dtype = self._cpp_ffi_type_name( self.dtype )
            return [ ( jax.ShapeDtypeStruct( [], self.dtype ), name, True, f"Ret<xla::ffi::Buffer<{ dtype }>>", f"xla::ffi::ResultBuffer<{ dtype }>" ) ]

        if isinstance( obj, int ):
            itype = self._cpp_ffi_type_name( self.itype )
            return [ ( jax.ShapeDtypeStruct( [], self.itype ), name, True, f"Ret<xla::ffi::Buffer<{ itype }>>", f"xla::ffi::ResultBuffer<{ itype }>" ) ]

        if isinstance( obj, ( list, tuple ) ):
            res = []
            for num, value in enumerate( obj ):
                res += self.as_jax_ffi_compatible_rets( value, f"{ name }_{ num }" )
            return res

        # else, get attributes
        out = []
        for attr, _ in collect_attributes( obj ):
            out += self.as_jax_ffi_compatible_rets( getattr( obj, attr ), f"{ name }_{ attr }" )
        return out


    def python_update_from_jax_ffi_compatible_args( self, obj, flat_arg_iterator ):
        """ use flatten jax compatible args to update python object obj """

        # method
        if callable( getattr( obj, "python_update_from_jax_ffi_compatible_args", None ) ):
            return obj.python_update_from_jax_ffi_compatible_args( self, flat_arg_iterator )

        # non ref objects
        if isinstance( obj, ( jax_core.Tracer, jax.Array, numpy.ndarray, int, float ) ):
            raise RuntimeError( "Due to Python+Jax `sdot.driver.call( ..., sdot.Mutable( x ) )` works only with objects that store references." )

        if isinstance( obj, ( tuple ) ):
            raise RuntimeError( "Due to Python+Jax `sdot.driver.call( ..., sdot.Mutable( x ) )` does not work with tuple because it does not support item assignment." )

        # list
        if isinstance( obj, list ):
            for num in range( len( obj ) ):
                obj[ num ] = self.python_assembly_from_jax_ffi_compatible_args( obj[ num ], flat_arg_iterator )
            return

        # else, use attributes
        for name, _ in collect_attributes( obj ):
            setattr( obj, name, self.python_assembly_from_jax_ffi_compatible_args( getattr( obj, name ), flat_arg_iterator ) )


    def python_assembly_from_jax_ffi_compatible_args( self, obj, flat_arg_iterator ):
        """ construct a python object defined obj (can be an instance or a Return(...)) using jax compatible args in flat_arg_iterator """

        # method
        if callable( getattr( obj, "python_assembly_from_jax_ffi_compatible_args", None ) ):
            return obj.python_assembly_from_jax_ffi_compatible_args( self, flat_arg_iterator )

        if isinstance( obj, ( jax_core.Tracer, jax.Array, numpy.ndarray ) ):
            return next( flat_arg_iterator )

        if isinstance( obj, ( float, int ) ):
            return next( flat_arg_iterator ).item()

        if isinstance( obj, list ):
            res = []
            for item in obj:
                res.append( self.python_assembly_from_jax_ffi_compatible_args( item, flat_arg_iterator ) )
            return res

        # else, use attributes
        args = {}
        for attr, _ in collect_attributes( obj ):
            args[ attr ] = self.python_assembly_from_jax_ffi_compatible_args( getattr( obj, attr ), flat_arg_iterator )
        return type( obj )( **args )


    def cpp_assembly_from_jax_ffi_compatible_args( self, obj, name, pos_in_validity_bits: list[ int ] ):
        """ string to recompose an obj instance from a set of jax_ffi compatible object """

        # method to_nanobind_compatible_objects
        if callable( getattr( obj, "cpp_assembly_from_jax_ffi_compatible_args", None ) ):
            return obj.cpp_assembly_from_jax_ffi_compatible_args( self, name, pos_in_validity_bits )

        # std objects
        if isinstance( obj, ( float, int ) ):
            p = pos_in_validity_bits[ 0 ]
            pos_in_validity_bits[ 0 ] += 1
            return f"tensor_view( CtInt<0>(), { name }, validity_mask[ { p // 64 } ] & { 1 << ( p % 64 ) } )()"

        if isinstance( obj, ( list, tuple ) ):
            args = []
            for num, val in enumerate( obj ):
                args.append( self.cpp_assembly_from_jax_ffi_compatible_args( val, f"{ name }_{ num }", pos_in_validity_bits ) )
            return f"std::tie( { str.join( ", ", args ) } )"

        if isinstance( obj, ( jax_core.Tracer, jax.Array, numpy.ndarray ) ):
            p = pos_in_validity_bits[ 0 ]
            pos_in_validity_bits[ 0 ] += 1
            return f"tensor_view( CtInt<{ obj.ndim }>(), { name }, validity_mask[ { p // 64 } ] & { 1 << ( p % 64 ) } )"

        # else, get attributes
        args = []
        for attr, _ in collect_attributes( obj ):
            args.append( f"{ name }_{ attr }" )
        return f"{ cpp_class_name( obj ) }{{ { str.join( ", ", args ) } }}"

    def _module_name_for( self, func_name: str, includes: list[ str ], args: dict ):
        # get signature
        from .compilation.cpp_class_name import cpp_class_name
        signature_items = [ func_name ]
        for arg_name, arg_data in args.items():
            signature_items.append( arg_name + "_" + cpp_class_name( arg_data ) )
        signature_items += includes

        # module name
        from .compilation.encode_base_62 import encode_base_62
        res = re.sub( r'[^\w]', '_', str.join( "__", signature_items ) )
        if len( res ) > 40:
            res = res[ : 40 - 11 ] + encode_base_62( res[ 40 - 11: ] )

        return res

    _registered_ffi_targets = set()

    def _register_ffi_target( self, module_name: str, func_name: str, includes: list[ str ], args: dict, make_backward_binding = True ):
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


    def _make_dylib( self, func_name: str, includes: list[ str ], args: dict, module_name: str, make_backward_binding: bool ):
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
        lines += self._handler_source( func_name, args )
        lines.append( "" )

        # --- backward handler ---
        if make_backward_binding:
            lines += self._handler_source( func_name + "_backward", self._make_backwarg_args_instance( args ) )
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

    def _make_backwarg_args_instance( self, args: dict ) -> dict:
        """ make a (fake) list of args that can be used to compile the backward version """

        # transformation of the forward args (everything becomes an input)
        res = {}
        for arg_name, arg_data in args.items():
            if isinstance( arg_data, Return ):
                res[ arg_name ] = arg_data.fake_instance( self )
            elif isinstance( arg_data, Mutable ):
                res[ arg_name ] = arg_data.value
            else:
                res[ arg_name ] = arg_data

        # gradients
        # for arg in args:
        #     grad_inp = None
        #     grad_out = None
        #     if isinstance( arg, Return ):
        #         grad_out = arg.fake_instance( self )
        #     elif isinstance( arg, Mutable ):
        #         grad_out = arg.value
        #         grad_inp = arg.value
        #     else:
        #         grad_inp = arg

        #     if grad_out is not None:
        #         for obj, _, _, _, _ in self.as_jax_ffi_compatible_args( grad_out, "" ):
        #             if isinstance( obj, ( jax_core.Tracer, jax.Array, numpy.ndarray ) ):
        #                 if self.is_int_dtype( obj.dtype ):
        #                     continue
        #                 res.append( obj )
        #             else:
        #                 raise NotImplementedError

        #     if grad_inp is not None:
        #         for obj, _, _, _, _ in self.as_jax_ffi_compatible_args( arg, "" ):
        #             if isinstance( obj, ( jax_core.Tracer, jax.Array, numpy.ndarray ) ):
        #                 if self.is_int_dtype( obj.dtype ):
        #                     continue
        #                 res.append( Return( Tensor, obj.shape, obj.dtype ) )
        #             else:
        #                 raise NotImplementedError

        return res

    def _handler_source( self, func_name: str, args: dict ) -> list[ str ]:
        #
        lines = []

        # start impl: declaration. Split between inputs then outputs
        output_arg_decls = []
        input_arg_decls = []
        for arg_name, arg_data, has_input, has_output in JaxDriver._with_hio( args ):
            if has_input:
                for _, arg_name, _, _, cpp_type in self.as_jax_ffi_compatible_args( arg_data, "inp_" + arg_name ):
                    input_arg_decls.append( f"{ cpp_type } { arg_name }" )
            if has_output:
                for _, arg_name, _, _, cpp_type in self.as_jax_ffi_compatible_rets( arg_data, "out_" + arg_name ):
                    output_arg_decls.append( f"{ cpp_type } { arg_name }" )

        input_arg_decls.append( "xla::ffi::Buffer<xla::ffi::DataType::U64> validity_mask_buffer" )
        lines.append( f"xla::ffi::Error impl_{ func_name }( { str.join( ", ", input_arg_decls + output_arg_decls ) } ) {{" )

        # read the validity_mask
        lines.append( "    const PI64 *validity_mask = validity_mask_buffer.typed_data();" )

        # assembly of the C++ objects (splitted between inputs outputs)
        pos_in_validity_bits = [ 0 ]
        for arg_name, arg_data in args.items():
            if isinstance( arg_data, Mutable ):
                # copy input data to output value
                lines.append( f"    auto input_cpp_assembly_{ arg_name } = { self.cpp_assembly_from_jax_ffi_compatible_args( arg_data, f"inp_{ arg_name }", pos_in_validity_bits ) };" )
                lines.append( f"    auto cpp_assembly_{ arg_name } = { self.cpp_assembly_from_jax_ffi_compatible_args( arg_data, f"out_{ arg_name }", pos_in_validity_bits ) };" )
                lines.append( f"    cpp_assembly_{ arg_name } = input_cpp_assembly_{ arg_name };" )
            elif isinstance( arg_data, Return ):
                lines.append( f"    auto cpp_assembly_{ arg_name } = { self.cpp_assembly_from_jax_ffi_compatible_args( arg_data, f"out_{ arg_name }", pos_in_validity_bits ) };" )
            else:
                lines.append( f"    auto cpp_assembly_{ arg_name } = { self.cpp_assembly_from_jax_ffi_compatible_args( arg_data, f"inp_{ arg_name }", pos_in_validity_bits ) };" )

        # call the function
        lines.append( f"    { func_name }( { str.join( ", ", [ f"cpp_assembly_{ arg_name }" for arg_name in args.keys() ] ) } );" )

        # end impl
        lines.append( "    return xla::ffi::Error::Success();" )
        lines.append( "}" )

        # XLA_FFI_DEFINE_HANDLER_SYMBOL
        output_bind_chain = []
        input_bind_chain = []
        for arg_name, arg_data, has_input, has_output in JaxDriver._with_hio( args ):
            if has_output:
                for _, _, _, bind_type, _ in self.as_jax_ffi_compatible_rets( arg_data, arg_name ):
                    output_bind_chain.append( f"{ bind_type }()" )
            if has_input:
                for _, _, _, bind_type, _ in self.as_jax_ffi_compatible_args( arg_data, arg_name ):
                    input_bind_chain.append( f"{ bind_type }()" )
        input_bind_chain.append( "Arg<xla::ffi::Buffer<xla::ffi::DataType::U64>>()" )
        bind_chain = [ "xla::ffi::Ffi::Bind()" ] + input_bind_chain + output_bind_chain
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

