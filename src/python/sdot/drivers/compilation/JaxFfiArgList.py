from .collect_attributes import collect_attributes
from .JaxFfiArg import JaxFfiArg
from .CpyArg import CpyArg
from .Return import Return
from .Tensor import Tensor

from jax._src import ad_util
import jax.numpy
import jax.core
import numpy
import jax


class JaxFfiArgList:
    """ lists of arguments
    - flat lists of JaxFfi compatible items (tensors, returned tensors)
        - non_differentiable_jax_ffi_output_args
        - non_differentiable_jax_ffi_input_args
        - differentiable_jax_ffi_output_args
        - differentiable_jax_ffi_input_args
    - list of Python and Cpp side arguments

    cpy_args -> list
    """

    def __init__( self, driver, python_args: dict ):
        # rec find sub args
        self.non_differentiable_jax_ffi_output_args: list[ JaxFfiArg ] = []
        self.non_differentiable_jax_ffi_input_args: list[ JaxFfiArg ] = []
        self.differentiable_jax_ffi_output_args: list[ JaxFfiArg ] = []
        self.differentiable_jax_ffi_input_args: list[ JaxFfiArg ] = []
        self.validity_mask_values = []
        self.cpy_args: list[ CpyArg ] = [] # recursive list
        for python_arg_name, python_arg_value in python_args.items():
            cpy_arg = CpyArg( python_arg_name )
            cpy_arg.value = python_arg_value
            self.cpy_args.append( cpy_arg )
            self.get_jax_ffi_args( driver, python_arg_name, python_arg_value, cpy_arg )

        # make validity_input_arg
        validity_mask = numpy.zeros( [ ( len( self.validity_mask_values ) + 63 ) // 64 ], dtype = numpy.uint64 )
        for n, v in enumerate( self.validity_mask_values ):
           if v:
               validity_mask[ n // 64 ] |= ( 1 << ( n % 64 ) )


        self.non_differentiable_jax_ffi_input_args.append( JaxFfiArg(
            value = validity_mask,
            spec = self._tensor_spec( driver, [ ], jax.numpy.uint64 ),
            name = "validity_mask_buffer",
            valid = None,
            bind = "Arg<xla::ffi::Buffer<xla::ffi::U64>>",
            cpp_type = "xla::ffi::Buffer<xla::ffi::U64>"
        ) )

        # info
        # info( self.non_differentiable_jax_ffi_output_args,
        #     self.non_differentiable_jax_ffi_input_args,
        #     self.differentiable_jax_ffi_output_args,
        #     self.differentiable_jax_ffi_input_args,
        #     self.validity_mask_values,
        #     self.cpy_args )


    @property
    def non_differentiable_jax_ffi_output_specs( self ):
        return [ output.spec for output in self.non_differentiable_jax_ffi_output_args ]

    @property
    def differentiable_jax_ffi_output_specs( self ):
        return [ output.spec for output in self.differentiable_jax_ffi_output_args ]

    @property
    def non_differentiable_jax_ffi_input_values( self ):
        return [ input.value for input in self.non_differentiable_jax_ffi_input_args ]

    @property
    def differentiable_jax_ffi_input_values( self ):
        return [ input.value for input in self.differentiable_jax_ffi_input_args ]

    @property
    def jax_ffi_output_args( self ) -> list[ JaxFfiArg ]:
        return self.differentiable_jax_ffi_output_args + self.non_differentiable_jax_ffi_output_args

    @property
    def jax_ffi_input_args( self ) -> list[ JaxFfiArg ]:
        return self.differentiable_jax_ffi_input_args + self.non_differentiable_jax_ffi_input_args

    def backward_version( self, driver, differentiable_inputs, non_differentiable_inputs, differentiable_outputs, non_differentiable_outputs, grad_outputs ):
        # turn fwd arg into inputs
        nargs = {}
        for cpy_arg in self.cpy_args:
            value = cpy_arg.reassemble( differentiable_inputs, non_differentiable_inputs, differentiable_outputs, non_differentiable_outputs )
            nargs[ cpy_arg.name ] = value

        # grad of the outputs (-> inputs)
        for n in range( len( grad_outputs ) ):
            farg = self.differentiable_jax_ffi_output_args[ n ]
            nargs[ "grad_" + farg.name ] = grad_outputs[ n ]

        # return grad of the inputs (-> inputs)
        for n in range( len( differentiable_inputs ) ):
            farg = self.differentiable_jax_ffi_input_args[ n ]
            nargs[ "grad_" + farg.name ] = Return( Tensor, differentiable_inputs[ n ].shape )

        return JaxFfiArgList( driver, nargs )

    def get_jax_ffi_args( self, driver, name: str, value: any, cpy_arg: CpyArg ):
        """ recursively fill the jax_ffi_... lists """

        # method
        if callable( getattr( value, "get_jax_ffi_args", None ) ):
            return value.get_jax_ffi_args( self, driver, name, cpy_arg )

        # SymbolicZero
        if isinstance( value, ad_util.SymbolicZero ):
            array_value = self._tensor_value( driver, value.shape, value.dtype )
            array_spec = self._tensor_spec( driver, value.shape, value.dtype )
            return self._add_tensor_arg( driver, array_value, array_spec, name, cpy_arg, valid = False )

        # arrays
        if isinstance( value, ( jax.core.Tracer, jax.Array, numpy.ndarray, jax.ShapeDtypeStruct ) ):
            return self._add_tensor_arg( driver, value, jax.ShapeDtypeStruct( value.shape, value.dtype ), name, cpy_arg )

        # std objects
        if isinstance( value, float ):
            array = driver.tn( value, ndim = 0, dtype = driver.dtype )
            return self._add_tensor_arg( driver, array, jax.ShapeDtypeStruct( [], driver.dtype ), name, cpy_arg )

        if isinstance( value, int ):
            array = driver.tn( value, ndim = 0, dtype = driver.itype )
            return self._add_tensor_arg( driver, array, jax.ShapeDtypeStruct( [], driver.itype ), name, cpy_arg )

        if isinstance( value, ( list, tuple ) ):
            cpy_arg._python_ctor = type( value )
            cpy_arg.signature_type = "L"
            cpy_arg.code = "std::tie"
            for num, item in enumerate( value ):
                self.get_jax_ffi_args( driver, f"{ name }_{ num }", item, cpy_arg.arg( str( num ) ) )
            return

        if value is None:
            cpy_arg.signature_type = "N"
            cpy_arg.code = "{}"
            return

        # else, get attributes
        cpy_arg.code = JaxFfiArgList.cpp_class_name( driver, value )
        cpy_arg.signature_type = cpy_arg.code
        cpy_arg._python_ctor = type( value )
        for attr, _ in collect_attributes( value ):
            self.get_jax_ffi_args( driver, f"{ name }_{ attr }", getattr( value, attr ), cpy_arg.arg( attr ) )


    def _tensor_spec( self, driver, shape, dtype ):
        return jax.ShapeDtypeStruct( shape, dtype or driver.dtype )

    def _tensor_value( self, driver, shape, dtype ):
        return driver.empty( shape, dtype = dtype )

    @staticmethod
    def cpp_class_name( driver, value ):
        if callable( getattr( value, "cpp_class_name", None ) ):
            return value.cpp_class_name( driver )
        return value.__class__.__name__


    def _differentiable_dtype( self, dtype ):
        if dtype is None or dtype is float:
            return True
        if dtype is int:
            return False
        return not jax.numpy.issubdtype( dtype, jax.numpy.integer )


    def _jax_dtype( self, driver, dtype ):
        if dtype is None or dtype is float:
            return self._jax_dtype( driver, driver.dtype )
        if dtype is int:
            return self._jax_dtype( driver, driver.itype )
        """ C++ jax name for dtype """

        if dtype == jax.numpy.float32:
            return "xla::ffi::F32"
        if dtype == jax.numpy.float64:
            return "xla::ffi::F64"
        if dtype == jax.numpy.int32:
            return "xla::ffi::S32"
        if dtype == jax.numpy.int64:
            return "xla::ffi::S64"

        raise NotImplementedError( f"for dtype { dtype }" )


    def _add_tensor_arg( self, driver, value: any, spec: any, name: str, cpy_arg: CpyArg, valid: bool = True ):
        # differentiable
        differentiable = self._differentiable_dtype( value.dtype )

        # signature_type
        cpy_arg.signature_type = f"T{ len( value.shape ) }{ driver.normalized_type_for( value.dtype ) }"
        jax_dtype = self._jax_dtype( driver, value.dtype )

        # validity
        validity = len( self.validity_mask_values )
        self.validity_mask_values.append( valid )

        # cpy_arg.value
        tv_arg = name
        if cpy_arg.for_return == 1: # mutable => start out with a copy of inp
           tv_arg = f"{ name }_out, { name }"

        cpy_arg.code = f"tensor_view( CtInt<{ len( value.shape ) }>(), { tv_arg }, validity_mask[ { validity // 64 } ] & { 1 << ( validity % 64 ) } )"

        # input
        if cpy_arg.for_return <= 1:
            self._add_jax_ffi_arg_input( cpy_arg, valid, differentiable = differentiable, jax_ffi_arg = JaxFfiArg(
                cpp_type = f"xla::ffi::Buffer<{ jax_dtype }>",
                bind = f"Arg<xla::ffi::Buffer<{ jax_dtype }>>",
                value = value,
                valid = valid,
                spec = spec,
                name = name,
            ) )

        # output
        if cpy_arg.for_return >= 1:
            self._add_jax_ffi_arg_output( cpy_arg, valid, differentiable = differentiable, jax_ffi_arg = JaxFfiArg(
                cpp_type = f"xla::ffi::ResultBuffer<{ jax_dtype }>",
                bind = f"Ret<xla::ffi::Buffer<{ jax_dtype }>>",
                value = value,
                valid = valid,
                spec = spec,
                name = name + "_out" if cpy_arg.for_return == 1 else name,
            ) )


    def _add_jax_ffi_arg_output( self, cpy_arg, valid, differentiable, jax_ffi_arg ):
        if differentiable:
            cpy_arg._python_ctor = ( 1, len( self.differentiable_jax_ffi_output_args ), valid )
            self.differentiable_jax_ffi_output_args.append( jax_ffi_arg )
        else:
            cpy_arg._python_ctor = ( 0, len( self.non_differentiable_jax_ffi_output_args ), valid )
            self.non_differentiable_jax_ffi_output_args.append( jax_ffi_arg )

    def _add_jax_ffi_arg_input( self, cpy_arg, valid, differentiable, jax_ffi_arg ):
        if differentiable:
            cpy_arg._python_ctor = ( 1, len( self.differentiable_jax_ffi_input_args ), valid )
            self.differentiable_jax_ffi_input_args.append( jax_ffi_arg )
        else:
            cpy_arg._python_ctor = ( 0, len( self.differentiable_jax_ffi_input_args ), valid )
            self.non_differentiable_jax_ffi_input_args.append( jax_ffi_arg )

