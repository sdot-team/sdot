from .collect_attributes import collect_attributes
from .JaxFfiArg import JaxFfiArg
from .Mutable import Mutable
from .Return import Return
# from .Tensor import Tensor

from jax._src import ad_util
from typing import cast
import jax.numpy
import jax.core
import numpy
import jax

class CpyArg:
    def __init__( self, name: str, value: str = "", sub_list: list | None = None ):
        self.name = name
        self.value = value
        self.sub_list = sub_list
        self.for_return = 0
        self.signature_type = ""
        self._python_ctor: callable | tuple[ int, int ] = () # for output value. ( 1 if differentiable, num in list )

    def arg( self, name: str ):
        sub_cpp_arg = CpyArg( name )

        if self.sub_list is None:
            self.sub_list = []
        self.sub_list.append( sub_cpp_arg )

        return sub_cpp_arg

    def reassemble( self, differentiable_output_values, non_differentiable_output_values ):
        # tensor ?
        if isinstance( self._python_ctor, tuple ):
            assert len( self._python_ctor ) == 2
            info( self._python_ctor, differentiable_output_values, non_differentiable_output_values )
            if self._python_ctor[ 0 ]:
                return differentiable_output_values[ self._python_ctor[ 1 ] ]
            return non_differentiable_output_values[ self._python_ctor[ 1 ] ]

        # ctor
        if self.sub_list is None:
            return self._python_ctor()
        return self._python_ctor( *[ item.reassemble( differentiable_output_values, non_differentiable_output_values ) for item in self.sub_list ] )

    def assemble( self ) -> str:
        res = self.value
        if self.sub_list is not None:
            res += "( " + str.join( ", ", [ a.assemble() for a in self.sub_list ] ) + " )"
        return res


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
            self.cpy_args.append( cpy_arg )
            self.get_jax_ffi_args( driver, python_arg_name, python_arg_value, cpy_arg, 0 )

        # make validity_input_arg
        validity_mask = numpy.zeros( [ ( len( self.validity_mask_values ) + 63 ) // 64 ], dtype = numpy.uint64 )
        for n, v in enumerate( self.validity_mask_values ):
           if v:
               validity_mask[ n // 64 ] |= ( 1 << ( n % 64 ) )
        self.non_differentiable_jax_ffi_input_args.append( JaxFfiArg( validity_mask, "validity_mask_buffer", None,
            "Arg<xla::ffi::Buffer<xla::ffi::U64>>",
            "xla::ffi::Buffer<xla::ffi::U64>"
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
        return [ output.value for output in self.non_differentiable_jax_ffi_output_args ]

    @property
    def differentiable_jax_ffi_output_specs( self ):
        return [ output.value for output in self.differentiable_jax_ffi_output_args ]

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

    def get_jax_ffi_args( self, driver, name: str, value: any, cpy_arg: CpyArg, for_return: int ):
        """ recursively fill the jax_ffi_... lists """
        cpy_arg.for_return = for_return

        # method
        if callable( getattr( value, "get_jax_ffi_args", None ) ):
            return value.get_jax_ffi_args( self, driver, name, cpy_arg, for_return )

        # SymbolicZero
        if isinstance( value, ad_util.SymbolicZero ):
            array = self._tensor_value( driver, value.shape, value.dtype, for_return )
            return self._add_tensor_arg( driver, array, name, cpy_arg, for_return, valid = False )

        # arrays
        if isinstance( value, ( jax.core.Tracer, jax.Array, numpy.ndarray ) ):
            return self._add_tensor_arg( driver, value, name, cpy_arg, for_return )

        # std objects
        if isinstance( value, float ):
            array = driver.tn( value, ndim = 0, dtype = driver.dtype )
            return self._add_tensor_arg( driver, array, name, cpy_arg, for_return )

        if isinstance( value, int ):
            array = driver.tn( value, ndim = 0, dtype = driver.itype )
            return self._add_tensor_arg( driver, array, name, cpy_arg, for_return )

        if isinstance( value, ( list, tuple ) ):
            cpy_arg._python_ctor = type( value )
            cpy_arg.signature_type = "L"
            cpy_arg.value = "std::tie"
            for num, item in enumerate( value ):
                self.get_jax_ffi_args( driver, f"{ name }_{ num }", item, cpy_arg.arg( str( num ) ), for_return )
            return

        if value is None:
            cpy_arg.signature_type = "N"
            cpy_arg.value = "{}"
            return

        # else, get attributes
        cpy_arg.value = self.cpp_class_name( driver, value )
        cpy_arg.signature_type = cpy_arg.value
        cpy_arg._python_ctor = type( value )
        for attr, _ in collect_attributes( value ):
            self.get_jax_ffi_args( driver, f"{ name }_{ attr }", getattr( value, attr ), cpy_arg.arg( attr ), for_return )


    def _tensor_value( self, driver, shape, dtype, for_return ):
        if for_return:
            return jax.ShapeDtypeStruct( shape, dtype or driver.dtype )
        return driver.empty( [ 0 ] * len( cast( tuple, shape ) ), dtype = dtype )


    def cpp_class_name( self, driver, value ):
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


    def _add_tensor_arg( self, driver, value: any, name: str, cpy_arg: CpyArg, for_return: int, valid: bool = True ):
        # _python_ctor
        differentiable = self._differentiable_dtype( value.dtype )
        if for_return:
            if differentiable:
                cpy_arg._python_ctor = ( 1, len( self.differentiable_jax_ffi_output_args ) )
            else:
                cpy_arg._python_ctor = ( 0, len( self.non_differentiable_jax_ffi_output_args ) )

        # signature_type
        cpy_arg.signature_type = f"T{ len( value.shape ) }{ driver.normalized_type_for( value.dtype ) }"

        p = len( self.validity_mask_values )
        self.validity_mask_values.append( valid )
        cpy_arg.value = f"tensor_view( CtInt<{ len( value.shape ) }>(), { name }, validity_mask[ { p // 64 } ] & { 1 << ( p % 64 ) } )"

        #
        jax_dtype = self._jax_dtype( driver, value.dtype )
        if for_return:
            cpp_type = f"xla::ffi::ResultBuffer<{ jax_dtype }>"
            bind = f"Ret<xla::ffi::Buffer<{ jax_dtype }>>"
        else:
            cpp_type = f"xla::ffi::Buffer<{ jax_dtype }>"
            bind = f"Arg<xla::ffi::Buffer<{ jax_dtype }>>"

        self._add_jax_ffi_arg( for_return, differentiable, JaxFfiArg(
            cpp_type = cpp_type,
            value = value,
            valid = valid,
            name = name,
            bind = bind,
        ) )


    def _add_jax_ffi_arg( self, for_return, differentiable, jax_ffi_arg ):
        if for_return:
            if differentiable:
                self.differentiable_jax_ffi_output_args.append( jax_ffi_arg )
            else:
                self.non_differentiable_jax_ffi_output_args.append( jax_ffi_arg )
        else:
            if differentiable:
                self.differentiable_jax_ffi_input_args.append( jax_ffi_arg )
            else:
                self.non_differentiable_jax_ffi_input_args.append( jax_ffi_arg )
