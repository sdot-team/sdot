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


class JaxFfiArgList:
    """ lists of arguments
    - flat lists of JaxFfi compatible items (tensors, returned tensors)
        - jax_ffi_non_differentiable_outputs
        - jax_ffi_non_differentiable_inputs
        - jax_ffi_differentiable_outputs
        - jax_ffi_differentiable_inputs
    - list of Python and Cpp side arguments
    """


    def __init__( self, driver, python_args: dict ):
        self.jax_ffi_non_differentiable_outputs = []
        self.jax_ffi_non_differentiable_inputs = []
        self.jax_ffi_differentiable_outputs = []
        self.jax_ffi_differentiable_inputs = []
        self.validity_mask = []
        self.cpp_args = [] # recursive list
        for python_arg_name, python_arg_value in python_args.items():
            self.get_jax_ffi_args( driver, python_arg_name, python_arg_value, self.cpp_args, False )

        info( self.jax_ffi_non_differentiable_outputs,
            self.jax_ffi_non_differentiable_inputs,
            self.jax_ffi_differentiable_outputs,
            self.jax_ffi_differentiable_inputs,
            self.validity_mask,
            self.cpp_args )
        import sys
        sys.exit( 0 )

    def get_jax_ffi_args( self, driver, name: str, value: any, cpp_args: list, for_return: bool ):
        """ recursively fill the jax_ffi_... lists """

        # method
        if callable( getattr( value, "get_jax_ffi_args", None ) ):
            return value.get_jax_ffi_args( self, driver, name, cpp_args, for_return )

        # SymbolicZero
        if isinstance( value, ad_util.SymbolicZero ):
            array = self._tensor_value( driver, value.shape, value.dtype, for_return )
            return self._add_tensor_arg( driver, array, name, cpp_args, for_return, valid = False )

        # arrays
        if isinstance( value, ( jax.core.Tracer, jax.Array, numpy.ndarray ) ):
            return self._add_tensor_arg( driver, value, name, cpp_args, for_return )

        # std objects
        if isinstance( value, float ):
            array = driver.tn( value, ndim = 0, dtype = driver.dtype )
            return self._add_tensor_arg( driver, array, name, cpp_args, for_return )

        if isinstance( value, int ):
            array = driver.tn( value, ndim = 0, dtype = driver.itype )
            return self._add_tensor_arg( driver, array, name, cpp_args, for_return )

        if isinstance( value, ( list, tuple ) ):
            sublist = []
            cpp_args.append( ( "std::tie", sublist ) )
            for num, item in enumerate( value ):
                self.get_jax_ffi_args( driver, f"{ name }_{ num }", item, sublist, for_return )
            return

        if value is None:
            cpp_args.append( "{}" )
            return

        # else, get attributes
        sublist = []
        cpp_args.append( ( self.cpp_class_name( driver, value ), sublist ) )
        for attr, _ in collect_attributes( value ):
            self.get_jax_ffi_args( driver, f"{ name }_{ attr }", getattr( value, attr ), sublist, for_return )


    def _tensor_value( self, driver, shape, dtype, for_return ):
        if for_return:
            return jax.ShapeDtypeStruct( shape, dtype )
        return driver.empty( [ 0 ] * len( cast( tuple, shape ) ), dtype = dtype )


    def cpp_class_name( self, driver, value ):
        if callable( getattr( value, "cpp_class_name", None ) ):
            return value.cpp_class_name( driver )
        return value.__class_name__


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


    def _add_tensor_arg( self, driver, value: any, name: str, cpp_args: list, for_return: bool, valid: bool = True ):
        #
        p = len( self.validity_mask )
        self.validity_mask.append( valid )
        cpp_args.append( f"tensor_view( CtInt<{ len( value.shape ) }>(), { name }, validity_mask[ { p // 64 } ] & { 1 << ( p % 64 ) } )" )

        #
        jax_dtype = self._jax_dtype( driver, value.dtype )
        if for_return:
            cpp_type = f"xla::ffi::ResultBuffer<{ jax_dtype }>"
            bind = f"Ret<xla::ffi::Buffer<{ jax_dtype }>>"
        else:
            cpp_type = f"xla::ffi::Buffer<{ jax_dtype }>"
            bind = f"Arg<xla::ffi::Buffer<{ jax_dtype }>>"

        self._add_jax_ffi_arg( for_return, self._differentiable_dtype( value.dtype ), JaxFfiArg(
            cpp_type = cpp_type,
            value = value,
            valid = valid,
            name = name,
            bind = bind,
        ) )


    def _add_jax_ffi_arg( self, for_return, differentiable, jax_ffi_arg ):
        if for_return:
            if differentiable:
                self.jax_ffi_differentiable_outputs.append( jax_ffi_arg )
            else:
                self.jax_ffi_non_differentiable_outputs.append( jax_ffi_arg )
        else:
            if differentiable:
                self.jax_ffi_differentiable_inputs.append( jax_ffi_arg )
            else:
                self.jax_ffi_non_differentiable_inputs.append( jax_ffi_arg )
