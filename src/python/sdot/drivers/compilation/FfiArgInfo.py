# from .collect_attributes import collect_attributes
from .FfiParameter import FfiParameter
from .FfiOutput import FfiOutput
from .FfiInput import FfiInput
from .CallArg import CallArg

import numpy

# from .Return import Return
# from .Tensor import Tensor

# from jax._src import ad_util
# import jax.numpy
# import jax.core
# import jax


class FfiArgInfo:
    """ take a list of generic arguments `args` that are sent to driver.call in the python side and get
        - flat lists of FfiInput and FfiOutput, splitted in the following categories
            - non_differentiable_ffi_outputs
            - non_differentiable_ffi_inputs
            - differentiable_ffi_outputs
            - differentiable_ffi_inputs
            - ffi_parameters
        - list of CpyArg for each arg in `arg` that allows for
            - re-assembly and update of `args` from outputs of the jax calls (python side)
            - generation of the assembly code in C++

        It also prepares validity_mask (added in non_differentiable_ffi_inputs) to take into account the invalid tensors and gradients
    """

    def __init__( self, call_args: dict, driver ):
        # rec find sub args
        self.non_differentiable_ffi_outputs: list[ FfiOutput ] = []
        self.non_differentiable_ffi_inputs: list[ FfiInput ] = []
        self.differentiable_ffi_outputs: list[ FfiOutput ] = []
        self.differentiable_ffi_inputs: list[ FfiInput ] = []
        self.ffi_parameters: list[ FfiParameter ] = []
        self.call_args: list[ CallArg ] = [] # one for each call arg
        self.validity_mask_values = []

        for arg_name, arg_value in call_args.items():
            self.call_args.append( CallArg.analysis_of_python_arg( arg_value, arg_name, self, False, driver ) )

        # make validity_input_arg
        validity_mask = numpy.zeros( [ ( len( self.validity_mask_values ) + 63 ) // 64 ], dtype = numpy.uint64 )
        for n, v in enumerate( self.validity_mask_values ):
           if v:
               validity_mask[ n // 64 ] |= ( 1 << ( n % 64 ) )

        # register the validity_mask
        self.validity_mask_index = len( self.non_differentiable_ffi_inputs )
        self.non_differentiable_ffi_inputs.append( FfiInput(
            python_value = validity_mask,
            cpp_type = driver.ffi_tensor_input_arg_code( 1, numpy.uint64 ),
            valid = None,
            bind = driver.ffi_tensor_input_bind_code( 1, numpy.uint64 ),
        ) )

    def add_input_tensor( self, python_value: any, driver, valid = None ) -> tuple[ str, int ]:
        if valid is None:
            valid = True
            if driver.is_zero_tensor( python_value ):
                python_value = numpy.empty( [ 0 ] * len( python_value.shape ), dtype = python_value.dtype )
                valid = False

        validity_index = len( self.validity_mask_values )
        self.validity_mask_values.append( valid )

        differentiable = True
        if isinstance( python_value, numpy.ndarray ) or not driver.differentiable_type( python_value.dtype ):
            differentiable = False

        ffi_input = FfiInput(
            python_value = python_value,
            cpp_type = driver.ffi_tensor_input_arg_code( len( python_value.shape ), python_value.dtype ),
            bind = driver.ffi_tensor_input_bind_code( len( python_value.shape ), python_value.dtype ),
            valid = valid,
        )

        if differentiable:
            num_variable = len( self.differentiable_ffi_inputs )
            self.differentiable_ffi_inputs.append( ffi_input )
            return f"di{ num_variable }", validity_index
        else:
            num_variable = len( self.non_differentiable_ffi_inputs )
            self.non_differentiable_ffi_inputs.append( ffi_input )
            return f"ni{ num_variable }", validity_index

    def add_output_tensor( self, driver, shape, dtype, valid = None ) -> tuple[ str, int ]:
        """ return name, validity index """

        if valid is None:
            valid = True
            #     if driver.is_zero_tensor( python_value ):
            #         python_value = numpy.empty( [ 0 ] * len( python_value.shape ), dtype = python_value.dtype )
            #         valid = False

        if dtype is None:
            dtype = driver.dtype

        validity_index = len( self.validity_mask_values )
        self.validity_mask_values.append( valid )

        differentiable = True
        if not driver.differentiable_type( dtype ):
            differentiable = False

        ffi_output = FfiOutput(
            cpp_type = driver.ffi_tensor_output_arg_code( len( shape ), dtype ),
            valid = valid,
            bind = driver.ffi_tensor_output_bind_code( len( shape ), dtype ),
            spec = driver.ffi_tensor_output_spec( shape, dtype ),
        )

        if differentiable:
            num_variable = len( self.differentiable_ffi_outputs )
            self.differentiable_ffi_outputs.append( ffi_output )
            return f"do{ num_variable }", validity_index
        else:
            num_variable = len( self.non_differentiable_ffi_outputs )
            self.non_differentiable_ffi_outputs.append( ffi_output )
            return f"no{ num_variable }", validity_index

    def add_raw_return( self, return_type, valid, driver, *args, **kwargs ) -> tuple[ str, int ]:
        n = len( self.non_differentiable_ffi_outputs )

        cpp_type = str( return_type )

        self.ffi_parameters.append( FfiOutput(
            cpp_type = cpp_type,
            valid = valid,
            spec = "...",
            bind = "...",
            # bind = "Attr<float>"
        ) )

        return f"p{ n }"


    def add_parameter( self, python_value: any, cpp_type: str ) -> str:
        n = len( self.ffi_parameters )

        self.ffi_parameters.append( FfiParameter(
            python_value = python_value,
            cpp_type = cpp_type,
            bind = "Attr<float>"
        ) )

        return f"p{ n }"

    @property
    def output_specs( self ):
        return [ output.spec for output in self.non_differentiable_ffi_outputs + self.differentiable_ffi_outputs ]

    @property
    def input_values( self ):
        return [ input.python_value for input in self.non_differentiable_ffi_inputs + self.differentiable_ffi_inputs + self.ffi_parameters ]

    @property
    def name_of_validity_mask( self ):
        return f"ni{ self.validity_mask_index }"

    @property
    def named_ffi_args( self ) -> list[ tuple[ str, FfiOutput | FfiInput | FfiOutput | FfiInput | FfiParameter ] ]:
        res = []

        for n, arg in enumerate( self.non_differentiable_ffi_inputs ):
            res.append( ( f"ni{ n }", arg ) )
        for n, arg in enumerate( self.differentiable_ffi_inputs ):
            res.append( ( f"di{ n }", arg ) )
        for n, arg in enumerate( self.ffi_parameters ):
            res.append( ( f"p{ n }", arg ) )

        for n, arg in enumerate( self.non_differentiable_ffi_outputs ):
            res.append( ( f"no{ n }", arg ) )
        for n, arg in enumerate( self.differentiable_ffi_outputs ):
            res.append( ( f"do{ n }", arg ) )

        return res

    @property
    def bind_chain( self ) -> list[ str ]:
        res = []

        for arg in self.non_differentiable_ffi_inputs:
            res.append( arg.bind + "()" )
        for arg in self.differentiable_ffi_inputs:
            res.append( arg.bind + "()" )
        for n, arg in enumerate( self.ffi_parameters ):
            res.append( arg.bind + f"( \"p{ n }\" )" )

        for arg in self.non_differentiable_ffi_outputs:
            res.append( arg.bind + "()" )
        for arg in self.differentiable_ffi_outputs:
            res.append( arg.bind + "()" )

        return res


    # def non_differentiable_jax_ffi_output_specs( self ):
    #     return [ output.spec for output in self.non_differentiable_ffi_outputs ]

    # @property
    # def differentiable_jax_ffi_output_specs( self ):
    #     return [ output.spec for output in self.differentiable_ffi_outputs ]

    # @property
    # def non_differentiable_jax_ffi_input_values( self ):
    #     return [ input.value for input in self.non_differentiable_ffi_inputs ]

    # @property
    # def differentiable_jax_ffi_input_values( self ):
    #     return [ input.value for input in self.differentiable_ffi_inputs ]

    # @property
    # def jax_ffi_output_args( self ) -> list[ FfiOutput ]:
    #     return self.differentiable_ffi_outputs + self.non_differentiable_ffi_outputs

    # @property
    # def jax_ffi_input_args( self ) -> list[ FfiOutput ]:
    #     return self.differentiable_ffi_inputs + self.non_differentiable_ffi_inputs

    # def backward_version( self, driver, differentiable_inputs, non_differentiable_inputs, differentiable_outputs, non_differentiable_outputs, grad_outputs ):
    #     # turn fwd arg into inputs
    #     nargs = {}
    #     for cpy_arg in self.call_args:
    #         value = cpy_arg.reassemble( differentiable_inputs, non_differentiable_inputs, differentiable_outputs, non_differentiable_outputs )
    #         nargs[ cpy_arg.name ] = value

    #     # grad of the outputs (-> inputs)
    #     for n in range( len( grad_outputs ) ):
    #         farg = self.differentiable_ffi_outputs[ n ]
    #         nargs[ "grad_" + farg.name ] = grad_outputs[ n ]

    #     # return grad of the inputs (-> inputs)
    #     for n in range( len( differentiable_inputs ) ):
    #         farg = self.differentiable_ffi_inputs[ n ]
    #         nargs[ "grad_" + farg.name ] = Return( Tensor, differentiable_inputs[ n ].shape )

    #     return FfiArgInfo( driver, nargs )

    # def configure_call_arg( self, driver, name: str, value: any, cpy_arg: CallArg ):
    #     """ recursively fill the jax_ffi_... lists """

    #     # method
    #     if callable( getattr( value, "configure_call_arg", None ) ):
    #         return value.configure_call_arg( self, driver, name, cpy_arg )

    #     # arrays
    #     if isinstance( value, ( jax.core.Tracer, jax.Array, numpy.ndarray, jax.ShapeDtypeStruct ) ):
    #         return self._add_tensor_arg( driver, value, jax.ShapeDtypeStruct( value.shape, value.dtype ), name, cpy_arg )

    #     # std objects
    #     if isinstance( value, float ):
    #         array = driver.tn( value, ndim = 0, dtype = driver.dtype )
    #         return self._add_tensor_arg( driver, array, jax.ShapeDtypeStruct( [], driver.dtype ), name, cpy_arg )

    #     if isinstance( value, int ):
    #         array = driver.tn( value, ndim = 0, dtype = driver.itype )
    #         return self._add_tensor_arg( driver, array, jax.ShapeDtypeStruct( [], driver.itype ), name, cpy_arg )

    #     if isinstance( value, ( list, tuple ) ):
    #         cpy_arg._python_ctor = type( value )
    #         cpy_arg.signature_type = "L"
    #         cpy_arg.code = "std::tie"
    #         for num, item in enumerate( value ):
    #             self.configure_call_arg( driver, f"{ name }_{ num }", item, cpy_arg.arg( str( num ) ) )
    #         return

    #     if value is None:
    #         cpy_arg.signature_type = "N"
    #         cpy_arg.code = "{}"
    #         return

    #     # else, get attributes
    #     cpy_arg.code = FfiArgInfo.cpp_class_name( driver, value )
    #     cpy_arg.signature_type = cpy_arg.code
    #     cpy_arg._python_ctor = type( value )
    #     for attr, _ in collect_attributes( value ):
    #         self.configure_call_arg( driver, f"{ name }_{ attr }", getattr( value, attr ), cpy_arg.arg( attr ) )


    # def _tensor_spec( self, driver, shape, dtype ):
    #     return jax.ShapeDtypeStruct( shape, dtype or driver.dtype )

    # def _tensor_value( self, driver, shape, dtype ):
    #     return driver.empty( shape, dtype = dtype )

    # @staticmethod
    # def cpp_class_name( driver, value ):
    #     if callable( getattr( value, "cpp_class_name", None ) ):
    #         return value.cpp_class_name( driver )
    #     return value.__class__.__name__


    # def _differentiable_dtype( self, dtype ):
    #     if dtype is None or dtype is float:
    #         return True
    #     if dtype is int:
    #         return False
    #     return not jax.numpy.issubdtype( dtype, jax.numpy.integer )




    # def _add_tensor_arg( self, driver, value: any, spec: any, name: str, cpy_arg: CallArg, valid: bool = True ):
    #     # differentiable
    #     differentiable = self._differentiable_dtype( value.dtype )

    #     # signature_type
    #     cpy_arg.signature_type = f"T{ len( value.shape ) }{ driver.normalized_type_for( value.dtype ) }"
    #     jax_dtype = self._jax_dtype( driver, value.dtype )

    #     # validity
    #     validity = len( self.validity_mask_values )
    #     self.validity_mask_values.append( valid )

    #     # cpy_arg.value
    #     tv_arg = name
    #     if cpy_arg.for_return == 1: # mutable => start out with a copy of inp
    #        tv_arg = f"{ name }_out, { name }"

    #     cpy_arg.code = f"tensor_view( CtInt<{ len( value.shape ) }>(), { tv_arg }, validity_mask[ { validity // 64 } ] & { 1 << ( validity % 64 ) } )"

    #     # input
    #     if cpy_arg.for_return <= 1:
    #         self._add_jax_ffi_arg_input( cpy_arg, valid, differentiable = differentiable, jax_ffi_arg = FfiValue(
    #             cpp_type = f"xla::ffi::Buffer<{ jax_dtype }>",
    #             bind = f"Arg<xla::ffi::Buffer<{ jax_dtype }>>",
    #             value = value,
    #             valid = valid,
    #             spec = spec,
    #             name = name,
    #         ) )

    #     # output
    #     if cpy_arg.for_return >= 1:
    #         self._add_jax_ffi_arg_output( cpy_arg, valid, differentiable = differentiable, jax_ffi_arg = FfiValue(
    #             cpp_type = f"xla::ffi::ResultBuffer<{ jax_dtype }>",
    #             bind = f"Ret<xla::ffi::Buffer<{ jax_dtype }>>",
    #             value = value,
    #             valid = valid,
    #             spec = spec,
    #             name = name + "_out" if cpy_arg.for_return == 1 else name,
    #         ) )


    # def _add_jax_ffi_arg_output( self, cpy_arg, valid, differentiable, jax_ffi_arg ):
    #     if differentiable:
    #         cpy_arg._python_ctor = ( 1, len( self.differentiable_ffi_outputs ), valid )
    #         self.differentiable_ffi_outputs.append( jax_ffi_arg )
    #     else:
    #         cpy_arg._python_ctor = ( 0, len( self.non_differentiable_ffi_outputs ), valid )
    #         self.non_differentiable_ffi_outputs.append( jax_ffi_arg )

    # def _add_jax_ffi_arg_input( self, cpy_arg, valid, differentiable, jax_ffi_arg ):
    #     if differentiable:
    #         cpy_arg._python_ctor = ( 1, len( self.differentiable_ffi_inputs ), valid )
    #         self.differentiable_ffi_inputs.append( jax_ffi_arg )
    #     else:
    #         cpy_arg._python_ctor = ( 0, len( self.differentiable_ffi_inputs ), valid )
    #         self.non_differentiable_ffi_inputs.append( jax_ffi_arg )

