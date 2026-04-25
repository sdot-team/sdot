from .FfiParameter import FfiParameter
from .FfiOutput import FfiOutput
from .FfiInput import FfiInput
from .CallArg import CallArg

import numpy

class FfiArgInfo:
    """ take a list of generic arguments `args` that are sent to driver.call in the python side and get
        - flat lists for ffi arguments
            - non_differentiable_ffi_inputs
            - differentiable_ffi_inputs
            - ffi_parameters
            - ffi_outputs
        - list of CpyArg for each arg in `arg` that allows for
            - re-assembly and update of `args` from outputs of the jax calls (python side)
            - generation of the assembly code in C++

        It also prepares validity_mask (added in non_differentiable_ffi_inputs) to take into account the invalid tensors and gradients
    """

    def __init__( self, call_args: dict, driver ):
        # rec find sub args
        self.non_differentiable_ffi_inputs: list[ FfiInput ] = []
        self.differentiable_ffi_inputs: list[ FfiInput ] = []
        self.ffi_parameters: list[ FfiParameter ] = []
        self.ffi_outputs: list[ FfiOutput ] = []
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
        # self.validity_mask_index = len( self.non_differentiable_ffi_inputs )
        self.non_differentiable_ffi_inputs.append( FfiInput(
            num_in_sub_list = -1,
            differentiable = False,
            validity_index = -1,
            requires_grad = False,
            python_value = validity_mask,
            arg_name = "validity_mask_buffer",
            cpp_type = driver.ffi_tensor_input_arg_code( 1, numpy.uint64 ),
            valid = None,
            bind = driver.ffi_tensor_input_bind_code( 1, numpy.uint64 ),
        ) )

    def add_input_tensor( self, python_value: any, driver, valid = None ) -> tuple[ str, int, FfiInput ]:
        if valid is None:
            valid = True
            if driver.is_zero_tensor( python_value ):
                python_value = numpy.empty( [ 0 ] * len( python_value.shape ), dtype = python_value.dtype )
                valid = False

        validity_index = len( self.validity_mask_values )
        self.validity_mask_values.append( valid )

        differentiable = True
        requires_grad = True
        if not driver.differentiable_type( python_value.dtype ): # isinstance( python_value, numpy.ndarray ) or
            differentiable = False
            requires_grad = False
        if isinstance( python_value, numpy.ndarray ):
            requires_grad = False


        if differentiable:
            num_in_sub_list = len( self.differentiable_ffi_inputs )
            arg_name = f"di{ num_in_sub_list }"
        else:
            num_in_sub_list = len( self.non_differentiable_ffi_inputs )
            arg_name = f"ni{ num_in_sub_list }"

        ffi_input = FfiInput(
            num_in_sub_list = num_in_sub_list,
            validity_index = validity_index,
            differentiable = differentiable,
            requires_grad = requires_grad,
            python_value = python_value,
            arg_name = arg_name,
            cpp_type = driver.ffi_tensor_input_arg_code( len( python_value.shape ), python_value.dtype ),
            bind = driver.ffi_tensor_input_bind_code( len( python_value.shape ), python_value.dtype ),
            valid = valid,
        )

        if differentiable:
            self.differentiable_ffi_inputs.append( ffi_input )
        else:
            self.non_differentiable_ffi_inputs.append( ffi_input )

        return ffi_input

    def add_output_tensor( self, driver, shape, dtype, valid = None ) -> FfiOutput:
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

        num_in_sub_list = len( self.ffi_outputs )
        arg_name = f"out_{ num_in_sub_list }"

        ffi_output = FfiOutput(
            num_in_sub_list = num_in_sub_list,
            validity_index = validity_index,
            differentiable = differentiable, # we keep this information because we're not going to pass non differentiable outputs to the C++ call
            arg_name = arg_name,
            cpp_type = driver.ffi_tensor_output_arg_code( len( shape ), dtype ),
            valid = valid,
            bind = driver.ffi_tensor_output_bind_code( len( shape ), dtype ),
            spec = driver.ffi_tensor_output_spec( shape, dtype ),
        )

        self.ffi_outputs.append( ffi_output )

        return ffi_output

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

    def update_differentiable_input_values_with( self, differentiable_input_values ):
        # replace differentiable_ffi_inputs
        for n, v in enumerate( differentiable_input_values ):
            self.differentiable_ffi_inputs[ n ].python_value = v

        # update CallArgs
        for call_arg in self.call_args:
            call_arg.update_differentiable_input_values()


    @property
    def output_specs( self ):
        return [ output.spec for output in self.ffi_outputs ]

    @property
    def input_values( self ):
        return [ input.python_value for input in self.non_differentiable_ffi_inputs + self.differentiable_ffi_inputs + self.ffi_parameters ]

    @property
    def named_ffi_args( self ) -> list[ tuple[ str, FfiOutput | FfiInput | FfiOutput | FfiInput | FfiParameter ] ]:
        """ in the right order """
        res = []
        for arg in self.non_differentiable_ffi_inputs + self.differentiable_ffi_inputs + self.ffi_parameters:
            res.append( ( arg.arg_name, arg ) )
        for arg in self.ffi_outputs:
            res.append( ( arg.arg_name, arg ) )
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

        for arg in self.ffi_outputs:
            res.append( arg.bind + "()" )

        return res

    def backward_version( self, driver, outputs = [], grads_of_the_outputs = [] ):
        # turn fwd arg into inputs
        nargs = {}
        for call_arg in self.call_args:
            if call_arg.io_category == 2:
                nargs[ call_arg.attribute_name ] = call_arg.construct( self, outputs )
            else:
                nargs[ call_arg.attribute_name ] = call_arg.python_value

        # add gradients
        for call_arg in self.call_args:
            call_arg.add_gradients_to( nargs, call_arg.attribute_name, driver, grads_of_the_outputs )

        # make the analysis (again)
        return FfiArgInfo( nargs, driver )


