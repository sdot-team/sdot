from .FfiParameter import FfiParameter
from .Workspace import Workspace
from .FfiOutput import FfiOutput
from .FfiInput import FfiInput
from .CallArg import CallArg
from .Return import Return
from .Tensor import Tensor
from ...Dyn import Dyn

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

        It also prepares a U64 tensor input (added in non_differentiable_ffi_inputs) with
        - a validity mask, to store in a dynamic fashion the invalid tensors and gradients
        - dynamic axe sizes

        Also, it can add a U64 output tensor for
        - update dynamic axe sizes
    """

    def __init__( self, call_args: dict, driver, parameters_struct = None ):
        # ffi values (for XLA ffi)
        self.non_differentiable_ffi_inputs: list[ FfiInput ] = []
        self.differentiable_ffi_inputs: list[ FfiInput ] = []
        self.ffi_parameters: list[ FfiParameter ] = []
        self.ffi_outputs: list[ FfiOutput ] = []

        # python/cpp values (for the user)
        self.call_args: list[ CallArg ] = [] # one for each call arg

        # code generation
        self.aggregates: dict[ CallArg ] = {} # type -> instance for each aggregate, i.e. object handled using collect_attributes

        # configuration
        self.parameters_struct = parameters_struct # struct name to use instead of the parameter list

        # input u64 value
        self.u64_input_bit_offset = 0 # where to set the next input bit (64 means that we will have to allocate a new U64 in self.u64_input_values)
        self.u64_input_values = []
        self.u64_ffi_input = None

        # input u64 value
        self.u64_output_bit_offset = -1 # where to set the next output bit (multiple of 64 means that we will have to allocate a new U64)
        self.u64_output_size = 0
        self.u64_ffi_output = None

        # update call_arg with DynamicSizes if some args contain dynamic axes
        FfiArgInfo.add_dynamic_size_tensors_to( call_args, driver )

        # recursive analysis
        for arg_name, arg_value in call_args.items():
            self.call_args.append( CallArg.analysis_of_python_arg( arg_value, arg_name, self, None, driver ) )

        # second pass : updates for dynamic axes, ...
        CallArg.second_pass_analysis( self.call_args, self, driver )

        # reservation of a bit to get info on dynamic size exception
        self.bit_offset_dynamic_size_exception = self.reserve_u64_output_bit()
        self.offset_needed_dynamic_size = self.reserve_u64_output( 1 )
        self.offset_dynamic_size_name = self.reserve_u64_output( ( self.max_axis_name_size() + 8 ) // 8 )
        self.end_dynamic_size_name = self.u64_output_size

        # register u64_input
        if self.u64_input_values:
            self.u64_ffi_input = FfiInput(
                represents_a_dynamic_axis = False,
                num_in_sub_list = -1,
                differentiable = False,
                validity_index = -1,
                requires_grad = False,
                python_value = numpy.array( self.u64_input_values, numpy.uint64 ),
                arg_name = "u64_input_buffer",
                cpp_type = driver.ffi_tensor_input_arg_code( 1, numpy.uint64 ),
                valid = None,
                bind = driver.ffi_tensor_input_bind_code( 1, numpy.uint64 ),
            )
            self.non_differentiable_ffi_inputs.append( self.u64_ffi_input )

        # register u64_output
        if self.u64_output_size:
            self.u64_ffi_output = FfiOutput(
                represents_a_dynamic_axis = False,
                num_in_sub_list = len( self.ffi_outputs ),
                differentiable = False,
                validity_index = -1,
                axis_names = [ "" ],
                ct_axes = {},
                arg_name = "u64_output_buffer",
                cpp_type = driver.ffi_tensor_output_arg_code( 1, numpy.uint64 ),
                valid = None,
                spec = driver.ffi_tensor_output_spec( [ self.u64_output_size ], numpy.uint64 ),
                bind = driver.ffi_tensor_output_bind_code( 1, numpy.uint64 ),
            )
            self.ffi_outputs.append( self.u64_ffi_output )

    def generate_structures( self ):
        for name, call_arg in self.aggregates.items():
            code = call_arg.generated_structure()
            incl = f"{ name }.h"

            from ...generated_files.compilation_directories import generated_includes_dir
            path = generated_includes_dir() / incl
            path.write_text( code )


    def reserve_u64_output_bit( self ):
        self.u64_output_bit_offset += 1
        if self.u64_output_bit_offset % 64 == 0:
            self.u64_output_bit_offset = 64 * self.u64_output_size
            self.u64_output_size += 1

        return self.u64_output_bit_offset

    def reserve_u64_output( self, nb_u64s = 1 ):
        res = self.u64_output_size
        self.u64_output_size += nb_u64s
        return res

    def max_axis_name_size( self ):
        max_size = 0
        for fo in self.ffi_outputs:
            for an in fo.axis_names:
                max_size = max( max_size, len( an ) )
        return max_size

    @staticmethod
    def add_dynamic_size_tensors_to( call_args: dict, driver ):
        dynamic_axes: list[ Dyn ] = []
        for call_arg in call_args.values():
            if isinstance( call_arg, Return ) and call_arg.return_type == Tensor:
                def get_shape( shape, **kwargs ):
                    return shape
                shape = get_shape( *call_arg.type_args, **call_arg.type_kwargs )
                for s in shape:
                    if isinstance( s, Dyn ):
                        item = next( ( da for da in dynamic_axes if da.name == s.name ), None )
                        if item is None:
                            dynamic_axes.append( s )

        for dynamic_axis in dynamic_axes:
            if dynamic_axis.name not in call_args:
                shape = []
                for a in dynamic_axis.one_value_for_each:
                    def get_shape():
                        for call_arg in call_args.values():
                            attr = getattr( call_arg, a, None )
                            if attr is not None:
                                return attr
                        raise RuntimeError( f"Unable to get '{ a }' axis size " )
                    shape.append( get_shape() )
                tf = Workspace( Tensor, shape = shape, axis_names = dynamic_axis.one_value_for_each, dtype = driver.uint64, represents_a_dynamic_axis = True )
                call_args[ dynamic_axis.name ] = tf


    def append_u64_input_bit( self, value ) -> int:
        """ append a bit in u64_input. return index """

        # position
        if self.u64_input_bit_offset % 64 == 0:
            res = 64 * len( self.u64_input_values )
            self.u64_input_values.append( 0 )
        else:
            res = self.u64_input_bit_offset

        self.u64_input_bit_offset = res + 1

        # value
        self.u64_input_values[ res // 64 ] |= ( 1 << ( res % 64 ) )

        return res

    def add_input_tensor( self, python_value: any, driver, valid = None, represents_a_dynamic_axis = False ) -> tuple[ str, int, FfiInput ]:
        if valid is None:
            valid = True
            if driver.is_zero_tensor( python_value ):
                python_value = numpy.empty( [ 0 ] * len( python_value.shape ), dtype = python_value.dtype )
                valid = False

        validity_index = self.append_u64_input_bit( valid )

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
            represents_a_dynamic_axis = represents_a_dynamic_axis,
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

    def add_output_tensor( self, driver, shape, dtype, axis_names, ct_axes: dict[ int ], valid = None, represents_a_dynamic_axis = False ) -> FfiOutput:
        """ return name, validity index """

        if valid is None:
            valid = True
            #     if driver.is_zero_tensor( python_value ):
            #         python_value = numpy.empty( [ 0 ] * len( python_value.shape ), dtype = python_value.dtype )
            #         valid = False

        if dtype is None:
            dtype = driver.dtype

        validity_index = self.append_u64_input_bit( valid )

        differentiable = True
        if not driver.differentiable_type( dtype ):
            differentiable = False

        num_in_sub_list = len( self.ffi_outputs )
        arg_name = f"out_{ num_in_sub_list }"

        ffi_output = FfiOutput(
            represents_a_dynamic_axis = represents_a_dynamic_axis,
            num_in_sub_list = num_in_sub_list,
            validity_index = validity_index,
            differentiable = differentiable, # we keep this information because we're not going to pass non differentiable outputs to the C++ call
            arg_name = arg_name,
            cpp_type = driver.ffi_tensor_output_arg_code( len( shape ), dtype ),
            valid = valid,
            bind = driver.ffi_tensor_output_bind_code( len( shape ), dtype ),
            spec = driver.ffi_tensor_output_spec( shape, dtype ),

            axis_names = axis_names,
            ct_axes = ct_axes,
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

        res = FfiParameter(
            python_value = python_value,
            arg_name = f"p{ n }",
            cpp_type = cpp_type,
            bind = f"Attr<{ cpp_type }>"
        )

        self.ffi_parameters.append( res )

        return res

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
        return [ input.python_value for input in self.non_differentiable_ffi_inputs + self.differentiable_ffi_inputs ]

    @property
    def attributes( self ):
        res = {}
        for p in self.ffi_parameters:
            res[ p.arg_name ] = p.python_value
        return res

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
            if call_arg.io_category == 3:  # DynamicAxis — structural only, no Python value
                continue
            if call_arg.io_category == 2:
                nargs[ call_arg.attribute_name ] = call_arg.construct( self, outputs )
            else:
                nargs[ call_arg.attribute_name ] = call_arg.python_value

        # add gradients
        for call_arg in self.call_args:
            call_arg.add_gradients_to( nargs, call_arg.attribute_name, driver, grads_of_the_outputs )

        # make the analysis (again)
        return FfiArgInfo( nargs, driver )


