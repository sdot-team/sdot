from .FfiOutput import FfiOutput
from .FfiInput import FfiInput


class FfiDynamicAxis:
    """
    Represents a named dynamic axis shared across output tensors.

    Populated progressively as tensors are registered, then finalized
    once the shared sizes output buffer is known.
    """

    def __init__( self, name: str, pos_in_u64_output: int, pos_in_u64_input: int ):
        self.non_differentiable_input_capacity_sources: list[ tuple[ FfiInput, int ] ] = [] # list of ( ffi_input, tensor_axis_index )
        self.differentiable_input_capacity_sources: list[ tuple[ FfiInput, int ] ] = [] # list of ( ffi_input, tensor_axis_index )
        self.output_capacity_sources: list[ tuple[ FfiOutput, int ] ] = [] # list of ( ffi_output, tensor_axis_index )

        self.pos_in_u64_output = pos_in_u64_output
        self.pos_in_u64_input = pos_in_u64_input
        self.name = name

    def add_output_capacity_source( self, ffi_output: FfiOutput, tensor_axis: int ):
        self.output_capacity_sources.append( ( ffi_output, tensor_axis ) )

    def ctor_code( self ) -> str:
        init_value = "0"
        if self.pos_in_u64_input >= 0:
            init_value = f"u64_input[ { self.pos_in_u64_input } ]"
        return f"DynamicAxis( reinterpret_cast<PI *>( u64_output + { self.pos_in_u64_output } ), { init_value }, { self.capacity_code() } )"

    def capacity_code( self ) -> str:
        sources = []
        for ffi_input, axis_index in self.non_differentiable_input_capacity_sources + self.differentiable_input_capacity_sources:
            sources.append( f"{ ffi_input.arg_name }, { ffi_input.validity_index }, { axis_index }" )
        for ffi_output, axis_index in self.output_capacity_sources:
            sources.append( f"{ ffi_output.arg_name }, { ffi_output.validity_index }, { axis_index }" )
        return f"first_valid_dimension( u64_input, { str.join( ", ", sources ) } )"

