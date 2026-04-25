
class FfiDynamicAxis:
    """
    Represents a named dynamic axis shared across output tensors.

    Populated progressively as tensors are registered, then finalized
    once the shared sizes output buffer is known.
    """

    def __init__( self, name: str, pos_in_u64_output: int ):
        self.pos_in_u64_output = pos_in_u64_output
        self.capacity_sources = [] # list of ( ffi_output, tensor_axis_index )
        self.name = name #

    def add_capacity_source( self, ffi_output, tensor_axis: int ):
        self.capacity_sources.append( ( ffi_output, tensor_axis ) )

    # def finalize( self, ffi_output_for_dynamic_sizes ):
    #     if not self.capacity_sources:
    #         raise RuntimeError( f"DynamicAxis '{ self.name }' has no tensor registered for capacity" )
    #     src_output, tensor_axis = self.capacity_sources[ 0 ]
    #     self.dyn_call_arg.base_code = (
    #         f"DynamicAxis{{ { ffi_output_for_dynamic_sizes.arg_name }->typed_data() + { self.axis_index }, "
    #         f"SI( { src_output.arg_name }->dimensions()[{ tensor_axis }] ) }}"
    #     )
