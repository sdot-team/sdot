# from .FfiOutput import FfiOutput
# from .FfiInput import FfiInput


# class FfiDynamicAxis:
#     """
#     Represents a named dynamic axis shared across output tensors.

#     Sizes are stored in dedicated non-differentiable tensors:
#     - ffi_input  : initial sizes (read-only, supplied by Python)
#     - ffi_output : current sizes (written by C++ kernel)

#     For a scalar axis (one_value_for_each=[]):  both tensors are 0-D (shape [])
#     For a per-item axis (one_value_for_each=[...]): both tensors are 1-D (shape [N])
#     """

#     def __init__( self, name: str, ffi_input: FfiInput, ffi_output: FfiOutput ):
#         self.non_differentiable_input_capacity_sources: list[ tuple[ FfiInput, int ] ] = []
#         self.differentiable_input_capacity_sources: list[ tuple[ FfiInput, int ] ] = []
#         self.output_capacity_sources: list[ tuple[ FfiOutput, int ] ] = []

#         self.ffi_input = ffi_input
#         self.ffi_output = ffi_output
#         self.name = name

#     @property
#     def n_values( self ) -> int:
#         return int( self.ffi_input.python_value.size )  # numpy .size = total number of elements

#     def add_output_capacity_source( self, ffi_output: FfiOutput, tensor_axis: int ):
#         self.output_capacity_sources.append( ( ffi_output, tensor_axis ) )

#     def add_input_capacity_source( self, ffi_input: FfiInput, tensor_axis: int, differentiable: bool ):
#         if differentiable:
#             self.differentiable_input_capacity_sources.append( ( ffi_input, tensor_axis ) )
#         else:
#             self.non_differentiable_input_capacity_sources.append( ( ffi_input, tensor_axis ) )

#     def ctor_code( self ) -> str:
#         cap = self.capacity_code()
#         rank = self.ffi_input.python_value.ndim

#         def tv( ffi_arg ):
#             vi = f"u64_input[ { ffi_arg.validity_index // 64 } ] & { 1 << ( ffi_arg.validity_index % 64 ) }"
#             return f"tensor_view( CtInt<{ rank }>(), { ffi_arg.arg_name }, { vi } )"

#         return f"DynamicAxis<{ rank },Arch>( { tv( self.ffi_output ) }, { cap }, { tv( self.ffi_input ) } )"

#     def capacity_code( self ) -> str:
#         sources = []
#         for ffi_input, axis_index in self.non_differentiable_input_capacity_sources + self.differentiable_input_capacity_sources:
#             sources.append( f"{ ffi_input.arg_name }, { ffi_input.validity_index }, { axis_index }" )
#         for ffi_output, axis_index in self.output_capacity_sources:
#             sources.append( f"{ ffi_output.arg_name }, { ffi_output.validity_index }, { axis_index }" )
#         return f"first_valid_dimension( u64_input, { str.join( ', ', sources ) } )"
