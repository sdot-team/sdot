from dataclasses import dataclass

@dataclass
class FfiInput:
    """
        description of an input argument for ffi calls

    """

    num_in_sub_list : int #
    differentiable  : bool #
    validity_index  : int #
    requires_grad   : bool #
    python_value    : any # python value
    arg_name        : str #
    cpp_type        : str # the cpp argument type (like xla::ffi::ResultBuffer<FP32>)
    valid           : bool # False if comes from UndefinedTensor, Zero grad, etc...
    bind            : str # binding name (like Ret<xla::ffi::Buffer<FP32>>)
