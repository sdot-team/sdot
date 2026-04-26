from dataclasses import dataclass

@dataclass
class FfiOutput:
    """
        description of an output argument for ffi calls

    """

    represents_a_dynamic_axis: bool #
    num_in_sub_list          : int #
    differentiable           : bool #
    validity_index           : int #
    axis_names               : list[ str ] #
    arg_name                 : str #
    cpp_type                 : str # the cpp argument type (like xla::ffi::ResultBuffer<FP32>)
    valid                    : bool # False if comes from UndefinedTensor, Zero grad, etc...
    spec                     : any # python spec (jax.ShapeDtypeStruct for Jax, empty array for Torch)
    bind                     : str # binding name (like Ret<xla::ffi::Buffer<FP32>>)
