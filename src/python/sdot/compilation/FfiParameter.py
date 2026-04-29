from dataclasses import dataclass

@dataclass
class FfiParameter:
    """
        description of an input argument for ffi calls

    """

    python_value : str #
    arg_name     : str #
    cpp_type     : str #
    bind         : str # binding name (like Attr<xla::ffi::Buffer<FP32>>)


