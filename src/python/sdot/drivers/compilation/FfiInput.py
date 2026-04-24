class FfiInput:
    """
        description of an input argument for ffi calls

    """

    python_value: any # python value
    cpp_type    : str # the cpp argument type (like xla::ffi::ResultBuffer<FP32>)
    valid       : bool # False if comes from UndefinedTensor, Zero grad, etc...
    bind        : str # binding name (like Ret<xla::ffi::Buffer<FP32>>)

    def __init__( self, python_value, valid, bind, cpp_type ):
        self.python_value = python_value
        self.cpp_type = cpp_type
        self.valid = valid
        self.bind = bind

