class FfiInput:
    """
        description of an input argument for ffi calls

        - cpp_type : the cpp argument type (like xla::ffi::ResultBuffer<FP32>)
        - value    : python value
        - valid    : False if comes from UndefinedTensor, Zero grad, etc...
        - bind     : binding name (like Ret<xla::ffi::Buffer<FP32>>)
    """

    def __init__( self, value, valid, bind, cpp_type ):
        self.cpp_type = cpp_type
        self.value = value
        self.valid = valid
        self.bind = bind

