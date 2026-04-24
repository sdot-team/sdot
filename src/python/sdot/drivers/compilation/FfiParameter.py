class FfiParameter:
    """
        description of an input argument for ffi calls

        - cpp_type : the cpp argument type (like xla::ffi::ResultBuffer<FP32>)
        - value    : python value
        - valid    : False if comes from UndefinedTensor, Zero grad, etc...
        - bind     : binding name (like Ret<xla::ffi::Buffer<FP32>>)
        - name     : the cpp argument name
    """


    def __init__( self, python_value: any, bind: str, cpp_type: str ):
        self.python_value = python_value
        self.cpp_type = cpp_type
        self.bind = bind

