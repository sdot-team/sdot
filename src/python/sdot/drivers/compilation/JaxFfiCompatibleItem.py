class JaxFfiCompatibleItem:
    """
        - the compatibie argument value
        - a boolean to say if it's valid
        - the binding name (like Ret<xla::ffi::Buffer<FP32>>)
        - the argument name (like xla::ffi::ResultBuffer<FP32>)
        - ...
    """
    def __init__( self, value, name, valid, bind, cpp_type, differentiable, spec = None ):
        self.differentiable = differentiable
        self.cpp_type = cpp_type
        self.value = value
        self.valid = valid
        self.spec = spec
        self.name = name
        self.bind = bind

