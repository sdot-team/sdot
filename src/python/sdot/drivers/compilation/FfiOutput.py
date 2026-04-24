class FfiOutput:
    """
        description of an output argument for ffi calls

    """

    cpp_type : str # the cpp argument type (like xla::ffi::ResultBuffer<FP32>)
    valid    : bool # False if comes from UndefinedTensor, Zero grad, etc...
    spec     : any # python spec (jax.ShapeDtypeStruct for Jax, empty array for Torch)
    bind     : str # binding name (like Ret<xla::ffi::Buffer<FP32>>)

    def __init__( self, spec: any, valid: bool, bind: str, cpp_type: str ):
        self.cpp_type = cpp_type
        self.valid = valid
        self.spec = spec
        self.bind = bind

