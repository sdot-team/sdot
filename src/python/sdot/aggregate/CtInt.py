class CtInt:
    """
    used in binding arg to specify a compile time integer value
    """

    def __init__( self, value: int ):
        self.value = value

    def cpp_class_name( self ):
        return f"CtInt<{ self.value }>"

    def configure_call_arg( self, call_arg, fai, mutable, driver ):
        call_arg.base_code = f"CtInt<{ self.value }>()"
        call_arg.signature = f"C{ self.value }"
