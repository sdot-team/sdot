class CtInt:
    """
    used in binding arg to specify a compile time integer value
    """

    def __init__( self, value: int ):
        self.value = value

    def cpp_class_name( self ):
        return f"CtInt<{ self.value }>"

    def to_nanobind_compatible_objects( self ):
        return []
