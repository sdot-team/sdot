
class Return:
    """

    """
    def __init__( self, value ):
        self.value = value

    def cpp_class_name( self ):
        from ._types import cpp_class_name
        return cpp_class_name( self.value )

    def to_standard_objects( self ):
        from ._types import to_standard_objects
        return to_standard_objects( self.value )

    def diffentiable_tensors( self ):
        from ._types import diffentiable_tensors_of
        return diffentiable_tensors_of( self.value )
