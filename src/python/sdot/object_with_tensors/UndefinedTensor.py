class UndefinedTensor:
    def __init__( self, shape, dtype ):
        self.shape = shape
        self.dtype = dtype

    def cpp_class_name( self ):
        if self.dtype == int:
            return "MI"

        if self.dtype is not None:
            raise NotImplementedError( f"cpp_class_name with dtype = { self.dtype }" )

        return "MF"

    def to_standard_objects( self ):
        if self.dtype == int:
            return [ ( None, "MI" ) ]

        if self.dtype is not None:
            raise NotImplementedError( f"to_standard_objects with dtype = { self.dtype }" )

        return [ ( None, "MF" ) ]

    def from_standard_objects( self, obj, arg_names ):
        return f"tensor_view_{ len( self.shape ) }( { arg_names.pop( 0 ) } )"
