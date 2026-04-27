
class UndefinedTensor:
    def __init__( self, shape, dtype, axis_names ):
        self.axis_names = axis_names
        self.shape = [ s or 0 for s in shape ]
        self.dtype = dtype

    @property
    def ndim( self ):
        return len( self.shape )

    def cpp_class_name( self, driver ):
        return f"R{ len( self.shape ) }{ driver.normalized_type_for( self.dtype ) }"

    def configure_call_arg( self, call_arg, fai, mutable, driver ):
        python_value = driver.array( [ 0 for _ in self.shape ], dtype = self.dtype )
        return call_arg.configure_as_input_tensor( python_value, mutable, fai, driver, axis_names = self.axis_names, valid = False )

    # def to_nanobind_compatible_objects( self ):
    #     if self.dtype == int:
    #         return [ ( self, "MI" ) ]

    #     if self.dtype is not None:
    #         raise NotImplementedError( f"to_nanobind_compatible_objects with dtype = { self.dtype }" )

    #     return [ ( self, "MF" ) ]

