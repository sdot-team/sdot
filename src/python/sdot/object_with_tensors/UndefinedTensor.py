# from ..driver import driver

class UndefinedTensor:
    def __init__( self, shape, dtype ):
        self.shape = shape
        self.dtype = dtype

    @property
    def ndim( self ):
        return len( self.shape )

    def cpp_class_name( self ):
        if self.dtype == int:
            return "MI"

        if self.dtype is not None:
            raise NotImplementedError( f"cpp_class_name with dtype = { self.dtype }" )

        return "MF"

    def to_nanobind_compatible_objects( self ):
        if self.dtype == int:
            return [ ( self, "MI" ) ]

        if self.dtype is not None:
            raise NotImplementedError( f"to_nanobind_compatible_objects with dtype = { self.dtype }" )

        return [ ( self, "MF" ) ]

    def diffentiable_tensors( self ):
        if self.dtype == int:
            return []
        return [ self ]

    def cpp_assembly_from_nanobind_compatible_objects( self, obj, arg_names, use_view = False ):
        name = arg_names.pop( 0 )
        if use_view:
            return name
        return f"tensor_view_{ self.ndim }( { name } )"

    def as_jax_ffi_compatible_args( self, driver, name ) -> list[ tuple[ any, str, bool, str, str ] ]:
        dtype = driver._cpp_ffi_type_name( self.dtype )
        return [ ( driver.empty( [ 0 ] * self.ndim, dtype = self.dtype ), name, False, f"Arg<xla::ffi::Buffer<{ dtype }>>", f"xla::ffi::Buffer<{ dtype }>" ) ]

    def as_jax_ffi_compatible_rets( self, driver, name ) -> list[ tuple[ any, str, bool, str, str ] ]:
        dtype = driver._cpp_ffi_type_name( self.dtype )
        return [ ( driver._jax_shape_out( self.shape, self.dtype ), name, False, f"Arg<xla::ffi::Buffer<{ dtype }>>", f"xla::ffi::Buffer<{ dtype }>" ) ]
