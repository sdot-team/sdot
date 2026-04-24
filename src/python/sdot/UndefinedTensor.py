# from sdot.drivers.compilation.FfiValueList import JaxFfiCompatibleItem

class UndefinedTensor:
    def __init__( self, shape, dtype = None ):
        self.shape = [ s or 0 for s in shape ]
        self.dtype = dtype

    @property
    def ndim( self ):
        return len( self.shape )

    def cpp_class_name( self, driver ):
        return f"R{ len( self.shape ) }{ driver.normalized_type_for( self.dtype ) }"

    def configure_call_arg( self, jax_ffi_arg_list, driver, name, cpp_arg ):
        jax_ffi_arg_list._add_tensor_arg(
            driver,
            jax_ffi_arg_list._tensor_value( driver, [ 0 for _ in self.shape ], self.dtype ),
            jax_ffi_arg_list._tensor_spec( driver, [ 0 for _ in self.shape ], self.dtype ),
            name,
            cpp_arg,
            valid = False
        )

    def to_nanobind_compatible_objects( self ):
        if self.dtype == int:
            return [ ( self, "MI" ) ]

        if self.dtype is not None:
            raise NotImplementedError( f"to_nanobind_compatible_objects with dtype = { self.dtype }" )

        return [ ( self, "MF" ) ]

    # def diffentiable_tensors( self ):
    #     if self.dtype == int:
    #         return []
    #     return [ self ]

    # def cpp_assembly_from_nanobind_compatible_objects( self, obj, arg_names, use_view = False ):
    #     name = arg_names.pop( 0 )
    #     if use_view:
    #         return name
    #     return f"tensor_view_{ self.ndim }( { name } )"

    # def cpp_assembly_from_jax_ffi_compatible_args( self, driver, name, pos_in_validity_bits ):
    #     p = pos_in_validity_bits[ 0 ]
    #     pos_in_validity_bits[ 0 ] += 1
    #     return f"tensor_view( CtInt<{ len( self.shape ) }>(), { name }, validity_mask[ { p // 64 } ] & { 1 << ( p % 64 ) } )"

    # def as_jax_ffi_compatible_args( self, driver, name ) -> list[ JaxFfiCompatibleItem ]:
    #     dtype = driver._cpp_ffi_type_name( self.dtype )
    #     differentiable = not driver.is_int_dtype( self.dtype )
    #     return [ JaxFfiCompatibleItem( driver.empty( [ 0 ] * self.ndim, dtype = self.dtype ), name, False, f"Arg<xla::ffi::Buffer<{ dtype }>>", f"xla::ffi::Buffer<{ dtype }>", differentiable ) ]

    # def as_jax_ffi_compatible_rets( self, driver, name ) -> list[ JaxFfiCompatibleItem ]:
    #     dtype = driver._cpp_ffi_type_name( self.dtype )
    #     differentiable = not driver.is_int_dtype( self.dtype )
    #     return [ JaxFfiCompatibleItem( driver._jax_shape_out( self.shape, self.dtype ), name, False, f"Arg<xla::ffi::Buffer<{ dtype }>>", f"xla::ffi::Buffer<{ dtype }>", differentiable ) ]
