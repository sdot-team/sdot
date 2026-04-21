from ...driver import driver


class Tensor:
    """Sentinel type for returning a raw array from driver.call.

    Usage:
        driver.call( "measure", includes,
            Return( Tensor, shape=[], dtype=float ),
            cell
        )
    """

    @staticmethod
    def cpp_class_name_for( shape, dtype = None ):
        return f"R{ len( shape ) }{ driver.normalized_type_for( dtype ) }"

    @staticmethod
    def as_jax_ffi_compatible_rets( shape, dtype = None ):
        import jax
        return [ ( jax.ShapeDtypeStruct( shape, dtype or driver.dtype ), f"Ret<xla::ffi::Buffer<{ driver.cpp_ffi_type_name( dtype ) }>>", f"xla::ffi::ResultBuffer<{ driver.cpp_ffi_type_name( dtype ) }>" ) ]

    @staticmethod
    def cpp_assembly_from_jax_ffi_compatible_args( flat_arg_iterator, shape, dtype = None ):
        return f"tensor_view( CtInt<{ len( shape ) }>(), { next( flat_arg_iterator ) } )"

    @staticmethod
    def make_fake_instance( shape, dtype = None ):
        return driver.empty( shape, dtype = dtype )
