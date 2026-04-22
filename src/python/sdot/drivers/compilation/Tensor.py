from .JaxFfiCompatibleItem import JaxFfiCompatibleItem

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
        from ...driver import driver
        return f"R{ len( shape ) }{ driver.normalized_type_for( dtype ) }"

    @staticmethod
    def as_jax_ffi_compatible_rets( driver, name, shape, dtype = None ) -> list[ JaxFfiCompatibleItem ]:
        import jax
        dn = driver._cpp_ffi_type_name( dtype or driver.dtype )
        differentiable = not driver.is_int_dtype( dtype or driver.dtype )
        return [ JaxFfiCompatibleItem( None, name, True, f"Ret<xla::ffi::Buffer<{ dn }>>", f"xla::ffi::ResultBuffer<{ dn }>", differentiable, spec = jax.ShapeDtypeStruct( shape, dtype or driver.dtype ) ) ]

    @staticmethod
    def cpp_assembly_from_jax_ffi_compatible_args( name, pos_in_validity_bits, shape, dtype = None ):
        p = pos_in_validity_bits[ 0 ]
        pos_in_validity_bits[ 0 ] += 1
        return f"tensor_view( CtInt<{ len( shape ) }>(), { name }, validity_mask[ { p // 64 } ] & { 1 << ( p % 64 ) } )"

    @staticmethod
    def python_assembly_from_jax_ffi_compatible_args( driver, flat_arg_iterator, shape, dtype = None ):
        try:
            return next( flat_arg_iterator )
        except StopIteration:
            return driver.empty( [ 0 ] * len( shape ), dtype = dtype )

    @staticmethod
    def fake_instance( driver, shape, dtype = None ):
        return driver.empty( shape, dtype = dtype )
