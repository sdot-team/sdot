
class Return:
    """Declares what type a C++ function returns.

    Usage:
        driver.call( "make_hypercube", includes,
            Return( Cell, dim=2 ),
            frame, bnd
        )
        driver.call( "measure", includes,
            Return( Tensor, shape=[], dtype=float ),
            cell
        )
    The return_type must implement the protocol:
        return_type.output_specs( drv, **kwargs ) -> list[ (name, shape, dtype) ]
        return_type.from_outputs( arrays, **kwargs ) -> instance
    """
    def __init__( self, return_type, *args, **kwargs ):
        self.return_type = return_type
        self.type_kwargs = kwargs
        self.type_args   = args

    def cpp_class_name( self ):
        if self.return_type is float:
            return "TF"

        if self.return_type is int:
            return "PI"

        return self.return_type.cpp_class_name_for( *self.type_args, **self.type_kwargs )

    def as_jax_ffi_compatible_args( self, driver ):
        return self.return_type.as_jax_ffi_compatible_rets( *self.type_args, **self.type_kwargs )

    def make_fake_instance( self, driver ):
        return self.return_type.make_fake_instance( *self.type_args, **self.type_kwargs )

    def cpp_assembly_from_jax_ffi_compatible_args( self, driver, flat_arg_iterator ):
        return self.return_type.cpp_assembly_from_jax_ffi_compatible_args( flat_arg_iterator, *self.type_args, **self.type_kwargs )
