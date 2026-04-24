
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

    def cpp_class_name( self, driver ):
        if self.return_type is float:
            return "TF"
        if self.return_type is int:
            return "PI"
        return self.return_type.cpp_class_name_for( *self.type_args, **self.type_kwargs )

    def call_arg_analysis( self, jax_ffi_arg_list, driver, name: str, cpy_arg ):
        cpy_arg.for_return = 2

        if self.return_type is float:
            raise NotImplementedError
        if self.return_type is int:
            raise NotImplementedError

        # method
        if callable( getattr( self.return_type, "call_arg_analysis_for", None ) ):
            return self.return_type.call_arg_analysis_for( jax_ffi_arg_list, driver, name, cpy_arg, *self.type_args, **self.type_kwargs )

        # else, make an instance (slow but may be ok)
        instance = self.return_type( *self.type_args, **self.type_kwargs )
        jax_ffi_arg_list.call_arg_analysis( driver, name, instance, cpy_arg )

    def fake_instance( self, driver ):
        """ make a fake instance to help find how to compile a function with a value that comes from a return """
        # special method ?
        if callable( getattr( self.return_type, "fake_instance", None ) ):
            return self.return_type.fake_instance( driver, *self.type_args, **self.type_kwargs )

        # call ctor
        return self.return_type( *self.type_args, **self.type_kwargs )

