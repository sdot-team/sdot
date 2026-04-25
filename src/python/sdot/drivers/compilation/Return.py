from .CallArg import CallArg


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

    def configure_call_arg( self, call_arg: CallArg, fai, driver ):
        return call_arg.configure_as_return( fai, driver, self.return_type, *self.type_args, **self.type_kwargs )

    # def fake_instance( self, driver ):
    #     """ make a fake instance to help find how to compile a function with a value that comes from a return """
    #     # special method ?
    #     if callable( getattr( self.return_type, "fake_instance", None ) ):
    #         return self.return_type.fake_instance( driver, *self.type_args, **self.type_kwargs )

    #     # call ctor
    #     return self.return_type( *self.type_args, **self.type_kwargs )

