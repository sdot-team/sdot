# from .fake_instance import fake_instance
# from .CallArg import CallArg


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

    def configure_call_arg( self, call_arg, fai, mutable, driver ):
        return call_arg.configure_as_return( fai, driver, 2, self.return_type, *self.type_args, **self.type_kwargs )
