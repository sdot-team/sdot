
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
        self.type_args   = args
        self.type_kwargs = kwargs
