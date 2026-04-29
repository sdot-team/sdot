
class Workspace:
    """ Declares a worskpace.

    Usage:
        driver.call( "for_each_cell", includes,
            Worskpace( Cell, dim = 2, ... ),
            Return( Tensor, ... ),
        )

    Works like a Return excepted that call does not output the python object
    """
    def __init__( self, return_type, *args, **kwargs ):
        self.return_type = return_type
        self.type_kwargs = kwargs
        self.type_args   = args

    def configure_call_arg( self, call_arg, fai, mutable, driver ):
        return call_arg.configure_as_return( fai, driver, 3, self.return_type, *self.type_args, **self.type_kwargs )
