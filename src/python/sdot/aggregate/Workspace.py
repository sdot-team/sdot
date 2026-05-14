from ..compilation.IoCategory import IoCategory
from ..compilation.CallArg import CallArg

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

    def call_arg_factory( self, call_args, parent, name_in_parent, io_category: IoCategory, ctor_args, ctor_kwargs ):
        new_io_category = IoCategory(
            want_return = False,
            want_output = True,
            has_input = False
        )
        return CallArg.factory( call_args, parent, name_in_parent, self.return_type, None, io_category = new_io_category, ctor_args = self.type_args, ctor_kwargs = self.type_kwargs )

    def coerce( self, value ):
        if value is None:
            return None

        if coerce := getattr( self.return_type, "coerce", None ):
            return coerce( value )

        if not isinstance( value, self.return_type ):
            return self.return_type( value )
