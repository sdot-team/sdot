from ..compilation.IoCategory import IoCategory
from ..compilation.CallArg import CallArg

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

    def call_arg_factory( self, call_args, parent, io_category: IoCategory ):
        new_io_category = IoCategory(
            want_return = True,
            want_output = True,
            want_input = False
        )
        return CallArg.factory( call_args, parent, self.return_type, None, io_category = new_io_category )
