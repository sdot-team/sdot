from ..compilation.IoCategory import IoCategory
from ..compilation.CallArg import CallArg

class Mutable:
    """Marks an existing object as read+write in driver.call.

    The object's arrays are passed as XLA FFI inputs (C++ reads them),
    and the same-shaped fresh arrays are produced as XLA FFI outputs
    (C++ writes the updated state). After the call, the object's
    array attributes are reassigned to the new outputs.
    """
    def __init__( self, value, **dyn_axis_capacities ):
        self.dyn_axis_capacities = dyn_axis_capacities
        self.value = value

    def configure_call_arg( self, call_arg: CallArg, fai, mutable, driver ):
        return call_arg.configure( self.value, fai, self.dyn_axis_capacities, driver )

    def call_arg_factory( self, call_args, parent, name_in_parent, io_category: IoCategory, ctor_args, ctor_kwargs ):
        new_io_category = IoCategory(
            want_return = False,
            want_output = True,
            has_input = True
        )
        return CallArg.factory( call_args, parent, name_in_parent, type( self.value ), self.value, io_category = new_io_category, ctor_args = [], ctor_kwargs = self.dyn_axis_capacities )
