from .CallArg import CallArg

class Mutable:
    """Marks an existing object as read+write in driver.call.

    The object's arrays are passed as XLA FFI inputs (C++ reads them),
    and the same-shaped fresh arrays are produced as XLA FFI outputs
    (C++ writes the updated state). After the call, the object's
    array attributes are reassigned to the new outputs.
    """
    def __init__( self, value ):
        self.value = value

    def configure_call_arg( self, call_arg: CallArg, fai, driver ):
        return call_arg.configure( self.value, fai, True, driver )

