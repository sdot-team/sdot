from .FfiArgInfo import FfiArgInfo

class Mutable:
    """Marks an existing object as read+write in driver.call.

    The object's arrays are passed as XLA FFI inputs (C++ reads them),
    and the same-shaped fresh arrays are produced as XLA FFI outputs
    (C++ writes the updated state). After the call, the object's
    array attributes are reassigned to the new outputs.
    """
    def __init__( self, value ):
        self.value = value

    def cpp_class_name( self, driver ):
        return FfiArgInfo.cpp_class_name( self.value, driver )

    def configure_call_arg( self, jax_ffi_arg_list, driver, name: str, cpy_arg ):
        cpy_arg.for_return = 1
        return jax_ffi_arg_list.configure_call_arg( driver, name, self.value, cpy_arg )
