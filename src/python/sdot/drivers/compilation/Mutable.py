from .cpp_class_name import cpp_class_name

class Mutable:
    """Marks an existing object as read+write in driver.call.

    The object's arrays are passed as XLA FFI inputs (C++ reads them),
    and the same-shaped fresh arrays are produced as XLA FFI outputs
    (C++ writes the updated state). After the call, the object's
    array attributes are reassigned to the new outputs.
    """
    def __init__( self, value ):
        self.value = value

    def cpp_class_name( self ):
        return cpp_class_name( self.value )

    def as_jax_ffi_compatible_args( self, driver ):
        return driver.as_jax_ffi_compatible_args( self.value )

    def as_jax_ffi_compatible_rets( self, driver ):
        return driver.as_jax_ffi_compatible_rets( self.value )

    def cpp_assembly_from_jax_ffi_compatible_args( self, driver, arg_iter ):
        return driver.cpp_assembly_from_jax_ffi_compatible_args( self.value, arg_iter )

    def python_assembly_from_jax_ffi_compatible_args( self, driver, arg_iter ):
        raise RuntimeError( "Mutable are not meant to assemble" )

    def python_update_from_jax_ffi_compatible_args( self, driver, arg_iter ):
        return driver.python_update_from_jax_ffi_compatible_args( self.value, arg_iter )

