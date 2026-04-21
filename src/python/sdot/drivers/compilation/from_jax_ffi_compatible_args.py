from .as_jax_ffi_compatible_args import as_jax_ffi_compatible_args
from .cpp_class_name import cpp_class_name
from ...driver import driver

def from_jax_ffi_compatible_args( obj, flat_arg_iterator ):
    """string to recompose an obj instance from a set of jax_ffi compatible object

    It's the opposite to as_jax_compatible_args
    """

    # method to_nanobind_compatible_objects
    if callable( getattr( obj, "from_jax_ffi_compatible_args", None ) ):
        return obj.from_jax_ffi_compatible_args( flat_arg_iterator )

    # std objects
    if isinstance( obj, ( float, int ) ):
        raise RuntimeError( "In the Jax world, run-time arguments must be tensors. If you want Compile-time arguments, please use for objects like CtInt" )

    # driver
    conv = driver.from_jax_ffi_compatible_args( obj, flat_arg_iterator )
    if conv:
        return conv

    # if obj is None:
    #     return []

    # else, get attributes
    args = []
    for _ in as_jax_ffi_compatible_args( obj ):
        args.append( next( flat_arg_iterator ) )
    return f"{ cpp_class_name( obj ) }{{ { str.join( ", ", args ) } }}"
