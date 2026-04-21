from .collect_attributes import collect_attributes
from ...driver import driver

def as_jax_ffi_compatible_rets( obj ) -> list[ tuple[ any, str, str ] ]:
    """Decompose *arg* into the flat list of jax-compatible return objects
    that represent it on the binding boundary.

    return a list of tuples with
    - the compatibie argument value
    - the binding name (like Ret<xla::ffi::Buffer<FP32>>)
    - the argument name (like xla::ffi::ResultBuffer<FP32>)
    """

    # method to_nanobind_compatible_objects
    if callable( getattr( obj, "as_jax_ffi_compatible_rets", None ) ):
        return obj.as_jax_ffi_compatible_rets()

    # std objects
    if isinstance( obj, ( float, int ) ):
        raise RuntimeError( "In the Jax world, run-time arguments must be tensors. If you want Compile-time arguments, please use for objects like CtInt" )

    # driver
    conv = driver.as_jax_ffi_compatible_rets( obj )
    if conv:
        return conv

    if obj is None:
        return []

    # else, get attributes
    out = []
    for name, _ in collect_attributes( obj ):
        out += as_jax_ffi_compatible_rets( getattr( obj, name ) )
    return out
