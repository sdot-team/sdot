# from .collect_attributes import collect_attributes
# from ...driver import driver

# def as_jax_ffi_compatible_specs( obj ) -> list:
#     """
#     """

#     # method to_nanobind_compatible_objects

#     # std objects
#     if isinstance( obj, ( float, int ) ):
#         raise RuntimeError( "In the Jax world, run-time arguments must be tensors. If you want Compile-time arguments, please use for objects like CtInt" )

#     # driver
#     conv = driver.as_jax_ffi_compatible_specs( obj )
#     if conv:
#         return conv

#     if obj is None:
#         return []
