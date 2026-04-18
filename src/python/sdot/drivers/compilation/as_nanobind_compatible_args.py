from .collect_attributes import collect_attributes
from ...driver import driver

def as_nanobind_compatible_args( obj ) -> list[ tuple[ any, str ] ]:
    """Decompose *arg* into the flat list of nanobind-compatible Python objects
    that represent it on the binding boundary.

    """

    # method to_nanobind_compatible_objects
    if callable( getattr( obj, "as_nanobind_compatible_args", None ) ):
        return obj.as_nanobind_compatible_args()

    # std objects
    if isinstance( obj, float ):
        return [ ( obj, "TF" ) ]

    if isinstance( obj, int ):
        return [ ( obj, "SI" ) ]

    # driver
    conv = driver.as_nanobind_compatible_args( obj )
    if conv:
        return conv

    if obj is None:
        return []

    # else, get attributes
    out = []
    for name, _ in collect_attributes( obj ):
        ic( name )
        out += as_nanobind_compatible_args( getattr( obj, name ) )
    return out
