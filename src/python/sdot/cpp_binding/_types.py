"""Type helpers: map Python objects to C++ class names and nanobind argument lists."""
from ..driver import driver
from .Output import Output
from .Return import Return


def cpp_class_name( obj ) -> str:
    # common types
    if isinstance( obj, driver.array_type ):
        if driver.is_int_dtype( obj.dtype ):
            return "MI"
        return "MF"

    if isinstance( obj, float ):
        return driver.normalized_dtype

    if isinstance( obj, int ):
        return "SI"

    # method
    if callable( getattr( obj, "cpp_class_name", None ) ):
        return obj.cpp_class_name()

    # fail
    raise NotImplementedError( f"cpp_class_name_for: no mapping for { type( obj ) }" )


def to_standard_objects( obj ) -> list[ tuple[ any, str ] ]:
    """Decompose *arg* into the flat list of nanobind-compatible Python objects
    that represent it on the binding boundary.

    """

    # method to_standard_objects
    if callable( getattr( obj, "to_standard_objects", None ) ):
        return obj.to_standard_objects()

    # std objects
    if isinstance( obj, int ):
        return [ ( obj, "SI" ) ]

    if isinstance( obj, float ):
        return [ ( obj, "TF" ) ]

    if isinstance( obj, driver.array_type ):
        if driver.is_int_dtype( obj.dtype ):
            return [ ( obj, "MI" ) ]
        return [ ( obj, "MF" ) ]

    if obj is None:
        return []

    # else, get attributes
    out = []
    for name, _ in _collect_attributes( obj ):
        out += to_standard_objects( getattr( obj, name ) )
    return out


def diffentiable_tensors_of( obj ) -> list:
    """Flatten *obj* into a list of tensors (driver array_type instances only)."""
    if callable( getattr( obj, "diffentiable_tensors", None ) ):
        return obj.diffentiable_tensors()

    if isinstance( obj, driver.array_type ):
        if driver.is_int_dtype( obj.dtype ):
            return []
        return [ obj ]

    if isinstance( obj, ( int, float ) ) or obj is None:
        return []

    # else, get attributes
    out = []
    for name, _ in _collect_attributes( obj ):
        out += diffentiable_tensors_of( getattr( obj, name ) )
    return out


def from_standard_objects( obj, arg_names ):
    # method to_standard_objects
    if callable( getattr( obj, "from_standard_objects", None ) ):
        return obj.from_standard_objects( obj, arg_names )

    # std objects
    if isinstance( obj, ( int, float ) ):
        return arg_names.pop( 0 )

    if isinstance( obj, driver.array_type ):
        return f"tensor_view_{ obj.ndim }( { arg_names.pop( 0 ) } )"

    if isinstance( obj, ( Output, Return ) ):
        return from_standard_objects( obj.value, arg_names )

    # else, get attributes
    largs = []
    for name, _ in _collect_attributes( obj ):
        largs.append( from_standard_objects( getattr( obj, name ), arg_names ) )
    return f"{ cpp_class_name( obj ) }( { str.join( ", ", largs ) } )"


def _collect_attributes( obj ):
    res = []
    name_indices = {} # if attribute appears in a subclass and in a parent class, we want to take
    for klass in reversed( type( obj ).__mro__ ):
        for name, value in vars( klass ).items():
            if isinstance( value, ( classmethod, staticmethod ) ) or callable( value ):
                continue
            if isinstance( value, property ): #  and value.fset is None
                continue
            if name.startswith( '_' ):
                continue

            # register value
            if name in name_indices:
                res[ name_indices[ name ] ] = ( name, value )
            else:
                name_indices[ name ] = len( res )
                res.append( ( name, value ) )

    return res

