"""Type helpers: map Python objects to C++ class names and nanobind argument lists."""
from ..driver import driver

def cpp_class_name_for( arg ) -> str:
    # common types
    if isinstance( arg, int ):
        return "SI"

    if isinstance( arg, float ):
        return driver.normalized_dtype

    if isinstance( arg, driver.array_type ):
        if arg.dtype == driver.int_type:
            return "MI"
        else:
            return "MF"

    if hasattr( arg, "cpp_class_name" ):
        return arg.cpp_class_name()

    raise NotImplementedError( f"cpp_class_name_for: no mapping for { type(arg) }" )


def to_standard_objects( obj ) -> list:
    """Decompose *arg* into the flat list of nanobind-compatible Python objects
    that represent it on the binding boundary."""

    # common types
    if isinstance( obj, ( int, float, driver.array_type ) ):
        return [ obj ]

    # method to_standard_objects
    if hasattr( obj, "to_standard_objects" ):
        return obj.to_standard_objects()

    # else, get attributes
    out = []
    for name, _ in _collect_attributes( obj ):
        out += to_standard_objects( getattr( obj, name ) )
    return out


def from_standard_objects( obj, arg_names ):
    if isinstance( obj, ( int, float ) ):
        return arg_names.pop( 0 )

    if isinstance( obj, ( driver.array_type ) ):
        return f"tensor_view_{ obj.ndim }( { arg_names.pop( 0 ) } )"

    largs = []
    for name, _ in _collect_attributes( obj ):
        largs.append( from_standard_objects( getattr( obj, name ), arg_names ) )
    return f"{ cpp_class_name_for( obj ) }( { str.join( ", ", largs )  } )"


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
