"""Type helpers: map Python objects to C++ class names and nanobind argument lists."""
from ..object_with_tensors.ListOfTensorFields import ListOfTensorFields
from ..object_with_tensors.TensorField import TensorField
from ..driver import driver
from .Output import Output
from .Return import Return

def cpp_class_name_for( obj ) -> str:
    # common types
    if isinstance( obj, int ):
        return "SI"

    if isinstance( obj, float ):
        return driver.normalized_dtype

    if isinstance( obj, ( driver.array_type, TensorField ) ):
        if driver.is_int_dtype( obj.dtype ):
            return "MI"
        return "MF"

    if isinstance( obj, ( Output, Return ) ):
        return cpp_class_name_for( obj.value )

    if hasattr( obj, "cpp_class_name" ):
        return obj.cpp_class_name()

    raise NotImplementedError( f"cpp_class_name_for: no mapping for { type(obj) }" )


def to_standard_objects( obj, attr = None ) -> list[ tuple[ any, str ] ]:
    """Decompose *arg* into the flat list of nanobind-compatible Python objects
    that represent it on the binding boundary.

    attr can be used to get class TensorField, ... which have get __get__ method
    """
    if attr is None:
        attr = obj

    if isinstance( attr, TensorField ):
        if attr.dtype == int:
            return [ ( obj, "MI" ) ]
        if attr.dtype is not None:
            raise NotImplementedError( f"to_standard_objects with dtype = { attr.dtype }" )
        return [ ( obj, "MF" ) ]

    if isinstance( attr, ListOfTensorFields ):
        raise NotImplementedError

    if isinstance( obj, int ):
        return [ ( obj, "SI" ) ]

    if isinstance( obj, float ):
        return [ ( obj, "TF" ) ]

    if isinstance( obj, driver.array_type ):
        if driver.is_int_dtype( obj.dtype ):
            return [ ( obj, "MI" ) ]
        return [ ( obj, "MF" ) ]

    # method to_standard_objects
    if hasattr( obj, "to_standard_objects" ):
        return obj.to_standard_objects()

    if isinstance( obj, ( Output, Return ) ):
        return to_standard_objects( obj.value )

    if obj is None:
        return []

    # else, get attributes
    out = []
    for name, attr in _collect_attributes( obj ):
        out += to_standard_objects( getattr( obj, name ), attr )
    return out


def from_standard_objects( obj, arg_names, attr = None ):
    if attr is None:
        attr = obj

    if isinstance( attr, TensorField ):
        return f"tensor_view_{ attr.ndim }( { arg_names.pop( 0 ) } )"

    if isinstance( attr, ListOfTensorFields ):
        raise NotImplementedError

    if isinstance( obj, ( int, float ) ):
        return arg_names.pop( 0 )

    if isinstance( obj, driver.array_type ):
        return f"tensor_view_{ obj.ndim }( { arg_names.pop( 0 ) } )"

    if isinstance( obj, ( Output, Return ) ):
        return from_standard_objects( obj.value, arg_names )

    largs = []
    for name, attr in _collect_attributes( obj ):
        largs.append( from_standard_objects( getattr( obj, name ), arg_names, attr ) )
    return f"{ cpp_class_name_for( obj ) }( { str.join( ", ", largs ) } )"


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

