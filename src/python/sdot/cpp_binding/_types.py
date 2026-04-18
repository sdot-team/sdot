"""Type helpers: map Python objects to C++ class names and nanobind argument lists."""
from ..driver import driver
from sdot.drivers.compilation.Output import Output
from sdot.drivers.compilation.Return import Return
from ..object_with_tensors.UndefinedTensor import UndefinedTensor


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


def to_nanobind_compatible_objects( obj ) -> list[ tuple[ any, str ] ]:
    """Decompose *arg* into the flat list of nanobind-compatible Python objects
    that represent it on the binding boundary.

    """

    # method to_nanobind_compatible_objects
    if callable( getattr( obj, "to_nanobind_compatible_objects", None ) ):
        return obj.to_nanobind_compatible_objects()

    # std objects
    if isinstance( obj, int ):
        return [ ( obj, "SI" ) ]

    if isinstance( obj, float ):
        return [ ( obj, "TF" ) ]

    # driver
    conv = driver.to_nanobind_compatible_objects( obj )
    if conv:
        if driver.is_int_dtype( obj.dtype ):
            return [ ( obj, "MI" ) ]
        return [ ( obj, "MF" ) ]

    if obj is None:
        return []

    # else, get attributes
    out = []
    for name, _ in _collect_attributes( obj ):
        out += to_nanobind_compatible_objects( getattr( obj, name ) )
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


def write_back_diffentiable_tensors( obj, tensors_iter ) -> None:
    """Set float-tensor attributes of *obj* from *tensors_iter*, mirroring diffentiable_tensors_of."""
    if callable( getattr( obj, "write_back_diffentiable_tensors", None ) ):
        obj.write_back_diffentiable_tensors( tensors_iter )
        return

    if isinstance( obj, driver.array_type ):
        if not driver.is_int_dtype( obj.dtype ):
            next( tensors_iter, None )  # consume; standalone tensors can't be re-assigned here
        return

    if isinstance( obj, ( int, float ) ) or obj is None:
        return

    for name, _ in _collect_attributes( obj ):
        attr = getattr( obj, name )
        if isinstance( attr, driver.array_type ):
            if not driver.is_int_dtype( attr.dtype ):
                setattr( obj, name, next( tensors_iter ) )
        elif isinstance( attr, UndefinedTensor ):
            if not driver.is_int_dtype( attr.dtype ):
                setattr( obj, name, next( tensors_iter ) )
        else:
            write_back_diffentiable_tensors( attr, tensors_iter )


def cpp_assembly_from_nanobind_compatible_objects( obj, arg_names, use_view = False ):
    # method to_nanobind_compatible_objects
    if callable( getattr( obj, "cpp_assembly_from_nanobind_compatible_objects", None ) ):
        return obj.cpp_assembly_from_nanobind_compatible_objects( obj, arg_names, use_view = use_view )

    # std objects
    if isinstance( obj, ( int, float ) ):
        return arg_names.pop( 0 )

    if isinstance( obj, driver.array_type ):
        name = arg_names.pop( 0 )
        if use_view:
            return name
        return f"tensor_view_{ obj.ndim }( { name } )"

    if isinstance( obj, ( Output, Return ) ):
        return cpp_assembly_from_nanobind_compatible_objects( obj.value, arg_names, use_view = use_view )

    # else, get attributes
    largs = []
    for name, _ in _collect_attributes( obj ):
        largs.append( cpp_assembly_from_nanobind_compatible_objects( getattr( obj, name ), arg_names, use_view = use_view ) )
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

