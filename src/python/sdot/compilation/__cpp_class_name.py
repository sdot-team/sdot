from ...driver import driver

def cpp_class_name( obj ):
    # common types
    if isinstance( obj, float ):
        return driver.normalized_dtype

    if isinstance( obj, int ):
        return "SI"

    if isinstance( obj, ( list, tuple ) ):
        return f"std::tuple<{ str.join( ", ", [ cpp_class_name( v ) for v in obj ] ) }>"

    # method
    if callable( getattr( obj, "cpp_class_name", None ) ):
        return obj.cpp_class_name()

    # driver
    out = driver.cpp_class_name( obj )
    if out:
        return out

    # fail
    raise NotImplementedError( f"cpp_class_name: no mapping for { type( obj ) }" )
