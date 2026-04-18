from ...driver import driver

def cpp_class_name( obj ):
    # common types
    if isinstance( obj, float ):
        return driver.normalized_dtype

    if isinstance( obj, int ):
        return "SI"

    # method
    if callable( getattr( obj, "cpp_class_name", None ) ):
        return obj.cpp_class_name()

    # driver
    out = driver.cpp_class_name( obj )
    if out:
        return out

    # fail
    raise NotImplementedError( f"cpp_class_name: no mapping for { type( obj ) }" )
