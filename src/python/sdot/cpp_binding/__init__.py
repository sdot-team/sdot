"""sdot.cpp_binding — generic C++ binding machinery.

Example::

    cpp_binding( "test", "my_include.h" )( 10 )

Flow:
* from generic python object
* conversion to "standard" ones (int, array, …)
* call the binding with "standard" python objects as arguments
* conversion (in C++) to C++ objects with the same name
* call a C++ function with the same name
* conversion of the result to "standard" python outputs

Env variables
* SDOT_FORCE_BUILD: "1" to force build
* SDOT_CACHE_DIR: where to place the generated sources and dylibs
"""

from ._types import to_standard_objects
from ._func  import get_func_for


class CppBinding:
    """Entry point: ``cpp_binding[ "func_name", "header.h" ]( *args )``."""

    class Item:
        def __init__( self, func_name, includes: list[ str ] ):
            self.func_name = func_name
            self.includes = includes

        def __call__( self, *args ):
            func = get_func_for( self.func_name, args, self.includes )

            fargs = []
            for arg in args:
                fargs += to_standard_objects( arg )

            return func( *fargs )


    def __call__( self, func_name, includes: str | list[ str ] = [] ):
        if isinstance( includes, str ):
            includes = [ includes ]
        return CppBinding.Item( func_name, includes )


cpp_binding = CppBinding()
