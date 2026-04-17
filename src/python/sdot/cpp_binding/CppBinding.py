from ._types import to_standard_objects
from ._func  import get_func_for
from .Return  import Return


class CppBinding:
    """Entry point: ``cpp_binding[ "func_name", "header.h" ]( *args )``."""

    class Item:
        def __init__( self, func_name, includes: list[ str ] ):
            self.func_name = func_name
            self.includes = includes

        def __call__( self, *args ):
            func = get_func_for( self.func_name, args, self.includes )

            fargs = []
            res = None
            for arg in args:
                fargs += to_standard_objects( arg )
                if isinstance( arg, Return ):
                    res = arg.value

            output = func( *fargs )

            if output is None:
                return res
            return output


    def __call__( self, func_name, includes: str | list[ str ] = [] ):
        if isinstance( includes, str ):
            includes = [ includes ]
        return CppBinding.Item( func_name, includes )
