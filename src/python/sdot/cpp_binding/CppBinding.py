from ._types import to_standard_objects, diffentiable_tensors_of, write_back_diffentiable_tensors
from ._func  import get_func_for, get_forward_and_backward_for
from ..driver import driver
from .Return  import Return
from .Output  import Output


class CppBinding:
    """Entry point: ``cpp_binding[ "func_name", "header.h" ]( *args )``."""

    class Item:
        def __init__( self, func_name, includes: list[ str ] ):
            self.func_name = func_name
            self.includes = includes

        def __call__( self, *args ):
            output_tensors = []
            input_tensors = []
            output_args = []
            fargs = []
            res = None
            for arg in args:
                fargs += [ val for val, _ in to_standard_objects( arg ) ]

                if isinstance( arg, Return ):
                    res = arg.value

                if isinstance( arg, Output ):  # covers Return (which inherits Output)
                    output_tensors += diffentiable_tensors_of( arg )
                    output_args.append( arg )
                else:
                    input_tensors += diffentiable_tensors_of( arg )

            if driver.any_requires_grad( input_tensors ):
                fwd, bwd = get_forward_and_backward_for( self.func_name, args, self.includes, input_tensors, output_tensors )
                output = driver.forward( fwd, bwd, fargs, input_tensors, output_args, output_tensors )
            else:
                func = get_func_for( self.func_name, args, self.includes )
                output = func( *fargs )

            if output is None:
                return res
            return output


    def __call__( self, func_name, includes: str | list[ str ] = [] ):
        if isinstance( includes, str ):
            includes = [ includes ]
        return CppBinding.Item( func_name, includes )
