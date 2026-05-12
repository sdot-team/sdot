from .IoCategory import IoCategory
from ..driver import driver

from typing import TYPE_CHECKING, Optional, Self
from weakref import ref

class CallArg:
    """
        Recursive analysis of an argument sent to driver.call( ... )

        CallArg is a pure virtual class

        Allows for
        - re-assembly and update of outputs of the ffi calls (which are basically flat lists of tensors)
        - generation of the assembly code for the C++ side

    """

    if TYPE_CHECKING:
        def get_template_args( self, template_args, names ): ...
        def backward_version( self, fai, driver, outputs, grads_of_the_outputs, parent, differentiable_inputs=None ) -> Self: ...
        def cpp_type_name( self, main_list ) -> str: ...
        def signature( self ) -> str: ...

        io_category    : IoCategory

        python_value   : Optional[ any ]
        python_class   : any #

        name_in_parent : Optional[ str ]
        parent         : Optional[ ref[ 'CallArg' ] ]

        ctor_kwargs    : Optional[ dict ]
        ctor_args      : Optional[ list ]


    @staticmethod
    def factory( call_args, parent, name_in_parent, python_class, python_value, io_category: IoCategory, ctor_args, ctor_kwargs ):
        # value method
        if python_value is not None and callable( getattr( python_value, "call_arg_factory", None ) ):
            return python_value.call_arg_factory( call_args, parent, name_in_parent, io_category, ctor_args, ctor_kwargs )

        # class method. Used for instance for Tensor() for which `value` is an array
        if callable( getattr( python_class, "call_arg_factory", None ) ):
            return python_class.call_arg_factory( call_args, parent, name_in_parent, python_value, io_category, ctor_args, ctor_kwargs )

        # arrays
        if driver.is_a_tensor( python_value ):
            from .CallArg_Tensor import CallArg_Tensor
            return CallArg_Tensor.factory( call_args, parent, name_in_parent, python_class, python_value, io_category, ctor_args, ctor_kwargs, comes_from_basic_array = True )

        # std objects
        if isinstance( python_value, ( int, float ) ):
            if io_category.want_output:
                raise NotImplementedError # A tensor returning item() ?
            from .CallArg_Parameter import CallArg_Parameter
            return CallArg_Parameter.factory( call_args, name_in_parent, python_value )

        if isinstance( python_value, ( list, tuple ) ):
            raise NotImplementedError

        if python_class.__name__ == "NoneType":
            raise RuntimeError( f"for { name_in_parent }" )

        # else, get attributes
        from .CallArg_Aggregate import CallArg_Aggregate
        return CallArg_Aggregate.factory( call_args, parent, name_in_parent, python_class, python_value, io_category, ctor_args, ctor_kwargs )

    def generate_structures( self, already_visited ):
        pass

    def get_includes( self, includes: set ):
        pass

    def get_axes( self, axes, ct_axes ):
        pass

    def beg_with_same_shape( self, name, s, lines ):
        lines.append( s + f"{ name }.with_same_shape( [&]( auto &{ name } ) {{" )
        return s + "  "

    def end_with_same_shape( self, name, s, lines ):
        s = s[ :-2 ]
        lines.append( s + "} );" )
        return s

    def assembled_code( self, beg_line: str ):
        raise NotImplementedError

    def assemble_return( self ):
        raise NotImplementedError

    def fully_qualified_name( self ) -> str:
        l = [ self.name_in_parent or "" ]
        p = self.parent
        while p is not None:
            if name := p().name_in_parent:
                l.append( name )
            p = p().parent
        return str.join( "_", l[ :-1 ] )

    def init_CallArgs_backward_version( self, res, parent ):
        res.io_category = IoCategory.pure_input()

        res.python_value = self.python_value
        res.python_class = self.python_class

        res.name_in_parent = self.name_in_parent
        res.parent = parent

        res.ctor_kwargs = None
        res.ctor_args = None
