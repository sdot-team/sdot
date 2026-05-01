from .IoCategory import IoCategory
from ..driver import driver


from typing import TYPE_CHECKING


class CallArg:
    """
        Recursive analysis of an argument sent to driver.call( ... )

        CallArg is a pure virtual class

        Allows for
        - re-assembly and update of outputs of the ffi calls (which are basically flat lists of tensors)
        - generation of the assembly code for the C++ side

    """

    if TYPE_CHECKING:
        def cpp_type_name( self, main_list ) -> str: ...
        def signature( self ) -> str: ...

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
            return CallArg_Tensor.factory( call_args, parent, name_in_parent, python_class, python_value, io_category, ctor_args, ctor_kwargs )

        # std objects
        if isinstance( python_value, float ):
            # return self.configure_as_parameter( python_value, "FP64", mutable, fai, driver )
            raise NotADirectoryError

        if isinstance( python_value, int ):
            #     return self.configure_as_parameter( python_value, "SI64", mutable, fai, driver )
            raise NotADirectoryError

        if isinstance( python_value, ( list, tuple ) ):
            raise NotImplementedError

        if python_value is None:
            raise NotImplementedError

        # else, get attributes
        from .CallArg_Aggregate import CallArg_Aggregate
        return CallArg_Aggregate.factory( call_args, parent, name_in_parent, python_class, python_value, io_category, ctor_args, ctor_kwargs )

    def generate_structures( self ):
        pass

    def get_includes( self, includes: set ):
        pass

