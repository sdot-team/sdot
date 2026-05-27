from ..drivers.driver import driver

from .IoCategory import IoCategory
from .CallArg import CallArg

from typing import Optional


class CallArg_Parameter( CallArg ):
    """
    A scalar argument ( int / float / ... ) passed to the FFI call as an attribute,
    not as a tensor. It has no children and no axis variables.
    """

    num_in_parameters : int
    name : str

    def __init__(
        self,
        call_args    : any,
        name         : str,
        python_value : Optional[ any ]
    ):
        super().__init__(
            name_in_parent = None,
            parent         = None,
            python_class   = None,
            python_value   = python_value,
            io_category    = IoCategory( want_return = False, want_output = False, has_input = True ),
            ctor_args      = None,
            ctor_kwargs    = None
        )
        self.name = name

        # analysis: register this scalar in the flat parameter list
        self.num_in_parameters = call_args.add_parameter( self )

    def signature( self ) -> str:
        """(analysis, override) Contribution to the per-binding signature: the scalar type."""
        return f"P{ driver.normalized_type_for( type( self.python_value ) ) }"

    def get_template_args( self, template_args, names ):
        """(code generation, override) A scalar carries no template argument."""
        pass

    def cpp_type_name( self, main_list = None ):
        """(code generation, override) C++ type of the scalar."""
        return driver.normalized_type_for( type( self.python_value ) )

    def get_input_arg_decl( self, declarations: list ):
        """(code generation) Append the handler argument declaration for this scalar."""
        declarations.append( f"{ self.cpp_type_name() } p{ self.num_in_parameters }" )

    def get_input_bind_chain( self, bind_chain: list ):
        """(code generation) Append the FFI binding that reads this scalar as an attribute."""
        bind_chain.append( driver.ffi_parameter_bind_code( type( self.python_value ), self.name ) )

    def ffi_name( self ):
        """(code generation) Name of the scalar in the generated handler."""
        return f"p{ self.num_in_parameters }"

    def assembled_code( self, beg_line ):
        """(code generation, override) The scalar is used directly by its name."""
        return self.ffi_name()

    def beg_with_same_shape( self, name, s, lines ):
        """(code generation, override) No-op: a scalar has no shape."""
        return s

    def end_with_same_shape( self, name, s, lines ):
        """(code generation, override) No-op: a scalar has no shape."""
        return s

    def backward_version( self, call_args, driver, outputs, grads_of_the_outputs, parent, differentiable_inputs = None ):
        """(analysis, override) A scalar is unchanged in the backward pass."""
        return CallArg_Parameter( call_args, self.name, self.python_value )
