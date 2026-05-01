from ..driver import driver

from .IoCategory import IoCategory
from .CallArg import CallArg

class CallArg_Parameter( CallArg ):
    """
    """

    num_in_parameters : int
    python_value : any
    io_category : IoCategory
    name : str

    @staticmethod
    def factory( call_args, name, python_value ):
        """  """
        res = CallArg_Parameter()
        res.num_in_parameters = call_args.add_parameter( res )
        res.python_value = python_value
        res.io_category = IoCategory( want_return = False, want_output = False, has_input = True )
        res.name = name
        return res

    def signature( self ) -> str:
        return f"P{ driver.normalized_type_for( type( self.python_value ) ) }"

    def get_template_args( self, template_args, names ):
        pass

    def cpp_type_name( self, main_list = None ):
        return driver.normalized_type_for( type( self.python_value ) )

    def get_axes( self, axes: dict, ct_axes: dict[ int ] ):
        pass

    def get_input_arg_decl( self, declarations: list ):
        declarations.append( f"{ self.cpp_type_name() } p{ self.num_in_parameters }" )

    def get_input_bind_chain( self, bind_chain: list ):
        bind_chain.append( driver.ffi_parameter_bind_code( type( self.python_value ), self.name ) )

    def ffi_name( self ):
        return f"p{ self.num_in_parameters }"

    def assembled_code( self, beg_line ):
        return self.ffi_name()
