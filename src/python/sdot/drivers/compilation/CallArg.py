from .collect_attributes import collect_attributes, Annotation
from typing import Self

class CallArg:
    """
        Recursive analysis of an argument sent to driver.call( ... )

        Allows for
        - re-assembly and update of outputs of the ffi calls (which are basically flat lists of tensors)
        - generation of the assembly code for the C++ side

    """

    attribute_name : str # attribute name
    updated_value  : callable # for update of mutable object. If None, `update` will use `sub_list` to go deeper
    python_value   : any #
    io_category    : int # 0 => pure input, 1 => mutable, 2 => return
    python_ctor    : callable # for reassembly of python objects. Called using an object that contains lists like differentiable_ffi_outputs, ...
    brace_ctor     : bool # true to use '{}' in generated code for instanciation
    base_code      : str # name of function or class for assembly of C++ objects (e.g. "Cell<TF,2>")
    signature      : str # name used to make the signature of the module (e.g. T0F64 for a tensor)
    sub_list       : list[ Self ] | None # arguments for python_ctor or base_code

    def __init__( self ):
        self.python_ctor = None
        self.brace_ctor = False
        self.sub_list = None
        self.updated_value = None

    @staticmethod
    def analysis_of_python_arg( python_value: any, attribute_name: str, fai, mutable: bool, driver ):
        """ recursively construct a CallArg """

        # common info
        res = CallArg()
        res.attribute_name = attribute_name
        res.sub_list = None

        # fill the remaining arguments
        res.configure( python_value, fai, mutable, driver )

        return res

    def configure( self, python_value: any, fai, mutable: bool, driver ):
        self.io_category = 1 if mutable else 0
        self.python_value = python_value

        # method
        if callable( getattr( python_value, "configure_call_arg", None ) ):
            return python_value.configure_call_arg( self, fai, driver )

        # arrays
        if driver.is_a_tensor( python_value ):
            return self.configure_as_input_tensor( python_value, mutable, fai, driver )

        # std objects
        if isinstance( python_value, float ):
            return self.configure_as_parameter( python_value, "FP64", mutable, fai, driver )

        if isinstance( python_value, int ):
            return self.configure_as_parameter( python_value, "PI64", mutable, fai, driver )

        if isinstance( python_value, ( list, tuple ) ):
            #     cpy_arg._python_ctor = type( value )
            #     cpy_arg.signature_type = "L"
            #     cpy_arg.code = "std::tie"
            #     for num, item in enumerate( value ):
            #         self.configure_call_arg( driver, f"{ name }_{ num }", item, cpy_arg.arg( str( num ) ) )
            #     return
            raise NotImplementedError

        if python_value is None:
            #     cpy_arg.signature_type = "N"
            #     cpy_arg.code = "{}"
            #     return
            raise NotImplementedError

        # else, get attributes
        self.configure_as_input_object_with_collect_attributes( python_value, mutable, fai, driver )

    def configure_as_parameter( self, python_value: any, cpp_type: str, mutable: bool, fai, driver ) -> Self:
        if mutable:
            raise NotImplementedError
        else:
            n = fai.add_parameter( python_value, cpp_type )

            self.python_ctor = None # not mutable -> we can directly use python_value
            self.signature = cpp_type
            self.base_code = n

    def configure_as_input_tensor( self, python_value: any, mutable: bool, fai, driver, valid = True ) -> Self:
        ndim = len( python_value.shape )
        self.signature = f"T{ ndim }{ driver.normalized_type_for( python_value.dtype ) }"

        if mutable:
            output_arg_name, output_validity_index = fai.add_output_tensor( driver, python_value.shape, python_value.dtype, valid )
            input_arg_name, input_validity_index = fai.add_input_tensor( python_value, driver, valid )
            self.base_code = f"tensor_view( CtInt<{ ndim }>(), { output_arg_name }, { input_arg_name }, ( validity_mask[ { output_validity_index // 64 } ] & { 1 << ( output_validity_index % 64 ) } ) && ( validity_mask[ { input_validity_index // 64 } ] & { 1 << ( input_validity_index % 64 ) } ) )"

            def updated_value( fai, differentiable_outputs, non_differentiable_outputs ):
                return CallArg.get_output_tensor( output_arg_name, fai, differentiable_outputs, non_differentiable_outputs )
            self.updated_value = updated_value
        else:
            arg_name, validity_index = fai.add_input_tensor( python_value, driver, valid )
            self.base_code = f"tensor_view( CtInt<{ ndim }>(), { arg_name }, validity_mask[ { validity_index // 64 } ] & { 1 << ( validity_index % 64 ) } )"

    @staticmethod
    def get_output_tensor( arg_name, fai, differentiable_outputs, non_differentiable_outputs ):
        if arg_name[ 0:2 ] == "no":
            return non_differentiable_outputs[ int( arg_name[ 2: ] ) ]
        if arg_name[ 0:2 ] == "do":
            return differentiable_outputs[ int( arg_name[ 2: ] ) ]
        raise NotImplementedError

    def configure_as_output_tensor( self, fai, driver, shape, dtype ):
        arg_name, validity_index = fai.add_output_tensor( driver, shape, dtype )
        dim = len( shape )

        self.base_code = f"tensor_view( CtInt<{ dim }>(), { arg_name }, validity_mask[ { validity_index // 64 } ] & { 1 << ( validity_index % 64 ) } )"
        self.signature = f"T{ dim }{ driver.normalized_type_for( dtype or driver.dtype ) }"

        def python_ctor( fai, differentiable_outputs, non_differentiable_outputs ):
            return CallArg.get_output_tensor( arg_name, fai, differentiable_outputs, non_differentiable_outputs )
        self.python_ctor = python_ctor

    def configure_as_return( self, fai, driver, return_type, *type_args, **type_kwargs ):
        """ complete self, with information for return """
        self.io_category = 2

        # method
        if callable( getattr( return_type, "configure_call_ret_for", None ) ):
            return_type.configure_call_ret_for( self, fai, driver, *type_args, **type_kwargs )
            return self

        # base types
        if return_type is float:
            # return self.configure_as_raw_return( fai, driver, return_type, *args, **kwargs )
            raise NotImplementedError

        if return_type is int:
            # return self.configure_as_raw_return( fai, driver, return_type, *args, **kwargs )
            raise NotImplementedError

        # -> use collect_attributes
        self.configure_as_output_object_with_collect_attributes( fai, driver, return_type, *type_args, **type_kwargs )

    def configure_as_input_object_with_collect_attributes( self, python_value, mutable, fai, driver ):
        if callable( getattr( python_value, "base_code", None ) ):
            self.base_code = python_value.base_code( driver )
        else:
            self.base_code = python_value.__class__.__name__

        if callable( getattr( python_value, "signature", None ) ):
            self.signature = python_value.signature( driver )
        else:
            self.signature = self.base_code

        self.brace_ctor = True

        self.sub_list = []
        for name, _ in collect_attributes( type( python_value ) ):
            self.sub_list.append( self.analysis_of_python_arg( getattr( python_value, name ), name, fai, mutable, driver ) )

    def configure_as_output_object_with_collect_attributes( self, fai, driver, return_type, *type_args, **type_kwargs ):
        if callable( getattr( return_type, "base_code_for", None ) ):
            self.base_code = return_type.base_code_for( driver, *type_args, **type_kwargs )
        else:
            self.base_code = return_type.__name__

        if callable( getattr( return_type, "signature_for", None ) ):
            self.signature = return_type.signature_for( driver, *type_args, **type_kwargs )
        else:
            self.signature = self.base_code

        def python_ctor( fai, differentiable_outputs, non_differentiable_outputs ):
            args = {}
            for item in self.sub_list:
                args[ item.attribute_name ] = item.python_ctor( fai, differentiable_outputs, non_differentiable_outputs )

            if callable( getattr( return_type, "__default_init__", None ) ):
                res = object.__new__( return_type )
                res.__default_init__( **args )
                return res

            raise return_type( **args )

        self.python_ctor = python_ctor

        self.brace_ctor = True

        self.sub_list = []
        for name, value in collect_attributes( return_type ):
            self.sub_list.append( self.return_child( fai, driver, name, value, *type_args, **type_kwargs ) )

    def return_child( self, fai, driver, attribute_name: str, return_type: any, *type_args, **type_kwargs ) -> Self:
        # base attributes
        res = CallArg()
        res.attribute_name = attribute_name
        res.python_value = None
        res.io_category = 2
        res.sub_list = None

        # analysis
        res.configure_as_return( fai, driver, return_type, *type_args, **type_kwargs )

        return res

    def configure_as_raw_return( self, fai, valid, driver, return_type, *args, **kwargs ):
        arg_name, validity_index = fai.add_raw_return( return_type, valid, driver, *args, **kwargs )

        def python_ctor( fai, differentiable_outputs, non_differentiable_outputs ):
            #if arg_name[ 0 ] == "n":
            raise NotImplementedError

        self.python_ctor = python_ctor #
        self.base_code = f"*{ arg_name }"
        self.signature = f"{ return_type }{ args }{ kwargs }"

    def assembled_code( self ) -> str:
        res = self.base_code
        if self.sub_list is not None:
            if self.brace_ctor:
                lst = []
                for a in self.sub_list:
                    if a.attribute_name.isdigit():
                        lst.append( a.assembled_code() )
                    else:
                        lst.append( f".{ a.attribute_name } = { a.assembled_code() }" )
                res += "{ " + str.join( ", ", lst ) + " }"
            else:
                res += "( " + str.join( ", ", [ a.assembled_code() for a in self.sub_list ] ) + " )"
        return res

    def construct( self, fai, differentiable_outputs, non_differentiable_outputs ):
        return self.python_ctor( fai, differentiable_outputs, non_differentiable_outputs )

    def update( self, fai, differentiable_outputs, non_differentiable_outputs ):
        for item in self.sub_list:
            if callable( item.updated_value ):
                setattr( self.python_value, item.attribute_name, item.updated_value( fai, differentiable_outputs, non_differentiable_outputs ) )
            else:
                item.update( fai, differentiable_outputs, non_differentiable_outputs )
