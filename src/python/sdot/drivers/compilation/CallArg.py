from .collect_attributes import collect_attributes_inst, collect_attributes #, Annotation
from ...UndefinedTensor import UndefinedTensor
from ...util import index
from ...Dyn import Dyn

from .FfiOutput import FfiOutput
from .FfiInput import FfiInput
from .Return import Return
from .Tensor import Tensor

from typing import Self
import weakref


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
    ffi_output     : FfiOutput | None #
    ffi_input      : FfiInput | None #
    base_code      : str # name of function or class for assembly of C++ objects (e.g. "Cell<TF,2>")
    signature      : str # name used to make the signature of the module (e.g. T0F64 for a tensor)
    sub_list       : list[ Self ] | None # arguments for python_ctor or base_code
    parent         : Self | None #

    def __init__( self ):
        self.updated_value = None
        self.python_ctor = None
        self.brace_ctor = False
        self.ffi_output = None
        self.ffi_input = None
        self.sub_list = None
        self.parent = None

    @staticmethod
    def analysis_of_python_arg( python_value: any, attribute_name: str, fai, mutable: dict[ int ] | bool, driver, parent = None, configure = True ):
        """ recursively construct a CallArg """

        # common info
        res = CallArg()
        res.attribute_name = attribute_name
        res.sub_list = None

        if parent is not None:
            res.parent = weakref.ref( parent )

        # fill the remaining arguments
        if configure:
            res.configure( python_value, fai, mutable, driver )

        return res

    @staticmethod
    def second_pass_analysis( call_args: list[ 'CallArg' ], fai, driver ):
        # dynamic output axes
        #   replace __capacity__
        #   update python_ctor and updated_value
        for call_arg in call_args:
            if call_arg.ffi_output and call_arg.ffi_output.represents_a_dynamic_axis:
                axis_name = call_arg.attribute_name + "_capacity"
                capacity_list = []
                for gra_llac in call_args:
                    if gra_llac.ffi_output:
                        num_axis = index( gra_llac.ffi_output.axis_names, axis_name )
                        if num_axis >= 0:
                            # __capacity__ inputs
                            capacity_list.append( gra_llac.ffi_output.arg_name )
                            capacity_list.append( f"u64_input[ { gra_llac.ffi_output.validity_index // 64 } ] & { 1 << ( gra_llac.ffi_output.validity_index % 64 ) }" )
                            capacity_list.append( str( num_axis ) )

                            # resize outputs with this axis (is one_value_for_each was [])
                            if len( call_arg.ffi_output.axis_names ) == 0:
                                if gra_llac.python_ctor is not None:
                                    def new_python_ctor( fai, outputs, n = num_axis, o = gra_llac.python_ctor, i = call_arg.ffi_output.num_in_sub_list ):
                                        res = o( fai, outputs )
                                        if i >= len( outputs ):
                                            return res
                                        slices = [ slice( None ) for _ in range( res.ndim ) ]
                                        slices[ n ] = slice( None, outputs[ i ].item() )
                                        return res[ tuple( slices ) ]
                                    gra_llac.python_ctor = new_python_ctor

                                if gra_llac.updated_value is not None:
                                    def new_updated_value( fai, outputs, n = num_axis, o = gra_llac.updated_value, i = call_arg.ffi_output.num_in_sub_list ):
                                        res = o( fai, outputs )
                                        if i >= len( outputs ):
                                            return res
                                        slices = [ slice( None ) for _ in range( res.ndim ) ]
                                        slices[ n ] = slice( None, outputs[ i ].item() )
                                        return res[ tuple( slices ) ]
                                    gra_llac.updated_value = new_updated_value

                capacity = f"first_valid_dimension( u64_input, { str.join( ", ", capacity_list ) } )"
                while "__capacity__" in call_arg.base_code:
                    call_arg.base_code = call_arg.base_code.replace( "__capacity__", capacity )

            # recursion
            if call_arg.sub_list:
                CallArg.second_pass_analysis( call_arg.sub_list, fai, driver )

    def configure( self, python_value: any, fai, mutable: dict[ int ] | None, driver ):
        assert not isinstance( mutable, bool )

        self.io_category = 1 if mutable is not None else 0
        self.python_value = python_value

        # method
        if callable( getattr( python_value, "configure_call_arg", None ) ):
            return python_value.configure_call_arg( self, fai, mutable, driver )

        # arrays
        if driver.is_a_tensor( python_value ):
            return self.configure_as_input_tensor( python_value, mutable, fai, driver, axis_names = [ "" ] * python_value.ndim )

        # std objects
        if isinstance( python_value, float ):
            return self.configure_as_parameter( python_value, "FP64", mutable, fai, driver )

        if isinstance( python_value, int ):
            return self.configure_as_parameter( python_value, "SI64", mutable, fai, driver )

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

    def configure_as_parameter( self, python_value: any, cpp_type: str, mutable: dict[ int ] | None, fai, driver ) -> Self:
        if mutable is not None:
            raise NotImplementedError
        else:
            ffi_parameter = fai.add_parameter( python_value, cpp_type )
            self.base_code = ffi_parameter.arg_name
            self.signature = cpp_type

    @staticmethod
    def get_output_tensor( ffi_output: FfiOutput, fai, driver, outputs, fallback_shape, fallback_dtype ):
        index = int( ffi_output.num_in_sub_list )
        if index < len( outputs ):
            return outputs[ index ]
        return driver.empty( [ 0 for _ in fallback_shape ], dtype = fallback_dtype )

    def configure_as_input_tensor( self, python_value: any, mutable: dict[ int ] | None, fai, driver, axis_names, valid = True, represents_a_dynamic_axis = False ) -> Self:
        if driver.is_zero_tensor( python_value ):
            return self.configure( UndefinedTensor( python_value.shape, python_value.dtype ), fai, mutable, driver )

        ndim = len( python_value.shape )
        self.signature = f"T{ ndim }{ driver.normalized_type_for( python_value.dtype ) }"

        if mutable is not None:
            shape = list( python_value.shape )
            assert( len( shape ) == len( axis_names ) )
            for n, axis_name in enumerate( axis_names ):
                if not axis_name.isidentifier():
                    raise NotImplementedError
                if axis_name in mutable:
                    shape[ n ] = mutable[ axis_name ]

            ffi_output = fai.add_output_tensor( driver, shape, python_value.dtype, axis_names = axis_names, valid = valid, represents_a_dynamic_axis = represents_a_dynamic_axis )
            ffi_input = fai.add_input_tensor( python_value, driver, valid = valid, represents_a_dynamic_axis = represents_a_dynamic_axis )
            self.ffi_output = ffi_output
            self.ffi_input = ffi_input

            self.base_code = f"tensor_view( CtInt<{ ndim }>(), { ffi_output.arg_name }, { ffi_input.arg_name }, ( u64_input[ { ffi_output.validity_index // 64 } ] & { 1 << ( ffi_output.validity_index % 64 ) } ) && ( u64_input[ { ffi_input.validity_index // 64 } ] & { 1 << ( ffi_input.validity_index % 64 ) } ) )"
            if represents_a_dynamic_axis:
                self.base_code = f"DynamicAxis<{ ndim },Arch>( { self.base_code }, __capacity__ )"

            def updated_value( fai, outputs ):
                return CallArg.get_output_tensor( ffi_output, fai, driver, outputs, fallback_shape = python_value.shape, fallback_dtype = python_value.dtype )
            self.updated_value = updated_value
        else:
            ffi_input = fai.add_input_tensor( python_value, driver, valid = valid, represents_a_dynamic_axis = represents_a_dynamic_axis )
            self.ffi_input = ffi_input

            self.base_code = f"tensor_view( CtInt<{ ndim }>(), { ffi_input.arg_name }, u64_input[ { ffi_input.validity_index // 64 } ] & { 1 << ( ffi_input.validity_index % 64 ) } )"
            if represents_a_dynamic_axis:
                self.base_code = f"DynamicAxis<{ ndim },Arch>( { self.base_code }, 0 )"

    def configure_as_output_tensor( self, fai, driver, _shape, dtype, axis_names, for_single_item = False, resize_dyn_axes = True, list_of_dynamic_axes = None, represents_a_dynamic_axis = False ):
        shape = []
        for s in _shape:
            if isinstance( s, Dyn ):
                if s.capacity is None:
                    raise RuntimeError( "for output tensors, capacity must be specified for dynamic axes (in this case for '{ s.name }')" )
                shape.append( s.capacity )
            else:
                shape.append( s )

        # create and register the ffi_output
        ffi_output = fai.add_output_tensor( driver, shape, dtype, axis_names, represents_a_dynamic_axis = represents_a_dynamic_axis )
        self.ffi_output = ffi_output

        self.base_code = f"tensor_view( CtInt<{ len( shape ) }>(), { ffi_output.arg_name }, u64_input[ { ffi_output.validity_index // 64 } ] & { 1 << ( ffi_output.validity_index % 64 ) } )"
        if represents_a_dynamic_axis:
            self.base_code = f"DynamicAxis<{ len( shape ) },Arch>( { self.base_code }, __capacity__ )"

        self.signature = f"T{ len( shape ) }{ driver.normalized_type_for( dtype or driver.dtype ) }"

        def _python_ctor( fai, outputs ):
            # get tensor
            output = CallArg.get_output_tensor( ffi_output, fai, driver, outputs, fallback_shape = shape, fallback_dtype = dtype )

            # output format
            if for_single_item:
                return output.item()
            return output

        self.python_ctor = _python_ctor

    def configure_as_return( self, fai, driver, io_category, return_type, *type_args, **type_kwargs ):
        """ complete self, with information for return """
        self.io_category = io_category

        # method
        if callable( getattr( return_type, "configure_call_ret_for", None ) ):
            return_type.configure_call_ret_for( self, fai, driver, *type_args, **type_kwargs )
            return self

        # base types
        if return_type is float:
            return self.configure_as_output_tensor( fai, driver, [], float, [], for_single_item = True )

        if return_type is int:
            return self.configure_as_output_tensor( fai, driver, [], int, [], for_single_item = True )

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
        for name, inst in collect_attributes_inst( python_value, use_annotations = True ):
            analysis_of_python_arg = getattr( inst, "analysis_of_python_arg", None )
            if analysis_of_python_arg:
                sc = analysis_of_python_arg( getattr( python_value, name ), name, fai, mutable, driver, parent = self )
            else:
                sc = CallArg.analysis_of_python_arg( getattr( python_value, name ), name, fai, mutable, driver, parent = self )
            self.sub_list.append( sc )


    def configure_as_output_object_with_collect_attributes( self, fai, driver, return_type, *type_args, **type_kwargs ):
        if callable( getattr( return_type, "base_code_for", None ) ):
            self.base_code = return_type.base_code_for( driver, *type_args, **type_kwargs )
        else:
            self.base_code = return_type.__name__

        if callable( getattr( return_type, "signature_for", None ) ):
            self.signature = return_type.signature_for( driver, *type_args, **type_kwargs )
        else:
            self.signature = self.base_code

        def python_ctor( fai, outputs ):
            args = {}
            for item in self.sub_list:
                args[ item.attribute_name ] = item.python_ctor( fai, outputs )

            if callable( getattr( return_type, "__default_init__", None ) ):
                res = object.__new__( return_type )
                res.__default_init__( **args )
            else:
                res = return_type( **args )

            return res

        self.python_ctor = python_ctor

        self.brace_ctor = True

        self.sub_list = []
        for name, value in collect_attributes( return_type, use_annotations = True ):
            # if isinstance( value, Annotation ):
            #     value = value.value
            #     # info( name, value )
            #     # import sys
            #     # sys.exit( 0 )
            self.sub_list.append( self.return_child( fai, driver, name, value, *type_args, **type_kwargs ) )

    def return_child( self, fai, driver, attribute_name: str, return_type: any, *type_args, **type_kwargs ) -> Self:
        # base attributes
        res = CallArg()
        res.attribute_name = attribute_name
        res.python_value = None
        res.io_category = 2
        res.sub_list = None
        res.parent = weakref.ref( self )

        # analysis
        res.configure_as_return( fai, driver, 2, return_type, *type_args, **type_kwargs )

        return res

    def configure_as_raw_return( self, fai, valid, driver, return_type, *args, **kwargs ):
        arg_name, validity_index = fai.add_raw_return( return_type, valid, driver, *args, **kwargs )

        def python_ctor( fai, outputs ):
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

    def construct( self, fai, outputs ):
        return self.python_ctor( fai, outputs )

    def update( self, fai, outputs ):
        for item in self.sub_list:
            if callable( item.updated_value ):
                setattr( self.python_value, item.attribute_name, item.updated_value( fai, outputs ) )
            else:
                item.update( fai, outputs )

    def as_input( self, driver ):
        if self.io_category == 0:
            return self

    def add_gradients_to( self, nargs: dict, name: str, driver, grads_of_the_outputs ):
        if self.ffi_input is not None and self.ffi_input.differentiable:
            nargs[ "grad_of_inp_" + name ] = Return( Tensor, self.ffi_input.python_value.shape, dtype = self.ffi_input.python_value.dtype )

        if self.ffi_output is not None and self.ffi_output.differentiable:
            n = self.ffi_output.num_in_sub_list
            if n < len( grads_of_the_outputs ):
                grad = grads_of_the_outputs[ n ]
            else:
                grad = driver.empty( [ 0 for _ in self.ffi_output.spec.shape ], dtype = self.ffi_output.spec.dtype )
            nargs[ "grad_of_out_" + name ] = grad

        if self.sub_list is not None:
            for s in self.sub_list:
                s.add_gradients_to( nargs, name + "_" + s.attribute_name, driver, grads_of_the_outputs )

    def update_differentiable_input_values( self ):
        if self.ffi_input is not None and self.ffi_input.differentiable:
            self.python_value = self.ffi_input.python_value
        if self.sub_list is not None:
            for s in self.sub_list:
                s.update_differentiable_input_values()
