from typing import Self

class CallArg:
    """
        Recursive analysis of an argument sent to driver.call( ... )

        Allows for
        - re-assembly and update of outputs of the ffi calls (which are basically flat lists of tensors)
        - generation of the assembly code for the C++ side

    """

    attribute_name : str # attribute name
    python_value   : any #
    io_category    : int # 0 => pure input, 1 => mutable, 2 => return
    python_ctor    : callable # for reassembly of python objects. Called using an object that contains lists like differentiable_ffi_outputs, ...
    base_code      : str # name of function or class for assembly of C++ objects (e.g. "Cell<TF,2>")
    signature      : str # name used to make the signature of the module (e.g. T0F64 for a tensor)
    sub_list       : list[ Self ] | None # arguments for python_ctor or base_code


    @staticmethod
    def analysis_of_python_arg( python_value: any, attribute_name: str, fai, mutable: bool, driver ):
        """ recursively construct a CallArg


        """

        # common info
        res = CallArg()
        res.attribute_name = attribute_name
        res.python_value = python_value
        res.io_category = 1 if mutable else 0
        res.sub_list = None

        # method
        if callable( getattr( python_value, "configure_call_arg", None ) ):
            python_value.configure_call_arg( res, fai, driver )
            return res

        # arrays
        if driver.is_a_tensor( python_value ):
            res.configure_as_input_tensor( python_value, mutable, fai, driver )
            return res

        # std objects
        if isinstance( python_value, float ):
            res.configure_as_parameter( python_value, "FP64", mutable, fai, driver )
            return res

        if isinstance( python_value, int ):
            res.configure_as_parameter( python_value, "PI64", mutable, fai, driver )
            return res

        # if isinstance( value, ( list, tuple ) ):
        #     cpy_arg._python_ctor = type( value )
        #     cpy_arg.signature_type = "L"
        #     cpy_arg.code = "std::tie"
        #     for num, item in enumerate( value ):
        #         self.configure_call_arg( driver, f"{ name }_{ num }", item, cpy_arg.arg( str( num ) ) )
        #     return

        # if value is None:
        #     cpy_arg.signature_type = "N"
        #     cpy_arg.code = "{}"
        #     return

        # # else, get attributes
        # cpy_arg.code = FfiArgInfo.cpp_class_name( driver, value )
        # cpy_arg.signature_type = cpy_arg.code
        # cpy_arg._python_ctor = type( value )
        # for attr, _ in collect_attributes( value ):
        #     self.configure_call_arg( driver, f"{ name }_{ attr }", getattr( value, attr ), cpy_arg.arg( attr ) )
        raise NotImplementedError

    def configure_as_parameter( self, python_value: any, cpp_type: str, mutable: bool, fai, driver ) -> Self:
        if mutable:
            raise NotImplementedError
        else:
            n = fai.add_parameter( python_value, cpp_type )

            self.python_ctor = None # not mutable -> we can directly use python_value
            self.signature = cpp_type
            self.base_code = n
        return self

    def configure_as_input_tensor( self, python_value: any, mutable: bool, fai, driver ) -> Self:
        arg_name, validity_index = fai.add_input_tensor( python_value, driver )
        dim = len( python_value.shape )

        self.python_ctor = None # not mutable -> we can used python_value
        self.base_code = f"tensor_view( CtInt<{ dim }>(), { arg_name }, validity_mask[ { validity_index // 64 } ] & { 1 << ( validity_index % 64 ) } )"
        self.signature = f"T{ dim }{ driver.normalized_type_for( python_value.dtype ) }"

        if mutable:
            raise NotImplementedError

        return self

    def configure_as_return( self, fai, driver, return_type, *type_args, **type_kwargs ):
        """ complete self, with information for return """
        self.io_category = 2

        if return_type is float:
            # return self.configure_as_raw_return( fai, driver, return_type, *args, **kwargs )
            raise NotImplementedError

        if return_type is int:
            # return self.configure_as_raw_return( fai, driver, return_type, *args, **kwargs )
            raise NotImplementedError


        # method
        if callable( getattr( return_type, "configure_call_ret_for", None ) ):
            return_type.configure_call_ret_for( self, fai, driver, *type_args, **type_kwargs )
            return self

        raise NotImplementedError
        # # else, make an instance (slow but may be ok)
        # instance = self.return_type( *self.type_args, **self.type_kwargs )
        # jax_ffi_arg_list.configure_call_arg( driver, name, instance, cpy_arg )
        # pouet()

    def configure_as_raw_return( self, fai, valid, driver, return_type, *args, **kwargs ) -> Self:
        arg_name, validity_index = fai.add_raw_return( return_type, valid, driver, *args, **kwargs )
        self.python_ctor = lambda x: x #
        self.base_code = f"*{ arg_name }"
        self.signature = f"{ return_type }{ args }{ kwargs }"
        return self

    def assembled_code( self ) -> str:
        res = self.base_code
        if self.sub_list is not None:
            res += "( " + str.join( ", ", [ a.assembled_code() for a in self.sub_list ] ) + " )"
        return res

    def construct( fai, differentiable_outputs, non_differentiable_outputs ):
        raise NotImplementedError


    # def arg( self, name: str ):
    #     cpy_arg = CallArg( name )

    #     cpy_arg.for_return = self.for_return

    #     if self.sub_list is None:
    #         self.sub_list = []
    #     self.sub_list.append( cpy_arg )

    #     return cpy_arg

    # def reassemble( self, differentiable_inputs, non_differentiable_inputs, differentiable_outputs, non_differentiable_outputs ):
    #     # tensor ?
    #     if isinstance( self._python_ctor, tuple ):
    #         assert len( self._python_ctor ) == 3

    #         # not valid ?
    #         if not self._python_ctor[ 2 ]:
    #             return None

    #         # else, get tensor
    #         if self.for_return:
    #             if self._python_ctor[ 0 ]:
    #                 return differentiable_outputs[ self._python_ctor[ 1 ] ]
    #             return non_differentiable_outputs[ self._python_ctor[ 1 ] ]
    #         if self._python_ctor[ 0 ]:
    #             return differentiable_inputs[ self._python_ctor[ 1 ] ]
    #         return non_differentiable_inputs[ self._python_ctor[ 1 ] ]

    #     #
    #     if self._python_ctor is None:
    #         raise RuntimeError( f"No python ctor for { self.name }" )

    #     # ctor
    #     if self.sub_list is None:
    #         return self._python_ctor()
    #     return self._python_ctor( *[ item.reassemble( differentiable_inputs, non_differentiable_inputs, differentiable_outputs, non_differentiable_outputs ) for item in self.sub_list ] )

    # def update( self, obj, cb, differentiable_output_values, non_differentiable_output_values ):
    #     """ obj = ref. cb = how to modify the ref """
    #     # tensor ?
    #     if isinstance( self._python_ctor, tuple ):
    #         assert len( self._python_ctor ) == 3

    #         # not valid ?
    #         if not self._python_ctor[ 2 ]:
    #             return

    #         if cb is None:
    #             raise RuntimeError( "Mutable() takes only objects that contains other objects (like lists for instance)" )
    #         if self._python_ctor[ 0 ]:
    #             return cb( differentiable_output_values[ self._python_ctor[ 1 ] ] )
    #         return cb( non_differentiable_output_values[ self._python_ctor[ 1 ] ] )

    #     # ctor
    #     if self.sub_list is None:
    #         return

    #     for s in self.sub_list:
    #         # get sub obj + how to update it
    #         s = cast( CallArg, s )
    #         if s.name.isdigit():
    #             k = int( s.name )
    #             nobj = obj[ k ]
    #             def ncb( x ):
    #                 obj[ k ] = x
    #         else:
    #             nobj = getattr( obj, s.name )
    #             def ncb( x ):
    #                 setattr( obj, s.name, x )

    #         # recursive call
    #         s.update( nobj, ncb, differentiable_output_values, non_differentiable_output_values )


