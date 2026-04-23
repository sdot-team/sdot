# from ...driver import driver
from .collect_attributes import collect_attributes

class Return:
    """Declares what type a C++ function returns.

    Usage:
        driver.call( "make_hypercube", includes,
            Return( Cell, dim=2 ),
            frame, bnd
        )
        driver.call( "measure", includes,
            Return( Tensor, shape=[], dtype=float ),
            cell
        )
    The return_type must implement the protocol:
        return_type.output_specs( drv, **kwargs ) -> list[ (name, shape, dtype) ]
        return_type.from_outputs( arrays, **kwargs ) -> instance
    """
    def __init__( self, return_type, *args, **kwargs ):
        self.return_type = return_type
        self.type_kwargs = kwargs
        self.type_args   = args

    def cpp_class_name( self, driver ):
        if self.return_type is float:
            return "TF"
        if self.return_type is int:
            return "PI"
        return self.return_type.cpp_class_name_for( *self.type_args, **self.type_kwargs )

    def get_jax_ffi_args( self, jax_ffi_arg_list, driver, name: str, cpy_arg, for_return: bool ):
        if self.return_type is float:
            raise NotImplementedError
        if self.return_type is int:
            raise NotImplementedError

        # method
        if callable( getattr( self.return_type, "get_jax_ffi_args_for", None ) ):
            return self.return_type.get_jax_ffi_args_for( jax_ffi_arg_list, driver, name, cpy_arg, True, *self.type_args, **self.type_kwargs )

        # else, make an instance (slow but may be ok)
        instance = self.return_type( *self.type_args, **self.type_kwargs )
        jax_ffi_arg_list.get_jax_ffi_args( driver, name, instance, cpy_arg, True )

    def fake_instance( self, driver ):
        """ make a fake instance to help find how to compile a function with a value that comes from a return """
        # special method ?
        if callable( getattr( self.return_type, "fake_instance", None ) ):
            return self.return_type.fake_instance( driver, *self.type_args, **self.type_kwargs )

        # call ctor
        return self.return_type( *self.type_args, **self.type_kwargs )

    # def cpp_assembly_from_jax_ffi_compatible_args( self, driver, flat_arg_iterator, pos_in_validity_bits: list[ int ] ):
    #     return self.return_type.cpp_assembly_from_jax_ffi_compatible_args( flat_arg_iterator, pos_in_validity_bits, *self.type_args, **self.type_kwargs )

    # def python_assembly_from_jax_ffi_compatible_args( self, driver, flat_arg_iterator ):
    #     if callable( getattr( self.return_type, "python_assembly_from_jax_ffi_compatible_args", None ) ):
    #         return self.return_type.python_assembly_from_jax_ffi_compatible_args( driver, flat_arg_iterator, *self.type_args, **self.type_kwargs )
    #     raise NotImplementedError
