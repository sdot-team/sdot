from .TensorField import GenericTensor, _rank
from typing import Self, overload
from ..driver import driver

class ListOfTensorFields:
    """
    """

    def __init__( self, main_axis_name : str, tensor_axis_names : list[ str ], dtype = None ):
        self.tensor_axis_names = tensor_axis_names
        self.main_axis_name = main_axis_name
        self.dtype = dtype
        self.name = None

    def __set_name__( self, enclosing, name ):
        self.name = name

    # overloads for typing
    @overload
    def __get__( self, enclosing: None, _type: type ) -> Self: ...
    @overload
    def __get__( self, enclosing: object, _type: type ) -> GenericTensor: ...

    def __get__( self, enclosing, _type = None ):
        # not in a Distribution ?
        if enclosing is None:
            return self

        # we have a value ?
        value = enclosing.__dict__.get( f'_{ self.name }' )
        if value is not None:
            return value

        # we have a default method ?
        default_method = getattr( type( enclosing ), f'default_{ self.name }', None )
        if default_method is not None:
            raise NotImplementedError
            #     # make the new value
            #     sig = signature( default_method )
            #     if len( sig.parameters ) == 2:
            #         value = default_method( distribution, isinstance( distribution, BatchOfDistributions ) )
            #     else:
            #         value = default_method( distribution )

            #     # register it
            #     self.__set__( distribution, value )

            #     # return the normalized value
            #     return distribution.__dict__[ f'_{ self.name }' ]

        # not found :(
        return None

    def __set__( self, enclosing, values ):
        if values is None:
            return

        # check the size
        curr = len( values )
        size = getattr( enclosing, self.main_axis_name )
        if size is not None:
            if curr != size:
                raise RuntimeError( f"list used to define the '{ self.name }' attribute is of a wrong size: expecting { size } ('{ self.main_axis_name }' axis), provided tensor len is { curr }" )

        # make the tensor list
        tensors = [ driver.tn( value, _rank( enclosing, self.tensor_axis_names ), self.name ) for value in values ]

        # TODO # check the dimensions
        # for axis_name in _names( self.axis_names ):
        #     size = getattr( distribution, axis_name )
        #     if size is not None:
        #         currs = self._get_axis_sizes( distribution, tensor, axis_name )
        #         for curr in currs:
        #             if curr is not None and curr != size:
        #                 raise RuntimeError( f"tensor used to define the '{ self.name }' attribute is not of the correct size along the '{ axis_name }' axis (expecting { size }, provided tensor shape - minus offset - is { curr })" )

        # register the tensor list
        enclosing.__dict__[ f'_{ self.name }' ] = tensors

    def _rank( self, enclosing ):
        return _rank( enclosing, self.tensor_axis_names )

    # def cpp_class_name( self ):
    #     if driver.is_int_dtype( self.dtype ):
    #         return "std::vector<MI>"
    #     return "std::vector<MF>"

    # def to_nanobind_compatible_objects( self, obj ):
    #     raise NotImplementedError

    # def cpp_assembly_from_nanobind_compatible_objects( self, obj, arg_names ):
    #     #return f"tensor_view_{ self.ndim }( { arg_names.pop( 0 ) } )"
    #     raise NotImplementedError

    def _get_axis_sizes( self, enclosing, v, axis_name ) -> list[ tuple[ int, ... ] | int ]:
        raise NotImplementedError
        # # nb fields with "*"
        # nb_fields_with_mul = 0
        # for field_axis_name in self.axis_names:
        #     nb_fields_with_mul += "*" in field_axis_name

        # # try
        # out = []
        # num_axis = 0
        # for field_axis_name in self.axis_names:
        #     field_axis_name = field_axis_name.replace( ' ', '' )

        #     if "+" in field_axis_name:
        #         lhs, rhs = field_axis_name.split( "+" )
        #         rhs = int( rhs )

        #         if axis_name == lhs:
        #             out.append( v.shape[ num_axis ] - rhs )
        #         num_axis += 1
        #         continue

        #     if "*" in field_axis_name:
        #         field_name, field_axis = field_axis_name.split( "*" )

        #         if axis_name == field_name:
        #             dim = getattr( distribution, field_axis )
        #             if dim is None:
        #                 break

        #             res = []
        #             for d in range( dim ):
        #                 res.append( v.shape[ num_axis + d ] )
        #             out.append( tuple( res ) )
        #             continue

        #         if axis_name == field_axis:
        #             if nb_fields_with_mul > 1:
        #                 raise NotImplementedError( "handle tensors with multiple a * b axes" )
        #             out.append( v.ndim - ( len( self.axis_names ) - 1 ) )
        #             continue

        #         break # num_axis += dim -> TODO: find a simple way to avoid the infinite loop

        #     if axis_name == field_axis_name:
        #         out.append( v.shape[ num_axis ] )
        #     num_axis += 1

        # return out
