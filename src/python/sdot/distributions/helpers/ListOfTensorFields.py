from .TensorField import GenericTensor, _rank
from typing import Self, overload
from ...driver import driver

class ListOfTensorFields:
    """
    """

    def __init__( self, main_axis_name : str, tensor_axis_names : list[ str ] ):
        self.tensor_axis_names = tensor_axis_names
        self.main_axis_name = main_axis_name
        self.name = None

    def __set_name__( self, distribution, name ):
        self.name = name

    # overloads for typing
    @overload
    def __get__( self, distribution: None, _type: type ) -> Self: ...
    @overload
    def __get__( self, distribution: object, _type: type ) -> GenericTensor: ...

    def __get__( self, distribution, _type = None ):
        # not in a Distribution ?
        if distribution is None:
            return self

        # we have a value ?
        value = distribution.__dict__.get( f'_{ self.name }' )
        if value is not None:
            return value

        # we have a default method ?
        default_method = getattr( type( distribution ), f'default_{ self.name }', None )
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

    def __set__( self, distribution, values ):
        if values is None:
            return

        # check the size
        curr = len( values )
        size = getattr( distribution, self.main_axis_name )
        if size is not None:
            if curr != size:
                raise RuntimeError( f"list used to define the '{ self.name }' attribute is of a wrong size: expecting { size } ('{ self.main_axis_name }' axis), provided tensor len is { curr }" )

        # make the tensor list
        tensors = [ driver.tn( value, _rank( distribution, self.tensor_axis_names ), self.name ) for value in values ]

        # TODO # check the dimensions
        # for axis_name in _names( self.axis_names ):
        #     size = getattr( distribution, axis_name )
        #     if size is not None:
        #         currs = self._get_axis_sizes( distribution, tensor, axis_name )
        #         for curr in currs:
        #             if curr is not None and curr != size:
        #                 raise RuntimeError( f"tensor used to define the '{ self.name }' attribute is not of the correct size along the '{ axis_name }' axis (expecting { size }, provided tensor shape - minus offset - is { curr })" )

        # register the tensor list
        distribution.__dict__[ f'_{ self.name }' ] = tensors

    def _rank( self, distribution ):
        return _rank( distribution, self.tensor_axis_names )

    def _get_axis_sizes( self, distribution, v, axis_name ) -> list[ tuple[ int, ... ] | int ]:
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
