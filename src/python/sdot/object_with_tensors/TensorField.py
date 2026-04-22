from ..UndefinedTensor import UndefinedTensor
from ..driver import driver

from typing import Self, overload
from inspect import signature
import numpy

class GenericTensor( numpy.ndarray ):
    """ created only to please typing """
    requires_grad : bool


class TensorField:
    """
    Descriptor for a tensor field in a Distribution (or BatchOfDistributions, ...).

    axis_names : names of each axis (e.g. "nb_diracs", "dim").

    The tensor rank equals len( axis_names ) excepted if there's a "*dim" in the axis names

    If the field value is None and the host class defines a method
    ``default_<name>(self)``, that method is called to supply the value. ``default_<name>`` can take an additionnal argument,
    ``default_<name>( self, batch_version )``

    In distribution, we store the tensor value in 'distribution._{ name }'.

    When we set a TensorField, we update 'distribution._{ name }' with a tensor compatible with the choices in sdot.driver
    """

    def __init__( self, *axis_names: str, dtype = None ):
        self.comes_from_a_dim_list = False
        self.removed_dim_axes = []
        self.axis_names = axis_names
        self.dtype = dtype
        self.name = None

    @property
    def ndim( self ):
        for axis_name in self.axis_names:
            if "*" in axis_name:
                raise NotImplementedError
        return len( self.axis_names )

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
            # make the new value
            sig = signature( default_method )
            if len( sig.parameters ) == 2:
                value = default_method( enclosing, isinstance( enclosing, BatchOfDistributions ) )
            else:
                value = default_method( enclosing )

            # register it
            self.__set__( enclosing, value )

            # return the normalized value
            return enclosing.__dict__[ f'_{ self.name }' ]

        # not found :(
        return UndefinedTensor( _shape( self, enclosing ), self.dtype )

    def __set__( self, distribution, value ):
        if value is None:
            return

        # make the tensor
        tensor = driver.tn( value, _rank( distribution, self.axis_names ), self.name, self.dtype )

        # check the dimensions
        for axis_name in _axis_names( self.axis_names ):
            size = getattr( distribution, axis_name )
            if size is not None:
                currs = distribution._axis_count_for( self, tensor, axis_name, [] )
                for curr in currs:
                    if curr is not None and curr != size:
                        raise RuntimeError( f"tensor used to define the '{ self.name }' attribute is not of the correct size along the '{ axis_name }' axis (expecting { size }, provided tensor shape - minus potential offset - is { curr })" )

        # register the tensor
        distribution.__dict__[ f'_{ self.name }' ] = tensor

    # def cpp_class_name( self ):
    #     if driver.is_int_dtype( self.dtype ):
    #         return "MI"
    #     return "MF"

    # def to_nanobind_compatible_objects( self, obj ):
    #     if self.dtype == int:
    #         return [ ( obj, "MI" ) ]

    #     if self.dtype is not None:
    #         raise NotImplementedError( f"to_nanobind_compatible_objects with dtype = { self.dtype }" )

    #     return [ ( obj, "MF" ) ]

    # def cpp_assembly_from_nanobind_compatible_objects( self, obj, arg_names ):
    #     return f"tensor_view_{ self.ndim }( { arg_names.pop( 0 ) } )"

    def _rank( self, distribution ):
        return _rank( distribution, self.axis_names )

# ------------------------------------------------------ helpers ------------------------------------------------------
def _rank( distribution, base_axis_names ):
    res = 0
    for axis_name in base_axis_names:
        axis_name = axis_name.replace( ' ', '' )
        if "*" in axis_name:
            _, axis = axis_name.split( "*" )
            axis_size = getattr( distribution, axis )
            if axis_size is None:
                return None

            assert isinstance( axis_size, int )
            res += axis_size
            continue

        res += 1

    return res


def _shape( field, enclosing ):
    res = []
    for axis_name in field.axis_names:
        axis_name = axis_name.replace( ' ', '' )
        if "*" in axis_name:
            # _, axis = axis_name.split( "*" )
            # axis_size = getattr( enclosing, axis )
            # if axis_size is None:
            #     return None

            # assert isinstance( axis_size, int )
            # res += axis_size
            # continue
            raise NotImplementedError

        if "+" in axis_name:
            lhs, rhs = axis_name.split( "+" )
            rhs = int( rhs )

            res.append( getattr( enclosing, lhs ) + rhs )
            continue

        res.append( getattr( enclosing, axis_name ) )

    return res


def _axis_names( axis_names: tuple[ str, ... ] ) -> list[ str ]:
    res = []
    for axis_name in axis_names:
        axis_name = axis_name.replace( ' ', '' )

        if "," in axis_name:
            lhs = axis_name.split( "," )[ 0 ]
            if lhs not in res:
                res.append( lhs )
            continue

        if "[" in axis_name:
            lhs = axis_name.split( "[]" )[ 0 ]
            if lhs not in res:
                res.append( lhs )
            continue

        if "+" in axis_name:
            lhs = axis_name.split( "+" )[ 0 ]
            # rhs = axis_name.split( "+" )[ 1 ]
            if lhs not in res:
                res.append( lhs )
            continue

        if "*" in axis_name:
            name = axis_name.split( "*" )[ 0 ]
            axis = axis_name.split( "*" )[ 1 ]
            if name not in res:
                res.append( name )
            if axis not in res:
                res.append( axis )
            continue

        if axis_name not in res:
            res.append( axis_name )

    return res
