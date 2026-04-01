from ..BatchOfDistributions import BatchOfDistributions
from ...driver import driver

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

    def __init__( self, *axis_names: str ):
        self.axis_names = axis_names
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
            # make the new value
            sig = signature( default_method )
            if len( sig.parameters ) == 2:
                value = default_method( distribution, isinstance( distribution, BatchOfDistributions ) )
            else:
                value = default_method( distribution )

            # register it
            self.__set__( distribution, value )

            # return the normalized value
            return distribution.__dict__[ f'_{ self.name }' ]

        # not foud :(
        return None

    def __set__( self, distribution, value ):
        if value is None:
            return

        # make the tensor
        tensor = driver.tn( value, _rank( distribution, self.axis_names ), self.name )

        # check the dimensions
        for axis_name in _axis_names( self.axis_names ):
            size = getattr( distribution, axis_name )
            if size is not None:
                currs = self._get_axis_sizes( distribution, tensor, axis_name )
                for curr in currs:
                    if curr is not None and curr != size:
                        raise RuntimeError( f"tensor used to define the '{ self.name }' attribute is not of the correct size along the '{ axis_name }' axis (expecting { size }, provided tensor shape - minus potential offset - is { curr })" )

        # register the tensor
        distribution.__dict__[ f'_{ self.name }' ] = tensor


    def _get_axis_sizes( self, distribution, v, axis_name ) -> list[ tuple[ int, ... ] | int ]:
        # nb fields with "*"
        nb_fields_with_mul = 0
        for field_axis_name in self.axis_names:
            nb_fields_with_mul += "*" in field_axis_name

        # try
        out = []
        num_axis = 0
        for field_axis_name in self.axis_names:
            field_axis_name = field_axis_name.replace( ' ', '' )

            if "," in field_axis_name:
                lhs = field_axis_name.split( "," )[ 0 ]

                if axis_name == lhs:
                    out.append( ( v.shape[ num_axis ], ) )
                num_axis += 1
                continue

            if "+" in field_axis_name:
                lhs, rhs = field_axis_name.split( "+" )
                rhs = int( rhs )

                if axis_name == lhs:
                    out.append( v.shape[ num_axis ] - rhs )
                num_axis += 1
                continue

            if "*" in field_axis_name:
                field_name, field_axis = field_axis_name.split( "*" )

                if axis_name == field_name:
                    dim = getattr( distribution, field_axis )
                    if dim is None:
                        break

                    res = []
                    for d in range( dim ):
                        res.append( v.shape[ num_axis + d ] )
                    out.append( tuple( res ) )
                    continue

                if axis_name == field_axis:
                    if nb_fields_with_mul > 1:
                        raise NotImplementedError( "handle tensors with multiple a * b axes" )
                    out.append( v.ndim - ( len( self.axis_names ) - 1 ) )
                    continue

                break # num_axis += dim -> TODO: find a simple way to avoid the infinite loop

            if axis_name == field_axis_name:
                out.append( v.shape[ num_axis ] )
            num_axis += 1

        return out


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
