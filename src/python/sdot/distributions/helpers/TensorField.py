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

    def __init__( self, *axis_names ):
        self.from_a_unidimensionnal_version = False
        self.from_a_batch_version = False
        self.axis_names = axis_names
        self.name = None

        if "*dim" not in axis_names:
            self.rank = len( axis_names )
        else:
            self.rank = None

    def __set_name__( self, distribution, name ):
        self.name = name

    # overloads for typing
    @overload
    def __get__( self, distribution: None, _type: type ) -> Self: ...
    @overload
    def __get__( self, distribution: object, _type: type ) -> GenericTensor: ...

    def __get__( self, distribution, _type = None ):
        if distribution is None:
            return self

        val = distribution.__dict__.get( f'_{ self.name }' )

        if val is None:
            default_method = getattr( type( distribution ), f'default_{ self.name }', None )
            if default_method is not None:
                sig = signature( default_method )
                if len( sig.parameters ) == 2:
                    val = default_method( distribution, isinstance( distribution, BatchOfDistributions ) )
                else:
                    val = default_method( distribution )

                val = driver.tn( val, self.rank )

                distribution.__dict__[ f'_{ self.name }' ] = val

        return val

    def __set__( self, obj, value ):
        if value is not None:
            obj.__dict__[ f'_{ self.name }' ] = driver.tn( value, self.rank )
        else:
            obj.__dict__[ f'_{ self.name }' ] = None


    # # @overload
    # # def __get__( self, obj: None, objtype: type ) -> 'TensorField': ...
    # # @overload
    # # def __get__( self, obj: object, objtype: type ) -> Any: ...
    # def __get__( self, obj, objtype = None ):
    #     if obj is None:
    #         return self
    #     val = obj.__dict__.get( f'_{ self.name }' )
    #     if val is None:
    #         default_method = getattr( type( obj ), f'default_{ self.name }', None )
    #         if default_method is not None:
    #             return default_method( obj )
    #     return val

    # def _batch_version( self, value, batch_size ):
    #     """ Add a leading batch dimension and repeat. """
    #     # if "..." in self.axis_names:
    #     #     idx = ( None, ) + ( slice( None ), ) * value.ndim
    #     #     rpt = [ batch_size ] + [ 1 ] * value.ndim
    #     #     return driver.repeat( value[ idx ], rpt )

    #     idx = ( None, ) + ( slice( None ), ) * self.rank
    #     rpt = [ batch_size ] + [ 1 ] * self.rank
    #     return driver.repeat( value[ idx ], rpt )

