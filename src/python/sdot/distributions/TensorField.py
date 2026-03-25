from typing import Any, overload
from ..driver import driver

class TensorField:
    """
    Descriptor for a tensor field in a Distribution or BatchOfDistributions.

    axis_names : names of each axis (e.g. "nb_diracs", "dim").
                 The tensor rank equals len(axis_names).

    If the field value is None and the host class defines a method
    ``default_<name>(self)``, that method is called to supply the value.

    @generate_distribution_methods auto-generates a property for each axis
    name that returns the corresponding dimension of the first non-None field
    that carries that axis name.
    """

    def __init__( self, *axis_names ):
        self.from_a_dim_reduction = False
        self.axis_names = axis_names
        self.rank = len( axis_names )
        self.name = None  # filled by __set_name__

    def __set_name__( self, owner, name ):
        self.name = name

    @overload
    def __get__( self, obj: None, objtype: type ) -> 'TensorField': ...
    @overload
    def __get__( self, obj: object, objtype: type ) -> Any: ...
    def __get__( self, obj, objtype = None ):
        if obj is None:
            return self
        val = obj.__dict__.get( f'_{ self.name }' )
        if val is None:
            default_method = getattr( type( obj ), f'default_{ self.name }', None )
            if default_method is not None:
                return default_method( obj )
        return val

    def __set__( self, obj, value ):
        obj.__dict__[ f'_{ self.name }' ] = getattr( driver, f't{ self.rank }' )( value ) if value is not None else None

    def expand_for_batch( self, value, batch_size ):
        """ Add a leading batch dimension and repeat. """
        idx = ( None, ) + ( slice( None ), ) * self.rank
        rpt = [ batch_size ] + [ 1 ] * self.rank
        return driver.repeat( value[ idx ], rpt )
