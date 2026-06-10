from ..util.get_all_annotations import get_all_annotations
from ..util.append_if_unique import append_if_unique
# from ..drivers.driver import driver

from .AxisVariableSystem import AxisVariableSystem
from .Workspace import Workspace
from .Tensor import Tensor

from typing import TypeVar
_T = TypeVar( '_T' )

def aggregate( cls: type[ _T ] ) -> type[ _T ]:
    """
    Class decorator that generates boilerplate for classes with Tensor field declarations.

    Generated:
      - __init__( self, fields in declaration order )
      - __setattr__ with field coercion
      - one property per unique axis name (dim, nb_points, ...)
      - batch_axes = []
      - BaseVersion = cls

    Usage::
        @aggregate
        class MyStruct:
            positions : Tensor( "nb_points", "dim" )
            weights   : Tensor( "nb_points" )
    """
    fields = get_all_annotations( cls )

    # inject Tensor fields for dynamic shape axes (e.g. nb_vertices[], nb_cuts[])
    dynamic_shapes = {}
    for field in fields.values():
        if isinstance( field, Workspace ):
            field = field.return_type
        if isinstance( field, Tensor ):
            for expr in field.shape:
                for term in expr.terms:
                    if term.selection is not None:
                        dynamic_shapes[ term.name ] = term.selection

    for axis_name, selection in dynamic_shapes.items():
        if axis_name not in fields:
            t = Tensor( *selection, dtype = int, represents_a_dynamic_axis = axis_name )
            cls.__annotations__[ axis_name ] = t
            fields[ axis_name ] = t

    # batch data
    cls.BaseVersion = cls
    cls.batch_axes = []

    # append methods
    _setup_aggregate( cls )

    return cls


# ---------------------------------------------------------------------------
# Internal helpers (also used by batch_variant_of)
# ---------------------------------------------------------------------------

def _setup_aggregate( cls ):
    fields = get_all_annotations( cls )

    if '__aggregate_init__' not in vars( cls ):
        def __aggregate_init__( self, *args, **kwargs ):
            values = {}
            for name, value in kwargs.items():
                values[ name ] = value
            for i, name in enumerate( fields.keys() ):
                if i >= len( args ):
                    break
                if name in values:
                    raise RuntimeError( f"argument '{ name }' has already been specified" )
                values[ name ] = args[ i ]

            for name in values.keys():
                if name not in fields:
                    raise RuntimeError( f"'{ name }' is not a valid argument for ctor of '{ cls.__name__ }'" )

            for name, field in fields.items():
                value = None
                if name in values:
                    value = values[ name ]

                if coerce := getattr( field, "coerce", None ):
                    value = coerce( value )
                elif value is not None and not isinstance( value, field ):
                    value = field( value )

                setattr( self, name, value )

        cls.__aggregate_init__ = __aggregate_init__

    if '__init__' not in vars( cls ):
        cls.__init__ = cls.__aggregate_init__

    if '__setattr__' not in vars( cls ):
        def __setattr__( self, name, value ):
            annotation = fields.get( name )
            if annotation is not None:
                if coerce := getattr( annotation, "coerce", None ):
                    value = coerce( value )
                elif value is not None and not isinstance( value, annotation ):
                    value = annotation( value )
            object.__setattr__( self, name, value )
        cls.__setattr__ = __setattr__

    if 'batch_axes_dict' not in vars( cls ):
        def batch_axes_dict( self ):
            res = {}
            for batch_axis in self.batch_axes:
                res[ batch_axis ] = getattr( self, batch_axis )
            return res
        cls.batch_axes_dict = property( batch_axes_dict )

    if '_aggregate_items' not in vars( cls ):
        cls._aggregate_items = _aggregate_items

    if 'with_prepended_batch_axis' not in vars( cls ):
        # vmap hook (see JaxDriver batch rule): return a same-typed instance carrying the
        # moved leading-N tensors, with one batch axis prepended to its instance batch_axes.
        # Type-stable — Cell stays Cell. `N` is unused (the size lives in the moved shapes).
        def with_prepended_batch_axis( self, N, moved_leaves ):
            inst = type( self )( **moved_leaves )
            inst.batch_axes = [ f"vmap_{ len( self.batch_axes ) }" ] + list( self.batch_axes )
            return inst
        cls.with_prepended_batch_axis = with_prepended_batch_axis

    # value for each axis name
    for axis_name in _axis_variable_names_of( cls ):
        if axis_name not in vars( cls ):
            def get_axis_size( self, a = axis_name ):
                return _axis_count( self, a )
            setattr( cls, axis_name, property( get_axis_size ) )

    return cls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _axis_variable_names_of( cls ) -> list[ str ]:
    res = []
    for attr in get_all_annotations( cls ).values():
        for name in getattr( attr, "axis_variable_names", [] ):
            append_if_unique( res, name )
    return res


def _aggregate_items( self ) -> list[ str ]:
    res = {}
    for name, attr in get_all_annotations( type( self ) ).items():
        res[ name ] = ( attr, getattr( self, name ) )
    return res


def _axis_count( self, axis_name ):
    """Resolve an axis variable value (int, or list for an expansion axis) for `distribution`."""
    return AxisVariableSystem( self ).value_of( axis_name )


