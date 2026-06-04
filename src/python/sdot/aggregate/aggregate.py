from ..util.get_all_annotations import get_all_annotations
from ..drivers.driver import driver
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
            positions = Tensor( "nb_points", "dim" )
            weights   = Tensor( "nb_points" )
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
                    if term.variable.selection is not None:
                        dynamic_shapes[ term.variable.name ] = term.variable.selection

    for axis_name, selection in dynamic_shapes.items():
        if axis_name not in fields:
            t = Tensor( *selection, dtype = int, represents_a_dynamic_axis = axis_name )
            cls.__annotations__[ axis_name ] = t
            fields[ axis_name ] = t

    cls.BaseVersion = cls

    _setup_distribution_class( cls )

    return cls


# ---------------------------------------------------------------------------
# Internal helpers (also used by batch_variant_of)
# ---------------------------------------------------------------------------

def _setup_distribution_class( cls ):
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

    for axis_name in _axis_names_of( cls ):
        if axis_name not in vars( cls ):
            def get_axis_size( self, a = axis_name ):
                return _axis_count( self, a )
            setattr( cls, axis_name, property( get_axis_size ) )

    return cls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _axis_names_of( cls ) -> list[ str ]:
    res = []
    for attr in get_all_annotations( cls ).values():
        if getattr( attr, "get_axis_variable_names", None ):
            attr.get_axis_variable_names( res )
    return res

def _axis_sources_of( distribution, fields_to_avoid ):
    """AxisTensorSource list for an @aggregate instance (same view the CallArg side uses)."""
    from .AxisVariableSystem import AxisTensorSource

    sources = []
    for tensor_name, tensor_field in get_all_annotations( type( distribution ) ).items():
        if tensor_field in fields_to_avoid or not isinstance( tensor_field, Tensor ):
            continue
        sources.append( AxisTensorSource(
            shape        = tensor_field.shape,
            ct_variables = tensor_field.ct_variables,
            numpy_value  = getattr( distribution, tensor_name, None )
        ) )
    return sources


def _axis_count( distribution, axis_name, fields_to_avoid = None ):
    if fields_to_avoid is None:
        fields_to_avoid = []

    from .AxisVariableSystem import AxisVariableSystem
    value = AxisVariableSystem( _axis_sources_of( distribution, fields_to_avoid ) ).local_value_of( axis_name )
    if value is not None:
        return value

    for tensor_name, tensor_field in get_all_annotations( type( distribution ) ).items():
        if tensor_field in fields_to_avoid or not isinstance( tensor_field, Tensor ):
            continue
        array = getattr( distribution, tensor_name, None )
        if array is None:
            continue

        n = 0
        for expr in tensor_field.shape:
            if len( expr.terms ) == 1 and expr.terms[ 0 ].variable.name == axis_name and expr.terms[ 0 ].variable.arguments:
                ndim = 1
                for argument in expr.terms[ 0 ].variable.arguments:
                    name, offset, coeff = argument.as_single_name()
                    ac = None if name is None else _axis_count( distribution, name )
                    if ac is None:
                        ndim = 0
                        break
                    ndim *= offset + coeff * ac
                if ndim:
                    return [ ( array.shape[ n + i ] - expr.offset ) // expr.terms[ 0 ].coeff for i in range( ndim ) ]

            all_the_arguments = [ a for term in expr.terms if term.variable.arguments for a in term.variable.arguments ]
            if len( all_the_arguments ) == 1:
                arg_name, arg_offset, arg_coeff = all_the_arguments[ 0 ].as_single_name()
                if arg_name == axis_name:
                    nb_with_argument = sum( 1 for e in tensor_field.shape if e.has_argument() )
                    if nb_with_argument == 1:
                        return ( array.ndim - ( len( tensor_field.shape ) - nb_with_argument ) - arg_offset ) // arg_coeff

            n += expr.ndim( lambda n, _: getattr( distribution, n ) )

    return None
