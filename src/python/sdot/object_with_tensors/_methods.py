from .ListOfTensorFields import ListOfTensorFields
from .TensorField import TensorField, _axis_names

from ..driver import driver

from typing import TypeVar, Type, cast
# import numpy
import re

_D = TypeVar( '_D' ) # , bound=Distribution
_T = TypeVar( '_T' )


def object_with_tensors( cls: type[ _T ] ) -> type[ _T ]:
    """
    Class decorator that auto-generates boilerplate for classes with TensorField or ListOfTensorFields
    based on their attribute declarations

    Fields with a ``default_<name>(self)`` method in the class will use that
    method when the field value is None.

    Generated method and attributes
      - def __init__( self, attributes by their order of apparition )
      - multidimensionnal_version()
      - batch_version( batch_size )
      - properties
        - tensors (summary of all the tensors with name and value)
        - dim, nb_points, ... (one property per unique axis name across all TensorFields)

    Usage::

        @object_with_tensors
        class MyDist:
            positions = TensorField("nb_points", "dim")
            weights   = TensorField("nb_points")

            def default_weights(self,batch_version):
                return driver.ones(self.nb_points)

    """
    # fields
    fields = _collect_attributes( cls )

    # all the axis names
    static_axis_names = [ "batch_size" ]
    for _, field in fields:
        if isinstance( field, TensorField ):
            for name in _axis_names( field.static_axis_names ):
                if name not in static_axis_names:
                    static_axis_names.append( name )

    dynamic_axis_names = []
    for _, field in fields:
        if isinstance( field, TensorField ):
            for name in field.dynamic_axis_names:
                if name not in dynamic_axis_names:
                    dynamic_axis_names.append( name )
                    if name in static_axis_names:
                        raise RuntimeError( f"axes can't ne both dynamic and static (for axis { name })" )

    # make the variants (batch, unidim, ...)
    clu = _make_variant( cls, fields, batch_version = 0, unidimensional_version = 1 )
    clb = _make_variant( cls, fields, batch_version = 1, unidimensional_version = 0 )
    clt = _make_variant( cls, fields, batch_version = 1, unidimensional_version = 1 )

    # links beteen variants
    cls.UnidimensionalBatchVersion = clt
    cls.UnidimensionalVersion = clu
    cls.BatchVersion = clb
    cls.BaseVersion = cls

    clu.MultidimensionalVersion = cls
    clu.BatchVersion = clt
    clu.BaseVersion = cls

    clb.BatchItemVersion = cls
    clb.BaseVersion = cls

    clt.MultidimensionalVersion = clb
    clt.BatchItemVersion = clu
    clt.BaseVersion = cls

    # add generated methods and properties for these variants
    _setup_distribution_class( cls.UnidimensionalBatchVersion, static_axis_names, dynamic_axis_names )
    _setup_distribution_class( cls.UnidimensionalVersion, static_axis_names, dynamic_axis_names )
    _setup_distribution_class( cls.BatchVersion, static_axis_names, dynamic_axis_names )
    _setup_distribution_class( cls, static_axis_names, dynamic_axis_names )

    return cls


def variants_of( cls ): # : Type[ _D ] ) -> tuple[ Type[ _D ], Type[ _D ], Type[ _D ] ]:
    t = cls.UnidimensionalBatchVersion
    u = cls.UnidimensionalVersion
    b = cls.BatchVersion
    return u, b, t


def _make_variant( cls, fields, batch_version : int, unidimensional_version : int ):
    class_name = cls.__name__
    if batch_version:
        class_name = 'BatchOf' + class_name
    if unidimensional_version:
        class_name += "1d"

    # parents = ( Distribution, )
    # if batch_version:
    #     parents = ( BatchOfDistributions, )
    parents = ()

    res = type( class_name, parents, { '__annotations__': { "pouet": int } } )
    for class_name, field in fields:
        new_field = field

        # if ListOfTensorFields, add/remove axes
        comes_from_a_dim_list = False
        if isinstance( new_field, ListOfTensorFields ):
            if unidimensional_version and new_field.main_axis_name == "dim":
                tensor_axis_names = []
                for tensor_axis_name in new_field.tensor_axis_names:
                    n = tensor_axis_name.replace( ' ', '' ).replace( "[index]", "," )
                    tensor_axis_names.append( n )
                new_field = TensorField( *tensor_axis_names )
                comes_from_a_dim_list = True
            else:
                new_tensor_axis_names = [ "batch_size" ] * batch_version + list( new_field.tensor_axis_names )
                if unidimensional_version:
                    new_tensor_axis_names = list( filter( lambda x: x != "dim", new_tensor_axis_names ) )
                new_field = ListOfTensorFields( new_field.main_axis_name, new_tensor_axis_names )

        # if TensorField, add/remove axes
        if isinstance( new_field, TensorField ):
            removed_dim_axes = []
            new_axis_names = [ "batch_size" ] * batch_version + list( new_field.axis_names )
            if unidimensional_version:
                removed_dim_axes = [ i for i, x in enumerate( new_axis_names ) if x == "dim" ]
                new_axis_names = list( filter( lambda x: x != "dim", new_axis_names ) )
            new_field = TensorField( *new_axis_names )
            new_field.removed_dim_axes = removed_dim_axes
            new_field.comes_from_a_dim_list = comes_from_a_dim_list

        setattr( res, class_name, new_field )
        if hasattr( new_field, '__set_name__' ):
            new_field.__set_name__( res, class_name )

    return res


def _setup_distribution_class( cls, static_axis_names : list[ str ], dynamic_axis_names : list[ str ] ):
    fields = _collect_attributes( cls )

    # --- __init__ -----------------------------------------------------------
    if '__default_init__' not in vars( cls ):
        def __default_init__( self, *args, **kwargs ):
            for i, ( name, _ ) in enumerate( fields ):
                if i < len( args ):
                    kwargs.setdefault( name, args[ i ] )
            for name, val in fields:
                default = val
                if isinstance( val, ( ListOfTensorFields, TensorField, property ) ):
                    default = None
                setattr( self, name, kwargs.get( name, default ) )
        cls.__default_init__ = __default_init__

    # --- __init__ -----------------------------------------------------------
    if '__init__' not in vars( cls ):
        cls.__init__ = cls.__default_init__

    # --- dim -----------
    if 'dim' not in vars( cls ) and cls.__name__.endswith( "1d" ):
        cls.dim = property( lambda self: 1 )

    # --- properties from axes (dim, nb_points, ...) --------------------------
    for axis_name in static_axis_names:
        if axis_name not in vars( cls ):
            setattr( cls, axis_name, property( lambda self, a = axis_name: _axis_count( self, a ) ) )

    for axis_name in dynamic_axis_names:
        # tensor info -> axis_name + "_capacity"
        setattr( cls, axis_name + "_capacity", property( lambda self, a = axis_name + "_capacity": _axis_count( self, a ) ) )

        # else, add an attributen handled by a property
        def get_dyn( self ):
            return getattr( self, "_" + axis_name, 0 )
        def set_dyn( self, value: int ):
            if value >= getattr( axis_name + "_capacity" ):
                raise RuntimeError( f"no enough capacity (for '{ axis_name }')" )
            return setattr( self, "_" + axis_name, value )
        setattr( cls, axis_name, property( get_dyn, set_dyn ) )

    # --- tensor_list -----------
    if 'tensors' not in vars( cls ):
        def tensors( self ):
            res = []
            for name, field in fields:
                if isinstance( field, TensorField ):
                    res.append( ( name, getattr( self, name ) ) )
            return res
        setattr( cls, 'tensors', property( tensors ) )

    # --- dynamic_axes: {axis_name -> actual_size} for all Dyn axes -----------
    if 'dynamic_axes' not in vars( cls ):
        def _dynamic_axes( self ):
            res = {}
            for name, field in fields:
                if isinstance( field, TensorField ) and field.dynamic_axis_names:
                    tensor_value = self.__dict__.get( f'_{ name }' )
                    if tensor_value is not None:
                        for i, axis_name in enumerate( field.axis_names ):
                            if axis_name in field.dynamic_axis_names and axis_name not in res:
                                res[ axis_name ] = int( tensor_value.shape[ i ] )
            return res
        setattr( cls, 'dynamic_axes', property( _dynamic_axes ) )

    # --- batch_version -----------
    if hasattr( cls, "BatchVersion" ) and "batch_version" not in vars( cls ):
        def batch_version( self, batch_size ):
            # make ctor args
            kw = {}
            for name, field in fields:
                if isinstance( field, TensorField ):
                    v = getattr( self, name ) # .__dict__.get( name )
                    if v is not None:
                        rpt = [ batch_size ] + [ 1 ] * v.ndim
                        kw[ name ] = driver.repeat( v[ None, ... ], rpt )
                elif isinstance( field, ListOfTensorFields ):
                    v = getattr( self, name ) # .__dict__.get( name )
                    if v is not None:
                        # rpt = [ batch_size ] + [ 1 ] * v.ndim
                        # kw[ name ] = driver.repeat( v[ None, ... ], rpt )
                        raise NotImplementedError
                else:
                    kw[ name ] = self.__dict__.get( name )
            # make the new instance
            return cls.BatchVersion( **kw )
        setattr( cls, 'batch_version', batch_version )

    # --- batch_item -----------
    if hasattr( cls, "BatchItemVersion" ) and "batch_item" not in vars( cls ):
        def batch_item( self, batch_index ):
            # make ctor args
            kw = {}
            for name, field in fields:
                if isinstance( field, TensorField ):
                    v = getattr( self, name )
                    if v is not None:
                        kw[ name ] = v[ batch_index, ... ]
                elif isinstance( field, ListOfTensorFields ):
                    lst = getattr( self, name )
                    if lst is not None:
                        kw[ name ] = [ v[ batch_index, ... ] for v in lst ]
                else:
                    kw[ name ] = self.__dict__.get( name )
            # make the new instance
            return cls.BatchItemVersion( **kw )
        setattr( cls, 'batch_item', batch_item )

    # --- _axis_count_for -----------
    if "_axis_count_for" not in vars( cls ):
        setattr( cls, '_axis_count_for', _axis_count_for )

    # --- multidimensional_version -----------
    if hasattr( cls, "MultidimensionalVersion" ) and "multidimensional_version" not in vars( cls ):
        def multidimensional_version( self ):
            # make ctor args
            kw = {}
            for name, field in fields:
                if isinstance( field, TensorField ):
                    v = getattr( self, name ) # .__dict__.get( name )
                    if v is not None:
                        for n in field.removed_dim_axes:
                            v = driver.expand_dims( v, n )
                        if field.comes_from_a_dim_list:
                            v = [ v ] # raise NotImplementedError
                        kw[ name ] = v
                elif isinstance( field, ListOfTensorFields ):
                    raise NotImplementedError
                else:
                    kw[ name ] = self.__dict__.get( name )
            # make the new instance
            return cls.MultidimensionalVersion( **kw )
        setattr( cls, 'multidimensional_version', multidimensional_version )

    return cls

# ---------------------------------------------------------------------------
# Helpers used by @generate_distribution_methods
# ---------------------------------------------------------------------------

def _collect_attributes( cls, stop_classes = [] ):
    """ All attributes visible from cls (MRO), excluding stop_classes. """
    res = []
    name_indices = {} # if attribute appears in a subclass and in a parent class, we want to take
    for klass in reversed( cls.__mro__ ):
        if klass in stop_classes:
            continue
        for name, val in vars( klass ).items():
            if name.startswith( '_' ) or isinstance( val, ( classmethod, staticmethod ) ) or callable( val ):
                continue
            if isinstance( val, property ) and val.fset is None:
                continue

            if name in name_indices:
                res[ name_indices[ name ] ] = ( name, val )
            else:
                name_indices[ name ] = len( res )
                res.append( ( name, val ) )
    return res


def _axis_count( distribution, axis_name, fields_to_avoid = [] ):
    for tensor_name, tensor_field in _collect_attributes( type( distribution ) ):
        if isinstance( tensor_field, TensorField ):
            # if we have a value for this tensor field (to get the shape)
            tensor_value = distribution.__dict__.get( f'_{ tensor_name }' )
            if tensor_value is not None:
                # -> we can try to use the shape
                res = _axis_count_for( distribution, tensor_field, tensor_value, axis_name, fields_to_avoid )
                if len( res ):
                    return res[ 0 ]
    return None


def _axis_count_for( distribution, tensor_field, tensor_value, axis_name, fields_to_avoid ) -> list[ tuple[ int, ... ] | int ]:
    if tensor_value is None:
        return []

    # count nb fields with "*"
    nb_fields_with_mul = 0
    for field_axis_name in tensor_field.axis_names:
        nb_fields_with_mul += "*" in field_axis_name

    # try
    out = []
    num_axis = 0
    for field_axis_name in tensor_field.axis_names:
        field_axis_name = field_axis_name.replace( ' ', '' )

        if "," in field_axis_name:
            lhs = field_axis_name.split( "," )[ 0 ]

            if axis_name == lhs:
                out.append( ( tensor_value.shape[ num_axis ], ) )
            num_axis += 1
            continue

        if "+" in field_axis_name:
            lhs, rhs = field_axis_name.split( "+" )
            rhs = int( rhs )

            if axis_name == lhs:
                out.append( tensor_value.shape[ num_axis ] - rhs )
            num_axis += 1
            continue

        if "*" in field_axis_name:
            if nb_fields_with_mul > 1:
                raise NotImplementedError( "handle tensors with multiple a * b axes" )

            # ex shape * dim -> "shape", "dim"
            name_shape, name_dim = map( str.strip, field_axis_name.split( "*" ) )

            # want "dim"
            dim = tensor_value.ndim - ( len( tensor_field.axis_names ) - 1 )
            if axis_name == name_dim:
                out.append( dim )

            # want "shape"
            if axis_name == name_shape:
                res = [ tensor_value.shape[ num_axis + d ] for d in range( dim ) ]
                out.append( tuple( res ) )

            num_axis += dim
            continue

        if axis_name == field_axis_name:
            out.append( tensor_value.shape[ num_axis ] )

        num_axis += 1

    return out
