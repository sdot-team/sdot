from ..drivers.compilation.collect_attributes import collect_attributes
from .ListOfTensorFields import ListOfTensorFields
from .TensorField import TensorField, _axis_names
from ..Dyn import Dyn

from ..driver import driver

from typing import TypeVar #, Type, cast
# import numpy
# import re

# _D = TypeVar( '_D' ) # , bound=Distribution
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
    # base fields
    fields = collect_attributes( cls )

    # all the static axes
    static_axes_names = [ "batch_size" ]
    for _, field in fields:
        if isinstance( field, TensorField ):
            for name in _axis_names( field.static_axis_names ):
                if name not in static_axes_names:
                    static_axes_names.append( name )

    # all the dynamic axes
    dynamic_axes = []  # name -> Dyn object (preserves one_value_for_each and other metadata)
    for _, field in fields:
        if isinstance( field, TensorField ):
            for dynamic_axis in field.dynamic_axes:
                # already registered ?
                item = next( ( da for da in dynamic_axes if da.name == dynamic_axis.name ), None )
                if item is not None:
                    assert item.one_value_for_each == dynamic_axis.one_value_for_each
                    continue

                # else, check if not presetn in static args and register it
                if dynamic_axis.name in static_axes_names:
                    raise RuntimeError( f"axes can't be both dynamic and static (for axis '{ name }')" )
                dynamic_axes.append( dynamic_axis )

    # add a TensorField for each Dynamic axis — inserted at the beginning of fields so they are
    for dynamic_axis in dynamic_axes:
        if dynamic_axis.name not in vars( cls ):
            tf = TensorField( *dynamic_axis.one_value_for_each, dtype = int, represents_a_dynamic_size = True )
            tf.__set_name__( cls, dynamic_axis.name )

            fields.append( ( dynamic_axis.name, tf ) )
            setattr( cls, dynamic_axis.name, tf )

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
    _setup_distribution_class( cls.UnidimensionalBatchVersion, static_axes_names, dynamic_axes )
    _setup_distribution_class( cls.UnidimensionalVersion, static_axes_names, dynamic_axes )
    _setup_distribution_class( cls.BatchVersion, static_axes_names, dynamic_axes )
    _setup_distribution_class( cls, static_axes_names, dynamic_axes )

    return cls


def variants_of( cls ): # : Type[ _D ] ) -> tuple[ Type[ _D ], Type[ _D ], Type[ _D ] ]:
    t = cls.UnidimensionalBatchVersion
    u = cls.UnidimensionalVersion
    b = cls.BatchVersion
    return u, b, t


def _make_variant( cls, fields, batch_version : int, unidimensional_version : int ):
    variant_name = cls.__name__
    if batch_version:
        variant_name = 'BatchOf' + variant_name
    if unidimensional_version:
        variant_name += "1d"

    parents = ()

    res = type( variant_name, parents, { '__annotations__': cls.__annotations__ } )
    for name, field in fields:
        if isinstance( field, ( TensorField, ListOfTensorFields ) ):
            field = field.make_variant( batch_version, unidimensional_version )
        setattr( res, name, field )

    return res


def _setup_distribution_class( cls, static_axis_names : list[ str ], dynamic_axes : list[ Dyn ] ):
    fields = collect_attributes( cls )

    # --- __init__ -----------------------------------------------------------
    if '__default_init__' not in vars( cls ):
        def __default_init__( self, *args, **kwargs ):
            for i, ( name, _ ) in enumerate( fields ):
                if i < len( args ):
                    kwargs.setdefault( name, args[ i ] )
            # basic setattr
            for name, value in kwargs.items():
                setattr( self, name, value )

            # init fields if not already done
            for name, val in fields:
                if name not in kwargs and isinstance( val, ( ListOfTensorFields, TensorField ) ):
                    setattr( self, name, None )

        cls.__default_init__ = __default_init__

    # --- __init__ -----------------------------------------------------------
    if '__init__' not in vars( cls ):
        cls.__init__ = cls.__default_init__

    # --- dim -----------
    if 'dim' not in vars( cls ) and cls.__name__.endswith( "1d" ):
        cls.dim = property( lambda self: 1 )

    # --- getters for axes (dim, nb_points, ...) --------------------------
    # for dynamic axes, $name refers to a dynamic tensor. Shape are refered as $( name + "_capacity" )
    for axis_name in static_axis_names + [ a.name + "_capacity" for a in dynamic_axes ]:
        if axis_name not in vars( cls ):
            def get_axis_size( self, a = axis_name ):
                return _axis_count( self, a )
            setattr( cls, axis_name, property( get_axis_size ) )

    # --- tensor_list -----------
    if 'tensors' not in vars( cls ):
        def tensors( self ):
            res = []
            for name, field in fields:
                if isinstance( field, TensorField ):
                    res.append( ( name, getattr( self, name ) ) )
            return res
        setattr( cls, 'tensors', property( tensors ) )

    # --- dynamic_axes: {axis_name -> size} — int for scalar axes, list for per-item axes -----------
    # if 'dynamic_axes' not in vars( cls ):
    #     def _dynamic_axes( self, _info = dynamic_axes_info ) -> dict:
    #         res = {}
    #         for dyn_name, dyn in _info.items():
    #             val = getattr( self, dyn_name, None )
    #             if val is not None:
    #                 res[ dyn_name ] = val  # int for scalar, list/array for per-item
    #         return res
    #     setattr( cls, 'dynamic_axes', property( _dynamic_axes ) )

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

# def _collect_attributes( cls, stop_classes = [] ):
#     """ All attributes visible from cls (MRO), excluding stop_classes. """
#     res = []
#     name_indices = {} # if attribute appears in a subclass and in a parent class, we want to take
#     for klass in reversed( cls.__mro__ ):
#         if klass in stop_classes:
#             continue
#         for name, val in vars( klass ).items():
#             if name.startswith( '_' ) or isinstance( val, ( classmethod, staticmethod ) ) or callable( val ):
#                 continue
#             if isinstance( val, property ) and val.fset is None:
#                 continue

#             if name in name_indices:
#                 res[ name_indices[ name ] ] = ( name, val )
#             else:
#                 name_indices[ name ] = len( res )
#                 res.append( ( name, val ) )
#     return res


def _axis_count( distribution, axis_name, fields_to_avoid = [] ):
    for tensor_name, tensor_field in collect_attributes( type( distribution ) ):
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
