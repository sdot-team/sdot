from ..util.get_all_annotations import get_all_annotations
from ..driver import driver
from .Tensor import Tensor

from typing import TypeVar
_T = TypeVar( '_T' )

def aggregate( cls: type[ _T ] ) -> type[ _T ]:
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

        @aggregate
        class MyDist:
            positions = Tensor( "nb_points", "dim")
            weights   = Tensor( "nb_points" )

            def default_weights(self,batch_version):
                return driver.ones(self.nb_points)

    """
    # base fields
    fields = get_all_annotations( cls )

    # add tensors for dynamic shapes
    dynamic_shapes = {}
    for field in fields.values():
        if isinstance( field, Tensor ):
            for expr in field.shape:
                for term in expr.terms:
                    if term.variable.selection is not None:
                        dynamic_shapes[ term.variable.name ] = term.variable.selection

    for name, selection in dynamic_shapes.items():
        if name not in fields:
            t = Tensor( *selection, dtype = int, represents_a_dynamic_axis = name )
            cls.__annotations__[ name ] = t
            fields[ name ] = t

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
    _setup_distribution_class( cls.UnidimensionalBatchVersion )
    _setup_distribution_class( cls.UnidimensionalVersion )
    _setup_distribution_class( cls.BatchVersion )
    _setup_distribution_class( cls )

    return cls


def variants_of( cls ): # : Type[ _D ] ) -> tuple[ Type[ _D ], Type[ _D ], Type[ _D ] ]:
    t = cls.UnidimensionalBatchVersion
    u = cls.UnidimensionalVersion
    b = cls.BatchVersion
    return u, b, t


def _make_variant( cls, fields: dict, batch_version : int, unidimensional_version : int ):
    variant_name = cls.__name__
    if batch_version:
        variant_name = 'BatchOf' + variant_name
    if unidimensional_version:
        variant_name += "1d"

    parents = ()

    res = type( variant_name, parents, {} )
    for name, field in fields.items():
        # synthetic attribtues
        # if getattr( field, "represents_a_dynamic_axis", None ):
        #     continue

        # make the new field
        if make_variant := getattr( field, "make_variant", None ):
            field = make_variant( batch_version, unidimensional_version )

        # store it
        res.__annotations__[ name ] = field

    return res


def _setup_distribution_class( cls ):
    fields = get_all_annotations( cls )

    # --- __init__ -----------------------------------------------------------
    if '__aggregate_init__' not in vars( cls ):
        def __aggregate_init__( self, *args, **kwargs ):
            # make a dict
            values = {}
            for name, value in kwargs.items():
                values[ name ] = value
            for i, name in enumerate( fields.keys() ):
                if i >= len( args ):
                    break
                if name in values:
                    raise RuntimeError( f"argument '{ name }' has already been specified" )
                values[ name ] = args[ i ]

            # check that all arguments correspond to a field
            for name in values.keys():
                if name not in fields:
                    raise RuntimeError( f"'{ name } is no a valid argument for ctor of '{ cls.__name__ }''" )

            # set values
            for name, field in fields.items():
                if name in values:
                    cted = field( values[ name ] )
                else:
                    cted = field()
                setattr( self, name, cted )

        cls.__aggregate_init__ = __aggregate_init__

    # --- __init__ -----------------------------------------------------------
    if '__init__' not in vars( cls ):
        cls.__init__ = cls.__aggregate_init__

    # --- __setattr__ --------------------------------------------------------
    if '__setattr__' not in vars( cls ):
        def __setattr__( self, name, value ):
            annotation = fields.get( name )
            if coerce := getattr( annotation, "coerce", None ):
                value = coerce( value )
            else:
                assert isinstance( value, annotation )
            object.__setattr__( self, name, value )
        cls.__setattr__ = __setattr__

    # --- dim -----------
    if 'dim' not in vars( cls ) and cls.__name__.endswith( "1d" ):
        cls.dim = property( lambda self: 1 )

    # --- getters for axes (dim, nb_points, ...) --------------------------
    # for dynamic axes, $name refers to a dynamic tensor. Shape are refered as $( name + "_capacity" )
    for axis_name in _axis_names_of( cls ):
        if axis_name not in vars( cls ):
            def get_axis_size( self, a = axis_name ):
                return _axis_count( self, a )
            setattr( cls, axis_name, property( get_axis_size ) )

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

def _axis_names_of( cls ) -> set[ str ]:
    res = set()
    for _, attr in get_all_annotations( cls ).items():
        if getattr( attr, "get_axis_names", None ):
            attr.get_axis_names( res )
    return res

def _axis_count( distribution, axis_name, fields_to_avoid = [] ):
    for tensor_name, tensor_field in get_all_annotations( type( distribution ) ).items():
        if tensor_field in fields_to_avoid:
            continue
        if isinstance( tensor_field, Tensor ):
            for n, expr in enumerate( tensor_field.shape ):
                if len( expr.terms ) == 1 and expr.terms[ 0 ].variable.name == axis_name:
                    array = getattr( distribution, tensor_name )
                    return ( array.shape[ n ] - expr.offset ) // expr.terms[ 0 ].coeff

    return None


