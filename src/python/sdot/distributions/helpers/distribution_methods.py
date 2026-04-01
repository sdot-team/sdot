from ..BatchOfDistributions import BatchOfDistributions
from ..Distribution import Distribution

from .ListOfTensorFields import ListOfTensorFields
from .TensorField import TensorField, _axis_names

from ...driver import driver

from typing import TypeVar, Type

_D = TypeVar( '_D', bound=Distribution )
_T = TypeVar( '_T' )


def generate_distribution_methods( cls: type[ _T ] ) -> type[ _T ]:
    """
    Class decorator that auto-generates boilerplate for Distribution and
    BatchOfDistributions subclasses based on their attribute declarations
    (notably TensorField ones).

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

        @generate_distribution_methods
        class MyDist(Distribution):
            positions = TensorField("nb_points", "dim")
            weights   = TensorField("nb_points")

            def default_weights(self,batch_version):
                return driver.ones(self.nb_points)

    """
    assert issubclass( cls, Distribution )

    # fields
    fields = _collect_attributes( cls )

    # all the axis names
    axis_names = [ "batch_size" ]
    for _, field in fields:
        if isinstance( field, TensorField ):
            for name in _axis_names( field.axis_names ):
                if name not in axis_names:
                    axis_names.append( name )

    # make the varians (batch, unidim, ...)
    _make_variant( cls, fields, axis_names, batch_version = 0, unidimensional_version = 1 )
    _make_variant( cls, fields, axis_names, batch_version = 1, unidimensional_version = 0 )
    _make_variant( cls, fields, axis_names, batch_version = 1, unidimensional_version = 1 )
    cls.UnidimensionalBatchVersion.MultidimensionalVersion = cls.BatchVersion
    cls.BatchVersion.UnidimensionalVersion = cls.UnidimensionalBatchVersion
    cls.UnidimensionalVersion.MultidimensionalVersion = cls

    # add generated methods and properties for these variants
    _setup_distribution_class( cls.UnidimensionalBatchVersion, fields, axis_names )
    _setup_distribution_class( cls.UnidimensionalVersion, fields, axis_names )
    _setup_distribution_class( cls.BatchVersion, fields, axis_names )
    _setup_distribution_class( cls, fields, axis_names )

    return cls


def variants_of( cls: Type[ _D ] ) -> tuple[ Type[ _D ], Type[ _D ], Type[ _D ] ]:
    t = cls.UnidimensionalBatchVersion
    u = cls.UnidimensionalVersion
    b = cls.BatchVersion
    return u, b, t


def _make_variant( cls, fields, axis_names : list[ str ], batch_version : int, unidimensional_version : int ):
    class_name = cls.__name__
    if batch_version:
        class_name = 'BatchOf' + class_name
    if unidimensional_version:
        class_name += "1d"

    parent = Distribution
    if batch_version:
        parent = BatchOfDistributions

    variant_name = ""
    if unidimensional_version:
        variant_name += "Unidimensional"
    if batch_version:
        variant_name += 'Batch'

    res = type( class_name, ( parent, ), { '__annotations__': { "pouet": int } } )
    setattr( cls, variant_name + "Version", res )
    for class_name, field in fields:
        new_field = field

        # if ListOfTensorFields, add/remove axes
        if isinstance( new_field, ListOfTensorFields ):
            if new_field.main_axis_name == "dim":
                tensor_axis_names = []
                for tensor_axis_name in new_field.tensor_axis_names:
                    n = tensor_axis_name.replace( ' ', '' ).replace( "[index]", "," )
                    tensor_axis_names.append( n )
                new_field = TensorField( *tensor_axis_names )
            else:
                new_tensor_axis_names = [ "batch_size" ] * batch_version + list( new_field.tensor_axis_names )
                if unidimensional_version:
                    new_tensor_axis_names = list( filter( lambda x: x != "dim", new_tensor_axis_names ) )
                new_field = ListOfTensorFields( new_field.main_axis_name, new_tensor_axis_names )

        # if TensorField, add/remove axes
        if isinstance( new_field, TensorField ):
            new_axis_names = [ "batch_size" ] * batch_version + list( new_field.axis_names )
            if unidimensional_version:
                new_axis_names = list( filter( lambda x: x != "dim", new_axis_names ) )
            new_field = TensorField( *new_axis_names )

        setattr( res, class_name, new_field )
        if hasattr( new_field, '__set_name__' ):
            new_field.__set_name__( res, class_name )


def _setup_distribution_class( cls, fields, axis_names : list[ str ] ):
    # --- __init__ -----------------------------------------------------------
    if '__init__' not in vars( cls ):
        def __init__( self, *args, **kwargs ):
            for i, ( name, _ ) in enumerate( fields ):
                if i < len( args ):
                    kwargs.setdefault( name, args[ i ] )
            for name, val in fields:
                default = val
                if isinstance( val, ( ListOfTensorFields, TensorField, property ) ):
                    default = None
                setattr( self, name, kwargs.get( name, default ) )
        cls.__init__ = __init__

    # --- dim -----------
    if 'dim' not in vars( cls ) and cls.__name__.endswith( "1d" ):
        cls.dim = property( lambda self: 1 )

    # --- properties from axes (dim, nb_points, ...) --------------------------
    for axis_name in axis_names:
        if axis_name not in vars( cls ):
            setattr( cls, axis_name, property( lambda self, a = axis_name: _axis_count( self, a ) ) )

    # --- tensor_list -----------
    if 'tensors' not in vars( cls ):
        def tensors( self ):
            res = []
            for name, field in fields:
                if isinstance( field, TensorField ):
                    res.append( ( name, getattr( self, name ) ) )
            return res
        setattr( cls, 'tensors', property( tensors ) )

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
                else:
                    kw[ name ] = self.__dict__.get( name )
            # make the new instance
            return cls.BatchVersion( **kw )
        setattr( cls, 'batch_version', batch_version )

    # --- multidimensional_version -----------
    if hasattr( cls, "MultidimensionalVersion" ) and "multidimensional_version" not in vars( cls ):
        def multidimensional_version( self ):
            # make ctor args
            kw = {}
            for name, field in fields:
                if isinstance( field, TensorField ):
                    v = getattr( self, name ) # .__dict__.get( name )
                    if v is not None:
                        if "dim" in field.axis_names:
                            kw[ name ] = driver.expand_dims( v, field.axis_names.index( "dim" ) )
                        else:
                            kw[ name ] = v
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


def _axis_count( distribution, axis_name ):
    for name, field in _collect_attributes( type( distribution ) ):
        if isinstance( field, TensorField ):
            # if we have a value for this tensor field (to get the shape)
            value = distribution.__dict__.get( f'_{ name }' )
            if value is not None:
                # -> we can try to use the shape
                res = field._get_axis_sizes( distribution, value, axis_name )
                if len( res ):
                    return res[ 0 ]
    return None
