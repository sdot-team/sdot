from ..BatchOfDistributions import BatchOfDistributions
from ..Distribution import Distribution

from .ListOfTensorFields import ListOfTensorFields
from .TensorField import TensorField, _axis_names

from ...driver import driver

from typing import TypeVar, Type
import numpy

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
    all_the_axis_names = [ "batch_size" ]
    for _, field in fields:
        if isinstance( field, TensorField ):
            for name in _axis_names( field.axis_names ):
                if name not in all_the_axis_names:
                    all_the_axis_names.append( name )

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
    _setup_distribution_class( cls.UnidimensionalBatchVersion, all_the_axis_names )
    _setup_distribution_class( cls.UnidimensionalVersion, all_the_axis_names )
    _setup_distribution_class( cls.BatchVersion, all_the_axis_names )
    _setup_distribution_class( cls, all_the_axis_names )

    return cls


def variants_of( cls: Type[ _D ] ) -> tuple[ Type[ _D ], Type[ _D ], Type[ _D ] ]:
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

    parent = Distribution
    if batch_version:
        parent = BatchOfDistributions

    res = type( class_name, ( parent, ), { '__annotations__': { "pouet": int } } )
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


def _setup_distribution_class( cls, axis_names : list[ str ] ):
    fields = _collect_attributes( cls )

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


def to_tensor_list( obj ) -> list:
    if isinstance( obj, ( tuple, list ) ):
        res = []
        for o in obj:
            res += to_tensor_list( o )
        return res

    if isinstance( obj, numpy.ndarray ):
        return [ driver.array( obj ) ]

    if isinstance( obj, driver.array_type ):
        return [ obj ]

    raise RuntimeError( f"Unable to transform object of type { type( obj ) } to tensor list" )


def flat_tensor_list( distribution ) -> list:
    res = []
    for a_name, a_data in _collect_attributes( type( distribution ) ):
        if isinstance( a_data, ListOfTensorFields ):
            v = getattr( distribution, a_name )
            if v is not None:
                for item in v:
                    res.append( item )
            else:
                for _ in range( getattr( distribution, a_data.main_axis_name ) ):
                    res.append( driver.empty( [] ) )
        if isinstance( a_data, TensorField ):
            v = getattr( distribution, a_name )
            if v is not None:
                res.append( v )
            else:
                res.append( driver.empty( [] ) )
    return res

def unflat_tensor_list( distribution, out: list, inp: list ):
    for _, a_data in _collect_attributes( type( distribution ) ):
        if isinstance( a_data, ListOfTensorFields ):
            loc = []
            for _ in range( getattr( distribution, a_data.main_axis_name ) ):
                loc.append( inp.pop( 0 ) )
            out.append( loc )
        if isinstance( a_data, TensorField ):
            out.append( inp.pop( 0 ) )


def unflatten_args( f, g, args ):
    """ Convert a flat list of tensors to the structured list expected by the C++ binding. """
    res = []
    inp = list( args )
    unflat_tensor_list( f, res, inp )
    unflat_tensor_list( g, res, inp )
    return res
