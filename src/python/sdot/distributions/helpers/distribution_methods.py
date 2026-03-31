from ..BatchOfDistributions import BatchOfDistributions
from ..Distribution import Distribution
from .TensorField import TensorField
from ...driver import driver

from typing import TypeVar, Type
from inspect import signature

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

    # axis_names
    axis_names = set()
    for name, field in fields:
        if isinstance( field, TensorField ):
            for name in field.axis_names:
                if "*" not in name:
                    axis_names.add( name )

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


def _make_variant( cls, fields, axis_names : set[ str ], batch_version : int, unidimensional_version : int ):
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
        if isinstance( field, TensorField ):
            new_axis_names = [ "batch_size" ] * batch_version + list( field.axis_names )
            if unidimensional_version:
                new_axis_names = list( filter( lambda x: x != "dim", new_axis_names ) )
            new_field = TensorField( *new_axis_names )
        setattr( res, class_name, new_field )
        if hasattr( new_field, '__set_name__' ):
            new_field.__set_name__( res, class_name )


def _setup_distribution_class( cls, fields, axis_names ):
    # --- __init__ -----------------------------------------------------------
    if '__init__' not in vars( cls ):
        def __init__( self, *args, **kwargs ):
            for i, ( name, _ ) in enumerate( fields ):
                if i < len( args ):
                    kwargs.setdefault( name, args[ i ] )
            for name, val in fields:
                default = val
                if isinstance( val, ( TensorField, property ) ):
                    default = None
                setattr( self, name, kwargs.get( name, default ) )
        cls.__init__ = __init__

    # --- properties from axes (dim, nb_points, ...) --------------------------
    for axis_name in axis_names:
        if axis_name not in vars( cls ):
            def axis_count( self ):
                for name, field in fields:
                    if isinstance( field, TensorField ) and axis_name in field.axis_names:
                        v = self.__dict__.get( f'_{ name }' )
                        if v is not None:
                            return v.shape[ field.axis_names.index( axis_name ) ]
            setattr( cls, axis_name, property( axis_count ) )

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


# def _setup_distribution_class( cls, stop_classes, is_batch = False ):
#     """
#     Auto-generates the following if absent from the class:
#       - __init__
#       - one property per unique axis name across all TensorFields
#       - dim = 1          (when the class name ends with '1d')
#       - is_a_1d_version  (idem, mainly for batch classes)
#       - _nd_positions    (when a TensorField 'positions' is (re)defined in cls)
#       - batch_version    (only for Distribution subclasses, not BatchOfDistributions)
#       - tensor_list      (summary of all the tensors in TensorField)
#     """

#     all_fields = _collect_attributes( cls, stop_classes )
#     if not all_fields:
#         return

#     scalar_params = _collect_scalar_params( cls, stop_classes )
#     is_1d = cls.__name__.endswith( '1d' )

#     # --- axis-name properties -----------------------------------------------
#     # For each unique axis name, generate a property that returns
#     # field.shape[axis_idx] for the first non-None field carrying that axis.
#     axis_sources: dict[ str, list ] = {}
#     for field_name, field in all_fields.items():
#         for axis_idx, axis_name in enumerate( field.axis_names ):
#             axis_sources.setdefault( axis_name, [] ).append( ( field_name, axis_idx ) )

#     for axis_name, sources in axis_sources.items():
#         if axis_name not in vars( cls ):
#             def _make_prop( src = sources ):
#                 def prop( self ):
#                     for fn, ai in src:
#                         v = self.__dict__.get( f'_{ fn }' )
#                         if v is not None:
#                             return v.shape[ ai ]
#                     return None
#                 return property( prop )
#             setattr( cls, axis_name, _make_prop() )

#     # --- dim = 1 for classes whose name ends with '1d' ----------------------
#     if is_1d and 'dim' not in vars( cls ):
#         cls.dim = property( lambda self: 1 )

#     # --- _nd_positions: auto if 'positions' is (re)defined in cls -----------
#     own_fields = { n: f for n, f in vars( cls ).items() if isinstance( f, TensorField ) }
#     if 'positions' in own_fields and '_nd_positions' not in vars( cls ):
#         pos_rank  = own_fields[ 'positions' ].rank
#         # batch adds one dimension -> threshold 3; single -> threshold 2
#         threshold = 3 if is_batch else 2
#         if pos_rank < threshold:
#             def _nd_positions( self ):
#                 assert self.positions is not None
#                 return self.positions[ ..., None ]
#         else:
#             def _nd_positions( self ):
#                 assert self.positions is not None
#                 return self.positions
#         cls._nd_positions = _nd_positions

#     # --- tensor_list -----------
#     if 'tensor_list':
#         def tensor_list( self, with_names = False ):
#             res = []
#             name_indices = {}
#             for klass in reversed( type( self ).__mro__ ):
#                 if klass in { object }:
#                     continue
#                 for name, val in vars( klass ).items():
#                     if isinstance( val, TensorField ):
#                         # print( " ----------------from_a_dim_reduction", self, name, val.from_a_dim_reduction )
#                         tensor = getattr( self, name )
#                         if val.from_a_dim_reduction:
#                             tensor = tensor[ :, :, None ]
#                         if with_names:
#                             tensor = ( name, tensor )
#                         if name in name_indices:
#                             res[ name_indices[ name ] ] = tensor
#                         else:
#                             name_indices[ name ] = len( res )
#                             res.append( tensor )
#             return res

#         cls.tensor_list = tensor_list

#     # --- batch_version (only for non-batch Distribution subclasses) ---------
#     if is_batch or 'batch_version' in vars( cls ):
#         return

#     cls._batch_type  = None
#     batch_class_name = 'BatchOf' + cls.__name__
#     cls_module       = cls.__module__

#     def _get_batch_class( _cls = cls, _bcn = batch_class_name, _mod = cls_module ):
#         BatchCls = _cls._batch_type
#         if BatchCls is None:
#             import importlib
#             pkg = '.'.join( _mod.split( '.' )[ :-1 ] )
#             try:
#                 BatchCls = getattr( importlib.import_module( '.' + _bcn, package = pkg ), _bcn )
#             except ImportError:
#                 raise RuntimeError( "please create the batch type (as in __init__.py, in lines like BatchOfPiecewiseConstantImage = generate_batch_version_of( PiecewiseConstantImage ))" )
#             _cls._batch_type = BatchCls
#         return BatchCls

#     cls.batch_class = staticmethod( _get_batch_class )

#     def batch_version( self, batch_size ):
#         BatchCls = type( self ).batch_class()
#         kw = {}
#         for name, field in all_fields.items():
#             v = self.__dict__.get( f'_{ name }' )
#             kw[ name ] = field.expand_for_batch( v, batch_size ) if v is not None else None
#         for name in scalar_params:
#             kw[ name ] = getattr( self, name )
#         return BatchCls( **kw )

#     cls.batch_version = batch_version


# def generate_batch_of_distribution_methods( source_cls: type ) -> 'Callable[[type[_T]], type[_T]]':
#     """
#     Decorator factory that generates a BatchOf distribution class from the
#     corresponding non-batch Distribution class.

#     For each TensorField in source_cls (and its parents up to Distribution),
#     a new TensorField with "batch_size" prepended to the axis names is created
#     on the decorated class (unless already explicitly defined there).
#     Scalar parameters (e.g. x0, x1) are also copied from source_cls.

#     Usage::

#         @generate_batch_of_distribution_methods(SumOfWeightedDiracs)
#         class BatchOfSumOfWeightedDiracs(BatchOfDistributions):
#             nb_diracs: int

#             def default_weights(self, batch_version): ...
#     """
#     def decorator( cls: type[ _T ] ) -> type[ _T ]:
#         from .BatchOfDistributions import BatchOfDistributions
#         from .Distribution import Distribution

#         src_stop    = { Distribution, object }
#         src_fields  = _collect_attributes( source_cls, src_stop )
#         src_scalars = _collect_scalar_params( source_cls, src_stop )

#         # Add batch TensorFields for each source field not already defined in cls
#         for field_name, field in src_fields.items():
#             if field_name not in vars( cls ):
#                 batch_field      = TensorField( "batch_size", *field.axis_names )

#                 batch_field.from_a_dim_reduction = field.from_a_dim_reduction
#                 batch_field.name = field_name

#                 setattr( cls, field_name, batch_field )

#         # Copy scalar params (e.g. x0, x1) not already defined in cls
#         for param_name, param_val in src_scalars.items():
#             if param_name not in vars( cls ):
#                 setattr( cls, param_name, param_val )

#         # Auto-wrap default_* methods from source: if the result has rank one
#         # less than expected (missing batch_size dim), repeat it.
#         for field_name, field in src_fields.items():
#             default_name = f'default_{ field_name }'
#             src_default  = getattr( source_cls, default_name, None )
#             if src_default is not None and default_name not in vars( cls ):
#                 expected_rank = field.rank + 1  # source rank + 1 for batch_size
#                 def _wrap( src_fn = src_default, exp_rank = expected_rank ):
#                     def batch_default( self ):
#                         sig = signature( src_fn )
#                         if len( sig.parameters ) == 2:
#                             return src_fn( self, True )
#                         result = src_fn( self )
#                         if result is None or result.ndim == exp_rank:
#                             return result
#                         return driver.repeat(
#                             result[ None ],
#                             [ self.batch_size ] + [ 1 ] * ( exp_rank - 1 ),
#                         )
#                     return batch_default
#                 setattr( cls, default_name, _wrap() )

#         _setup_distribution_class(
#             cls,
#             stop_classes = { BatchOfDistributions, object },
#             is_batch     = True,
#         )
#         return cls

#     return decorator



# def generate_1d_version_of( source_cls: type[ _T ] ) -> type:
#     """
#     Create a 1d variant of a Distribution class by removing the ``"dim"`` axis
#     from any TensorField that carries it.

#     The generated class inherits from source_cls, overrides the affected fields,
#     and is named ``<SourceClass>1d``.

#     Example::

#         SumOfWeightedDiracs1d = generate_1d_version_of( SumOfWeightedDiracs )
#         # equivalent to:
#         # @generate_distribution_methods
#         # class SumOfWeightedDiracs1d( SumOfWeightedDiracs ):
#         #     positions = TensorField( "nb_diracs" )
#     """
#     from .Distribution import Distribution

#     src_fields = _collect_attributes( source_cls, { Distribution, object } )
#     namespace  = {}
#     for field_name, field in src_fields.items():
#         if 'dim' in field.axis_names:
#             new_axes = tuple( a for a in field.axis_names if a != 'dim' )
#             new_field = TensorField( *new_axes )

#             new_field.from_a_dim_reduction = True
#             new_field.name = field_name

#             namespace[ field_name ] = new_field

#     cls = type( source_cls.__name__ + '1d', ( source_cls, ), namespace )
#     _setup_distribution_class( cls, stop_classes = { Distribution, object }, is_batch = False )

#     # --- is_a_1d_version = True --------------------------------------------
#     if 'is_a_1d_version' not in vars( cls ):
#         cls.is_a_1d_version = property( lambda self: True )

#     return cls


# def generate_batch_version_of( source_cls: type[ _T ], *, parent: type | None = None, annotations: dict | None = None, **methods ) -> type:
#     """
#     Programmatically create a BatchOf<SourceClass> without a separate file.

#     The generated class is registered so that ``source_cls.batch_version()``
#     finds it without importlib.

#     Parameters
#     ----------
#     source_cls  : the non-batch Distribution subclass to batch-ify
#     parent      : base class for the generated batch class
#                   (defaults to BatchOfDistributions)
#     annotations : dict of bare annotations for PyLance visibility
#                   e.g. ``{'nb_diracs': int}``
#     **methods   : any extra methods / attributes to inject
#                   (e.g. ``default_weights=...``, ``__init__=...``)

#     Example::

#         BatchOfSumOfWeightedDiracs = generate_batch_version_of(
#             SumOfWeightedDiracs,
#             annotations    = {'nb_diracs': int},
#             default_weights = lambda self: ...,
#         )
#     """
#     from .BatchOfDistributions import BatchOfDistributions
#     from .Distribution import Distribution

#     if parent is None:
#         # Inherit from the batch version of the first source ancestor that has one
#         for ancestor in source_cls.__mro__[ 1: ]:
#             if ancestor in ( Distribution, object ):
#                 break
#             bt = getattr( ancestor, '_batch_type', None )
#             if bt is not None:
#                 parent = bt
#                 break
#         else:
#             parent = BatchOfDistributions
#         if parent is None:
#             parent = BatchOfDistributions

#     batch_name = 'BatchOf' + source_cls.__name__
#     namespace  = dict( methods )
#     if annotations:
#         namespace[ '__annotations__' ] = annotations

#     cls = type( batch_name, ( parent, ), namespace )
#     cls = generate_batch_of_distribution_methods( source_cls )( cls )  # type: ignore[assignment]
#     source_cls._batch_type = cls  # type: ignore[attr-defined]
#     return cls
