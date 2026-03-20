from typing import Any, Callable, TypeVar, overload
from ..driver import driver

_T = TypeVar( '_T' )



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
        self.axis_names = axis_names
        self.rank       = len( axis_names )
        self.name       = None  # filled by __set_name__

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


# ---------------------------------------------------------------------------
# Helpers used by @generate_distribution_methods
# ---------------------------------------------------------------------------

def _collect_fields( cls, stop_classes ):
    """ All TensorFields visible from cls (MRO), excluding stop_classes. """
    result = {}
    for klass in reversed( cls.__mro__ ):
        if klass in stop_classes:
            continue
        for name, val in vars( klass ).items():
            if isinstance( val, TensorField ):
                result[ name ] = val
    return result


def _collect_scalar_params( cls, stop_classes ):
    """ Non-TensorField, non-callable, non-dunder class attributes -> default values. """
    result = {}
    for klass in reversed( cls.__mro__ ):
        if klass in stop_classes:
            continue
        for name, val in vars( klass ).items():
            if (
                not name.startswith( '_' )
                and not isinstance( val, ( TensorField, property, classmethod, staticmethod ) )
                and not callable( val )
            ):
                result[ name ] = val
    return result


def _setup_distribution_class( cls, stop_classes, is_batch = False ):
    """
    Auto-generates the following if absent from the class:
      - __init__
      - one property per unique axis name across all TensorFields
      - dim = 1          (when the class name ends with '1d')
      - always_1d = True (idem, mainly for batch classes)
      - _nd_positions    (when a TensorField 'positions' is (re)defined in cls)
      - batch_version    (only for Distribution subclasses, not BatchOfDistributions)
    """

    all_fields = _collect_fields( cls, stop_classes )
    if not all_fields:
        return

    scalar_params = _collect_scalar_params( cls, stop_classes )
    is_1d = cls.__name__.endswith( '1d' )

    # --- __init__ -----------------------------------------------------------
    if '__init__' not in vars( cls ):
        field_names = list( all_fields.keys() )

        def __init__( self, *args, **kwargs ):
            for i, name in enumerate( field_names ):
                if i < len( args ):
                    kwargs.setdefault( name, args[ i ] )
            for name, default in scalar_params.items():
                setattr( self, name, kwargs.get( name, default ) )
            for name in field_names:
                setattr( self, name, kwargs.get( name ) )

        cls.__init__ = __init__

    # --- axis-name properties -----------------------------------------------
    # For each unique axis name, generate a property that returns
    # field.shape[axis_idx] for the first non-None field carrying that axis.
    axis_sources: dict[ str, list ] = {}
    for field_name, field in all_fields.items():
        for axis_idx, axis_name in enumerate( field.axis_names ):
            axis_sources.setdefault( axis_name, [] ).append( ( field_name, axis_idx ) )

    for axis_name, sources in axis_sources.items():
        if axis_name not in vars( cls ):
            def _make_prop( src = sources ):
                def prop( self ):
                    for fn, ai in src:
                        v = self.__dict__.get( f'_{ fn }' )
                        if v is not None:
                            return v.shape[ ai ]
                    return None
                return property( prop )
            setattr( cls, axis_name, _make_prop() )

    # --- dim = 1 for classes whose name ends with '1d' ----------------------
    if is_1d and 'dim' not in vars( cls ):
        cls.dim = property( lambda self: 1 )

    # --- always_1d = True (idem) --------------------------------------------
    if is_1d and 'always_1d' not in vars( cls ):
        cls.always_1d = property( lambda self: True )

    # --- _nd_positions: auto if 'positions' is (re)defined in cls -----------
    own_fields = { n: f for n, f in vars( cls ).items() if isinstance( f, TensorField ) }
    if 'positions' in own_fields and '_nd_positions' not in vars( cls ):
        pos_rank  = own_fields[ 'positions' ].rank
        # batch adds one dimension -> threshold 3; single -> threshold 2
        threshold = 3 if is_batch else 2
        if pos_rank < threshold:
            def _nd_positions( self ):
                assert self.positions is not None
                return self.positions[ ..., None ]
        else:
            def _nd_positions( self ):
                assert self.positions is not None
                return self.positions
        cls._nd_positions = _nd_positions

    # --- batch_version (only for non-batch Distribution subclasses) ---------
    if is_batch or 'batch_version' in vars( cls ):
        return

    cls._batch_type  = None
    batch_class_name = 'BatchOf' + cls.__name__
    cls_module       = cls.__module__

    def _get_batch_class( _cls = cls, _bcn = batch_class_name, _mod = cls_module ):
        BatchCls = _cls._batch_type
        if BatchCls is None:
            import importlib
            pkg      = '.'.join( _mod.split( '.' )[ :-1 ] )
            BatchCls = getattr( importlib.import_module( '.' + _bcn, package = pkg ), _bcn )
            _cls._batch_type = BatchCls
        return BatchCls

    cls.batch_class = staticmethod( _get_batch_class )

    def batch_version( self, batch_size ):
        BatchCls = type( self ).batch_class()
        kw = {}
        for name, field in all_fields.items():
            v = self.__dict__.get( f'_{ name }' )
            kw[ name ] = field.expand_for_batch( v, batch_size ) if v is not None else None
        for name in scalar_params:
            kw[ name ] = getattr( self, name )
        return BatchCls( **kw )

    cls.batch_version = batch_version


def generate_batch_of_distribution_methods( source_cls: type ) -> 'Callable[[type[_T]], type[_T]]':
    """
    Decorator factory that generates a BatchOf distribution class from the
    corresponding non-batch Distribution class.

    For each TensorField in source_cls (and its parents up to Distribution),
    a new TensorField with "batch_size" prepended to the axis names is created
    on the decorated class (unless already explicitly defined there).
    Scalar parameters (e.g. x0, x1) are also copied from source_cls.

    Usage::

        @generate_batch_of_distribution_methods(SumOfWeightedDiracs)
        class BatchOfSumOfWeightedDiracs(BatchOfDistributions):
            nb_diracs: int

            def default_weights(self): ...
    """
    def decorator( cls: type[ _T ] ) -> type[ _T ]:
        from .BatchOfDistributions import BatchOfDistributions
        from .Distribution import Distribution

        src_stop    = { Distribution, object }
        src_fields  = _collect_fields( source_cls, src_stop )
        src_scalars = _collect_scalar_params( source_cls, src_stop )

        # Add batch TensorFields for each source field not already defined in cls
        for field_name, field in src_fields.items():
            if field_name not in vars( cls ):
                batch_field      = TensorField( "batch_size", *field.axis_names )
                batch_field.name = field_name
                setattr( cls, field_name, batch_field )

        # Copy scalar params (e.g. x0, x1) not already defined in cls
        for param_name, param_val in src_scalars.items():
            if param_name not in vars( cls ):
                setattr( cls, param_name, param_val )

        # Auto-wrap default_* methods from source: if the result has rank one
        # less than expected (missing batch_size dim), repeat it.
        for field_name, field in src_fields.items():
            default_name = f'default_{ field_name }'
            src_default  = getattr( source_cls, default_name, None )
            if src_default is not None and default_name not in vars( cls ):
                expected_rank = field.rank + 1  # source rank + 1 for batch_size
                def _wrap( src_fn = src_default, exp_rank = expected_rank ):
                    def batch_default( self ):
                        result = src_fn( self )
                        if result is None or result.ndim == exp_rank:
                            return result
                        return driver.repeat(
                            result[ None ],
                            [ self.batch_size ] + [ 1 ] * ( exp_rank - 1 ),
                        )
                    return batch_default
                setattr( cls, default_name, _wrap() )

        _setup_distribution_class(
            cls,
            stop_classes = { BatchOfDistributions, object },
            is_batch     = True,
        )
        return cls

    return decorator


def generate_distribution_methods( cls: type[ _T ] ) -> type[ _T ]:
    """
    Class decorator that auto-generates boilerplate for Distribution and
    BatchOfDistributions subclasses based on their TensorField declarations.

    Fields with a ``default_<name>(self)`` method in the class will use that
    method when the field value is None.

    Usage::

        @generate_distribution_methods
        class MyDist(Distribution):
            positions = TensorField("nb_points", "dim")
            weights   = TensorField("nb_points")

            def default_weights(self):
                return driver.ones(self.nb_points)
    """
    from .BatchOfDistributions import BatchOfDistributions
    from .Distribution import Distribution

    if issubclass( cls, BatchOfDistributions ):
        _setup_distribution_class( cls, stop_classes = { BatchOfDistributions, object }, is_batch = True )
        return cls

    if issubclass( cls, Distribution ):
        _setup_distribution_class( cls, stop_classes = { Distribution, object }, is_batch = False )
        return cls

    raise TypeError( f"@generate_distribution_methods requires a Distribution or BatchOfDistributions subclass, got { cls }" )


def generate_1d_version_of( source_cls: type[ _T ] ) -> type:
    """
    Create a 1d variant of a Distribution class by removing the ``"dim"`` axis
    from any TensorField that carries it.

    The generated class inherits from source_cls, overrides the affected fields,
    and is named ``<SourceClass>1d``.

    Example::

        SumOfWeightedDiracs1d = generate_1d_version_of( SumOfWeightedDiracs )
        # equivalent to:
        # @generate_distribution_methods
        # class SumOfWeightedDiracs1d( SumOfWeightedDiracs ):
        #     positions = TensorField( "nb_diracs" )
    """
    from .Distribution import Distribution

    src_fields = _collect_fields( source_cls, { Distribution, object } )
    namespace  = {}
    for field_name, field in src_fields.items():
        if 'dim' in field.axis_names:
            new_axes       = tuple( a for a in field.axis_names if a != 'dim' )
            new_field      = TensorField( *new_axes )
            new_field.name = field_name
            namespace[ field_name ] = new_field

    cls = type( source_cls.__name__ + '1d', ( source_cls, ), namespace )
    _setup_distribution_class( cls, stop_classes = { Distribution, object }, is_batch = False )
    return cls


def generate_batch_version_of( source_cls: type[ _T ], *, parent: type | None = None, annotations: dict | None = None, **methods ) -> type:
    """
    Programmatically create a BatchOf<SourceClass> without a separate file.

    The generated class is registered so that ``source_cls.batch_version()``
    finds it without importlib.

    Parameters
    ----------
    source_cls  : the non-batch Distribution subclass to batch-ify
    parent      : base class for the generated batch class
                  (defaults to BatchOfDistributions)
    annotations : dict of bare annotations for PyLance visibility
                  e.g. ``{'nb_diracs': int}``
    **methods   : any extra methods / attributes to inject
                  (e.g. ``default_weights=...``, ``__init__=...``)

    Example::

        BatchOfSumOfWeightedDiracs = generate_batch_version_of(
            SumOfWeightedDiracs,
            annotations    = {'nb_diracs': int},
            default_weights = lambda self: ...,
        )
    """
    from .BatchOfDistributions import BatchOfDistributions
    from .Distribution import Distribution

    if parent is None:
        # Inherit from the batch version of the first source ancestor that has one
        for ancestor in source_cls.__mro__[ 1: ]:
            if ancestor in ( Distribution, object ):
                break
            bt = getattr( ancestor, '_batch_type', None )
            if bt is not None:
                parent = bt
                break
        else:
            parent = BatchOfDistributions
        if parent is None:
            parent = BatchOfDistributions

    batch_name = 'BatchOf' + source_cls.__name__
    namespace  = dict( methods )
    if annotations:
        namespace[ '__annotations__' ] = annotations

    cls = type( batch_name, ( parent, ), namespace )
    cls = generate_batch_of_distribution_methods( source_cls )( cls )  # type: ignore[assignment]
    source_cls._batch_type = cls  # type: ignore[attr-defined]
    return cls
