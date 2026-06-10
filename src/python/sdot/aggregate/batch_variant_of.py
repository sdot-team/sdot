from ..util.get_all_annotations import get_all_annotations
from .aggregate import _setup_aggregate

from typing import TypeVar
_T = TypeVar( '_T' )


def batch_variant_of( base_cls ):
    """
    Class decorator that creates a batch variant of an @aggregate class.

    The decorated class gets:
      - tensor fields from base_cls with the batch axis (batch_size_<BaseName>) prepended
      - BatchItemVersion link back to base_cls, and base_cls.BatchVersion pointing here
      - BaseVersion link on both sides
      - all boilerplate from @aggregate (axis getters, __init__, __setattr__, batch_axes_dict)
      - staticmethods from base_cls not explicitly overridden

    Classmethods, properties and regular methods are intentionally NOT copied:
    each variant must define its own (with proper naming, e.g. measure vs measures).

    Usage::

        @batch_variant_of( Cell )
        class BatchOfCell:
            @property
            def measures( self ):
                ...
    """
    def decorator( cls: type[ _T ] ) -> type[ _T ]:
        annotations = get_all_annotations( base_cls )
        bn = f"batch_size_{ base_cls.__name__ }"

        # Annotations are copied unchanged (base, unbatched). The batch axis is no longer
        # baked into the tensor shapes here — it is `batch_axes` (below) and gets prepended
        # to each tensor as a leading dimension at analysis time (see CallArg_Aggregate).
        # A batch variant is thus just "the base type with batch_axes preset + nicer ctors".
        for name, annotation in annotations.items():
            if name not in cls.__annotations__:
                cls.__annotations__[ name ] = annotation

        cls.BaseVersion  = base_cls
        cls.batch_axes   = [ bn ]
        base_cls.BatchVersion = cls

        _setup_aggregate( cls )

        # copy only staticmethods; everything else must be explicitly redefined
        for attr_name, val in vars( base_cls ).items():
            if attr_name.startswith( '__' ):
                continue
            if isinstance( val, staticmethod ) and attr_name not in vars( cls ):
                setattr( cls, attr_name, val )

        return cls

    return decorator
