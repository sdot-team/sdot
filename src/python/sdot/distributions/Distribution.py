from typing import Self

class Distribution:
    """
    Base class for all (non-batch) distributions.

    Subclasses should be decorated with @generate_distribution_methods to
    automatically receive __init__, batch_version, _nd_positions, dim,
    is_a_1d_version, and one property per TensorField axis name.
    """

    UnidimensionalBatchVersion : type
    MultidimensionalVersion    : type
    UnidimensionalVersion      : type
    BatchVersion               : type
    BaseVersion                : type
    batch_size                 : int
    dim                        : int

    def multidimensional_version( self, *_ ) -> Self: ...
    def unidimensional_version( self, *_ ) -> Self: ...
    def batch_version( self, *_ ) -> Self: ...

    @property
    def always_1d( self ) -> bool:
        return self.__class__.__name__.endswith( "1d" )

    @property
    def is_a_1d_version( self ) -> bool:
        """ true if comes from a multidimensional version (meaning that we can call .multidimensionnal_version()) """
        return False

    @property
    def tensors( self ) -> list:
        raise RuntimeError( f"To be redefined for { type( self ) }" )

    def normalized_version( self ):
        return self
