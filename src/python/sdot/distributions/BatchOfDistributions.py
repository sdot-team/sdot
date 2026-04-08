from ..driver import driver
from typing import Self


class BatchOfDistributions:
    """
    Base class for all batch distributions.

    Subclasses should be decorated with @generate_distribution_methods to
    automatically receive __init__, _nd_positions, dim, is_a_1d_version, and one
    property per TensorField axis name.
    """

    MultidimensionalVersion    : type
    UnidimensionalVersion      : type
    BaseVersion                : type
    batch_size                 : int
    dim                        : int

    def multidimensional_version( self, *_ ) -> Self: ...
    def batch_item( self, batch_index: int ) -> Self: ...

    @property
    def is_an_unidimensional_verion( self ) -> bool:
        return self.__class__.__name__.endswith( "1d" )

    @property
    def is_a_1d_version( self ) -> bool:
        """ true if comes from a multidimensional version (meaning that we can call .multidimensionnal_version()) """
        return False
