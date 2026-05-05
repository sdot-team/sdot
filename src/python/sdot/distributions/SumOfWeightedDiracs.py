from ..aggregate import Tensor, aggregate
from .Distribution import Distribution
from typing import TYPE_CHECKING

@aggregate
class SumOfWeightedDiracs( Distribution ):
    """
    positions : tensor[ nb_diracs, dim ]
    weights   : tensor[ nb_diracs ] (ones by default)
    """

    positions : Tensor( "nb_diracs", "dim" )
    weights   : Tensor( "nb_diracs" )

    if TYPE_CHECKING:
        nb_diracs : int

    def normalized_version( self ):
        return self

