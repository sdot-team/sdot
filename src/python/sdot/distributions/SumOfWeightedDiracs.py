from ..aggregate import Tensor, aggregate
from .Distribution import Distribution
from ..drivers.driver import driver
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

    def __init__( self, positions, weights = None, mass = 1 ):
        self.positions = positions
        self.weights = weights
        self.mass = mass

    def normalized_version( self ):
        if self.weights is None:
            c = 1 if self.mass is None else self.mass
            c /= self.nb_diracs
            new_weights = c * driver.ones( self.nb_diracs )
            return SumOfWeightedDiracs( self.positions, new_weights, None )

        if self.mass is not None:
            s = self.weights.sum()
            if s == 0:
                raise RuntimeError( "null weights !" )
            new_weights = self.mass / s * self.weights
            return SumOfWeightedDiracs( self.positions, new_weights, None )

        return self
