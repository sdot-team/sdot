from .aggregate import aggregate, Tensor
from .drivers.driver import driver
from typing import TYPE_CHECKING


@aggregate
class Norm2:
    """
    Distance norm

    Possibility to handle periodicity using transformations
    """

    transformations : Tensor( "nb_transformations", "dim + 1", "dim + 1", ct_axes = [ "dim" ] )

    if TYPE_CHECKING:
        nb_transformations : int
        dim : int

    def __init__( self, transformations = None ):
        if transformations is None:
            transformations = driver.empty( [ 0, 1, 1 ] )
        self.transformations = transformations
