from .aggregate import aggregate, Tensor, Return
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

    def __init__( self ):
        self.transformations = None
