from sdot.object_with_tensors import object_with_tensors, TensorField, CtKnown
# from .driver import driver, Workspace, Return, Tensor, CtInt, Dyn
# from .distributions.Distribution import Distribution
# from .Cell import BatchOfCell
# from .Plotter import Plotter
# from .Bsp import Bsp

from typing import TYPE_CHECKING
# if TYPE_CHECKING:


@object_with_tensors
class Norm2:
    """
    Distance norm

    Possibility to handle periodicity using transformations
    """

    transformations = TensorField( "nb_transformations", CtKnown( "dim + 1" ), CtKnown( "dim + 1" ) )

    if TYPE_CHECKING:
        nb_transformations : int
        dim : int

    def __init__( self ):
        self.transformations = None
