from sdot.object_with_tensors import object_with_tensors, TensorField
# from .driver import driver, Return, Mutable, Tensor, CtInt
from .Bsp import Bsp


@object_with_tensors
class PowerDiagram:
    """

    """

    positions = TensorField( "nb_vertices", "dim" ) # vertex index ->
    weights = TensorField( "nb_vertices" ) # vertex index ->

    def __init__( self, positions, weights = None ):
        self.positions = positions
        self.weights = weights
        self.bsp = Bsp( positions, weights )

