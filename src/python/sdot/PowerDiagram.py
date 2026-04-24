from sdot.object_with_tensors import object_with_tensors, TensorField
from .driver import driver, Return, Tensor, CtInt
from .distributions.Distribution import Distribution
from .Bsp import Bsp


@object_with_tensors
class PowerDiagram:
    """

    """

    positions = TensorField( "nb_vertices", "dim" ) # vertex index ->
    weights = TensorField( "nb_vertices" ) # vertex index ->
    bsp : Bsp

    def __init__( self, positions, weights = None ):
        self.positions = positions
        self.weights = weights

        self.bsp = Bsp.make_from( self.positions, self.weights )

    def newton_dir( self, g: Distribution ):
        """ On a la pos """
        res = driver.call( "newton_dir", "sdot/PowerDiagram/newton_dir.h", ct_dim = CtInt( self.dim ), res = Return( Tensor, [ ] ), pd = self )
        info( res )
