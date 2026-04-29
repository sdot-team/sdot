# from sdot.object_with_tensors._methods import TensorField, object_with_tensors
from .Distribution import Distribution
from ..driver import driver


# @object_with_tensors
class SumOfWeightedDiracs( Distribution ):
    """
    positions : tensor[ nb_diracs, dim ]
    weights   : tensor[ nb_diracs ] (ones by default)
    """

    # positions = TensorField( "nb_diracs", "dim" )
    # weights   = TensorField( "nb_diracs" )

    # nb_diracs : int

    # def __init__(self, positions = None, weights = None):
    #     self.positions = positions
    #     self.weights = weights

    # def default_weights( self ):
    #     return driver.ones( ( self.nb_diracs, ) ) if self.nb_diracs is not None else None

