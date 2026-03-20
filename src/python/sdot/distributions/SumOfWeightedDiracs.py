from .distribution_methods import TensorField, generate_distribution_methods
from .Distribution import Distribution
from ..driver import driver


@generate_distribution_methods
class SumOfWeightedDiracs( Distribution ):
    """
    positions : tensor[ nb_diracs, dim ]
    weights   : tensor[ nb_diracs ]         (ones by default)
    """

    positions = TensorField( "nb_diracs", "dim" )
    weights   = TensorField( "nb_diracs" )
    nb_diracs : int

    def default_weights( self ):
        return driver.ones( ( self.nb_diracs, ) ) if self.nb_diracs is not None else None

