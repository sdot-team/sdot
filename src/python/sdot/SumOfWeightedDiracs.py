from .Distribution import Distribution
from .driver import driver


class SumOfWeightedDiracs( Distribution ):
    """
    positions : tensor[ batch_size, nb_diracs, dim ] (even in 1d, where dim is expected to be equal to 1)
    """
    def __init__( self, positions = None, weights = None ):
        self._positions = None
        self._weights = None

        self.positions = positions
        self.weights = weights

    @property
    def positions( self ):
        return self._positions

    @positions.setter
    def positions( self, value ):
        self._positions = driver.t2( value )

    @property
    def weights( self ):
        if self._weights is None:
            positions = self.positions
            if positions is None:
                return None
            print( positions.shape[ : -1 ] )
            return driver.ones( positions.shape[ : -1 ] )
        return self._weights

    @weights.setter
    def weights( self, value ):
        self._weights = driver.t1( value )

    @property
    def batch_size( self ):
        return self.positions.shape[ 0 ]

    @property
    def nb_diracs( self ):
        return self.positions.shape[ 0 ]
