from .BatchOfSumOfWeighted1dDiracs import BatchOfSumOfWeighted1dDiracs
from .SumOfWeightedDiracs import SumOfWeightedDiracs
from ..driver import driver


class SumOf1dWeightedDiracs( SumOfWeightedDiracs ):
    """
    positions : tensor[ nb_diracs ]
    weights : tensor[ nb_diracs ]
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
        self._positions = driver.t1( value )

    @property
    def weights( self ):
        if self._weights is None:
            positions = self.positions
            if positions is None:
                return None
            return driver.ones( positions.shape )
        return self._weights

    @weights.setter
    def weights( self, value ):
        self._weights = driver.t1( value )

    @property
    def nb_diracs( self ):
        return self.positions.shape[ 0 ]

    @property
    def dim( self ):
        return 1

    def batch_version( self, batch_size ):
        p = self.positions
        if p is not None:
            p = driver.repeat( p[ None, : ], [ batch_size, 1 ] )
        w = self.weights
        if w is not None:
            w = driver.repeat( w[ None, : ], [ batch_size, 1 ] )
        return BatchOfSumOfWeighted1dDiracs( p, w )
