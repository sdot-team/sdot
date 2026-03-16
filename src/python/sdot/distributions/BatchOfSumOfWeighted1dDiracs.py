from .BatchOfSumOfWeightedDiracs import BatchOfSumOfWeightedDiracs
from ..driver import driver


class BatchOfSumOfWeighted1dDiracs( BatchOfSumOfWeightedDiracs ):
    """
    positions : tensor[ batch_size, nb_diracs ] (no need to add a rank in 1d)
    weights : tensor[ batch_size, nb_diracs ]
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
            return driver.ones( positions.shape )
        return self._weights

    @weights.setter
    def weights( self, value ):
        self._weights = driver.t2( value )

    @property
    def batch_size( self ) -> int:
        return self.positions.shape[ 0 ]

    @property
    def nb_diracs( self ) -> int:
        return self.positions.shape[ 1 ]

    @property
    def dim( self ) -> int:
        return 1

    @property
    def always_1d( self ) -> bool:
        return True

    def _nd_positions( self ):
        """ return a position tensor compatible with the sdot procedures """
        assert self.positions is not None
        return self.positions[ :, :, None ]
