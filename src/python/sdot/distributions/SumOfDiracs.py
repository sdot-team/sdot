from .Distribution import Distribution
from ..PoomVec import PoomVec

class SumOfDiracs( Distribution ):
    """
        sum of potentially weighted diracs


    """
    
    def __init__( self, positions, masses = None ) -> None:
        """
            By default, masses are equal to 1 / len( positions )
        """

        # init values
        self._positions = None

        # ctor arguments
        self.positions = positions

    @property
    def positions( self ):
        return self._positions

    @positions.setter
    def positions( self, values ):
        if values is None:
            return
        self._positions = PoomVec( values )
        
    @property
    def ndim( self ):
        return self._positions.shape[ 1 ]
