from .densities import SumOfDiracs
import numpy as np

class SdotSolver:
    """
        Find a transport plan between a sum of dirac and a () density 

        If norm == 2, `self.power_diagram` will contain the used instance of `PowerDiagram` 

    """
    def __init__( self, source_distribution, target_distribution, norm = 2 ):
        # internal attributes
        self._source_distribution = None
        self._target_distribution = None
        self._diracs_in_src = None        

        # properties and attributes from ctor arguments
        self.source_distribution = source_distribution
        self.target_distribution = target_distribution
        self.norm = norm

        # stopping criteria
        self.mass_error_ratio = None

        # 

    def solve( self ):
        print( self._source_distribution )
        print( self._target_distribution )

    @property
    def target_distribution( self ):
        return self._target_distribution

    @target_distribution.setter
    def target_distribution( self, distribution ):
        self._target_distribution = SdotSolver._normalize_distribution( distribution )

    @property
    def source_distribution( self ):
        return self._source_distribution

    @source_distribution.setter
    def source_distribution( self, distribution ):
        self._source_distribution = SdotSolver._normalize_distribution( distribution )

    @staticmethod
    def _normalize_distribution( distribution ):
        if isinstance( distribution, np.ndarray ):
            return SumOfDiracs( distribution )
        return distribution

    def _update_power_diagram( self ):
        # if self.source_distribution:
        pass

def optimal_transport_plan( source_distribution, target_distribution, norm = 2, mass_error_ratio = 1e-2 ):
    os = SdotSolver( source_distribution, target_distribution, norm )
    os.mass_error_ratio = mass_error_ratio
    return os.solve()
