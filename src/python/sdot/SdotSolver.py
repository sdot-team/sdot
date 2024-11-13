from .distributions.IndicatorFunction import IndicatorFunction
from .distributions.Distribution import Distribution
from .distributions.SumOfDiracs import SumOfDiracs

from .space_subsets.SpaceSubset import SpaceSubset

from .PowerDiagram import PowerDiagram

import numpy as np

class SdotSolver:
    """
        Class to find a transport plan between a sum of dirac and a "generic" distribution.


    """
    def __init__( self, source_distribution, target_distribution, norm = 2 ):
        # internal attributes
        self._source_distribution = None
        self._target_distribution = None

        # the PowerDiagram used for the computations stores the acceleration structure with the positions and the weights
        self.power_diagram = PowerDiagram()

        # default values
        self.mass_error_ratio = None

        # properties and attributes from ctor arguments
        self.source_distribution = source_distribution
        self.target_distribution = target_distribution
        self.norm = norm

    def solve( self ):
        generic_dist, dirac_dist, dirac_is_tgt = self._check_inputs()

        # print( self.power_diagram.measures() )

    def plot_in_pyplot( self, plt ):
        generic_dist, dirac_dist, dirac_is_tgt = self._check_inputs()
        self.power_diagram.plot_in_pyplot( plt )

    @property
    def target_distribution( self ):
        return self._target_distribution

    @target_distribution.setter
    def target_distribution( self, distribution ):
        self._target_distribution = self._normalize_and_use_distribution( distribution )

    @property
    def source_distribution( self ):
        return self._source_distribution

    @source_distribution.setter
    def source_distribution( self, distribution ):
        self._source_distribution = self._normalize_and_use_distribution( distribution )

    @property
    def ndim( self ):
        for dist in [ self._source_distribution, self._target_distribution ]:
            p = dist.ndim
            if p is not None:
                return p
        return None

    def _normalize_and_use_distribution( self, distribution ):
        if distribution is None:
            return distribution
        
        # check type
        if not isinstance( distribution, Distribution ):
            if isinstance( distribution, np.ndarray ):
                distribution = SumOfDiracs( distribution )
            elif isinstance( distribution, SpaceSubset ):
                distribution = IndicatorFunction( distribution )
            else:
                raise RuntimeError( f"Don't know how to transform a { type( distribution ) } to a Distribution" )

        # register power diagram data        
        if isinstance( distribution, SumOfDiracs ):
            self.power_diagram.positions = distribution.positions

        # return the normalized version
        return distribution


    def _check_inputs( self ):
        """ return ( generic_dist, dirac_dist, dirac_is_target ) """

        # check distribution compatibility
        sd = isinstance( self._source_distribution, SumOfDiracs )
        td = isinstance( self._target_distribution, SumOfDiracs )
        if sd + td == 0:
            raise RuntimeError( "SdotSolver expects at least one distribution to be a SumOfDiracs" )
        if sd + td == 2:
            raise RuntimeError( "SdotSolver expects at most one distribution to be a SumOfDiracs" )

        # distributions in the expected ordering
        if td:
            generic_dist, dirac_dist, dirac_is_tgt = self._source_distribution, self._target_distribution, True
        else:
            generic_dist, dirac_dist, dirac_is_tgt = self._target_distribution, self._source_distribution, False

        # update boundaries if possible
        if bnds := generic_dist.convex_boundaries( self.ndim ):
            self.power_diagram.boundaries = bnds

        # output
        return generic_dist, dirac_dist, dirac_is_tgt
    