from .distributions.normalized_distribution import normalized_distribution
from .distributions.Distribution import Distribution
from .distributions.SumOfDiracs import SumOfDiracs

from .space_subsets.UnitSquare import UnitSquare

from .PowerDiagram import PowerDiagram

import numpy as np

class SdotSolver:
    """
        Class to find a transport plan between a sum of dirac and a density.

    """
    def __init__( self, source_measure, target_measure = None, norm = 2 ):
        # internal attributes
        self._source_density = None
        self._target_density = None

        # the PowerDiagram used for the computations stores the acceleration structure with the positions and the weights
        self.power_diagram = PowerDiagram()

        # stopping criteria
        self.max_mass_error_ratio_target = None
        self.max_mass_error_target = None

        # solver parameters
        self.max_nb_newton_iterations = 500
        self.max_lg2_relaxation = 20

        # error history
        self.max_mass_error_ratio_history = []
        self.norm_2_mass_error_history = []
        self.max_mass_error_history = []
        self.raise_if_error = True
        self.log_items = []
        self.status = 'none'

        # properties and attributes from ctor arguments
        self.source_density = source_measure
        self.target_density = target_measure
        self.norm = norm

    def solve( self ):
        return self.newton_solve( )

    def newton_solve( self, relaxation_coefficient = 1.0 ):
        # first system
        m_rows, m_cols, m_vals, v_vals, n2_err, error_code = self.newton_system()
        if error_code:
            return self._set_status( 'void cells before the first iteration' )

        # check convergence
        if self._convergence_test( v_vals, n2_err ):
            return self._set_status( 'ok' )

        # newton iteration
        for _ in range( self.max_nb_newton_iterations ):
            # linear solve
            mw = self._system_solve( m_rows, m_cols, m_vals, v_vals )
            dw = relaxation_coefficient * mw

            # update the weights
            self.power_diagram._weights -= dw
            self.power_diagram._weight_have_been_modified() # TODO: keep the same acceleration structure

            # loop if void cell
            for _ in range( self.max_lg2_relaxation ):
                # solve again the linear system
                m_rows, m_cols, m_vals, v_vals, n2_err, error_code = self.newton_system()
                if error_code == 0:
                    break

                # smaller weights update
                dw /= 2
                self.power_diagram._weights += dw
                self.power_diagram._weight_have_been_modified() # TODO: keep the same acceleration structure
            else:
                return self._set_status( 'max_lg2_relaxation reached' )

            # check convergence
            if self._convergence_test( v_vals, n2_err ):
                break
        else:
            return self._set_status( 'max_nb_newton_iterations reached' )

        return self._set_status( 'ok' )
                

    def _convergence_test( self, v_vals, n2_err ):
        self.max_mass_error_ratio_history.append( np.max( v_vals ) * self.nb_unknowns )
        self.norm_2_mass_error_history.append( np.linalg.norm( v_vals ) )
        self.max_mass_error_history.append( np.max( v_vals ) )

        if 'max_mass_error_ratio' in self.log_items:
            print( 'max_mass_error_ratio:', self.max_mass_error_ratio_history[ -1 ] )
        if 'norm_2_mass_error' in self.log_items:
            print( 'norm_2_mass_error:', self.norm_2_mass_error_history[ -1 ] )
        if 'max_mass_error' in self.log_items:
            print( 'max_mass_error:', self.max_mass_error_history[ -1 ] )

        ok = True
        tested = False
        if self.max_mass_error_ratio_target is not None:
            ok = ok and self.max_mass_error_ratio_history[ -1 ] <= self.max_mass_error_ratio_target
            tested = True
        if self.max_mass_error_target is not None:
            ok = ok and self.max_mass_error_history[ -1 ] <= self.max_mass_error_target
            tested = True

        if tested:
            return ok
        return self.max_mass_error_ratio_history[ -1 ] <= 1e-4

    def _set_status( self, status ):
        self.status = status
        if self.raise_if_error and status != 'ok':
            raise SdotSolverError( status )
        return self.status

    # def measure_ratio_error( self, norm = 'inf' ):
    #     f = np.full( [ self.nb_unknowns ], 1 / self.nb_unknowns )
    #     m = self.power_diagram.measures()
    #     if norm == 'inf':
    #         return np.max( np.abs( m - f ) / f )
    #     if norm == 2:
    #         return np.linalg.norm( ( m - f ) / f )
    #     raise RuntimeError( "TODO" )

    # def n2_error( self ):
    #     f = np.full( [ self.nb_unknowns ], 1 / self.nb_unknowns )
    #     m = self.power_diagram.measures()
    #     return np.linalg.norm( m - f )

    def newton_system( self ):
        self._check_inputs()

        # data from the power diagram
        m_rows, m_cols, m_vals, v_vals, n2_err, error_code = self.power_diagram.dmeasures_dweights()

        # force a delta weight to be equal to 0 (the first one...)
        if error_code == 0:
            for i in range( v_vals.size ):
                if m_rows[ i ] == m_cols[ i ]:
                    m_vals[ i ] *= 2

        return m_rows, m_cols, m_vals, v_vals, n2_err, error_code
    
    def _system_solve( self, m_rows, m_cols, m_vals, v_vals ):
        from scipy.sparse.linalg import spsolve
        from scipy.sparse import coo_matrix

        M = coo_matrix( ( m_vals, ( m_rows, m_cols ) ), shape = ( v_vals.size, v_vals.size ) ).tocsr()
        return spsolve( M, v_vals )

    def plot_in_pyplot( self, plt ):
        generic_dist, dirac_dist, dirac_is_tgt = self._check_inputs()
        self.power_diagram.plot_in_pyplot( plt )

    @property
    def nb_unknowns( self ):
        self._check_inputs()
        return self.power_diagram._weights.size

    @property
    def target_density( self ):
        return self._target_density

    @target_density.setter
    def target_density( self, distribution ):
        self._target_density = self._normalize_and_use_density( distribution )

    @property
    def source_density( self ):
        return self._source_density

    @source_density.setter
    def source_density( self, distribution ):
        self._source_density = self._normalize_and_use_density( distribution )

    @property
    def ndim( self ):
        for dist in [ self._source_density, self._target_density ]:
            p = dist.ndim
            if p is not None:
                return p
        return None

    def _normalize_and_use_density( self, distribution ):
        if distribution is None:
            return distribution

        # check type
        if not isinstance( distribution, Distribution ):
            # for this class, we consider that as np.ndarray represents a sum of uniformly weighted diracs
            if isinstance( distribution, np.ndarray ):
                distribution = SumOfDiracs( distribution )
            else:
                distribution = normalized_distribution( distribution )

        # register power diagram data        
        if isinstance( distribution, SumOfDiracs ):
            self.power_diagram.positions = distribution.positions
        else:
            self.power_diagram.underlying_measure = distribution

        # return the normalized version
        return distribution


    def _check_inputs( self ):
        """ return ( generic_dist, dirac_dist, dirac_is_target ) """

        if self._target_density is None:
            self.target_density = UnitSquare()

        # check distribution compatibility
        sd = isinstance( self._source_density, SumOfDiracs )
        td = isinstance( self._target_density, SumOfDiracs )
        if sd + td == 0:
            raise RuntimeError( "SdotSolver expects at least one distribution to be a SumOfDiracs" )
        if sd + td == 2:
            raise RuntimeError( "SdotSolver expects at most one distribution to be a SumOfDiracs" )

        # distributions in the expected ordering
        if td:
            generic_dist, dirac_dist, dirac_is_tgt = self._source_density, self._target_density, True
        else:
            generic_dist, dirac_dist, dirac_is_tgt = self._target_density, self._source_density, False

        # output
        return generic_dist, dirac_dist, dirac_is_tgt
    

class SdotSolverError( RuntimeError ):
    pass