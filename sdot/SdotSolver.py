from .distributions.normalized_distribution import normalized_distribution
from .distributions.Distribution import Distribution
from .distributions.SumOfDiracs import SumOfDiracs
from .distributions.UnitBox import UnitBox

from .D2GTransportMap import D2GTransportMap
from .G2DTransportMap import G2DTransportMap
from .PowerDiagram import PowerDiagram

import numpy as np

class DisplayParameters:
    num_newton_iteration = 0
    nb_relaxation_steps = 1
    max_error_ratio = 2
    norm_2_error = 0
    max_error = 0
    max_dw = 0 # max delta weight

    def __init__( self, list = [ 'max_error_ratio' ] ):
        if list is not None:
            for n, attr in enumerate( list ):
                assert attr in self
                setattr( self, attr, n + 1 )

    def write( self, solver ): 
        l = [ 'max_error_ratio', 'max_error', 'norm_2_error', 'nb_relaxation_steps', 'max_dw' ]
        display_first_iteration = False
        displayed_something = False
        for attr in l:
            if getattr( self, attr ):
                if len( getattr( solver, attr + '_history' ) ):
                    value = getattr( solver, attr + '_history' )[ -1 ]
                    if isinstance( value, float ):
                        print( f'{ attr }: {value:.4e}', end=' ' )
                    else:
                        print( f'{ attr }: {value}', end=' ' )
                    displayed_something = True
                else:
                    display_first_iteration = True

        if displayed_something:
            if display_first_iteration:
                print( '(first iteration)', end=' ' )
            print()


class SdotSolver:
    """
        Class to find a transport plan between a sum of dirac and a density.

    """
    def __init__( self, source_measure, target_measure = None, norm = 2, display: DisplayParameters = None ):
        # internal attributes
        self._source_measure = None
        self._target_measure = None

        # type update
        if not isinstance( display, DisplayParameters ):
            display = DisplayParameters( display )

        # the PowerDiagram used for the computations stores the acceleration structure with the positions and the weights
        self.power_diagram = PowerDiagram()

        # stopping criteria
        self.max_error_ratio_target = None
        self.max_error_target = None

        # solver parameters
        self.max_nb_newton_iterations = 500
        self.max_nb_relaxation_steps = 20

        # error history
        self.nb_relaxation_steps_history = []
        self.max_error_ratio_history = []
        self.norm_2_error_history = []
        self.max_error_history = []
        self.max_dw_history = []

        self.raise_if_error = True
        self.display = display
        self.status = 'none'

        # properties and attributes from ctor arguments
        self.source_measure = source_measure
        self.target_measure = target_measure
        self.norm = norm

    def forward_map( self ):
        return D2GTransportMap()

    def backward_map( self ):
        return D2GTransportMap()

    def adjust_potentials( self ):
        return self.newton_solve()

    def newton_solve( self, relaxation_coefficient = 1.0 ):
        # first system
        m_rows, m_cols, m_vals, residual, error_code = self.newton_system()
        if error_code:
            return self._set_status( 'void cells before the first iteration' )

        # check convergence
        if self._convergence_test( residual, None ):
            return self._set_status( 'ok' )

        # newton iteration
        for _ in range( self.max_nb_newton_iterations ):
            # linear solve
            mw = self._system_solve( m_rows, m_cols, m_vals, residual )
            dw = relaxation_coefficient * mw

            # update the weights
            self.power_diagram._weights -= dw
            self.power_diagram._weight_have_been_modified() # TODO: keep the same acceleration structure

            # loop if void cell
            for nb_relax_divs in range( self.max_nb_relaxation_steps ):
                # solve again the linear system
                m_rows, m_cols, m_vals, residual, error_code = self.newton_system()

                if error_code == 0:
                    self.nb_relaxation_steps_history.append( nb_relax_divs )
                    break

                # smaller weights update
                dw /= 2
                self.power_diagram._weights += dw
                self.power_diagram._weight_have_been_modified() # TODO: keep the same acceleration structure
            else:
                return self._set_status( 'max_lg2_relaxation reached' )

            # check convergence
            if self._convergence_test( residual, mw ):
                break
        else:
            return self._set_status( 'max_nb_newton_iterations reached' )

        return self._set_status( 'ok' )
                

    def _convergence_test( self, residual, dw ):
        # update history
        self.max_error_ratio_history.append( np.max( residual ) * self.nb_unknowns )
        self.norm_2_error_history.append( np.linalg.norm( residual ) )
        self.max_error_history.append( np.max( residual ) )
        if dw is not None:
            self.max_dw_history.append( np.max( np.abs( dw ) ) )

        # display
        self.display.write( self )

        # test convergence
        ok = True
        tested = False
        if self.max_error_ratio_target is not None:
            ok = ok and self.max_error_ratio_history[ -1 ] <= self.max_error_ratio_target
            tested = True
        if self.max_error_target is not None:
            ok = ok and self.max_error_history[ -1 ] <= self.max_error_target
            tested = True

        if tested:
            return ok
        return self.max_error_ratio_history[ -1 ] <= 1e-4

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
        m_rows, m_cols, m_vals, v_vals, error_code = self.power_diagram.dintegrals_dweights()
        residual = v_vals - np.full( [ self.nb_unknowns ], 1 / self.nb_unknowns )

        # force a delta weight to be equal to 0 (the first one...)
        for i in range( m_rows.size ):
            if m_rows[ i ] == m_cols[ i ]:
                m_vals[ i ] *= 2
                break

        return m_rows, m_cols, m_vals, residual, error_code
    
    def _system_solve( self, m_rows, m_cols, m_vals, v_vals ):
        from scipy.sparse.linalg import spsolve
        from scipy.sparse import coo_matrix

        M = coo_matrix( ( m_vals, ( m_rows, m_cols ) ), shape = ( v_vals.size, v_vals.size ) ).tocsr()
        return spsolve( M, v_vals )

    def plot( self, plt = None ):
        if plt is None:
            import matplotlib.pyplot as plt
            self.plot( plt )
            plt.show()
            return

        self._check_inputs()
        self.power_diagram.plot( plt )

    @property
    def nb_unknowns( self ):
        self._check_inputs()
        return self.power_diagram._weights.size

    @property
    def target_measure( self ):
        return self._target_measure

    @target_measure.setter
    def target_measure( self, measure ):
        self._target_measure = self._normalize_and_use_measure( measure )

    @property
    def source_measure( self ):
        return self._source_measure

    @source_measure.setter
    def source_measure( self, measure ):
        self._source_measure = self._normalize_and_use_measure( measure )

    @property
    def ndim( self ):
        for dist in [ self._source_measure, self._target_measure ]:
            p = dist.ndim
            if p is not None:
                return p
        return None

    def _normalize_and_use_measure( self, distribution ):
        """ register distribution (use it as self.power_diagram.positions or self.power_diagram.underlying_measure) """
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

        if self._target_measure is None:
            self.target_measure = UnitBox()

        # check distribution compatibility
        sd = isinstance( self._source_measure, SumOfDiracs )
        td = isinstance( self._target_measure, SumOfDiracs )
        if sd + td == 0:
            raise RuntimeError( "SdotSolver expects at least one distribution to be a SumOfDiracs" )
        if sd + td == 2:
            raise RuntimeError( "SdotSolver expects at most one distribution to be a SumOfDiracs" )

        # distributions in the expected ordering
        if td:
            generic_dist, dirac_dist, dirac_is_tgt = self._source_measure, self._target_measure, True
        else:
            generic_dist, dirac_dist, dirac_is_tgt = self._target_measure, self._source_measure, False

        # output
        return generic_dist, dirac_dist, dirac_is_tgt
    

class SdotSolverError( RuntimeError ):
    pass
