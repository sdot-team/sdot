"""`pysdot.OptimalTransport` on top of `sdot.OtProblem`: `adjust_weights` is one `solve` of sdot's Newton method, which starts from
the last solution ( or from the weights the caller set ). The weights are defined up to a constant, which sdot fixes differently
( `weights[ 0 ] == 0` ): the diagram is the same.
"""
from .radial_funcs import RadialFuncUnit, RadialFuncInBall, RadialFuncEntropy, sdot_radial_args
from .PowerDiagram import PowerDiagram, default_domain, _old_call_forms
import numpy as np


class BadInitialGuess( Exception ):
    def __init__( self, ot ):
        self.ot = ot


class OptimalTransport:
    def __init__( self, positions = None, weights = None, domain = None, masses = None, radial_func = RadialFuncUnit(),
                  obj_max_dw = 1e-8, obj_max_dm = 0, linear_solver = "Petsc", verbosity = 0 ):
        """the stopping criterion is the first `obj_max_*` that is not 0: `obj_max_dm` on the masses of the cells ( absolute ),
        `obj_max_dw` on the weights ( turned into a criterion on the normalized masses, sdot's: `tol = obj_max_dw` ).
        `linear_solver` is ignored: sdot has its own. The older call form `OptimalTransport( domain, radial_func )` is accepted."""
        positions, weights, domain, radial_func = _old_call_forms( positions, weights, domain, radial_func )
        self.pd = PowerDiagram( positions, weights, domain, radial_func )
        self.obj_max_dw = obj_max_dw
        self.obj_max_dm = obj_max_dm
        self.masses = masses

        self.linear_solver = linear_solver
        self.verbosity = verbosity
        self.max_iter = 1000
        self.delta_m = []
        self.delta_w = []

        self._problem = None
        self._weights_given = weights is not None
        self.plan = None                    # the last `sdot` plan

    # ---- inputs ------------------------------------------------------------------------------------------------
    def get_positions( self ):
        return self.pd.positions

    def set_positions( self, new_positions ):
        self.pd.set_positions( new_positions )

    def get_masses( self ):
        return self.masses

    def set_masses( self, new_masses ):
        self.masses = new_masses

    def get_domain( self ):
        return self.pd.get_domain()

    def set_domain( self, new_domain ):
        self.pd.set_domain( new_domain )
        self._problem = None

    def get_weights( self ):
        return self.pd.weights

    def set_weights( self, new_weights ):
        self.pd.set_weights( new_weights )
        self._weights_given = True

    def set_stopping_criterion( self, value, type = "max delta masses" ):
        """`type`: "max delta masses" ( `max | mass - target |` ) or "max delta weights" """
        self.obj_max_dw = 0
        self.obj_max_dm = 0
        if type == "max delta masses":
            self.obj_max_dm = value
        elif type == "max delta weights":
            self.obj_max_dw = value
        else:
            raise ValueError( "'{}' is not a known stopping criterion type".format( type ) )

    # ---- the solve -----------------------------------------------------------------------------------------------
    def adjust_weights( self, initial_weights = None, ret_if_err = False, relax = 1.0 ):
        """the weights for which each cell has its mass. Returns `True` when the solver did not converge ( pysdot's convention ),
        `False` otherwise. `relax` and `ret_if_err` are ignored: the Newton method of sdot damps its steps by itself."""
        import sdot
        assert self.obj_max_dw or self.obj_max_dm

        if initial_weights is not None:
            self.set_weights( initial_weights )
        pd = self.pd
        n, d = pd.positions.shape
        if pd.domain is None:
            pd.set_domain( default_domain( d ) )
        domain_mass = float( pd.domain.measure() )

        if self.masses is None:
            rf = pd.radial_func
            if isinstance( rf, RadialFuncInBall ):
                self.masses = np.ones( n ) * 1e-6
            elif isinstance( rf, ( RadialFuncUnit, RadialFuncEntropy ) ):
                self.masses = np.ones( n ) * domain_mass / n
            else:
                sdot_radial_args( rf )                    # ( raises for what sdot does not do )
        masses = np.asarray( self.masses, dtype = float ).reshape( -1 )
        total = float( masses.sum() )

        # the problem: absolute masses for a partial transport ( the rest of the domain goes to the sink ), normalized ones
        # otherwise ( a sum of masses that is not exactly the measure of the domain stays a full transport, as in pysdot )
        sink, radial = sdot_radial_args( pd.radial_func )
        partial = sink is not None
        target = pd.domain.to_sdot( target_mass = domain_mass if partial else 1.0 )
        source = sdot.SumOfDiracs( pd.positions, masses, target_mass = total if partial else 1.0 )
        if self._problem is None or self._problem.periodicity != pd.sdot_periodicity():
            self._problem = sdot.OtProblem( source, target, periodicity = pd.sdot_periodicity(), radial_func = radial )
        else:
            self._problem.source = source
            self._problem.target = target

        # the stopping criterion, in the units of the problem
        scale = 1.0 if partial else total
        tol = self.obj_max_dm / scale if self.obj_max_dm else self.obj_max_dw
        settings = dict( tol = tol, max_iter = self.max_iter, precision = "fp64", on_failure = "ignore", keep_weights = True )
        if self._weights_given and pd.weights is not None:
            w0 = np.asarray( pd.weights, dtype = float )
            if isinstance( pd.radial_func, RadialFuncEntropy ):
                w0 = w0 - self._entropy_shift( total, domain_mass )
            settings[ "weights0" ] = w0
        plan = self._problem.solve( sdot.Iterative( **settings ), verbose = self.verbosity > 1 )
        self.plan = plan

        w = np.asarray( plan.weights, dtype = float ).reshape( -1 )
        if isinstance( pd.radial_func, RadialFuncEntropy ):
            w = w + self._entropy_shift( total, domain_mass )
        pd.set_weights( w )
        self._weights_given = False

        hist = plan.history
        self.delta_m += [ float( h[ "max_abs_residual" ] ) * scale for h in hist if "max_abs_residual" in h ]
        ws = [ np.asarray( h[ "weights" ], dtype = float ) for h in hist if h.get( "weights" ) is not None ]
        self.delta_w += [ float( np.max( np.abs( b - a ) ) ) for a, b in zip( ws[ :-1 ], ws[ 1: ] ) ]
        if self.verbosity > 0:
            print( "sdot.pysdot: {} after {} iterations".format( "converged" if plan.converged else "NOT converged", plan.stats.get( "nb_iter" ) ) )
        return not plan.converged

    def _entropy_shift( self, total, domain_mass ):
        """sdot normalizes the density and the masses to 1: with `exp( ( w - r^2 ) / eps )`, the weights of pysdot ( absolute
        masses ) are those of sdot plus `eps log( total / domain_mass )`"""
        return self.pd.radial_func.eps * np.log( total / domain_mass )

    # ---- what is read ------------------------------------------------------------------------------------------
    def get_centroids( self ):
        return self.pd.centroids()

    def display_vtk( self, filename, points = False, centroids = False ):
        self.pd.display_vtk( filename, points, centroids )

    def display_asy( self, *args, **kwargs ):
        self.pd.display_asy( *args, **kwargs )

    def nb_diracs( self ):
        return self.pd.positions.shape[ 0 ]

    def dim( self ):
        return self.pd.positions.shape[ 1 ]

