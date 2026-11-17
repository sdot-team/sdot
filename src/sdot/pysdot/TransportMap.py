"""`pysdot.TransportMap` and `find_transport_map`, on top of `OptimalTransport`"""
from .OptimalTransport import OptimalTransport
from .PowerDiagram import default_domain
import numpy as np


class TransportMap:
    def __init__( self, positions, masses = None, initial_weights = None, domain = None ) -> None:
        positions = np.asarray( positions, dtype = float )
        if domain is None:
            domain = default_domain( positions.shape[ 1 ] )
        if masses is None:
            masses = np.ones( positions.shape[ 0 ] ) * domain.measure() / positions.shape[ 0 ]
        self._ot = OptimalTransport( positions, initial_weights, domain, masses )
        self.pd = self._ot.pd
        self.masses = masses

        self.obj_max_dw = 1e-6
        self.max_iter = 100
        self.verbosity = 2
        self.linear_solver = "Petsc"

    def optimize_weights( self ) -> None:
        self._ot.masses = self.masses
        self._ot.obj_max_dw = self.obj_max_dw
        self._ot.max_iter = self.max_iter
        self._ot.verbosity = self.verbosity
        return self._ot.adjust_weights()

    def show( self, arrows = False, **kwargs ):
        """the diagram in the sdot viewer ( a notebook, or the browser ), with the transport when `arrows`"""
        plan = self._ot.plan
        if plan is not None:
            return plan.show( transport = arrows )
        return self.pd.sdot_diagram().show()

    def write_vtk( self, filename, points = False, centroids = False ):
        self.pd.display_vtk( filename, points, centroids )


def find_transport_map( positions, masses = None, initial_weights = None, domain = None ):
    res = TransportMap( positions, masses, initial_weights, domain )
    res.optimize_weights()
    return res
