import numpy as np

class D2GTransportMap:
    """
        Discrete (diracs) to generic measure transport map (optimal or not)

        => properties return vectors (one value for each dirac)
    """
    def __init__( self, sdot_solver ):
        self.sdot_solver = sdot_solver

    @property
    def kantorovitch_potentials( self ):
        return self.sdot_solver.power_diagram.weights

    @property
    def brenier_potentials( self ):
        p = self.sdot_solver.dirac_positions
        return self.kantorovitch_potentials + np.linalg.norm( p )

    @property
    def barycenters( self ):
        return self.sdot_solver.power_diagram.cell_barycenters()
