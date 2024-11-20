from .CellFunction import CellFunction
import numpy as np

class G2DTransportMap:
    """
        Generic measure to Discrete (dirac) transport map (optimal or not)

        => properties return function of $x$ (position in the space)
    """
    def __init__( self, sdot_solver ):
        self.sdot_solver = sdot_solver

    @property
    def kantorovitch_potential( self ):
        def cell_function( cell, x ):
            return cell.weight + np.linalg.norm( x - cell.seed_position )**2
        return CellFunction( self.sdot_solver.power_diagram, cell_function )

    @property
    def brenier_potential( self ):
        def cell_function( cell, x ):
            # return cell.weight / 2 + np.linalg.norm( x - cell.seed_position )**2 - np.linalg.norm( x )**2
            return np.dot( x, cell.seed_position ) - ( np.linalg.norm( cell.seed_position )**2 - cell.weight ) / 2
        return CellFunction( self.sdot_solver.power_diagram, cell_function )

    @property
    def barycenters( self ):
        def cell_function( cell, x ):
            return cell.seed_position
        return CellFunction( self.sdot_solver.power_diagram, cell_function )
