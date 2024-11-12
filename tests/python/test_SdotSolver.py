from sdot import PowerDiagram, SdotSolver, set_auto_rebuild
import matplotlib.pyplot as plt
import numpy as np
import pytest

set_auto_rebuild( True )

def test_SdotSolver():
    # pd = PowerDiagram( positions = [ [ 0.25, 0.5 ], [ 0.75, 0.5 ] ] )
    pd = PowerDiagram( positions = np.random.random( [ 40, 2 ] ) )
    # pd.add_cube_boundaries()

    pd.plot_in_pyplot( plt )
    plt.show()
    

    # pd.display_vtk( vo )

np.random.seed(0)

test_SdotSolver()

# p = PoomVec( [1, 2] )
# print( p.dtype )
# print( p.shape )

# p = PoomVec( [[1, 2]] )
# print( p.dtype )
# print( p.shape )

# p = PoomVec( [[1, 2]] )
# print( p.dtype )
# print( p.shape )
