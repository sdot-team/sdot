import matplotlib.pyplot as plt
from sdot import PoomVec
import numpy as np
import pytest

# def test_PowerDiagram():
#     # pd = PowerDiagram( positions = [ [ 0.25, 0.5 ], [ 0.75, 0.5 ] ] )
#     pd = PowerDiagram( positions = np.random.random( [ 40, 2 ] ) )
#     pd.add_cube_boundaries()

#     # pd.for_each_cell( print )

#     # pd.plot_in_pyplot( plt )
#     # plt.show()
    

#     # pd.display_vtk( vo )
    
# test_PowerDiagram()

p = PoomVec( [ 1, 2, 3 ] )
print( p )
