from sdot import SdotSolver, SumOfDiracs, IndicatorFunction, UnitSquare
import matplotlib.pyplot as plt
import numpy as np
import pytest

def test_SdotSolver():
    sd = SdotSolver(
        source_distribution = SumOfDiracs( np.random.random( [ 40, 2 ] ) ),
        target_distribution = IndicatorFunction( UnitSquare() ),
    )

    # tp = sd.solve()  
    sd.solve()  

    # tp.plot_in_vtk_output( vo )
    sd.plot_in_pyplot( plt )
    plt.show()

np.random.seed( 2 )

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
