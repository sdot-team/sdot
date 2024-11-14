from sdot import SdotSolver, SumOfDiracs, IndicatorFunction, UnitSquare, Cell
import matplotlib.pyplot as plt
import numpy as np
import pytest
import pysdot


# def test_SdotSolver():
#     p = np.random.random( [ 400, 2 ] )
#     p[ :, 0 ] *= 0.5

#     # ot = pysdot.OptimalTransport( p )
#     # ot.verbosity = 2
#     # ot.adjust_weights()
#     # ot.display_vtk( "yo.vtk" )

#     sd = SdotSolver( p )
#     sd.max_nb_newton_iterations = 15
    
#     # m_rows, m_cols, m_vals, v_vals, n2_err, error_code = sd.newton_system()
#     # print( coo_matrix( ( m_vals, ( m_rows, m_cols ) ) ).toarray() )

#     sd.log_items = [ 'nb_relaxation_divs', 'max_dw' ]
#     sd.newton_solve( relaxation_coefficient = 1.0 )  

#     # sd.power_diagram.weights = ot.get_weights()
#     # print( sd.power_diagram.measures() )

#     # # tp.plot_in_vtk_output( vo )
#     sd.plot_in_pyplot( plt )
#     plt.show()

# np.random.seed( 2 )

# test_SdotSolver()

# p = PoomVec( [1, 2] )
# print( p.dtype )
# print( p.shape )

# p = PoomVec( [[1, 2]] )
# print( p.dtype )
# print( p.shape )

# p = PoomVec( [[1, 2]] )
# print( p.dtype )
# print( p.shape )

cell = Cell( ndim = 3 )

# we create a triangle, infinitely extruded
cell.cut( [ -1,  0, 0 ], 0 )
cell.cut( [  0, -1, 0 ], 0 )
cell.cut( [ +1, +1, 0 ], 1 )

# naturally, there's no 3D vertex...
print( cell.nb_vertices ) # => 0

# and this cell is sill unbounded (in 3D)
print( cell.bounded ) # => False

# It's because we're in 2D
print( cell.true_dimensionality ) # => 2

# "td" is the shortcut for "true dimensionality".
# Method with with prefix return the information for the subspace defined by `cell.base`
print( cell.nb_vertices_td ) # => 3 (the 3 vertices of the triangle)

# we can get coordinates to represent these points in 3D
print( cell.vertex_coords_td @ cell.base ) # => [[0. 0. 0.] [1. 0. 0.] [0. 1. 0.]]

cell.plot( plt )
plt.show()
