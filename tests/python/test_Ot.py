from sdot import SdotSolver, SumOfDiracs, UnitBox, optimal_transport_plan
import matplotlib.pyplot as plt
import numpy as np
import pytest


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

positions = np.random.random( [ 40, 2 ] )
positions[ :, 0 ] *= 0.5

sp = SdotSolver( positions )
sp.display.max_dw = 1


# print( sum( sp.power_diagram.cell_integrals() ) )
sp.adjust_potentials()

# pl = optimal_transport_plan(
#     SumOfDiracs( positions ),
#     UnitBox()
# )

# pl.plot()

# ot = pysdot.OptimalTransport( p )
# ot.verbosity = 2
# ot.adjust_weights()
# ot.display_vtk( "yo.vtk" )

# m_rows, m_cols, m_vals, v_vals, n2_err, error_code = sd.newton_system()
# print( coo_matrix( ( m_vals, ( m_rows, m_cols ) ) ).toarray() )

# sd.log_items = [ 'nb_relaxation_divs', 'max_dw' ]
# sd.newton_solve( relaxation_coefficient = 1.0 )  

# sd.power_diagram.weights = ot.get_weights()
# print( sd.power_diagram.measures() )

# # tp.plot_in_vtk_output( vo )
# sd.plot_in_pyplot( plt )
