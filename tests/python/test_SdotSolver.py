from sdot import SdotSolver, SumOfDiracs, IndicatorFunction, UnitSquare
import matplotlib.pyplot as plt
import numpy as np
import pytest
import pysdot


def test_SdotSolver():
    p = np.random.random( [ 40, 2 ] )

    ot = pysdot.OptimalTransport( p )
    # print( ot.pd.integrals() )

    # mvs = ot.pd.der_integrals_wrt_weights()

    # from scipy.sparse import csr_matrix, coo_matrix
    # print( csr_matrix( ( mvs.m_values, mvs.m_columns, mvs.m_offsets ) ).toarray() )

    ot.verbosity = 2
    ot.adjust_weights()

    sd = SdotSolver( p )
    
    # m_rows, m_cols, m_vals, v_vals, n2_err, error_code = sd.newton_system()
    # print( coo_matrix( ( m_vals, ( m_rows, m_cols ) ) ).toarray() )

    sd.log_items = [ 'max_dw' ]
    sd.newton_solve( relaxation_coefficient = 1.0 )  

    # sd.power_diagram.weights = ot.get_weights()
    # print( sd.power_diagram.measures() )

    # # tp.plot_in_vtk_output( vo )
    # sd.plot_in_pyplot( plt )
    # plt.show()

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
