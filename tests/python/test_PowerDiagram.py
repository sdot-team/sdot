from sdot import PowerDiagram, UnitSquare, IndicatorFunction
import matplotlib.pyplot as plt
import numpy as np
import pytest


def test_PowerDiagram():
    # pd = PowerDiagram( positions = [ [ 0.25, 0.5 ], [ 0.75, 0.5 ] ] )
    best_ptp = 100000000
    best_p = None
    for i in range( 3000 ):
        p = np.random.random( [ 30, 2 ] )
        pd = PowerDiagram( positions = p )

        pd.periodicity_transformations = [
            ( np.eye( 2 ), [ 0, 1 ] )
        ]

        ptp = np.ptp( pd.summary().vertex_coords[ :, 0 ] )
        if best_ptp > ptp:
            best_ptp = ptp
            best_p = p

    print( best_ptp )
    pd = PowerDiagram( positions = best_p )

    pd.periodicity_transformations = [
        ( np.eye( 2 ), [ 0, 1 ] )
    ]

    pd.plot()

test_PowerDiagram()

# p = PoomVec( [1, 2] )
# print( p.dtype )
# print( p.shape )

# p = PoomVec( [[1, 2]] )
# print( p.dtype )
# print( p.shape )

# p = PoomVec( [[1, 2]] )
# print( p.dtype )
# print( p.shape )
