from munch import Munch
from sdot import Cell
import pytest
import numpy

def check_cut_seq( ndim, cuts ):
    cell = Cell( ndim = ndim )
    for num_cut, cut in enumerate( map( Munch, cuts ) ):
        # do the cut
        cell.cut( cut.dir, cut.off )

        # check scalar products
        for coords, refs in zip( cell.vertex_coords, cell.vertex_refs ):
            for num_cut in refs:
                assert pytest.approx( numpy.dot( cell.base.T @ coords, cuts[ num_cut ][ 'dir' ] ) - cuts[ num_cut ][ 'off' ], abs = 1e-10 ) == 0

        # check expected data
        if 'true_dimensionality' in cut:
            assert cut.true_dimensionality == cell.true_dimensionality
        if 'bounded' in cut:
            assert cut.bounded == cell.bounded
        if 'empty' in cut:
            assert cut.empty == cell.empty

        if 'coords' in cut:
            assert pytest.approx( numpy.sort( cell.vertex_coords @ cell.base.T, axis = 0 ), abs = 1e-6 ) == numpy.sort( cut.coords, axis = 0 )


def test_3D():
    # wedge
    check_cut_seq( 3, [
        { 'dir': [ -0.5, -0.7, +0.0 ], 'off': +1.0, 'true_dimensionality': 1,  },
        { 'dir': [ -0.5, +0.8, +0.0 ], 'off': +1.0, 'true_dimensionality': 2,  },
        { 'dir': [ +1.0, +0.0, +0.0 ], 'off': +1.0, 'true_dimensionality': 2,  },
        { 'dir': [ +0.0, +0.0, +1.0 ], 'off': +1.0, 'true_dimensionality': 3, 'bounded': 0 },
        { 'dir': [ +0.0, +0.0, -1.0 ], 'off': +1.0, 'true_dimensionality': 3, 'bounded': 1, 'empty': 0, 'coords': [[-2,0,1],[1,1.875,1],[1,-2.14285714,1],[-2,0,-1],[1,1.875,-1],[1,-2.14285714,-1]] },
        { 'dir': [ +0.0, +0.0, +1.0 ], 'off': -2.0, 'empty': 1, 'coords': numpy.zeros( [ 0, 3 ] ) },
        { 'dir': [ +0.0, +0.0, +1.0 ], 'off': -1.0, 'empty': 1, 'coords': numpy.zeros( [ 0, 3 ] ) },
    ] )

    # fail before true_dimensionality == nb_dims
    check_cut_seq( 3, [
        { 'dir': [ +1.0, +0.0, +0.0 ], 'off': +1.0, 'empty': 0 },
        { 'dir': [ -1.0, +0.0, +0.0 ], 'off': -2.0, 'empty': 1 },
        { 'dir': [ +0.0, +1.0, +0.0 ], 'off': -1.0, 'empty': 1 },
    ] )

    # simple cube
    check_cut_seq( 3, [
        { 'dir': [ +1.0, +0.0, +0.0 ], 'off': +1.0, },
        { 'dir': [ +0.0, +1.0, +0.0 ], 'off': +1.0, },
        { 'dir': [ +0.0, +0.0, +1.0 ], 'off': +1.0, },
        { 'dir': [ -1.0, +0.0, +0.0 ], 'off': +1.0, },
        { 'dir': [ +0.0, -1.0, +0.0 ], 'off': +1.0, 'bounded': 0, },
        { 'dir': [ +0.0, +0.0, -1.0 ], 'off': +1.0, 'bounded': 1, 'empty': 0, 'coords': [[-1,-1,-1],[1,-1,-1],[-1,1,-1],[1,1,-1],[-1,-1,1],[1,-1,1],[-1,1,1],[1,1,1],] },
    ] )

# print( cell )

# vo = module.VtkOutput()
# cell.display_vtk( vo )
# vo.save( "pd.vtk" )
