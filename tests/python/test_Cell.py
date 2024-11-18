import matplotlib.pyplot as plt
from munch import Munch
from sdot import Cell
import pytest
import numpy

def check_cut_seq( ndim, cuts ):
    cell = Cell( ndim = ndim )
    for num_cut, cut in enumerate( map( Munch, cuts ) ):
        # do the cut
        if 'dir' in cut:
            cell.cut( cut.dir, cut.off )

        # check scalar products
        for coords, refs in zip( cell.vertex_coords_td, cell.vertex_refs_td ):
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
            assert pytest.approx( numpy.sort( cell.vertex_coords_td @ cell.base, axis = 0 ), abs = 1e-6 ) == numpy.sort( cut.coords, axis = 0 )
        if 'faces' in cut:
            faces = []

            def on_2_rays( cut_refs, vertex_indices ):
                # normalize
                refs = list( list( x ) for x in cell.vertex_refs[ vertex_indices ] )
                if str( refs[ -1 ] ) < str( refs[ 1 ] ):
                    refs = reversed( refs )
                    refs = refs[ -1: ] + refs[ :-1 ]
                faces.append( ( "2_rays", cut_refs, refs ) )
            
            def on_closed( cut_refs, vertex_indices ):
                # normalize 
                refs = list( list( x ) for x in cell.vertex_refs[ vertex_indices ] )
                amin = min( range( len( refs ) ), key = lambda x: refs[ x ] )
                refs = list( refs[ amin: ] ) + list( refs[ :amin ] )
                if str( refs[ -1 ] ) < str( refs[ 1 ] ):
                    refs = reversed( refs )
                    refs = refs[ -1: ] + refs[ :-1 ]
                faces.append( ( "closed", cut_refs, refs ) )
            
            def on_1_ray( cut_refs, ray_refs ):
                faces.append( ( "1_ray", cut_refs, ray_refs ) )
            
            def on_free( cut_refs ):
                faces.append( ( "free", cut_refs ) )

            cell.for_each_face( on_closed, on_2_rays, on_1_ray, on_free )

            assert faces == cut.faces


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

def test_faces():
    check_cut_seq( 2, [
        { 'dir': [ +1.0, +0.0 ], 'off': +1.0, 'faces': [ ('1_ray', [], [ 0 ] ) ] }, # single face
        { 'dir': [ -1.0, +0.0 ], 'off': +1.0, 'faces': [ ('1_ray', [], [ 0 ] ), ( '1_ray', [], [ 1 ] ) ] }, # parallel faces
        { 'dir': [ +0.0, +1.0 ], 'off': +1.0, 'faces': [ ('2_rays', [], [ [ 1, 2 ], [ 0, 2 ] ] ) ] }, # 2 rays (not yet a closed loop)
        { 'dir': [ +0.0, -1.0 ], 'off': +1.0, 'faces': [ ('closed', [], [ [ 0, 2 ], [ 0, 3 ], [ 1, 3 ], [ 1, 2 ] ] ) ] }, # closed loop
    ] )


# cell = Cell( ndim = 2 )

# cell.cut( [ -1,  0 ], 0 )
# cell.cut( [  0, -1 ], 0 )
# cell.cut( [ +1, +1 ], 1 )



# print( cell.integral() )

from sdot import Expr
ima = [ [ 10, 1 ], [ 2, 3 ] ]
img = Expr.img_interpolation( ima, interpolation_order = 0 )
print( img[ 0.4, 0.4 ] )
print( img[ 0.54, 0.4 ] )

