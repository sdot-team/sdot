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

c = Cell( ndim = 2 )

# c.cut( [ -1, 0 ], 0 )
# c.cut( [ 0, -1 ], 0 )
# c.cut( [ 1, 1 ], 1 )
# c.cut( [ 1, 0 ], 0.2 )


# c.cut( [ 0.0032212639998749903, -0.11510865119116831 ], -0.020422031329160022 )
# c.cut( [ -0.18679755260909214, -0.00886601053817937 ], -0.10675875690110406 )
# c.cut( [ 0.044522870461866426, -0.19306613091051228 ], -0.00019536387275698175 )
# c.cut( [ -0.08291155504751879, 0.18530991092253823 ], 0.013394261948515177 )
# c.cut( [ -0.13126000371532676, 0.16137033745074147 ], -0.02321846265965842 )
# c.cut( [ 0.16788490438253667, -0.15619032674672084 ], 0.09637571025369839 )
# c.cut( [ 0.18483658203340547, -0.15719319464581905 ], 0.11033972564677288 )
# c.cut( [ -0.05034494939375456, 0.29159158045711475 ], 0.08475712496836402 )
# c.cut( [ 0.13861671261726616, 0.27560331721312237 ], 0.20792562567902012)
# c.cut( [ -0.2936004248916124, 0.18374035125955934 ], -0.0852321307463734)
# c.cut( [ -0.33767997454121457, 0.1104191684028405 ], -0.1294632756214804)
# c.cut( [ -0.3569681279432535, -0.13456388358553806 ], -0.1944559164364274 )

# c.plot_in_pyplot( plt )

# plt.show()
