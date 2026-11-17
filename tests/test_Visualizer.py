from errand import test


if test( "shared primitives are in every frame, stored once" ):
    import numpy as np
    from sdot.viz.Visualizer import Visualizer
    square = np.array( [ [ 0, 0 ], [ 1, 0 ], [ 1, 1 ], [ 0, 1 ] ], float )
    v = Visualizer()
    v.add_points( square[ :2 ] )
    v.add_edges( square, closed = True, shared = True )                      # ( added in the first frame )
    v.new_frame()
    v.add_points( square )
    v.add_faces( square, [ [ 0, 1, 2, 3 ] ], shared = True )               # ( added in the second one )
    f0, f1 = v.frame( 0 ), v.frame( 1 )
    assert len( f0[ "points" ] ) == 2 and len( f1[ "points" ] ) == 4
    assert len( f0[ "edges" ] ) == len( f1[ "edges" ] ) == 4 and len( v.edges ) == 4
    assert len( f0[ "polygons" ] ) == len( f1[ "polygons" ] ) == 1 and len( v.polygons ) == 1
    assert "SHARED" in v.to_html()

if test( "ellipses: a matrix per point, their radius the largest half-axis" ):
    import numpy as np
    from sdot.viz.Visualizer import Visualizer
    M = np.array( [ [ [ 0.3, 0.0 ], [ 0.0, 0.1 ] ], [ [ 0.1, 0.0 ], [ 0.2, 0.1 ] ] ] )
    v = Visualizer( flat_points = True ).add_points( np.zeros( ( 2, 2 ) ), matrices = M )
    v.add_points( np.ones( ( 1, 2 ) ), radius = 0.5 )
    fr = v.frame( 0 )
    assert np.allclose( fr[ "point_matrices" ], [ M[ 0 ].ravel(), M[ 1 ].ravel(), [ 0, 0, 0, 0 ] ] )
    assert np.allclose( fr[ "point_radii" ], [ 0.3, np.linalg.norm( M[ 1 ], 2 ), 0.5 ] )
    v.to_html()
