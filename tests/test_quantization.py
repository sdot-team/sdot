from errand import test, skip

if test( "lloyd_on_a_square_reaches_the_regular_grid" ):
    import numpy
    from sdot import ConvexHull, quantization        # the public API: arrays of the caller, views given to the callback
    # 4 diracs of equal mass on the unit square: the centers of the 4 quadrants
    square = ConvexHull( [ [ 0, 0 ], [ 1, 0 ], [ 1, 1 ], [ 0, 1 ] ] )
    grid = numpy.array( [ [ 0.25, 0.25 ], [ 0.75, 0.25 ], [ 0.25, 0.75 ], [ 0.75, 0.75 ] ] )
    x0 = grid + numpy.random.default_rng( 0 ).uniform( -0.1, 0.1, grid.shape )
    costs = []
    x = quantization( square, x0, tol = 1e-6, callback = lambda it, pos, plan: costs.append( float( plan.cost ) ) )
    assert isinstance( x, numpy.ndarray ) and x.shape == ( 4, 2 )
    assert numpy.abs( x - grid ).max() < 1e-4
    # Lloyd is a descent on W_2^2
    assert all( b <= a + 1e-12 for a, b in zip( costs, costs[ 1: ] ) )


if test( "the_callback_receives_the_positions_of_each_solve" ):
    import numpy
    from sdot import ConvexHull, quantization        # the public API: arrays of the caller, views given to the callback
    square = ConvexHull( [ [ 0, 0 ], [ 1, 0 ], [ 1, 1 ], [ 0, 1 ] ] )
    x0 = numpy.random.default_rng( 1 ).uniform( 0, 1, ( 50, 2 ) )
    seen = []
    x = quantization( square, x0, callback = lambda it, pos, plan: seen.append( ( it, numpy.asarray( pos ) ) ) )
    assert [ it for it, _ in seen ] == list( range( len( seen ) ) )
    assert numpy.array_equal( seen[ 0 ][ 1 ], x0 )
    assert len( seen ) > 1 and not numpy.allclose( seen[ -1 ][ 1 ], x0 )


if test( "max_iter_reached_warns" ):
    import numpy
    import pytest
    from sdot import ConvexHull, quantization        # the public API: arrays of the caller, views given to the callback
    square = ConvexHull( [ [ 0, 0 ], [ 1, 0 ], [ 1, 1 ], [ 0, 1 ] ] )
    x0 = numpy.random.default_rng( 2 ).uniform( 0, 1, ( 50, 2 ) )
    with pytest.warns( UserWarning, match = "iterations without reaching" ):
        quantization( square, x0, max_iter = 2 )


if test( "a_number_of_diracs_starts_them_in_the_support" ):
    import numpy
    from sdot import ConvexHull, quantization        # the public API: arrays of the caller, views given to the callback
    from sdot.quantization import uniform_in_support
    tri = ConvexHull( [ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ] ] )
    x0 = uniform_in_support( tri, 500, seed = 3 )
    assert x0.shape == ( 500, 2 ) and numpy.all( x0 >= 0 ) and numpy.all( x0.sum( axis = 1 ) <= 1 )
    seen = []
    x = quantization( tri, 500, seed = 3, callback = lambda it, pos, plan: seen.append( numpy.asarray( pos ) ) )
    assert numpy.array_equal( seen[ 0 ], x0 ) and x.shape == ( 500, 2 )


if test( "a_mesh_starts_its_diracs_by_mass" ):
    # drawn in the box of the domain, the seeds around a mesh ( zero density ) had no mass, and the first solve did not converge
    import numpy
    from scipy.spatial import Delaunay
    from sdot import Mesh, quantization
    rng = numpy.random.default_rng( 0 )
    nodes = numpy.concatenate( [ [ [ 0, 0 ], [ 1, 0 ], [ 0, 1 ], [ 1, 1 ] ], rng.random( ( 300, 2 ) ) ] )
    sx = Delaunay( nodes ).simplices
    values = numpy.ones( sx.shape )
    values[ numpy.linalg.norm( nodes[ sx ].mean( 1 ) - 0.5, axis = 1 ) < 0.25 ] = 0           # a hole
    mesh = Mesh( nodes, sx, values = values )
    x0 = mesh.random_points( 2000, seed = 1 )
    assert numpy.all( ( x0 >= 0 ) & ( x0 <= 1 ) ) and not numpy.any( numpy.linalg.norm( x0 - 0.5, axis = 1 ) < 0.2 )
    x = quantization( mesh, 500, seed = 0, max_iter = 5 )
    assert x.shape == ( 500, 2 )
