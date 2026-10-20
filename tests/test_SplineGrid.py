from errand import test

# `SplineGrid`: a B-spline of control coefficients on the centers of the pixels, written as an `Image( degree = p )`, one Bernstein
# polynomial per span ( `distributions/SplineGrid.py` )


def spline_at( values, p, x ):
    """the spline at `x` ( `[ m, d ]`, pixel units ), from its definition: the tensor product of the basis matrices"""
    import numpy
    from sdot.distributions.SplineGrid import _basis_matrix
    d = values.ndim
    Ms = [ _basis_matrix( values.shape[ a ], p, x[ :, a ] ) for a in range( d ) ]
    res = numpy.empty( len( x ) )
    for i in range( len( x ) ):
        t = values
        for a in range( d ):
            t = numpy.tensordot( Ms[ a ][ i ], t, axes = ( [ 0 ], [ 0 ] ) )
        res[ i ] = t
    return res


def image_at( img, x ):
    """an image at `x` ( pixel units of the spline: the default frame, whose longest side is 1 )"""
    import numpy
    from sdot.distributions._bernstein import evaluate
    d = int( img.nb_dims.value )
    shape = numpy.asarray( img.shape.value, dtype = int ).reshape( -1 )
    if img.knots.is_defined:
        kn = numpy.asarray( img.knots, dtype = float ).reshape( d, -1 )
        knots = [ kn[ a, : shape[ a ] + 1 ] for a in range( d ) ]
    else:
        knots = [ numpy.arange( shape[ a ] + 1, dtype = float ) for a in range( d ) ]
    pix, loc = [], []
    for a in range( d ):
        k = knots[ a ]
        j = numpy.clip( numpy.searchsorted( k, x[ :, a ], side = "right" ) - 1, 0, len( k ) - 2 )
        pix.append( j )
        loc.append( ( x[ :, a ] - k[ j ] ) / ( k[ j + 1 ] - k[ j ] ) )
    return evaluate( numpy.asarray( img.coeffs, dtype = float ), d, numpy.stack( pix, 1 ), numpy.stack( loc, 1 ) )


if test( "is_the_spline" ):
    import numpy
    from sdot.distributions.SplineGrid import SplineGrid
    rng = numpy.random.default_rng( 0 )
    for p in ( 1, 2, 3, 4 ):
        for shape in ( ( 5, 4 ), ( 3, 4, 2 ) ):
            v = rng.uniform( 0, 1, shape )
            img = SplineGrid( v, degree = p )
            assert img.degree == p
            assert numpy.asarray( img.coeffs ).min() >= 0                       # ( non-negative coefficients: a non-negative density )
            x = rng.uniform( 0, 1, ( 60, len( shape ) ) ) * numpy.array( shape )
            err = numpy.abs( image_at( img, x ) - spline_at( v, p, x ) ).max()
            assert err < 1e-14, ( p, shape, err )
            one = SplineGrid( numpy.ones( shape ), degree = p )
            assert numpy.abs( numpy.asarray( one.coeffs ) - 1 ).max() < 1e-14     # ( the basis sums to one on the whole box )


if test( "a_constant_stays_a_constant" ):
    # the same cells as the constant image: the knots of an odd degree ( half pixels at the ends ) included
    import numpy
    from sdot.PowerDiagram import PowerDiagram
    from sdot.distributions.Image import Image
    from sdot.distributions.SplineGrid import SplineGrid
    rng = numpy.random.default_rng( 1 )
    for d, shape in ( ( 2, ( 5, 3 ) ), ( 3, ( 3, 2, 4 ) ) ):
        pos = rng.uniform( 0, 1, size = ( 40, d ) ) * numpy.array( shape ) / max( shape )
        ref = numpy.asarray( PowerDiagram( pos, distribution = Image( numpy.ones( shape ) ), kernel_dtype = "FP64" ).measures )
        for p in ( 1, 2, 3 ):
            m = numpy.asarray( PowerDiagram( pos, distribution = SplineGrid( numpy.ones( shape ), degree = p ), kernel_dtype = "FP64" ).measures )
            assert numpy.abs( m - ref ).max() < 1e-13, ( d, p, numpy.abs( m - ref ).max() )


if test( "interpolates" ):
    import numpy
    from sdot.distributions.SplineGrid import SplineGrid
    rng = numpy.random.default_rng( 2 )
    shape = ( 6, 5 )
    s = rng.uniform( 0.5, 1.5, shape )                           # ( smooth enough: the coefficients stay positive )
    centers = numpy.stack( numpy.meshgrid( *[ numpy.arange( n ) + 0.5 for n in shape ], indexing = "ij" ), -1 ).reshape( -1, 2 )
    for p in ( 1, 2, 3 ):
        img = SplineGrid( s, degree = p, interpolate = True )
        err = numpy.abs( image_at( img, centers ) - s.reshape( -1 ) ).max()
        assert err < 1e-13, ( p, err )

    # a sharp edge: the cubic that goes through it overshoots below zero, and is refused
    step = numpy.zeros( ( 8, 8 ) )
    step[ 4:, : ] = 1
    try:
        SplineGrid( step, degree = 3, interpolate = True )
        raise AssertionError( "a negative spline was taken" )
    except ValueError as e:
        assert "below zero" in str( e )


if test( "derives" ):
    # through the map from the control coefficients to the Bernstein ones ( in the framework of `values` ), then the plan
    import numpy
    from loom.testing import check_grad, need
    from sdot.distributions.SplineGrid import SplineGrid
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from sdot.OtProblem import Iterative, OtProblem
    need( "cpu" )
    rng = numpy.random.default_rng( 3 )
    pos = rng.uniform( 0.1, 0.9, size = ( 12, 2 ) ) * [ 1, 0.8 ]
    v = rng.uniform( 0.5, 1.5, ( 5, 4 ) )
    settings = Iterative( tol = 1e-13, max_iter = 200 )
    for p in ( 2, 3 ):
        for output in ( "cost", "barycenters" ):
            check_grad( lambda v: getattr( OtProblem( SumOfDiracs( pos ), SplineGrid( v, degree = p ) ).solve( settings ), output ), v,
                        seed = 60 + p, eps = 1e-6, rtol = 1e-6, atol = 1e-9 )


if test( "solves" ):
    # transports to a smooth density, 2D and 3D: the masses of the cells are the source's
    import numpy
    from sdot.distributions.SplineGrid import SplineGrid
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from sdot.OtProblem import OtProblem
    from sdot.PowerDiagram import PowerDiagram
    rng = numpy.random.default_rng( 4 )
    for d, shape, n in ( ( 2, ( 16, 12 ), 300 ), ( 3, ( 6, 5, 4 ), 200 ) ):
        x = numpy.stack( numpy.meshgrid( *[ ( numpy.arange( k ) + 0.5 ) / k for k in shape ], indexing = "ij" ), -1 )
        v = 0.2 + numpy.exp( -( ( x - 0.5 ) ** 2 ).sum( -1 ) / 0.05 )
        target = SplineGrid( v, degree = 3 )
        pos = rng.uniform( 0, 1, size = ( n, d ) ) * numpy.array( shape ) / max( shape )
        sol = OtProblem( SumOfDiracs( pos ), target ).solve()
        assert sol.converged
        m = numpy.asarray( PowerDiagram( pos, numpy.asarray( sol.weights ), distribution = target.normalized_version(), kernel_dtype = "FP64" ).measures )
        assert numpy.abs( m - 1 / n ).max() < 1e-6 / n, ( d, numpy.abs( m - 1 / n ).max() * n )
