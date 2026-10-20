from errand import test

if test( "basic" ):
    # ds = SumOfDiracs( [ 1, 2, 3, 4 ] )
    # op = SdotPlanNd1D( ds, di )
    from sdot.distributions.Image import Image
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from loom.testing import check_grad
    di = Image( values = [[ 1, 0, 1 ]], pixel_frame = [ [ 1, 0 ], [ 0, 1 ] ] )   # pixels of 1: a mass of 2
    assert float( di.mass ) == 2


if test( "the_frame_and_the_origin_derive" ):
    import numpy
    from loom.testing import check_grad, need
    from sdot.PowerDiagram import PowerDiagram
    from sdot.distributions.Box import Box
    from sdot.distributions.Image import Image
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from sdot.OtProblem import Iterative, OtProblem
    need( "cpu" )
    # the image is the affine image `x = origin + F^T t` of a fixed grid: in `t` only the facets of the seeds move, so the integrals
    # of the cells derive with respect to `F` and `origin` from the vertex adjoint and closed forms ( `diagram::AffineAcc` ). A
    # sheared frame: the sides of the grid are the domain's up to a rounding, and must not cut the cells again ( `Image.cxx` )
    rng = numpy.random.default_rng( 0 )
    pos = rng.uniform( 0.2, 0.8, size = ( 10, 2 ) )
    values = rng.uniform( 0.5, 1.5, size = ( 3, 3 ) )
    w = rng.uniform( 0, 0.01, size = 10 )
    F = numpy.array( [ [ 0.36, 0.03 ], [ -0.02, 0.33 ] ] )
    o = numpy.array( [ 0.02, -0.01 ] )
    tight = dict( eps = 1e-6, rtol = 1e-6, atol = 1e-9 )
    img = lambda F, o: Image( values, pixel_frame = F, origin = o )
    for k in ( None, 0, 1, 2, 3 ):
        integral = lambda pd: pd.measures if k is None else pd.moment( k )
        check_grad( lambda F, o: integral( PowerDiagram( pos, w, distribution = img( F, o ), kernel_dtype = "FP64" ) ), F, o, seed = 10 + ( k or 0 ), **tight )

    # the whole plan, an image and a box
    settings = Iterative( tol = 1e-13, max_iter = 200 )
    for output in ( "weights", "cost", "barycenters" ):
        check_grad( lambda F, o: getattr( OtProblem( SumOfDiracs( pos ), img( F, o ) ).solve( settings ), output ), F, o, seed = 20, **tight )
        check_grad( lambda f, o: getattr( OtProblem( SumOfDiracs( pos ), Box( origin = o, frame = f ) ).solve( settings ), output ),
                    numpy.array( [ 1.0, 1.1 ] ), numpy.array( [ 0.05, -0.02 ] ), seed = 21, **tight )


# ---- polynomial images ( `degree > 0`, Bernstein coefficients per pixel: `PolynomialPixel.h` ) ---------------------------------

if test( "polynomial_degree_elevation" ):
    # the same density written with a higher degree: the same integrals, to the rounding ( the rule is exact )
    import numpy
    from sdot.PowerDiagram import PowerDiagram
    from sdot.distributions.Image import Image
    from sdot.distributions._bernstein import elevate
    rng = numpy.random.default_rng( 1 )
    for d in ( 2, 3 ):
        pos = rng.uniform( 0.05, 0.95, size = ( 40, d ) ) * ( 0.75 if d == 3 else 1 )
        shape = ( 4, 3 ) if d == 2 else ( 3, 2, 3 )
        v = rng.uniform( 0.5, 1.5, size = shape )
        c1 = rng.uniform( 0.2, 1.5, size = shape + ( 2 ** d, ) )
        ref = PowerDiagram( pos, distribution = Image( v ), kernel_dtype = "FP64" )
        for p in ( 1, 2 ):
            c = numpy.repeat( v[ ..., None ], ( p + 1 ) ** d, axis = -1 )
            pd = PowerDiagram( pos, distribution = Image( c, degree = p ), kernel_dtype = "FP64" )
            assert numpy.abs( numpy.asarray( pd.measures ) - numpy.asarray( ref.measures ) ).max() < 1e-14
        ref = PowerDiagram( pos, distribution = Image( c1, degree = 1 ), kernel_dtype = "FP64" )
        for by in ( 1, 2 ):
            pd = PowerDiagram( pos, distribution = Image( elevate( c1, d, by ), degree = 1 + by ), kernel_dtype = "FP64" )
            for k in ( None, 0, 1, d ):
                a = numpy.asarray( pd.measures if k is None else pd.moment( k ) )
                b = numpy.asarray( ref.measures if k is None else ref.moment( k ) )
                assert numpy.abs( a - b ).max() < 1e-13, ( d, by, k, numpy.abs( a - b ).max() )


if test( "polynomial_matches_an_affine_mesh" ):
    # an affine density, as a degree 1 image and as a P1 mesh of the same box: the same cells, the same integrals
    import numpy
    from sdot.PowerDiagram import PowerDiagram
    from sdot.distributions.Image import Image
    from sdot.distributions.Mesh import Mesh
    from sdot.distributions._bernstein import from_function
    rng = numpy.random.default_rng( 2 )
    for d in ( 2, 3 ):
        shape = ( 3, 2 ) if d == 2 else ( 2, 3, 2 )
        hi = numpy.asarray( shape, dtype = float ) / max( shape )                  # ( the default frame: the longest side is 1 )
        rho = lambda x: 1 + x @ numpy.arange( 1, d + 1 ) * 0.4
        img = Image( from_function( rho, shape, 1, hi = hi ), degree = 1 )
        corners = numpy.stack( numpy.meshgrid( *[ [ 0, h ] for h in hi ], indexing = "ij" ), -1 ).reshape( -1, d )
        from scipy.spatial import Delaunay
        mesh = Mesh( corners, Delaunay( corners ).simplices, values = rho( corners ) )
        pos = rng.uniform( 0, 1, size = ( 30, d ) ) * hi
        a = PowerDiagram( pos, distribution = img, kernel_dtype = "FP64" )
        b = PowerDiagram( pos, distribution = mesh, kernel_dtype = "FP64" )
        for k in ( None, 0, d ):
            x = numpy.asarray( a.measures if k is None else a.moment( k ) )
            y = numpy.asarray( b.measures if k is None else b.moment( k ) )
            assert numpy.abs( x - y ).max() < 1e-13, ( d, k, numpy.abs( x - y ).max() )


if test( "polynomial_derives" ):
    # the integrals of the cells, with respect to the coefficients, the seeds, the frame and the origin; then the whole plan
    import numpy
    from loom.testing import check_grad, need
    from sdot.PowerDiagram import PowerDiagram
    from sdot.distributions.Image import Image
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from sdot.OtProblem import Iterative, OtProblem
    need( "cpu" )
    rng = numpy.random.default_rng( 3 )
    pos = rng.uniform( 0.2, 0.8, size = ( 10, 2 ) )
    w = rng.uniform( 0, 0.01, size = 10 )
    c = rng.uniform( 0.3, 1.5, size = ( 3, 3, 9 ) )
    F = numpy.array( [ [ 0.36, 0.03 ], [ -0.02, 0.33 ] ] )
    o = numpy.array( [ 0.02, -0.01 ] )
    tight = dict( eps = 1e-6, rtol = 1e-6, atol = 1e-9 )
    img = lambda c, F, o: Image( c, degree = 2, pixel_frame = F, origin = o )
    for k in ( None, 0, 1, 2, 3 ):
        integral = lambda pd: pd.measures if k is None else pd.moment( k )
        check_grad( lambda c, p, w: integral( PowerDiagram( p, w, distribution = img( c, F, o ), kernel_dtype = "FP64" ) ), c, pos, w, seed = 30 + ( k or 0 ), **tight )
        check_grad( lambda F, o: integral( PowerDiagram( pos, w, distribution = img( c, F, o ), kernel_dtype = "FP64" ) ), F, o, seed = 40 + ( k or 0 ), **tight )

    settings = Iterative( tol = 1e-13, max_iter = 200 )
    for output in ( "weights", "cost", "barycenters" ):
        check_grad( lambda c, F, o: getattr( OtProblem( SumOfDiracs( pos ), img( c, F, o ) ).solve( settings ), output ), c, F, o, seed = 50, **tight )


if test( "polynomial_continuation" ):
    # a peaked target that the plain Newton does not reach: the stages are coarser grids of the blurred density
    # ( `sdotplan/convolved/Image.h` ), in 2D and 3D, for a polynomial image as for a constant one
    import numpy
    from sdot.distributions.Image import Image
    from sdot.distributions.SplineGrid import SplineGrid
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from sdot.OtProblem import Iterative, OtProblem
    rng = numpy.random.default_rng( 5 )
    for d, k, n in ( ( 2, 64, 2000 ), ( 3, 16, 1000 ) ):
        x = numpy.stack( numpy.meshgrid( *[ ( numpy.arange( k ) + 0.5 ) / k ] * d, indexing = "ij" ), -1 )
        v = 1e-3 + numpy.exp( -( ( x - 0.35 ) ** 2 ).sum( -1 ) / 0.004 ) + 0.5 * ( x[ ..., 0 ] > 0.6 )
        pos = rng.uniform( 0, 1, size = ( n, d ) )
        for p in ( 1, 3 ):
            target = SplineGrid( v, degree = p )
            for cont in ( "auto", "always" ):
                sol = OtProblem( SumOfDiracs( pos ), target ).solve( Iterative( continuation = cont ) )
                assert sol.converged, ( d, p, cont, sol.stats[ "status" ] )
