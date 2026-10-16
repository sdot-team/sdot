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
