from errand import test


def _host( x ):
    import numpy
    from loom.tensor.host import to_host
    return numpy.asarray( to_host( getattr( x, "value", x ) ), dtype = float )


def _image():
    """a smooth density on the unit square, not constant"""
    import numpy
    from sdot.distributions.Image import Image
    x = ( numpy.arange( 24 ) + 0.5 ) / 24
    return Image( values = 1 + 0.5 * numpy.sin( 5 * x[ :, None ] ) * numpy.cos( 3 * x[ None, : ] ) )


if test( "the_diracs_are_at_the_centres_of_the_cells" ):
    import numpy
    from sdot.distributions.SumOfDiracsOnAGrid import SumOfDiracsOnAGrid
    # C order, the first axis the slowest; the longest side of the grid is 1 by default, as for an `Image`
    g = SumOfDiracsOnAGrid( ( 2, 4 ) )
    assert g.shape == ( 2, 4 ) and float( _host( g.mass ) ) == 8
    expected = numpy.array( [ [ ( i + 0.5 ) / 4, ( j + 0.5 ) / 4 ] for i in range( 2 ) for j in range( 4 ) ] )
    assert numpy.allclose( _host( g.positions ), expected )

    # `frame` ( the whole grid, lengths or rows ), `cell_frame` ( one cell ), `origin`
    g = SumOfDiracsOnAGrid( ( 2, 2 ), frame = [ 2.0, 1.0 ], origin = numpy.array( [ -1.0, 0.0 ] ) )
    assert numpy.allclose( _host( g.positions ), [ [ -0.5, 0.25 ], [ -0.5, 0.75 ], [ 0.5, 0.25 ], [ 0.5, 0.75 ] ] )
    shear = numpy.array( [ [ 1.0, 0.5 ], [ 0.0, 1.0 ] ] )
    g = SumOfDiracsOnAGrid( ( 3, 2 ), cell_frame = shear )
    k = numpy.array( [ [ i + 0.5, j + 0.5 ] for i in range( 3 ) for j in range( 2 ) ] )
    assert numpy.allclose( _host( g.positions ), k @ shear )
    assert not g.grid.axis_aligned and g.grid.shape == ( 3, 2 )

    # the weights, flat or of the shape of the grid, give the mass
    g = SumOfDiracsOnAGrid( ( 2, 2 ), weights = [ [ 1, 2 ], [ 3, 4 ] ] )
    assert float( _host( g.mass ) ) == 10 and numpy.allclose( _host( g.weights ), [ 1, 2, 3, 4 ] )
    try:
        SumOfDiracsOnAGrid( ( 2, 2 ), weights = [ 1, 2, 3 ] )
        assert False, "3 weights for 4 cells"
    except ValueError:
        pass

if test( "the_normalized_grid_is_still_a_grid" ):
    import numpy
    from sdot.distributions.SumOfDiracsOnAGrid import SumOfDiracsOnAGrid
    from sdot.OtProblem import OtProblem
    g = SumOfDiracsOnAGrid( ( 3, 5 ), frame = [ 0.6, 1.0 ], weights = numpy.arange( 1.0, 16.0 ) )
    n = OtProblem( g, _image() ).source
    assert isinstance( n, SumOfDiracsOnAGrid ) and n.grid.shape == ( 3, 5 )
    assert numpy.isclose( float( _host( n.mass ) ), 1 ) and numpy.allclose( _host( n.weights ), numpy.arange( 1.0, 16.0 ) / 120 )
    assert numpy.allclose( _host( n.positions ), _host( g.positions ) )

if test( "the_grid_diagram_is_the_diagram_of_the_positions" ):
    import numpy
    from loom.testing import need
    from sdot.PowerDiagram import PowerDiagram
    from sdot.distributions.SumOfDiracsOnAGrid import SumOfDiracsOnAGrid
    need( "cpu" )
    # weights with a smooth part and noise, a sheared grid, a grid larger than the image ( empty cells ): same cells, same
    # facets, same moments as the BSP's diagram on the positions
    rng = numpy.random.default_rng( 2 )
    for shape, kw in ( ( ( 7, 5 ), dict( frame = [ 1.0, 0.8 ] ) ), ( ( 13, 16 ), dict() ),
                       ( ( 9, 11 ), dict( cell_frame = numpy.array( [ [ 0.08, 0.02 ], [ 0.0, 0.07 ] ] ) ) ),
                       ( ( 20, 3 ), dict( frame = [ 1.5, 0.3 ], origin = numpy.array( [ -0.2, 0.3 ] ) ) ) ):
        g = SumOfDiracsOnAGrid( shape, **kw )
        pos = _host( g.positions )
        w = 0.002 * numpy.sin( 7 * pos[ :, 0 ] ) * numpy.cos( 5 * pos[ :, 1 ] ) + 0.001 * rng.standard_normal( len( pos ) )
        a = PowerDiagram( None, w, accelerator = g.grid, distribution = _image(), kernel_dtype = "FP64" )
        b = PowerDiagram( pos, w, distribution = _image(), kernel_dtype = "FP64" )
        assert type( a ).__name__ == "PowerDiagram_Grid"
        assert numpy.allclose( _host( a.measures ), _host( b.measures ), atol = 1e-14 ), shape
        assert numpy.allclose( _host( a.positions ), pos )
        ma, mb = a.moments, b.moments
        for x, y in zip( ma, mb ):
            assert numpy.allclose( _host( x ), _host( y ), atol = 1e-13 ), shape
        ra, rb = a.hessian_rows(), b.hessian_rows()
        assert numpy.array_equal( _host( ra[ 0 ] ), _host( rb[ 0 ] ) ), shape

if test( "a_solve_on_the_grid_is_that_on_its_diracs" ):
    import numpy
    from loom.testing import need
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from sdot.distributions.SumOfDiracsOnAGrid import SumOfDiracsOnAGrid
    from sdot.OtProblem import ot_solve
    need( "cpu" )
    # uniform masses, then masses that vary from one cell to the next ( in C order )
    rng = numpy.random.default_rng( 1 )
    for weights in ( None, rng.uniform( 0.5, 1.5, size = ( 12, 9 ) ) ):
        g = SumOfDiracsOnAGrid( ( 12, 9 ), weights = weights, frame = [ 0.9, 0.7 ], origin = numpy.array( [ 0.05, 0.1 ] ) )
        ref = SumOfDiracs( _host( g.positions ), weights = None if weights is None else weights.reshape( -1 ) )
        p = ot_solve( g, _image(), tol = 1e-10 )
        q = ot_solve( ref, _image(), tol = 1e-10 )
        assert p.converged and q.converged
        assert type( p._pd ).__name__ == "PowerDiagram_Grid" and type( q._pd ).__name__ != "PowerDiagram_Grid"
        assert numpy.allclose( _host( p.weights ), _host( q.weights ), atol = 1e-9 ), numpy.abs( _host( p.weights ) - _host( q.weights ) ).max()
        assert numpy.isclose( float( _host( p.cost ) ), float( _host( q.cost ) ), rtol = 1e-9 )
        m = _host( p.cell_masses ).reshape( -1 )
        nu = numpy.ones( 108 ) if weights is None else weights.reshape( -1 )
        assert numpy.allclose( m, nu / nu.sum(), atol = 1e-9 )

if test( "a_periodic_solve_on_the_grid" ):
    import numpy
    from loom.testing import need
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from sdot.distributions.SumOfDiracsOnAGrid import SumOfDiracsOnAGrid
    from sdot.OtProblem import Iterative, OtProblem
    need( "cpu" )
    g = SumOfDiracsOnAGrid( ( 8, 8 ), frame = [ 0.5, 0.5 ], origin = numpy.array( [ 0.4, 0.3 ] ) )
    p = OtProblem( g, _image(), periodicity = True ).solve( Iterative( tol = 1e-10 ) )
    q = OtProblem( SumOfDiracs( _host( g.positions ) ), _image(), periodicity = True ).solve( Iterative( tol = 1e-10 ) )
    assert p.converged and numpy.allclose( _host( p.weights ), _host( q.weights ), atol = 1e-9 )
    assert type( p._pd ).__name__ != "PowerDiagram_Grid"         # ( a periodicity: the positions, made for the solve )

if test( "a_3d_solve_on_the_grid" ):
    import numpy
    from loom.testing import need
    from sdot.distributions.Image import Image
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from sdot.distributions.SumOfDiracsOnAGrid import SumOfDiracsOnAGrid
    from sdot.OtProblem import ot_solve
    need( "cpu" )
    x = ( numpy.arange( 6 ) + 0.5 ) / 6
    img = Image( values = 1 + 0.5 * x[ :, None, None ] * x[ None, :, None ] + 0.3 * x[ None, None, : ] )
    g = SumOfDiracsOnAGrid( ( 4, 5, 3 ), frame = [ 0.9, 1.0, 0.8 ], weights = numpy.linspace( 1, 2, 60 ) )
    p = ot_solve( g, img, tol = 1e-10 )
    q = ot_solve( SumOfDiracs( _host( g.positions ), weights = numpy.linspace( 1, 2, 60 ) ), img, tol = 1e-10 )
    assert p.converged and type( p._pd ).__name__ == "PowerDiagram_Grid"
    assert numpy.allclose( _host( p.weights ), _host( q.weights ), atol = 1e-9 )
    assert numpy.allclose( _host( p.barycenters ), _host( q.barycenters ), atol = 1e-9 )

if test( "a_1d_grid_goes_to_the_1d_plan" ):
    import numpy
    from loom.testing import need
    from sdot.SdotPlan1d import SdotPlan1d
    from sdot.distributions.SumOfDiracs1d import SumOfDiracs1d
    from sdot.distributions.SumOfDiracsOnAGrid import SumOfDiracsOnAGrid
    from sdot.distributions.Image import Image
    need( "cpu" )
    values = numpy.array( [ 1.0, 0.2, 3.0, 1.0, 0.5 ] )
    g = SumOfDiracsOnAGrid( 7, frame = [ 1.0 ], weights = numpy.arange( 1.0, 8.0 ) )
    p = SdotPlan1d( g, Image( values = values ), with_barycenters = True )
    q = SdotPlan1d( SumOfDiracs1d( positions = _host( g.positions )[ :, 0 ], weights = numpy.arange( 1.0, 8.0 ) ),
                    Image( values = values ), with_barycenters = True )
    assert numpy.isclose( float( _host( p.cost ) ), float( _host( q.cost ) ) )
    assert numpy.allclose( _host( p.barycenters ).reshape( -1 ), _host( q.barycenters ).reshape( -1 ) )

def _grid_masses( N ):
    import numpy
    u = ( numpy.arange( N ) + 0.5 ) / N
    X, Y = numpy.meshgrid( u, u, indexing = "ij" )
    m = numpy.exp( -( ( X - 0.35 ) ** 2 + ( Y - 0.6 ) ** 2 ) / 0.08 ) + 0.3
    return m.ravel() / m.sum()


if test( "the_dedicated_solver_is_the_solver_of_the_cells" ):
    import numpy
    from loom.testing import need
    from sdot.distributions.Box import Box
    from sdot.distributions.SumOfGaussians import SumOfGaussians
    from sdot.distributions.SumOfDiracsOnAGrid import SumOfDiracsOnAGrid
    from sdot.OtProblem import Iterative, OtProblem, Tuning
    need( "cpu" )
    from sdot.distributions.Image import Image
    # the uniform density, images of constant pixels ( square, then 40 x 24 on the square ), a sum of gaussians ( the densities
    # of `GridDensities.h` ) and a polynomial image ( sdot's cells ), straight on 64 x 64 then coarse to fine from 32: the same
    # plan as the generic solver on the grid's diagram ( `grid_solver = "cells"` )
    x, y = ( numpy.arange( 40 ) + 0.5 ) / 40, ( numpy.arange( 24 ) + 0.5 ) / 24
    coeffs = 0.5 + numpy.random.default_rng( 0 ).random( ( 6, 6, 4 ) )
    targets = ( ( "uniform density", lambda: Box( [ 0.0, 0.0 ], [ 1.0, 1.0 ] ) ), ( "image", _image ),
                ( "image", lambda: Image( values = 1 + 0.6 * numpy.cos( 4 * x[ :, None ] + 2 * y[ None, : ] ), frame = [ 1.0, 1.0 ] ) ),
                ( "gaussians", lambda: SumOfGaussians( numpy.array( [ [ 0.3, 0.4 ], [ 0.7, 0.6 ] ] ), numpy.array( [ 0.2, 0.25 ] ),
                                                       weights = numpy.array( [ 1.0, 0.7 ] ), support_box = ( [ 0.0, 0.0 ], [ 1.0, 1.0 ] ) ) ),
                ( "gaussians", lambda: SumOfGaussians( numpy.array( [ [ 0.5, 0.5 ] ] ), numpy.array( [ 0.08 ] ), support_box = ( [ 0.0, 0.0 ], [ 1.0, 1.0 ] ) ) ),
                ( "sdot cells", lambda: Image( values = coeffs, degree = 1 ) ) )
    m = _grid_masses( 64 )
    for kind, target in targets:
        ref = OtProblem( SumOfDiracsOnAGrid( ( 64, 64 ), weights = m ), target() ).solve( Iterative( tol = 1e-12, tuning = Tuning( grid_solver = "cells" ) ) )
        for n0 in ( 64, 32 ):
            p = OtProblem( SumOfDiracsOnAGrid( ( 64, 64 ), weights = m ), target() ).solve(
                Iterative( tol = 1e-12, tuning = Tuning( grid_solver = "dedicated", grid_n0 = n0 ) ) )
            assert p.converged and p.stats[ "grid_solver" ] == f"dedicated ( { kind } )", p.stats
            # ( the gaussians: the steps of the width continuation first, on the coarsest level )
            steps = ( 1 if n0 == 64 else 2 )
            assert p.stats[ "nb_continuation_steps" ] == steps or ( kind == "gaussians" and p.stats[ "nb_continuation_steps" ] > steps ), p.stats
            assert numpy.allclose( _host( p.weights ), _host( ref.weights ), atol = 1e-9 ), ( kind, numpy.abs( _host( p.weights ) - _host( ref.weights ) ).max() )
            assert numpy.allclose( _host( p.cell_masses ), _host( ref.cell_masses ), atol = 1e-11 )
            assert numpy.allclose( _host( p.barycenters ), _host( ref.barycenters ), atol = 1e-8 )
            assert numpy.isclose( float( _host( p.cost ) ), float( _host( ref.cost ) ), rtol = 1e-8 )

if test( "the_dedicated_solver_starts_from_given_weights" ):
    import numpy
    from loom.testing import need
    from sdot.distributions.SumOfDiracsOnAGrid import SumOfDiracsOnAGrid
    from sdot.OtProblem import Iterative, OtProblem, Tuning
    need( "cpu" )
    m = _grid_masses( 32 )
    pb = OtProblem( SumOfDiracsOnAGrid( ( 32, 32 ), weights = m ), _image() )
    p = pb.solve( Iterative( tol = 1e-12 ) )
    q = OtProblem( SumOfDiracsOnAGrid( ( 32, 32 ), weights = m ), _image() ).solve( Iterative( tol = 1e-12, weights0 = _host( p.weights ) ) )
    assert q.converged and q.stats[ "nb_iter" ] <= 1 and q.stats[ "nb_continuation_steps" ] == 1
    assert numpy.allclose( _host( q.weights ), _host( p.weights ), atol = 1e-10 )

if test( "what_the_dedicated_solver_does_not_take_goes_to_the_cells" ):
    import numpy
    from loom.testing import need
    from sdot.distributions.Box import Box
    from sdot.distributions.SumOfDiracsOnAGrid import SumOfDiracsOnAGrid
    from sdot.OtProblem import Iterative, OtProblem, Tuning
    need( "cpu" )
    # not N x N with N a power of two, rectangular cells, a target that is not on the box of the grid
    for g, target in ( ( SumOfDiracsOnAGrid( ( 24, 24 ) ), Box( [ 0.0, 0.0 ], [ 1.0, 1.0 ] ) ),
                       ( SumOfDiracsOnAGrid( ( 32, 32 ), frame = [ 1.0, 0.5 ] ), Box( [ 0.0, 0.0 ], [ 1.0, 0.5 ] ) ),
                       ( SumOfDiracsOnAGrid( ( 32, 32 ), frame = [ 0.5, 0.5 ] ), Box( [ 0.0, 0.0 ], [ 1.0, 1.0 ] ) ) ):
        p = OtProblem( g, target ).solve( Iterative( tol = 1e-10 ) )
        assert p.converged and "grid_solver" not in p.stats and type( p._pd ).__name__ == "PowerDiagram_Grid"
        try:
            OtProblem( g, target ).solve( Iterative( tol = 1e-10, tuning = Tuning( grid_solver = "dedicated" ) ) )
            assert False, "grid_solver = 'dedicated' where it does not apply"
        except ValueError:
            pass

if test( "the_potential_integrals_of_a_p1_image_are_the_derivative_of_the_cost" ):
    import numpy
    from loom.testing import need
    from sdot.distributions.Image import Image
    from sdot.distributions.SumOfDiracsOnAGrid import SumOfDiracsOnAGrid
    from sdot.OtProblem import Iterative, OtProblem
    from sdot.PowerDiagram import PowerDiagram
    need( "cpu" )
    # the diracs of a grid against a P1 image ( nodal values at the corners of its pixels ); the derivative of the cost with
    # respect to the nodal values, by the envelope theorem: the potential against each basis function, through the
    # normalization of the image
    N = 32
    x = numpy.arange( N + 1 ) / N
    X, Y = numpy.meshgrid( x, x, indexing = "ij" )
    nodes = 1 + 0.5 * numpy.cos( 5 * X + 2 * Y ) ** 2 + 0.3 * numpy.exp( -30 * ( ( X - 0.4 ) ** 2 + ( Y - 0.6 ) ** 2 ) )
    c = ( numpy.arange( N ) + 0.5 ) / N
    masses = 1 + 0.4 * numpy.sin( 3 * c[ :, None ] + 1 ) * numpy.cos( 4 * c[ None, : ] )
    corners = ( numpy.s_[ :-1, :-1 ], numpy.s_[ :-1, 1: ], numpy.s_[ 1:, :-1 ], numpy.s_[ 1:, 1: ] )
    coeffs = lambda v: numpy.stack( [ v[ s ] for s in corners ], -1 )

    def solve( v ):
        return OtProblem( SumOfDiracsOnAGrid( ( N, N ), weights = masses ), Image( values = coeffs( v ), degree = 1 ) ).solve( Iterative( tol = 1e-12, max_iter = 200 ) )

    p = solve( nodes )
    pd = PowerDiagram( _host( SumOfDiracsOnAGrid( ( N, N ) ).positions ), _host( p.weights ), distribution = Image( values = coeffs( nodes ), degree = 1 ),
                       kernel_dtype = "FP64" )
    I = _host( pd.potential_integrals() )
    assert I.shape == ( N, N, 4 )
    In, q = numpy.zeros( ( N + 1, N + 1 ) ), numpy.zeros( ( N + 1, N + 1 ) )
    for k, s in enumerate( corners ):
        In[ s ] += I[ ..., k ]
        q[ s ] += 0.25 / N ** 2
    S = float( ( nodes * q ).sum() )
    g = In / S - q * float( ( nodes * In ).sum() ) / S ** 2
    d, eps = numpy.random.default_rng( 0 ).normal( size = nodes.shape ), 1e-5
    fd = ( float( _host( solve( nodes + eps * d ).cost ) ) - float( _host( solve( nodes - eps * d ).cost ) ) ) / ( 2 * eps )
    an = float( ( g * d ).sum() )
    assert abs( fd - an ) < 1e-6 * abs( an ), ( fd, an )

if test( "the_grid_solver_takes_sharp_targets" ):
    import numpy
    from loom.testing import need
    from sdot.OtProblem import OtProblem, Iterative, Tuning
    from sdot.distributions.Image import Image
    from sdot.distributions.SumOfDiracsOnAGrid import SumOfDiracsOnAGrid
    need( "cpu" )
    # thin rings on a dark background, the diracs and the image a fraction of a pixel apart ( a radiograph of hollow spheres and
    # its model ): cells of a hundredth of a pixel beside cells of ten pixels, a Newton step of 100 at the rim of the rings
    N = 128
    h = 1 / N
    def rings( x, y, shift ):
        v = 0 * x
        for cx, cy, r in ( ( 0.35, 0.4, 0.2 ), ( 0.7, 0.65, 0.15 ) ):
            d = numpy.hypot( x - cx - shift, y - cy ) - r
            v = v + numpy.maximum( 1 - ( d / ( 2.5 * h ) ) ** 2, 0 ) ** 2.5
        return v + 3e-3 * v.mean()
    p = ( numpy.arange( N ) + 0.5 ) * h
    q = numpy.arange( N + 1 ) * h
    def solve( shift, weights0 = None ):
        v = rings( q[ :, None ], q[ None, : ], shift )
        img = Image( values = numpy.stack( [ v[ :-1, :-1 ], v[ :-1, 1: ], v[ 1:, :-1 ], v[ 1:, 1: ] ], -1 ), degree = 1 )
        src = SumOfDiracsOnAGrid( ( N, N ), weights = rings( p[ :, None ], p[ None, : ], 0.0 ).reshape( -1 ) )
        return OtProblem( src, img ).solve( Iterative( tol = 1e-3 / N ** 2, max_iter = 100, weights0 = weights0, tuning = Tuning( grid_solver = "dedicated" ) ) )

    cold = solve( 0.7 * h )
    print( { k: cold.stats[ k ] for k in ( "status", "nb_iter", "nb_backtracks", "residual" ) } )
    nu = rings( p[ :, None ], p[ None, : ], 0.0 ).reshape( -1 )
    assert numpy.abs( _host( cold.cell_masses ) - nu / nu.sum() ).max() < 1e-3 / N ** 2
    assert float( _host( cold.cost ) ) > 0

    # the same target a quarter of a pixel further: a few steps from the weights of the first solve
    warm = solve( 0.95 * h, cold.weights )
    print( { k: warm.stats[ k ] for k in ( "status", "nb_iter", "nb_backtracks", "residual" ) } )
    assert int( warm.stats[ "nb_iter" ] ) <= 25 and int( warm.stats[ "nb_backtracks" ] ) <= 40
