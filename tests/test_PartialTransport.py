from errand import test


# PARTIAL TRANSPORT: the cell of seed `i` is integrated inside its ball `|x - p_i|^2 <= w_i - w_s` only ( `integration/InBall.h` ).
# THE REFERENCE is the same diagram without a sink, each cell clipped by a fine polygon of its disc in numpy: no code in common
# but the cutting of the Laguerre cells.


def _disc_in_cell( vertices, center, r, nb = 4096 ):
    """the convex polygon `vertices` inter the disc ( a regular `nb`-gon of the same area )"""
    import numpy
    t = 2 * numpy.pi * numpy.arange( nb ) / nb
    r = r * numpy.sqrt( 2 * numpy.pi / nb / numpy.sin( 2 * numpy.pi / nb ) )
    poly = center + r * numpy.stack( [ numpy.cos( t ), numpy.sin( t ) ], axis = 1 )
    v = numpy.asarray( vertices, dtype = float ).reshape( -1, 2 )
    if len( v ) < 3:
        return poly[ :0 ]                                # an empty cell
    if numpy.sum( v[ :, 0 ] * numpy.roll( v[ :, 1 ], -1 ) - numpy.roll( v[ :, 0 ], -1 ) * v[ :, 1 ] ) < 0:
        v = v[ ::-1 ]
    for i in range( len( v ) ):
        a, b = v[ i ], v[ ( i + 1 ) % len( v ) ]
        s = ( poly - a ) @ numpy.array( [ b[ 1 ] - a[ 1 ], a[ 0 ] - b[ 0 ] ] )
        q, sq = numpy.roll( poly, -1, axis = 0 ), numpy.roll( s, -1 )
        out = []
        for j in range( len( poly ) ):
            if s[ j ] <= 0:
                out.append( poly[ j ] )
            if ( s[ j ] <= 0 ) != ( sq[ j ] <= 0 ):
                out.append( poly[ j ] + ( q[ j ] - poly[ j ] ) * ( s[ j ] / ( s[ j ] - sq[ j ] ) ) )
        poly = numpy.asarray( out ).reshape( -1, 2 )
        if len( poly ) == 0:
            break
    return poly


def _area_and_first( poly ):
    import numpy
    if len( poly ) < 3:
        return 0.0, numpy.zeros( 2 )
    x, y = poly[ :, 0 ], poly[ :, 1 ]
    x1, y1 = numpy.roll( x, -1 ), numpy.roll( y, -1 )
    c = x * y1 - x1 * y
    return c.sum() / 2, numpy.array( [ ( ( x + x1 ) * c ).sum() / 6, ( ( y + y1 ) * c ).sum() / 6 ] )


if test( "measures_inside_balls_match_clipped_cells" ):
    import numpy
    from sdot.distributions.Box import Box
    from sdot.PowerDiagram import PowerDiagram
    rng = numpy.random.default_rng( 0 )
    n, ws = 15, -0.004
    pos = rng.uniform( 0, 1, size = ( n, 2 ) )
    w = rng.uniform( 0.0, 0.05, size = n )
    pd = PowerDiagram( pos, w, distribution = Box( nb_dims = 2 ), kernel_dtype = "FP64", sink_weight = ws )
    m = numpy.asarray( pd.measures ).reshape( -1 )
    mm, f, _ = pd.moments
    f = numpy.asarray( f.value ).reshape( n, 2 )
    cells = PowerDiagram( pos, w, distribution = Box( nb_dims = 2 ), kernel_dtype = "FP64" ).cells
    for i in range( n ):
        a, fi = _area_and_first( _disc_in_cell( cells.vertices( i ), pos[ i ], numpy.sqrt( w[ i ] - ws ) ) )
        assert abs( m[ i ] - a ) < 2e-7, ( i, m[ i ], a )
        assert numpy.abs( f[ i ] - fi ).max() < 2e-7, ( i, f[ i ], fi )
    assert numpy.allclose( numpy.asarray( mm.value ).reshape( -1 ), m, atol = 1e-13 )


if test( "a_ball_inside_its_cell_holds_pi_r2" ):
    import numpy
    from sdot.distributions.Box import Box
    from sdot.PowerDiagram import PowerDiagram
    pos = numpy.array( [ [ 0.3, 0.3 ], [ 0.7, 0.7 ] ] )
    w = numpy.array( [ 0.01, 0.02 ] )
    pd = PowerDiagram( pos, w, distribution = Box( nb_dims = 2 ), kernel_dtype = "FP64", sink_weight = 0.0 )
    assert numpy.allclose( numpy.asarray( pd.measures ).reshape( -1 ), numpy.pi * w, atol = 1e-13 )
    _, f, q = pd.moments
    assert numpy.allclose( numpy.asarray( f.value ).reshape( 2, 2 ), pos * ( numpy.pi * w )[ :, None ], atol = 1e-13 )
    # int |x|^2 = int |x - p|^2 + |p|^2 m = pi r^4 / 2 + |p|^2 pi r^2
    assert numpy.allclose( numpy.asarray( q.value ).reshape( -1 ), numpy.pi * w ** 2 / 2 + ( pos ** 2 ).sum( 1 ) * numpy.pi * w, atol = 1e-13 )


if test( "the_hessian_has_the_sphere_on_its_diagonal" ):
    import numpy
    from sdot.distributions.Box import Box
    from sdot.distributions.Image import Image
    from sdot.PowerDiagram import PowerDiagram
    rng = numpy.random.default_rng( 0 )
    n, ws, eps = 12, -0.005, 1e-6
    pos = rng.uniform( 0, 1, size = ( n, 2 ) )
    w = rng.uniform( 0.0, 0.04, size = n )
    for dist in ( Box( nb_dims = 2 ), Image( rng.uniform( 0.5, 2, size = ( 7, 5 ) ), frame = [ 1.0, 1.0 ] ) ):
        meas = lambda w: numpy.asarray( PowerDiagram( pos, w, distribution = dist, kernel_dtype = "FP64", sink_weight = ws ).measures ).reshape( -1 )
        # the facets, and the sphere: the last slot of a row, under the cell's own identifier
        cnt, ids, vals = PowerDiagram( pos, w, distribution = dist, kernel_dtype = "FP64", sink_weight = ws ).hessian_rows()
        H = numpy.zeros( ( n, n ) )
        for i in range( n ):
            for r in range( cnt[ i ] ):
                j = ids[ i, r ]
                if j == i:
                    H[ i, i ] += vals[ i, r ]
                elif j >= 0:
                    H[ i, j ] -= vals[ i, r ]
                    H[ i, i ] += vals[ i, r ]
        F = numpy.stack( [ ( meas( w + eps * e ) - meas( w - eps * e ) ) / ( 2 * eps ) for e in numpy.eye( n ) ], axis = 1 )
        assert numpy.abs( H - F ).max() < 1e-6 * numpy.abs( F ).max(), numpy.abs( H - F ).max()


if test( "measures_and_moments_inside_balls_derive" ):
    import numpy
    from loom.testing import need, check_grad
    from sdot.distributions.Box import Box
    from sdot.distributions.Image import Image
    from sdot.PowerDiagram import PowerDiagram
    from sdot.Cell import set_kernel_dtype
    set_kernel_dtype( "FP64" )
    need( "grad" )
    # the ball moves with its seed: its center with the position, its radius with the weight ( `InBallDensity::seed_bwd` ),
    # and the polynomial of a moment is integrated inside it ( `diagram/Moments.h` )
    rng = numpy.random.default_rng( 0 )
    n, ws = 10, -0.005
    pos = rng.uniform( 0, 1, size = ( n, 2 ) )
    w = rng.uniform( 0.0, 0.04, size = n )
    for dist in ( Box( nb_dims = 2 ), Image( rng.uniform( 0.5, 2, size = ( 7, 5 ) ), frame = [ 1.0, 1.0 ] ) ):
        mk = lambda p, q: PowerDiagram( p, weights = q, distribution = dist, sink_weight = ws, accelerator = "plain" )
        check_grad( lambda p, q: mk( p, q ).measures, pos, w, seed = 1, eps = 1e-6, rtol = 1e-5, atol = 1e-7 )
        for k in range( 4 ):
            check_grad( lambda p, q, k = k: mk( p, q ).moment( k ), pos, w, seed = 2 + k, eps = 1e-6, rtol = 1e-5, atol = 1e-7 )


if test( "isolated_balls_hold_their_mass_in_closed_form" ):
    import numpy
    from sdot.OtProblem import ot_solve
    from sdot.distributions.Box import Box
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from loom.testing import need
    need( "cpu" )
    # well apart and light: every cell is a disc of mass `nu` -- `w = nu / pi`, a cost `nu^2 / ( 2 pi )` each
    g = ( numpy.arange( 4 ) + 0.5 ) / 4
    pos = numpy.stack( numpy.meshgrid( g, g, indexing = "ij" ), axis = -1 ).reshape( -1, 2 )
    nu = 0.1 / len( pos )
    p = ot_solve( SumOfDiracs( pos, target_mass = 0.1 ), Box(), tol = 1e-13 )
    assert p.converged
    assert numpy.allclose( numpy.asarray( p.weights ).reshape( -1 ), nu / numpy.pi, atol = 1e-12 )
    assert abs( float( numpy.asarray( p.cost ) ) - len( pos ) * nu ** 2 / ( 2 * numpy.pi ) ) < 1e-13
    assert numpy.allclose( numpy.asarray( p.barycenters ).reshape( -1, 2 ), pos, atol = 1e-12 )


if test( "a_partial_plan_reaches_the_masses" ):
    import numpy
    from loom.testing import need
    from sdot.OtProblem import ot_solve
    from sdot.distributions.Image import Image
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    need( "cpu" )
    rng = numpy.random.default_rng( 1 )
    n = 40
    pos = rng.uniform( 0, 1, size = ( n, 2 ) )
    img = Image( rng.uniform( 0.2, 2, size = ( 9, 7 ) ), frame = [ 1.0, 1.0 ] )
    for tm in ( 0.2, 0.6, 0.95 ):
        p = ot_solve( SumOfDiracs( pos, target_mass = tm ), img, tol = 1e-11 )
        assert p.converged, p.stats[ "status" ]
        assert numpy.abs( numpy.asarray( p.cell_masses ).reshape( -1 ) - tm / n ).max() < 1e-10
        assert numpy.allclose( numpy.asarray( p.target_masses ).reshape( -1 ), tm / n )
        # the cost and the barycenters: those of the diagram of the plan
        mm, f, q = p.power_diagram().moments
        mm, f, q = ( numpy.asarray( x.value ).reshape( n, -1 ) for x in ( mm, f, q ) )
        cost = ( q[ :, 0 ] - 2 * ( pos * f ).sum( 1 ) + ( pos ** 2 ).sum( 1 ) * mm[ :, 0 ] ).sum()
        assert abs( cost - float( numpy.asarray( p.cost ) ) ) < 1e-12
        assert numpy.allclose( numpy.asarray( p.barycenters ).reshape( n, 2 ), f / mm, atol = 1e-10 )


if test( "a_partial_plan_is_differentiable" ):
    import numpy
    from loom.testing import check_grad, need
    from sdot.OtProblem import Iterative, OtProblem
    from sdot.distributions.Image import Image
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    need( "cpu" )
    need( "grad" )
    # through the implicit function theorem ( `_attach_derivatives` ), the sink being the gauge: against the positions, the
    # masses of the diracs ( a source of mass 0.6 ) and the values of the image
    rng = numpy.random.default_rng( 3 )
    n = 10
    pos = rng.uniform( 0.1, 0.9, size = ( n, 2 ) )
    nu = rng.uniform( 0.5, 1.5, size = n ) * 0.6 / n
    values = rng.uniform( 0.5, 1.5, size = ( 3, 3 ) )
    settings = Iterative( tol = 1e-13, max_iter = 200 )

    def solve( p, q, v ):
        return OtProblem( SumOfDiracs( p, weights = q, target_mass = None ), Image( values = v, frame = [ 1.0, 1.0 ] ) ).solve( settings )

    for output, tol in ( ( "weights", 1e-5 ), ( "cell_masses", 1e-5 ), ( "cost", 1e-4 ), ( "barycenters", 1e-4 ) ):
        check_grad( lambda p, q, v: getattr( solve( p, q, v ), output ), pos, nu, values, seed = 3, eps = 1e-6, rtol = tol, atol = tol * 1e-2 )
    # the envelope: `2 m ( p - b )`, the weights for the masses
    plan = solve( pos, nu, values )
    _, grad = plan.cost_and_position_grad()
    assert numpy.asarray( grad ).shape == ( n, 2 )


if test( "what_partial_transport_refuses" ):
    import numpy
    from sdot.OtProblem import Tuning, ot_solve
    from sdot.distributions.Box import Box
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from sdot.distributions.SumOfGaussians import SumOfGaussians
    from sdot.PowerDiagram import PowerDiagram
    rng = numpy.random.default_rng( 0 )
    pos = rng.uniform( 0, 1, size = ( 8, 2 ) )

    def raises( f, what ):
        try:
            f()
        except ( NotImplementedError, ValueError ) as e:
            assert what in str( e ), str( e )
            return
        raise AssertionError( f"no error ( expected: { what } )" )

    raises( lambda: ot_solve( SumOfDiracs( pos, target_mass = 1.5 ), Box() ), "weighs more than the target" )
    raises( lambda: PowerDiagram( rng.uniform( 0, 1, size = ( 8, 3 ) ), distribution = Box( nb_dims = 3 ), sink_weight = 0.0 ), "2D only" )
    raises( lambda: PowerDiagram( pos, distribution = SumOfGaussians( [ [ 0.5, 0.5 ] ], [ 0.2 ] ), sink_weight = 0.0 ), "constant on pieces" )
    raises( lambda: ot_solve( SumOfDiracs( pos, target_mass = 0.5 ), Box(), periodicity = True ), "periodicity" )
    # ( on a CUDA device, the partial solve is refused before its settings are read )
    import loom
    raises( lambda: ot_solve( SumOfDiracs( pos, target_mass = 0.5 ), Box(), tuning = Tuning( step = "limits" ) ),
            "limits" if loom.resolved_device().is_cpu else "on the CPU only" )


if test( "equal_masses_balance" ):
    import numpy
    from loom.testing import need
    from sdot.OtProblem import ot_solve
    from sdot.distributions.Box import Box
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    need( "cpu" )
    pos = numpy.random.default_rng( 0 ).uniform( 0, 1, size = ( 8, 2 ) )
    # two equal `target_mass` balance exactly ( no sink ), a heavier box of the same density too
    assert not ot_solve( SumOfDiracs( pos, target_mass = 0.5 ), Box( target_mass = 0.5 ) ).power_diagram().sink_weight.is_defined
    assert not ot_solve( SumOfDiracs( pos ), Box() ).power_diagram().sink_weight.is_defined
    assert ot_solve( SumOfDiracs( pos, target_mass = 0.5 ), Box() ).power_diagram().sink_weight.is_defined


if test( "a_partial_plan_is_drawn_with_its_balls" ):
    import numpy
    from loom.testing import need
    from sdot.OtProblem import ot_solve
    from sdot.distributions.Box import Box
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from sdot.PowerDiagram import _disc_in_polygon
    from sdot.viz.Visualizer import Visualizer
    need( "cpu" )
    pos = numpy.random.default_rng( 1 ).uniform( 0, 1, size = ( 10, 2 ) )
    viz = ot_solve( SumOfDiracs( pos, target_mass = 0.3 ), Box() ).add_to_viz( Visualizer() )
    assert viz.nb_frames == 1
    sq = numpy.array( [ [ 0, 0 ], [ 1, 0 ], [ 1, 1 ], [ 0, 1 ] ], dtype = float )
    x, y = _disc_in_polygon( sq, numpy.array( [ 0.0, 0.0 ] ), 0.5, nb = 400 ).T
    assert abs( ( x * numpy.roll( y, -1 ) - numpy.roll( x, -1 ) * y ).sum() / 2 - numpy.pi / 16 ) < 1e-4
