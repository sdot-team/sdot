from errand import test


# PARTIAL TRANSPORT IN 3D: the cell of seed `i` is integrated inside its ball `|x - p_i|^2 <= w_i - w_s` only
# ( `integration/InBall3.h` ). THE REFERENCES: closed forms for a ball cut by a plane, and for a Voronoi diagram ( equal weights:
# every seed inside its cell ) an integration along the rays from the seed, `int min( t( e ), r )^( 3 + k ) / ( 3 + k ) de`
# over a fine set of directions, `t( e )` the exit of the ray from the cell -- no code in common with the kernels.


def _fibonacci_sphere( n ):
    import numpy
    k = numpy.arange( n ) + 0.5
    z = 1 - 2 * k / n
    phi = numpy.pi * ( 1 + 5 ** 0.5 ) * k
    s = numpy.sqrt( 1 - z * z )
    return numpy.stack( [ s * numpy.cos( phi ), s * numpy.sin( phi ), z ], axis = 1 )


def _voronoi_ball_moments( pos, i, r, nb_dirs = 400_000 ):
    """mass and first moment of ( the Voronoi cell of `pos[ i ]` in the unit cube ) inter the ball of radius `r` about it"""
    import numpy
    e = _fibonacci_sphere( nb_dirs )
    p = pos[ i ]
    t = numpy.full( nb_dirs, numpy.inf )
    for j in range( len( pos ) ):                        # the bisectors: ( x - m ) . d <= 0
        if j == i:
            continue
        d = pos[ j ] - p
        den = e @ d
        dist = ( d @ d ) / 2
        ok = den > 0
        t[ ok ] = numpy.minimum( t[ ok ], dist / den[ ok ] )
    for a in range( 3 ):                                 # the walls of the cube
        ok = e[ :, a ] > 0
        t[ ok ] = numpy.minimum( t[ ok ], ( 1 - p[ a ] ) / e[ ok, a ] )
        ok = e[ :, a ] < 0
        t[ ok ] = numpy.minimum( t[ ok ], - p[ a ] / e[ ok, a ] )
    s = numpy.minimum( t, r )
    dO = 4 * numpy.pi / nb_dirs
    return ( s ** 3 / 3 ).sum() * dO, p * ( s ** 3 / 3 ).sum() * dO + ( e * ( s ** 4 / 4 )[ :, None ] ).sum( 0 ) * dO


if test( "a_ball_inside_its_cell_holds_four_thirds_pi_r3" ):
    import numpy
    from sdot.distributions.Box import Box
    from sdot.PowerDiagram import PowerDiagram
    pos = numpy.array( [ [ 0.3, 0.3, 0.3 ], [ 0.7, 0.7, 0.6 ] ] )
    w = numpy.array( [ 0.01, 0.02 ] )
    pd = PowerDiagram( pos, w, distribution = Box( nb_dims = 3 ), kernel_dtype = "FP64", sink_weight = 0.0 )
    vol = 4 / 3 * numpy.pi * w ** 1.5
    assert numpy.allclose( numpy.asarray( pd.measures ).reshape( -1 ), vol, atol = 1e-13 )
    _, f, q = pd.moments
    assert numpy.allclose( numpy.asarray( f.value ).reshape( 2, 3 ), pos * vol[ :, None ], atol = 1e-13 )
    # int |x|^2 = int |x - p|^2 + |p|^2 m = 4 pi r^5 / 5 + |p|^2 m
    assert numpy.allclose( numpy.asarray( q.value ).reshape( -1 ), 4 * numpy.pi * w ** 2.5 / 5 + ( pos ** 2 ).sum( 1 ) * vol, atol = 1e-13 )


if test( "a_ball_cut_by_a_wall_in_closed_form" ):
    import numpy
    from sdot.distributions.Box import Box
    from sdot.PowerDiagram import PowerDiagram
    # the ball about `( d, y, z )` cut by the plane `x = 0`: it loses the cap of height `h = r - d`, of volume `pi h^2 ( 3 r - h ) / 3`
    # and of first moment `- pi ( r^2 - d^2 )^2 / 4` along `x` about the center
    r, d = 0.2, 0.07
    pos = numpy.array( [ [ d, 0.5, 0.4 ] ] )
    pd = PowerDiagram( pos, numpy.array( [ r * r ] ), distribution = Box( nb_dims = 3 ), kernel_dtype = "FP64", sink_weight = 0.0 )
    h = r - d
    vol = 4 / 3 * numpy.pi * r ** 3 - numpy.pi * h * h * ( 3 * r - h ) / 3
    assert abs( float( numpy.asarray( pd.measures ).reshape( -1 )[ 0 ] ) - vol ) < 1e-13
    _, f, _ = pd.moments
    first = pos[ 0 ] * vol + numpy.array( [ numpy.pi * ( r * r - d * d ) ** 2 / 4, 0, 0 ] )
    assert numpy.abs( numpy.asarray( f.value ).reshape( -1 ) - first ).max() < 1e-13


if test( "measures_inside_balls_match_the_rays" ):
    import numpy
    from sdot.distributions.Box import Box
    from sdot.PowerDiagram import PowerDiagram
    rng = numpy.random.default_rng( 0 )
    n = 12
    pos = rng.uniform( 0.1, 0.9, size = ( n, 3 ) )
    for ws in ( -0.01, -0.03, -0.08 ):                   # balls inside, across, and over the Voronoi cells
        pd = PowerDiagram( pos, numpy.zeros( n ), distribution = Box( nb_dims = 3 ), kernel_dtype = "FP64", sink_weight = ws )
        m = numpy.asarray( pd.measures ).reshape( -1 )
        _, f, _ = pd.moments
        f = numpy.asarray( f.value ).reshape( n, 3 )
        for i in range( n ):
            mi, fi = _voronoi_ball_moments( pos, i, numpy.sqrt( - ws ) )
            assert abs( m[ i ] - mi ) < 1e-4 * mi + 1e-9, ( ws, i, m[ i ], mi )
            assert numpy.abs( f[ i ] - fi ).max() < 1e-4 * mi + 1e-9, ( ws, i, f[ i ], fi )


if test( "the_hessian_3d_has_the_sphere_on_its_diagonal" ):
    import numpy
    from sdot.distributions.Box import Box
    from sdot.distributions.Image import Image
    from sdot.PowerDiagram import PowerDiagram
    rng = numpy.random.default_rng( 0 )
    n, ws, eps = 10, -0.01, 1e-6
    pos = rng.uniform( 0, 1, size = ( n, 3 ) )
    w = rng.uniform( 0.0, 0.03, size = n )
    for dist in ( Box( nb_dims = 3 ), Image( rng.uniform( 0.5, 2, size = ( 3, 4, 3 ) ), frame = [ 1.0, 1.0, 1.0 ] ) ):
        meas = lambda w: numpy.asarray( PowerDiagram( pos, w, distribution = dist, kernel_dtype = "FP64", sink_weight = ws ).measures ).reshape( -1 )
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


if test( "measures_and_moments_inside_balls_derive_in_3d" ):
    import numpy
    from loom.testing import need, check_grad
    from sdot.distributions.Box import Box
    from sdot.distributions.Image import Image
    from sdot.PowerDiagram import PowerDiagram
    from sdot.Cell import set_kernel_dtype
    set_kernel_dtype( "FP64" )
    need( "grad" )
    # the faces of the cells inside the balls ( `simplex_bwd` ), and the balls themselves ( `seed_bwd`: the solid moments )
    rng = numpy.random.default_rng( 0 )
    n, ws = 8, -0.01
    pos = rng.uniform( 0, 1, size = ( n, 3 ) )
    w = rng.uniform( 0.0, 0.03, size = n )
    for dist in ( Box( nb_dims = 3 ), Image( rng.uniform( 0.5, 2, size = ( 3, 4, 3 ) ), frame = [ 1.0, 1.0, 1.0 ] ) ):
        mk = lambda p, q: PowerDiagram( p, weights = q, distribution = dist, sink_weight = ws, accelerator = "plain" )
        check_grad( lambda p, q: mk( p, q ).measures, pos, w, seed = 1, eps = 1e-6, rtol = 1e-5, atol = 1e-7 )
        for k in range( 5 ):
            check_grad( lambda p, q, k = k: mk( p, q ).moment( k ), pos, w, seed = 2 + k, eps = 1e-6, rtol = 1e-5, atol = 1e-7 )


def _cube_mesh( values ):
    """the unit cube, a grid of `values.shape` cubes cut into 6 tetrahedra each, every tetrahedron with the value of its cube: `Mesh(
    degree = 0 )` with the density of `Image( values )`"""
    import numpy
    from sdot.distributions.Mesh import Mesh
    k = values.shape
    g = [ numpy.linspace( 0, 1, n + 1 ) for n in k ]
    nodes = numpy.stack( numpy.meshgrid( *g, indexing = "ij" ), -1 ).reshape( -1, 3 )
    I = lambda i, j, l: ( i * ( k[ 1 ] + 1 ) + j ) * ( k[ 2 ] + 1 ) + l
    tets, vals = [], []
    for i in range( k[ 0 ] ):
        for j in range( k[ 1 ] ):
            for l in range( k[ 2 ] ):
                c = [ I( i + a, j + b, l + e ) for a in ( 0, 1 ) for b in ( 0, 1 ) for e in ( 0, 1 ) ]
                for q in ( ( 1, 3 ), ( 3, 2 ), ( 2, 6 ), ( 6, 4 ), ( 4, 5 ), ( 5, 1 ) ):
                    tets.append( [ c[ 0 ], c[ q[ 0 ] ], c[ q[ 1 ] ], c[ 7 ] ] )
                    vals.append( values[ i, j, l ] )
    return Mesh( nodes, numpy.array( tets ), values = numpy.repeat( numpy.array( vals )[ :, None ], 4, axis = 1 ),
                 degree = 0 )


if test( "a_mesh_of_degree_0_in_3d_is_its_image" ):
    import numpy
    from sdot.distributions.Image import Image
    from sdot.PowerDiagram import PowerDiagram
    # the same density, cut by the tetrahedra or by the pixels: the measures, the moments and the hessian ( its facets inside the
    # balls, `InBallDist::polygon_area_in_ball` ) agree
    rng = numpy.random.default_rng( 0 )
    n = 10
    values = rng.uniform( 0.5, 2, size = ( 3, 4, 3 ) )
    mesh, img = _cube_mesh( values ), Image( values, frame = [ 1.0, 1.0, 1.0 ] )
    pos = rng.uniform( 0.05, 0.95, size = ( n, 3 ) )
    w = rng.uniform( 0.0, 0.03, size = n )
    for ws in ( -0.01, -0.04 ):
        a, b = ( PowerDiagram( pos, w, distribution = d, kernel_dtype = "FP64", sink_weight = ws ) for d in ( mesh, img ) )
        assert numpy.allclose( numpy.asarray( a.measures ), numpy.asarray( b.measures ), rtol = 1e-10, atol = 1e-13 )
        for ma, mb in zip( a.moments, b.moments ):
            assert numpy.allclose( numpy.asarray( ma.value ), numpy.asarray( mb.value ), rtol = 1e-10, atol = 1e-13 )
        assert numpy.allclose( numpy.asarray( a.sphere_areas ), numpy.asarray( b.sphere_areas ), rtol = 1e-10, atol = 1e-13 )

        def dense( pd ):
            cnt, ids, vals = pd.hessian_rows()
            H = numpy.zeros( ( n, n ) )
            for i in range( n ):
                for r in range( cnt[ i ] ):
                    j = ids[ i, r ]
                    if j == i:
                        H[ i, i ] += vals[ i, r ]
                    elif j >= 0:
                        H[ i, j ] -= vals[ i, r ]
                        H[ i, i ] += vals[ i, r ]
            return H
        Ha, Hb = dense( a ), dense( b )
        assert numpy.abs( Ha - Hb ).max() < 1e-10 * numpy.abs( Hb ).max(), numpy.abs( Ha - Hb ).max()


if test( "a_mesh_of_degree_0_derives_in_3d" ):
    import numpy
    from loom.testing import need, check_grad
    from sdot.PowerDiagram import PowerDiagram
    from sdot.Cell import set_kernel_dtype
    set_kernel_dtype( "FP64" )
    need( "grad" )
    rng = numpy.random.default_rng( 0 )
    n, ws = 8, -0.01
    mesh = _cube_mesh( rng.uniform( 0.5, 2, size = ( 2, 3, 2 ) ) )
    pos = rng.uniform( 0.05, 0.95, size = ( n, 3 ) )
    w = rng.uniform( 0.0, 0.03, size = n )
    mk = lambda p, q: PowerDiagram( p, weights = q, distribution = mesh, sink_weight = ws, accelerator = "plain" )
    check_grad( lambda p, q: mk( p, q ).measures, pos, w, seed = 1, eps = 1e-6, rtol = 1e-5, atol = 1e-7 )
    for k in range( 5 ):
        check_grad( lambda p, q, k = k: mk( p, q ).moment( k ), pos, w, seed = 2 + k, eps = 1e-6, rtol = 1e-5, atol = 1e-7 )


if test( "a_partial_plan_in_3d_reaches_its_masses" ):
    import numpy
    from sdot.OtProblem import ot_solve
    from sdot.distributions.Box import Box
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from loom.testing import need
    need( "cpu" )
    rng = numpy.random.default_rng( 0 )
    for mass in ( 0.05, 0.3, 0.8 ):                      # isolated balls, touching, nearly balanced
        pos = rng.uniform( 0.1, 0.9, size = ( 40, 3 ) )
        p = ot_solve( SumOfDiracs( pos, target_mass = mass ), Box( nb_dims = 3 ) )
        assert p.converged
        assert numpy.allclose( numpy.asarray( p.cell_masses ).reshape( -1 ), mass / 40, rtol = 1e-6 )


if test( "what_partial_transport_refuses_in_3d" ):
    import numpy
    from sdot.distributions.Image import Image
    from sdot.PowerDiagram import PowerDiagram
    rng = numpy.random.default_rng( 0 )

    def raises( f, what ):
        try:
            f()
        except ( NotImplementedError, ValueError ) as e:
            assert what in str( e ), str( e )
            return
        raise AssertionError( f"no error ( expected: { what } )" )

    pos = rng.uniform( 0, 1, size = ( 8, 3 ) )
    smooth = Image( rng.uniform( 0.5, 2, size = ( 3, 3, 3, 8 ) ), frame = [ 1.0, 1.0, 1.0 ], degree = 1 )
    raises( lambda: PowerDiagram( pos, distribution = smooth, sink_weight = 0.0 ), "constant on pieces" )
    raises( lambda: PowerDiagram( rng.uniform( 0, 1, size = ( 8, 4 ) ), sink_weight = 0.0 ), "2D and 3D only" )
    from sdot.distributions.Mesh import Mesh
    affine = Mesh( rng.uniform( 0, 1, size = ( 4, 3 ) ), numpy.array( [ [ 0, 1, 2, 3 ] ] ) )
    raises( lambda: PowerDiagram( pos, distribution = affine, sink_weight = 0.0 ), "constant on pieces" )


if test( "a_partial_plan_in_3d_is_drawn_by_the_boundary_of_its_matter" ):
    import numpy
    import sdot
    from sdot.distributions.Box import Box
    from sdot.PowerDiagram import PowerDiagram
    # one ball inside its cell ( a whole icosphere ), one cut by the wall `x = 0` ( a cap missing, a disc on the wall )
    pd = PowerDiagram( numpy.array( [ [ 0.5, 0.5, 0.5 ], [ 0.05, 0.2, 0.2 ] ] ), numpy.array( [ 0.01, 0.01 ] ), distribution = Box( nb_dims = 3 ),
                       sink_weight = 0.0 )
    for res in ( 10, 40 ):
        viz = sdot.Visualizer()
        pd.add_to_viz( viz, seeds = False, opacity_beyond_balls = 0, ball_resolution = res, support_only = False )
        P = viz.positions
        r = numpy.linalg.norm( P - numpy.array( [ 0.5, 0.5, 0.5 ] ), axis = 1 )
        assert ( numpy.abs( r - 0.1 ) < 1e-6 ).any()     # the whole sphere
        assert ( numpy.abs( P[ :, 0 ] ) < 1e-9 ).sum() >= 3   # the disc on the wall
        assert P[ :, 0 ].min() > -1e-9                   # nothing beyond the wall


if test( "sphere_areas_in_closed_form" ):
    import numpy
    from sdot.distributions.Box import Box
    from sdot.PowerDiagram import PowerDiagram
    # the sphere inside each cell: a whole one, and one cut by the wall `x = 0` at a distance `d` ( a zone of height `r + d` )
    r, d = 0.2, 0.07
    for dim, whole, cut in ( ( 2, 2 * numpy.pi * r, r * ( 2 * numpy.pi - 2 * numpy.arccos( d / r ) ) ),
                             ( 3, 4 * numpy.pi * r * r, 2 * numpy.pi * r * ( r + d ) ) ):
        pos = numpy.array( [ [ 0.5 ] * dim, [ d ] + [ 0.25 ] * ( dim - 1 ) ] )
        pd = PowerDiagram( pos, numpy.array( [ r * r ] * 2 ), distribution = Box( nb_dims = dim ), kernel_dtype = "FP64", sink_weight = 0.0 )
        assert numpy.allclose( numpy.asarray( pd.sphere_areas ).reshape( -1 ), [ whole, cut ], atol = 1e-12 ), ( dim, pd.sphere_areas )
        # no sink: no sphere
        assert numpy.allclose( numpy.asarray( PowerDiagram( pos, distribution = Box( nb_dims = dim ) ).sphere_areas ), 0 )


if test( "sphere_areas_derive" ):
    import numpy
    from loom.testing import need, check_grad
    from sdot.distributions.Box import Box
    from sdot.distributions.Image import Image
    from sdot.PowerDiagram import PowerDiagram
    from sdot.Cell import set_kernel_dtype
    set_kernel_dtype( "FP64" )
    need( "grad" )
    # the faces of the cells crossing the spheres, the centers and the radii ( `sphere_length_bwd`, `sphere_area_bwd`, the seed's share );
    # an image with a zero pixel: the sphere there is not counted, its edges move like walls
    rng = numpy.random.default_rng( 0 )
    for dim, n, ws in ( ( 2, 10, -0.005 ), ( 3, 8, -0.01 ) ):
        pos = rng.uniform( 0, 1, size = ( n, dim ) )
        w = rng.uniform( 0.0, 0.03, size = n )
        img = rng.uniform( 0.5, 2, size = ( 3, ) * dim )
        img.reshape( -1 )[ 4 ] = 0
        for dist in ( Box( nb_dims = dim ), Image( img, frame = [ 1.0 ] * dim ) ):
            mk = lambda p, q: PowerDiagram( p, weights = q, distribution = dist, sink_weight = ws, accelerator = "plain" )
            check_grad( lambda p, q: mk( p, q ).sphere_areas, pos, w, seed = 1, eps = 1e-6, rtol = 1e-5, atol = 1e-7 )


if test( "a_partial_plan_gives_its_sphere_areas_with_their_derivatives" ):
    import numpy
    from loom.testing import need, check_grad
    from sdot.OtProblem import OtProblem
    from sdot.distributions.Box import Box
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    need( "grad" )
    # through the weights: the masses stay what they are when a dirac moves
    rng = numpy.random.default_rng( 1 )
    for dim in ( 2, 3 ):
        pos = rng.uniform( 0.2, 0.8, size = ( 12, dim ) )
        masses = numpy.full( 12, 0.4 / 12 )
        f = lambda p, m: OtProblem( SumOfDiracs( p, m, target_mass = None ), Box( nb_dims = dim ) ).solve().sphere_areas
        check_grad( f, pos, masses, seed = 3, eps = 1e-6, rtol = 1e-4, atol = 1e-6 )
