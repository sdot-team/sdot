"""A sum of gaussians on the FACETS of a cell ( `SumOfGaussians::facet_mass` ): what the laplacian of a transport reads
( `sdotplan/Sweep.h`, `PowerDiagram.hessian_rows` ). In 3D a facet is a planar polygon, on which an isotropic gaussian splits
into a 1D factor across the plane and the exact 2D reduction in it."""

from errand import test


def _dense_hessian( pd, n ):
    import numpy
    counts, ids, vals = pd.hessian_rows()
    H = numpy.zeros( ( n, n ) )
    for i in range( n ):
        for q in range( counts[ i ] ):
            if ids[ i, q ] >= 0:
                H[ i, ids[ i, q ] ] -= vals[ i, q ]
                H[ i, i ] += vals[ i, q ]
    return H


if test( "the_3d_facet_mass_of_gaussians_is_exact_on_any_plane" ):
    # two seeds in the unit cube: their facet is the square `x = 1/2`, on which a gaussian is a product of `erf`s. Then the same
    # configuration ROTATED ( seeds, centers, the half-spaces of the cube ): an oblique facet, the same masses -- the fan of
    # triangles and the in-plane frame of `SumOfGaussians::facet_mass_3d`, against the closed form
    import numpy
    from math import erf, exp, pi, sqrt
    from sdot.PowerDiagram import PowerDiagram, box_half_spaces
    from sdot.distributions.SumOfGaussians import SumOfGaussians
    c = numpy.array( [ [ 0.43, 0.58, 0.47 ], [ 0.52, 0.31, 0.66 ], [ 0.2, 0.7, 0.5 ] ] )
    s = numpy.array( [ 0.17, 0.05, 0.3 ] )                 # narrow ( the exact 2D reduction matters ), mid, wide
    w = numpy.array( [ 0.5, 1.0, 0.7 ] )
    pos = numpy.array( [ [ 0.25, 0.5, 0.5 ], [ 0.75, 0.5, 0.5 ] ] )
    seg = lambda a, s: 0.5 * ( erf( ( 1 - a ) / ( s * sqrt( 2 ) ) ) - erf( ( 0 - a ) / ( s * sqrt( 2 ) ) ) )
    facet = sum( w[ i ] / w.sum() * exp( - ( 0.5 - c[ i, 0 ] ) ** 2 / ( 2 * s[ i ] ** 2 ) ) / ( s[ i ] * sqrt( 2 * pi ) )
                 * seg( c[ i, 1 ], s[ i ] ) * seg( c[ i, 2 ], s[ i ] ) for i in range( 3 ) )
    ref = facet / ( 2 * 0.5 )                              # `d mass_0 / d w_1 = facet / ( 2 | p_1 - p_0 | )`
    dirs, offs = box_half_spaces( [ 0 ] * 3, [ 1 ] * 3 )
    rng = numpy.random.default_rng( 35 )
    for r in range( 3 ):
        R = numpy.eye( 3 ) if r == 0 else numpy.linalg.qr( rng.normal( size = ( 3, 3 ) ) )[ 0 ]
        pd = PowerDiagram( pos @ R.T, boundaries = ( dirs @ R.T, offs ), kernel_dtype = "FP64",
                           distribution = SumOfGaussians( positions = c @ R.T, sigmas = s, weights = w ) )
        H = _dense_hessian( pd, 2 )
        assert abs( - H[ 0, 1 ] - ref ) < 1e-12 * ref, ( r, - H[ 0, 1 ], ref )


if test( "the_3d_facet_mass_of_gaussians_is_symmetric" ):
    # many seeds, oblique facets cut by their neighbours: each facet is integrated from both sides ( two cells, two fans, two
    # frames ), and must agree. ( No finite difference here: the measures of a 3D gaussian come from an adaptive quadrature,
    # `PointwiseDensity`, whose noise is above the checks this one makes. )
    import numpy
    from sdot.PowerDiagram import PowerDiagram, box_half_spaces
    from sdot.distributions.SumOfGaussians import SumOfGaussians
    rng = numpy.random.default_rng( 33 )
    n = 30
    pos = rng.uniform( 0.1, 0.9, size = ( n, 3 ) )
    w0 = rng.uniform( -0.01, 0.01, n )
    dens = SumOfGaussians( positions = rng.uniform( 0.3, 0.7, size = ( 3, 3 ) ), sigmas = [ 0.04, 0.2, 0.35 ], weights = [ 0.5, 1.0, 0.7 ] )
    pd = PowerDiagram( pos, w0, boundaries = box_half_spaces( [ 0 ] * 3, [ 1 ] * 3 ), distribution = dens, kernel_dtype = "FP64" )
    H = _dense_hessian( pd, n )
    assert ( H - numpy.diag( numpy.diag( H ) ) < 0 ).sum() > 2 * n, "too few facets"
    assert numpy.abs( H - H.T ).max() < 1e-12 * numpy.abs( H ).max(), numpy.abs( H - H.T ).max()


if test( "the_3d_transport_towards_gaussians_is_solved" ):
    # the solver's laplacian reads `facet_mass`: in 3D it did not compile before ( `sdotplan/Sweep.h`, `density_facet_mass` )
    import numpy
    from sdot.OtProblem import Iterative, OtProblem
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from sdot.distributions.SumOfGaussians import SumOfGaussians
    rng = numpy.random.default_rng( 34 )
    n = 40
    dens = SumOfGaussians( positions = rng.uniform( 0.35, 0.65, size = ( 4, 3 ) ), sigmas = [ 0.15, 0.2, 0.18, 0.25 ] )
    # ( the measures of a 3D gaussian are exact too -- `SumOfGaussians::face3` -- so Newton converges as tight as with a constant )
    plan = OtProblem( SumOfDiracs( rng.uniform( 0.2, 0.8, size = ( n, 3 ) ) ), dens ).solve( Iterative( tol = 1e-10 / n ) )
    m = numpy.asarray( plan.cell_masses ).reshape( -1 )
    assert plan.converged and numpy.abs( m * n - 1 ).max() < 1e-8, ( plan.stats[ "status" ], numpy.abs( m * n - 1 ).max() )


if test( "the_3d_cells_integrate_gaussians_by_their_edges" ):
    # `measures` and `moments` integrate a 3D cell at once ( `SumOfGaussians::cell_integrals`: one cone edge per edge and face of
    # the polyhedron ), `moment( k )` still tetrahedron by tetrahedron: the same integrals, and in a box the cells sum to the closed form
    import numpy
    from math import erf, sqrt
    from sdot.PowerDiagram import PowerDiagram, box_half_spaces
    from sdot.distributions.SumOfGaussians import SumOfGaussians
    rng = numpy.random.default_rng( 36 )
    ng, n = 12, 60
    c = rng.uniform( 0.2, 0.8, size = ( ng, 3 ) )
    s = rng.choice( [ 0.02, 0.07, 0.25 ], ng )            # narrow ( against the cells ), mid, wide
    w = rng.uniform( 0.5, 1.0, ng )
    pd = PowerDiagram( rng.uniform( 0, 1, size = ( n, 3 ) ), rng.uniform( -0.002, 0.002, n ), boundaries = box_half_spaces( [ 0 ] * 3, [ 1 ] * 3 ),
                       kernel_dtype = "FP64", distribution = SumOfGaussians( positions = c, sigmas = s, weights = w ) )
    m = numpy.asarray( pd.measures )
    ref = [ numpy.asarray( pd.moment( k ) ) for k in range( 5 ) ]
    mass, first, second = [ numpy.asarray( a ) for a in pd.moments ]
    a = w / w.sum()
    box = sum( a[ i ] * numpy.prod( [ ( erf( ( 1 - c[ i, d ] ) / ( s[ i ] * sqrt( 2 ) ) ) + erf( c[ i, d ] / ( s[ i ] * sqrt( 2 ) ) ) ) / 2 for d in range( 3 ) ] )
               for i in range( ng ) )
    assert abs( m.sum() - box ) < 1e-13, ( m.sum(), box )
    assert numpy.abs( m - ref[ 4 ] ).max() < 1e-13, numpy.abs( m - ref[ 4 ] ).max()
    assert numpy.abs( mass.reshape( -1 ) - ref[ 4 ] ).max() < 1e-13
    assert numpy.abs( first.reshape( n, 3 ) - numpy.stack( ref[ :3 ], -1 ) ).max() < 1e-13
    assert numpy.abs( second.reshape( -1 ) - ref[ 3 ] ).max() < 1e-13


if test( "the_support_follows_traced_parameters" ):
    import numpy
    from loom.testing import need
    import sdot
    need( "cpu" )
    need( "grad" )
    import jax
    import jax.numpy as jnp
    # no `support_box`: the domain is the box of the gaussians up to `support_sigmas`, read off their parameters even when
    # they are traced ( `bounding_half_spaces`: traced offsets, detached ) -- what lets a fit differentiate, and jit, with the
    # domain following the gaussians
    rng = numpy.random.default_rng( 1 )
    x = jnp.asarray( rng.normal( 0, 0.5, size = ( 40, 2 ) ) )
    def w2( c, s, m ):
        return sdot.w2( sdot.SumOfDiracs( x ), sdot.SumOfGaussians( c, s, weights = m ) )
    args = ( jnp.asarray( [ [ -0.3, 0.1 ], [ 0.4, -0.2 ] ] ), jnp.asarray( [ 0.5, 0.4 ] ), jnp.asarray( [ 1.0, 0.7 ] ) )
    g = jax.grad( w2, argnums = ( 0, 1, 2 ) )( *args )
    gj = jax.jit( jax.grad( w2, argnums = ( 0, 1, 2 ) ) )( *args )
    for a, b in zip( g, gj ):
        assert numpy.isfinite( numpy.asarray( a ) ).all() and numpy.abs( numpy.asarray( a ) ).max() > 0
        assert numpy.allclose( numpy.asarray( a ), numpy.asarray( b ), rtol = 1e-8, atol = 1e-12 ), ( a, b )


if test( "the_2d_moments_of_gaussians_are_exact" ):
    import numpy
    from loom.testing import need
    import sdot
    need( "cpu" )
    # `PowerDiagram.moment` on gaussians in 2D: the divergence theorem reduces the moments to edge integrals ( `erf`s ) and
    # to the exact mass -- they no longer go through the adaptive quadrature, which missed a large boundary cell by 12 %
    # depending on how it was cut into triangles ( the BSP and the plain search do not cut it the same way )
    rng = numpy.random.default_rng( 0 )
    k = rng.choice( 4, 200, p = [ 0.35, 0.25, 0.25, 0.15 ] )
    C = numpy.array( [ [ -1.0, -0.6 ], [ 1.1, -0.4 ], [ 0.2, 1.0 ], [ -0.9, 0.9 ] ] )
    x = C[ k ] + numpy.array( [ 0.35, 0.25, 0.3, 0.2 ] )[ k, None ] * rng.standard_normal( ( 200, 2 ) )
    c = numpy.array( [ [ -0.97, -0.70 ], [ 1.13, -0.42 ], [ 0.19, 1.02 ], [ -0.96, 0.90 ] ] )
    s, m = numpy.array( [ 0.35, 0.26, 0.28, 0.17 ] ), numpy.array( [ 0.3, 0.25, 0.24, 0.21 ] )
    w = numpy.asarray( sdot.OtProblem( sdot.SumOfDiracs( x ), sdot.SumOfGaussians( c, s, weights = m ) ).solve().weights )

    rho = sdot.SumOfGaussians( c, s, weights = m )
    a = m / m.sum()
    ref = [ ( a * c[ :, 0 ] ).sum(), ( a * c[ :, 1 ] ).sum(), ( a * ( ( c ** 2 ).sum( 1 ) + 2 * s ** 2 ) ).sum(), 1.0 ]
    moments = { acc: [ numpy.asarray( sdot.PowerDiagram( x, w, distribution = rho, accelerator = acc ).moment( k ) ) for k in range( 4 ) ]
                for acc in ( None, "plain" ) }
    measures = numpy.asarray( sdot.PowerDiagram( x, w, distribution = rho ).measures )
    for k in range( 4 ):
        assert abs( moments[ None ][ k ].sum() - ref[ k ] ) < 1e-6, ( k, moments[ None ][ k ].sum(), ref[ k ] )     # the tails beyond 6 sigmas
        assert numpy.abs( moments[ None ][ k ] - moments[ "plain" ][ k ] ).max() < 1e-6, k
    assert numpy.abs( moments[ None ][ 3 ] - measures ).max() < 1e-6
