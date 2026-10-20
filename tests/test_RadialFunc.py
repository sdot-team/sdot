from errand import test


# A RADIAL INTEGRAND ( `PowerDiagram( radial_func = RadialExp( eps ) )` ): the cell of seed `i` integrates
# `rho( x ) exp( ( w_i - |x - p_i|^2 ) / ( 2 eps ) )` ( `integration/RadialExp.h` ). THE REFERENCE is the same diagram without it,
# each cell integrated in numpy by a Gauss rule on the triangles of a fan ( Duffy ): no code in common but the cutting of the cells.


def _gauss_on_polygon( vertices, f, k = 48 ):
    """`int f` and `int x f` over the convex polygon `vertices`: a fan from its first vertex, `k x k` Gauss points per triangle"""
    import numpy
    v = numpy.asarray( vertices, dtype = float ).reshape( -1, 2 )
    g, gw = numpy.polynomial.legendre.leggauss( k )
    g, gw = ( g + 1 ) / 2, gw / 2
    u, s = numpy.meshgrid( g, g, indexing = "ij" )
    wu = numpy.outer( gw, gw ) * u                       # Duffy: `( u, s ) -> ( u ( 1 - s ), u s )`, jacobian `u`
    m, first = 0.0, numpy.zeros( 2 )
    for i in range( 1, len( v ) - 1 ):
        a, b, c = v[ 0 ], v[ i ], v[ i + 1 ]
        det = abs( ( b[ 0 ] - a[ 0 ] ) * ( c[ 1 ] - a[ 1 ] ) - ( b[ 1 ] - a[ 1 ] ) * ( c[ 0 ] - a[ 0 ] ) )
        x = a + ( u * ( 1 - s ) )[ ..., None ] * ( b - a ) + ( u * s )[ ..., None ] * ( c - a )
        val = f( x ) * wu * det
        m += val.sum()
        first += ( val[ ..., None ] * x ).sum( ( 0, 1 ) )
    return m, first


def _setup( n = 25, eps = 0.01, seed = 0 ):
    import numpy
    rng = numpy.random.default_rng( seed )
    pos = rng.uniform( 0.1, 0.9, size = ( n, 2 ) )
    w = rng.normal( 0, 0.001, size = n )                 # ( small: no cell emptied )
    return pos, w, eps


def _diagram( pos, w, eps, **kwargs ):
    from sdot.distributions.Box import Box
    from sdot.PowerDiagram import PowerDiagram
    from sdot.radial_funcs import RadialExp
    return PowerDiagram( pos, w, distribution = Box( nb_dims = 2 ), kernel_dtype = "FP64", radial_func = RadialExp( eps ), **kwargs )


if test( "measures_and_moments_match_quadrature" ):
    import numpy
    from sdot.distributions.Box import Box
    from sdot.PowerDiagram import PowerDiagram
    pos, w, eps = _setup()
    n = len( pos )
    pd = _diagram( pos, w, eps )
    m = numpy.asarray( pd.measures ).reshape( -1 )
    mm, f, _ = pd.moments
    mm, f = numpy.asarray( mm.value ).reshape( -1 ), numpy.asarray( f.value ).reshape( n, 2 )
    cells = PowerDiagram( pos, w, distribution = Box( nb_dims = 2 ), kernel_dtype = "FP64" ).cells
    for i in range( n ):
        ri, fi = _gauss_on_polygon( cells.vertices( i ), lambda x: numpy.exp( ( w[ i ] - ( ( x - pos[ i ] ) ** 2 ).sum( -1 ) ) / ( 2 * eps ) ) )
        assert abs( m[ i ] - ri ) < 1e-9 * max( ri, 1e-3 ), ( i, m[ i ], ri )
        assert abs( mm[ i ] - ri ) < 1e-9 * max( ri, 1e-3 ), ( i, mm[ i ], ri )
        assert numpy.abs( f[ i ] - fi ).max() < 1e-9 * max( ri, 1e-3 ), ( i, f[ i ], fi )


if test( "a_density_scales_the_integrand" ):
    import numpy
    from sdot.distributions.Box import Box
    from sdot.PowerDiagram import PowerDiagram
    from sdot.radial_funcs import RadialExp
    pos, w, eps = _setup( n = 12 )
    # a box of side 2: the density, normalized, is 1/4 -- the same cells as in the unit box, around the seeds
    big = PowerDiagram( pos, w, distribution = Box( origin = numpy.array( [ 0.0, 0.0 ] ), frame = numpy.array( [ 2.0, 2.0 ] ) ),
                        kernel_dtype = "FP64", radial_func = RadialExp( eps ) )
    cells = PowerDiagram( pos, w, distribution = Box( origin = numpy.array( [ 0.0, 0.0 ] ), frame = numpy.array( [ 2.0, 2.0 ] ) ), kernel_dtype = "FP64" ).cells
    m = numpy.asarray( big.measures ).reshape( -1 )
    for i in range( len( pos ) ):
        ri, _ = _gauss_on_polygon( cells.vertices( i ), lambda x: numpy.exp( ( w[ i ] - ( ( x - pos[ i ] ) ** 2 ).sum( -1 ) ) / ( 2 * eps ) ) / 4 )
        assert abs( m[ i ] - ri ) < 1e-9 * max( ri, 1e-3 ), ( i, m[ i ], ri )


if test( "hessian_rows_match_finite_differences" ):
    # `d m_i / d w_j = - vals` on the facets, and the last slot ( the cell's own id ) is `m_i / ( 2 eps )`
    import numpy
    pos, w, eps = _setup( n = 20 )
    n = len( pos )
    pd = _diagram( pos, w, eps )
    cnt, ids, vals = pd.hessian_rows()
    J = numpy.zeros( ( n, n ) )
    m = numpy.asarray( pd.measures ).reshape( -1 )
    for i in range( n ):
        own = [ r for r in range( cnt[ i ] ) if ids[ i, r ] == i ]
        assert len( own ) == 1 and abs( vals[ i, own[ 0 ] ] - m[ i ] / ( 2 * eps ) ) < 1e-12 * m[ i ] / eps
        for r in range( cnt[ i ] ):
            j = ids[ i, r ]
            if j == i:
                J[ i, i ] += vals[ i, r ]
            elif j >= 0:
                J[ i, j ] -= vals[ i, r ]
                J[ i, i ] += vals[ i, r ]
    h = 1e-6
    for j in range( n ):
        wp, wm = w.copy(), w.copy()
        wp[ j ] += h
        wm[ j ] -= h
        fd = ( numpy.asarray( _diagram( pos, wp, eps ).measures ) - numpy.asarray( _diagram( pos, wm, eps ).measures ) ).reshape( -1 ) / ( 2 * h )
        assert numpy.abs( J[ :, j ] - fd ).max() < 1e-6 * numpy.abs( J ).max(), ( j, numpy.abs( J[ :, j ] - fd ).max() )


if test( "newton_reaches_the_masses" ):
    # the weights of the Moreau-Yosida regularization of the entropy: every cell weighs `1 / n`, with no gauge to fix
    import numpy
    import scipy.sparse
    import scipy.sparse.linalg
    pos, _, eps = _setup( n = 40, eps = 0.003 )
    n = len( pos )
    w = numpy.zeros( n )
    w -= 2 * eps * numpy.log( numpy.asarray( _diagram( pos, w, eps ).measures ).sum() )     # sigma weighs 1
    for it in range( 30 ):
        pd = _diagram( pos, w, eps )
        m = numpy.asarray( pd.measures ).reshape( -1 )
        if numpy.abs( m * n - 1 ).max() < 1e-10:
            break
        cnt, ids, vals = pd.hessian_rows()
        J = numpy.zeros( ( n, n ) )
        for i in range( n ):
            for r in range( cnt[ i ] ):
                j = ids[ i, r ]
                if j >= 0:
                    J[ i, i ] += vals[ i, r ]
                    if j != i:
                        J[ i, j ] -= vals[ i, r ]
        d = numpy.linalg.solve( J, 1 / n - m )
        t = 1.0
        while not ( numpy.asarray( _diagram( pos, w + t * d, eps ).measures ) > 0 ).all():
            t /= 2
        w = w + t * d
    assert numpy.abs( m * n - 1 ).max() < 1e-10, numpy.abs( m * n - 1 ).max()
    assert it < 15, it


if test( "measures_and_moments_derive" ):
    # the integrand moves with its seed: the gaussian's center with the position, its height with the weight
    # ( `RadialExpDensity::seed_bwd` ), the cells' vertices with both; and it is linear in the values of the density
    import numpy
    from loom.testing import need, check_grad
    from sdot.distributions.Box import Box
    from sdot.distributions.Image import Image
    from sdot.PowerDiagram import PowerDiagram
    from sdot.radial_funcs import RadialExp
    from sdot.Cell import set_kernel_dtype
    set_kernel_dtype( "FP64" )
    need( "grad" )
    pos, w, eps = _setup( n = 10, eps = 0.01 )
    rng = numpy.random.default_rng( 3 )
    img = rng.uniform( 0.5, 2, size = ( 7, 5 ) )
    for dist in ( Box( nb_dims = 2 ), Image( img, frame = [ 1.0, 1.0 ] ) ):
        mk = lambda p, q: PowerDiagram( p, weights = q, distribution = dist, radial_func = RadialExp( eps ), accelerator = "plain" )
        check_grad( lambda p, q: mk( p, q ).measures, pos, w, seed = 1, eps = 1e-6, rtol = 1e-5, atol = 1e-7 )
        for k in range( 4 ):
            check_grad( lambda p, q, k = k: mk( p, q ).moment( k ), pos, w, seed = 2 + k, eps = 1e-6, rtol = 1e-5, atol = 1e-7 )
    mk = lambda v: PowerDiagram( pos, weights = w, distribution = Image( v, frame = [ 1.0, 1.0 ] ), radial_func = RadialExp( eps ) )
    check_grad( lambda v: mk( v ).measures, img, seed = 7, eps = 1e-6, rtol = 1e-5, atol = 1e-7 )
    check_grad( lambda v: mk( v ).moment( 0 ), img, seed = 8, eps = 1e-6, rtol = 1e-5, atol = 1e-7 )


if test( "a_plan_reaches_the_masses" ):
    # `ot_solve( ..., radial_func = ... )`: the solver's Newton, the radial factor being a ground ( `sdotplan/Sweep.h` ): every cell
    # weighs its dirac's mass, not rescaled; barycenters and cost are those of the integrand
    import numpy
    from loom.testing import need
    from sdot.OtProblem import ot_solve
    from sdot.distributions.Box import Box
    from sdot.distributions.Image import Image
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from sdot.radial_funcs import RadialExp
    need( "cpu" )
    rng = numpy.random.default_rng( 1 )
    n, eps = 60, 0.004
    pos = rng.uniform( 0.2, 0.8, size = ( n, 2 ) )
    nu = rng.uniform( 0.5, 1.5, size = n )
    nu /= nu.sum()
    for dist in ( Box( nb_dims = 2 ), Image( rng.uniform( 0.2, 2, size = ( 9, 7 ) ), frame = [ 1.0, 1.0 ] ) ):
        p = ot_solve( SumOfDiracs( pos, nu ), dist, radial_func = RadialExp( eps ), tol = 1e-13 )
        assert p.converged, p.stats[ "status" ]
        assert numpy.abs( numpy.asarray( p.cell_masses ).reshape( -1 ) - nu ).max() < 1e-12
        assert numpy.allclose( numpy.asarray( p.target_masses ).reshape( -1 ), nu )
        # ... those of the diagram of the plan, with the same radial function
        pd = p.power_diagram()
        assert numpy.abs( numpy.asarray( pd.measures ).reshape( -1 ) - nu ).max() < 1e-12
        mm, f, q = pd.moments
        mm, f, q = ( numpy.asarray( x.value ).reshape( n, -1 ) for x in ( mm, f, q ) )
        cost = ( q[ :, 0 ] - 2 * ( pos * f ).sum( 1 ) + ( pos ** 2 ).sum( 1 ) * mm[ :, 0 ] ).sum()
        assert abs( cost - float( numpy.asarray( p.cost ) ) ) < 1e-12
        assert numpy.allclose( numpy.asarray( p.barycenters ).reshape( n, 2 ), f / mm, atol = 1e-10 )
        # warm: from the plan of positions nearby
        p2 = ot_solve( SumOfDiracs( pos + 0.002, nu ), dist, radial_func = RadialExp( eps ), tol = 1e-13, ot_plan = p )
        assert p2.converged and numpy.abs( numpy.asarray( p2.cell_masses ).reshape( -1 ) - nu ).max() < 1e-12


if test( "a_plan_is_differentiable" ):
    # the implicit function theorem through the ground ( `_attach_derivatives`, `sdotplan/Adjoint.h` ): against the positions,
    # the masses of the diracs and the values of the image
    import numpy
    from loom.testing import check_grad, need
    from sdot.OtProblem import Iterative, OtProblem
    from sdot.distributions.Image import Image
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from sdot.radial_funcs import RadialExp
    need( "cpu" )
    need( "grad" )
    rng = numpy.random.default_rng( 3 )
    n = 10
    pos = rng.uniform( 0.2, 0.8, size = ( n, 2 ) )
    nu = rng.uniform( 0.5, 1.5, size = n ) / n
    values = rng.uniform( 0.5, 1.5, size = ( 3, 3 ) )
    settings = Iterative( tol = 1e-13, max_iter = 200 )

    def solve( p, q, v ):
        return OtProblem( SumOfDiracs( p, weights = q, target_mass = None ), Image( values = v, frame = [ 1.0, 1.0 ] ),
                          radial_func = RadialExp( 0.01 ) ).solve( settings )

    for output, tol in ( ( "weights", 1e-5 ), ( "cell_masses", 1e-5 ), ( "cost", 1e-4 ), ( "barycenters", 1e-4 ) ):
        check_grad( lambda p, q, v: getattr( solve( p, q, v ), output ), pos, nu, values, seed = 3, eps = 1e-6, rtol = tol, atol = tol * 1e-2 )


if test( "unsupported_cases_are_refused" ):
    import numpy
    from sdot.distributions.SumOfGaussians import SumOfGaussians
    from sdot.PowerDiagram import PowerDiagram
    from sdot.radial_funcs import RadialExp
    for args, kwargs in [ ( ( numpy.random.default_rng( 0 ).uniform( size = ( 5, 3 ) ), ), {} ),
                          ( ( numpy.random.default_rng( 0 ).uniform( size = ( 5, 2 ) ), ),
                            dict( distribution = SumOfGaussians( numpy.zeros( ( 1, 2 ) ), numpy.ones( 1 ) ) ) ),
                          ( ( numpy.random.default_rng( 0 ).uniform( size = ( 5, 2 ) ), ), dict( sink_weight = 0.0 ) ) ]:
        try:
            PowerDiagram( *args, radial_func = RadialExp( 0.01 ), **kwargs )
        except NotImplementedError:
            continue
        raise AssertionError( f"accepted: { kwargs }" )
