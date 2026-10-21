from errand import test


# THE REFERENCE of a periodic diagram is the same diagram WITHOUT periodicity, on the seeds copied by the short words of the
# group ( a translation lattice: its ring `{ -1, 0, 1 }^d` ), integrated on the support of the density only: the cell of seed
# `i` on the quotient is the union of the pieces of its copies, each brought back next to `i` by the inverse map
# ( `M^T ( first - m V )` ). No code in common with the periodic path but the cutting itself: neither the images, nor the
# start box, nor the moved densities.


def _maps( gens, d, length = 3 ):
    """the elements of the group of words of at most `length` generators ( affine maps `[ d, d + 1 ]` ), the identity first"""
    import numpy
    def affine( g ):
        if isinstance( g, tuple ):
            return numpy.concatenate( [ numpy.asarray( g[ 0 ], dtype = float ), numpy.asarray( g[ 1 ], dtype = float ).reshape( d, 1 ) ], axis = 1 )
        return numpy.concatenate( [ numpy.eye( d ), numpy.asarray( g, dtype = float ).reshape( d, 1 ) ], axis = 1 )
    def compose( a, b ):
        return numpy.concatenate( [ a[ :, :d ] @ b[ :, :d ], ( a[ :, :d ] @ b[ :, d ] + a[ :, d ] )[ :, None ] ], axis = 1 )
    def inverse( a ):
        return numpy.concatenate( [ a[ :, :d ].T, ( - a[ :, :d ].T @ a[ :, d ] )[ :, None ] ], axis = 1 )
    moves = [ affine( g ) for g in gens ]
    moves += [ inverse( m ) for m in moves ]
    res = [ numpy.concatenate( [ numpy.eye( d ), numpy.zeros( ( d, 1 ) ) ], axis = 1 ) ]
    keys = { tuple( numpy.round( res[ 0 ].ravel(), 9 ) ) }
    front = list( res )
    for _ in range( length ):
        nxt = []
        for a in front:
            for m in moves:
                b = compose( m, a )
                k = tuple( numpy.round( b.ravel(), 9 ) )
                if k not in keys:
                    keys.add( k )
                    nxt.append( b )
        res += nxt
        front = nxt
    return res


def _reference( folded, weights, dist, gens, length = 3, **kw ):
    """`( masses, first )` of the periodic diagram, from the copies of the seeds in a plain diagram"""
    import numpy
    from sdot.PowerDiagram import PowerDiagram
    n, d = folded.shape
    maps = _maps( gens, d, length )
    pos = numpy.concatenate( [ folded @ a[ :, :d ].T + a[ :, d ] for a in maps ] )
    w = numpy.concatenate( [ weights for a in maps ] )
    pd = PowerDiagram( pos, w, distribution = dist, kernel_dtype = "FP64", **kw )
    m, f, _ = pd.moments
    m = numpy.asarray( m.value ).reshape( len( maps ), n )
    f = numpy.asarray( f.value ).reshape( len( maps ), n, d )
    first = sum( ( f[ t ] - m[ t ][ :, None ] * maps[ t ][ :, d ][ None, : ] ) @ maps[ t ][ :, :d ] for t in range( len( maps ) ) )
    return m.sum( axis = 0 ), first


def _compare( pos, weights, dist, periodicity, gens, accelerator = None, tol = 1e-10, length = 3 ):
    import numpy
    from sdot.PowerDiagram import PowerDiagram
    pd = PowerDiagram( pos, weights, distribution = dist, periodicity = periodicity, kernel_dtype = "FP64", accelerator = accelerator )
    folded = numpy.asarray( pd._stored_positions().raw ).reshape( pos.shape )
    assert numpy.allclose( numpy.asarray( pd.unfold( folded ) ), pos, atol = 1e-12 )
    assert numpy.allclose( numpy.asarray( pd.positions ).reshape( pos.shape ), pos, atol = 1e-12 )    # ( as given )
    m, f, _ = pd.moments
    m = numpy.asarray( m.value ).reshape( -1 )
    f = numpy.asarray( f.value ).reshape( len( m ), -1 )
    meas = numpy.asarray( pd.measures.value ).reshape( -1 )
    rm, rf = _reference( folded, weights, dist, gens, length )
    assert numpy.allclose( m, rm, atol = tol ), ( m, rm )
    assert numpy.allclose( meas, rm, atol = tol ), ( meas, rm )
    # ( the moments are those of the cells around the GIVEN positions )
    rb = numpy.asarray( pd.unfold( rf / numpy.maximum( rm, 1e-300 )[ :, None ] ) )
    assert numpy.allclose( f, rb * rm[ :, None ], atol = tol ), numpy.abs( f - rb * rm[ :, None ] ).max()
    assert abs( m.sum() - 1 ) < tol, m.sum()
    return pd


if test( "periodic_measures_match_the_copies_2d" ):
    import numpy
    from loom.testing import need
    from sdot.distributions.Image import Image
    rng = numpy.random.default_rng( 0 )
    img = Image( values = rng.uniform( 0.5, 1.5, size = ( 4, 3 ) ), frame = [ 1.0, 1.0 ] )
    gens = numpy.eye( 2 )
    pos = rng.uniform( -0.5, 1.7, size = ( 15, 2 ) )        # some outside: folded
    w = rng.uniform( -0.01, 0.01, size = 15 )
    for acc in ( None, "plain" ):
        _compare( pos, w, img, True, gens, accelerator = acc )
    # a large weight: a cell reaching beyond half the support
    w2 = w.copy()
    w2[ 0 ] = 0.15
    _compare( pos, w2, img, True, gens )


if test( "periodic_measures_cylinder_and_few_seeds" ):
    import numpy
    from loom.testing import need
    from sdot.distributions.Image import Image
    rng = numpy.random.default_rng( 1 )
    img = Image( values = rng.uniform( 0.5, 1.5, size = ( 3, 5 ) ), frame = [ 2.0, 1.0 ] )
    # a cylinder: periodic along x only ( period 2 ), walls in y
    pos = rng.uniform( 0, 1, size = ( 9, 2 ) ) * [ 4.0, 1.0 ] - [ 1.0, 0.0 ]
    _compare( pos, numpy.zeros( 9 ), img, [ True, False ], numpy.array( [ [ 2.0, 0.0 ] ] ) )
    # two seeds on a torus: each cell neighbours the other through two images, and itself through its own
    pos = numpy.array( [ [ 0.3, 0.2 ], [ 1.2, 0.8 ] ] )
    _compare( pos, numpy.array( [ 0.0, 0.05 ] ), img, True, numpy.diag( [ 2.0, 1.0 ] ) )


if test( "periodic_measures_polynomial_image" ):
    # a polynomial on each pixel ( `Image( degree = 2 )`, then a cubic spline with its half pixels at the ends ): the pieces of the
    # copies are the same polynomials, moved
    import numpy
    from sdot.distributions.Image import Image
    from sdot.distributions.SplineGrid import SplineGrid
    rng = numpy.random.default_rng( 7 )
    pos = rng.uniform( -0.5, 1.7, size = ( 15, 2 ) )
    w = rng.uniform( -0.01, 0.01, size = 15 )
    for img in ( Image( rng.uniform( 0.3, 1.5, size = ( 4, 3, 9 ) ), degree = 2, frame = [ 1.0, 1.0 ] ),
                 SplineGrid( rng.uniform( 0.3, 1.5, size = ( 4, 3 ) ), degree = 3, frame = [ 1.0, 1.0 ] ) ):
        _compare( pos, w, img, True, numpy.eye( 2 ) )


if test( "periodic_measures_skewed_lattice_3d" ):
    import numpy
    from loom.testing import need
    from sdot.distributions.Box import Box
    rng = numpy.random.default_rng( 2 )
    # a box support of volume 1 and a lattice of the same covolume, sheared: the box is a fundamental domain
    gens = numpy.array( [ [ 1.0, 0.0, 0.0 ], [ 0.5, 1.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ] )
    pos = rng.uniform( 0, 1, size = ( 12, 3 ) )
    w = rng.uniform( -0.01, 0.01, size = 12 )
    _compare( pos, w, Box( nb_dims = 3 ), [ g for g in gens ], gens )


if test( "periodic_measures_derive_wrt_positions_and_weights" ):
    import numpy
    from loom.testing import need, check_grad
    from sdot.distributions.Image import Image
    from sdot.PowerDiagram import PowerDiagram
    from sdot.Cell import set_kernel_dtype
    set_kernel_dtype( "FP64" )
    need( "grad" )
    # the images carry the derivative of their seed: a cut by `j + n t` sends its share to `j`. Some seeds outside the
    # support, folded: the folding is constant by pieces, it does not derive.
    rng = numpy.random.default_rng( 3 )
    img = Image( values = rng.uniform( 0.5, 1.5, size = ( 4, 3 ) ), frame = [ 1.0, 1.0 ] )
    pos = rng.uniform( -0.4, 1.4, size = ( 8, 2 ) )
    w = rng.uniform( -0.02, 0.02, size = 8 )
    check_grad( lambda p, q: PowerDiagram( p, weights = q, distribution = img, periodicity = True ).measures, pos, w )
    # few seeds: cells that touch their own images
    pos = rng.uniform( 0, 1, size = ( 3, 2 ) )
    check_grad( lambda p, q: PowerDiagram( p, weights = q, distribution = img, periodicity = True ).measures, pos, numpy.array( [ 0.0, 0.03, -0.02 ] ) )


if test( "periodic_hessian_rows_match_finite_differences" ):
    import numpy
    from loom.testing import need
    from sdot.distributions.Image import Image
    from sdot.PowerDiagram import PowerDiagram
    need( "cpu" )
    # `d m_i / d w_j = - vals`, summed over the images of `j`; the facets of a cell with its own images do not move
    rng = numpy.random.default_rng( 4 )
    img = Image( values = rng.uniform( 0.5, 1.5, size = ( 4, 3 ) ), frame = [ 1.0, 1.0 ] )
    for n in ( 2, 3, 10 ):
        pos = rng.uniform( 0, 1, size = ( n, 2 ) )
        w = rng.uniform( -0.01, 0.01, size = n )

        def meas( w ):
            return numpy.asarray( PowerDiagram( pos, w, distribution = img, periodicity = True, kernel_dtype = "FP64" ).measures.value ).reshape( -1 )

        counts, ids, vals = PowerDiagram( pos, w, distribution = img, periodicity = True, kernel_dtype = "FP64" ).hessian_rows()
        J = numpy.zeros( ( n, n ) )
        for i in range( n ):
            for r in range( counts[ i ] ):
                if ids[ i, r ] >= 0:
                    J[ i, ids[ i, r ] ] -= vals[ i, r ]
                    J[ i, i ] += vals[ i, r ]
        eps = 1e-6
        Jfd = numpy.stack( [ ( meas( w + eps * e ) - meas( w - eps * e ) ) / ( 2 * eps ) for e in numpy.eye( n ) ], axis = 1 )
        assert numpy.allclose( J, Jfd, atol = 1e-6 ), ( n, numpy.abs( J - Jfd ).max() )


if test( "periodic_plan_of_a_regular_grid" ):
    import numpy
    from loom.testing import need
    from sdot import plan
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from sdot.distributions.Box import Box
    # a regular grid on the torus, shifted off the sides of the box and partly by whole periods: every cell is the square
    # around its dirac, its barycenter the dirac itself ( next to it as given ), and the cost `n h^4 / 6` with `h = 1 / 4`
    g = ( numpy.arange( 4 ) + 0.5 ) / 4
    pos = numpy.stack( numpy.meshgrid( g, g, indexing = "ij" ), axis = -1 ).reshape( -1, 2 ) + [ 0.07, -0.11 ]
    pos[ ::3 ] += [ 2.0, -1.0 ]
    p = plan( SumOfDiracs( pos ), Box(), periodicity = True )
    assert p.converged
    assert numpy.allclose( numpy.asarray( p.cell_masses ).reshape( -1 ), 1 / 16, atol = 1e-9 )
    assert numpy.allclose( numpy.asarray( p.barycenters ).reshape( -1, 2 ), pos, atol = 1e-9 )
    assert abs( float( p.cost ) - 16 * 0.25 ** 4 / 6 ) < 1e-9, float( p.cost )
    _, grad = p.cost_and_position_grad()
    assert numpy.abs( numpy.asarray( grad ) ).max() < 1e-8


if test( "periodic_plan_matches_the_copies" ):
    import numpy
    from loom.testing import need
    from sdot.OtProblem import Iterative, OtProblem
    from sdot.distributions.Image import Image
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from sdot.PowerDiagram import PowerDiagram
    need( "cpu" )
    rng = numpy.random.default_rng( 5 )
    img = Image( values = rng.uniform( 0.5, 1.5, size = ( 5, 4 ) ), frame = [ 1.0, 1.0 ] )
    for d_n in ( 3, 40 ):
        pos = rng.uniform( -1, 2, size = ( d_n, 2 ) )
        mass = rng.uniform( 0.5, 1.5, size = d_n )
        pb = OtProblem( SumOfDiracs( pos, mass ), img, periodicity = True )
        p = pb.solve( Iterative( tol = 1e-11 ) )
        assert p.converged
        nu = mass / mass.sum()
        w = numpy.asarray( p.weights ).reshape( -1 )
        cm = numpy.asarray( p.cell_masses ).reshape( -1 )
        assert numpy.allclose( cm, nu, atol = 1e-9 ), numpy.abs( cm - nu ).max()
        # the reference at the fitted weights: the copies of the folded diracs
        pd = PowerDiagram( pos, w, distribution = img, periodicity = True, kernel_dtype = "FP64" )
        folded = numpy.asarray( pd._stored_positions().raw ).reshape( pos.shape )
        rm, rf = _reference( folded, w, img.normalized_version( nb_dims = 2 ), numpy.eye( 2 ) )
        assert numpy.allclose( rm, nu, atol = 1e-8 ), numpy.abs( rm - nu ).max()
        bar = numpy.asarray( p.barycenters ).reshape( -1, 2 )
        ref = numpy.asarray( pd.unfold( rf / rm[ :, None ] ) )
        assert numpy.allclose( bar, ref, atol = 1e-8 ), numpy.abs( bar - ref ).max()
        # the cost, from the moments ( around the given positions ): `int |x - p|^2 = second - 2 p . first + |p|^2 m`
        mm, ff, sec = pd.moments
        ff = numpy.asarray( ff.value ).reshape( -1, 2 )
        cost = ( numpy.asarray( sec.value ).reshape( -1 ) - 2 * ( pos * ff ).sum( axis = 1 ) + ( pos ** 2 ).sum( axis = 1 ) * rm ).sum()
        assert abs( float( p.cost ) - cost ) < 1e-8, ( float( p.cost ), cost )


if test( "periodic_plan_3d_partly_periodic" ):
    import numpy
    from loom.testing import need
    from sdot.OtProblem import Iterative, OtProblem
    from sdot.distributions.Box import Box
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from sdot.PowerDiagram import PowerDiagram
    need( "cpu" )
    rng = numpy.random.default_rng( 6 )
    pos = rng.uniform( -0.5, 1.5, size = ( 30, 3 ) )
    pos[ :, 2 ] = rng.uniform( 0, 1, size = 30 )          # the free axis: inside
    box = Box( nb_dims = 3 )
    p = OtProblem( SumOfDiracs( pos ), box, periodicity = [ True, True, False ] ).solve( Iterative( tol = 1e-11 ) )
    assert p.converged
    assert numpy.allclose( numpy.asarray( p.cell_masses ).reshape( -1 ), 1 / 30, atol = 1e-9 )
    w = numpy.asarray( p.weights ).reshape( -1 )
    pd = PowerDiagram( pos, w, distribution = box, periodicity = [ True, True, False ], kernel_dtype = "FP64" )
    folded = numpy.asarray( pd._stored_positions().raw ).reshape( pos.shape )
    rm, rf = _reference( folded, w, box.normalized_version( nb_dims = 3 ), numpy.eye( 3 )[ :2 ] )
    assert numpy.allclose( rm, 1 / 30, atol = 1e-8 )
    bar = numpy.asarray( p.barycenters ).reshape( -1, 3 )
    assert numpy.allclose( bar, numpy.asarray( pd.unfold( rf / rm[ :, None ] ) ), atol = 1e-8 )


if test( "periodic_plan_is_differentiable" ):
    import numpy
    from loom.testing import check_grad, need
    from sdot.OtProblem import Iterative, OtProblem
    from sdot.distributions.Image import Image
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    need( "cpu" )
    # through the implicit function theorem ( `_attach_derivatives` ), on the periodic diagrams: against the positions ( some
    # folded ), the masses of the diracs and the values of the image
    rng = numpy.random.default_rng( 7 )
    n = 12
    pos = rng.uniform( -0.5, 1.5, size = ( n, 2 ) )
    nu = rng.uniform( 0.5, 1.5, size = n )
    values = rng.uniform( 0.5, 1.5, size = ( 3, 3 ) )
    settings = Iterative( tol = 1e-13, max_iter = 200 )

    def solve( p, q, v ):
        return OtProblem( SumOfDiracs( p, weights = q ), Image( values = v, frame = [ 1.0, 1.0 ] ), periodicity = True ).solve( settings )

    for output, tol in ( ( "weights", 1e-5 ), ( "cell_masses", 1e-5 ), ( "cost", 1e-3 ), ( "barycenters", 1e-3 ) ):
        check_grad( lambda p, q, v: getattr( solve( p, q, v ), output ), pos, nu, values, seed = 3, eps = 1e-5, rtol = tol, atol = tol * 1e-2 )


if test( "periodicity_descriptions" ):
    import numpy
    from sdot.periodicity import generators_of, wrap
    from sdot.distributions.Image import Image
    from sdot.distributions.Polytope import ConvexHull
    img = Image( values = numpy.ones( ( 2, 2 ) ), frame = [ 2.0, 1.0 ], origin = [ 1.0, 0.0 ] )
    sup = img.bounding_half_spaces()

    def translations( g ):
        assert numpy.allclose( g[ :, :, :2 ], numpy.eye( 2 ) )
        return g[ :, :, 2 ]

    assert numpy.allclose( translations( generators_of( True, 2, sup ) ), [ [ 2, 0 ], [ 0, 1 ] ] )
    assert numpy.allclose( translations( generators_of( [ False, True ], 2, sup ) ), [ [ 0, 1 ] ] )
    assert generators_of( [ False, False ], 2, sup ) is None
    assert numpy.allclose( sorted( map( tuple, numpy.abs( translations( generators_of( img.domain, 2 ) ) ) ) ), [ ( 0, 1 ), ( 2, 0 ) ] )
    # the three forms of an isometry
    R = numpy.array( [ [ 0.0, -1.0 ], [ 1.0, 0.0 ] ] )
    T = numpy.eye( 3 ); T[ :2, :2 ] = R; T[ :2, 2 ] = [ 0.5, 1.0 ]
    for g in ( ( R, [ 0.5, 1.0 ] ), T ):
        assert numpy.allclose( generators_of( [ g ], 2 ), [ numpy.concatenate( [ R, [ [ 0.5 ], [ 1.0 ] ] ], axis = 1 ) ] )
    assert numpy.allclose( translations( generators_of( [ [ 0.5, 1.0 ] ], 2 ) ), [ [ 0.5, 1.0 ] ] )
    # the booleans read an AXIS-ALIGNED box: a hexagon refuses them, and pairs its opposite sides through its domain
    hexa = ConvexHull( [ [ numpy.cos( a ), numpy.sin( a ) ] for a in numpy.arange( 6 ) * numpy.pi / 3 ] )
    for bad in ( True, [ True, False ] ):
        try:
            generators_of( bad, 2, hexa.domain )
        except ValueError as e:
            assert "AXIS-ALIGNED" in str( e )
        else:
            raise AssertionError( bad )
    for bad in ( [ [ 1, 0 ], [ numpy.sqrt( 2 ), 0 ] ], [ [ 0, 0 ] ], [ ( 2 * numpy.eye( 2 ), [ 1, 0 ] ) ] ):
        try:
            generators_of( bad, 2 )
        except ValueError:
            pass
        else:
            raise AssertionError( bad )
    # translations that are not independent: the basis of their lattice
    assert numpy.allclose( translations( generators_of( [ [ 1, 0 ], [ 2, 0 ] ], 2 ) ), [ [ 1, 0 ] ] )
    assert numpy.allclose( numpy.abs( translations( generators_of( [ [ 1, 0 ], [ 1.1, 0 ] ], 2 ) ) ), [ [ 0.1, 0 ] ] )
    assert numpy.allclose( wrap( [ [ 3.5, -0.25 ] ], True, img ), [ [ 1.5, 0.75 ] ] )
    # a Klein bottle: a point below the square comes back flipped
    klein = [ ( numpy.diag( [ 1.0, -1.0 ] ), [ 1.0, 1.0 ] ), [ 0.0, 1.0 ] ]
    unit = Image( values = numpy.ones( ( 1, 1 ) ), frame = [ 1.0, 1.0 ] )
    assert numpy.allclose( wrap( [ [ 1.25, 0.25 ] ], klein, unit ), [ [ 0.25, 0.75 ] ] )


if test( "domain_gluings_match_the_copies" ):
    # `periodicity = target.domain`: each facet is sent ONTO its opposite one -- along the side of a parallelogram, not across it --
    # and the three pairs of sides of a hexagon give a lattice of rank 2
    import numpy
    from sdot.periodicity import generators_of
    from sdot.distributions.Polytope import ConvexHull
    from sdot.distributions.Image import Image
    rng = numpy.random.default_rng( 9 )
    t = numpy.arange( 6 ) * numpy.pi / 3
    cases = (
        ( ConvexHull( numpy.stack( [ numpy.cos( t ), numpy.sin( t ) ], -1 ) ), [ [ 1.5, numpy.sqrt( 3 ) / 2 ], [ 0.0, numpy.sqrt( 3 ) ] ] ),
        ( ConvexHull( [ [ 0, 0 ], [ 1, 0 ], [ 1.5, 1 ], [ 0.5, 1 ] ] ), [ [ 1.0, 0.0 ], [ 0.5, 1.0 ] ] ),
        # ( an image with an oblique frame: the block of its pixels is a parallelogram )
        ( Image( values = rng.uniform( 0.5, 1.5, size = ( 4, 3 ) ), frame = [ [ 1.0, 0.0 ], [ 0.5, 1.0 ] ] ), [ [ 1.0, 0.0 ], [ 0.5, 1.0 ] ] ),
    )
    for hull, basis in cases:
        g = generators_of( hull.domain, 2 )
        assert len( g ) == 2 and numpy.allclose( g[ :, :, :2 ], numpy.eye( 2 ) )
        # ( the same lattice: each basis has integer coordinates in the other, of determinant +-1 )
        c = numpy.asarray( basis ) @ numpy.linalg.inv( g[ :, :, 2 ] )
        assert numpy.allclose( c, numpy.round( c ), atol = 1e-9 ) and abs( abs( numpy.linalg.det( c ) ) - 1 ) < 1e-9, c
        pos = rng.uniform( -1.0, 1.5, size = ( 12, 2 ) )
        w = rng.uniform( -0.01, 0.01, size = 12 )
        _compare( pos, w, hull, hull.domain, numpy.asarray( basis ), tol = 1e-9, length = 6 )


if test( "periodicity_is_checked" ):
    # a support that is not inside a fundamental domain, opposite sides that no translation glues, gaussians without their domain
    import numpy
    from sdot.PowerDiagram import PowerDiagram
    from sdot.distributions.Box import Box
    from sdot.distributions.Polytope import ConvexHull
    from sdot.distributions.SumOfGaussians import SumOfGaussians

    def raises( f, what ):
        try:
            f()
        except ValueError as e:
            assert what in str( e ), str( e )
        else:
            raise AssertionError( what )

    pos = numpy.random.default_rng( 10 ).uniform( 0, 1, size = ( 6, 2 ) )
    flip, rot = numpy.diag( [ 1.0, -1.0 ] ), numpy.array( [ [ 0.0, -1.0 ], [ 1.0, 0.0 ] ] )
    for gens in ( [ [ 0.5, 0.0 ], [ 0.0, 1.0 ] ],                   # a period shorter than the box
                  [ [ 1.0, 0.0 ], [ 0.5, 0.5 ] ],                   # a sheared lattice of covolume 1 / 2
                  [ ( rot, [ 0.5, 0.5 ] ) ],                        # a quarter turn about the center
                  [ ( flip, [ 0.5, 1.0 ] ), [ 0.0, 1.0 ] ] ):       # a Klein bottle of half the width
        raises( lambda: PowerDiagram( pos, distribution = Box(), periodicity = gens ).measures, "not inside a fundamental domain" )
    c72 = 2 * numpy.pi / 5
    R72 = numpy.array( [ [ numpy.cos( c72 ), -numpy.sin( c72 ) ], [ numpy.sin( c72 ), numpy.cos( c72 ) ] ] )
    R1 = numpy.array( [ [ numpy.cos( 1 ), -numpy.sin( 1 ) ], [ numpy.sin( 1 ), numpy.cos( 1 ) ] ] )
    small = ConvexHull( [ [ 0, 0 ], [ 0.1, 0 ], [ 0.1, 0.01 ] ] )
    for dist, gens, what in ( ( Box(), [ [ 1, 0 ], [ numpy.sqrt( 2 ), 0 ] ], "not discrete" ),          # dense translations
                              ( Box(), [ [ 1, 0 ], [ 1.1, 0 ] ], "not inside a fundamental domain" ),     # steps of 0.1
                              ( Box(), [ [ 0, 0 ] ], "identity" ),
                              ( small, [ ( R1, [ 0, 0 ] ) ], "too many elements" ),                       # a turn of 1 radian
                              ( small, [ ( R72, [ 0, 0 ] ), [ 1, 0 ] ], "not discrete" ),                 # a fivefold turn and a lattice
                              ( small, [ ( flip, [ 0, 0 ] ), ( R1 @ flip @ R1.T, [ 0, 0 ] ) ], "too many elements" ),   # mirrors at 1 radian
                              ( Box(), [ ( R72, [ 0, 0 ] ) ], "not inside a fundamental domain" ),
                              ( Box(), [ ( - numpy.eye( 2 ), [ 1, 1 ] ) ], "not inside a fundamental domain" ) ):
        raises( lambda: PowerDiagram( pos, distribution = dist, periodicity = gens ).measures, what )
    # a hexagon whose opposite sides are parallel but not of the same length
    hexa = ConvexHull( [ [ 0, 0 ], [ 3, 0 ], [ 4, 1 ], [ 4, 3 ], [ 2, 3 ], [ 0, 1 ] ] )
    raises( lambda: PowerDiagram( pos, distribution = hexa, periodicity = hexa.domain ).measures, "not translates" )
    raises( lambda: PowerDiagram( pos, distribution = SumOfGaussians( [ [ 0.5, 0.5 ] ], [ 0.1 ] ), periodicity = True ).measures, "support_box" )


if test( "more_groups_match_the_copies" ):
    # valid groups that the other tests do not cover: redundant translations, a fivefold cone, a glide, a point reflection, a mirror
    # and a translation, quarter turns with an oblique translation, a screw by an eighth of a turn ( no translation in short words )
    import numpy
    from sdot.distributions.Box import Box
    from sdot.distributions.Polytope import ConvexHull
    from sdot.periodicity import support_box
    def rot( a ):
        return numpy.array( [ [ numpy.cos( a ), -numpy.sin( a ) ], [ numpy.sin( a ), numpy.cos( a ) ] ] )
    c72 = 2 * numpy.pi / 5
    rz = numpy.eye( 3 )
    rz[ :2, :2 ] = rot( numpy.pi / 4 )
    cases = (
        ( Box( nb_dims = 2 ), [ [ 1, 0 ], [ 2, 0 ] ], 6 ),
        ( ConvexHull( [ [ 0, 0 ], [ 0.1, 0 ], [ 0.1, 1 ], [ 0, 1 ] ] ), [ [ 1, 0 ], [ 1.1, 0 ] ], 6 ),
        ( ConvexHull( [ [ 0, 0 ], [ 1, 0 ], [ numpy.cos( c72 ), numpy.sin( c72 ) ] ] ), [ ( rot( c72 ), [ 0, 0 ] ) ], 6 ),
        ( Box( nb_dims = 2 ), [ ( numpy.diag( [ 1.0, -1.0 ] ), [ 1, 0 ] ) ], 6 ),
        ( Box( nb_dims = 2 ), [ ( - numpy.eye( 2 ), [ 0, 0 ] ) ], 6 ),
        ( ConvexHull( [ [ 0, 0 ], [ 0.5, 0 ], [ 0.5, 1 ], [ 0, 1 ] ] ), [ ( numpy.diag( [ -1.0, 1.0 ] ), [ 0, 0 ] ), [ 1, 0 ] ], 6 ),
        ( ConvexHull( [ [ 0, 0 ], [ 0.3, 0 ], [ 0.3, 0.3 ], [ 0, 0.3 ] ] ), [ ( rot( numpy.pi / 2 ), [ 0, 0 ] ), [ 1, 0.3 ] ], 6 ),
        ( Box( nb_dims = 3 ), [ ( rz, [ 0, 0, 1 ] ) ], 8 ),
    )
    rng = numpy.random.default_rng( 11 )
    for dist, gens, length in cases:
        lo, hi = support_box( dist.bounding_half_spaces(), len( numpy.asarray( gens[ 0 ][ 1 ] if isinstance( gens[ 0 ], tuple ) else gens[ 0 ] ) ) )
        pos = lo + ( hi - lo ) * rng.uniform( 0.1, 0.9, size = ( 6, len( lo ) ) )
        _compare( pos, rng.uniform( -1e-3, 1e-3, size = 6 ), dist, gens, gens, tol = 1e-8, length = length )


if test( "isometry_groups_match_the_copies" ):
    import numpy
    from loom.testing import need
    from sdot.distributions.Image import Image
    from sdot.distributions.Box import Box
    rng = numpy.random.default_rng( 8 )
    flip = numpy.diag( [ 1.0, -1.0 ] )
    rot = numpy.array( [ [ 0.0, -1.0 ], [ 1.0, 0.0 ] ] )
    img = Image( values = rng.uniform( 0.5, 1.5, size = ( 4, 3 ) ), frame = [ 1.0, 1.0 ] )
    pos = rng.uniform( -1.5, 2.5, size = ( 14, 2 ) )
    w = rng.uniform( -0.01, 0.01, size = 14 )
    cases = {
        # a Klein bottle: across x, the square comes back upside down
        "klein"  : [ ( flip, [ 1.0, 1.0 ] ), [ 0.0, 1.0 ] ],
        # a Moebius strip: the same glue, walls at y = 0 and y = 1 ( swapped by the group: kept )
        "moebius": [ ( flip, [ 1.0, 1.0 ] ) ],
        # a cone: the quarter turn about the corner, no translation at all
        "cone"   : [ ( rot, [ 0.0, 0.0 ] ) ],
        # an oblique cylinder: glued along a diagonal translation ( its images of the square only touch at a corner )
        "oblique": [ [ 1.0, 1.0 ] ],
    }
    for name, gens in cases.items():
        _compare( pos if name != "moebius" else numpy.clip( pos, [ -2, 0 ], [ 3, 1 ] ), w, img, gens, gens, tol = 1e-9, length = 4 )
    # the wallpaper group p4: quarter turns about the origin and the unit translations; its fundamental domain is a square of side 1 / 2
    quarter = Image( values = rng.uniform( 0.5, 1.5, size = ( 3, 3 ) ), frame = [ 0.5, 0.5 ] )
    gens = [ ( rot, [ 0.0, 0.0 ] ), [ 1.0, 0.0 ] ]
    _compare( rng.uniform( -1, 1, size = ( 10, 2 ) ), w[ :10 ], quarter, gens, gens, tol = 1e-9, length = 5 )


if test( "isometry_groups_3d_and_derivatives" ):
    import numpy
    from loom.testing import need, check_grad
    from sdot.distributions.Box import Box
    from sdot.distributions.Image import Image
    from sdot.PowerDiagram import PowerDiagram
    from sdot.Cell import set_kernel_dtype
    set_kernel_dtype( "FP64" )
    need( "grad" )
    rng = numpy.random.default_rng( 9 )
    # a screw: a quarter turn about the vertical axis of the cube for each step up, and the horizontal unit translations
    rot = numpy.array( [ [ 0.0, -1.0, 0.0 ], [ 1.0, 0.0, 0.0 ], [ 0.0, 0.0, 1.0 ] ] )
    gens = [ ( rot, [ 1.0, 0.0, 1.0 ] ), [ 1.0, 0.0, 0.0 ], [ 0.0, 1.0, 0.0 ] ]
    pos = rng.uniform( -0.5, 1.5, size = ( 12, 3 ) )
    w = rng.uniform( -0.01, 0.01, size = 12 )
    _compare( pos, w, Box( nb_dims = 3 ), gens, gens, tol = 1e-9 )
    # the derivatives go back to the seeds through the linear part of the images
    flip = numpy.diag( [ 1.0, -1.0 ] )
    img = Image( values = rng.uniform( 0.5, 1.5, size = ( 3, 3 ) ), frame = [ 1.0, 1.0 ] )
    klein = [ ( flip, [ 1.0, 1.0 ] ), [ 0.0, 1.0 ] ]
    check_grad( lambda p, q: PowerDiagram( p, weights = q, distribution = img, periodicity = klein ).measures,
                rng.uniform( -0.5, 1.5, size = ( 8, 2 ) ), rng.uniform( -0.02, 0.02, size = 8 ) )


if test( "isometry_plan_is_differentiable" ):
    import numpy
    from loom.testing import check_grad, need
    from sdot.OtProblem import Iterative, OtProblem
    from sdot.distributions.Image import Image
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    need( "cpu" )
    # a Klein bottle: the barycenters come back through the flips, and so do the derivatives
    rng = numpy.random.default_rng( 10 )
    flip = numpy.diag( [ 1.0, -1.0 ] )
    klein = [ ( flip, [ 1.0, 1.0 ] ), [ 0.0, 1.0 ] ]
    pos = rng.uniform( -0.5, 1.5, size = ( 10, 2 ) )
    nu = rng.uniform( 0.5, 1.5, size = 10 )
    values = rng.uniform( 0.5, 1.5, size = ( 3, 3 ) )
    settings = Iterative( tol = 1e-13, max_iter = 200 )

    def solve( p, q, v ):
        return OtProblem( SumOfDiracs( p, weights = q ), Image( values = v, frame = [ 1.0, 1.0 ] ), periodicity = klein ).solve( settings )

    sol = solve( pos, nu, values )
    assert sol.converged
    assert numpy.allclose( numpy.asarray( sol.cell_masses ).reshape( -1 ), nu / nu.sum(), atol = 1e-9 )
    for output, tol in ( ( "weights", 1e-5 ), ( "cost", 1e-3 ), ( "barycenters", 1e-3 ) ):
        check_grad( lambda p, q, v: getattr( solve( p, q, v ), output ), pos, nu, values, seed = 3, eps = 1e-5, rtol = tol, atol = tol * 1e-2 )


if test( "periodic_card_cells_and_solve" ):
    import numpy
    import loom
    from errand import skip
    from sdot import plan
    from sdot.distributions.Box import Box
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from sdot.PowerDiagram import PowerDiagram
    if not getattr( loom.resolved_device(), "is_cuda_gpu", False ):
        skip( "the card's cells only exist on a CUDA device" )
    # the card's 2D cells walk the translations of the images ( `gpu/Cell2D.cuh` ): the same measures, moments and laplacian as the
    # generic cells, and the card's solve the same plan as the copies
    rng = numpy.random.default_rng( 11 )
    box = Box( nb_dims = 2 )
    for n in ( 3, 200 ):
        pos = rng.uniform( -0.5, 1.5, size = ( n, 2 ) )
        w = rng.uniform( -0.01, 0.01, size = n ) / n
        for kd in ( "FP64", "FP32" ):
            pd = PowerDiagram( pos, w, distribution = box, periodicity = True, kernel_dtype = kd )
            card = pd._card_cells( facets = True, moments = True )
            assert card is not None, "the card should take a periodic diagram of translations"
            pd.use_card_cells = False
            gen = numpy.asarray( pd.measures.value ).reshape( -1 )
            tol = 1e-10 if kd == "FP64" else 1e-6
            assert numpy.allclose( numpy.asarray( card[ "measures" ].raw ).reshape( -1 ), gen, atol = tol ), ( n, kd )
            m, f, _ = pd.moments
            bary = numpy.asarray( f.value ).reshape( n, 2 ) / numpy.maximum( numpy.asarray( m.value ).reshape( n, 1 ), 1e-300 )
            # ( the card's barycentres are next to the folded seeds, the moments' next to the given ones )
            assert numpy.allclose( numpy.asarray( pd.unfold( numpy.asarray( card[ "bary" ].raw ).reshape( n, 2 ) ) ), bary, atol = 1e-5 ), ( n, kd )
            # the laplacian: its rows sum to zero and it is symmetric ( two images of a neighbour merged into one weight )
            nnz = int( card[ "nnz" ] )
            row = numpy.asarray( card[ "row" ].raw ).reshape( -1 )[ : n + 1 ].astype( int )
            col = numpy.asarray( card[ "col" ].raw ).reshape( -1 )[ : nnz ].astype( int )
            val = numpy.asarray( card[ "val" ].raw ).reshape( -1 )[ : nnz ]
            L = numpy.zeros( ( n, n ) )
            for i in range( n ):
                for e in range( row[ i ], row[ i + 1 ] ):
                    L[ i, col[ e ] ] += val[ e ]
            assert numpy.allclose( L, L.T, atol = 1e-12 )
    # the card's solve
    pos = rng.uniform( -1, 2, size = ( 300, 2 ) )
    p = plan( SumOfDiracs( pos ), box, periodicity = True )
    assert p.converged, p.stats[ "status" ]
    assert numpy.allclose( numpy.asarray( p.cell_masses ).reshape( -1 ), 1 / 300, atol = 1e-8 )
    pd = PowerDiagram( pos, numpy.asarray( p.weights ).reshape( -1 ), distribution = box, periodicity = True, kernel_dtype = "FP64" )
    folded = numpy.asarray( pd._stored_positions().raw ).reshape( pos.shape )
    rm, rf = _reference( folded, numpy.asarray( p.weights ).reshape( -1 ), box.normalized_version( nb_dims = 2 ), numpy.eye( 2 ) )
    assert numpy.allclose( numpy.asarray( p.barycenters ).reshape( -1, 2 ), numpy.asarray( pd.unfold( rf / rm[ :, None ] ) ), atol = 1e-7 )


if test( "periodic_card_cells_3d" ):
    import numpy
    import loom
    from errand import skip
    from sdot import plan
    from sdot.distributions.Box import Box
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from sdot.PowerDiagram import PowerDiagram
    if not getattr( loom.resolved_device(), "is_cuda_gpu", False ):
        skip( "the card's cells only exist on a CUDA device" )
    # the card's 3D cells ( `gpu/Cell3D.cuh` ) with the images of a torus and of a cylinder, against the generic cells
    rng = numpy.random.default_rng( 12 )
    box = Box( nb_dims = 3 )
    for per in ( True, [ True, False, True ] ):
        pos = rng.uniform( -0.5, 1.5, size = ( 150, 3 ) )
        w = rng.uniform( -0.01, 0.01, size = 150 ) / 150
        pd = PowerDiagram( pos, w, distribution = box, periodicity = per, kernel_dtype = "FP64" )
        card = pd._card_cells( facets = True, moments = True )
        assert card is not None
        pd.use_card_cells = False
        assert numpy.allclose( numpy.asarray( card[ "measures" ].raw ).reshape( -1 ), numpy.asarray( pd.measures.value ).reshape( -1 ), atol = 1e-10 )
    p = plan( SumOfDiracs( rng.uniform( -1, 2, size = ( 200, 3 ) ) ), box, periodicity = True )
    assert p.converged, p.stats[ "status" ]
    assert numpy.allclose( numpy.asarray( p.cell_masses ).reshape( -1 ), 1 / 200, atol = 1e-8 )


if test( "periodic_cells_are_drawn_whole_around_the_given_diracs" ):
    import numpy
    from sdot.distributions.Box import Box
    from sdot.PowerDiagram import PowerDiagram
    from sdot.viz.Visualizer import Visualizer
    # the cells are those of the diracs as given ( outside the domain, away from the others ), their moments too; the rings of the
    # group and the dashed outline are what a display repeats
    rng = numpy.random.default_rng( 13 )
    pos = rng.uniform( 0, 1, size = ( 10, 2 ) )
    pos[ :3 ] += [ [ 2.0, 0.0 ], [ -1.0, 3.0 ], [ 0.0, -2.0 ] ]
    w = rng.uniform( -0.01, 0.01, size = 10 )
    pd = PowerDiagram( pos, w, distribution = Box( nb_dims = 2 ), periodicity = True, kernel_dtype = "FP64" )
    cells = pd.cells
    m, f, _ = pd.moments
    bary = numpy.asarray( f.value ).reshape( -1, 2 ) / numpy.asarray( m.value ).reshape( -1, 1 )
    for i in range( 10 ):
        v = cells.vertices( i )
        assert ( v.min( axis = 0 ) - 1e-9 <= bary[ i ] ).all() and ( bary[ i ] <= v.max( axis = 0 ) + 1e-9 ).all()
        assert numpy.linalg.norm( bary[ i ] - pos[ i ] ) < 1.0
    assert len( pd.periodicity_rings( 1 ) ) == 9 and len( pd.periodicity_rings( 2 ) ) == 25
    viz = pd.add_to_viz( Visualizer() )
    assert viz.nb_frames == 1


def _square_mesh( k, values ):
    """the unit square cut into `2 k^2` triangles, P1 `values` per node ( a function of the nodes )"""
    import itertools
    import numpy
    from sdot.distributions.Mesh import Mesh
    g = numpy.linspace( 0, 1, k + 1 )
    nodes = numpy.array( list( itertools.product( g, g ) ) )
    sx = []
    for i, j in itertools.product( range( k ), range( k ) ):
        a, b, c, d = i * ( k + 1 ) + j, i * ( k + 1 ) + j + 1, ( i + 1 ) * ( k + 1 ) + j, ( i + 1 ) * ( k + 1 ) + j + 1
        sx += [ [ a, b, d ], [ a, d, c ] ]
    return Mesh( nodes, numpy.array( sx ), values( nodes ) )


if test( "periodic_measures_gaussians_match_the_copies" ):
    # the WRAPPED gaussians: the periodic diagram against the copies of the seeds in the plane, integrating the gaussians folded
    # onto the square and unbounded ( `support_sigmas` wide ) -- the tails cross the sides, and one center is given outside
    import numpy
    from sdot.distributions.SumOfGaussians import SumOfGaussians
    from sdot.PowerDiagram import PowerDiagram
    from sdot.periodicity import wrap
    rng = numpy.random.default_rng( 14 )
    c = numpy.array( [ [ 0.05, 0.5 ], [ 0.9, 0.95 ], [ 0.5, 0.4 ], [ 1.2, -0.1 ] ] )
    s = numpy.array( [ 0.08, 0.12, 0.2, 0.1 ] )
    wg = numpy.array( [ 1.0, 2.0, 1.5, 0.7 ] )
    flip = numpy.diag( [ 1.0, -1.0 ] )
    for gens, per in ( ( numpy.eye( 2 ), True ), ( [ ( flip, [ 1.0, 1.0 ] ), [ 0.0, 1.0 ] ], [ ( flip, [ 1.0, 1.0 ] ), [ 0.0, 1.0 ] ] ) ):
        dist = SumOfGaussians( c, s, wg, support_box = ( [ 0, 0 ], [ 1, 1 ] ), support_sigmas = 8 )
        ref = SumOfGaussians( wrap( c, per, dist ), s, wg, support_sigmas = 12 )
        pos = rng.uniform( -0.3, 1.3, size = ( 12, 2 ) )
        w = rng.uniform( -0.005, 0.005, size = 12 )
        pd = PowerDiagram( pos, w, distribution = dist, periodicity = per, kernel_dtype = "FP64" )
        folded = numpy.asarray( pd._stored_positions().raw ).reshape( pos.shape )
        m = numpy.asarray( pd.measures.value ).reshape( -1 )
        _, f, _ = pd.moments
        f = numpy.asarray( f.value ).reshape( -1, 2 )
        rm, rf = _reference( folded, w, ref, gens, 4 )
        assert numpy.allclose( m, rm, atol = 1e-11 ), numpy.abs( m - rm ).max()
        rb = numpy.asarray( pd.unfold( rf / rm[ :, None ] ) )
        assert numpy.allclose( f, rb * rm[ :, None ], atol = 1e-11 ), numpy.abs( f - rb * rm[ :, None ] ).max()
        assert abs( m.sum() - 1 ) < 1e-11


if test( "periodic_measures_mesh_match_the_copies" ):
    # a mesh of the unit square: the fundamental domain is its box, without the margin of its support
    import numpy
    rng = numpy.random.default_rng( 15 )
    mesh = _square_mesh( 5, lambda x: 0.5 + x[ :, 0 ] + 2 * x[ :, 1 ] ** 2 )
    pos = rng.uniform( -0.5, 1.5, size = ( 14, 2 ) )
    w = rng.uniform( -0.01, 0.01, size = 14 )
    _compare( pos, w, mesh, True, numpy.eye( 2 ) )
    flip = numpy.diag( [ 1.0, -1.0 ] )
    klein = [ ( flip, [ 1.0, 1.0 ] ), [ 0.0, 1.0 ] ]
    _compare( pos, w, mesh, klein, klein, tol = 1e-9, length = 4 )


if test( "periodic_gaussians_and_mesh_derive" ):
    import numpy
    from loom.testing import check_grad, need
    from sdot.distributions.SumOfGaussians import SumOfGaussians
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from sdot.distributions.Mesh import Mesh
    from sdot.OtProblem import Iterative, OtProblem
    from sdot.PowerDiagram import PowerDiagram
    need( "grad" )
    # the images are copies made by the framework: the derivatives go back to the parameters of the gaussians and to the values
    # of the mesh, through the diagrams and through the plan ( the implicit function theorem )
    rng = numpy.random.default_rng( 16 )
    pos = rng.uniform( -1, 2, size = ( 12, 2 ) )
    w = rng.uniform( -0.01, 0.01, size = 12 )
    c, s = numpy.array( [ [ 0.05, 0.5 ], [ 0.9, 0.95 ], [ 0.5, 0.4 ] ] ), numpy.array( [ 0.08, 0.12, 0.2 ] )
    box = ( [ 0, 0 ], [ 1, 1 ] )
    check_grad( lambda q, cc, ss, ww: PowerDiagram( q, w, distribution = SumOfGaussians( cc, ss, ww, support_box = box ), periodicity = True ).measures,
                pos, c, s, numpy.array( [ 1.0, 2.0, 1.5 ] ) )
    mesh = _square_mesh( 4, lambda x: 1 + x[ :, 0 ] )
    nodes, sx = numpy.asarray( mesh.nodes ), numpy.asarray( mesh.simplices )
    check_grad( lambda q, v: PowerDiagram( q, w, distribution = Mesh( nodes, sx, v ), periodicity = True ).measures,
                pos, rng.uniform( 0.3, 2.0, size = len( nodes ) ) )
    nu = rng.uniform( 0.5, 1.5, size = 12 )
    settings = Iterative( tol = 1e-13, max_iter = 200 )

    def solve( q, cc ):
        return OtProblem( SumOfDiracs( q, weights = nu ), SumOfGaussians( cc, s, support_box = box ), periodicity = True ).solve( settings )

    for output, tol in ( ( "weights", 1e-5 ), ( "cost", 1e-3 ), ( "barycenters", 1e-3 ) ):
        check_grad( lambda q, cc: getattr( solve( q, cc ), output ), pos, c, seed = 3, eps = 1e-5, rtol = tol, atol = tol * 1e-2 )


if test( "periodic_plan_gaussians_and_mesh" ):
    import numpy
    from loom.testing import need
    from sdot.OtProblem import Iterative, OtProblem
    from sdot.distributions.SumOfGaussians import SumOfGaussians
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    need( "cpu" )
    rng = numpy.random.default_rng( 17 )
    pos = rng.uniform( -1, 2, size = ( 40, 2 ) )
    mass = rng.uniform( 0.5, 1.5, size = 40 )
    flip = numpy.diag( [ 1.0, -1.0 ] )
    targets = ( SumOfGaussians( [ [ 0.05, 0.5 ], [ 0.9, 0.95 ], [ 0.5, 0.4 ] ], [ 0.08, 0.12, 0.2 ], support_box = ( [ 0, 0 ], [ 1, 1 ] ) ),
                _square_mesh( 6, lambda x: 0.3 + 2 * x[ :, 0 ] * x[ :, 1 ] ) )
    for tgt in targets:
        for per in ( True, [ ( flip, [ 1.0, 1.0 ] ), [ 0.0, 1.0 ] ] ):
            p = OtProblem( SumOfDiracs( pos, mass ), tgt, periodicity = per ).solve( Iterative( tol = 1e-11 ) )
            assert p.converged, p.stats[ "status" ]
            assert numpy.allclose( numpy.asarray( p.cell_masses ).reshape( -1 ), mass / mass.sum(), atol = 1e-9 )


if test( "periodic_card_solve_of_gaussians_and_meshes" ):
    # the card's solve of a group of translations, towards targets that repeat themselves ( `Distribution.periodized` ): the card
    # integrates the copies as any other gaussians or mesh. The barycenters: those of the generic cells at the solved weights
    import numpy
    import loom
    from errand import skip
    from sdot import plan
    from sdot.distributions.SumOfGaussians import SumOfGaussians
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from sdot.PowerDiagram import PowerDiagram
    if not getattr( loom.resolved_device(), "is_cuda_gpu", False ):
        skip( "the card's solve only exists on a CUDA device" )
    rng = numpy.random.default_rng( 18 )
    targets = [
        ( 2, SumOfGaussians( [ [ 0.05, 0.5 ], [ 0.9, 0.95 ], [ 0.5, 0.4 ] ], [ 0.08, 0.12, 0.2 ], support_box = ( [ 0, 0 ], [ 1, 1 ] ) ) ),
        ( 2, _square_mesh( 6, lambda x: 0.3 + 2 * x[ :, 0 ] * x[ :, 1 ] ) ),
        ( 3, SumOfGaussians( [ [ 0.05, 0.5, 0.5 ], [ 0.9, 0.95, 0.1 ] ], [ 0.12, 0.2 ], support_box = ( [ 0, 0, 0 ], [ 1, 1, 1 ] ) ) ),
    ]
    for d, tgt in targets:
        n = 300 if d == 2 else 200
        pos = rng.uniform( -1, 2, size = ( n, d ) )
        p = plan( SumOfDiracs( pos ), tgt, periodicity = True )
        assert p.converged, ( d, type( tgt ).__name__, p.stats[ "status" ] )
        assert numpy.allclose( numpy.asarray( p.cell_masses ).reshape( -1 ), 1 / n, atol = 1e-8 )
        pd = PowerDiagram( pos, numpy.asarray( p.weights ).reshape( -1 ), distribution = tgt, periodicity = True, kernel_dtype = "FP64" )
        pd.use_card_cells = False
        m, f, _ = pd.moments
        bary = numpy.asarray( f.value ).reshape( n, d ) / numpy.asarray( m.value ).reshape( n, 1 )
        assert numpy.allclose( numpy.asarray( m.value ).reshape( -1 ), 1 / n, atol = 1e-7 ), ( d, type( tgt ).__name__ )
        assert numpy.allclose( numpy.asarray( p.barycenters ).reshape( n, d ), bary, atol = 1e-6 ), ( d, type( tgt ).__name__ )


if test( "exact_grids_on_a_torus" ):
    # a grid of seeds on a torus: every cell is the cube of its seed. The bisectors of the neighbors go through vertices and edges
    # that other bisectors made, exactly: a vertex that should lie on a plane is there up to the rounding of its LARGEST coordinate,
    # and the band that tells "on the plane" must be relative to that ( `LocalN::magnitude` ) -- relative to the terms of the dot
    # product, it vanished, the cut could not be carved and was dropped, and cells of 7 / 6 h^3 came out ( 14^3 was one )
    import numpy
    from sdot.Cell import set_kernel_dtype
    from sdot.PowerDiagram import PowerDiagram, box_half_spaces
    for kd, tol in ( ( "FP64", 1e-9 ), ( "FP32", 1e-4 ) ):
        set_kernel_dtype( kd )
        for d in ( 2, 3 ):
            for m in ( 4, 7, 8, 14, 16 ):
                g = ( numpy.stack( numpy.meshgrid( *[ numpy.arange( m ) + 0.5 ] * d, indexing = "ij" ), -1 ).reshape( -1, d ) ) / m
                for acc in ( None, "plain" ):
                    pd = PowerDiagram( g, boundaries = box_half_spaces( numpy.zeros( d ), numpy.ones( d ) ), periodicity = True, accelerator = acc )
                    mz = numpy.asarray( pd.measures ).reshape( -1 ) * m ** d
                    assert numpy.abs( mz - 1 ).max() < tol, ( kd, d, m, acc, numpy.flatnonzero( numpy.abs( mz - 1 ) >= tol ) )
    set_kernel_dtype( "FP64" )
