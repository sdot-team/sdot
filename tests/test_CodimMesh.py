"""A density carried by a mesh of lower dimension ( `src/sdot/distributions/CodimMesh.py`, `src/sdot/include/sdot/CodimMesh.h` ):
`rho exp( -d^2 / width^2 )` on the envelope of the mesh, integrated in closed form.

The witnesses: the closed forms against the quadrature where the quadrature converges ( a wide density ), the cells summing to
the mass whatever the width, the limit `width -> 0` ( `sqrt( pi ) width int_M rho` on lines in 2D and surfaces, `pi width^2
int_M rho` on lines in 3D, the weights of the fans ), the derivatives, the Hessian against its finite difference, and the
transport, solved through the width continuation of the C++ ( `Convolved<CodimMesh>` ) down to the measure on M itself, the last
stage; the measure on M is integrated exactly ( against a sampling of M ), with its derivatives and its Hessian."""

from errand import test


def _ellipse( n = 40, a = 0.3, b = 0.15 ):
    import numpy
    t = numpy.linspace( 0, 2 * numpy.pi, n, endpoint = False )
    nodes = 0.5 + numpy.stack( [ a * numpy.cos( t ), b * numpy.sin( t ) ], 1 )
    return nodes, numpy.stack( [ numpy.arange( n ), ( numpy.arange( n ) + 1 ) % n ], 1 )


def _tet_surface():
    import numpy
    nodes = 0.3 + 0.4 * numpy.array( [ [ 0, 0, 0 ], [ 1, 0, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ] ], float )
    return nodes, numpy.array( [ [ 0, 1, 2 ], [ 0, 1, 3 ], [ 0, 2, 3 ], [ 1, 2, 3 ] ] )


def _loop_3d( n = 24 ):
    import numpy
    t = numpy.linspace( 0, 2 * numpy.pi, n, endpoint = False )
    nodes = 0.5 + numpy.stack( [ 0.3 * numpy.cos( t ), 0.3 * numpy.sin( t ), 0.1 * numpy.sin( 2 * t ) ], 1 )
    return nodes, numpy.stack( [ numpy.arange( n ), ( numpy.arange( n ) + 1 ) % n ], 1 )


def _length_or_area( nodes, sx ):
    import numpy
    if sx.shape[ 1 ] == 2:
        return numpy.linalg.norm( nodes[ sx[ :, 1 ] ] - nodes[ sx[ :, 0 ] ], axis = 1 ).sum()
    return 0.5 * numpy.linalg.norm( numpy.cross( nodes[ sx[ :, 1 ] ] - nodes[ sx[ :, 0 ] ], nodes[ sx[ :, 2 ] ] - nodes[ sx[ :, 0 ] ] ), axis = 1 ).sum()


if test( "mesh_with_segments_is_a_codim_mesh" ):
    from sdot.distributions.CodimMesh import CodimMesh
    from sdot.distributions.Mesh import Mesh
    nodes, sx = _ellipse()
    assert isinstance( Mesh( nodes, sx ), CodimMesh )


if test( "the_mass_tends_to_the_measure_on_m" ):
    # closed curves and surfaces: no end, the relative error is `O( width^2 )` ( the curvature )
    import numpy
    from sdot.distributions.Mesh import Mesh
    for name, ( nodes, sx ), k in ( ( "ellipse", _ellipse( 80 ), 1 ), ( "tet_surface", _tet_surface(), 1 ), ( "loop_3d", _loop_3d(), 2 ) ):
        L = _length_or_area( nodes, sx )
        errs = []
        for width in ( 0.02, 0.005, 0.00125 ):
            m = Mesh( nodes, sx, width = width, alpha = 15, smoothed = True )
            ref = ( numpy.sqrt( numpy.pi ) * width if k == 1 else numpy.pi * width ** 2 ) * L
            errs.append( abs( float( m.mass ) / ref - 1 ) )
        assert errs[ -1 ] < 1e-2 and errs[ -1 ] < errs[ 0 ], ( name, errs )


if test( "the_closed_forms_are_the_quadrature" ):
    import numpy
    from sdot.distributions.Mesh import Mesh
    from sdot.PowerDiagram import PowerDiagram
    rng = numpy.random.default_rng( 0 )
    for nodes, sx in ( _ellipse(), _tet_surface(), _loop_3d( 12 ) ):
        D = nodes.shape[ 1 ]
        pos = rng.uniform( 0.2, 0.8, ( 12, D ) )
        m = Mesh( nodes, sx, width = 0.3, alpha = 30, smoothed = True )
        a = numpy.asarray( PowerDiagram( pos, distribution = m, kernel_dtype = "FP64" ).measures ).reshape( -1 )
        b = numpy.asarray( PowerDiagram( pos, distribution = m.with_width( 0.3, quadrature = True ), kernel_dtype = "FP64" ).measures ).reshape( -1 )
        assert numpy.abs( a - b ).max() < 2e-4 * a.max(), ( D, numpy.abs( a - b ).max() / a.max() )


if test( "the_cells_sum_to_the_mass_at_any_width" ):
    import numpy
    from sdot.distributions.Mesh import Mesh
    from sdot.PowerDiagram import PowerDiagram
    rng = numpy.random.default_rng( 1 )
    for nodes, sx in ( _ellipse(), _tet_surface(), _loop_3d( 12 ) ):
        D = nodes.shape[ 1 ]
        pos = rng.uniform( 0.2, 0.8, ( 30, D ) )
        for width in ( 0.3, 0.01, 1e-4 ):
            m = Mesh( nodes, sx, width = width, alpha = 30, smoothed = True )
            a = numpy.asarray( PowerDiagram( pos, distribution = m, kernel_dtype = "FP64" ).measures ).reshape( -1 )   # ( normalized )
            assert abs( a.sum() - 1 ) < 1e-10 and ( a >= 0 ).all(), ( D, width, a.sum() - 1 )


def _sampled_masses( nodes, sx, rho, pos, w, k = 400 ):
    """the masses of the power cells of `( pos, w )` for `rho` on M, by sampling M ( `k` points per edge, midpoints )"""
    import numpy
    if sx.shape[ 1 ] == 2:
        t = ( numpy.arange( k ) + 0.5 ) / k
        A, B = nodes[ sx[ :, 0 ] ], nodes[ sx[ :, 1 ] ]
        x = A[ :, None ] + t[ None, :, None ] * ( B - A )[ :, None ]
        r = rho[ sx[ :, 0 ], None ] + t[ None ] * ( rho[ sx[ :, 1 ] ] - rho[ sx[ :, 0 ] ] )[ :, None ]
        dm = numpy.linalg.norm( B - A, axis = 1 )[ :, None ] / k * r
    else:
        i, j = numpy.meshgrid( numpy.arange( k ), numpy.arange( k ), indexing = "ij" )
        ok = i + j < k                                          # ( the midpoints of the k^2 small triangles, up and down )
        a = numpy.concatenate( [ ( i[ ok ] + 1 / 3 ), ( i[ ok & ( i + j < k - 1 ) ] + 2 / 3 ) ] ) / k
        b = numpy.concatenate( [ ( j[ ok ] + 1 / 3 ), ( j[ ok & ( i + j < k - 1 ) ] + 2 / 3 ) ] ) / k
        A, B, C = ( nodes[ sx[ :, q ] ] for q in range( 3 ) )
        x = A[ :, None ] + a[ None, :, None ] * ( B - A )[ :, None ] + b[ None, :, None ] * ( C - A )[ :, None ]
        r = rho[ sx[ :, 0 ], None ] + a[ None ] * ( rho[ sx[ :, 1 ] ] - rho[ sx[ :, 0 ] ] )[ :, None ] + b[ None ] * ( rho[ sx[ :, 2 ] ] - rho[ sx[ :, 0 ] ] )[ :, None ]
        area = 0.5 * numpy.linalg.norm( numpy.cross( B - A, C - A ), axis = 1 )
        dm = area[ :, None ] / k ** 2 * r
    x = x.reshape( -1, nodes.shape[ 1 ] )
    owner = ( ( ( x[ :, None ] - pos[ None ] ) ** 2 ).sum( 2 ) - w[ None ] ).argmin( 1 )
    return numpy.bincount( owner, weights = dm.reshape( -1 ), minlength = len( pos ) )


if test( "the_measure_on_m_is_integrated_exactly" ):
    # the distribution itself ( the last stage ): the length ( area ) of M in each cell, weighted by `rho`, against a sampling of M
    import numpy
    from sdot.distributions.Mesh import Mesh
    from sdot.PowerDiagram import PowerDiagram
    rng = numpy.random.default_rng( 5 )
    for nodes, sx in ( _ellipse(), _tet_surface(), _loop_3d( 12 ) ):
        D = nodes.shape[ 1 ]
        rho = 1 + nodes @ numpy.arange( 1, D + 1 )
        pos = rng.uniform( 0.25, 0.75, ( 15, D ) )
        w = rng.uniform( -0.01, 0.01, 15 )
        m = Mesh( nodes, sx, values = rho )
        a = numpy.asarray( PowerDiagram( pos, w, distribution = m, kernel_dtype = "FP64" ).measures ).reshape( -1 )
        ref = _sampled_masses( nodes, sx, rho, pos, w )
        ref /= ref.sum()
        assert abs( a.sum() - 1 ) < 1e-12 and numpy.abs( a - ref ).max() < 2e-3 * ref.max(), ( D, a.sum() - 1, numpy.abs( a - ref ).max() / ref.max() )


if test( "a_codim_mesh_carries_its_derivative" ):
    import numpy
    from loom.testing import check_grad, need
    from sdot.distributions.Mesh import Mesh
    from sdot.PowerDiagram import PowerDiagram
    from sdot.Cell import set_kernel_dtype
    set_kernel_dtype( "FP64" )
    need( "grad" )
    nodes, sx = _ellipse( 20 )
    rng = numpy.random.default_rng( 2 )
    pos = rng.uniform( 0.25, 0.75, ( 6, 2 ) )
    w = rng.uniform( -0.01, 0.01, 6 )
    for m in ( Mesh( nodes, sx, width = 0.05, smoothed = True ), Mesh( nodes, sx ) ):              # ( smoothed, the measure on M )
        check_grad( lambda p, q: PowerDiagram( p, weights = q, distribution = m ).measures, pos, w )


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


if test( "the_hessian_of_a_codim_mesh_is_its_finite_difference" ):
    import numpy
    from sdot.distributions.Mesh import Mesh
    from sdot.PowerDiagram import PowerDiagram
    rng = numpy.random.default_rng( 3 )
    for nodes, sx, width, smoothed in ( ( *_ellipse(), 0.03, True ), ( *_tet_surface(), 0.05, True ),
                                        ( *_ellipse(), 0.03, False ), ( *_tet_surface(), 0.05, False ), ( *_loop_3d( 12 ), 0.05, False ) ):
        D = nodes.shape[ 1 ]
        n = 12
        pos = rng.uniform( 0.25, 0.75, ( n, D ) )
        w0 = rng.uniform( -0.01, 0.01, n )
        pd = PowerDiagram( pos, w0, distribution = Mesh( nodes, sx, width = width, alpha = 30, smoothed = smoothed ), kernel_dtype = "FP64" )
        H = _dense_hessian( pd, n )
        h = 1e-6
        for j in ( 0, 5, 11 ):
            e = numpy.zeros( n ); e[ j ] = h
            pd.weights = w0 + e; mp = numpy.asarray( pd.measures ).reshape( -1 )
            pd.weights = w0 - e; mm = numpy.asarray( pd.measures ).reshape( -1 )
            err = numpy.abs( ( mp - mm ) / ( 2 * h ) - H[ :, j ] ).max()
            assert err < 1e-5 * numpy.abs( H ).max(), ( D, smoothed, j, err, numpy.abs( H ).max() )


if test( "the_transport_towards_a_codim_mesh_is_solved" ):
    import numpy
    from sdot.distributions.Mesh import Mesh
    from sdot.distributions.SumOfDiracs import SumOfDiracs
    from sdot.OtProblem import Iterative, OtProblem
    rng = numpy.random.default_rng( 4 )
    for nodes, sx, n in ( ( *_ellipse( 60 ), 200 ), ( *_tet_surface(), 60 ) ):
        D = nodes.shape[ 1 ]
        for width in ( None, 1e-3 ):                              # ( the default, then a smoothed density at `1e-3` )
            target = Mesh( nodes, sx, width = width, smoothed = width is not None )
            lo, hi = target.domain_box()                         # ( a seed out of the domain may keep an empty cell at every width )
            plan = OtProblem( SumOfDiracs( rng.uniform( lo, hi, ( n, D ) ) ), target ).solve( Iterative() )
            m = numpy.asarray( plan.cell_masses ).reshape( -1 )
            assert numpy.abs( m * n - 1 ).max() < 1e-6, ( D, width, numpy.abs( m * n - 1 ).max(), plan.stats.get( "status" ) )


if test( "the_quantization_of_a_codim_mesh_starts_on_m" ):
    # the seeds drawn on M ( `random_points` ): the Voronoi start is admissible for the measure on M, the first solve needs no
    # continuation, and the diracs end next to M ( at the sagitta of their cells )
    import numpy
    from sdot.distributions.Mesh import Mesh
    from sdot.quantization import quantization, uniform_in_support
    for nodes, sx, n in ( ( *_ellipse( 60 ), 60 ), ( *_tet_surface(), 80 ) ):
        target = Mesh( nodes, sx )
        x0 = uniform_in_support( target, n, 0 )
        stats = []
        x = numpy.asarray( quantization( target, x0, callback = lambda it, pos, plan: stats.append( plan.stats ) ) )
        assert stats[ 0 ][ "nb_continuation_steps" ] == 1, stats[ 0 ].get( "nb_continuation_steps" )
        assert len( stats ) < 60, len( stats )
        P = nodes[ sx ]                                          # ( the distance to the nearest node of a fine sampling of M )
        t = numpy.random.default_rng( 1 ).dirichlet( numpy.ones( sx.shape[ 1 ] ), 4000 )
        S = numpy.einsum( "ki,eic->ekc", t, P ).reshape( -1, nodes.shape[ 1 ] )
        d = numpy.sqrt( ( ( x[ :, None, : ] - S[ None, :, : ] ) ** 2 ).sum( -1 ).min( 1 ) )
        h = _length_or_area( nodes, sx ) / n                     # ( a cell: its length, or its area )
        assert d.max() < ( 0.5 * h if sx.shape[ 1 ] == 2 else 0.5 * h ** 0.5 ), ( d.max(), h )
