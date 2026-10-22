"""`ProjectedSumOfTents` against an `Image`, through `ContinuousOtPlan1d` ( the generic 1D transport on typed pieces,
`ot1d/transport_1d.h` ): the cost and the barycenters against a numpy reference in closed form, the gradients with
respect to the points and the half-widths against finite differences, a batch against its items.
"""
import numpy as np

from errand import test


# ── the numpy reference: exact W2 between a sum of tents and a piecewise-constant density ───────────────────────────

_GL_X = np.array( [ -np.sqrt( 0.6 ), 0.0, np.sqrt( 0.6 ) ] )
_GL_W = np.array( [ 5.0, 8.0, 5.0 ] ) / 9.0


def reference( s, w, m, values, s_min, ds ):
    """( cost, b ): s, w, m the tents' centers, half-widths and masses; values the density on bins of width ds"""
    v = np.asarray( values, float )
    bm = v * ds / ( v * ds ).sum()
    e = s_min + ds * np.arange( len( v ) + 1 )
    C = np.concatenate( [ [ 0.0 ], np.cumsum( bm ) ] )

    def quantile( q ):
        j = np.clip( np.searchsorted( C, q, side = "right" ) - 1, 0, len( bm ) - 1 )
        with np.errstate( divide = "ignore", invalid = "ignore" ):
            t = np.where( bm[ j ] > 0, ( q - C[ j ] ) / bm[ j ], 0.0 )
        return e[ j ] + np.clip( t, 0, 1 ) * ds

    m = np.asarray( m, float ) / np.sum( m )
    p = np.concatenate( [ s - w, s, s + w ] )
    c = np.concatenate( [ m / w ** 2, -2 * m / w ** 2, m / w ** 2 ] )
    o = np.argsort( p, kind = "stable" )
    yb, cb = p[ o ], c[ o ]
    sig = np.cumsum( cb )
    dy = np.diff( yb )
    f = np.maximum( np.concatenate( [ [ 0.0 ], np.cumsum( sig[ :-1 ] * dy ) ] ), 0.0 )
    F = np.concatenate( [ [ 0.0 ], np.cumsum( f[ :-1 ] * dy + 0.5 * sig[ :-1 ] * dy * dy ) ] )
    F /= F[ -1 ]

    def cdf( y ):
        k = np.clip( np.searchsorted( yb, y, side = "right" ) - 1, 0, len( yb ) - 2 )
        t = y - yb[ k ]
        return np.clip( F[ k ] + f[ k ] * t + 0.5 * sig[ k ] * t * t, 0, 1 ), f[ k ] + sig[ k ] * t

    lev = C[ 1:-1 ]
    lev = lev[ ( lev > 0 ) & ( lev < 1 ) ]
    k = np.clip( np.searchsorted( F, lev, side = "right" ) - 1, 0, len( yb ) - 2 )
    r = lev - F[ k ]
    disc = np.sqrt( np.maximum( f[ k ] ** 2 + 2 * sig[ k ] * r, 0.0 ) )
    with np.errstate( divide = "ignore", invalid = "ignore" ):
        t = np.where( f[ k ] + disc > 0, 2 * r / ( f[ k ] + disc ), 0.0 )
    P = np.unique( np.concatenate( [ yb, yb[ k ] + np.clip( t, 0, np.diff( yb )[ k ] ) ] ) )
    a, b = P[ :-1 ], P[ 1: ]
    hl = 0.5 * ( b - a )
    yq = 0.5 * ( a + b )[ :, None ] + hl[ :, None ] * _GL_X[ None ]
    Fq, fq = cdf( yq )
    Tq = quantile( Fq )
    wq = hl[ :, None ] * _GL_W[ None ]
    cost = float( ( wq * ( yq - Tq ) ** 2 * fq ).sum() )
    P0 = np.concatenate( [ [ 0.0 ], np.cumsum( ( wq * Tq ).sum( 1 ) ) ] )
    P1 = np.concatenate( [ [ 0.0 ], np.cumsum( ( wq * yq * Tq ).sum( 1 ) ) ] )
    i0, i1, i2 = ( np.searchsorted( P, x ) for x in ( s - w, s, s + w ) )
    bary = ( ( P1[ i1 ] - P1[ i0 ] ) - ( s - w ) * ( P0[ i1 ] - P0[ i0 ] ) + ( s + w ) * ( P0[ i2 ] - P0[ i1 ] ) - ( P1[ i2 ] - P1[ i1 ] ) ) / w ** 2
    return cost, bary


def _problem( nb_angles = 4, n = 30, B = 48, seed = 0 ):
    rng = np.random.default_rng( seed )
    pts = rng.uniform( -0.6, 0.6, ( n, 2 ) )
    hw = rng.uniform( 0.02, 0.08, n )
    angles = np.pi * np.arange( nb_angles ) / nb_angles
    normals = np.stack( [ np.cos( angles ), np.sin( angles ) ], 1 )
    values = rng.random( ( nb_angles, B ) ) + 0.05
    values[ :, 5:9 ] = 0.0
    return pts, hw, normals, values, -1.0, 2.0 / B


def _plan( pts, hw, normals, values, s_min, ds, **kw ):
    from loom.tensor import Axis, ShapeVar
    from sdot import ContinuousOtPlan1d, Image, ProjectedSumOfTents
    num_angle = Axis( ShapeVar( len( normals ) ), name = "num_angle" )
    src = ProjectedSumOfTents( points = pts, half_widths = hw, normal = normals, batch_axes = [ num_angle ] )
    dst = Image( values = values, origin = [ s_min ], pixel_frame = [ [ ds ] ], batch_axes = [ num_angle ] )
    return ContinuousOtPlan1d( src, dst, **kw )


if test( "cost and barycenters against the numpy reference" ):
    pts, hw, normals, values, s_min, ds = _problem()
    plan = _plan( pts, hw, normals, values, s_min, ds, with_barycenters = True )
    cost = np.asarray( plan.cost )
    bary = np.asarray( plan.barycenters )
    for a, nrm in enumerate( normals ):
        c_ref, b_ref = reference( pts @ nrm, hw, np.ones( len( hw ) ), values[ a ], s_min, ds )
        assert abs( cost[ a ] - c_ref ) < 1e-10 * c_ref, ( a, cost[ a ], c_ref )
        assert np.abs( bary[ a ] - b_ref ).max() < 1e-10, ( a, np.abs( bary[ a ] - b_ref ).max() )


if test( "gradients with respect to the points and the half-widths, by finite differences" ):
    import jax
    jax.config.update( "jax_enable_x64", True )
    import jax.numpy as jnp
    pts, hw, normals, values, s_min, ds = _problem( seed = 1 )
    f = lambda p, w: _plan( p, w, normals, values, s_min, ds ).cost.sum()
    gp, gw = jax.grad( f, argnums = ( 0, 1 ) )( jnp.asarray( pts ), jnp.asarray( hw ) )
    gp, gw = np.asarray( gp ), np.asarray( gw )
    rng = np.random.default_rng( 2 )
    dp, dw = rng.normal( size = pts.shape ), rng.normal( size = hw.shape )
    eps = 1e-6
    fd_p = ( float( f( pts + eps * dp, hw ) ) - float( f( pts - eps * dp, hw ) ) ) / ( 2 * eps )
    fd_w = ( float( f( pts, hw + eps * dw ) ) - float( f( pts, hw - eps * dw ) ) ) / ( 2 * eps )
    an_p, an_w = float( ( gp * dp ).sum() ), float( ( gw * dw ).sum() )
    assert abs( fd_p - an_p ) < 1e-6 * abs( an_p ), ( fd_p, an_p )
    assert abs( fd_w - an_w ) < 1e-6 * abs( an_w ), ( fd_w, an_w )


if test( "a batch of views is the views one by one" ):
    pts, hw, normals, values, s_min, ds = _problem( nb_angles = 5, seed = 3 )
    cost = np.asarray( _plan( pts, hw, normals, values, s_min, ds ).cost )
    for a in range( len( normals ) ):
        one = np.asarray( _plan( pts, hw, normals[ a:a + 1 ], values[ a:a + 1 ], s_min, ds ).cost )[ 0 ]
        assert abs( one - cost[ a ] ) < 1e-12 * cost[ a ], ( a, one, cost[ a ] )


if test( "large views ( on a card, the device-wide sort ) against the numpy reference" ):
    pts, hw, normals, values, s_min, ds = _problem( nb_angles = 3, n = 25000, B = 256, seed = 4 )
    hw = hw / 10
    cost = np.asarray( _plan( pts, hw, normals, values, s_min, ds ).cost )
    for a, nrm in enumerate( normals ):
        c_ref, _ = reference( pts @ nrm, hw, np.ones( len( hw ) ), values[ a ], s_min, ds )
        assert abs( cost[ a ] - c_ref ) < 1e-9 * c_ref, ( a, cost[ a ], c_ref )


if test( "with_gradient: the cost and the gradient of its sum in one pass, as the adjoint's" ):
    import jax
    jax.config.update( "jax_enable_x64", True )
    import jax.numpy as jnp
    pts, hw, normals, values, s_min, ds = _problem( nb_angles = 5, seed = 5 )
    f = lambda p, w: _plan( p, w, normals, values, s_min, ds ).cost.sum()
    c, ( gp, gw ) = jax.value_and_grad( f, argnums = ( 0, 1 ) )( jnp.asarray( pts ), jnp.asarray( hw ) )
    plan = _plan( pts, hw, normals, values, s_min, ds, with_gradient = True )
    assert abs( float( np.asarray( plan.cost ).sum() ) - float( c ) ) < 1e-12 * float( c )
    assert np.abs( np.asarray( plan.points_grad ) - np.asarray( gp ) ).max() < 1e-12 * np.abs( np.asarray( gp ) ).max()
    assert np.abs( np.asarray( plan.half_widths_grad ) - np.asarray( gw ) ).max() < 1e-12 * np.abs( np.asarray( gw ) ).max()


if test( "grid method: the cost and the barycenters converge to the exact ones as the grid refines" ):
    pts, hw, normals, values, s_min, ds = _problem( nb_angles = 4, n = 200, B = 48, seed = 6 )
    exact = _plan( pts, hw, normals, values, s_min, ds, with_barycenters = True )
    c_ex, b_ex = np.asarray( exact.cost ), np.asarray( exact.barycenters )
    errs = [ ]
    for scale in ( 4, 16, 64 ):
        grid = _plan( pts, hw, normals, values, s_min, ds, with_barycenters = True, method = "grid", scale = scale )
        errs.append( ( np.abs( np.asarray( grid.cost ) - c_ex ).max() / c_ex.max(), np.abs( np.asarray( grid.barycenters ) - b_ex ).max() ) )
    assert errs[ 2 ][ 0 ] < 1e-3 and errs[ 2 ][ 1 ] < 1e-3, errs
    assert errs[ 2 ][ 0 ] < errs[ 0 ][ 0 ] and errs[ 2 ][ 1 ] < errs[ 0 ][ 1 ], errs


if test( "grid method: the gradients by finite differences, and in one pass as the adjoint's" ):
    import jax
    jax.config.update( "jax_enable_x64", True )
    import jax.numpy as jnp
    pts, hw, normals, values, s_min, ds = _problem( nb_angles = 4, n = 60, seed = 7 )
    f = lambda p, w: _plan( p, w, normals, values, s_min, ds, method = "grid", scale = 8 ).cost.sum()
    c, ( gp, gw ) = jax.value_and_grad( f, argnums = ( 0, 1 ) )( jnp.asarray( pts ), jnp.asarray( hw ) )
    gp, gw = np.asarray( gp ), np.asarray( gw )
    rng = np.random.default_rng( 8 )
    dp, dw = rng.normal( size = pts.shape ), rng.normal( size = hw.shape )
    eps = 1e-7
    fd_p = ( float( f( pts + eps * dp, hw ) ) - float( f( pts - eps * dp, hw ) ) ) / ( 2 * eps )
    fd_w = ( float( f( pts, hw + eps * dw ) ) - float( f( pts, hw - eps * dw ) ) ) / ( 2 * eps )
    an_p, an_w = float( ( gp * dp ).sum() ), float( ( gw * dw ).sum() )
    assert abs( fd_p - an_p ) < 1e-5 * abs( an_p ), ( fd_p, an_p )
    assert abs( fd_w - an_w ) < 1e-5 * abs( an_w ), ( fd_w, an_w )
    plan = _plan( pts, hw, normals, values, s_min, ds, method = "grid", scale = 8, with_gradient = True )
    assert abs( float( np.asarray( plan.cost ).sum() ) - float( c ) ) < 1e-12 * float( c )
    assert np.abs( np.asarray( plan.points_grad ) - gp ).max() < 1e-12 * np.abs( gp ).max()
    assert np.abs( np.asarray( plan.half_widths_grad ) - gw ).max() < 1e-12 * np.abs( gw ).max()


if test( "grid method: tents across the grid's ends, the gradient by finite differences" ):
    import jax
    jax.config.update( "jax_enable_x64", True )
    import jax.numpy as jnp
    rng = np.random.default_rng( 9 )
    A, B, n = 12, 64, 300
    pts = rng.uniform( -0.8, 0.8, ( n, 2 ) )        # projections up to 1.13: beyond the detector [ -1, 1 ]
    hw = np.full( n, 1.0 / B )
    ang = np.pi * np.arange( A ) / A
    normals = np.stack( [ np.cos( ang ), np.sin( ang ) ], 1 )
    values = rng.random( ( A, B ) ) + 0.05
    for scale in ( 8, 32 ):
        f = lambda p: _plan( p, hw, normals, values, -1.0, 2.0 / B, method = "grid", scale = scale ).cost.sum()
        g = np.asarray( jax.grad( f )( jnp.asarray( pts ) ) )
        d, eps = rng.normal( size = pts.shape ), 1e-7
        fd = ( float( f( pts + eps * d ) ) - float( f( pts - eps * d ) ) ) / ( 2 * eps )
        an = float( ( g * d ).sum() )
        assert abs( fd - an ) < 1e-5 * abs( an ), ( scale, fd, an )


if test( "grid method, float32 points: the cost and the gradients close to float64's ( narrow, wide and mixed tents )" ):
    rng = np.random.default_rng( 10 )
    A, B, n = 16, 128, 20000
    pts = rng.uniform( -0.7, 0.7, ( n, 2 ) )
    ang = np.pi * np.arange( A ) / A
    normals = np.stack( [ np.cos( ang ), np.sin( ang ) ], 1 )
    values = rng.random( ( A, B ) ) + 0.05
    # half-widths in fine cells ( scale 8 ): 4 ( deposited directly ), 8 ( by kinks ), and both
    for name, hw in ( ( "narrow", np.full( n, 0.5 / B ) ), ( "wide", np.full( n, 1.0 / B ) ), ( "mixed", rng.uniform( 0.2, 1.2, n ) / B ) ):
        res = { }
        for dt in ( np.float64, np.float32 ):
            plan = _plan( pts.astype( dt ), hw.astype( dt ), normals.astype( dt ), values.astype( dt ), -1.0, 2.0 / B, method = "grid", with_gradient = True )
            res[ dt ] = ( np.asarray( plan.cost, np.float64 ), np.asarray( plan.points_grad, np.float64 ), np.asarray( plan.half_widths_grad, np.float64 ) )
        c64, g64, w64 = res[ np.float64 ]
        c32, g32, w32 = res[ np.float32 ]
        errs = ( np.abs( c32 - c64 ).max() / c64.max(), np.abs( g32 - g64 ).max() / np.abs( g64 ).max(), np.abs( w32 - w64 ).max() / np.abs( w64 ).max() )
        print( f"float32 vs float64, { name } ( cost, points grad, half-widths grad ):", errs, flush = True )
        assert errs[ 0 ] < 1e-5 and errs[ 1 ] < 1e-4 and errs[ 2 ] < 1e-3, ( name, errs )


if test( "3D points, a normal per ( view, direction ): the 1D reference" ):
    from loom.tensor import Axis, ShapeVar
    from sdot import ContinuousOtPlan1d, Image, ProjectedSumOfTents
    rng = np.random.default_rng( 11 )
    n, V, T, B = 40, 3, 4, 48
    pts = rng.uniform( -0.5, 0.5, ( n, 3 ) )
    hw = rng.uniform( 0.02, 0.06, n )
    # the views' detector bases [ V, 2, 3 ] ( orthonormal rows ), the directions in each detector plane
    q, _ = np.linalg.qr( rng.normal( size = ( V, 3, 3 ) ) )
    bases = q[ :, :2, : ]
    ang = np.pi * ( np.arange( T )[ None ] + rng.random( ( V, 1 ) ) ) / T
    theta = np.stack( [ np.cos( ang ), np.sin( ang ) ], -1 )                              # [ V, T, 2 ]
    normals = np.einsum( "vtk,vkd->vtd", theta, bases ).reshape( V * T, 3 )
    values = rng.random( ( V * T, B ) ) + 0.05
    num = Axis( ShapeVar( V * T ), name = "num_view_direction" )
    src = ProjectedSumOfTents( points = pts, half_widths = hw, normal = normals, proj_dims = 3, batch_axes = [ num ] )
    dst = Image( values = values, origin = [ -1.0 ], pixel_frame = [ [ 2.0 / B ] ], batch_axes = [ num ] )
    cost = np.asarray( ContinuousOtPlan1d( src, dst ).cost )
    for q_ in range( V * T ):
        c_ref, _ = reference( pts @ normals[ q_ ], hw, np.ones( n ), values[ q_ ], -1.0, 2.0 / B )
        assert abs( cost[ q_ ] - c_ref ) < 1e-10 * c_ref, ( q_, cost[ q_ ], c_ref )
