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
