"""`sdot.projected_profiles.image_profiles`: the exact 1D profiles of images along directions, against column sums
( a pixel seen edge-on ), against the closed-form cumulative mass of each pixel's trapezoid, and for the total mass."""
import numpy as np

from errand import test


def _reference( values, origin, frame, theta, s_min, ds, nb_bins ):
    """the bins' masses by the closed-form cdf of each pixel's trapezoid ( an independent formula )"""
    e0, e1 = np.asarray( frame[ 0 ], float ), np.asarray( frame[ 1 ], float )
    a, b = abs( e0 @ theta ), abs( e1 @ theta )
    area = abs( e0[ 0 ] * e1[ 1 ] - e0[ 1 ] * e1[ 0 ] )
    N0, N1 = values.shape
    edges = s_min + ds * np.arange( nb_bins + 1 )
    out = np.zeros( nb_bins )
    R = lambda u: np.maximum( u, 0 ) ** 2 / 2
    for i in range( N0 ):
        for j in range( N1 ):
            c = np.asarray( origin ) @ theta + ( i + 0.5 ) * ( e0 @ theta ) + ( j + 0.5 ) * ( e1 @ theta )
            lo = c - 0.5 * ( a + b )
            x = [ lo, lo + min( a, b ), lo + max( a, b ), lo + a + b ]
            F = ( R( edges - x[ 0 ] ) - R( edges - x[ 1 ] ) - R( edges - x[ 2 ] ) + R( edges - x[ 3 ] ) ) / ( a * b )
            out += values[ i, j ] * area * np.diff( F )
    return out


if test( "a direction along a pixel axis: the column sums" ):
    import jax
    jax.config.update( "jax_enable_x64", True )
    from sdot.projected_profiles import image_profiles
    rng = np.random.default_rng( 0 )
    vals = rng.random( ( 2, 7, 5 ) )
    du, dv = 0.3, 0.2
    dirs = np.array( [ [ [ 1.0, 0.0 ] ], [ [ 0.0, 1.0 ] ] ] )
    s_min = np.array( [ [ -1.0 ], [ 0.5 ] ] )
    # the origin, then bins aligned with the pixels along each direction
    p0 = np.asarray( image_profiles( vals[ :1 ], [ -1.0, 0.5 ], [ [ du, 0 ], [ 0, dv ] ], dirs[ :1 ], s_min[ :1 ], du, 7 ) )[ 0, 0 ]
    p1 = np.asarray( image_profiles( vals[ 1: ], [ -1.0, 0.5 ], [ [ du, 0 ], [ 0, dv ] ], dirs[ 1: ], s_min[ 1: ], dv, 5 ) )[ 0, 0 ]
    assert np.abs( p0 - vals[ 0 ].sum( 1 ) * du * dv ).max() < 1e-13
    assert np.abs( p1 - vals[ 1 ].sum( 0 ) * du * dv ).max() < 1e-13


if test( "random directions: the closed-form trapezoids, and the total mass" ):
    import jax
    jax.config.update( "jax_enable_x64", True )
    from sdot.projected_profiles import image_profiles
    rng = np.random.default_rng( 1 )
    V, T, N0, N1 = 2, 3, 6, 4
    vals = rng.random( ( V, N0, N1 ) )
    origin, frame = [ -0.7, -0.4 ], [ [ 0.25, 0.0 ], [ 0.0, 0.2 ] ]
    ang = rng.uniform( 0, np.pi, ( V, T ) )
    dirs = np.stack( [ np.cos( ang ), np.sin( ang ) ], -1 )
    ds, nb = 0.07, 40
    s_min = np.full( ( V, T ), -1.4 )
    prof = np.asarray( image_profiles( vals, origin, frame, dirs, s_min, ds, nb ) )
    for v in range( V ):
        for t in range( T ):
            ref = _reference( vals[ v ], origin, frame, dirs[ v, t ], s_min[ v, t ], ds, nb )
            assert np.abs( prof[ v, t ] - ref ).max() < 1e-12, ( v, t, np.abs( prof[ v, t ] - ref ).max() )
            assert abs( prof[ v, t ].sum() - vals[ v ].sum() * 0.05 ) < 1e-12
