"""The EXACT 1D profiles of images along directions: what a radiograph looks like once projected once more on a line
of its plane ( the target of a sliced transport, `ProjectedSumOfTents` with 3D points ).

An image is constant on its pixels, parallelograms of edges `pixel_frame[ 0 ]` and `pixel_frame[ 1 ]` starting at
`origin`. Projected on a unit direction theta, a pixel spreads its mass over a TRAPEZOID ( the convolution of two
segments of lengths a = |e_0 . theta| and b = |e_1 . theta| ): a density linear between four kinks, with slope jumps
+k, -k, -k, +k, k = mass / ( a b ). The mass of each bin then comes, as for the tents of the grid method
( `ContinuousOtPlan1dGrid.h` ), from three sums per bin and two cumulative sums -- no loop over the bins a pixel covers:
a kink at offset delta inside bin j adds c ( h - delta )^2 / 2 to the mass of bin j, c ( h - delta ) to the density at its
right edge and c to the slope after it ( a jump of the density itself, for a pixel seen edge-on, adds d ( h - delta ) and d ).

The bins should cover the projections of the pixels: the mass that falls outside stays in the two end bins ( as the
grid method does with its border cells ).

    profiles = image_profiles( values, origin, pixel_frame, directions, s_min, ds, nb_bins )

`values` `[ V, N0, N1 ]` ( V images ), `directions` `[ V, T, 2 ]` ( T unit directions per image ), `s_min` `[ V, T ]`
( or a scalar ): the start of the bins of each profile, the position along theta being `x . theta`; the result is
`[ V, T, nb_bins ]`, the masses of the bins ( an image's total mass in each profile ). jax: CPU or card, jit-able.
"""
import jax
import jax.numpy as jnp


#: below this ratio of the two projected lengths of a pixel, it is taken as seen edge-on ( a box ): the kinks' slope
#: jumps, mass / ( a b ), would grow without bound
_EDGE_ON = 1e-5


def _one_profile( values, origin, frame, theta, s_min, ds, nb_bins ):
    """the profile of one image along one direction ( see the module )"""
    N0, N1 = values.shape
    dt = values.dtype
    e0, e1 = frame[ 0 ].astype( dt ), frame[ 1 ].astype( dt )
    a, b = jnp.abs( e0 @ theta ), jnp.abs( e1 @ theta )
    lo_ab, hi_ab = jnp.minimum( a, b ), jnp.maximum( a, b )
    area = jnp.abs( e0[ 0 ] * e1[ 1 ] - e0[ 1 ] * e1[ 0 ] )
    i = jnp.arange( N0, dtype = dt )[ :, None ]
    j = jnp.arange( N1, dtype = dt )[ None, : ]
    centers = ( origin.astype( dt ) @ theta ) + ( i + 0.5 ) * ( e0 @ theta ) + ( j + 0.5 ) * ( e1 @ theta )     # [ N0, N1 ]
    m = ( values * area ).ravel()
    lo = ( centers - 0.5 * ( a + b ) ).ravel()
    edge_on = lo_ab < _EDGE_ON * hi_ab
    # the kinks of every pixel: positions, slope jumps ( trapezoid ) or density jumps ( a pixel seen edge-on )
    k = m / jnp.where( edge_on, 1.0, a * b )
    d = m / hi_ab
    xs = jnp.stack( [ lo, lo + lo_ab, lo + hi_ab, lo + a + b ] )                       # [ 4, P ]
    slope = jnp.where( edge_on, 0.0, jnp.stack( [ k, -k, -k, k ] ) )
    jump = jnp.where( edge_on, jnp.stack( [ d, jnp.zeros_like( d ), jnp.zeros_like( d ), -d ] ), 0.0 )
    xs, slope, jump = xs.ravel(), slope.ravel(), jump.ravel()
    u = ( xs - s_min ) / ds
    cell = jnp.floor( u ).astype( jnp.int32 )
    inside = cell < nb_bins - 1                     # kinks in the last bin or beyond: not needed ( complement )
    left = cell < 0
    delta = jnp.where( left, xs - s_min, xs - ( s_min + cell * ds ) )
    cell = jnp.clip( cell, 0, nb_bins - 1 )
    r = ds - delta
    w = jnp.where( inside, 1.0, 0.0 )
    A = jnp.zeros( nb_bins, dt ).at[ cell ].add( w * slope )
    B = jnp.zeros( nb_bins, dt ).at[ cell ].add( w * ( slope * r + jump ) )
    Q = jnp.zeros( nb_bins, dt ).at[ cell ].add( w * ( 0.5 * slope * r * r + jump * r ) )
    # slope and density at the left edge of each bin, then its mass; the last bin takes the rest
    sl = jnp.concatenate( [ jnp.zeros( 1, dt ), jnp.cumsum( A )[ :-1 ] ] )
    den = jnp.concatenate( [ jnp.zeros( 1, dt ), jnp.cumsum( sl * ds + B )[ :-1 ] ] )
    mass = den * ds + 0.5 * sl * ds * ds + Q
    mass = jnp.maximum( mass, 0 )
    head = mass[ :-1 ]
    return jnp.concatenate( [ head, jnp.maximum( m.sum() - head.sum(), 0 )[ None ] ] )


def image_profiles( values, origin, pixel_frame, directions, s_min, ds, nb_bins ):
    """see the module: `[ V, T, nb_bins ]`"""
    values = jnp.asarray( values )
    dt = values.dtype
    origin = jnp.asarray( origin, dt )
    frame = jnp.asarray( pixel_frame, dt )
    directions = jnp.asarray( directions, dt )
    V, T = directions.shape[ :2 ]
    s_min = jnp.broadcast_to( jnp.asarray( s_min, dt ), ( V, T ) )
    ds = float( ds )
    nb_bins = int( nb_bins )

    def per_image( args ):
        vals, dirs, smin = args
        return jax.vmap( lambda th, s0: _one_profile( vals, origin, frame, th, s0, ds, nb_bins ) )( dirs, smin )

    # one image at a time ( its T directions together ): never every kink of every image at once
    return jax.lax.map( per_image, ( values, directions, s_min ) )
