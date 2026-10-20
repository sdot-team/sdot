import numpy

import loom

from .Image import Image, _grid_shape, _pixel_frame
from ._bernstein import basis


def SplineGrid( values, degree = 3, frame = None, pixel_frame = None, origin = None, interpolate = False, target_mass = 1.0 ):
    """A smooth density on a grid of pixels: the tensor-product B-spline of degree `degree` ( `C^{ degree - 1 }` ) whose control
    coefficients are `values`, one per pixel, at its center. Non-negative `values` give a non-negative density. The grid is placed
    like that of an `Image` ( `frame`, `pixel_frame`, `origin` ), and the density is zero outside it.

    `interpolate = True`: `values` are SAMPLES at the centers of the pixels instead, and the coefficients are those of the spline that
    goes through them. Such a spline can dip below zero near a sharp edge: it is then refused ( a `ValueError` ), and a lower degree,
    or `interpolate = False`, keeps it non-negative.

    It is an `Image( degree = degree )`: each span of the spline is a polynomial, written in the Bernstein basis of its pixel.
    Differentiable with respect to `values`.

    = Internals

    Along an axis of `n` pixels ( in pixel units, the box `[ 0, n ]` ), with `N` the centered cardinal B-spline of degree `p`,

        S( x ) = sum_{ i in Z } c_{ clamp( i ) } N( x - i - 1/2 ),

    the coefficients extended beyond the border by their edge value: the basis sums to one on the whole box, so a constant stays a
    constant. Its breakpoints are the borders of the pixels for an even `p`, their centers ( and the two ends of the box ) for an odd
    one -- then the image has `n + 1` pixels, the first and the last of half a width ( `knots` ). Each span is the image of
    `p + 1` coefficients by a fixed linear map ( `_axis_map` ), and the whole is the tensor product of these maps, applied axis by
    axis with `matmul` -- in the framework of `values`, which stays differentiable.
    """
    p = int( degree )
    if p < 0:
        raise ValueError( f"SplineGrid: `degree` has to be >= 0 ( got { p } )" )
    shape = _grid_shape( values, None )
    d = len( shape )
    if d < 1:
        raise ValueError( "SplineGrid: `values` has to be an array with one axis per dimension" )
    if p == 0:                                           # ( a constant per pixel: the image itself )
        place = dict( origin = origin ) if origin is not None else {}
        return Image( values, frame = frame, pixel_frame = pixel_frame, target_mass = target_mass, **place )

    maps, knots = [], []
    for n in shape:
        A, k = _axis_map( n, p )
        if interpolate:
            A = numpy.einsum( "kbj,ji->kbi", A, numpy.linalg.inv( _collocation( n, p ) ) )
        maps.append( A )
        knots.append( k )

    if interpolate:
        try:
            host = numpy.asarray( values.value if isinstance( values, loom.Tensor ) else values, dtype = float )
        except ( TypeError, ValueError ):
            host = None                                  # ( traced: not checked )
        if host is not None:
            c = _apply( host, maps, p, numpy )
            if c.min() < - 1e-12 * max( abs( c ).max(), 1e-300 ):
                raise ValueError( f"SplineGrid: the interpolating spline of degree { p } dips below zero ( min of its coefficients: "
                                  f"{ c.min():.3g} ) -- take `interpolate = False`, or a lower degree" )

    coeffs = _apply( values, maps, p, None )
    pf = _pixel_frame( frame, shape ) if pixel_frame is None else pixel_frame
    kwargs = dict( degree = p, pixel_frame = pf, target_mass = target_mass )
    if origin is not None:
        kwargs[ "origin" ] = origin
    if p % 2:
        width = max( len( k ) for k in knots )
        kwargs[ "knots" ] = numpy.stack( [ numpy.concatenate( [ k, numpy.full( width - len( k ), k[ -1 ] ) ] ) for k in knots ] )
    return Image( coeffs, **kwargs )


def _cardinal( p, x ):
    """the centered cardinal B-spline of degree `p` at `x`, by the recurrence
    `N_k( x ) = ( ( x + ( k + 1 ) / 2 ) N_{ k - 1 }( x + 1/2 ) + ( ( k + 1 ) / 2 - x ) N_{ k - 1 }( x - 1/2 ) ) / k` ( no cancellation )"""
    x = numpy.asarray( x, dtype = float )
    if p == 0:
        return ( ( x >= -0.5 ) & ( x < 0.5 ) ).astype( float )
    h = ( p + 1 ) / 2
    return ( ( x + h ) * _cardinal( p - 1, x + 0.5 ) + ( h - x ) * _cardinal( p - 1, x - 0.5 ) ) / p


def _breakpoints( n, p ):
    if p % 2 == 0:
        return numpy.arange( n + 1, dtype = float )
    return numpy.concatenate( [ [ 0.0 ], numpy.arange( n, dtype = float ) + 0.5, [ float( n ) ] ] )


def _basis_matrix( n, p, x ):
    """`[ len( x ), n ]`: the weight of each coefficient in `S( x )`, the extension beyond the border folded onto the edge"""
    r = ( p + 1 ) // 2 + 1
    M = numpy.zeros( ( len( x ), n ) )
    for i in range( -r, n + r ):
        M[ :, min( max( i, 0 ), n - 1 ) ] += _cardinal( p, x - i - 0.5 )
    return M


def _axis_map( n, p ):
    """`( A, knots )`: `A [ pieces, p + 1, n ]` the Bernstein coefficients of each span from the coefficients of the axis, `knots`
    the breakpoints. A span is sampled at `p + 1` points and the values brought back to the Bernstein basis: exact, it is a
    polynomial of degree `p`"""
    k = _breakpoints( n, p )
    t = numpy.linspace( 0, 1, p + 1 )
    Binv = numpy.linalg.inv( basis( p, t ) )                                     # values at `t` -> Bernstein
    A = numpy.empty( ( len( k ) - 1, p + 1, n ) )
    for s in range( len( k ) - 1 ):
        A[ s ] = Binv @ _basis_matrix( n, p, k[ s ] + t * ( k[ s + 1 ] - k[ s ] ) )
    return A, k


def _collocation( n, p ):
    """`[ n, n ]`: the spline at the centers of the pixels from its coefficients"""
    return _basis_matrix( n, p, numpy.arange( n ) + 0.5 )


def _apply( values, maps, p, xp ):
    """the coefficients of the image, `[ pieces..., ( p + 1 )^d ]`, from `values [ n... ]`: axis by axis, `[ pre, n, post ]` ->
    `[ pre, pieces, p + 1, post ]` by a `matmul`, then the Bernstein axes moved last"""
    d = len( maps )
    raw = values.value if isinstance( values, loom.Tensor ) else values
    if xp is None:
        ops = loom.ops( raw ) if not isinstance( raw, ( list, tuple, numpy.ndarray ) ) else None
        if ops is None:
            raw = numpy.asarray( raw, dtype = float )
    else:
        ops = None
    mm = ( lambda a, b: ops.matmul( a, b ) ) if ops is not None else numpy.matmul
    rs = ( lambda a, s: ops.reshape( a, s ) ) if ops is not None else numpy.reshape
    tr = ( lambda a, axes: ops.transpose( a, axes ) ) if ops is not None else numpy.transpose

    c = raw
    pieces = []
    q = p + 1
    for a, A in enumerate( maps ):
        pre = int( numpy.prod( [ len( m ) for m in maps[ :a ] ] ) ) * q ** a
        n = A.shape[ 2 ]
        post = int( numpy.prod( [ m.shape[ 2 ] for m in maps[ a + 1: ] ] ) )
        c = rs( c, ( pre, n, post ) )
        c = mm( A.reshape( -1, n ), c )                                           # [ pre, pieces * q, post ]
        pieces.append( A.shape[ 0 ] )
    # the order is now [ k_0, b_0, k_1, b_1, ..., k_{ d - 1 }, b_{ d - 1 } ]: the pieces first, then the Bernstein indices
    c = rs( c, tuple( x for a in range( d ) for x in ( pieces[ a ], q ) ) )
    c = tr( c, tuple( 2 * a for a in range( d ) ) + tuple( 2 * a + 1 for a in range( d ) ) )
    return rs( c, tuple( pieces ) + ( q ** d, ) )
