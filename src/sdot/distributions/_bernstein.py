"""= Internals

The Bernstein coefficients of a polynomial image ( `Image( degree = P )` ), on the host: `[ pixels..., ( P + 1 )^d ]`, flattened in C
order ( the first axis of the image the slowest ), in the coordinates `t` in `[ 0, 1 ]^d` of each pixel ( `PolynomialPixel.h` ).
"""
import math
import numpy


def basis( p, t ):
    """`B_{ i, p }( t )` for `i <= p`: `[ ..., p + 1 ]`"""
    t = numpy.asarray( t, dtype = float )[ ..., None ]
    i = numpy.arange( p + 1 )
    binom = numpy.array( [ math.comb( p, k ) for k in range( p + 1 ) ], dtype = float )
    return binom * t ** i * ( 1 - t ) ** ( p - i )


def as_tensor( coeffs, d ):
    """`[ pixels..., ( p + 1 )^d ]` -> `[ pixels..., p + 1, ..., p + 1 ]` ( and the degree )"""
    c = numpy.asarray( coeffs, dtype = float )
    p = round( c.shape[ -1 ] ** ( 1 / d ) ) - 1
    return c.reshape( c.shape[ :-1 ] + ( p + 1, ) * d ), p


def elevate( coeffs, d, by = 1 ):
    """the same polynomials written with the degree raised by `by` along each axis ( exact ):
    `c'_i = i / ( p + 1 ) c_{ i - 1 } + ( 1 - i / ( p + 1 ) ) c_i`, axis by axis"""
    c, p = as_tensor( coeffs, d )
    for _ in range( by ):
        for a in range( d ):
            ax = c.ndim - d + a
            c = numpy.moveaxis( c, ax, -1 )
            i = numpy.arange( p + 2 ) / ( p + 1 )
            lo = numpy.concatenate( [ numpy.zeros_like( c[ ..., :1 ] ), c ], axis = -1 )
            hi = numpy.concatenate( [ c, numpy.zeros_like( c[ ..., :1 ] ) ], axis = -1 )
            c = numpy.moveaxis( i * lo + ( 1 - i ) * hi, -1, ax )
        p += 1
    return c.reshape( c.shape[ :c.ndim - d ] + ( ( p + 1 ) ** d, ) )


def evaluate( coeffs, d, pixel, t ):
    """the density at local coordinates `t` ( `[ n, d ]` ) of the pixels `pixel` ( `[ n, d ]` integer indices )"""
    c, p = as_tensor( coeffs, d )
    t = numpy.asarray( t, dtype = float )
    res = c[ tuple( numpy.asarray( pixel ).T ) ]                                    # [ n, p + 1, ..., p + 1 ]
    for a in range( d ):
        b = basis( p, t[ :, a ] )                                                   # [ n, p + 1 ]
        res = numpy.einsum( "ni,ni...->n...", b, res )
    return res


def from_function( func, shape, p, lo = None, hi = None ):
    """the coefficients that INTERPOLATE `func` ( `[ n, d ] -> [ n ]`, the coordinates of the grid `[ lo, hi ]`, the unit box
    by default ) at the `( p + 1 )^d` equispaced points of each pixel: exact for a polynomial of degree `<= p` along each axis"""
    d = len( shape )
    lo = numpy.zeros( d ) if lo is None else numpy.asarray( lo, dtype = float )
    hi = numpy.ones( d ) if hi is None else numpy.asarray( hi, dtype = float )
    nodes = numpy.linspace( 0, 1, p + 1 ) if p else numpy.array( [ 0.5 ] )
    B = basis( p, nodes )                                                           # [ node, coefficient ]
    Binv = numpy.linalg.inv( B )
    pix = numpy.stack( numpy.meshgrid( *[ numpy.arange( s ) for s in shape ], indexing = "ij" ), -1 ).reshape( -1, d )
    loc = numpy.stack( numpy.meshgrid( *[ nodes ] * d, indexing = "ij" ), -1 ).reshape( -1, d )
    h = ( hi - lo ) / numpy.asarray( shape )
    x = lo + ( pix[ :, None, : ] + loc[ None, :, : ] ) * h                          # [ pixel, node, d ]
    v = numpy.asarray( func( x.reshape( -1, d ) ), dtype = float ).reshape( ( len( pix ), ) + ( p + 1, ) * d )
    for a in range( d ):
        v = numpy.moveaxis( numpy.tensordot( v, Binv.T, axes = ( [ 1 + a ], [ 0 ] ) ), -1, 1 + a )
    return v.reshape( tuple( shape ) + ( ( p + 1 ) ** d, ) )
