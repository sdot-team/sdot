"""The envelope of a codimension >= 1 mesh ( `src/sdot/envelope.py`, `src/sdot/include/sdot/Envelope.h` ): the generalized Voronoi
diagram of the elements of M, as simplices with an affine `d` ( distance to M ) and `rho`.

The witnesses: the simplices tile the box ( no hole, no overlap ); `d` is between the exact distance and the exact distance over
`cos( alpha / 2 )` ( the slabs are exact, the fans P1 ); `d` has a slope of 1 on the simplices with a face on M ( the limit
`s -> 0` is then exact ); `rho` on M is the P1 of M."""

from errand import test


def _seg_dist( x, a, b ):
    """the distance of the points `x [ n, D ]` to the segments `[ a, b ]` ( `[ m, D ]` ), and the parameter of the closest point:
    `[ n, m ]` each"""
    import numpy
    ab = b - a
    t = numpy.clip( ( ( x[ :, None ] - a ) * ab ).sum( 2 ) / ( ab ** 2 ).sum( 1 ), 0, 1 )
    return numpy.linalg.norm( x[ :, None ] - ( a + t[ ..., None ] * ab ), axis = 2 ), t


def _distance( x, nodes, simplices ):
    """the exact distance of the points `x` to M ( segments, or triangles in 3D )"""
    import numpy
    if simplices.shape[ 1 ] == 2:
        return _seg_dist( x, nodes[ simplices[ :, 0 ] ], nodes[ simplices[ :, 1 ] ] )[ 0 ].min( 1 )
    a, b, c = ( nodes[ simplices[ :, i ] ] for i in range( 3 ) )
    n = numpy.cross( b - a, c - a )
    n /= numpy.linalg.norm( n, axis = 1 )[ :, None ]
    h = ( ( x[ :, None ] - a ) * n ).sum( 2 )                    # [ n, m ]
    p = x[ :, None ] - h[ ..., None ] * n                       # ( the projections on the planes )
    inside = numpy.ones( h.shape, dtype = bool )
    for u, v in ( ( a, b ), ( b, c ), ( c, a ) ):
        inside &= ( numpy.cross( v - u, p - u ) * n ).sum( 2 ) >= 0
    d = numpy.where( inside, numpy.abs( h ), numpy.inf )
    for u, v in ( ( a, b ), ( b, c ), ( c, a ) ):
        d = numpy.minimum( d, _seg_dist( x, u, v )[ 0 ] )
    return d.min( 1 )


def _locate( P, x ):
    """for each point, the number of simplices holding it ( barycentrics >= -1e-9 ), the number holding it strictly ( > 1e-9 ), and
    the barycentrics in the last one holding it: `( loose, strict, idx, bary )`"""
    import numpy
    D = P.shape[ 2 ]
    M = numpy.swapaxes( P[ :, 1: ] - P[ :, :1 ], 1, 2 )
    inv = numpy.linalg.inv( M )
    loose = numpy.zeros( len( x ), int ); strict = numpy.zeros( len( x ), int )
    idx = numpy.full( len( x ), -1 ); bary = numpy.zeros( ( len( x ), D + 1 ) )
    lo, hi = P.min( 1 ), P.max( 1 )
    for t in range( len( P ) ):
        c = numpy.nonzero( ( x >= lo[ t ] - 1e-9 ).all( 1 ) & ( x <= hi[ t ] + 1e-9 ).all( 1 ) )[ 0 ]
        l = ( x[ c ] - P[ t, 0 ] ) @ inv[ t ].T
        b = numpy.concatenate( [ 1 - l.sum( 1, keepdims = True ), l ], 1 )
        ok = ( b >= -1e-9 ).all( 1 )
        loose[ c ] += ok
        strict[ c ] += ( b > 1e-9 ).all( 1 )
        idx[ c[ ok ] ] = t
        bary[ c[ ok ] ] = b[ ok ]
    return loose, strict, idx, bary


def _box( nodes, margin = 0.25 ):
    import numpy
    a, b = nodes.min( 0 ), nodes.max( 0 )
    m = margin * numpy.linalg.norm( b - a )
    return a - m, b + m


def _measures( P ):
    import numpy
    import math
    M = P[ :, 1: ] - P[ :, :1 ]
    return numpy.abs( numpy.linalg.det( M ) ) / math.factorial( P.shape[ 2 ] )


def _cases():
    """`name -> ( nodes, simplices )`: lines in 2D, surfaces and lines in 3D, with sharp corners, junctions, crossings, close
    branches"""
    import numpy
    out = {}
    t = numpy.linspace( 0, 2 * numpy.pi, 40, endpoint = False )
    n = 0.5 + numpy.stack( [ 0.3 * numpy.cos( t ), 0.04 * numpy.sin( t ) ], 1 )
    out[ "ellipse_2d" ] = ( n, numpy.stack( [ numpy.arange( 40 ), ( numpy.arange( 40 ) + 1 ) % 40 ], 1 ) )
    out[ "corner_and_cross_2d" ] = ( numpy.array( [ [ 0, 0 ], [ 1, 0.1 ], [ 0, 0.2 ], [ 0.5, -0.3 ], [ 0.6, 0.5 ], [ 0.3, 0.06 ] ], float ),
                                     numpy.array( [ [ 0, 1 ], [ 1, 2 ], [ 3, 4 ], [ 5, 1 ] ] ) )
    out[ "close_2d" ] = ( numpy.array( [ [ 0, 0 ], [ 1, 0 ], [ 0, 1e-3 ], [ 1, 1e-3 ] ], float ), numpy.array( [ [ 0, 1 ], [ 2, 3 ] ] ) )
    out[ "tet_surface_3d" ] = ( numpy.array( [ [ 0, 0, 0 ], [ 1, 0, 0 ], [ 0, 1, 0 ], [ 0, 0, 1 ] ], float ),
                                numpy.array( [ [ 0, 1, 2 ], [ 0, 1, 3 ], [ 0, 2, 3 ], [ 1, 2, 3 ] ] ) )
    out[ "sharp_fold_3d" ] = ( numpy.array( [ [ 0, 0, 0 ], [ 1, 0, 0 ], [ 0.5, 1, 0 ], [ 0.5, 1, 0.15 ], [ 0.5, -0.6, 0.3 ] ], float ),
                               numpy.array( [ [ 0, 1, 2 ], [ 0, 1, 3 ], [ 0, 1, 4 ] ] ) )
    out[ "lines_3d" ] = ( numpy.array( [ [ 0, 0, 0 ], [ 1, 0, 0 ], [ 1, 1, 0.3 ], [ 0.2, 0.1, 0.1 ], [ 1, 0, 1 ] ], float ),
                          numpy.array( [ [ 0, 1 ], [ 1, 2 ], [ 2, 3 ], [ 1, 4 ] ] ) )
    # ( a closed surface around a hole: the candidates of its inner side all overlap, and their fragments are merged back )
    nu, nv = 12, 6
    u, v = numpy.meshgrid( 2 * numpy.pi * numpy.arange( nu ) / nu, 2 * numpy.pi * numpy.arange( nv ) / nv, indexing = "ij" )
    n = numpy.stack( [ ( 1 + 0.35 * numpy.cos( v ) ) * numpy.cos( u ), ( 1 + 0.35 * numpy.cos( v ) ) * numpy.sin( u ), 0.35 * numpy.sin( v ) ], -1 ).reshape( -1, 3 )
    I = lambda i, j: ( i % nu ) * nv + ( j % nv )
    out[ "torus_3d" ] = ( n, numpy.array( [ t for i in range( nu ) for j in range( nv ) for t in ( [ I( i, j ), I( i + 1, j ), I( i + 1, j + 1 ) ], [ I( i, j ), I( i + 1, j + 1 ), I( i, j + 1 ) ] ) ] ) )
    return out


if test( "the_envelope_tiles_the_box" ):
    import numpy
    from sdot.envelope import envelope
    rng = numpy.random.default_rng( 0 )
    for name, ( nodes, sx ) in _cases().items():
        lo, hi = _box( nodes )
        o = envelope( nodes, sx, lo = lo, hi = hi, alpha = 30 )
        assert abs( _measures( o[ "pts" ] ).sum() / numpy.prod( hi - lo ) - 1 ) < 1e-9, ( name, _measures( o[ "pts" ] ).sum() / numpy.prod( hi - lo ) - 1 )
        loose, strict, _, _ = _locate( o[ "pts" ], rng.uniform( lo, hi, ( 20000, len( lo ) ) ) )
        assert ( loose >= 1 ).all() and ( strict <= 1 ).all(), ( name, ( loose == 0 ).sum(), ( strict > 1 ).sum() )


if test( "d_is_the_distance_up_to_the_fans" ):
    import numpy
    from sdot.envelope import envelope
    rng = numpy.random.default_rng( 1 )
    alpha = 30
    for name, ( nodes, sx ) in _cases().items():
        lo, hi = _box( nodes )
        o = envelope( nodes, sx, lo = lo, hi = hi, alpha = alpha )
        x = rng.uniform( lo, hi, ( 20000, len( lo ) ) )
        _, _, idx, b = _locate( o[ "pts" ], x )
        d = ( b * o[ "d" ][ idx ] ).sum( 1 )
        ex = _distance( x, nodes, sx )
        assert ( d >= ex - 1e-10 ).all(), ( name, ( ex - d ).max() )
        assert ( d <= ex / numpy.cos( numpy.radians( alpha ) / 2 ) + 1e-10 ).all(), ( name, ( d / ex ).max() )


if test( "d_has_a_slope_of_one_on_the_faces_of_m" ):
    import numpy
    from sdot.envelope import envelope
    for name, ( nodes, sx ) in _cases().items():
        D = nodes.shape[ 1 ]
        if sx.shape[ 1 ] != D:
            continue                                            # ( lines in 3D: the fans, see the weights )
        o = envelope( nodes, sx, alpha = 30 )
        P, d = o[ "pts" ], o[ "d" ]
        on = ( d == 0 ).sum( 1 ) == D
        g = numpy.linalg.solve( P[ on, 1: ] - P[ on, :1 ], ( d[ on, 1: ] - d[ on, :1 ] )[ ..., None ] )[ ..., 0 ]
        assert on.sum() > 0 and numpy.abs( numpy.linalg.norm( g, axis = 1 ) - 1 ).max() < 1e-9, name


if test( "the_fans_around_lines_in_3d_are_weighted" ):
    # a sector of angle theta around a segment: `int exp( -d^2 / s^2 )` over its section is `s^2 sin( theta ) / 2` with `d` P1, so
    # `w = theta / sin( theta )` brings it back to the exact `s^2 theta / 2`
    import numpy
    from sdot.envelope import envelope
    nodes, sx = _cases()[ "lines_3d" ]
    for alpha in ( 30, 15 ):
        o = envelope( nodes, sx, alpha = alpha )
        th = 2 * numpy.pi / numpy.ceil( 2 * numpy.pi / numpy.radians( alpha ) - 1e-9 )
        w = numpy.unique( numpy.round( o[ "w" ], 12 ) )
        assert numpy.allclose( w, [ 1, th / numpy.sin( th ) ] ), ( alpha, w )


if test( "rho_on_m_is_the_p1_of_m" ):
    import numpy
    from sdot.envelope import envelope
    for name, ( nodes, sx ) in _cases().items():
        rho = 1 + nodes @ numpy.arange( 1, nodes.shape[ 1 ] + 1 )   # ( affine: its P1 on M is itself )
        o = envelope( nodes, sx, rho = rho, alpha = 30 )
        z = o[ "d" ] == 0
        x = o[ "pts" ][ z ]
        assert numpy.abs( o[ "rho" ][ z ] - ( 1 + x @ numpy.arange( 1, nodes.shape[ 1 ] + 1 ) ) ).max() < 1e-9, name
