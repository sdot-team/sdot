from errand import test

# `ContinuousOtPlan1d` between two images ( `ImageOt1d.h` ): pixels of degree 0 or 1 on both sides, the cost only


def _cost( a, b ):
    import numpy
    from sdot.ContinuousOtPlan1d import ContinuousOtPlan1d
    return numpy.asarray( ContinuousOtPlan1d( a, b ).cost )


if test( "a shifted uniform density costs the square of the shift" ):
    from sdot.distributions.Image import Image
    c = float( _cost( Image( values = [ 1.0, 1.0 ], pixel_frame = [ [ 0.5 ] ] ), Image( values = [ 1.0 ], origin = [ 0.3 ] ) ) )
    assert abs( c - 0.09 ) < 1e-14, c
    c = float( _cost( Image( values = [ [ 1.0, 1.0 ] ], degree = 1 ), Image( values = [ [ 2.0, 2.0 ] ], degree = 1, origin = [ -0.2 ] ) ) )
    assert abs( c - 0.04 ) < 1e-14, c

if test( "affine pixels are the limit of constant ones" ):
    # both images of degree 1, against the same densities cut into R constant pixels per pixel: O( 1 / R^2 )
    import numpy
    from sdot.distributions.Image import Image
    B = 16
    na = 1 + 0.8 * numpy.sin( 3 * numpy.linspace( -1, 1, B + 1 ) )
    nb = 1 + 0.6 * numpy.cos( 2 * numpy.linspace( -1, 1, B + 1 ) )
    ca, cb = numpy.stack( [ na[ :-1 ], na[ 1: ] ], 1 ), numpy.stack( [ nb[ :-1 ], nb[ 1: ] ], 1 )
    p1 = float( _cost( Image( values = ca, degree = 1, origin = [ -1.0 ], pixel_frame = [ [ 2 / B ] ] ),
                       Image( values = cb, degree = 1, origin = [ -0.7 ], pixel_frame = [ [ 1.6 / B ] ] ) ) )
    errs = [ ]
    for R in ( 16, 128, 1024 ):
        t = ( numpy.arange( R ) + 0.5 ) / R
        va, vb = ( ca[ :, :1 ] * ( 1 - t ) + ca[ :, 1: ] * t ).ravel(), ( cb[ :, :1 ] * ( 1 - t ) + cb[ :, 1: ] * t ).ravel()
        p0 = float( _cost( Image( values = va, origin = [ -1.0 ], pixel_frame = [ [ 2 / B / R ] ] ),
                           Image( values = vb, origin = [ -0.7 ], pixel_frame = [ [ 1.6 / B / R ] ] ) ) )
        errs.append( abs( p0 - p1 ) / p1 )
    print( "  relative differences at R = 16, 128, 1024:", errs )
    assert errs[ -1 ] < 1e-7 and errs[ 1 ] < errs[ 0 ] / 30 and errs[ 2 ] < errs[ 1 ] / 30, errs

if test( "a batch of pairs is the pairs one by one" ):
    import numpy
    from loom.tensor import Axis, ShapeVar
    from sdot.distributions.Image import Image
    rng = numpy.random.default_rng( 0 )
    V, B = 5, 32
    ca, cb = 0.2 + rng.random( ( V, B, 2 ) ), 0.2 + rng.random( ( V, B, 2 ) )
    ax = Axis( ShapeVar( V ), name = "num_angle" )
    batched = _cost( Image( values = ca, degree = 1, batch_axes = [ ax ] ), Image( values = cb, degree = 1, origin = [ 0.1 ], batch_axes = [ ax ] ) ).ravel()
    one = [ float( _cost( Image( values = ca[ v ], degree = 1 ), Image( values = cb[ v ], degree = 1, origin = [ 0.1 ] ) ) ) for v in range( V ) ]
    assert numpy.allclose( batched, one, rtol = 1e-13 ), ( batched, one )
