"""`sdot.pixel_transport`: pyramids rasterized exactly on a fine grid ( the source ) against the pixels of an image as diracs
( the target ), by sdot's semi-discrete solver: the raster's masses, the cost's derivatives by finite differences, the
warm start."""
import numpy as np

from errand import test


if test( "the raster: exact cell masses, the mass kept at the borders" ):
    import jax
    jax.config.update( "jax_enable_x64", True )
    from sdot.pixel_transport import raster_pyramids
    h = 0.1
    # a pyramid of half-width h on a cell corner: a quarter of its mass in each of the 4 cells around
    d = np.asarray( raster_pyramids( np.array( [ [ 0.3, 0.5 ] ] ), 0.1, 1.0, [ 0.0, 0.0 ], ( h, h ), ( 8, 8 ), 0.1 ) ) * h * h
    assert np.abs( d[ 2:4, 4:6 ] - 0.25 ).max() < 1e-14 and abs( d.sum() - 1 ) < 1e-14
    # a pyramid half out of the grid: its mass stays in the border cells
    d = np.asarray( raster_pyramids( np.array( [ [ -0.05, 0.4 ] ] ), 0.17, 2.0, [ 0.0, 0.0 ], ( h, h ), ( 8, 8 ), 0.17 ) ) * h * h
    assert abs( d.sum() - 2 ) < 1e-14 and d[ 1:, : ].sum() < d[ 0, : ].sum()
    # against a quadrature of the product of tents
    rng = np.random.default_rng( 0 )
    uv, w = rng.uniform( 0.2, 0.6, ( 3, 2 ) ), rng.uniform( 0.05, 0.15, 3 )
    d = np.asarray( raster_pyramids( uv, w, 1.0, [ 0.0, 0.0 ], ( h, h ), ( 8, 8 ), 0.15 ) )
    x = ( np.arange( 800 ) + 0.5 ) / 1000.0
    X, Y = np.meshgrid( x, x, indexing = "ij" )
    ref = np.zeros( ( 800, 800 ) )
    for ( u, v ), wi in zip( uv, w ):
        ref += np.maximum( 1 - np.abs( X - u ) / wi, 0 ) * np.maximum( 1 - np.abs( Y - v ) / wi, 0 ) / wi ** 2
    ref = ref.reshape( 8, 100, 8, 100 ).mean( axis = ( 1, 3 ) )
    assert np.abs( d - ref ).max() < 1e-4 * ref.max(), np.abs( d - ref ).max()


if test( "the cost: smaller on the image's own pyramids, derivatives by finite differences, warm start" ):
    import jax
    jax.config.update( "jax_enable_x64", True )
    from sdot.pixel_transport import PixelTransport2d
    rng = np.random.default_rng( 1 )
    N0, N1, du, dv = 12, 10, 0.1, 0.1
    truth = rng.uniform( 0.3, 0.9, ( 30, 2 ) )
    hw = 0.05
    pt0 = PixelTransport2d( np.ones( ( N0, N1 ) ), [ 0.0, 0.0 ], ( du, dv ), scale = 4 )
    img = np.asarray( pt0.raster( truth, hw ) ).reshape( N0, 4, N1, 4 ).mean( axis = ( 1, 3 ) ) + 1e-3
    pt = PixelTransport2d( img, [ 0.0, 0.0 ], ( du, dv ), scale = 4, tol = 1e-13, max_iter = 200 )
    uv = truth + rng.normal( scale = 0.03, size = truth.shape )
    assert pt.cost( truth, hw ) < pt.cost( uv, hw )
    c, ( g_uv, g_hw ) = pt.value_and_grad( uv, np.full( 30, hw ) )
    # the envelope derivative is sdot's own ( a frozen Newton step and its adjoint )
    c2, ( g_uv2, g_hw2 ) = pt.value_and_grad_adjoint( uv, np.full( 30, hw ) )
    assert abs( c - c2 ) < 1e-14 and np.abs( np.asarray( g_uv - g_uv2 ) ).max() < 1e-9 * np.abs( np.asarray( g_uv2 ) ).max()
    assert np.abs( np.asarray( g_hw - g_hw2 ) ).max() < 1e-9 * np.abs( np.asarray( g_hw2 ) ).max()
    d, eps = rng.normal( size = uv.shape ), 1e-5
    fd = ( pt.cost( uv + eps * d, hw ) - pt.cost( uv - eps * d, hw ) ) / ( 2 * eps )
    an = float( ( np.asarray( g_uv ) * d ).sum() )
    assert abs( fd - an ) < 1e-5 * abs( an ), ( fd, an )
    dw = rng.normal( size = 30 )
    fdw = ( pt.cost( uv, hw + eps * dw ) - pt.cost( uv, hw - eps * dw ) ) / ( 2 * eps )
    anw = float( ( np.asarray( g_hw ) * dw ).sum() )
    assert abs( fdw - anw ) < 1e-5 * abs( anw ), ( fdw, anw )
    # the warm start: a nearby problem from the last weights takes fewer Newton steps than from scratch
    pt.weights = None
    pt.cost( uv, hw )
    cold = int( pt.stats[ "nb_iter" ] )
    pt.cost( uv + 1e-3 * d, hw )
    warm = int( pt.stats[ "nb_iter" ] )
    assert warm < cold, ( warm, cold )


if test( "split pixels: several diracs per pixel, the same raster grid, a cost between the pixel one and the continuous one" ):
    import jax
    jax.config.update( "jax_enable_x64", True )
    from sdot.pixel_transport import PixelTransport2d
    rng = np.random.default_rng( 4 )
    img = rng.random( ( 10, 8 ) ) + 0.2
    uv = rng.uniform( 0.1, 0.7, ( 200, 2 ) )
    costs = [ ]
    for split in ( 1, 2, 4 ):
        pt = PixelTransport2d( img, [ 0.0, 0.0 ], ( 0.08, 0.1 ), scale = 4, dirac_split = split, tol = 1e-12, max_iter = 200 )
        assert pt.shape == ( 40, 32 ) and pt.centers.shape[ 0 ] == 80 * split * split
        c, ( g, _ ) = pt.value_and_grad( uv, 0.04 )
        costs.append( c )
    # the finer the target, the smaller its own spread ( the part of the cost the discretization adds )
    assert costs[ 0 ] > costs[ 1 ] > costs[ 2 ], costs
