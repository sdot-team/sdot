"""`sdot.projected1d`: the sort-free transport of projected objects on a fine detector grid ( the pure
jax reference ). Float64 everywhere here: these are checks of the formulas, not of the float32 path.
"""
from errand import test


def _setup( n = 40, A = 6, B = 64, seed = 0 ):
    import jax
    jax.config.update( "jax_enable_x64", True )
    import numpy as np
    rng = np.random.default_rng( seed )
    angles = np.pi * np.arange( A ) / A
    s_min, extent = -2.0, 4.0
    dw = extent / B
    edges = s_min + dw * np.arange( B + 1 )
    # the target: a disk of radius 0.8 off center ( its exact chord, integrated per bin ) + a small background
    c, r = np.array( [ 0.3, -0.2 ] ), 0.8
    s0 = np.cos( angles ) * c[ 0 ] + np.sin( angles ) * c[ 1 ]
    t = np.clip( edges[ None, : ] - s0[ :, None ], -r, r )
    Gc = t * np.sqrt( r * r - t * t ) + r * r * np.arcsin( t / r )
    values = ( Gc[ :, 1: ] - Gc[ :, :-1 ] ) / dw + 1e-3
    pts = rng.uniform( -0.7, 0.7, ( n, 2 ) )
    return angles, values, s_min, dw, pts


if test( "profiles have mass one and G is their primitive" ):
    import jax
    jax.config.update( "jax_enable_x64", True )
    import jax.numpy as jnp
    import numpy as np
    from sdot.projected1d import profile
    t = jnp.linspace( -1.2, 1.2, 20001 )
    for shape, k in [ ( "tent", 0 ), ( "dome", 1 ), ( "dome", 2 ), ( "dome", 3 ) ]:
        f, G = profile( shape, k )
        assert abs( float( G( -1.0 ) ) ) < 1e-14 and abs( float( G( 1.0 ) ) - 1 ) < 1e-14, ( shape, k )
        dG = jax.vmap( jax.grad( G ) )( t[ 1:-1 ] )
        inside = jnp.abs( t[ 1:-1 ] ) < 0.999
        assert np.allclose( np.asarray( dG )[ inside ], np.asarray( f( t[ 1:-1 ] ) )[ inside ], atol = 1e-12 ), ( shape, k )


if test( "deposit is exact and keeps the mass" ):
    import numpy as np
    from scipy.integrate import quad
    from sdot.projected1d import ProjectedLoss1d, profile
    angles, values, s_min, dw, pts = _setup( n = 7 )
    for shape in [ "dome", "tent" ]:
        loss = ProjectedLoss1d( angles, values, s_min, dw, scale = 4, shape = shape, max_radius = 0.05,
                                dtype = np.float64 )
        radii = np.linspace( 0.005, 0.05, len( pts ) )
        mu = np.asarray( loss.deposit( pts, radii ) )
        assert np.allclose( mu.sum( axis = 1 ), 1, atol = 1e-13 )
        # one angle, one cell, by quadrature
        f, _ = profile( shape, 1 )
        a = 2
        n = np.array( [ np.cos( angles[ a ] ), np.sin( angles[ a ] ) ] )
        w = radii if shape == "dome" else np.full( len( pts ), loss.h )
        for i in range( len( pts ) ):
            s = pts[ i ] @ n
            j = int( ( s - s_min ) // loss.h )
            lo = s_min + j * loss.h
            ref = sum( quad( lambda x: float( f( ( x - pts[ q ] @ n ) / w[ q ] ) ) / w[ q ], lo, lo + loss.h,
                             points = [ pts[ q ] @ n ] )[ 0 ] for q in range( len( pts ) ) ) / len( pts )
            assert abs( mu[ a, j ] - ref ) < 1e-10, ( shape, i, mu[ a, j ], ref )


if test( "grid transport: zero against itself, delta^2 for a shift" ):
    import jax
    jax.config.update( "jax_enable_x64", True )
    import jax.numpy as jnp
    import numpy as np
    from sdot.projected1d import grid_w2
    rng = np.random.default_rng( 1 )
    M, h = 200, 0.01
    nu = np.zeros( M )
    nu[ 50:120 ] = rng.random( 70 )
    nu[ 60:64 ] = 0                      # empty cells inside
    nu /= nu.sum()
    assert abs( float( grid_w2( jnp.asarray( nu ), jnp.asarray( nu ), h ) ) ) < 1e-15
    for shift in [ 1, 7, 30 ]:
        mu = np.roll( nu, shift )
        assert abs( float( grid_w2( jnp.asarray( mu ), jnp.asarray( nu ), h ) ) - ( shift * h ) ** 2 ) < 1e-13, shift
    # two uniform densities: [ 0, 1 ] against [ 0, 2 ] ( cells of 1/100 ): int_0^1 ( t - 2t )^2 dt = 1/3
    mu = np.zeros( M ); mu[ :100 ] = 1 / 100
    nu = np.full( M, 1 / M )
    assert abs( float( grid_w2( jnp.asarray( mu ), jnp.asarray( nu ), 0.01 ) ) - 1 / 3 ) < 1e-13


if test( "gradient matches finite differences" ):
    import jax
    import numpy as np
    from sdot.projected1d import ProjectedLoss1d
    angles, values, s_min, dw, pts = _setup( n = 30 )
    rng = np.random.default_rng( 3 )
    for shape, k in [ ( "dome", 1 ), ( "dome", 2 ), ( "tent", 1 ) ]:
        loss = ProjectedLoss1d( angles, values, s_min, dw, scale = 4, shape = shape, k = k, max_radius = 0.2,
                                dtype = np.float64 )
        radii = rng.uniform( 0.05, 0.15, len( pts ) )
        val, ( gp, gr ) = jax.value_and_grad( loss, argnums = ( 0, 1 ) )( pts, radii )
        assert abs( float( val ) - float( loss( pts, radii ) ) ) < 1e-15
        d = rng.standard_normal( pts.shape )
        dr = rng.standard_normal( radii.shape ) * ( shape == "dome" )
        eps = 1e-6
        fd = ( float( loss( pts + eps * d, radii + eps * dr ) ) - float( loss( pts - eps * d, radii - eps * dr ) ) ) / ( 2 * eps )
        an = float( np.sum( np.asarray( gp ) * d ) + np.sum( np.asarray( gr ) * dr ) )
        assert abs( fd - an ) < 1e-6 * max( 1, abs( fd ) ), ( shape, k, fd, an )


if test( "grid loss converges to the exact one as the grid refines" ):
    import numpy as np
    from sdot.projected1d import ProjectedLoss1d, exact_w2
    angles, values, s_min, dw, pts = _setup( n = 25, A = 3, B = 32 )
    radii = np.full( len( pts ), 0.12 )
    for k in [ 1, 2 ]:
        exact = sum( exact_w2( "dome", k, pts @ np.array( [ np.cos( a ), np.sin( a ) ] ), radii, values[ q ], s_min, dw )
                     for q, a in enumerate( angles ) )
        errs = [ ]
        for scale in [ 2, 8, 32 ]:
            loss = ProjectedLoss1d( angles, values, s_min, dw, scale = scale, shape = "dome", k = k,
                                    max_radius = 0.12, dtype = np.float64 )
            errs.append( abs( float( loss( pts, radii ) ) - exact ) / exact )
        print( f"  dome k={ k }: exact { exact:.6e}, relative errors at scale 2, 8, 32: { errs }" )
        assert errs[ -1 ] < 1e-3 and errs[ 2 ] < errs[ 0 ], errs


if test( "tents approach the semi-discrete transport of diracs" ):
    import jax.numpy as jnp
    import numpy as np
    from sdot.projected1d import ProjectedLoss1d
    from sdot.distributions._pure_jax_cost1d import cost_1d_ot
    angles, values, s_min, dw, pts = _setup( n = 200, A = 4, B = 64 )
    sd = sum( float( cost_1d_ot( jnp.asarray( pts @ np.array( [ np.cos( a ), np.sin( a ) ] ) ), jnp.ones( len( pts ) ),
                                 jnp.asarray( values[ q ] ), s_min, dw ) ) for q, a in enumerate( angles ) )
    prev = None
    for scale in [ 4, 16, 64 ]:
        g = float( ProjectedLoss1d( angles, values, s_min, dw, scale = scale, shape = "tent", dtype = np.float64 )( pts ) )
        err = abs( g - sd ) / sd
        print( f"  scale { scale }: grid { g:.6e} semi-discrete { sd:.6e} ( { err:.2e} )" )
        assert prev is None or err < prev
        prev = err
    assert prev < 2e-2


if test( "the fused kernels match the pure jax passes" ):
    # `backend = "kernel"` ( `sdot/Projected1d.h`, on loom's device ) against `backend = "jax"`, float64: the same
    # formulas in the same order, so equal to rounding
    import jax
    import jax.numpy as jnp
    import numpy as np
    from sdot.projected1d import ProjectedLoss1d
    angles, values, s_min, dw, pts = _setup( n = 500, A = 12, B = 64 )
    rng = np.random.default_rng( 5 )
    for shape, k in [ ( "tent", 1 ), ( "dome", 1 ), ( "dome", 2 ) ]:
        radii = jnp.asarray( rng.uniform( 0.01, 0.08, len( pts ) ) )
        res = { }
        for be in [ "jax", "kernel" ]:
            L = ProjectedLoss1d( angles, values, s_min, dw, scale = 4, shape = shape, k = k, max_radius = 0.06,
                                 dtype = np.float64, backend = be )
            mu = L.deposit( jnp.asarray( pts ), radii )
            # random edge differences ( what `gather` takes ), the end edges at 0 as `grid` writes them
            dg = np.pad( rng.standard_normal( ( mu.shape[ 0 ], mu.shape[ 1 ] - 1 ) ), ( ( 0, 0 ), ( 1, 1 ) ) )
            g = jnp.asarray( dg ) if be == "jax" else res[ "jax" ][ "g" ]
            v, ( gp, gr ) = jax.jit( jax.value_and_grad( lambda p, r: L( p, r ), argnums = ( 0, 1 ) ) )( jnp.asarray( pts ), radii )
            res[ be ] = dict( mu = mu, g = g, gather = L.gather( jnp.asarray( pts ), radii, g ), v = v, gp = gp, gr = gr )
        a, b = res[ "jax" ], res[ "kernel" ]
        rel = lambda x, y: float( np.abs( np.asarray( x ) - np.asarray( y ) ).max() / max( np.abs( np.asarray( x ) ).max(), 1e-300 ) )
        errs = [ rel( a[ "mu" ], b[ "mu" ] ), rel( a[ "gather" ][ 0 ], b[ "gather" ][ 0 ] ), rel( a[ "gather" ][ 1 ], b[ "gather" ][ 1 ] ) if shape == "dome" else 0,
                 rel( a[ "v" ], b[ "v" ] ), rel( a[ "gp" ], b[ "gp" ] ) ]
        assert max( errs ) < 1e-11, ( shape, k, errs )


if test( "a batch of problems is the sum of the problems" ):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from sdot.projected1d import ProjectedLoss1d
    angles, values, s_min, dw, pts = _setup( n = 300, A = 8, B = 64 )
    rng = np.random.default_rng( 2 )
    K = 3
    vals = np.stack( [ values * rng.uniform( 0.5, 2, ( 1, values.shape[ 1 ] ) ) for _ in range( K ) ] )
    P = np.stack( [ pts + 0.05 * rng.standard_normal( pts.shape ) for _ in range( K ) ] )
    R = rng.uniform( 0.01, 0.05, P.shape[ :2 ] )
    for be in [ "jax", "kernel" ]:
        Lb = ProjectedLoss1d( angles, vals, s_min, dw, scale = 4, max_radius = 0.05, dtype = np.float64, backend = be )
        vb, ( gb, rb ) = jax.value_and_grad( Lb, argnums = ( 0, 1 ) )( jnp.asarray( P ), jnp.asarray( R ) )
        for q in range( K ):
            L1 = ProjectedLoss1d( angles, vals[ q ], s_min, dw, scale = 4, max_radius = 0.05, dtype = np.float64, backend = be )
            v1, ( g1, r1 ) = jax.value_and_grad( L1, argnums = ( 0, 1 ) )( jnp.asarray( P[ q ] ), jnp.asarray( R[ q ] ) )
            assert np.allclose( gb[ q ], g1, rtol = 1e-10, atol = 1e-14 ) and np.allclose( rb[ q ], r1, rtol = 1e-10, atol = 1e-14 ), ( be, q )
        tot = sum( float( ProjectedLoss1d( angles, vals[ q ], s_min, dw, scale = 4, max_radius = 0.05, dtype = np.float64,
                                           backend = be )( P[ q ], R[ q ] ) ) for q in range( K ) )
        assert abs( float( vb ) - tot ) < 1e-12 * tot, ( be, float( vb ), tot )
