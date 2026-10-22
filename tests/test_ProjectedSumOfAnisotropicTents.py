"""`ProjectedSumOfAnisotropicTents` ( a matrix per tent: its half-width on a line of direction n is | M^T n | ) against an `Image`
through `ContinuousOtPlan1d`: round tents give the isotropic class's costs, the cost of each item is that of isotropic tents of
the half-widths of that direction, and the gradients with respect to the points and the matrices match finite differences.
"""
import numpy as np

from errand import test


def _problem( nb_angles = 5, n = 30, B = 48, seed = 0 ):
    rng = np.random.default_rng( seed )
    pts = rng.uniform( -0.6, 0.6, ( n, 2 ) )
    mats = rng.uniform( 0.02, 0.08, ( n, 2, 2 ) ) * np.array( [ [ 1.0, 0.3 ], [ 0.3, 1.0 ] ] )   # not symmetric, not diagonal
    angles = np.pi * np.arange( nb_angles ) / nb_angles
    normals = np.stack( [ np.cos( angles ), np.sin( angles ) ], 1 )
    values = rng.random( ( nb_angles, B ) ) + 0.05
    return pts, mats, normals, values, -1.0, 2.0 / B


def _image( values, s_min, ds, axis ):
    from sdot import Image
    return Image( values = values, origin = [ s_min ], pixel_frame = [ [ ds ] ], batch_axes = [ axis ] )


def _aniso( pts, mats, normals, values, s_min, ds, **kw ):
    from loom.tensor import Axis, ShapeVar
    from sdot import ContinuousOtPlan1d, ProjectedSumOfAnisotropicTents
    num_angle = Axis( ShapeVar( len( normals ) ), name = "num_angle" )
    src = ProjectedSumOfAnisotropicTents( points = pts, matrices = mats, normal = normals, batch_axes = [ num_angle ] )
    return ContinuousOtPlan1d( src, _image( values, s_min, ds, num_angle ), **kw )


def _iso( pts, hw, normals, values, s_min, ds, **kw ):
    from loom.tensor import Axis, ShapeVar
    from sdot import ContinuousOtPlan1d, ProjectedSumOfTents
    num_angle = Axis( ShapeVar( len( normals ) ), name = "num_angle" )
    src = ProjectedSumOfTents( points = pts, half_widths = hw, normal = normals, batch_axes = [ num_angle ] )
    return ContinuousOtPlan1d( src, _image( values, s_min, ds, num_angle ), **kw )


if test( "round tents: the isotropic class's cost and barycenters" ):
    pts, _, normals, values, s_min, ds = _problem()
    hw = np.random.default_rng( 1 ).uniform( 0.02, 0.08, len( pts ) )
    mats = hw[ :, None, None ] * np.eye( 2 )
    a = _aniso( pts, mats, normals, values, s_min, ds, with_barycenters = True )
    b = _iso( pts, hw, normals, values, s_min, ds, with_barycenters = True )
    assert np.abs( np.asarray( a.cost ) - np.asarray( b.cost ) ).max() < 1e-12
    assert np.abs( np.asarray( a.barycenters ) - np.asarray( b.barycenters ) ).max() < 1e-12


if test( "each item is the isotropic problem of the half-widths of its direction" ):
    pts, mats, normals, values, s_min, ds = _problem()
    a = np.asarray( _aniso( pts, mats, normals, values, s_min, ds ).cost )
    for k, n in enumerate( normals ):
        hw = np.linalg.norm( np.einsum( "irc,r->ic", mats, n ), axis = 1 )          # | M^T n |
        ref = np.asarray( _iso( pts, hw, normals[ k:k + 1 ], values[ k:k + 1 ], s_min, ds ).cost )
        assert abs( a[ k ] - ref[ 0 ] ) < 1e-12 * abs( ref[ 0 ] ), ( k, a[ k ], ref[ 0 ] )


if test( "gradients with respect to the points and the matrices, by finite differences" ):
    import jax
    jax.config.update( "jax_enable_x64", True )
    import jax.numpy as jnp
    pts, mats, normals, values, s_min, ds = _problem( seed = 1 )
    f = lambda p, m: _aniso( p, m, normals, values, s_min, ds ).cost.sum()
    gp, gm = jax.grad( f, argnums = ( 0, 1 ) )( jnp.asarray( pts ), jnp.asarray( mats ) )
    gp, gm = np.asarray( gp ), np.asarray( gm )
    rng = np.random.default_rng( 2 )
    dp, dm = rng.normal( size = pts.shape ), rng.normal( size = mats.shape )
    eps = 1e-6
    fd_p = ( float( f( pts + eps * dp, mats ) ) - float( f( pts - eps * dp, mats ) ) ) / ( 2 * eps )
    fd_m = ( float( f( pts, mats + eps * dm ) ) - float( f( pts, mats - eps * dm ) ) ) / ( 2 * eps )
    assert abs( fd_p - ( gp * dp ).sum() ) < 1e-6 * abs( fd_p ), ( fd_p, ( gp * dp ).sum() )
    assert abs( fd_m - ( gm * dm ).sum() ) < 1e-6 * abs( fd_m ), ( fd_m, ( gm * dm ).sum() )


if test( "with_gradient is refused ( the half-widths are not what is differentiated )" ):
    pts, mats, normals, values, s_min, ds = _problem()
    try:
        _aniso( pts, mats, normals, values, s_min, ds, with_gradient = True )
    except NotImplementedError:
        pass
    else:
        raise AssertionError( "with_gradient should be refused" )


if test( "grid method: close to the exact cost, the gradient with respect to the matrices by finite differences" ):
    import jax
    jax.config.update( "jax_enable_x64", True )
    import jax.numpy as jnp
    pts, mats, normals, values, s_min, ds = _problem()
    exact = np.asarray( _aniso( pts, mats, normals, values, s_min, ds ).cost )
    grid = np.asarray( _aniso( pts, mats, normals, values, s_min, ds, method = "grid", scale = 32 ).cost )
    assert np.abs( grid / exact - 1 ).max() < 1e-4, np.abs( grid / exact - 1 ).max()
    f = lambda m: _aniso( pts, m, normals, values, s_min, ds, method = "grid", scale = 32 ).cost.sum()
    gm = np.asarray( jax.grad( f )( jnp.asarray( mats ) ) )
    dm = np.random.default_rng( 2 ).normal( size = mats.shape )
    eps = 1e-6
    fd = ( float( f( mats + eps * dm ) ) - float( f( mats - eps * dm ) ) ) / ( 2 * eps )
    assert abs( fd - ( gm * dm ).sum() ) < 1e-5 * abs( fd ), ( fd, ( gm * dm ).sum() )
