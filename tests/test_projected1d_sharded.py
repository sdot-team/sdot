"""`ProjectedLoss1d.sharded`: the angles split between devices ( here 4 CPU devices, forced before jax starts ). A
file of its own: the device count must be set before the first `import jax` of the process."""
import os

os.environ[ "XLA_FLAGS" ] = os.environ.get( "XLA_FLAGS", "" ) + " --xla_force_host_platform_device_count=4"

from errand import test


if test( "the angles split between devices give the same loss and gradient" ):
    import jax
    jax.config.update( "jax_enable_x64", True )
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh
    from sdot.projected1d import ProjectedLoss1d
    from loom.testing import need
    need( "cpu" )
    assert len( jax.devices() ) == 4, jax.devices()
    rng = np.random.default_rng( 0 )
    A, B, n = 12, 48, 400
    angles = np.pi * np.arange( A ) / A
    values = rng.random( ( A, B ) ) + 0.1
    pts = rng.uniform( -0.8, 0.8, ( n, 2 ) )
    radii = rng.uniform( 0.02, 0.06, n )
    mesh = Mesh( np.array( jax.devices() ), ( "a", ) )
    for be in [ "jax", "kernel" ]:
        L = ProjectedLoss1d( angles, values, -2.0, 4.0 / B, scale = 4, max_radius = 0.06, dtype = np.float64, backend = be )
        v, ( gp, gr ) = jax.value_and_grad( L, argnums = ( 0, 1 ) )( pts, radii )
        Ls = L.sharded( mesh )
        vs, ( gps, grs ) = jax.jit( jax.value_and_grad( Ls, argnums = ( 0, 1 ) ) )( jnp.asarray( pts ), jnp.asarray( radii ) )
        assert abs( float( vs ) - float( v ) ) < 1e-12 * float( v ), ( be, float( vs ), float( v ) )
        assert np.allclose( gps, gp, rtol = 1e-10, atol = 1e-15 ) and np.allclose( grs, gr, rtol = 1e-10, atol = 1e-15 ), be
