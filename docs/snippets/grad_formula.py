import jax
import jax.numpy as jnp
import sdot
jax.config.update( "jax_enable_x64", True )

positions = jax.random.uniform( jax.random.key( 0 ), ( 100, 2 ) )
g = sdot.Image( jnp.ones( ( 10, 10 ) ) )

grad = jax.grad( lambda x: sdot.w2( sdot.SumOfDiracs( x ), g ) )( positions )

p = sdot.plan( sdot.SumOfDiracs( positions ), g )
m, b = p.cell_masses, p.barycenters                # jax arrays
assert jnp.allclose( grad, 2 * m[ :, None ] * ( positions - b ) )
