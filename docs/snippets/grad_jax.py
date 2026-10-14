import jax
import jax.numpy as jnp
import sdot
jax.config.update( "jax_enable_x64", True )

k1, k2 = jax.random.split( jax.random.key( 0 ) )
positions = jax.random.uniform( k1, ( 100, 2 ) )
masses    = jnp.ones( 100 )                        # normalized by sdot: only the ratios count
pixels    = jax.random.uniform( k2, ( 10, 10 ) )

def loss( positions, masses, pixels ):
    return sdot.w2( sdot.SumOfDiracs( positions, masses ), sdot.Image( pixels ) )

value, grads = jax.jit( jax.value_and_grad( loss, argnums = ( 0, 1, 2 ) ) )( positions, masses, pixels )
d_positions, d_masses, d_pixels = grads            # [ 100, 2 ], [ 100 ], [ 10, 10 ]
