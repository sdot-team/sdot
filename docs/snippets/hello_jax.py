import jax
import sdot

k1, k2 = jax.random.split( jax.random.key( 0 ) )
f = sdot.SumOfDiracs( jax.random.uniform( k1, ( 100, 2 ) ) )  # 100 diracs of equal mass
g = sdot.Image( jax.random.uniform( k2, ( 10, 10 ) ) )  # a 10 x 10 image, on the unit square

print( sdot.w2( f, g ) )  # W2^2, as a jax array
