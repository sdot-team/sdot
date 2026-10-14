import numpy as np
import sdot

rng = np.random.default_rng( 0 )
f = sdot.SumOfDiracs( rng.random( ( 100, 2 ) ) )  # 100 diracs of equal mass
g = sdot.Image( rng.random( ( 10, 10 ) ) )  # a 10 x 10 image, on the unit square

print( sdot.w2( f, g ) )  # W2^2, as a numpy array
