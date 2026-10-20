import cupy as cp
import sdot

rng = cp.random.default_rng( 0 )
f = sdot.SumOfDiracs( rng.random( ( 100, 2 ) ) )  # 100 diracs of equal mass
g = sdot.Image( rng.random( ( 10, 10 ) ) )  # a 10 x 10 image, on the unit square

print( sdot.w2( f, g ) )  # W2^2, as a cupy array, computed on the GPU
