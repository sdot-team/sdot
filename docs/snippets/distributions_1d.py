import numpy as np
import sdot

rng = np.random.default_rng( 0 )
f = sdot.SumOfDiracs( rng.random( ( 1000, 1 ) ) )            # diracs on a line
g = sdot.Image( rng.random( 100 ) )                         # 100 pixels on [ 0, 1 ]

p = sdot.ot_solve( f, g, with_barycenters = True )            # 1D: a sort, exact, no tolerance
