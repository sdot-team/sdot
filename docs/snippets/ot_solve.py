import numpy as np
import sdot

rng = np.random.default_rng( 0 )
positions = rng.random( ( 100, 2 ) )
f = sdot.SumOfDiracs( positions )
g = sdot.Image( rng.random( ( 10, 10 ) ) )

p = sdot.ot_solve( f, g, tol = 1e-12, max_iter = 200 )   # the same plan, with the settings of the solver
print( p.stats[ "nb_iter" ], p.stats[ "residual" ] )

# a nearby problem restarts from a previous plan: a few Newton steps instead of a few dozen
f2 = sdot.SumOfDiracs( positions + 0.01 )
p2 = sdot.ot_solve( f2, g, ot_plan = p )
print( p2.stats[ "nb_iter" ] )
