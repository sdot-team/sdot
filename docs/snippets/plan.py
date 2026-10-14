import numpy as np
import sdot

rng = np.random.default_rng( 0 )
f = sdot.SumOfDiracs( rng.random( ( 100, 2 ) ) )
g = sdot.Image( rng.random( ( 10, 10 ) ) )

p = sdot.plan( f, g )

p.cost                # W2^2, the same value as `sdot.w2( f, g )`
p.barycenters         # [ 100, 2 ]: the barycenter of each cell
p.cell_masses         # [ 100 ]: the mass of each cell ( the mass of its dirac )
p.weights             # [ 100 ]: the weights of the power diagram
p.converged           # True

p.write_html( "plan.html" )    # the cells, the seeds and the density, in a self-contained page
