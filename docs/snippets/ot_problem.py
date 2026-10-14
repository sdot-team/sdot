import numpy as np
import sdot

rng = np.random.default_rng( 0 )
positions = rng.random( ( 100, 2 ) )

pb = sdot.OtProblem( sdot.SumOfDiracs( positions ), sdot.Image( rng.random( ( 10, 10 ) ) ) )
for _ in range( 10 ):
    p = pb.solve()                                       # restarts from the previous solution by itself
    positions = p.barycenters
    pb.source = sdot.SumOfDiracs( positions )
