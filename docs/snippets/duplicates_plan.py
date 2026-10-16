import numpy as np
import sdot

rng = np.random.default_rng( 0 )
positions = rng.random( ( 50, 2 ) )
positions[ 1 ] = positions[ 0 ]                          # an exact duplicate

plan = sdot.OtProblem( sdot.SumOfDiracs( positions ), sdot.Box() ).solve( sdot.Iterative( duplicates = "split" ) )   # the default
print( plan.groups.members() )                           # the groups of the plan
print( plan.stats[ "residual_full" ] )                   # each member's own mass, to the precision of the cells
b = plan.barycenters                                     # each member's own piece: 0 and 1 no longer share a barycentre
