import numpy as np
import sdot

rng = np.random.default_rng( 0 )
positions = rng.random( ( 50, 2 ) )
positions[ 1 ] = positions[ 0 ]                          # an exact duplicate
positions[ 3 ] = positions[ 2 ] + 1e-9                   # a pair closer than the doubles can separate

pd = sdot.PowerDiagram( positions, boundaries = sdot.box_half_spaces( [ 0, 0 ], [ 1, 1 ] ), groups = "auto" )
print( pd.groups.members() )                             # { 0: [ 0, 1 ], 2: [ 2, 3 ], ... }
m = pd.measures                                          # every member has a cell of its own, the cells tile the box
