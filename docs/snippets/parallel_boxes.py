import numpy as np
import sdot

rng = np.random.default_rng( 0 )
positions, weights = rng.random( ( 10000, 2 ) ), rng.random( 10000 ) * 1e-3

# 8 boxes of seeds, handled one after the other on this device
pd = sdot.PowerDiagram( positions, weights, boundaries = sdot.box_half_spaces( [ 0, 0 ], [ 1, 1 ] ), nb_boxes = 8 )
m = pd.measures
print( m.sum() )  # 1: the area of the unit square
