import numpy as np

# a program written for pysdot, unchanged but for these two lines
import sdot.pysdot
sdot.pysdot.install()

from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot import OptimalTransport

domain = ConvexPolyhedraAssembly()
domain.add_box( [ 0, 0 ], [ 1, 1 ] )
domain.sub_box( [ 0.4, 0.4 ], [ 0.6, 0.6 ] )          # a hole

positions = np.random.default_rng( 0 ).uniform( 0, 1, ( 100, 2 ) )
ot = OptimalTransport( positions, domain = domain )
for _ in range( 10 ):                                  # Lloyd iterations
    ot.adjust_weights()
    ot.set_positions( ot.get_centroids() )

print( ot.pd.integrals().sum(), domain.measure() )     # 0.96 0.96
