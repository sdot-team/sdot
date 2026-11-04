import os
os.environ[ "XLA_FLAGS" ] = "--xla_force_host_platform_device_count=4"  # 4 CPU devices, to try it on a laptop ( not on GPUs )

import jax
import numpy as np
import sdot
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

jax.config.update( "jax_enable_x64", True )
mesh = Mesh( np.array( jax.devices() ), ( "p", ) )

rng = np.random.default_rng( 0 )
positions = jax.device_put( rng.random( ( 10000, 2 ) ), NamedSharding( mesh, P( "p", None ) ) )  # split along the seeds
weights = rng.random( 10000 ) * 1e-3

pd = sdot.PowerDiagram( positions, weights, boundaries = sdot.box_half_spaces( np.zeros( 2 ), np.ones( 2 ) ) )
m = pd.measures         # one jax array, split between the devices like the positions
print( m.sharding.spec, m.sum() )
