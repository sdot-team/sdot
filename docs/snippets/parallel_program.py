import os
os.environ[ "XLA_FLAGS" ] = "--xla_force_host_platform_device_count=4"

import jax
import loom
import numpy as np
from jax.sharding import Mesh

ranks = loom.JaxRanks( Mesh( np.array( jax.devices() ), ( "p", ) ) )
slots = loom.Capacity( 2 )  # rows per destination: doubled, and the program run again, if too small


def program( ranks, values ):
    """each rank sends each of its values to the rank `value % size`, which adds up what it receives"""
    o = loom.ops( values )
    dest = values % ranks.size
    out = loom.route( dest, o.reshape( values, ( -1, 1 ) ), ranks.size, slots )  # [ size, slots, 1 ], -1 where empty
    got = o.reshape( ranks.exchange( out ), ( -1, ) )                            # what the others sent to this rank
    return o.reshape( o.sum( o.where( got >= 0, got, 0 ) ), ( 1, ) )


values = loom.spread_host( np.arange( 16, dtype = np.int32 ), ranks )  # 4 values per rank
sums = loom.spmd( program, ranks )( values )           # one sum per rank
print( loom.gather_to_host( sums ) )                    # [ 0+4+8+12, 1+5+9+13, 2+6+10+14, 3+7+11+15 ]
