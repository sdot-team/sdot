# mpiexec -n 4 python parallel_mpi.py
import loom
import numpy as np
import sdot
from mpi4py import MPI

comm = MPI.COMM_WORLD
ranks = loom.MpiRanks( comm )
rank, size = comm.Get_rank(), comm.Get_size()

# each process holds its block of the seeds ( cut here from a cloud they all draw the same way )
rng = np.random.default_rng( 0 )
positions, weights = rng.random( ( 10000, 2 ) ), rng.random( 10000 ) * 1e-3
n = len( positions ) // size
block = lambda a: a[ rank * n:( rank + 1 ) * n ]

pd = sdot.PowerDiagram( loom.spread( block( positions ), ranks ), loom.spread( block( weights ), ranks ),
                        boundaries = sdot.box_half_spaces( [ 0, 0 ], [ 1, 1 ] ) )
m = pd.measures         # spread like the positions: this process holds the measures of its seeds ( `m.value` )
total = m.sum()         # over all the processes: every one gets it
if rank == 0:
    print( float( total ) )  # 1: the area of the unit square
