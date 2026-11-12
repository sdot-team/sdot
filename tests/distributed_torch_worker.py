"""Run by `test_distributed.py`: the distributed `PowerDiagram` on 4 Torch processes ( gloo ), each holding a block of the seeds, compared
to the ordinary diagram. Prints `all ranks ok` when every rank agrees."""
import os
import socket


def _cases():
    import numpy as np
    rng = np.random.default_rng( 1 )
    n = 2048
    pos = rng.random( ( n, 2 ) )
    huge = rng.random( n ) * 0.01
    huge[ :5 ] = 0.5
    yield "voronoi", pos, np.zeros( n )
    yield "huge weights", pos, huge
    yield "clustered", rng.normal( 0.5, 0.05, ( n, 2 ) ).clip( 0.01, 0.99 ), rng.random( n ) * 1e-4


def work( rank, size, port ):
    import numpy as np
    import torch
    import torch.distributed as dist
    import loom
    os.environ.update( MASTER_ADDR = "127.0.0.1", MASTER_PORT = str( port ) )
    dist.init_process_group( "gloo", rank = rank, world_size = size )
    loom.default_framework = "torch"
    from sdot.Cell import set_kernel_dtype
    from sdot.PowerDiagram import PowerDiagram, box_half_spaces
    set_kernel_dtype( "FP64" )
    bnd = box_half_spaces( [ 0, 0 ], [ 1, 1 ] )
    ranks = loom.TorchRanks()
    # one box per rank, two ( boxes of one size ), three ( boxes of different sizes: lists of boxes )
    for ( name, pos, w ), nb_boxes in [ ( c, k ) for c in _cases() for k in ( None, 2 * size, 3 * size ) ]:
        m = len( pos ) // size
        block = lambda a: torch.as_tensor( a[ rank * m:( rank + 1 ) * m ].copy() )
        pd = PowerDiagram( loom.spread( block( pos ), ranks ), loom.spread( block( w ), ranks ), boundaries = bnd,
                           nb_boxes = nb_boxes, msg_capacity = 8, inbox_capacity = 4 )
        res = pd.measures
        assert isinstance( res.value, torch.Tensor ) and tuple( res.value.shape ) == ( m, ), ( name, res.value.shape )
        full = loom.gather_to_host( res )
        ref = np.asarray( loom.to_numpy( PowerDiagram( pos, w, boundaries = bnd ).measures.value ) )
        assert abs( full.sum() - 1 ) < 1e-12, ( name, full.sum() )
        assert np.abs( full - ref ).max() < 1e-12, ( name, np.abs( full - ref ).max() )
        # the reductions are over all the ranks ( every one gets them ), and the other tensors stay what they were
        for red in ( "sum", "max", "min", "mean" ):
            got = float( getattr( res, red )() )
            assert abs( got - getattr( np, red )( ref ) ) < 1e-12, ( name, red, got )
        assert type( loom.RealTensor( full ) ) is loom.RealTensor
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    import torch.multiprocessing as mp
    with socket.socket() as s:
        s.bind( ( "127.0.0.1", 0 ) )
        port = s.getsockname()[ 1 ]
    mp.spawn( work, args = ( 4, port ), nprocs = 4 )
    print( "all ranks ok" )
