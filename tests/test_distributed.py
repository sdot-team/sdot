"""The distributed sweep ( `sdot/distributed.py` ): the cells travelling between boxes must give the measures of the ordinary diagram,
seed by seed. A file of its own: the device count of the `shard_map` tests must be set before the first `import jax` of the process."""
import os

os.environ[ "XLA_FLAGS" ] = os.environ.get( "XLA_FLAGS", "" ) + " --xla_force_host_platform_device_count=4"

from errand import test


def _cases():
    """( name, positions, weights ): an even cloud, cells that cross many boxes ( huge weights ), a cluster, very few seeds"""
    import numpy as np
    rng = np.random.default_rng( 1 )
    n = 2048
    pos = rng.random( ( n, 2 ) )
    huge = rng.random( n ) * 0.01
    huge[ :5 ] = 0.5
    yield "voronoi", pos, np.zeros( n )
    yield "huge weights", pos, huge
    yield "clustered", rng.normal( 0.5, 0.05, ( n, 2 ) ).clip( 0.01, 0.99 ), rng.random( n ) * 1e-4
    yield "few seeds", rng.random( ( 64, 2 ) ), rng.random( 64 ) * 0.05


if test( "PowerDiagram( nb_boxes = K ): the boxes in turn give the measures of the ordinary diagram" ):
    import numpy as np
    from sdot.Cell import set_kernel_dtype
    from sdot.PowerDiagram import PowerDiagram, box_half_spaces
    set_kernel_dtype( "FP64" )
    bnd = box_half_spaces( [ 0, 0 ], [ 1, 1 ] )
    for name, pos, w in _cases():
        ref = np.asarray( PowerDiagram( pos, w, boundaries = bnd ).measures.value )
        for K in [ 1, 3, 16, 52 ]:
            if K > len( pos ) // 2:
                continue
            # ( rows of 8 vertices: the widening and the restart are part of the test )
            pd = PowerDiagram( pos, w, boundaries = bnd, nb_boxes = K, msg_capacity = 8 )
            assert isinstance( pd, PowerDiagram )
            m = np.asarray( pd.measures.value )
            assert abs( m.sum() - 1 ) < 1e-12, ( name, K, m.sum() )
            assert np.abs( m - ref ).max() < 1e-12, ( name, K, np.abs( m - ref ).max() )

if test( "PowerDiagram( nb_boxes = K ): differentiable ( Jax, Torch ), the gradient of the ordinary diagram wrt positions and weights" ):
    import numpy as np
    import loom
    from errand import has_tag
    from loom.tensor.Dtype import Dtype
    from sdot.Cell import set_kernel_dtype
    from sdot.PowerDiagram import PowerDiagram, box_half_spaces
    fw = next( ( n for n in ( "jax", "torch" ) if has_tag( f"driver={ n }" ) ), None )
    if fw is None:
        print( "numpy: forward only" )
    else:
        if fw == "jax":
            import jax
            jax.config.update( "jax_enable_x64", True )
        old, loom.default_framework = loom.default_framework, fw
        set_kernel_dtype( "FP64" )
        o = loom.ops( None )
        arr = lambda a: o.array( np.asarray( a, dtype = np.float64 ), Dtype.fp( 64 ) )
        bnd = box_half_spaces( [ 0, 0 ], [ 1, 1 ] )
        for name, pos, w in _cases():
            pos, w = pos[ :512 ], w[ :512 ]
            C = arr( np.random.default_rng( 2 ).random( len( pos ) ) )

            def grads( **kw ):
                return o.grad( lambda p, q: o.sum( PowerDiagram( p, q, boundaries = bnd, **kw ).measures.value * C ), argnums = ( 0, 1 ) )( arr( pos ), arr( w ) )

            gp, gw = grads()
            for K in [ 1, 3, 16 ]:
                if K > len( pos ) // 2:
                    continue
                # ( rows of 8 vertices: the widening, the restart and the tape's are part of the test )
                hp, hw = grads( nb_boxes = K, msg_capacity = 8 )
                ep = np.abs( np.asarray( loom.to_numpy( hp ) ) - np.asarray( loom.to_numpy( gp ) ) ).max()
                ew = np.abs( np.asarray( loom.to_numpy( hw ) ) - np.asarray( loom.to_numpy( gw ) ) ).max()
                assert ep < 1e-11 and ew < 1e-11, ( name, K, ep, ew )
        loom.default_framework = old

if test( "PowerDiagram( nb_boxes = K, max_resident = M ): out of core, the measures of the ordinary diagram" ):
    import numpy as np
    from sdot.Cell import set_kernel_dtype
    from sdot.PowerDiagram import PowerDiagram, box_half_spaces
    set_kernel_dtype( "FP64" )
    bnd = box_half_spaces( [ 0, 0 ], [ 1, 1 ] )
    for name, pos, w in _cases():
        ref = np.asarray( PowerDiagram( pos, w, boundaries = bnd ).measures.value )
        for K, M in [ ( 4, 1 ), ( 16, 2 ) ]:
            pd = PowerDiagram( pos, w, boundaries = bnd, nb_boxes = K, max_resident = M, msg_capacity = 8 )
            m = np.asarray( pd.measures.value )
            assert abs( m.sum() - 1 ) < 1e-12, ( name, K, m.sum() )
            assert np.abs( m - ref ).max() < 1e-12, ( name, K, np.abs( m - ref ).max() )

if test( "PowerDiagram of spread positions ( Jax ): one box or two per device give the measures of the ordinary diagram" ):
    import jax
    jax.config.update( "jax_enable_x64", True )
    import numpy as np
    import loom
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
    from loom.testing import need
    from sdot.Cell import set_kernel_dtype
    from sdot.PowerDiagram import PowerDiagram, box_half_spaces
    need( "cpu" )
    assert len( jax.devices() ) == 4, jax.devices()
    set_kernel_dtype( "FP64" )
    bnd = box_half_spaces( [ 0, 0 ], [ 1, 1 ] )
    for name, pos, w in _cases():
        ref = np.asarray( PowerDiagram( pos, w, boundaries = bnd ).measures.value )
        for nd in [ 2, 4 ]:          # ( one device: not spread, the ordinary diagram )
            mesh = Mesh( np.array( jax.devices()[ :nd ] ), ( "p", ) )
            ranks = loom.JaxRanks( mesh )
            # spread by loom, or a Jax array already sharded along the seeds: the same thing
            for p, nb_boxes in [ ( loom.spread( pos, ranks ), None ), ( jax.device_put( pos, NamedSharding( mesh, P( "p", None ) ) ), None ),
                                 ( loom.spread( pos, ranks ), 2 * nd ) ]:      # ( several boxes per device )
                # ( small capacities: the overflow of a row and of an inbox, and the restart, are part of the test )
                pd = PowerDiagram( p, w, boundaries = bnd, nb_boxes = nb_boxes, msg_capacity = 8, inbox_capacity = 4 )
                m = pd.measures
                assert loom.spread_of( m ) is not None and loom.spread_of( m ).ranks == ranks
                m = loom.gather_to_host( m )
                assert abs( m.sum() - 1 ) < 1e-12, ( name, nd, m.sum() )
                assert np.abs( m - ref ).max() < 1e-12, ( name, nd, np.abs( m - ref ).max() )

if test( "PowerDiagram of spread positions ( Torch, 4 processes ): the same program gives the same measures" ):
    # one process per rank ( `torch.distributed`, gloo ): a script of its own, `mp.spawn` re-imports its main module
    import os
    import subprocess
    import sys
    from loom.testing import need
    need( "torch" )
    script = os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), "distributed_torch_worker.py" )
    r = subprocess.run( [ sys.executable, script ], capture_output = True, text = True, timeout = 600 )
    assert r.returncode == 0 and "all ranks ok" in r.stdout, ( r.stdout[ -2000: ], r.stderr[ -4000: ] )

if test( "PowerDiagram of spread positions ( MPI + numpy ): each rank holds a block, the measures are those of the ordinary diagram",
         ranks = 4 ):
    # one process per rank: errand starts this body under the MPI launcher of the place ( `mpiexec -n 4` )
    import importlib.util
    from errand import skip
    if importlib.util.find_spec( "mpi4py" ) is None:
        skip( "needs mpi4py", hint = "micromamba install -c conda-forge mpi4py mpich" )
    import numpy as np
    from mpi4py import MPI
    import loom
    loom.default_framework = "numpy"
    from sdot.Cell import set_kernel_dtype
    from sdot.PowerDiagram import PowerDiagram, box_half_spaces
    set_kernel_dtype( "FP64" )
    rank, size = MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size()
    assert size > 1, "one process: not started by the launcher"     # ( 4 by default, `--ranks 8` asks for more )
    bnd = box_half_spaces( [ 0, 0 ], [ 1, 1 ] )
    ranks = loom.MpiRanks()
    # one box per rank, two ( boxes of one size ), three ( boxes of different sizes: lists of boxes )
    for ( name, pos, w ), nb_boxes in [ ( c, k ) for c in _cases() for k in ( None, 2 * size, 3 * size ) ]:
        m = len( pos ) // size
        block = lambda a: a[ rank * m:( rank + 1 ) * m ].copy()
        # ( small capacities: the overflow of a row and of an inbox, and the restart, are part of the test )
        pd = PowerDiagram( loom.spread( block( pos ), ranks ), loom.spread( block( w ), ranks ), boundaries = bnd,
                           nb_boxes = nb_boxes, msg_capacity = 8, inbox_capacity = 4 )
        res = pd.measures
        assert isinstance( res.value, np.ndarray ) and res.value.shape == ( m, ), ( name, res.value.shape )
        full = loom.gather_to_host( res )
        ref = np.asarray( PowerDiagram( pos, w, boundaries = bnd ).measures.value )
        assert abs( full.sum() - 1 ) < 1e-12, ( name, full.sum() )
        assert np.abs( full - ref ).max() < 1e-12, ( name, np.abs( full - ref ).max() )
        assert np.abs( res.value - ref[ rank * m:( rank + 1 ) * m ] ).max() < 1e-12, name     # ( this rank's block, in place )
        # the reductions are over all the ranks ( every one gets them ), and the other tensors stay what they were
        for red in ( "sum", "max", "min", "mean" ):
            got = float( getattr( res, red )() )
            assert abs( got - getattr( np, red )( ref ) ) < 1e-12, ( name, red, got )
        assert type( loom.RealTensor( full ) ) is loom.RealTensor
