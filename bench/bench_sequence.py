"""PREPARED ONCE, MANY DIAGRAMS: the cost of `K` power diagrams on the same positions with weights that change at each
one -- what a Newton solve, a Lloyd step or a gradient flow at fixed positions asks for. The preparation ( the tree, the
diagram's structures ) is paid once; each next diagram reuses it and the neighbour memory of the previous one.

    errand run --env lmo-numpy 'sdot-ffi/bench/bench_sequence.py::*' --case=uniform --dim=2 --sequence=10 --threads=8

The weights go from `W / K` to `W` in `K` steps ( a path from the Voronoi diagram to a Laguerre one, like a solve ): `W` is
the file's ( the `equal` cases: a transport solution ), otherwise uniform in `[ -wscale h^2, wscale h^2 ]`, `h = n^( -1 / d )`.
Timed: the preparation ( tree + diagram ), the first diagram, the mean of the next ones; the minimum over `reps` runs, each
from scratch. `--dump=FILE` writes the cloud with the weights `W` in the format of `cases/` ( for CGAL, which rebuilds its
triangulation for every weight vector: its cost per diagram is its whole chain on that file ).
"""

from errand import Param, bench


CASE_CHOICES = [ "uniform", "lines_voronoi", "lines_equal", "planes_voronoi", "planes_equal" ]

if p := bench( "sequence",
               case      = Param( "uniform", choices = CASE_CHOICES, help = "the case ( see cases.py ); `uniform` uses --dim" ),
               dim       = Param( 2, help = "dimension of the `uniform` case" ),
               n         = Param( 0, help = "number of seeds ( 0: 1e6 for the uniform ones, the file's size otherwise )" ),
               sequence  = Param( 10, help = "K, the number of diagrams per preparation" ),
               wscale    = Param( 0.3, help = "the final weights when the case has none: uniform in [ -w h^2, w h^2 ]" ),
               threads   = Param( 0, help = "workers ( 0: all the cores )" ),
               pin       = Param( "yes", choices = [ "yes", "no", "env" ], help = "pin worker w to core w" ),
               kernel    = Param( "double", choices = [ "double", "float" ], help = "the kernel float type" ),
               memory    = Param( -1, help = "neighbour memories per seed ( -1: the default of the dimension, 0: none )" ),
               reps      = Param( 3, help = "runs from scratch, we keep the fastest" ),
               seed      = Param( 0, help = "seed of the uniform draw" ),
               dump      = Param( "", help = "write the cloud with the final weights to this file ( the `cases/` format )" ) ):
    import numpy
    import benchlib
    import cases
    benchlib.set_threads( p.threads, p.pin )
    from sdot import AaBsp, PowerDiagram, box_half_spaces
    import time

    name = f"uniform{ p.dim }d" if p.case == "uniform" else p.case
    pos, w, d = cases.case( name, n = p.n or None, seed = p.seed )
    n, K = len( pos ), max( p.sequence, 2 )
    if not numpy.any( w ):
        h = n ** ( -1.0 / d )
        w = numpy.random.default_rng( p.seed + 1 ).uniform( -p.wscale * h * h, p.wscale * h * h, n )
    seq = [ w * ( k + 1 ) / K for k in range( K ) ]
    kernel = "FP64" if p.kernel == "double" else "FP32"
    memory = None if p.memory < 0 else p.memory
    box = box_half_spaces( [ 0 ] * d, [ 1 ] * d )
    print( "  " + benchlib.env_report() + "; " + benchlib.driver_line() )

    if p.dump:
        with open( p.dump, "w" ) as f:
            f.write( f"# { name }, n = { n }, the final weights of bench_sequence ( wscale { p.wscale } where the case has none )\n{ n }\n" )
            numpy.savetxt( f, numpy.column_stack( [ pos, w ] ), fmt = "%.17g" )

    def run( pos, seq ):
        t0 = time.perf_counter()
        bsp = AaBsp( pos, seq[ 0 ] )
        pd = PowerDiagram( pos, weights = seq[ 0 ], boundaries = box, accelerator = bsp, kernel_dtype = kernel, memory = memory )
        t1 = time.perf_counter()
        m = numpy.asarray( pd.measures ).reshape( -1 )
        t2 = time.perf_counter()
        for wk in seq[ 1: ]:
            pd.weights = wk
            m = numpy.asarray( pd.measures ).reshape( -1 )
        t3 = time.perf_counter()
        return t1 - t0, t2 - t1, ( t3 - t2 ) / ( len( seq ) - 1 ), m

    run( pos[ :1000 ], [ x[ :1000 ] for x in seq[ :2 ] ] )      # compiles
    best = None
    for _ in range( p.reps ):
        r = run( pos, seq )
        if best is None or sum( r[ :2 ] ) + ( K - 1 ) * r[ 2 ] < sum( best[ :2 ] ) + ( K - 1 ) * best[ 2 ]:
            best = r
    t_prep, t_first, t_next, m = best
    total = t_prep + t_first + ( K - 1 ) * t_next

    ns = lambda t: f"{ t / n * 1e9 :.0f}"
    p.results.update( n = n, dim = d, threads = benchlib.nb_threads(), kernel = p.kernel, sequence = K, t_prepare = t_prep, t_first = t_first,
                      t_next = t_next, t_total = total, ns_per_seed_next = t_next / n * 1e9, ns_per_seed_mean = total / K / n * 1e9,
                      sum_of_measures = float( m.sum() ) )
    header = [ "case", "n", "threads", "kernel", "K", "prepare ns/seed", "first ns/seed", "next ns/seed", "mean per diagram ns/seed" ]
    print( benchlib.table( header, [ [ name, n, benchlib.nb_threads(), p.kernel, K, ns( t_prep ), ns( t_first ), ns( t_next ), ns( total / K ) ] ] ) )
    print( f"  min of { p.reps } runs from scratch; total { total :.3f} s; sum of the last measures { m.sum() :.6f}" )
