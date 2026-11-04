"""1D TRANSPORT, BY BATCHES: sdot's 1D plans on `batch` independent problems at once ( the threads, or the card, share
them ), and POT on the discretized problem -- the page `docs/guide/performance-1d.md`.

    errand run --env lmo-numpy 'sdot-ffi/bench/bench_1d.py::*' --problem=diracs --n=1000,100000
    errand run --env lmo-torch 'sdot-ffi/bench/bench_1d.py::*' --problem=images --method=pot --m=1000,100000 --backend=torch

The runs of the page are jobs of `page.py` ( `one_d*`, `gpu_one_d*` ).

THE MEASURES. An image is affine pixels ( `Image( degree = 1 )` ) sampling a sum of two thin gaussians ( standard deviation
0.05 ) at its knots; item k of a batch is shifted by 0.3 cos( k ). The problems, `batch` items each:
  `--problem=diracs`      `n` diracs drawn uniformly on [ -1, 1 ] against a target of `pixels` pixels on [ -1.2, 1.2 ];
  `--problem=images`      a source image of `n` pixels on [ -1, 1 ] against a target of as many on [ -1.2, 1.2 ].
The inputs are in `--dtype` ( float64 by default ), placed where the computation runs before the clock
starts ( the arrays of the framework, on the card with `--device=default` ); what is timed ends with the costs on the
host. After a warm-up ( the compilation ), the smallest wall time of `reps`. `error`: the median over the items of the
relative difference of the W2 with sdot's in double ( computed apart, not timed ).

`--method`:
  sdot  `SdotPlan1d` ( diracs: `SumOfDiracs` batched ) or `ContinuousOtPlan1d` ( images ): W2^2, exact up to the rounding
  pot   ( images ) POT's `ot.wasserstein_1d( p = 2 )`, one column per item, on the discretized problem: each image replaced
        by `m` diracs, the centres of a regular grid of its support weighted by the exact masses of their cells, the same
        number on both sides, made before the clock starts. `--backend`: numpy ( one thread ), torch, or jax ( the card with
        `--device=default`: POT computes on it through these backends ).
"""

import numpy
from errand import Param, bench

import benchlib

S_MIN, S_MAX = -1.2, 1.2         # the targets' support
A_MIN, A_MAX = -1.0, 1.0         # the source images'
SIGMA = 0.05


def gaussians( x, a, b ):
    return numpy.exp( -0.5 * ( ( x - a ) / SIGMA ) ** 2 ) + 0.7 * numpy.exp( -0.5 * ( ( x - b ) / SIGMA ) ** 2 )


def coeffs( lo, hi, pixels, shifts, a, b ):
    """the Bernstein coefficients of the images of a batch, `[ batch, pixels, 2 ]`: the density at the knots of each pixel"""
    f = gaussians( numpy.linspace( lo, hi, pixels + 1 )[ None, : ] - shifts[ :, None ], a, b )
    return numpy.stack( [ f[ :, :-1 ], f[ :, 1: ] ], axis = 2 )


def discretized( c, lo, hi, m ):
    """the affine images `c` ( `[ batch, pixels, 2 ]` on [ lo, hi ] ) as `m` diracs each: the centres of a regular grid of
    [ lo, hi ], weighted by the exact masses of their cells ( the cumulated mass, quadratic in each pixel, at the edges of
    the grid ), normalized. `( positions [ m ], weights [ m, batch ] )`"""
    V, P, _ = c.shape
    h = ( hi - lo ) / P
    cum = numpy.concatenate( [ numpy.zeros( ( V, 1 ) ), numpy.cumsum( h * c.sum( axis = 2 ) / 2, axis = 1 ) ], axis = 1 )
    e = numpy.linspace( lo, hi, m + 1 )
    j = numpy.clip( ( ( e - lo ) / h ).astype( numpy.int64 ), 0, P - 1 )
    u = e - ( lo + h * j )                                                  # the offset in the pixel
    a, s = c[ :, j, 0 ], ( c[ :, j, 1 ] - c[ :, j, 0 ] ) / h
    w = numpy.maximum( numpy.diff( cum[ :, j ] + u * ( a + s * u / 2 ), axis = 1 ), 0 )
    return ( e[ :-1 ] + e[ 1: ] ) / 2, ( w / w.sum( axis = 1, keepdims = True ) ).T


if p := bench( "one_d",
               problem = Param( "diracs", choices = [ "diracs", "images" ], help = "the problem ( see above )" ),
               method  = Param( "sdot", choices = [ "sdot", "pot" ], help = "the method ( see above )" ),
               dtype   = Param( "float64", choices = [ "float32", "float64" ], help = "the float of the inputs ( and of the computation )" ),
               n       = Param( 100000, help = "the diracs or points, or the pixels of the source images" ),
               batch   = Param( 256, help = "the problems" ),
               pixels  = Param( 10000, help = "diracs: the pixels of the targets" ),
               m       = Param( 1000, help = "pot: the diracs of each discretized image" ),
               backend = Param( "numpy", choices = [ "numpy", "torch", "jax", "auto" ], help = "pot: the array library ( auto: the framework of the environment )" ),
               threads = Param( -1, help = "workers ( 0: all the cores, -1: the physical ones )" ),
               pin     = Param( "yes", choices = [ "yes", "no", "env" ], help = "pin worker w to core w" ),
               reps    = Param( 3, help = "repetitions ( the smallest wall time is kept )" ),
               seed    = Param( 0, help = "the draw" ),
               device  = Param( "cpu", choices = [ "cpu", "default" ], help = "cpu: the host, even with a card; default: the card if any" ),
               kernels = Param( "catalogue", choices = [ "catalogue", "compile" ], help = "`compile`: compiled for this machine" ) ):
    benchlib.set_device( p.device )                # BEFORE importing loom
    benchlib.set_kernels( p.kernels )
    benchlib.set_threads( p.threads, p.pin )
    benchlib.set_kernel_timing()
    import os
    import time

    n, V = p.n, p.batch
    if p.method == "pot" and p.problem != "images":
        raise ValueError( "--method=pot: the images only" )
    dt = numpy.dtype( p.dtype )
    P = n if p.problem == "images" else p.pixels                                 # the pixels of the targets
    shifts = 0.3 * numpy.cos( numpy.arange( V ) )
    tgt = coeffs( S_MIN, S_MAX, P, -shifts, -0.3, 0.4 )                          # [ batch, P, 2 ]
    rng = numpy.random.default_rng( p.seed )
    if p.problem == "diracs":
        xs = rng.uniform( -1, 1, ( V, n, 1 ) )
    else:
        src = coeffs( A_MIN, A_MAX, n, shifts, -0.4, 0.3 )
    print( "  " + benchlib.env_report() + "; " + benchlib.driver_line() )

    # the arrays of the framework, where the computation runs ( placed before the clock starts )
    import loom
    fw = loom.resolved_framework().module_name
    on_card = bool( getattr( loom.resolved_device(), "is_cuda_gpu", False ) )
    backend = ( fw if fw in ( "torch", "jax" ) else "numpy" ) if p.backend == "auto" else p.backend
    if fw == "torch" or p.method == "pot" and backend == "torch":
        import torch
        torch.set_num_threads( benchlib.nb_threads() )
        dev = "cuda" if p.device == "default" and torch.cuda.is_available() else "cpu"
        place = lambda a: torch.as_tensor( numpy.ascontiguousarray( a ), device = dev )
        sync = lambda: torch.cuda.synchronize() if dev == "cuda" else None
    elif fw == "jax" or p.method == "pot" and backend == "jax":
        import jax
        place = lambda a: jax.device_put( jax.numpy.asarray( a ) )
        sync = lambda: None
    else:
        place, sync = numpy.ascontiguousarray, lambda: None

    def host( c ):
        from loom.tensor.host import to_host
        sync()
        return numpy.asarray( to_host( getattr( c, "value", c ) ), dtype = float ).reshape( -1 )

    def sdot_runner( dtype ):
        """the plan of the batch, its inputs already placed"""
        import sdot
        from loom.tensor import Axis, ShapeVar
        ax = Axis( ShapeVar( V ), name = "num_angle" )
        image = lambda c, lo, hi: sdot.Image( values = c, degree = 1, origin = [ lo ], pixel_frame = [ [ ( hi - lo ) / c.shape[ 1 ] ] ], batch_axes = [ ax ] )
        T = place( tgt.astype( dtype ) )
        if p.problem == "images":
            S = place( src.astype( dtype ) )
            return lambda: host( sdot.ContinuousOtPlan1d( image( S, A_MIN, A_MAX ), image( T, S_MIN, S_MAX ) ).cost )
        X = place( xs.astype( dtype ) )
        return lambda: host( sdot.SdotPlan1d( sdot.SumOfDiracs( positions = X, batch_axes = [ ax ] ), image( T, S_MIN, S_MAX ) ).cost )

    def pot_runner():
        import ot
        a, aw = discretized( src, A_MIN, A_MAX, p.m )                      # [ m ], [ m, batch ]
        b, bw = discretized( tgt, S_MIN, S_MAX, p.m )
        a, b = numpy.broadcast_to( a[ :, None ], aw.shape ).copy(), numpy.broadcast_to( b[ :, None ], bw.shape ).copy()
        a, aw, b, bw = ( x.astype( dt ) for x in ( a, aw, b, bw ) )
        if backend == "numpy":
            return lambda: numpy.atleast_1d( ot.wasserstein_1d( a, b, aw, bw, p = 2 ) )
        a, aw, b, bw = map( place, ( a, aw, b, bw ) )
        return lambda: host( ot.wasserstein_1d( a, b, aw, bw, p = 2 ) )

    # the memory: POT's two densities of m diracs per item ( and their sorts ), against 80 % of the card's ( half the host's )
    def card_bytes():
        try:
            if fw == "jax" or backend == "jax":
                return jax.devices()[ 0 ].memory_stats()[ "bytes_limit" ]
            return torch.cuda.get_device_properties( 0 ).total_memory
        except Exception:
            return 20e9
    budget = 0.8 * card_bytes() if p.device == "default" else os.sysconf( "SC_PAGE_SIZE" ) * os.sysconf( "SC_PHYS_PAGES" ) / 2
    need = p.m * V * 8 * 8 if p.method == "pot" else 0
    if need > budget:
        p.results.update( benchlib.hardware() )
        p.results.update( n = n, batch = V, m = p.m, skipped = "memory" )
        print( f"  { p.method }: { need / 1e9 :.0f} GB, not run" )
    else:
        run = pot_runner() if p.method == "pot" else sdot_runner( dt )
        run()                                                                      # warm-up: the compilation
        kt = benchlib.KernelTiming() if p.method != "pot" and on_card else None
        best = None
        benchlib.start_clock()
        for _ in range( max( p.reps, 1 ) ):
            if kt: kt.reset()
            t = time.perf_counter()
            c = run()
            wall = time.perf_counter() - t
            k = kt.read() if kt else None
            if best is None or wall < best[ 0 ]:
                best = ( wall, c, k[ "ms" ] / 1e3 if k else None )
        wall, c, t_kernels = best
        ref = sdot_runner( numpy.float64 )()                               # on W2, not W2^2, against double
        err = float( numpy.median( numpy.abs( numpy.sqrt( c ) - numpy.sqrt( ref ) ) / numpy.sqrt( ref ) ) )
        p.results.update( benchlib.hardware() )
        p.results.update( n = n, batch = V, pixels = P, problem = p.problem, method = p.method, m = p.m if p.method == "pot" else None, dtype = p.dtype,
                          backend = backend if p.method == "pot" else fw, threads = benchlib.nb_threads(),
                          t_total_wall = wall, error = err, t_kernels = t_kernels )
        print( benchlib.table( [ "problem", "method", "backend", "batch", "n", "pixels", "m", "seconds", "kernels", "error" ],
                               [ [ p.problem, p.method, backend if p.method == "pot" else fw, f"{ V }", f"{ n }", f"{ P }",
                                   f"{ p.m if p.method == 'pot' else '-' }", f"{ wall :.3f}",
                                   "-" if t_kernels is None else f"{ t_kernels :.3f}", "-" if err is None else f"{ err :.1e}" ] ] ) )
