# What the capacity check costs under a jit, on the card: the same 3D solve ( uniform3d, n seeds, target masses traced ) compiled
# three ways, min of `reps` wall times of the compiled function called again:
#   callback : `jax.jit`, each call with a kernel-decided count checked at run time by a `jax.debug.callback`
#   watch    : `loom.jit`, the checks carried out of the program by `checkify`, read once after the run ( rerun on overflow )
#   none     : `jax.jit` with the callbacks removed -- no check at all ( the lower bound, unsafe )
#   errand run --env lmo-jax --no-queue python sdot-ffi/bench/experiments/jit_watch.py 100000 5
import os, sys, time
sys.path.insert( 0, os.path.join( os.path.dirname( os.path.abspath( __file__ ) ), ".." ) )
import cases

n    = int( sys.argv[ 1 ] ) if len( sys.argv ) > 1 else 100_000
reps = int( sys.argv[ 2 ] ) if len( sys.argv ) > 2 else 5

import numpy
import jax
import loom
from sdot import Image, Iterative, OtProblem, SumOfDiracs

pos, _, d = cases.case( "uniform3d", n = n, seed = 0 )
print( f"device { loom.resolved_device() }, n = { n }, d = { d }", flush = True )
target = lambda: Image( values = numpy.ones( ( 1, ) * d ), origin = [ 0.0 ] * d, pixel_frame = numpy.eye( d ) )


def solve( masses ):
    plan = OtProblem( SumOfDiracs( pos, masses ), target() ).solve( Iterative( tol = 1e-6 / n, max_iter = 60, differentiable = False ) )
    return plan.weights, plan.stats[ "nb_iter" ]


# the callbacks a trace emits: counted, or dropped for `none`
_callback = jax.debug.callback
counts = { "n": 0 }


def counting( *a, **k ):
    counts[ "n" ] += 1
    return _callback( *a, **k )


def dropped( *a, **k ):
    counts[ "n" ] += 1


def timed( name, f ):
    masses = numpy.ones( n )
    t = time.perf_counter()
    w, it = f( masses )
    jax.block_until_ready( w )
    first = time.perf_counter() - t
    best = None
    for _ in range( reps ):
        t = time.perf_counter()
        w, it = f( masses )
        jax.block_until_ready( w )
        wall = time.perf_counter() - t
        best = wall if best is None else min( best, wall )
    print( f"{ name:9s} first { first:7.3f} s   best { best:7.4f} s   nb_iter { int( it ) }   callbacks traced { counts[ 'n' ] }", flush = True )
    return numpy.asarray( w ).reshape( -1 )


results = {}
for name in ( "callback", "watch", "none", "callback", "watch", "none" ):   # twice, interleaved: the order should not matter
    counts[ "n" ] = 0
    jax.debug.callback = { "callback": counting, "watch": counting, "none": dropped }[ name ]
    try:
        # a fresh function each time: `jax.jit` caches its traces per function, and `none` must be traced without callbacks
        fresh = lambda m: solve( m )
        f = loom.jit( fresh ) if name == "watch" else jax.jit( fresh )
        results[ name ] = timed( name, f )
    finally:
        jax.debug.callback = _callback

ref = results[ "callback" ]
for name in ( "watch", "none" ):
    print( f"max |w_{ name } - w_callback| = { numpy.abs( results[ name ] - ref ).max():.3e}" )


# MANY SMALL CALLS: `k` calls in one jitted function, each with a count decided by the kernel ( `nb` items asked, capacity
# `nb` ): what one callback per call costs when the calls are short
from loom import ShapeVar, CtShapeVar, Axis, Aggregate, RealTensor, IntTensor
from loom.compilation.FfiCode import FfiCode
import jax.numpy as jnp


class Pts( Aggregate ):
    values      : RealTensor[ "num_value" ]
    num_value   : Axis[ "nb_values" ]
    nb_values   : ShapeVar


KERNEL = FfiCode.per_item( code = """
auto c = outputs.cell( batch_index );
c.nb_values.set( SI( inputs.wanted( batch_index ) ) );
for ( SI i = 0; i < SI( c.nb_values ); ++i )
    c.values( num_value = i ) = i;
""" )


def small( x, nb ):
    cell = Pts()
    wanted = IntTensor()
    wanted.set( nb )
    loom.ffi_call( "jit_watch_small", KERNEL, wanted = wanted, cell = loom.out( cell, writes = ( "nb_values", "values" ),
                                                                              capacities = { "nb_values": 64 } ) )
    return x + cell.values.raw[ 0 ]


for k in ( 10, 100, 1000 ):
    def chain( x ):
        for i in range( k ):
            # each call depends on the previous one ( `x` ) and differs from the others ( `i` ): XLA can neither merge identical
            # calls nor drop unused ones -- the count stays 32 ( x >= 0 )
            x = small( x, jnp.int32( 32 ) + ( x < -1 ).astype( jnp.int32 ) * i )
        return x
    for name in ( "callback", "watch", "none" ):
        counts[ "n" ] = 0
        jax.debug.callback = { "callback": counting, "watch": counting, "none": dropped }[ name ]
        try:
            fresh = lambda x: chain( x )
            f = loom.jit( fresh ) if name == "watch" else jax.jit( fresh )
            x = jnp.zeros( (), dtype = jnp.float32 )
            jax.block_until_ready( f( x ) )
            best = None
            for _ in range( reps ):
                t = time.perf_counter()
                jax.block_until_ready( f( x ) )
                wall = time.perf_counter() - t
                best = wall if best is None else min( best, wall )
            print( f"k = { k:4d}  { name:9s} best { 1e3 * best:8.3f} ms   ( { 1e6 * best / k:7.2f} us per call )   callbacks traced { counts[ 'n' ] }", flush = True )
        finally:
            jax.debug.callback = _callback
