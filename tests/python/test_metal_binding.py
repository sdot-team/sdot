"""
End-to-end smoke test for the JAX -> Metal binding (forward only).

The JAX graph runs on the CPU XLA backend; the generated Obj-C++ FFI handler launches a
hand-written Metal compute kernel on the unified-memory pointers. This proves the whole
pipeline: device selection, .mm build with the Metal framework, ctypes capsule registration,
and the kernel round-trip. The generic MSL codegen (from FfiCodeParallel) comes later.

Run on Apple Silicon only.
"""
import platform
import numpy
import pytest

import sdot
from sdot.drivers.driver import FfiCodeCustom, FfiCodeParallel


pytestmark = pytest.mark.skipif(
    platform.system() != "Darwin" or platform.machine() != "arm64",
    reason = "Metal binding requires Apple Silicon",
)


# MSL kernel: element-wise add over a flat 1D grid.
_ADD_MSL = """
#include <metal_stdlib>
using namespace metal;
kernel void add_kernel(
    device const float* a   [[buffer(0)]],
    device const float* b   [[buffer(1)]],
    device       float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    out[ id ] = a[ id ] + b[ id ];
}
"""

# Handler body: wrap each TensorView's unified-memory pointer and dispatch the kernel.
_ADD_FWD = """
    static const char *msl = R"MSL(%s)MSL";
    metal_launch_1d( msl, "add_kernel", {
        MetalBuf{ (void *)( p.a.data().raw ),      size_t( p.a.nb_items() )      * sizeof( TF ), false },
        MetalBuf{ (void *)( p.b.data().raw ),      size_t( p.b.nb_items() )      * sizeof( TF ), false },
        MetalBuf{ (void *)( p.output.data().raw ), size_t( p.output.nb_items() ) * sizeof( TF ), true  },
    }, size_t( p.output.nb_items() ) );
""" % _ADD_MSL


def test_metal_elementwise_add():
    sdot.driver.framework = "jax"
    sdot.driver.device = "metal"

    n = 1024
    a = sdot.driver.array( numpy.arange( n, dtype = numpy.float32 ) )
    b = sdot.driver.array( numpy.arange( n, dtype = numpy.float32 ) * 3.0 )

    out = sdot.driver.call(
        FfiCodeCustom( includes = [ "sdot/metal/metal_launch.h" ], fwd_code = _ADD_FWD, name = "metal_add" ),
        output = sdot.Return( sdot.Tensor( "n" ), n = n ),
        a = a,
        b = b,
    )

    ref = numpy.asarray( a ) + numpy.asarray( b )
    assert numpy.allclose( numpy.asarray( out ), ref ), ( numpy.asarray( out )[ :8 ], ref[ :8 ] )


def test_metal_parallel_codegen():
    """Same op, but through the generic FfiCodeParallel -> MSL codegen path (no hand-written
    kernel): the CallArgs emit the MSL struct + buffers, FfiCodeParallel provides the body."""
    sdot.driver.framework = "jax"
    sdot.driver.device = "metal"

    n = 2048
    a = sdot.driver.array( numpy.linspace( 0, 1, n, dtype = numpy.float32 ) )
    b = sdot.driver.array( numpy.linspace( 2, 5, n, dtype = numpy.float32 ) )

    out = sdot.driver.call(
        FfiCodeParallel(
            name       = "padd",
            parallel_over = [ "output" ],
            fwd_body   = "p.output( batch_index ) = p.a( batch_index ) + p.b( batch_index );",
        ),
        output = sdot.Return( sdot.Tensor( "n" ), n = n ),
        a = a,
        b = b,
    )

    ref = numpy.asarray( a ) + numpy.asarray( b )
    assert numpy.allclose( numpy.asarray( out ), ref ), ( numpy.asarray( out )[ :8 ], ref[ :8 ] )


def test_metal_scalar_param():
    """A scalar parameter ( `scale` ) reaches the MSL kernel as a 1-element buffer: `p.scale`."""
    sdot.driver.framework = "jax"
    sdot.driver.device = "metal"

    n = 512
    a = sdot.driver.array( numpy.linspace( 0, 1, n, dtype = numpy.float32 ) )

    out = sdot.driver.call(
        FfiCodeParallel(
            name       = "pscale",
            parallel_over = [ "output" ],
            fwd_body   = "p.output( batch_index ) = p.scale * p.a( batch_index );",
        ),
        output = sdot.Return( sdot.Tensor( "n" ), n = n ),
        a = a,
        scale = 2.5,
    )

    assert numpy.allclose( numpy.asarray( out ), numpy.asarray( a ) * 2.5 )


def test_metal_target_specific_body():
    """Per-context body selection: the "metal" variant must win over the "*" fallback."""
    sdot.driver.framework = "jax"
    sdot.driver.device = "metal"

    n = 256
    a = sdot.driver.array( numpy.linspace( 0, 1, n, dtype = numpy.float32 ) )

    out = sdot.driver.call(
        FfiCodeParallel(
            name       = "psel",
            parallel_over = [ "output" ],
            fwd_body   = {
                "*":     "p.output( batch_index ) = p.a( batch_index );",
                "metal": "p.output( batch_index ) = p.a( batch_index ) * 10.0f;",
            },
        ),
        output = sdot.Return( sdot.Tensor( "n" ), n = n ),
        a = a,
    )

    assert numpy.allclose( numpy.asarray( out ), numpy.asarray( a ) * 10.0 )


if __name__ == "__main__":
    test_metal_elementwise_add()
    test_metal_parallel_codegen()
    test_metal_scalar_param()
    test_metal_target_specific_body()
