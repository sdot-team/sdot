#pragma once

// Reusable test matrices, shared across C++ test files.
//
// Each "axis" is a small combinator invoking a callback once per case; axes are
// orthogonal and compose by nesting. On top of them, `check_binary_op_matrix`
// sweeps the full product (operand-A memory space × operand-B memory space ×
// shape × execution context) for an in-place binary op, transferring operands
// in/out as needed and checking the result against a scalar reference.
//
// Memory spaces swept: CpuRam always; PinnedCpuRam and GlobalCudaRam under nvcc.
// Execution contexts: top-level, and "already-parallelized" (operands routed
// through a 1-item run_parallel so a nested run_*() dispatches inline).

#include "../../src/cpp/sdot/support/containers/TensorView.h"
#include "../../src/cpp/sdot/support/hardware/Run.h"
#include "../../src/cpp/sdot/support/containers/Range.h"
#include "../../src/cpp/sdot/support/display.h"
#include "catch_main.h"

#include <sstream>
#include <vector>
#include <cstring>

namespace sdot_test {

using namespace sdot;

// --- memory-space axis -----------------------------------------------------

/// Calls `f( memory_space )` for each swept memory space.
void for_each_memory_space( auto &&f ) {
    f( MemorySpace_CpuRam{} );
#ifdef __CUDACC__
    f( MemorySpace_PinnedCpuRam{} );
    f( MemorySpace_GlobalCudaRam{} );
#endif
}

// --- shape axis ------------------------------------------------------------

/// Calls `f( shape )` for representative shapes of rank 0, 1, 2.
void for_each_shape( auto &&f ) {
    f( tuple() );        // rank 0 (scalar)
    f( tuple( 5 ) );     // rank 1
    f( tuple( 2, 3 ) );  // rank 2
}

// --- execution-context axis ------------------------------------------------

/// HD functor applying the operation under test; used to carry `op` into a kernel
/// by value (device code cannot capture host stack by reference).
struct ApplyOp {
    HD void operator()( auto /*item*/, auto op, auto &&...operands ) const { op( FORWARD( operands )... ); }
};

/// Runs the operation under test in one execution context.
struct ContextRunner {
    bool        nested;
    const char *label;

    /// `op( operands... )` either directly (top level) or wrapped in a 1-item run_parallel
    /// so the operands carry has_already_been_parallelized (nested -> any run_*() inside runs inline).
    void operator()( auto &&op, auto &&...operands ) const {
        if ( ! nested )
            op( FORWARD( operands )... );
        else
            run_parallel( range( 1 ), ApplyOp{}, FORWARD( op ), FORWARD( operands )... );
    }

    void display( auto &os ) const { os << label; }
};

/// Calls `f( ContextRunner )` once per execution context.
void for_each_execution_context( auto &&f ) {
    f( ContextRunner{ false, "top-level" } );
    f( ContextRunner{ true,  "already-parallelized" } );
}

// --- transfer helpers (host <-> any space) ---------------------------------

template<class T,class MS>
void seed_from_host( Ptr<T,MS> dst, const T *host, std::size_t n ) {
    if constexpr ( DECAYED_TYPE_OF( accessible_from( ExecutionContext_Cpu{}, MS{} ) )::value )
        std::memcpy( dst.raw, host, n * sizeof( T ) );
    else
        copy( dst, Ptr<T,MemorySpace_CpuRam>( const_cast<T *>( host ) ), n );
}

template<class T,class MS>
void read_to_host( T *host, Ptr<T,MS> src, std::size_t n ) {
    if constexpr ( DECAYED_TYPE_OF( accessible_from( ExecutionContext_Cpu{}, MS{} ) )::value )
        std::memcpy( host, src.raw, n * sizeof( T ) );
    else
        copy( Ptr<T,MemorySpace_CpuRam>( host ), src, n );
}

inline void sync_device() {
#ifdef __CUDACC__
    cudaStreamSynchronize( ExecutionContext_Cuda::default_stream );
#endif
}

// --- the binary-op matrix --------------------------------------------------

/// Sweep (ms_a × ms_b × shape × context) for the in-place binary op `op( a, b )`
/// (a, b same-shape tensors). For each cell: allocate a in ms_a and b in ms_b,
/// seed from host, run `op` in the context, read a back, and CHECK every element
/// against `reference( a0, b0 )` computed on the host seed values.
///   op        : HD callable, e.g. []  HD ( auto a, auto b ) { a += b; }
///   reference : host scalar, e.g. [] ( double a, double b ) { return a + b; }
template<class TF = double>
void check_binary_op_matrix( auto op, auto reference ) {
    for_each_memory_space( [&]( auto ms_a ) {
        for_each_memory_space( [&]( auto ms_b ) {
            for_each_shape( [&]( auto shape ) {
                for_each_execution_context( [&]( auto run ) {
                    using MSA = DECAYED_TYPE_OF( ms_a );
                    using MSB = DECAYED_TYPE_OF( ms_b );
                    const std::size_t n = shape.apply_values( []( auto... s ) { return ( std::size_t( 1 ) * ... * std::size_t( s ) ); } );

                    // host seed values (distinct, so a wrong index/element shows up)
                    std::vector<TF> a0( n ), b0( n ), got( n );
                    for ( std::size_t i = 0; i < n; ++i ) { a0[ i ] = TF( i + 1 ); b0[ i ] = TF( 10 * ( i + 1 ) ); }

                    ms_a.template with_reservation<TF>( n, [&]( Ptr<TF,MSA> pa ) {
                        ms_b.template with_reservation<TF>( n, [&]( Ptr<TF,MSB> pb ) {
                            seed_from_host( pa, a0.data(), n );
                            seed_from_host( pb, b0.data(), n );

                            TensorView va( pa.raw, shape, contiguous_strides<TF>( shape ), ms_a );
                            TensorView vb( pb.raw, shape, contiguous_strides<TF>( shape ), ms_b );

                            run( op, va, vb );
                            sync_device();

                            read_to_host( got.data(), pa, n );

                            std::ostringstream os;
                            os << "a="; display( os, ms_a );
                            os << " b="; display( os, ms_b );
                            os << " shape="; display( os, shape );
                            os << " ctx="; display( os, run );
                            INFO( os.str() );

                            for ( std::size_t i = 0; i < n; ++i )
                                CHECK( got[ i ] == reference( a0[ i ], b0[ i ] ) );
                        } );
                    } );
                } );
            } );
        } );
    } );
}

} // namespace sdot_test
