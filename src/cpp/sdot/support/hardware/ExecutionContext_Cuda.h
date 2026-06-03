#pragma once

#ifdef __CUDACC__

#include "../containers/for_each_item_split.h"
#include "ExecutionContext.h"
#include "CudaThreadInfo.h"
#include "RunTraits.h"

#include <cuda_runtime.h>
#include <cstdio> // fprintf (debug error reporting)

namespace sdot {

struct MemorySpace_GlobalCudaRam;


// Device kernel mirroring ExecutionContext_Cpu's per-thread body: each device thread runs
// the optional per_thread() setup, then walks its share of the items (split by global id
// over the whole grid) and calls func( item, args... ) on each.
template<class List,class Func,class... Args>
__global__ void _execution_space_cuda_run_parallel( int nb_workers, List list, Func func, Args... args ) {
    CudaThreadInfo thread_info;
    const int gid = thread_info.global_id();
    // `nb_workers` (= min( nb_items, max_gpu_threads ), chosen by launch_cuda_run_parallel) is the
    // intended worker count, which may be < the launched grid (block rounding) and < nb_items
    // (grid-stride). Excess threads must do nothing: per_thread() indexes per-thread scratch by
    // global_id(), sized for at most max_gpu_threads workers — running it past nb_workers reads out
    // of bounds (and has hung the kernel). Mirrors the CPU path's min(nb_items, threads) dispatch.
    if ( gid >= nb_workers )
        return;
    // grid-stride by nb_workers (NOT the rounded grid size): worker gid handles items gid,
    // gid+nb_workers, ... so the nb_workers workers cover every item exactly once.
    RunTraits::per_thread( func, thread_info, list, [&]( auto &&...args ) {
        for_each_item_split( list, gid, nb_workers, [&]( auto &&item ) {
            func( item, FORWARD( args )... );
        } );
    }, FORWARD( args )... );
}

// Kernel launch must live in a pure __host__ function: a <<<>>> launch reached from a
// __host__ __device__ function can leave the kernel instantiation as an "invalid device
// function" at runtime. run_parallel (HD, for device nesting) forwards its host branch here.
// NB: this is necessary but NOT sufficient — because run_parallel is HD, nvcc still instantiates
// the whole launch chain for the device pass, where it takes the __CUDA_ARCH__ branch (never the
// launch), so the kernel's *device* code is never emitted and the launch fails at runtime.
// The dummy launch in the __CUDA_ARCH__ branch (anchored by if(false)) forces this instantiation
// while allowing nvcc to deduce the kernel's template arguments automatically (including those
// for local/generic lambdas).
HD void launch_cuda_run_parallel( cudaStream_t stream, const auto &list, auto &&func, auto &&...args ) {
#ifdef __CUDA_ARCH__
    // Called from device code: run inline on the current thread (no nested launch).
    // per_thread() setup first, then walk items — mirrors the launched kernel path.
    CudaThreadInfo thread_info;
    RunTraits::per_thread( func, thread_info, list, [&]( auto &&...args ) {
        for_each_item( list, [&]( auto &&item ) {
            func( item, FORWARD( args )... );
        } );
    }, FORWARD( args )... );

    // Anchor the kernel. Labeled as if(false) to avoid requiring CUDA Dynamic Parallelism at
    // runtime, but the call expression forces nvcc to generate the kernel's device code.
    if ( false )
        _execution_space_cuda_run_parallel<<< 0, 0 >>>( 0, list, func, args... );
#else
    const int threads_per_block = 256;
    // bound the grid by the functor's max_gpu_threads (its per-thread scratch capacity): launching
    // one thread per item would over-subscribe scratch for large lists. Threads then grid-stride.
    const PI  nb_items   = list.nb_items();
    const PI  max_thr    = PI( RunTraits::max_gpu_threads( func, args... ) );
    const int nb_workers = int( nb_items < max_thr ? nb_items : max_thr );
    if ( nb_workers <= 0 )
        return;
    const int nb_blocks  = ( nb_workers + threads_per_block - 1 ) / threads_per_block;
    _execution_space_cuda_run_parallel<<< nb_blocks, threads_per_block, 0, stream >>>( nb_workers, list, FORWARD( func ), FORWARD( args )... );

    // Debug builds: surface CUDA errors at the kernel boundary (instead of a confusing failure at the
    // next unrelated CUDA call) and flush device-side info()/printf. The cudaStreamSynchronize is the
    // expensive part, so it is debug-only — release stays asynchronous. cudaGetLastError catches launch
    // (config) errors; the synchronize catches runtime errors (e.g. illegal address) and flushes printf.
    #ifndef NDEBUG
        if ( cudaError_t e = cudaGetLastError(); e != cudaSuccess )
            fprintf( stderr, "[sdot] CUDA launch error: %s\n", cudaGetErrorString( e ) );
        if ( cudaError_t e = cudaStreamSynchronize( stream ); e != cudaSuccess )
            fprintf( stderr, "[sdot] CUDA kernel error: %s\n", cudaGetErrorString( e ) );
    #endif
#endif
}

// CUDA device execution + stream — {}-constructible (uses the global default_stream; the
// bindings set default_stream). A specific stream can be passed explicitly; mind that
// using non-default streams requires care w.r.t. ordering/synchronization.
struct ExecutionContext_Cuda : public ExecutionContext {
    using MemorySpace = MemorySpace_GlobalCudaRam;
    static cudaStream_t default_stream; ///< process-wide default, set by the bindings
    cudaStream_t        stream;         ///< stream this execution space enqueues onto

    explicit HD constexpr ExecutionContext_Cuda( cudaStream_t s ) : stream( s ) {}
    // default_stream is a host symbol; in device code (where the context is only used as a
    // compile-time tag for accessible_from) the stream value is irrelevant, so use 0 there.
    #ifdef __CUDA_ARCH__
        HD constexpr ExecutionContext_Cuda() : stream( 0 ) {}
    #else
        HD ExecutionContext_Cuda() : stream( default_stream ) {}
    #endif

    HD void run_parallel( const auto &list, auto &&func, auto &&...args ) const {
        launch_cuda_run_parallel( stream, list, FORWARD( func ), FORWARD( args )... );
    }
};

} // namespace sdot

#endif // __CUDACC__
