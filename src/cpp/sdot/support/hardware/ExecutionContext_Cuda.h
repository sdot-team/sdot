#pragma once

#ifdef __CUDACC__

#include "../containers/for_each_item_split.h"
#include "ExecutionContext.h"
#include "CudaThreadInfo.h"
#include "RunTraits.h"

#include <cuda_runtime.h>

namespace sdot {

// Device kernel mirroring ExecutionContext_Cpu's per-thread body: each device thread runs
// the optional per_thread() setup, then walks its share of the items (split by global id
// over the whole grid) and calls func( item, args... ) on each.
template<class List,class Func,class... Args>
__global__ void _execution_space_cuda_run_parallel( List list, Func func, Args... args ) {
    CudaThreadInfo thread_info;
    const int nth = thread_info.nb_threads();
    const int gid = thread_info.global_id();
    RunTraits::per_thread( func, thread_info, list, [&]( auto &&...args ) {
        for_each_item_split( list, gid, nth, [&]( auto &&item ) {
            func( item, FORWARD( args )... );
        } );
    }, args... );
}

// CUDA device execution + stream — {}-constructible (uses the global default_stream; the
// bindings set default_stream). A specific stream can be passed explicitly; mind that
// using non-default streams requires care w.r.t. ordering/synchronization.
struct ExecutionContext_Cuda : public ExecutionContext {
    static cudaStream_t default_stream; ///< process-wide default, set by the bindings
    cudaStream_t        stream;         ///< stream this execution space enqueues onto

    explicit HD ExecutionContext_Cuda( cudaStream_t s ) : stream( s ) {}
    // default_stream is a host symbol; in device code (where the context is only used as a
    // compile-time tag for accessible_from) the stream value is irrelevant, so use 0 there.
    #ifdef __CUDA_ARCH__
        HD ExecutionContext_Cuda() : stream( 0 ) {}
    #else
        HD ExecutionContext_Cuda() : stream( default_stream ) {}
    #endif

    void run_parallel( const auto &list, auto &&func, auto &&...args ) {
        // one thread per item for now (round-robin in the kernel tolerates any grid size).
        // TODO: cap with RunTraits::max_gpu_threads once it accounts for registers/shared mem.
        const int threads_per_block = 256;
        const int nb_threads        = int( list.nb_items() );
        const int nb_blocks         = ( nb_threads + threads_per_block - 1 ) / threads_per_block;
        _execution_space_cuda_run_parallel<<< nb_blocks, threads_per_block, 0, stream >>>( list, func, FORWARD( args )... );
    }

};

// int block = max_tpb;
// int regs = kernel_nb_gpu_register_per_thread( func, args... ); // Ct<...> or runtime
// if ( regs > 0 && regs_per_block / regs < block )
//     block = regs_per_block / regs;
// PI shpt = kernel_local_gpu_memory_size( func, args... );        // Ct<...> or runtime
// if ( shpt > 0 && PI( shared_per_block ) / shpt < PI( block ) )
//     block = int( PI( shared_per_block ) / shpt );

// block = ( block / 32 ) * 32;          // warp multiple
// if ( block < 32 ) block = 32;

// PI grid = ( nb_items + block - 1 ) / block;
// return { grid, block };

} // namespace sdot

#endif
