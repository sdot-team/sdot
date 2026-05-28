#pragma once

#ifdef __CUDACC__

#include "../containers/for_each_item_split.h"
#include "ExecutionContext.h"
#include "CudaThreadInfo.h"
#include "RunTraits.h"

#include <cuda_runtime.h>

namespace sdot {

// Continuation for the device-inline (nested) branch of run_parallel: once RunTraits::per_thread
// has produced its per-thread args, walk the items and apply func( item, args... ). Expressed as a
// functor (not a lambda) because nvcc rejects generic extended __host__ __device__ lambdas and this
// lives inside the HD run_parallel. Mirrors the cont the launched kernel uses.
template<class Func,class List>
struct DeviceInlineForEach {
    Func func; ///< reference type (T&)
    List list; ///< reference type (const T&)
    GD void operator()( auto &&...args ) const {
        for_each_item( list, RunTraits::apply_to_item( func, FORWARD( args )... ) );
    }
};

// Device kernel mirroring ExecutionContext_Cpu's per-thread body: each device thread runs
// the optional per_thread() setup, then walks its share of the items (split by global id
// over the whole grid) and calls func( item, args... ) on each.
template<class List,class Func,class... Args>
__global__ void _execution_space_cuda_run_parallel( List list, Func func, Args... args ) {
    CudaThreadInfo thread_info;
    const int nth = thread_info.nb_threads();
    const int gid = thread_info.global_id();
    // Block rounding launches nb_blocks*threads_per_block threads, usually more than there are
    // items. Excess threads (gid >= nb_items) get no item from the round-robin split below, so
    // they must skip per_thread() too: that per-thread setup indexes scratch by global_id(), which
    // is only sized for the participating threads — running it here reads/writes out of bounds (and
    // has hung the kernel). This mirrors the CPU path, which dispatches min(nb_items, threads).
    if ( gid >= int( list.nb_items() ) )
        return;
    RunTraits::per_thread( func, thread_info, list, [&]( auto &&...args ) {
        for_each_item_split( list, gid, nth, [&]( auto &&item ) {
            func( item, FORWARD( args )... );
        } );
    }, FORWARD( args )... );
}

// Kernel launch must live in a pure __host__ function: a <<<>>> launch reached from a
// __host__ __device__ function can leave the kernel instantiation as an "invalid device
// function" at runtime. run_parallel (HD, for device nesting) forwards its host branch here.
// NB: this is necessary but NOT sufficient — because run_parallel is HD, nvcc still instantiates
// the whole launch chain for the device pass, where it takes the __CUDA_ARCH__ branch (never the
// launch), so the kernel's *device* code is never emitted and the launch fails at runtime. The
// device branch of run_parallel therefore anchors the kernel explicitly (see below).
CPU_ONLY void launch_cuda_run_parallel( cudaStream_t stream, const auto &list, auto &&func, auto &&...args ) {
    const int threads_per_block = 256;
    const int nb_threads        = int( list.nb_items() );
    const int nb_blocks         = ( nb_threads + threads_per_block - 1 ) / threads_per_block;
    _execution_space_cuda_run_parallel<<< nb_blocks, threads_per_block, 0, stream >>>( list, FORWARD( func ), FORWARD( args )... );
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

    HD void run_parallel( const auto &list, auto &&func, auto &&...args ) const {
    #ifdef __CUDA_ARCH__
        // Called from device code: we are already inside a kernel, so there is no nested launch
        // (that would need dynamic parallelism). Run inline on the current thread — the device
        // counterpart of ExecutionContext_Cpu's inline (already-parallelized) branch. Run the
        // optional per_thread() setup first (like the launched kernel and the CPU path do), then
        // walk the items. Functor (not lambda) continuation, since nvcc rejects generic extended
        // HD lambdas.
        CudaThreadInfo thread_info;
        RunTraits::per_thread( func, thread_info, list,
            DeviceInlineForEach<DECAYED_TYPE_OF( func )&,const DECAYED_TYPE_OF( list )&>{ func, list },
            FORWARD( args )... );

        // Anchor the kernel the HOST branch launches. nvcc emits a launched kernel's device code
        // only if device-compiled code references it; the HD launch chain alone does not, so a
        // top-level host launch would fail at runtime with cudaErrorInvalidDeviceFunction. This
        // never-executed address-of, compiled in the device pass, forces that device-side
        // instantiation. The template arguments mirror launch_cuda_run_parallel's by-value
        // (decayed) deduction so the anchored and launched instantiations are the same kernel.
        (void) &_execution_space_cuda_run_parallel<DECAYED_TYPE_OF( list ),DECAYED_TYPE_OF( func ),DECAYED_TYPE_OF( args )...>;
    #else
        // one thread per item for now (round-robin in the kernel tolerates any grid size).
        // TODO: cap with RunTraits::max_gpu_threads once it accounts for registers/shared mem.
        launch_cuda_run_parallel( stream, list, FORWARD( func ), FORWARD( args )... );
    #endif
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
