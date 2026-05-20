#pragma once

#include "KernelTraits.h"

namespace sdot {

// Result of the launch computation: how many groups (blocks/chunks) and the size
// of each group (threads-per-block on GPU; items-per-chunk on host).
struct LaunchConfig {
    PI  grid  = 1; ///< number of blocks (GPU) / chunks (host)
    int block = 1; ///< threads per block (GPU) / items per chunk (host)
};

// Host: a single logical pass; splitting across pool threads is the runner's job.
// local_cpu_memory_size( args... ) can later tune chunk granularity.
template<class F, class... Args>
inline LaunchConfig launch_for_host( PI nb_items, const F &, const Args &... ) {
    return { nb_items, 1 };
}

} // namespace sdot

#ifdef __CUDACC__
#include <cuda_runtime.h>
namespace sdot {

// GPU: pick a block size bounded by occupancy limits derived from the functor's
// resource hints (registers/thread, per-thread local/shared bytes) and device props,
// rounded down to a warp multiple; grid covers nb_items. The hints may depend on the
// run arguments. Deliberately a simple model — the place to refine occupancy later.
template<class F, class... Args>
inline LaunchConfig launch_for_cuda( PI nb_items, int device_id, const F &func, const Args &...args ) {
    int max_tpb = 1024, regs_per_block = 65536, shared_per_block = 49152;
    cudaDeviceGetAttribute( &max_tpb,          cudaDevAttrMaxThreadsPerBlock,      device_id );
    cudaDeviceGetAttribute( &regs_per_block,   cudaDevAttrMaxRegistersPerBlock,    device_id );
    cudaDeviceGetAttribute( &shared_per_block, cudaDevAttrMaxSharedMemoryPerBlock, device_id );

    int block = max_tpb;
    int regs = kernel_nb_gpu_register_per_thread( func, args... ); // Ct<...> or runtime
    if ( regs > 0 && regs_per_block / regs < block )
        block = regs_per_block / regs;
    PI shpt = kernel_local_gpu_memory_size( func, args... );        // Ct<...> or runtime
    if ( shpt > 0 && PI( shared_per_block ) / shpt < PI( block ) )
        block = int( PI( shared_per_block ) / shpt );

    block = ( block / 32 ) * 32;          // warp multiple
    if ( block < 32 ) block = 32;

    PI grid = ( nb_items + block - 1 ) / block;
    return { grid, block };
}

} // namespace sdot
#endif

namespace sdot {
// Unified entry: dispatch host-vs-GPU launch computation from the space; the functor
// and the run arguments are forwarded so resource hints can depend on them.
template<class Space, class F, class... Args>
inline LaunchConfig compute_launch( PI nb_items, const Space &space, const F &func, const Args &...args ) {
    if constexpr ( Space::runs_on_host )
        return launch_for_host( nb_items, func, args... );
#ifdef __CUDACC__
    else
        return launch_for_cuda( nb_items, space.device_id, func, args... );
#endif
}
} // namespace sdot
