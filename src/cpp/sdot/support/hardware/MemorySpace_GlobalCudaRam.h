#pragma once

#ifdef __CUDACC__

#include "ExecutionSpace_Cuda.h"
#include "ExecutionSpace_Cpu.h"
#include "MemorySpace.h"
#include "../Ct.h"

namespace sdot {

/// pageable host memory
struct MemorySpace_GlobalCudaRam : MemorySpace {
    auto execution_space() const { return ExecutionSpace_Cuda{}; }
};

constexpr auto operator==( MemorySpace_GlobalCudaRam, MemorySpace_GlobalCudaRam ) { return Ct<bool,true>(); }
constexpr auto operator==( MemorySpace_GlobalCudaRam, auto                      ) { return Ct<bool,false>(); }
constexpr auto operator==( auto                     , MemorySpace_GlobalCudaRam ) { return Ct<bool,false>(); }

auto transfer_cost( ExecutionSpace_Cuda, MemorySpace_GlobalCudaRam, auto &&/*get_nb_bytes*/ ) { return Ct<int,0>(); }
auto transfer_cost( ExecutionSpace_Cpu, MemorySpace_GlobalCudaRam, auto &&/*get_nb_bytes*/ ) { return Ct<int,1>(); }

// CPU → GPU  (enqueued on dst_arch.stream)
// inline void arch_copy( const CudaGpu &dst_arch, void *dst, const Cpu &, const void *src, PI n_bytes ) {
//     cudaMemcpyAsync( dst, src, n_bytes, cudaMemcpyHostToDevice, dst_arch.stream );
// }

// // GPU → CPU  (enqueued on src_arch.stream, then synced so the CPU can read immediately)
// inline void arch_copy( const Cpu &, void *dst, const CudaGpu &src_arch, const void *src, PI n_bytes ) {
//     cudaMemcpyAsync( dst, src, n_bytes, cudaMemcpyDeviceToHost, src_arch.stream );
//     cudaStreamSynchronize( src_arch.stream );
// }

// // GPU → GPU  (same device: D2D on dst_arch.stream; different devices: peer transfer)
// inline void arch_copy( const CudaGpu &dst_arch, void *dst, const CudaGpu &src_arch, const void *src, PI n_bytes ) {
//     if ( dst_arch.device_id == src_arch.device_id )
//         cudaMemcpyAsync( dst, src, n_bytes, cudaMemcpyDeviceToDevice, dst_arch.stream );
//     else
//         cudaMemcpyPeerAsync( dst, dst_arch.device_id, src, src_arch.device_id, n_bytes, dst_arch.stream );
// }

} // namespace sdot

#endif
