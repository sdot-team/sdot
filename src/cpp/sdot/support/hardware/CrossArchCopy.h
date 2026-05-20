#pragma once

#include "common_types.h"
#include "Cpu.h"
#ifdef __CUDACC__
#include "CudaGpu.h"
#endif

namespace sdot {

// arch_copy — generic cross-architecture byte copy.
// Using a free function with overloads lets callers express source and destination arches
// independently, covering CPU↔CPU, CPU↔GPU, GPU↔GPU (same and different devices).

// CPU → CPU
inline void arch_copy( const Cpu &, void *dst, const Cpu &, const void *src, PI n_bytes ) {
    std::memcpy( dst, src, n_bytes );
}

#ifdef __CUDACC__
// CPU → GPU  (enqueued on dst_arch.stream)
inline void arch_copy( const CudaGpu &dst_arch, void *dst, const Cpu &, const void *src, PI n_bytes ) {
    cudaMemcpyAsync( dst, src, n_bytes, cudaMemcpyHostToDevice, dst_arch.stream );
}

// GPU → CPU  (enqueued on src_arch.stream, then synced so the CPU can read immediately)
inline void arch_copy( const Cpu &, void *dst, const CudaGpu &src_arch, const void *src, PI n_bytes ) {
    cudaMemcpyAsync( dst, src, n_bytes, cudaMemcpyDeviceToHost, src_arch.stream );
    cudaStreamSynchronize( src_arch.stream );
}

// GPU → GPU  (same device: D2D on dst_arch.stream; different devices: peer transfer)
inline void arch_copy( const CudaGpu &dst_arch, void *dst, const CudaGpu &src_arch, const void *src, PI n_bytes ) {
    if ( dst_arch.device_id == src_arch.device_id )
        cudaMemcpyAsync( dst, src, n_bytes, cudaMemcpyDeviceToDevice, dst_arch.stream );
    else
        cudaMemcpyPeerAsync( dst, dst_arch.device_id, src, src_arch.device_id, n_bytes, dst_arch.stream );
}
#endif

} // namespace sdot
