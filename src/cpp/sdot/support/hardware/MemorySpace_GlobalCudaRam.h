#pragma once

#ifdef __CUDACC__

#include "ExecutionSpace_Cuda.h"
#include "ExecutionSpace_Cpu.h"
#include "MemorySpace_CpuRam.h"
#include "accessible_from.h"
#include "MemorySpace.h"
#include "Ptr.h"
#include "../Ct.h"

namespace sdot {

/// device global memory
struct MemorySpace_GlobalCudaRam : MemorySpace {
    auto execution_space() const { return ExecutionSpace_Cuda{}; }

    /// allocate `n` items of T on the device, expose them (as an informed Ptr) to `func`, then release
    T_T void with_reservation( PI n, auto &&func ) const {
        T *p = nullptr;
        cudaMalloc( reinterpret_cast<void **>( &p ), n * sizeof( T ) );
        func( Ptr<T,MemorySpace_GlobalCudaRam>( p, *this ) );
        cudaFree( p );
    }
};

constexpr auto operator==( MemorySpace_GlobalCudaRam, MemorySpace_GlobalCudaRam ) { return Ct<bool,true>(); }
constexpr auto operator==( MemorySpace_GlobalCudaRam, auto                      ) { return Ct<bool,false>(); }
constexpr auto operator==( auto                     , MemorySpace_GlobalCudaRam ) { return Ct<bool,false>(); }

auto accessible_from( ExecutionSpace_Cuda, MemorySpace_GlobalCudaRam ) { return Ct<bool,true>(); }

/// memory space a CUDA execution space allocates into when it must materialize data
auto native_memory_space( ExecutionSpace_Cuda ) { return MemorySpace_GlobalCudaRam{}; }

// Transfer primitive copy( execution_space, dst_ptr, src_ptr, nb_items ): the execution
// space carries the stream; the (dst,src) memory-space types select the direction.
// CPU → GPU (enqueued on the target stream)
T_T void copy( ExecutionSpace_Cuda es, Ptr<T,MemorySpace_GlobalCudaRam> dst, Ptr<T,MemorySpace_CpuRam> src, PI nb_items ) {
    cudaMemcpyAsync( dst.raw, src.raw, nb_items * sizeof( T ), cudaMemcpyHostToDevice, es.stream );
}
// GPU → GPU (same stream; TODO: peer copy for different devices)
T_T void copy( ExecutionSpace_Cuda es, Ptr<T,MemorySpace_GlobalCudaRam> dst, Ptr<T,MemorySpace_GlobalCudaRam> src, PI nb_items ) {
    cudaMemcpyAsync( dst.raw, src.raw, nb_items * sizeof( T ), cudaMemcpyDeviceToDevice, es.stream );
}
// GPU → CPU (target is the host: no host stream, so drive it on the GPU memory's stream, then sync)
T_T void copy( ExecutionSpace_Cpu, Ptr<T,MemorySpace_CpuRam> dst, Ptr<T,MemorySpace_GlobalCudaRam> src, PI nb_items ) {
    auto es = MemorySpace_GlobalCudaRam{}.execution_space();
    cudaMemcpyAsync( dst.raw, src.raw, nb_items * sizeof( T ), cudaMemcpyDeviceToHost, es.stream );
    cudaStreamSynchronize( es.stream );
}

} // namespace sdot

#endif
