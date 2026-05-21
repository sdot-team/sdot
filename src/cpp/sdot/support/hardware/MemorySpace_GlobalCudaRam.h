#pragma once

#ifdef __CUDACC__

#include "ExecutionSpace_Cuda.h"
#include "ExecutionSpace_Cpu.h"
#include "MemorySpace_CpuRam.h"
#include "accessible_from.h"
#include "MemorySpace.h"
#include "../Ct.h"
#include "Ptr.h"

namespace sdot {

/// device global memory
struct MemorySpace_GlobalCudaRam : MemorySpace {
    auto HD execution_space() const { return ExecutionSpace_Cuda{}; }

    /// allocate `n` items of T on the device, expose them (as an informed Ptr) to `func`, then release.
    /// cudaMalloc/cudaFree are host APIs -> __host__ only.
    T_T __host__ void with_reservation( PI n, auto &&func ) const {
        T *p = nullptr;
        cudaMalloc( reinterpret_cast<void **>( &p ), n * sizeof( T ) );
        func( Ptr<T,MemorySpace_GlobalCudaRam>( p, *this ) );
        cudaFree( p );
    }
};

constexpr auto operator==( MemorySpace_GlobalCudaRam, MemorySpace_GlobalCudaRam ) { return Ct<bool,true>(); }
constexpr auto operator==( MemorySpace_GlobalCudaRam, auto                      ) { return Ct<bool,false>(); }
constexpr auto operator==( auto                     , MemorySpace_GlobalCudaRam ) { return Ct<bool,false>(); }

HD auto accessible_from( ExecutionSpace_Cuda, MemorySpace_GlobalCudaRam ) { return Ct<bool,true>(); }

/// memory space a CUDA execution space allocates into when it must materialize data
HD auto native_memory_space( ExecutionSpace_Cuda ) { return MemorySpace_GlobalCudaRam{}; }

// Transfer primitive copy( dst_ptr, src_ptr, nb_items[, es] ): the (dst,src) memory-space types
// select the direction; the execution space carries the stream.
//
// Inter-space transfers go through cudaMemcpy, which is a host API -> __host__ only (calling them
// from device code is therefore a natural compile error).
T_T HD void copy( Ptr<T,MemorySpace_GlobalCudaRam> dst, Ptr<T,MemorySpace_CpuRam> src, PI nb_items, ExecutionSpace_Cuda es = {} ) {
    #ifdef __CUDA_ARCH__
        __trap();
    #else
        cudaMemcpyAsync( dst.raw, src.raw, nb_items * sizeof( T ), cudaMemcpyHostToDevice, es.stream );
    #endif
}
T_T HD void copy( Ptr<T,MemorySpace_CpuRam> dst, Ptr<T,MemorySpace_GlobalCudaRam> src, PI nb_items, ExecutionSpace_Cuda es = {} ) {
    #ifdef __CUDA_ARCH__
        __trap();
    #else
        cudaMemcpyAsync( dst.raw, src.raw, nb_items * sizeof( T ), cudaMemcpyDeviceToHost, es.stream );
        cudaStreamSynchronize( es.stream );
    #endif
}

// device -> device: feasible from both sides -> HD, and this is where __CUDA_ARCH__ legitimately
// picks an *implementation* (both branches are valid): in a kernel, an internal element copy; from
// the host, cudaMemcpy. (TODO: peer copy for different devices.)
T_T HD void copy( Ptr<T,MemorySpace_GlobalCudaRam> dst, Ptr<T,MemorySpace_GlobalCudaRam> src, PI nb_items, ExecutionSpace_Cuda es = {} ) {
    #ifdef __CUDA_ARCH__
        for ( PI i = 0; i < nb_items; ++i )
            dst.raw[ i ] = src.raw[ i ];
    #else
        cudaMemcpyAsync( dst.raw, src.raw, nb_items * sizeof( T ), cudaMemcpyDeviceToDevice, es.stream );
    #endif
}

} // namespace sdot

#endif
