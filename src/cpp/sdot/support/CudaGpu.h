#pragma once

#ifdef __CUDACC__

#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include "common_types.h"
#include <vector>

namespace sdot {
template<class TI, class Arch, int ct_rank, class... Attrs> class AxisTuple;
struct CudaGpu; // forward declaration for use in _gpu_run_kernel

// ------------------------------------------------------ GPU thread info ------------------------------------------------------
struct GpuThreadInfo {
    __device__ int global_id()         const { return blockIdx.x * blockDim.x + threadIdx.x; }
    __device__ int nb_threads()        const { return gridDim.x * blockDim.x; }
    __device__ int local_id()          const { return threadIdx.x; }
    __device__ int block_id()          const { return blockIdx.x; }
    __device__ int threads_per_block() const { return blockDim.x; }
};

// ------------------------------------------------------ GPU kernels ------------------------------------------------------
template<class F> __global__ void _gpu_run_single_kernel( F func ) { func(); }

template<class F, class BT>
__global__ void _gpu_run_kernel( F func, BT batch_sizes ) {
    using TI = typename BT::TI;
    constexpr int N = BT::ct_rank;
    batch_sizes.for_each_index( [&]( auto ...is ) {
        func( AxisTuple<TI,CudaGpu,N>( Values(), is... ) );
    } );
}

template<class F, class BT>
__global__ void _gpu_parallel_runner_kernel( F thread_func, BT batch_sizes ) {
    GpuThreadInfo ti;
    const int gid = ti.global_id();
    const int nb  = ti.nb_threads();
    thread_func( ti, [&]( auto &&for_each_bi ) {
        batch_sizes.for_each_index_split( gid, nb, for_each_bi );
    } );
}

// ------------------------------------------------------ ParallelRunner_Gpu ------------------------------------------------------
// ParallelRunner_Gpu stores cudaStream_t directly to avoid circular dependency with CudaGpu
template<class BatchSizes>
struct ParallelRunner_Gpu {
    ParallelRunner_Gpu( cudaStream_t stream, BatchSizes batch_sizes, int max_nb_threads, int threads_per_block )
        : stream( stream ), batch_sizes( batch_sizes ),
          nb_blocks( ( max_nb_threads + threads_per_block - 1 ) / threads_per_block ),
          threads_per_block( threads_per_block ) {}

    void for_each_thread( auto &&thread_func ) {
        _gpu_parallel_runner_kernel<<<nb_blocks, threads_per_block, 0, stream>>>( thread_func, batch_sizes );
    }

    cudaStream_t stream;
    BatchSizes   batch_sizes;
    int          nb_blocks;
    int          threads_per_block;
};

// ------------------------------------------------------ CudaGpu ------------------------------------------------------
struct CudaGpu {
    explicit           CudaGpu             ( cudaStream_t s ) : stream( s ) { cudaGetDevice( &device_id ); }

    T_T struct         Vector          { using type = thrust::device_vector<T>; };

    T_T void           with_reservation( PI size, auto &&func ) const { T *res = nullptr; cudaMalloc( reinterpret_cast<void **>( &res ), size * sizeof( T ) ); func( res ); cudaFree( res ); }
    auto               parallel_runner ( auto batch_sizes, int max_nb_threads, int threads_per_block = 256 ) const { return ParallelRunner_Gpu( stream, batch_sizes, max_nb_threads, threads_per_block ); }
    auto               run_single      ( auto &&func ) const { _gpu_run_single_kernel<<<1, 1, 0, stream>>>( func ); }
    void               run             ( auto batch_sizes, auto &&func ) const { _gpu_run_kernel<<<1, 1, 0, stream>>>( func, batch_sizes ); }
    void               run_parallel    ( auto batch_sizes, auto &&func ) const { _gpu_run_kernel<<<1, 1, 0, stream>>>( func, batch_sizes ); }
    __host__ void      zero_fill       ( void *ptr, PI n, PI elem_size ) const { cudaMemsetAsync( ptr, 0, n * elem_size, stream ); }
    static auto        raw_ptr         ( auto &&vec ) { return thrust::raw_pointer_cast( vec.data() ); }
    static const char *name            () { return "CudaGpu"; }

    cudaStream_t       stream;
    int                device_id;
};

} // namespace sdot

#endif
