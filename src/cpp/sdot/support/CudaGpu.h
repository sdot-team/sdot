#pragma once

#ifdef __CUDACC__

#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include "common_types.h"
#include <vector>

namespace sdot {
template<class TI, class Arch, int ct_rank, class... Attrs> class AxisTuple;

// ------------------------------------------------------ GPU ------------------------------------------------------
template<class F> __global__ void _gpu_run_single_kernel( F func ) { func(); }

template<class F, class BT>
__global__ void _gpu_run_kernel( F func, BT batch_sizes ) {
    using TI = typename BT::TI;
    constexpr int N = BT::ct_rank;
    batch_sizes.for_each_index( [&]( auto ...is ) {
        func( AxisTuple<TI,Gpu,N>( Values(), is... ) );
    } );
}

struct CudaGpu {
    explicit           CudaGpu             ( cudaStream_t s ) : stream( s ) { cudaGetDevice( &device_id ); }

    T_T struct         Vector          { using type = thrust::device_vector<T>; };

    T_T void           with_reservation( PI size, auto &&func ) const { T *res = nullptr; cudaMalloc( reinterpret_cast<void **>( &res ), size * sizeof( T ) ); func( res ); cudaFree( res ); }
    auto               run_single      ( auto &&func ) const { _gpu_run_single_kernel<<<1, 1, 0, stream>>>( func ); }
    void               run             ( auto batch_sizes, auto &&func ) const { _gpu_run_kernel<<<1, 1, 0, stream>>>( func, batch_sizes ); }
    void               run_parallel    ( auto batch_sizes, auto &&func ) const { _gpu_run_kernel<<<1, 1, 0, stream>>>( func, batch_sizes ); }
    static auto        raw_ptr         ( auto &&vec ) { return thrust::raw_pointer_cast( vec.data() ); }
    static const char *name            () { return "CudaGpu"; }

    cudaStream_t       stream;
    int                device_id;
};

} // namespace sdot

#endif
