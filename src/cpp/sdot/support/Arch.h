#pragma once

#include "common_types.h"
#include <vector>

namespace sdot {
// forward declaration so run() body can reference AxisTuple before AxisTuple.h is fully parsed
template<class TI, class Arch, int ct_rank, class... Attrs> class AxisTuple;
} // namespace sdot

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#endif

namespace sdot {

// ------------------------------------------------------ CPU ------------------------------------------------------
struct Cpu {
    T_T struct         Vector          { using type = std::vector<T>; };

    T_T void           with_reservation( PI size, auto &&func ) const { T *res = new T[ size ]; func( res ); delete [] res; }
    auto               run_single      ( auto &&func ) { func(); }
    void               run             ( auto batch_sizes, auto &&func ) {
        using BT = std::decay_t<decltype( batch_sizes )>;
        using TI = typename BT::TI;
        constexpr int N = BT::ct_rank;
        batch_sizes.for_each_index( [&]( auto ...is ) {
            func( AxisTuple<TI,Cpu,N>( Values(), is... ) );
        } );
    }
    void               run_parallel    ( auto batch_sizes, auto &&func ) { run( batch_sizes, func ); }
    static auto        raw_ptr         ( auto &&vec ) { return vec.data(); }
    static const char* name            () { return "cpu"; }

};

// ------------------------------------------------------ GPU ------------------------------------------------------
#ifdef __CUDACC__
template<class F> __global__ void _gpu_run_single_kernel( F func ) { func(); }

template<class F, class BT>
__global__ void _gpu_run_kernel( F func, BT batch_sizes ) {
    using TI = typename BT::TI;
    constexpr int N = BT::ct_rank;
    batch_sizes.for_each_index( [&]( auto ...is ) {
        func( AxisTuple<TI,Gpu,N>( Values(), is... ) );
    } );
}

struct Gpu {
    explicit           Gpu             ( cudaStream_t s ) : stream( s ) { cudaGetDevice( &device_id ); }

    T_T struct         Vector          { using type = thrust::device_vector<T>; };

    T_T void           with_reservation( PI size, auto &&func ) const { T *res = nullptr; cudaMalloc( reinterpret_cast<void **>( &res ), size * sizeof( T ) ); func( res ); cudaFree( res ); }
    auto               run_single      ( auto &&func ) const { _gpu_run_single_kernel<<<1, 1, 0, stream>>>( func ); }
    void               run             ( auto batch_sizes, auto &&func ) const { _gpu_run_kernel<<<1, 1, 0, stream>>>( func, batch_sizes ); }
    void               run_parallel    ( auto batch_sizes, auto &&func ) const { _gpu_run_kernel<<<1, 1, 0, stream>>>( func, batch_sizes ); }
    static auto        raw_ptr         ( auto &&vec ) { return thrust::raw_pointer_cast( vec.data() ); }
    static const char *name            () { return "gpu"; }

    cudaStream_t       stream;
    int                device_id;
};
// #else
//     struct Gpu {
//         int device_id = 0;

//         T_T struct Vector { using type = std::vector<T>; };

//         T_T void with_reservation( PI size, auto &&func ) const { T *res = new T[ size ]; func( res ); delete [] res; }
//         static auto        raw_ptr( auto &&vec ) { return vec.data(); }
//         static const char *name   () { return "gpu"; }
//     };
#endif

} // namespace sdot
