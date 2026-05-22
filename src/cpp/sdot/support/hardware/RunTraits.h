#pragma once

#include "../common_macros.h"
#include "../Ct.h"
#include <thread>

namespace sdot {

namespace RunTraits {

// ----------------- generic -----------------

/// add args computed for each thread
HD void per_thread( auto &func, auto &&thread_info, auto &&list, auto &&cont, auto &&...args ) {
    if constexpr( requires{ func.per_thread( FORWARD( thread_info ), FORWARD( list ), FORWARD( cont ), FORWARD( args )... ); } )
        func.per_thread( FORWARD( thread_info ), FORWARD( list ), FORWARD( cont ), FORWARD( args )... );
    else
        cont( FORWARD( args )... ); //
}

// ----------------- CPU -----------------

HD int max_cpu_threads( auto &&func, auto &&...args ) {
    if constexpr( requires{ func.max_cpu_threads( args... ); } )
        return func.max_cpu_threads( args... );
    else
        return std::thread::hardware_concurrency();
}

// ----------------- GPU -----------------

HD int local_gpu_memory_size( auto &&func, auto &&...args ) {
    if constexpr( requires{ func.local_gpu_memory_size( args... ); } )
        return func.local_gpu_memory_size( args... );
    else
        return Ct<int,1>();
}

HD int nb_gpu_register_per_thread( auto &&func, auto &&...args ) {
    if constexpr( requires{ func.nb_gpu_register_per_thread( args... ); } )
        return func.nb_gpu_register_per_thread( args... );
    else
        return Ct<int,16>();
}

HD int max_gpu_threads( auto &&func, auto &&...args ) {
    if constexpr( requires{ func.max_gpu_threads( args... ); } )
        return func.max_gpu_threads( args... );
    else
        // simple default to start; TODO: derive from nb_gpu_register_per_thread / local_gpu_memory_size
        return 1 << 20;
}

// ----------------- util -----------------

/// Wrap a func, allowing to surdefine methods
template<class Func>
struct RunFunctorWrapper {
    HD int  nb_gpu_register_per_thread( auto &&...args ) { return sdot::RunTraits::nb_gpu_register_per_thread( func, FORWARD( args )... ); }
    HD int  local_gpu_memory_size     ( auto &&...args ) { return sdot::RunTraits::local_gpu_memory_size( func, FORWARD( args )... ); }
    HD int  max_gpu_threads           ( auto &&...args ) { return sdot::RunTraits::max_gpu_threads( func, FORWARD( args )... ); }

    HD int  max_cpu_threads           ( auto &&...args ) { return sdot::RunTraits::max_cpu_threads( func, FORWARD( args )... ); }

    HD void per_thread                ( auto &&thread_info, auto &&list, auto &&cont, auto &&...args ) { return sdot::RunTraits::per_thread( func, FORWARD( thread_info ), FORWARD( list ), FORWARD( cont ), FORWARD( args )... ); }

    HD auto operator()                ( auto &&...args ) const { func( FORWARD( args )... ); }

    Func func;
};

} // RunTraits

} // namespace sdot
