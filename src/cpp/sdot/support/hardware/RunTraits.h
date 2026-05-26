#pragma once

#include "../containers/Tuple.h"
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

// ----------------- lambda-free item iteration (device-safe) -----------------
// nvcc forbids generic / by-reference extended __host__ __device__ lambdas, so the
// per-item application is expressed with functors (usable on host and device).

/// calls func( leading, vals... ) — used to prepend the current item to the stored args
template<class Func,class Leading>
struct CallWithLeading {
    Func    func;     ///< reference type (T&)
    Leading leading;
    HD void operator()( auto &&...vals ) const { func( leading, FORWARD( vals )... ); }
};

/// for_each_item callback applying func( item, args... ), args carried in a Tuple
template<class Func,class Args>
struct ApplyToItem {
    Func func;        ///< reference type (T&)
    Args args;        ///< Tuple of the trailing args
    HD void operator()( auto &&item ) const {
        args.apply_values( CallWithLeading<Func,DECAYED_TYPE_OF( item )>{ func, FORWARD( item ) } );
    }
};

/// build an ApplyToItem holding a reference to `func` and a copy of `args...`
HD auto apply_to_item( auto &func, auto &&...args ) {
    auto a = tuple( FORWARD( args )... );
    return ApplyToItem<DECAYED_TYPE_OF( func )&,DECAYED_TYPE_OF( a )>{ func, a };
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
