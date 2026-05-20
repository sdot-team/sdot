#pragma once

#include "for_each_possible_ExecutionSpace.h"
#include "approximate_execution_cost.h"
#include "make_accessible.h"
// #include "ParallelRunner_Cpu.h"
#include "transfer_cost.h"
#include "LaunchPolicy.h"
#include <type_traits>
#include <utility>
#include "Run.h"

namespace sdot {

// // One activated thread's work. `continuation( call_args... )` loops over this thread's
// // share of indices, invoking operator()( index, call_args... ).
// //
// // If the functor exposes per_thread( thread_info, continuation, args... ), it is called
// // once per activated thread and takes over: it sets up per-thread variables (whose
// // lifetime it controls — RAII) and calls `continuation` with the incoming args plus any
// // extra arguments it wants to inject. Default (no per_thread): continuation is called
// // directly with the accessible args.
// //
// // `for_each` runs a per-index callback over this thread's indices; inner callbacks are HD
// // so the same helper serves host threads and GPU device threads.
// template<class ThreadInfo, class ForEach, class F, class... Acc>
// HD void _run_thread_body( const ThreadInfo &ti, ForEach &&for_each, F &&func, Acc &&...acc ) {
//     auto continuation = [&] HD ( auto &&...call_args ) {
//         for_each( [&] HD ( auto bi ) { func( bi, call_args... ); } );
//     };
//     if constexpr ( requires { func.per_thread( ti, continuation, acc... ); } )
//         func.per_thread( ti, continuation, acc... );
//     else
//         continuation( acc... );
// }

// #ifdef __CUDACC__
// template<class F, class BatchSizes, class... Acc>
// __global__ void _run_kernel( F func, BatchSizes batch_sizes, Acc... acc ) {
//     GpuThreadInfo ti;
//     _run_thread_body( ti, [&] GD ( auto &&cb ) {
//         batch_sizes.for_each_index_split( ti.global_id(), ti.nb_threads(), cb );
//     }, func, acc... );
// }
// #endif

// template<class Space, class BatchSizes, class F, class... Args>
//     requires ExecutionSpace<Space>
// void _run( const Space &space, int max_nb_threads, const BatchSizes &batch_sizes, F &&func, Args &&...args ) {
//     auto body = [&]( auto &&...acc ) {
//         if constexpr ( Space::runs_on_host ) {
//             ParallelRunner_Cpu pr( batch_sizes, max_nb_threads );
//             pr.for_each_thread( [&]( auto ti, auto &&dispatch ) {
//                 _run_thread_body( ti, dispatch, func, acc... );
//             } );
//         }
// #ifdef __CUDACC__
//         else {
//             LaunchConfig lc = ( max_nb_threads == 1 )
//                 ? LaunchConfig{ 1, 1 }
//                 : compute_launch( batch_sizes.nb_items(), space, func, acc... );
//             _run_kernel<<< lc.grid, lc.block, 0, space.stream >>>( func, batch_sizes, acc... );
//         }
// #endif
//     };
//     body( make_accessible( space, std::forward<Args>( args ) )... );
// }

template<int n> void _get_args_on( auto execution_space, Ct<int,n>, auto &&head, auto &&...tail ) {
    make_accessible( execution_space, FORWARD( head ), [&]( auto &&head ) {
        _get_args_on( execution_space, Ct<int,n-1>(), FORWARD( tail )..., FORWARD( head ) );
    } );
}
void _get_args_on( auto /*execution_space*/, Ct<int,0>, auto &&func, auto &&...args ) {
    func( FORWARD( args )... );
}


void run_parallel( const auto &batch_sizes, auto &&func, auto &&...args ) {
    // prepare a cost list
    constexpr PI ne = decltype( nb_possible_ExecutionSpace() )::value;
    double cost_for_each_execution_space[ ne ];
    for( PI num_execution_space = 0; num_execution_space < ne; ++num_execution_space )
        cost_for_each_execution_space[ num_execution_space ] = 0;

    // get transfer cost
    auto add_cost_for_arg = [&]( const auto &arg ) {
        PI num_execution_space = 0;
        for_each_possible_ExecutionSpace( [&]( auto execution_space ) {
            cost_for_each_execution_space[ num_execution_space++ ] = transfer_cost( execution_space, arg );
        } );
    };
    ( add_cost_for_arg( args ), ... );

    // add approximate execution cost
    PI num_execution_space = 0;
    for_each_possible_ExecutionSpace( [&]( auto execution_space ) {
        cost_for_each_execution_space[ num_execution_space++ ] = approximate_execution_cost( execution_space, batch_sizes, func, args... );
    } );

    // find best_execution_space
    PI best_execution_space = 0;
    for( PI num_execution_space = 1; num_execution_space < ne; ++num_execution_space )
        if ( cost_for_each_execution_space[ best_execution_space ] > cost_for_each_execution_space[ num_execution_space ] )
        best_execution_space = num_execution_space;

    // exec on best_execution_space
    num_execution_space = 0;
    for_each_possible_ExecutionSpace( [&]( auto execution_space ) {
        if ( num_execution_space++ != best_execution_space )
            return;
        _get_args_on( execution_space, Ct<int,sizeof...( args )>(), FORWARD( args )..., [&]( auto &&...args ) {
            //_run_parallel_on( execution_space, batch_sizes, FORWARD( func ), FORWARD( args )... );
            func( args... );
        } );
    } );
}

// --- run_sequential ---
// template<class Space, class BatchSizes, class F, class... Args>
//     requires ExecutionSpace<Space>
// void run_sequential( const Space &space, const BatchSizes &batch_sizes, F &&func, Args &&...args ) {
//     _run( space, 1, batch_sizes, std::forward<F>( func ), std::forward<Args>( args )... );
// }

// template<class BatchSizes, class F, class... Args>
// void run_sequential( const BatchSizes &batch_sizes, F &&func, Args &&...args ) {
//     // #ifdef __CUDACC__
//     //     if constexpr ( any_arg_needs_device<Args...>() )
//     //         run_sequential( CudaSpace{}, batch_sizes, std::forward<F>( func ), std::forward<Args>( args )... );
//     //     else
//     // #endif
//     //         run_sequential( CpuExecutionSpace{}, batch_sizes, std::forward<F>( func ), std::forward<Args>( args )... );
// }

} // namespace sdot
