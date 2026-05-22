#pragma once

#include "execution_space_for.h"
#include "make_accessible.h"
#include "Run.h"

#include "../common_types.h"

namespace sdot {


// #ifdef __CUDACC__
// template<class F, class BatchSizes, class... Acc>
// __global__ void _run_kernel( F func, BatchSizes list, Acc... acc ) {
//     GpuThreadInfo ti;
//     _run_thread_body( ti, [&] GD ( auto &&cb ) {
//         list.for_each_index_split( ti.global_id(), ti.nb_threads(), cb );
//     }, func, acc... );
// }
// #endif

// template<class Space, class BatchSizes, class F, class... Args>
//     requires ExecutionContext<Space>
// void _run( const Space &space, int max_nb_threads, const BatchSizes &list, F &&func, Args &&...args ) {
//     auto body = [&]( auto &&...acc ) {
//         if constexpr ( Space::runs_on_host ) {
//             ParallelRunner_Cpu pr( list, max_nb_threads );
//             pr.for_each_thread( [&]( auto ti, auto &&dispatch ) {
//                 _run_thread_body( ti, dispatch, func, acc... );
//             } );
//         }
// #ifdef __CUDACC__
//         else {
//             LaunchConfig lc = ( max_nb_threads == 1 )
//                 ? LaunchConfig{ 1, 1 }
//                 : compute_launch( list.nb_items(), space, func, acc... );
//             _run_kernel<<< lc.grid, lc.block, 0, space.stream >>>( func, list, acc... );
//         }
// #endif
//     };
//     body( make_accessible( space, std::forward<Args>( args ) )... );
// }
namespace RunDetails {
    // _get_args_on -> accesible args from execution_space
    void _get_args_on( auto /*execution_space*/, Ct<int,0>, auto &&func, auto &&...args ) {
        func( FORWARD( args )... );
    }
    template<int n> void _get_args_on( auto execution_space, Ct<int,n>, auto &&head, auto &&...tail ) {
        make_accessible( execution_space, FORWARD( head ), [&]( auto &&head ) {
            _get_args_on( execution_space, Ct<int,n-1>(), FORWARD( tail )..., FORWARD( head ) );
        } );
    }

    // force max_cpu_threads to 1
    template<class Func>
    struct RunSequentialWrapper : RunTraits::RunFunctorWrapper<Func> {
        int max_cpu_threads( auto &&.../* args */ ) { return Ct<int,1>(); }
    };
} // namespace RunDetails

void run_parallel( auto &&list, auto &&func, auto &&...args ) {
    // statically chosen from the args' memory spaces (single type -> only this branch compiles)
    auto execution_space = execution_space_for( args... );

    // make every arg accessible from that space (pass-through or transfer), then run
    RunDetails::_get_args_on( execution_space, Ct<int,1+sizeof...( args )>(), FORWARD( list ), FORWARD( args )..., [&]( auto &&list, auto &&...args ) {
        execution_space.run_parallel( FORWARD( list ), FORWARD( func ), FORWARD( args )... );
    } );
}

void run_sequential( auto &&list, auto &&func, auto &&...args ) {
    run_parallel( FORWARD( list ), RunDetails::RunSequentialWrapper<DECAYED_TYPE_OF(func)>{ FORWARD( func ) }, FORWARD( args )... );
}

} // namespace sdot
