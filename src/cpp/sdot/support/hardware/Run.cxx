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
    // _get_args_on makes each of (list, args...) accessible from `execution_space` (pass-through
    // or transfer), one at a time, rotating the made-accessible ones to the back; when none are
    // left it hands them to the trailing continuation `cont`. The continuations are functors (not
    // lambdas) so the whole chain is callable from device code too (nvcc rejects generic / by-ref
    // extended __host__ __device__ lambdas).

    // terminal: every item is accessible -> invoke the continuation with them
    HD void _get_args_on( auto /*execution_space*/, Ct<int,0>, auto &&cont, auto &&...args ) {
        cont( FORWARD( args )... );
    }

    template<int n> HD void _get_args_on( auto execution_space, Ct<int,n>, auto &&head, auto &&...tail );

    /// resume the rotation once `head` has been made accessible: continue with (tail..., head)
    template<class ES,int n,class Head>
    struct Resume {
        ES   execution_space;
        Head head;
        HD void operator()( auto &&...tail ) const {
            _get_args_on( execution_space, Ct<int,n>(), FORWARD( tail )..., head );
        }
    };

    /// make_accessible callback for `head`: keeps the remaining `tail` (as a Tuple) and resumes
    template<class ES,int n,class Tail>
    struct WhenAccessible {
        ES   execution_space;
        Tail tail;
        HD void operator()( auto &&head ) const {
            tail.apply_values( Resume<ES,n-1,DECAYED_TYPE_OF( head )>{ execution_space, FORWARD( head ) } );
        }
    };

    template<int n,class ES,class... Tail>
    HD auto when_accessible( ES execution_space, Tail &&...tail ) {
        auto t = tuple( FORWARD( tail )... );
        return WhenAccessible<ES,n,DECAYED_TYPE_OF( t )>{ execution_space, t };
    }

    template<int n> HD void _get_args_on( auto execution_space, Ct<int,n>, auto &&head, auto &&...tail ) {
        make_accessible( execution_space, FORWARD( head ), when_accessible<n>( execution_space, FORWARD( tail )... ) );
    }

    /// terminal continuation of run_parallel: dispatch onto the chosen execution space
    template<class ES,class Func>
    struct LaunchOn {
        ES    execution_space;
        Func &func;
        HD void operator()( auto &&list, auto &&...args ) const {
            execution_space.run_parallel( FORWARD( list ), func, FORWARD( args )... );
        }
    };

    // force max_cpu_threads to 1
    template<class Func>
    struct RunSequentialWrapper : RunTraits::RunFunctorWrapper<Func> {
        HD int max_cpu_threads( auto &&.../* args */ ) { return Ct<int,1>(); }
    };
} // namespace RunDetails

HD void run_parallel( auto &&list, auto &&func, auto &&...args ) {
    // statically chosen from the args' memory spaces (single type -> only this branch compiles)
    auto execution_space = execution_space_for( args... );

    // make every arg accessible from that space (pass-through or transfer), then run
    RunDetails::_get_args_on( execution_space, Ct<int,1+sizeof...( args )>(), FORWARD( list ), FORWARD( args )...,
        RunDetails::LaunchOn<DECAYED_TYPE_OF( execution_space ),DECAYED_TYPE_OF( func )>{ execution_space, func } );
}

HD void run_sequential( auto &&list, auto &&func, auto &&...args ) {
    run_parallel( FORWARD( list ), RunDetails::RunSequentialWrapper<DECAYED_TYPE_OF(func)>{ FORWARD( func ) }, FORWARD( args )... );
}

} // namespace sdot
