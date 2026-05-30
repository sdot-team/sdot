#pragma once

#include "execution_space_for.h"
#include "make_accessible.h"
#include "Run.h"

#include "../common_types.h"

namespace sdot {

namespace RunDetails {
    // _get_args_on makes each of (list, args...) accessible from `execution_space` (pass-through
    // or transfer), one at a time, rotating the made-accessible ones to the back; when none are
    // left it hands them to the trailing continuation `cont`. The continuations are functors (not
    // lambdas) so the whole chain is callable from device code too (nvcc rejects generic / by-ref
    // extended __host__ __device__ lambdas).

    // terminal: every item is accessible -> invoke the continuation with them
    // HD void _get_args_on( auto /*execution_space*/, Ct<int,0>, auto &&cont, auto &&...args ) {
    //     cont( FORWARD( args )... );
    // }

    // template<int n> HD void _get_args_on( auto execution_space, Ct<int,n>, auto &&head, auto &&...tail );

    // /// resume the rotation once `head` has been made accessible: continue with (tail..., head)
    // template<class ES,int n,class Head>
    // struct Resume {
    //     ES   execution_space;
    //     Head head;
    //     HD void operator()( auto &&...tail ) const {
    //         _get_args_on( execution_space, Ct<int,n>(), FORWARD( tail )..., head );
    //     }
    // };

    // /// make_accessible callback for `head`: keeps the remaining `tail` (as a Tuple) and resumes
    // template<class ES,int n,class Tail>
    // struct WhenAccessible {
    //     ES   execution_space;
    //     Tail tail;
    //     HD void operator()( auto &&head ) const {
    //         tail.apply_values( Resume<ES,n-1,DECAYED_TYPE_OF( head )>{ execution_space, FORWARD( head ) } );
    //     }
    // };

    // template<int n,class ES,class... Tail>
    // HD auto when_accessible( ES execution_space, Tail &&...tail ) {
    //     auto t = tuple( FORWARD( tail )... );
    //     return WhenAccessible<ES,n,DECAYED_TYPE_OF( t )>{ execution_space, t };
    // }

    // template<int n> HD void _get_args_on( auto execution_space, Ct<int,n>, auto &&head, auto &&...tail ) {
    //     make_accessible( execution_space, FORWARD( head ), when_accessible<n>( execution_space, FORWARD( tail )... ) );
    // }

    // /// terminal continuation of run_parallel: dispatch onto the chosen execution space
    // template<class ES,class Func>
    // struct LaunchOn {
    //     ES    execution_space;
    //     Func &func;
    //     HD void operator()( auto &&list, auto &&...args ) const {
    //         execution_space.run_parallel( FORWARD( list ), func, FORWARD( args )... );
    //     }
    // };

    // force max_cpu_threads to 1
    template<class Func>
    struct RunSequentialWrapper : RunTraits::RunFunctorWrapper<Func> {
        HD int max_cpu_threads( auto &&.../* args */ ) { return 1_c; }
    };

    // decl
    HD void run_parallel_from( const auto &execution_space, Ct<int,0>, auto &&func, auto &&list, auto &&...args );
    template<int n> HD void run_parallel_from( const auto &execution_space, Ct<int,n> cn, auto &&func, auto &&head, auto &&...tail );
    template<int n> HD void run_parallel_from( const auto &execution_space, Ct<int,n> cn, auto &&func, Inp, auto &&head, auto &&...tail );
    template<int n> HD void run_parallel_from( const auto &execution_space, Ct<int,n> cn, auto &&func, Out, auto &&head, auto &&...tail );
    template<int n> HD void run_parallel_from( const auto &execution_space, Ct<int,n> cn, auto &&func, Mut, auto &&head, auto &&...tail );

    // end
    HD void run_parallel_from( const auto &execution_space, Ct<int,0>, auto &&func, auto &&list, auto &&...args ) {
        execution_space.run_parallel( FORWARD( list ), func, FORWARD( args )... );
    }

    // Inp
    template<int n> HD void run_parallel_from( const auto &execution_space, Ct<int,n> cn, auto &&func, Inp, auto &&head, auto &&...tail ) {
        make_accessible( execution_space, FORWARD( head ), 1_b, 0_b, [&]( auto &&head ) {
            run_parallel_from( execution_space, cn - 2_c, FORWARD( func ), FORWARD( tail )..., FORWARD( head ) );
        } );
    }

    // Out
    template<int n> HD void run_parallel_from( const auto &execution_space, Ct<int,n> cn, auto &&func, Out, auto &&head, auto &&...tail ) {
        make_accessible( execution_space, FORWARD( head ), 0_b, 1_b, [&]( auto &&head ) {
            run_parallel_from( execution_space, cn - 2_c, FORWARD( func ), FORWARD( tail )..., FORWARD( head ) );
        } );
    }

    // Mut
    template<int n> HD void run_parallel_from( const auto &execution_space, Ct<int,n> cn, auto &&func, Mut, auto &&head, auto &&...tail ) {
        make_accessible( execution_space, FORWARD( head ), 1_b, 1_b, [&]( auto &&head ) {
            run_parallel_from( execution_space, cn - 2_c, FORWARD( func ), FORWARD( tail )..., FORWARD( head ) );
        } );
    }

    // raw
    template<int n> HD void run_parallel_from( const auto &execution_space, Ct<int,n> cn, auto &&func, auto &&head, auto &&...tail ) {
        make_accessible( execution_space, FORWARD( head ), 0_b, 0_b, [&]( auto &&head ) {
            run_parallel_from( execution_space, cn - 1_c, FORWARD( func ), FORWARD( tail )..., FORWARD( head ) );
        } );
    }
} // namespace RunDetails

HD void run_parallel( auto &&list, auto &&func, auto &&...args ) {
    // statically chosen from the args memory spaces (single type -> only this branch compiles)
    auto execution_space = execution_space_for( args... );

    //
    RunDetails::run_parallel_from( execution_space, Ct<int,2+sizeof...( args )>(), func, Inp(), FORWARD( list ), FORWARD( args )... );

    // make every arg accessible from that space (pass-through or transfer), then run
    // RunDetails::_get_args_on( execution_space, Ct<int,1+sizeof...( args )>(), FORWARD( list ), FORWARD( args )...,
    //     RunDetails::LaunchOn<DECAYED_TYPE_OF( execution_space ),DECAYED_TYPE_OF( func )>{ execution_space, func }
    // );
}

HD void run_sequential( auto &&list, auto &&func, auto &&...args ) {
    run_parallel( FORWARD( list ), RunDetails::RunSequentialWrapper<DECAYED_TYPE_OF(func)>{ FORWARD( func ) }, FORWARD( args )... );
}

} // namespace sdot
