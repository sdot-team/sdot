#pragma once

#include "execution_space_for.h"
#include "make_accessible.h"
#include "Run.h"

#include "../common_types.h"

namespace sdot {

namespace RunDetails {
    // force max_cpu_threads to 1
    template<class Func>
    struct RunSequentialWrapper : RunTraits::RunFunctorWrapper<Func> {
        T_VA HD int max_cpu_threads( A &&.../* args */ ) { return 1_c; }
        T_VA HD int max_gpu_threads( A &&.../* args */ ) { return 1_c; }
    };

    // decl
    template<class ES,class F,class L,class... A>        HD void run_parallel_from( const ES &execution_space, Ct<int,0>, F &&func, L &&list, A &&...args );
    template<int n,class ES,class F,class H,class... A>  HD void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, H &&head, A &&...tail );
    template<int n,class ES,class F,class H,class... A>  HD void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, Inp, H &&head, A &&...tail );
    template<int n,class ES,class F,class H,class... A>  HD void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, Out, H &&head, A &&...tail );
    template<int n,class ES,class F,class H,class... A>  HD void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, Mut, H &&head, A &&...tail );

    // end
    template<class ES,class F,class L,class... A> HD void run_parallel_from( const ES &execution_space, Ct<int,0>, F &&func, L &&list, A &&...args ) {
        execution_space.run_parallel( FORWARD( list ), func, FORWARD( args )... );
    }

    // Inp
    template<int n,class ES,class F,class H,class... A> HD void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, Inp, H &&head, A &&...tail ) {
        make_accessible( execution_space, FORWARD( head ), 1_b, 0_b, [&]( auto &&head ) {
            run_parallel_from( execution_space, cn - 2_c, FORWARD( func ), FORWARD( tail )..., FORWARD( head ) );
        } );
    }

    // Out
    template<int n,class ES,class F,class H,class... A> HD void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, Out, H &&head, A &&...tail ) {
        make_accessible( execution_space, FORWARD( head ), 0_b, 1_b, [&]( auto &&head ) {
            run_parallel_from( execution_space, cn - 2_c, FORWARD( func ), FORWARD( tail )..., FORWARD( head ) );
        } );
    }

    // Mut
    template<int n,class ES,class F,class H,class... A> HD void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, Mut, H &&head, A &&...tail ) {
        make_accessible( execution_space, FORWARD( head ), 1_b, 1_b, [&]( auto &&head ) {
            run_parallel_from( execution_space, cn - 2_c, FORWARD( func ), FORWARD( tail )..., FORWARD( head ) );
        } );
    }

    // raw
    template<int n,class ES,class F,class H,class... A> HD void run_parallel_from( const ES &execution_space, Ct<int,n> cn, F &&func, H &&head, A &&...tail ) {
        make_accessible( execution_space, FORWARD( head ), 0_b, 0_b, [&]( auto &&head ) {
            run_parallel_from( execution_space, cn - 1_c, FORWARD( func ), FORWARD( tail )..., FORWARD( head ) );
        } );
    }
} // namespace RunDetails

template<class L,class F,class... A> HD void run_parallel( L &&list, F &&func, A &&...args ) {
    // statically chosen from the args memory spaces (single type -> only this branch compiles)
    auto execution_space = execution_space_for( args... );

    //
    RunDetails::run_parallel_from( execution_space, Ct<int,2+sizeof...( args )>(), func, Inp(), FORWARD( list ), FORWARD( args )... );

    // make every arg accessible from that space (pass-through or transfer), then run
    // RunDetails::_get_args_on( execution_space, Ct<int,1+sizeof...( args )>(), FORWARD( list ), FORWARD( args )...,
    //     RunDetails::LaunchOn<DECAYED_TYPE_OF( execution_space ),DECAYED_TYPE_OF( func )>{ execution_space, func }
    // );
}

template<class L,class F,class... A> HD void run_sequential( L &&list, F &&func, A &&...args ) {
    run_parallel( FORWARD( list ), RunDetails::RunSequentialWrapper<DECAYED_TYPE_OF(func)>{ FORWARD( func ) }, FORWARD( args )... );
}

} // namespace sdot
