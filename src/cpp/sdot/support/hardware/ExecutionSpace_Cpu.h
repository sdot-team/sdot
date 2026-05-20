#pragma once

#include "RunFunctorTraits.h"
#include "ExecutionSpace.h"
#include "WaitingThreads.h"
#include "CpuThreadInfo.h"

namespace sdot {

// host execution / host RAM — stateless, {}-constructible
struct ExecutionSpace_Cpu : ExecutionSpace {
    void run_parallel( const auto &list, auto &&func, auto &&...args ) {
        waiting_threads().run( [&]( int num_thread, int nb_threads ) {
            CpuThreadInfo thread_info{ num_thread, nb_threads };
            RunFunctorTraits::per_thread( func, thread_info, list, [&]( auto &&...args ) {
                list.for_each_index_split( num_thread, nb_threads, [&]( auto &&item ) {
                    func( item, FORWARD( args )... );
                } );
            }, FORWARD( args )... );
        }, RunFunctorTraits::max_cpu_threads( func, args... ) );
    }
};

} // namespace sdot

