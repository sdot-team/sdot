#pragma once

#include "../containers/for_each_item_split.h"
#include "RunTraits.h"
#include "ExecutionSpace.h"
#include "WaitingThreads.h"
#include "CpuThreadInfo.h"

namespace sdot {

// host execution / host RAM — stateless, {}-constructible
struct ExecutionSpace_Cpu : ExecutionSpace {
    void run_parallel( const auto &list, auto &&func, auto &&...args ) {
        waiting_threads().run( [&]( int num_thread, int nb_threads ) {
            CpuThreadInfo thread_info{ num_thread, nb_threads };
            RunTraits::per_thread( func, thread_info, list, [&]( auto &&...args ) {
                for_each_item_split( list, num_thread, nb_threads, [&]( auto &&item ) {
                    func( item, FORWARD( args )... );
                } );
            }, FORWARD( args )... );
        }, RunTraits::max_cpu_threads( func, args... ) );
    }
};

} // namespace sdot

