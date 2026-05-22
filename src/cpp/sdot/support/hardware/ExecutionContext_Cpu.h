#pragma once

#include "../containers/for_each_item_split.h"
#include "../containers/nb_items.h"
#include "ExecutionContext.h"
#include "WaitingThreads.h"
#include "CpuThreadInfo.h"
#include "RunTraits.h"

namespace sdot {

// host execution
struct ExecutionContext_Cpu : ExecutionContext {
    void run_parallel( const auto &list, auto &&func, auto &&...args ) {
       waiting_threads().run( [&]( int num_thread, int nb_threads ) {
            CpuThreadInfo thread_info{ num_thread, nb_threads };
            RunTraits::per_thread( func, thread_info, list, [&]( auto &&...args ) {
                for_each_item_split( list, num_thread, nb_threads, [&]( auto &&item ) {
                    func( item, FORWARD( args )... );
                } );
            }, FORWARD( args )... );
        }, std::min( nb_items( list ), PI( RunTraits::max_cpu_threads( func, args... ) ) ) );
    }
};

} // namespace sdot

