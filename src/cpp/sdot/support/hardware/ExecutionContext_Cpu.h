#pragma once

#include "../containers/for_each_item_split.h"
#include "../containers/nb_items.h"
#include "ExecutionContext.h"
#include "WaitingThreads.h"
#include "CpuThreadInfo.h"
#include "RunTraits.h"

namespace sdot {

struct MemorySpace_CpuRam;

// host execution
struct ExecutionContext_Cpu : ExecutionContext {
    using MemorySpace = MemorySpace_CpuRam;
    template<class L,class F,class... A>
    void run_parallel( const L &list, F &&func, A &&...args ) const {
        // run inline on a single thread: used for the nested case
        auto run_inline = [&]( int num_thread, int nb_threads ) {
            CpuThreadInfo thread_info{ num_thread, nb_threads };
            RunTraits::per_thread( func, thread_info, list, [&]( auto &&...args ) {
                for_each_item_split( list, num_thread, nb_threads, [&]( auto &&item ) {
                    func( item, FORWARD( args )... );
                } );
            }, FORWARD( args )... );
        };

        if ( is_pool_worker_thread ) {
            // runtime nesting: still must not call waiting_threads().run() (which would deadlock)
            run_inline( 0, 1 );
        } else {
            // top level: dispatch to the pool
            waiting_threads().run( [&]( int num_thread, int nb_threads ) {
                CpuThreadInfo thread_info{ num_thread, nb_threads };
                RunTraits::per_thread( func, thread_info, list, [&]( auto &&...args ) {
                    for_each_item_split( list, num_thread, nb_threads, [&]( auto &&item ) {
                        func( item, FORWARD( args )... );
                    } );
                }, FORWARD( args )... );
            }, std::min( nb_items( list ), PI( RunTraits::max_cpu_threads( func, args... ) ) ) );
        }
    }
};

} // namespace sdot


