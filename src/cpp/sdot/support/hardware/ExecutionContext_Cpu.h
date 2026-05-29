#pragma once

#include "../containers/for_each_item_split.h"
#include "../containers/container_tags.h"
#include "../containers/nb_items.h"
#include "ExecutionContext.h"
#include "WaitingThreads.h"
#include "CpuThreadInfo.h"
#include "RunTraits.h"

namespace sdot {

// host execution
struct ExecutionContext_Cpu : ExecutionContext {
    void run_parallel( const auto &list, auto &&func, auto &&...args ) const {
        // Are we already inside a parallel region? (statically known: at least one operand
        // carries the has_already_been_parallelized tag.) If so, run inline on the current
        // thread: dispatching to the pool here would deadlock (a worker would wait on itself).
        constexpr bool nested = ( false || ... || is_already_parallelized<DECAYED_TYPE_OF( args )>() );

        // run inline on a single thread: used for the nested case (static or runtime)
        auto run_inline = [&]( int num_thread, int nb_threads ) {
            CpuThreadInfo thread_info{ num_thread, nb_threads };
            RunTraits::per_thread( func, thread_info, list, [&]( auto &&...args ) {
                for_each_item_split( list, num_thread, nb_threads, [&]( auto &&item ) {
                    func( item, FORWARD( args )... );
                } );
            }, FORWARD( args )... );
        };

        if constexpr ( nested ) {
            // statically known nesting (tag propagated): stay on current thread
            run_inline( 0, 1 );
        } else if ( is_pool_worker_thread ) {
            // runtime nesting: tag did not propagate (e.g. fresh TensorView created
            // inside the worker body) — still must not call waiting_threads().run()
            run_inline( 0, 1 );
        } else {
            // top level: dispatch to the pool, and mark the operands the body sees as
            // has_already_been_parallelized so any run_*() they spawn stays inline
            waiting_threads().run( [&]( int num_thread, int nb_threads ) {
                CpuThreadInfo thread_info{ num_thread, nb_threads };
                RunTraits::per_thread( func, thread_info, list, [&]( auto &&...args ) {
                    for_each_item_split( list, num_thread, nb_threads, [&]( auto &&item ) {
                        func( item, FORWARD( args )... );
                    } );
                }, as_already_parallelized( FORWARD( args ) )... );
            }, std::min( nb_items( list ), PI( RunTraits::max_cpu_threads( func, args... ) ) ) );
        }
    }
};

} // namespace sdot

