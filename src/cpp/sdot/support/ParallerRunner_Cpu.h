#pragma once

#include "WaitingThreads.h"

namespace sdot {
template<class TI, class Arch, int ct_rank, class... Attrs> class AxisTuple;

// ------------------------------------------------------ CPU ------------------------------------------------------
template<class BatchSizes>
struct ParallelRunner_Cpu {
    /* */ ParallelRunner_Cpu( BatchSizes batch_sizes, std::size_t /*nb_bytes_per_thread*/ ) : waiting_threads( sdot::waiting_threads() ), batch_sizes( batch_sizes ) {
        nb_threads = waiting_threads.nb_threads; // std::min( unsigned(  ), std::thread::hardware_concurrency() );
    }

    void for_each_thread( auto &&thread_func ) { // thread_func will take args ( num_thread, callback ). callback must be a function that take batch_indices as parameters
        waiting_threads.run( [&]( int num_thread ) {
            thread_func( num_thread, [&]( auto &&for_each_bi ) {
                batch_sizes.for_each_index_split( num_thread, nb_threads, for_each_bi );
            } );
        }, nb_threads );
    }

    WaitingThreads &waiting_threads;
    BatchSizes batch_sizes;
    int nb_threads;
};

} // namespace sdot
