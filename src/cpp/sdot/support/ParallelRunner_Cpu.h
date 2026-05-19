#pragma once

#include "WaitingThreads.h"

namespace sdot {
template<class TI, class Arch, int ct_rank, class... Attrs> class AxisTuple;

// ------------------------------------------------------ CPU ------------------------------------------------------
template<class BatchSizes>
struct ParallelRunner_Cpu {
    /* */ ParallelRunner_Cpu( BatchSizes batch_sizes, int max_nb_threads ) : waiting_threads( sdot::waiting_threads() ), batch_sizes( batch_sizes ), max_nb_threads( max_nb_threads ) {
    }

    void for_each_thread( auto &&thread_func ) { // thread_func will take args ( num_thread, callback ). callback must be a function that take batch_indices as parameters
        waiting_threads.run( [&]( int num_thread, int nb_threads ) {
            thread_func( num_thread, nb_threads, [&]( auto &&for_each_bi ) {
                batch_sizes.for_each_index_split( num_thread, nb_threads, for_each_bi );
            } );
        }, max_nb_threads );
    }

    WaitingThreads &waiting_threads;
    BatchSizes batch_sizes;
    int max_nb_threads;
};

} // namespace sdot
