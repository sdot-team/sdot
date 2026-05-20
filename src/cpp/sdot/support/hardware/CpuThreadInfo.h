#pragma once

//#include "WaitingThreads.h"

namespace sdot {

struct CpuThreadInfo {
    int global_id()         const { return _id; }
    int nb_threads()        const { return _nb; }
    int local_id()          const { return _id; }
    int block_id()          const { return _id; }
    int threads_per_block() const { return 1; }
    int _id, _nb;
};

// // ------------------------------------------------------ CPU ------------------------------------------------------
// template<class BatchSizes>
// struct ParallelRunner_Cpu {
//     /* */ ParallelRunner_Cpu( BatchSizes batch_sizes, int max_nb_threads ) : waiting_threads( sdot::waiting_threads() ), batch_sizes( batch_sizes ), max_nb_threads( max_nb_threads ) {
//     }

//     void for_each_thread( auto &&thread_func ) {
//         waiting_threads.run( [&]( int num_thread, int nb_threads ) {
//             auto ti = CpuThreadInfo{ num_thread, nb_threads };
//             thread_func( ti, [&]( auto &&for_each_bi ) {
//                 batch_sizes.for_each_index_split( num_thread, nb_threads, for_each_bi );
//             } );
//         }, max_nb_threads );
//     }

//     WaitingThreads &waiting_threads;
//     BatchSizes batch_sizes;
//     int max_nb_threads;
// };

} // namespace sdot
