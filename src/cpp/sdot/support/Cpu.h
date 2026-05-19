#pragma once

#include "ParallelRunner_Cpu.h"
#include "common_types.h"

namespace sdot {
template<class TI, class Arch, int ct_rank, class... Attrs> class AxisTuple;

// ------------------------------------------------------ CPU ------------------------------------------------------
struct Cpu {
    T_T void           with_reservation( PI size, auto &&func ) const { T *res = new T[ size ]; func( res ); delete [] res; }
    auto               parallel_runner ( auto batch_sizes, int max_nb_threads ) { return ParallelRunner_Cpu( batch_sizes, max_nb_threads ); }
    auto               run_single      ( auto &&func ) { func(); }

    void               run             ( auto batch_sizes, auto &&func ) {
        using BT = std::decay_t<decltype( batch_sizes )>;
        using TI = typename BT::TI;
        constexpr int N = BT::ct_rank;
        batch_sizes.for_each_index( [&]( auto ...is ) {
            func( AxisTuple<TI,Cpu,N>( Values(), is... ) );
        } );
    }

    void               run_parallel    ( auto batch_sizes, auto &&func ) { run( batch_sizes, func ); }
    static const char* name            () { return "cpu"; }
};

} // namespace sdot
