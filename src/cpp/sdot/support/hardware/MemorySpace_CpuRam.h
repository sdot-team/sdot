#pragma once

#include "ExecutionContext_Cpu.h"
#include "MemorySpace.h"
#include "Ptr.h"

#include <cstring>

namespace sdot {

/// pageable host memory
struct MemorySpace_CpuRam : MemorySpace {
    /// allocate `n` items of T in this memory space, expose them (as an informed Ptr) to `func`, then release
    T_T CPU_ONLY auto with_reservation( PI n, auto &&func ) const { T *p = new T[ n ]; func( Ptr<T,MemorySpace_CpuRam>( p, *this ) ); delete [] p; }
    void         display( auto &os ) const { os << "CpuRam"; }
};

// functions
constexpr auto transfer_cost_per_byte( ExecutionContext_Cpu, MemorySpace_CpuRam ) { return 0_c; }
constexpr auto operator==            ( MemorySpace_CpuRam, MemorySpace_CpuRam ) { return Ct<bool,true>(); }
T_T T*         zero_for              ( MemorySpace_CpuRam /* memory_space */ ) { static T res = 0; return &res; }
T_T HD void    copy                  ( Ptr<T,MemorySpace_CpuRam> dst, Ptr<T,MemorySpace_CpuRam> src, PI nb_items ) { std::memcpy( dst.raw, src.raw, nb_items * sizeof( T ) ); }


} // namespace sdot
