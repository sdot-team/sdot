#pragma once

#include "ExecutionSpace_Cpu.h"
#include "accessible_from.h" // IWYU pragma: export
#include "MemorySpace.h"
#include "Ptr.h"

#include <cstring>

namespace sdot {

/// pageable host memory
struct MemorySpace_CpuRam : MemorySpace {
    /// allocate `n` items of T in this memory space, expose them (as an informed Ptr) to `func`, then release
    T_T HD auto with_reservation( PI n, auto &&func ) const { T *p = new T[ n ]; func( Ptr<T,MemorySpace_CpuRam>( p, *this ) ); delete [] p; }
};

constexpr auto operator==( MemorySpace_CpuRam, MemorySpace_CpuRam ) { return Ct<bool,true>(); }

auto accessible_from( ExecutionSpace_Cpu, MemorySpace_CpuRam ) { return Ct<bool,true>(); }

/// memory space a CPU execution space allocates into when it must materialize data
auto native_memory_space( ExecutionSpace_Cpu ) { return MemorySpace_CpuRam{}; }

T_T HD void copy( Ptr<T,MemorySpace_CpuRam> dst, Ptr<T,MemorySpace_CpuRam> src, PI nb_items ) { std::memcpy( dst.raw, src.raw, nb_items * sizeof( T ) ); }

auto memory_space( const auto &value ) requires ( ! requires { value.memory_space(); } ) { return MemorySpace_CpuRam(); }

} // namespace sdot
