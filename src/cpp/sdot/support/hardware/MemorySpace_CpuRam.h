#pragma once

#include "ExecutionSpace_Cpu.h"
#include "MemorySpace.h"
#include "../Ct.h"
#include "Ptr.h"

namespace sdot {

/// pageable host memory
struct MemorySpace_CpuRam : MemorySpace {
};

constexpr auto operator==( MemorySpace_CpuRam, MemorySpace_CpuRam ) { return Ct<bool,true>(); }

auto transfer_cost( ExecutionSpace_Cpu, MemorySpace_CpuRam, auto &&/*get_nb_bytes*/ ) { return Ct<int,0>(); }

T_T CPU_ONLY inline void copy( Ptr<T,MemorySpace_CpuRam> dst, Ptr<T,MemorySpace_CpuRam> src, PI nb_items ) { std::memcpy( dst.raw, src.raw, nb_items * sizeof( T ) ); }

auto memory_space( const auto &value ) requires ( ! requires { value.memory_space(); } ) { return MemorySpace_CpuRam(); }

} // namespace sdot
