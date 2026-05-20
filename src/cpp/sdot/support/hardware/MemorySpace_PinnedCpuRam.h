#pragma once

#ifdef __CUDACC__

#include "ExecutionSpace_Cuda.h"
#include "MemorySpace.h"
#include "../Ct.h"

namespace sdot {

/// pageable host memory
struct MemorySpace_PinnedCpuRam : MemorySpace {
};

constexpr auto operator==( MemorySpace_PinnedCpuRam, MemorySpace_PinnedCpuRam ) { return Ct<bool,true>(); }
constexpr auto operator==( MemorySpace_PinnedCpuRam, MemorySpace_CpuRam       ) { return Ct<bool,true>(); }
constexpr auto operator==( MemorySpace_CpuRam      , MemorySpace_PinnedCpuRam ) { return Ct<bool,true>(); }

auto transfer_cost( ExecutionSpace_Cuda, MemorySpace_PinnedCpuRam, auto &&/*get_nb_bytes*/ ) { return Ct<int,0>(); }
auto transfer_cost( ExecutionSpace_Cpu, MemorySpace_PinnedCpuRam, auto &&/*get_nb_bytes*/ ) { return Ct<int,0>(); }

} // namespace sdot

#endif
