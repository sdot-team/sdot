#pragma once

#include "ExecutionSpace_Cpu.h"
#include "MemorySpace.h"
#include "../Ct.h"

namespace sdot {

/// pageable host memory
struct MemorySpace_CpuRam : MemorySpace {
};

constexpr auto operator==( MemorySpace_CpuRam, MemorySpace_CpuRam ) { return Ct<bool,true>(); }

auto transfer_cost( ExecutionSpace_Cpu, MemorySpace_CpuRam, auto &&/*get_nb_bytes*/ ) { return Ct<int,0>(); }

} // namespace sdot
