#pragma once

#ifdef __CUDACC__

#include "ExecutionContext_Cuda.h"
#include "ExecutionContext_Cpu.h"
#include "accessible_from.h"
#include "MemorySpace.h"
#include "../Ct.h"

namespace sdot {

/// page-locked host memory: zero-copy reachable from both host and device
struct MemorySpace_PinnedCpuRam : MemorySpace {
};

constexpr auto operator==( MemorySpace_PinnedCpuRam, MemorySpace_PinnedCpuRam ) { return Ct<bool,true>(); }
constexpr auto operator==( MemorySpace_PinnedCpuRam, MemorySpace_CpuRam       ) { return Ct<bool,true>(); }
constexpr auto operator==( MemorySpace_CpuRam      , MemorySpace_PinnedCpuRam ) { return Ct<bool,true>(); }

auto accessible_from( ExecutionContext_Cuda, MemorySpace_PinnedCpuRam ) { return Ct<bool,true>(); }
auto accessible_from( ExecutionContext_Cpu , MemorySpace_PinnedCpuRam ) { return Ct<bool,true>(); }

} // namespace sdot

#endif
