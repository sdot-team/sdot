#pragma once

#ifdef __CUDACC__

#include "ExecutionSpace_Cuda.h"
#include "ExecutionSpace_Cpu.h"
#include "MemorySpace.h"
#include "../Ct.h"

namespace sdot {

/// pageable host memory
struct MemorySpace_GlobalCudaRam : MemorySpace {
    auto execution_space() const { return ExecutionSpace_Cuda{}; }
};

constexpr auto operator==( MemorySpace_GlobalCudaRam, MemorySpace_GlobalCudaRam ) { return Ct<bool,true>(); }
constexpr auto operator==( MemorySpace_GlobalCudaRam, auto                      ) { return Ct<bool,false>(); }
constexpr auto operator==( auto                     , MemorySpace_GlobalCudaRam ) { return Ct<bool,false>(); }

auto transfer_cost( ExecutionSpace_Cuda, MemorySpace_GlobalCudaRam, auto &&/*get_nb_bytes*/ ) { return Ct<int,0>(); }
auto transfer_cost( ExecutionSpace_Cpu, MemorySpace_GlobalCudaRam, auto &&/*get_nb_bytes*/ ) { return Ct<int,1>(); }

} // namespace sdot

#endif
