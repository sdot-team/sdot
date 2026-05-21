#pragma once

#include "../common_macros.h" // DECAYED_TYPE_OF
#include "accessible_from.h"
#include "ExecutionSpace_Cpu.h"
#ifdef __CUDACC__
#include "ExecutionSpace_Cuda.h"
#endif

namespace sdot {

// Compile-time accessibility of a memory space from the host / the device, derived from the
// single accessible_from matrix. Used to qualify data accessors as __host__- or __device__-only
// (see TensorView): an element accessor exists in a given context iff the data is reachable
// there, so dereferencing in the wrong context is a compile error rather than UB.
template<class MemorySpace>
constexpr bool host_accessible_v = DECAYED_TYPE_OF( accessible_from( ExecutionSpace_Cpu{}, MemorySpace{} ) )::value;

#ifdef __CUDACC__
template<class MemorySpace>
constexpr bool device_accessible_v = DECAYED_TYPE_OF( accessible_from( ExecutionSpace_Cuda{}, MemorySpace{} ) )::value;
#endif

} // namespace sdot
