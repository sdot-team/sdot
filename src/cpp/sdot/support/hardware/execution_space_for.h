#pragma once

#include "ExecutionContext_Cpu.h"
#ifdef __CUDACC__
#include "ExecutionContext_Cuda.h"
#endif

#include "transfer_cost.h" // IWYU pragma: keep

namespace sdot {

// ---------------------------------------------------------------------------
// execution_space_for( args... ) -> ExecutionContext_Cpu | ExecutionContext_Cuda
//
// Chooses at compile time the execution context with the lowest total transfer
// cost across all arguments.  The comparison is purely type-level: only the
// return TYPE of transfer_cost is inspected (via DECAYED_TYPE_OF),
// so runtime values of the args are never touched.
//
// Rules:
//   - From device code (__CUDA_ARCH__): always CUDA — no CPU dispatch possible.
//   - No CUDA build (__CUDACC__ absent): always CPU.
//   - CUDA build, host side:
//       cuda_cost < cpu_cost  → CUDA (GPU data avoids a transfer)
//       otherwise             → CPU  (tie or CPU-native data; current context is host)
// ---------------------------------------------------------------------------
HD auto execution_space_for( [[maybe_unused]] const auto &...args ) {
#ifdef __CUDA_ARCH__
    return ExecutionContext_Cuda{};
#elif defined( __CUDACC__ )
    constexpr int cpu_cost  = DECAYED_TYPE_OF( transfer_cost( ExecutionContext_Cpu {}, args... ) )::value;
    constexpr int cuda_cost = DECAYED_TYPE_OF( transfer_cost( ExecutionContext_Cuda{}, args... ) )::value;
    info( cpu_cost, cuda_cost );

    if constexpr ( cuda_cost < cpu_cost )
        return ExecutionContext_Cuda{};
    else
        return ExecutionContext_Cpu{};
#else
    return ExecutionContext_Cpu{};
#endif
}

} // namespace sdot
