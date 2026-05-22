#pragma once

#include "ExecutionContext_Cpu.h"
#ifdef __CUDACC__
#include "ExecutionContext_Cuda.h"
#endif

namespace sdot {

// ---------------------------------------------------------------------------
// Static choice of the execution space for run_*().
//
// The space is decided at compile time from the arguments' memory spaces — no runtime cost
// model. Each argument "pulls" toward the execution space native to its memory space (data
// living on the device pulls toward CUDA); host-resident / scalar args stay neutral (CPU).
// `promote` combines them, the device winning (host args are then transferred by
// make_accessible). Because the result is a single type, only the chosen branch is ever
// instantiated — so e.g. a host-only call never compiles a GPU kernel.
// ---------------------------------------------------------------------------

inline auto promote( ExecutionContext_Cpu, ExecutionContext_Cpu ) { return ExecutionContext_Cpu{}; }
#ifdef __CUDACC__
inline auto promote( ExecutionContext_Cpu , ExecutionContext_Cuda ) { return ExecutionContext_Cuda{}; }
inline auto promote( ExecutionContext_Cuda, ExecutionContext_Cpu  ) { return ExecutionContext_Cuda{}; }
inline auto promote( ExecutionContext_Cuda, ExecutionContext_Cuda ) { return ExecutionContext_Cuda{}; }
#endif

/// the execution space an argument pulls toward (its memory space's native one, else host)
auto pulled_execution_space( const auto &arg ) {
    if constexpr ( requires { arg.memory_space().execution_space(); } )
        return arg.memory_space().execution_space();
    else
        return ExecutionContext_Cpu{};
}

/// execution space chosen for a set of arguments (CPU by default, promoted to device as soon
/// as one argument's data lives there)
auto execution_space_for() { return ExecutionContext_Cpu{}; }
auto execution_space_for( const auto &arg, const auto &...rest ) {
    return promote( pulled_execution_space( arg ), execution_space_for( rest... ) );
}

} // namespace sdot
