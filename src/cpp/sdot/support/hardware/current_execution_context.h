#pragma once

#include "ExecutionContext_Cuda.h" // IWYU pragma: export
#include "ExecutionContext_Cpu.h"

namespace sdot {

#ifdef __CUDA_ARCH__
    HD auto current_execution_context() { return ExecutionContext_Cuda(); }
#else
    HD auto current_execution_context() { return ExecutionContext_Cpu(); }
#endif

} // namespace sdot
