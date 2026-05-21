#pragma once

#include "ExecutionSpace_Cuda.h" // IWYU pragma: export
#include "ExecutionSpace_Cpu.h"

namespace sdot {

#ifdef __CUDA_ARCH__
    HD auto current_execution_context() { return ExecutionSpace_Cuda(); }
#else
    HD auto current_execution_context() { return ExecutionSpace_Cpu(); }
#endif

} // namespace sdot
