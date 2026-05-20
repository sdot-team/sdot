#pragma once

#ifdef __CUDACC__
#include "MemorySpace_GlobalCudaRam.h"
#include "MemorySpace_PinnedCpuRam.h"
#include "ExecutionSpace.h"

#include <cuda_runtime.h>
namespace sdot {

// CUDA device global RAM + stream. {}-constructible => "main" (device 0, default stream).
// A future SpecificCudaSpace (no {} ctor) can force explicit device/stream selection.
struct ExecutionSpace_Cuda : public ExecutionSpace {
    int          device_id = 0;
    cudaStream_t stream    = 0; // default stream
};

} // namespace sdot

#endif
