#pragma once

#ifdef __CUDACC__
#include "ExecutionSpace.h"

#include <cuda_runtime.h>
namespace sdot {

// CUDA device global RAM + stream. {}-constructible => "main" (device 0, default stream).
// A future SpecificCudaSpace (no {} ctor) can force explicit device/stream selection.
struct ExecutionSpace_Cuda : public ExecutionSpace {
    ExecutionSpace_Cuda() : device_id( 0 ), stream( default_stream ) {}

    static cudaStream_t default_stream;

    int          device_id = 0;
    cudaStream_t stream = -1; // default stream
};

} // namespace sdot

#endif
