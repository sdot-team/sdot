#pragma once

#ifdef __CUDACC__

namespace sdot {

// GPU counterpart of CpuThreadInfo: identity of a device thread inside the grid.
struct CudaThreadInfo {
    __device__ int global_id()         const { return blockIdx.x * blockDim.x + threadIdx.x; }
    __device__ int nb_threads()        const { return gridDim.x * blockDim.x; }
    __device__ int local_id()          const { return threadIdx.x; }
    __device__ int block_id()          const { return blockIdx.x; }
    __device__ int threads_per_block() const { return blockDim.x; }
};

} // namespace sdot

#endif
