#include <cuda_runtime.h>
#include "sdot_l2.h"
#include <device_launch_parameters.h>

__global__ void sdot_l2_forward_kernel(const float* f, const float* g, size_t N, float* partial_sums) {
    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    float diff = (i < N) ? f[i] - g[i] : 0.0f;
    sdata[tid] = diff * diff;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) partial_sums[blockIdx.x] = sdata[0];
}

__global__ void sdot_l2_backward_kernel(const float* f, const float* g, size_t N, float* grad_f, float* grad_g, float invN) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float diff = f[i] - g[i];
        if (grad_f) grad_f[i] = 2.0f * diff * invN;
        if (grad_g) grad_g[i] = -2.0f * diff * invN;
    }
}

// C wrapper functions would go here, calling the kernels.
// Since we don't have a CUDA environment to test, we provide the kernels as requested.
