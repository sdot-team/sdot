#include <metal_stdlib>
using namespace metal;

kernel void sdot_l2_forward_kernel(
    device const float* f [[buffer(0)]],
    device const float* g [[buffer(1)]],
    device float* partial_sums [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint bid [[threadgroup_position_in_grid]],
    uint threads_per_group [[threads_per_threadgroup]]
) {
    // Each thread calculates (f - g)^2
    // For now, let's just do a simple per-element and then reduce in C++
    // or do a simple threadgroup reduction.
    
    threadgroup float sdata[1024]; // Max threadgroup size
    
    float diff = f[id] - g[id];
    sdata[tid] = diff * diff;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Simple reduction in threadgroup
    for (uint s = 1; s < threads_per_group; s <<= 1) {
        if (tid % (2 * s) == 0) {
            if (tid + s < threads_per_group) {
                sdata[tid] += sdata[tid + s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (tid == 0) {
        partial_sums[bid] = sdata[0];
    }
}

kernel void sdot_l2_backward_kernel(
    device const float* f [[buffer(0)]],
    device const float* g [[buffer(1)]],
    device float* grad_f [[buffer(2)]],
    device float* grad_g [[buffer(3)]],
    constant float& invN [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    float diff = f[id] - g[id];
    grad_f[id] = 2.0f * diff * invN;
    grad_g[id] = -2.0f * diff * invN;
}
