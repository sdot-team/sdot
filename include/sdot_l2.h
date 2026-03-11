#ifndef SDOT_L2_H
#define SDOT_L2_H

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute squared L2 distance: sum((f[i] - g[i])^2) / N
 * f: Array of function 1 values
 * g: Array of function 2 values
 * N: Number of samples
 * result: Output scalar (single float)
 */
void sdot_l2_cpu(const float* f, const float* g, size_t N, float* result);

/**
 * Compute gradients for squared L2 distance.
 * f, g: Input function values
 * N: Number of samples
 * grad_f: Output gradient wrt f (size N)
 * grad_g: Output gradient wrt g (size N)
 */
void sdot_l2_backward_cpu(const float* f, const float* g, size_t N, float* grad_f, float* grad_g);

/**
 * Metal implementations
 */
void sdot_l2_metal(const float* f, const float* g, size_t N, float* result);
void sdot_l2_backward_metal(const float* f, const float* g, size_t N, float* grad_f, float* grad_g);

#ifdef __cplusplus
}
#endif

#endif // SDOT_L2_H
