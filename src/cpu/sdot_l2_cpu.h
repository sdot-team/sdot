#ifndef SDOT_L2_CPU_H
#define SDOT_L2_CPU_H

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

void sdot_l2_cpu(const float* f, const float* g, size_t N, float* result);
void sdot_l2_backward_cpu(const float* f, const float* g, size_t N, float* grad_f, float* grad_g);

#ifdef __cplusplus
}
#endif

#endif // SDOT_L2_CPU_H
