#ifndef SDOT_L2_METAL_H
#define SDOT_L2_METAL_H

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

void sdot_l2_metal(const float* f, const float* g, size_t N, float* result);
void sdot_l2_backward_metal(const float* f, const float* g, size_t N, float* grad_f, float* grad_g);

#ifdef __cplusplus
}
#endif

#endif // SDOT_L2_METAL_H
