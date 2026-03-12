#ifndef SDOT_W2_METAL_H
#define SDOT_W2_METAL_H

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

void sdot_w2_metal(const float* Xf, const float* Wf, size_t Nf,
                   const float* Xg, const float* Yg, size_t Mg,
                   size_t batch_size,
                   float* result, float* barycenters);

#ifdef __cplusplus
}
#endif

#endif // SDOT_W2_METAL_H
