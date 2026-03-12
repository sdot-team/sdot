#include "sdot_w2_metal.h"

void sdot_w2_metal(const float* Xf, const float* Wf, size_t Nf,
                   const float* Xg, const float* Yg, size_t Mg,
                   size_t batch_size,
                   float* result, float* barycenters) {
    if (result) {
        for (size_t b = 0; b < batch_size; ++b) result[b] = 0.0f;
    }
    if (barycenters) {
        for (size_t i = 0; i < batch_size * Nf; ++i) barycenters[i] = 0.0f;
    }
}
