#ifndef SDOT_W2_H
#define SDOT_W2_H

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute W2 distance and barycenters for a batch of (f, g) pairs.
 * batch_size: Number of pairs to process.
 * result: Output array (size batch_size).
 * barycenters: Output array (size batch_size * Nf), ignored if NULL.
 */
void sdot_w2_cpu(const float* Xf, const float* Wf, size_t Nf,
                 const float* Xg, const float* Yg, size_t Mg,
                 size_t batch_size,
                 float* result, float* barycenters);

void sdot_w2_metal(const float* Xf, const float* Wf, size_t Nf,
                   const float* Xg, const float* Yg, size_t Mg,
                   size_t batch_size,
                   float* result, float* barycenters);

#ifdef __cplusplus
}
#endif

#endif // SDOT_W2_H
