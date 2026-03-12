#include "sdot_l2_metal.h"

void sdot_l2_metal(const float* f, const float* g, size_t N, float* result) {
    if (result) *result = 0.0f;
}

void sdot_l2_backward_metal(const float* f, const float* g, size_t N, float* grad_f, float* grad_g) {
    if (grad_f) {
        for (size_t i = 0; i < N; ++i) grad_f[i] = 0.0f;
    }
    if (grad_g) {
        for (size_t i = 0; i < N; ++i) grad_g[i] = 0.0f;
    }
}
