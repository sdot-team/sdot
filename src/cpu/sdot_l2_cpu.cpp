#include "sdot_l2_cpu.h"
#include <vector>
#include <cmath>

#if defined(_OPENMP)
#include <omp.h>
#endif

void sdot_l2_cpu(const float* f, const float* g, size_t N, float* result) {
    double sum = 0.0;
    // #pragma omp parallel for reduction(+:sum)
    for (size_t i = 0; i < N; ++i) {
        float diff = f[i] - g[i];
        sum += (double)(diff * diff);
    }
    *result = (float)(sum / (double)N);
}

void sdot_l2_backward_cpu(const float* f, const float* g, size_t N, float* grad_f, float* grad_g) {
    float invN = 1.0f / (float)N;
    // #pragma omp parallel for
    for (size_t i = 0; i < N; ++i) {
        float diff = f[i] - g[i];
        if (grad_f) grad_f[i] = 2.0f * diff * invN;
        if (grad_g) grad_g[i] = -2.0f * diff * invN;
    }
}
