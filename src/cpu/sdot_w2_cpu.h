#pragma once

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

void sdot_w2_cpu(
    const float *dirac_xs, const float *dirac_ws, size_t nb_diracs,
    const float *point_xs, const float *point_ys, size_t nb_points,
    size_t batch_size, float *w2_squared, float *w2_barycenters
);

void sdot_w2_backward_cpu(
    const float *grad_distance, const float *grad_barycenters,
    const float *w2_barycenters,
    const float *dirac_xs, const float *dirac_ws, size_t nb_diracs,
    const float *points_xs, const float *points_ys, size_t nb_points,
    size_t batch_size,
    float *grad_dirac_xs,
    float *grad_dirac_ws,
    float *grad_point_xs,
    float *grad_point_ys
);

#ifdef __cplusplus
}
#endif
