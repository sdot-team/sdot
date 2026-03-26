#pragma once

#include "support/common_macros.h"
#include "support/common_types.h"

namespace sdot {

template<class T>
struct PieceOfAffine1d {
    HD auto take_some_mass     ( T mass_to_take ) -> PieceOfAffine1d;
    HD T    value_at           ( T x ) const;

    HD void get_w2_dist_grad_ys( T point_scale, T dirac_pos, auto grad_ys ) const;
    HD T    w2_dist            ( T dirac_pos ) const;
    HD T    moment             () const;

    HD void w2_dist_backward   ( T dirac_pos, T g_out, T &g_dirac_pos, T &g_x0, T &g_x1, T &g_y0, T &g_y1 ) const;
    HD void moment_backward    ( T g_out, T &g_x0, T &g_x1, T &g_y0, T &g_y1 ) const;

    // Integration of ((dirac_pos - t)^2 - potential) * shape_function(t)
    // where shape_function(t) is (x1 - t) / (x1 - x0) (left) or (t - x0) / (x1 - x0) (right)
    // defined on the *original* interval of the piece.
    HD void integrate_w2_shape_functions( double dirac_pos, T potential, T original_x0, T original_x1, T &res_left, T &res_right ) const;
    HD void integrate_linear_shape_functions( T slope, T offset, T original_x0, T original_x1, T &res_left, T &res_right ) const;

    PI   index;
    T    mass;
    T    x0;
    T    x1;
    T    y0;
    T    y1;
};

} // namespace sdot

#include "PieceOfAffine1d.cxx"
