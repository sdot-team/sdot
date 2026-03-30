#pragma once

#include "support/solve_quadratic.h"
#include "PieceOfAffine1d.h"

namespace sdot {

#define UTP template<class T>
#define DTP PieceOfAffine1d<T>

UTP DTP DTP::take_some_mass( T mass_to_take ) {
    PieceOfAffine1d res = *this;
    if ( x0 == x1 )
       return res;

    const T xn = x0 + solve_quadratic<T>( ( y0 - y1 ) / ( x0 - x1 ), y0, - mass_to_take, 0 );
    const T yn = value_at( xn );
    res.mass = mass_to_take;
    res.x1 = xn;
    res.y1 = yn;

    mass -= mass_to_take;
    x0 = xn;
    y0 = yn;

    return res;
}

UTP T DTP::value_at( T xn ) const {
    return y0 + ( y1 - y0 ) * ( xn - x0 ) / ( x1 - x0 );
}

UTP T DTP::w2_dist( T dirac_pos ) const {
    return ( x0 - x1 ) * (
        + 4 * dirac_pos * ( x0 * ( 2 * y0 + y1 ) + x1 * ( y0 + 2 * y1 ) )
        - 6 * dirac_pos * dirac_pos * ( y0 + y1 )
        - x0 * x0 * ( 3 * y0 + y1 )
        - x1 * x1 * ( y0 + 3 * y1 )
        - 2 * x0 * x1 * ( y0 + y1 )
    ) / 12;
}

UTP T DTP::moment() const {
    if ( x0 == x1 )
       return 0.0;
    const double b = ( y1 - y0 ) / ( x1 - x0 );
    const double a = ( y0 - b * x0 );
    // TODO: precision handling
    return ( std::pow( x1, 2 ) - std::pow( x0, 2 ) ) * a / 2
         + ( std::pow( x1, 3 ) - std::pow( x0, 3 ) ) * b / 3;
}

UTP void DTP::w2_dist_backward( T dirac_pos, T g_out, T &g_dirac_pos, T &g_x0, T &g_x1, T &g_y0, T &g_y1 ) const {
    const T h = x1 - x0;
    if ( h == 0 ) return;
    const T inv_h = 1.0 / h;
    const T p = ( dirac_pos - x0 ) * inv_h;
    const T p2 = p * p;

    const T c0 = ( 1.0 - 4.0 * p + 6.0 * p2 ) / 12.0;
    const T c1 = ( 3.0 - 8.0 * p + 6.0 * p2 ) / 12.0;

    g_y0 += g_out * h * c0;
    g_y1 += g_out * h * c1;

    const T dF_dp = h * ( y0 * ( p - 1.0 / 3.0 ) + y1 * ( p - 2.0 / 3.0 ) );
    const T g_p = g_out * dF_dp;

    const T gp_h = g_p * inv_h;
    g_dirac_pos += gp_h;
    g_x1 += g_out * ( y0 * c0 + y1 * c1 ) - gp_h * p;
    g_x0 += -g_out * ( y0 * c0 + y1 * c1 ) + gp_h * p - gp_h;
}

UTP void DTP::moment_backward( T g_out, T &g_x0, T &g_x1, T &g_y0, T &g_y1 ) const {
    const T h = x1 - x0;
    if ( h == 0 ) return;

    g_y0 += g_out * h * ( 2 * x0 + x1 ) / 6.0;
    g_y1 += g_out * h * ( x0 + 2 * x1 ) / 6.0;

    const T M_h = ( 2 * x0 * y0 + x0 * y1 + x1 * y0 + 2 * x1 * y1 ) / 6.0;
    g_x1 += g_out * ( M_h + h * ( y0 + 2 * y1 ) / 6.0 );
    g_x0 += g_out * ( -M_h + h * ( 2 * y0 + y1 ) / 6.0 );
}

UTP void DTP::get_w2_dist_grad_ys( T point_scale, T dirac_pos, auto grad_ys ) const {
    using namespace std;

    // integrate( ( x - p )^2 / 2, x, x0, x1 )
    T contribution = point_scale / 6 * ( pow( x1 - dirac_pos, 3 ) - pow( x0 - dirac_pos, 3 ) );
    grad_ys[ index - 1 ] += contribution;
    grad_ys[ index - 0 ] += contribution;
}

UTP void DTP::integrate_w2_shape_functions( double dirac_pos, T potential, T original_x0, T original_x1, T &res_left, T &res_right ) const {
    if ( x0 == x1 || original_x0 == original_x1 )
        return;

    const T inv_len = 1.0 / ( original_x1 - original_x0 );
    const T dx0 = x0 - dirac_pos;
    const T dx1 = x1 - dirac_pos;
    const T dx0_2 = dx0 * dx0;
    const T dx1_2 = dx1 * dx1;
    const T dx0_3 = dx0_2 * dx0;
    const T dx1_3 = dx1_2 * dx1;
    const T dx0_4 = dx0_3 * dx0;
    const T dx1_4 = dx1_3 * dx1;

    // Integral of ((dirac_pos - t)^2 - potential) * (A*t + B) = [ A/4 * (t-x)^4 + (Ax+B)/3 * (t-x)^3 - potential * (A/2 * (t-x)^2 + (Ax+B)*(t-x)) ]_a^b

    // Left shape function: A = -inv_len, Ax+B = (original_x1 - dirac_pos) * inv_len
    {
        const T A = -inv_len;
        const T AxB = ( original_x1 - dirac_pos ) * inv_len;
        const T I_w2 = ( A / 4.0 ) * ( dx1_4 - dx0_4 ) + ( AxB / 3.0 ) * ( dx1_3 - dx0_3 );
        const T I_psi = ( A / 2.0 ) * ( dx1_2 - dx0_2 ) + ( AxB ) * ( dx1 - dx0 );
        res_left += I_w2 - potential * I_psi;
    }

    // Right shape function: A = inv_len, Ax+B = (dirac_pos - original_x0) * inv_len
    {
        const T A = inv_len;
        const T AxB = ( dirac_pos - original_x0 ) * inv_len;
        const T I_w2 = ( A / 4.0 ) * ( dx1_4 - dx0_4 ) + ( AxB / 3.0 ) * ( dx1_3 - dx0_3 );
        const T I_psi = ( A / 2.0 ) * ( dx1_2 - dx0_2 ) + ( AxB ) * ( dx1 - dx0 );
        res_right += I_w2 - potential * I_psi;
    }
}

UTP void DTP::integrate_linear_shape_functions( T slope, T offset, T original_x0, T original_x1, T &res_left, T &res_right ) const {
    if ( x0 == x1 || original_x0 == original_x1 )
        return;

    const T h = original_x1 - original_x0;
    const T inv_h = 1.0 / h;
    const T u0 = ( x0 - original_x0 ) * inv_h;
    const T u1 = ( x1 - original_x0 ) * inv_h;
    const T u0_2 = u0 * u0, u1_2 = u1 * u1;
    const T u0_3 = u0_2 * u0, u1_3 = u1_2 * u1;

    const T L0 = slope * original_x0 + offset;
    const T dL = slope * h;

    // Left shape function: 1-u
    {
        const T I = h * ( L0 * ( u1 - u0 ) + ( dL - L0 ) / 2.0 * ( u1_2 - u0_2 ) - dL / 3.0 * ( u1_3 - u0_3 ) );
        res_left += I;
    }

    // Right shape function: u
    {
        const T I = h * ( L0 / 2.0 * ( u1_2 - u0_2 ) + dL / 3.0 * ( u1_3 - u0_3 ) );
        res_right += I;
    }
}

#undef UTP
#undef DTP

} // namespace sdot
