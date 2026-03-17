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

UTP T DTP::w2_dist( double dirac_pos ) const {
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

#undef UTP
#undef DTP

} // namespace sdot
