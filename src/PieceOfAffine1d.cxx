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
    const double b = (y1 - y0) / (x1 - x0);
    const double a = (y0 - b * x0);
    // TODO: precision handling
    return ( std::pow( x1, 2 ) - std::pow( x0, 2 ) ) * a / 2
         + ( std::pow( x1, 3 ) - std::pow( x0, 3 ) ) * b / 3;
}

#undef UTP
#undef DTP

} // namespace sdot
