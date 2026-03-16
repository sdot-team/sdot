#pragma once

#include "common_macros.h"
#include <algorithm>
#include <cmath>

namespace sdot {

///< a is multiplied by 2
T_T T solve_quadratic( T a, T b, T c, T default_value = 0 ) {
    const T scale = std::max( { std::abs( a ), std::abs( b ), std::abs( c ) } );
    if ( scale == 0.0 )
       return 0.0;

    const T eps = std::numeric_limits<T>::epsilon() * scale;
    if ( std::abs( a ) < eps ) {
        if ( std::abs( b ) < eps )
           return default_value;
        return - c / b;
    }

    T delta = b * b - 2 * a * c;
    if ( delta < eps * eps )
       return - b / a;

    return ( std::sqrt( std::max( T( 0 ), delta ) ) - b ) / a;
}

} // namespace sdot
