#pragma once

#include "DsVec.h"

namespace sdot {

template<class TF>
TF dichotomy_find_largest_zero( TF beg, auto &&func, TF rel = 1e-3 ) {
    // grow
    TF end = 2 * beg + 1;
    while ( true ) {
        if ( func( end ) )
            break;
        end *= 2;
    }

    // fit
    while ( true ) {
        if ( ( end - beg ) <= rel * end )
            return beg;

        TF mid = ( beg + end ) / 2;
        auto val = func( end );
        if ( val )
            end = mid;
        else
            beg = mid;
    }
}

} // namespace sdot
