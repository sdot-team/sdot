#pragma once

#include "support/common_types.h"

namespace sdot {

template<class T>
struct PieceOfAffine1d {
    auto take_some_mass ( T mass_to_take ) -> PieceOfAffine1d;
    T    value_at       ( T x ) const;

    T    w2_dist        ( double dirac_pos ) const;
    T    moment         () const;

    PI   index;
    T    mass;
    T    x0;
    T    x1;
    T    y0;
    T    y1;
};

} // namespace sdot

#include "PieceOfAffine1d.cxx"
