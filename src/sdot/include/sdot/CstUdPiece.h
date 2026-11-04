#pragma once

#include <loom/support/common_macros.h> // HD

namespace sdot {

template<class TF>
struct CstUdPiece {
    // mass carried by the piece: Integral_{x0}^{x1} y dx
    HD TF mass() const { return y * ( x1 - x0 ); }

    // first moment: Integral_{x0}^{x1} x * y dx (used to build target barycenters)
    HD TF first_moment() const { return y * ( x1 * x1 - x0 * x0 ) / 2; }

    // Integral_{x0}^{x1} (x - p)^2 dx = [ (x1-p)^3 - (x0-p)^3 ] / 3, WITHOUT the density factor
    // (the direct term of d cost / d y_c, which survives even where y = 0).
    HD auto second_moment_about( auto &&p ) const {
        const TF a = x1 - p, b = x0 - p;
        return ( a * a * a - b * b * b ) / 3;
    }

    // Integral_{x0}^{x1} (x - dirac_pos)^2 * y dx = y ( b^3 - a^3 ) / 3 = y ( b - a ) ( a^2 + a b + b^2 ) / 3, with
    // a = x0-dirac_pos, b = x1-dirac_pos: RELATIVE to the dirac -- expanded in x0, x1 and dirac_pos, terms of the order
    // of x^2 cancel down to ( x - dirac_pos )^2, which single precision does not survive.
    HD auto w2_dist( auto &&dirac_pos ) const {
        const TF a = x0 - TF( dirac_pos ), b = x1 - TF( dirac_pos );
        return y * ( b - a ) * ( a * a + a * b + b * b ) / 3;
    }

    TF x0, x1;
    TF y;
};

}
