#pragma once

#include <loom/support/common_macros.h> // HD

namespace sdot {

/// A piece of a 1D walk on which the density is AFFINE: `y0 + s ( x - x0 )` on `[ x0, x1 ]` -- the `CstUdPiece` of a
/// pixel of degree 1 ( `Image( degree = 1 )` ), with the same interface, plus the integrals against the two Bernstein
/// basis functions of its pixel ( its adjoint ).
template<class TF>
struct LinUdPiece {
    // mass carried by the piece: Integral_{x0}^{x1} y dx
    HD TF mass() const { const TF L = x1 - x0; return L * ( y0 + s * L / 2 ); }

    // first moment: Integral_{x0}^{x1} x * y dx
    HD TF first_moment() const {
        const TF L = x1 - x0;
        return L * ( x0 * y0 + L * ( x0 * s + y0 ) / 2 + s * L * L / 3 );
    }

    // Integral_{x0}^{x1} (x - p)^2 dx, WITHOUT the density factor ( as `CstUdPiece`'s )
    HD auto second_moment_about( auto &&p ) const {
        const TF a = x1 - p, b = x0 - p;
        return ( a * a * a - b * b * b ) / 3;
    }

    // Integral_{x0}^{x1} (x - p)^2 * y dx, in u = x - x0 ( q = x0 - p ): y0 ( L^3/3 + q L^2 + q^2 L ) + s ( L^4/4 + 2 q L^3/3 + q^2 L^2/2 )
    HD auto w2_dist( auto &&dirac_pos ) const {
        const TF L = x1 - x0, q = x0 - TF( dirac_pos );
        return L * ( y0 * ( L * L / 3 + q * L + q * q ) + s * L * ( L * L / 4 + 2 * q * L / 3 + q * q / 2 ) );
    }

    // Integral_{x0}^{x1} f( x ) b_j( x ) dx, b_0 = ( xr - x ) / h and b_1 = ( x - xl ) / h the Bernstein basis of the
    // pixel [ xl, xr = xl + h ]: exact for f of degree <= 4 ( a 3-point Gauss-Legendre rule, the integrand of degree <= 5 )
    HD TF basis_integral( int j, auto &&f, TF xl, TF h ) const {
        const TF c = ( x0 + x1 ) / 2, r = ( x1 - x0 ) / 2, g = TF( 0.7745966692414834 );
        TF res = 0;
        for ( int k = 0; k < 3; ++k ) {
            const TF x = c + r * ( k == 0 ? -g : k == 1 ? TF( 0 ) : g );
            const TF w = k == 1 ? TF( 8 ) / 9 : TF( 5 ) / 9;
            const TF b = j == 0 ? ( xl + h - x ) / h : ( x - xl ) / h;
            res += w * f( x ) * b;
        }
        return r * res;
    }

    TF x0, x1;
    TF y0;      // the density at x0
    TF s;       // its slope
};

}
