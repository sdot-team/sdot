#pragma once

// =====================================================================================
// AN EXACT RULE FOR POLYNOMIALS ON A SIMPLEX, in any dimension: Grundmann-Moller ( 1978 ). Of index `s`, it integrates
// exactly every polynomial of degree `2 s + 1` on an `N`-simplex `S`,
//
//     int_S f = |S| sum_{ i = 0 }^{ s } w_i sum_{ |b| = s - i } f( x( ( 2 b + 1 ) / ( d + N - 2 i ) ) ),
//     w_i = ( -1 )^i 2^{ -2 s } ( d + N - 2 i )^d N! / ( i! ( d + N - i )! ),       d = 2 s + 1,
//
// `b` running over the `N + 1` non-negative integers of sum `s - i`, the point given by its barycentric coordinates. Some
// weights are negative, which a polynomial does not mind: the rule is exact, and the cancellation stays at a few ulps up to
// `s = 6` ( 1e-14 in 3D at `s = 4` ). It is what integrates a polynomial piece -- a pixel of an `Image` of degree `p > 0` -- on
// the simplices of a cell, its moments and its facets: no adaptivity, nothing to tune, the derivative of what is computed is
// the derivative of the integral.
// =====================================================================================

#include <loom/support/common_macros.h>

namespace sdot {
namespace integration {

/// the index of the rule that is exact up to `degree`
HD constexpr int simplex_rule_index( int degree ) { return degree < 0 ? 0 : degree / 2; }

/// `func( bary, w )` for each node of the rule exact up to `degree` on an `N`-simplex: `bary` its `N + 1` barycentric coordinates,
/// `w` its weight, such that `int_S f = |S| sum w f( x( bary ) )`
template<int N,class TF>
HD void for_each_simplex_node( int degree, auto &&func ) {
    const int s = simplex_rule_index( degree ), d = 2 * s + 1;
    auto fact = []( int n ) { double r = 1; for ( int i = 2; i <= n; ++i ) r *= i; return r; };
    for ( int i = 0; i <= s; ++i ) {
        const int den = d + N - 2 * i;
        double w = ( i % 2 ? -1.0 : 1.0 ) * fact( N ) / ( fact( i ) * fact( d + N - i ) );
        w *= den;                                        // `2^{ -2 s } den^d = den ( den^2 / 4 )^s`
        for ( int k = 0; k < s; ++k )
            w *= double( den ) * den / 4;
        // the compositions of `m = s - i` in `N + 1` parts: an odometer on the first `N`, the last one what is left
        const int m = s - i;
        int b[ N + 1 ];
        for ( int c = 0; c <= N; ++c )
            b[ c ] = 0;
        while ( true ) {
            int used = 0;
            for ( int c = 0; c < N; ++c )
                used += b[ c ];
            if ( used <= m ) {
                b[ N ] = m - used;
                TF bary[ N + 1 ];
                for ( int c = 0; c <= N; ++c )
                    bary[ c ] = TF( double( 2 * b[ c ] + 1 ) / den );
                func( bary, TF( w ) );
            }
            int c = 0;
            while ( c < N && ++b[ c ] > m ) { b[ c ] = 0; ++c; }
            if ( c == N )
                break;
        }
    }
}

} // namespace integration
} // namespace sdot
