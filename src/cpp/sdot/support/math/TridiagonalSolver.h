#pragma once

#include "common_types.h"
#include <vector>
#include <span>

namespace sdot {

/**
    Tridiagonal solver for Mx = d, where M has diagonals a, b, c:
    M = [ b0 c0  0  0 ]
        [ a1 b1 c1  0 ]
        [ 0  a2 b2 c2 ]
        [ 0  0  a3 b3 ]
*/
template<class TF>
struct TridiagonalSolver {
    static void solve( std::span<TF> x, std::span<const TF> a, std::span<const TF> b, std::span<const TF> c, std::span<const TF> d ) {
        PI n = x.size();
        if ( n == 0 ) return;
        if ( n == 1 ) { x[ 0 ] = d[ 0 ] / b[ 0 ]; return; }

        std::vector<TF> cp( n );
        std::vector<TF> dp( n );

        cp[ 0 ] = c[ 0 ] / b[ 0 ];
        dp[ 0 ] = d[ 0 ] / b[ 0 ];

        for ( PI i = 1; i < n; ++i ) {
            TF m = 1 / ( b[ i ] - a[ i ] * cp[ i - 1 ] );
            cp[ i ] = c[ i ] * m;
            dp[ i ] = ( d[ i ] - a[ i ] * dp[ i - 1 ] ) * m;
        }

        x[ n - 1 ] = dp[ n - 1 ];
        for ( SI i = n - 2; i >= 0; --i )
            x[ i ] = dp[ i ] - cp[ i ] * x[ i + 1 ];
    }

    /**
        Solve M^T y = g
        M^T = [ b0 a1  0  0 ]
              [ c0 b1 a2  0 ]
              [ 0  c1 b2 a3 ]
              [ 0  0  c2 b3 ]
    */
    static void solve_transpose( std::span<TF> y, std::span<const TF> a, std::span<const TF> b, std::span<const TF> c, std::span<const TF> g ) {
        PI n = y.size();
        if ( n == 0 ) return;
        if ( n == 1 ) { y[ 0 ] = g[ 0 ] / b[ 0 ]; return; }

        std::vector<TF> a_T( n ), b_T( n ), c_T( n );
        for ( PI i = 0; i < n; ++i ) {
            a_T[ i ] = ( i > 0     ? c[ i - 1 ] : 0 );
            b_T[ i ] = b[ i ];
            c_T[ i ] = ( i < n - 1 ? a[ i + 1 ] : 0 );
        }
        solve( y, a_T, b_T, c_T, g );
    }
};

} // namespace sdot
