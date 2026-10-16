#pragma once

// =====================================================================================
// THE GAUSSIAN OF AN AFFINE FUNCTION ON A SIMPLEX, by divided differences ( `CodimMesh.h` ).
//
// On a simplex `S` of dimension `n` where `u` is affine ( `u_i` at the vertices ), Hermite-Genocchi gives
//
//     int_S prod_i lambda_i^a_i g( u ) = |S| n! a! psi[ u_0 .. u_n, u_i repeated a_i more times ],     psi^( n + |a| ) = g,
//
// the divided difference over the vertices, each one repeated as often as its barycentric coordinate appears ( `a! = prod a_i!` ).
// With `g( u ) = exp( -u^2 )`, all is one function of a multiset of nodes `x_0 .. x_m` and an order `p` of derivation:
//
//     G( p, x ) = int_{Delta_m} g^( p )( sum_i t_i x_i ) dt = psi[ x_0 .. x_m ],   psi = g^( p - m )
//
// ( `Delta_m` the unit simplex, volume `1 / m!`; `g^( -q )` the `q`-th antiderivative that vanishes at `+inf`:
// `( -1 )^q sqrt( pi ) / 2 i^( q - 1 ) erfc` ). `p = 0` is the integrand, `p = 1` its derivative with respect to a node
// ( `d / dx_i psi[ x ] = psi[ x, x_i ]` ).
//
// THE EVALUATION never subtracts close numbers:
//   * on a cluster of nodes of spread `<= tau( c ) = min( 1, 1 / | c | )` ( `c` its center ), the Taylor series of the divided
//     difference, `psi[ x ] = sum_r psi^( m + r )( c ) h_r( x - c ) / ( m + r )!` ( `h_r` the complete homogeneous symmetric
//     polynomials ), whose terms fall at least like `2^r / r!`;
//   * over a wider spread, the recurrence `psi[ x_lo .. x_hi ] = ( psi[ x_lo+1 .. x_hi ] - psi[ x_lo .. x_hi-1 ] ) / ( x_hi - x_lo )`
//     on the sorted nodes, each sub-difference by the same rule ( a table, `O( m^2 )` entries ).
// =====================================================================================

#include <loom/support/math.h>
#include <algorithm>
#include <cmath>

namespace sdot {
namespace gauss_dd {

constexpr int MAX_NODES = 10;                                     ///< `m + 1`: 3D moments with their adjoint stay below
constexpr int MAX_TERMS = 48;                                     ///< of a Taylor series ( `2^r / r!` < 1e-30 at 48 )

/// `g^( q )( x )` for `q` in `[ q0, q0 + n )`, `g = exp( -x^2 )`: Hermite above 0, the iterated `erfc` below
template<class TF>
void derivatives( TF x, int q0, int n, TF *out ) {
    const TF e = sdot::exp( - x * x );
    for ( int k = 0; k < n; ++k ) {
        const int q = q0 + k;
        if ( q >= 0 ) {
            TF h0 = 1, h1 = 2 * x;                                // physicists' Hermite: H_q( x ), g^( q ) = ( -1 )^q H_q e^{-x^2}
            if ( q == 0 ) h1 = h0;
            for ( int j = 1; j < q; ++j ) { const TF h2 = 2 * x * h1 - 2 * j * h0; h0 = h1; h1 = h2; }
            out[ k ] = ( q % 2 ? -h1 : h1 ) * e;
        } else {
            const int a = -q;                                     // ( -1 )^a sqrt( pi ) / 2 i^( a - 1 ) erfc( x )
            TF ia;                                                // i^( a - 1 ) erfc( x ), `2 n i^n = i^( n - 2 ) - 2 x i^( n - 1 )`
            if ( x <= 1 || a <= 1 ) {                             // ( upwards: stable where x is small )
                TF im1 = TF( 2 / std::sqrt( M_PI ) ) * e, i0 = sdot::erfc( x );
                for ( int j = 1; j < a; ++j ) { const TF i1 = ( im1 - 2 * x * i0 ) / ( 2 * j ); im1 = i0; i0 = i1; }
                ia = i0;
            } else {                                              // ( downwards from far, normalized by erfc: Miller; the start
                const int top = a + 30 + int( 170 / ( x * x ) );  //   needed grows like 1 / x^2, measured to 1e-15 against mpmath )
                TF hi = 0, mid = TF( 1e-300 ), want = 0;          // i^( n + 1 ), i^n
                for ( int n = top; n >= 1; --n ) {
                    const TF lo = 2 * ( n + 1 ) * hi + 2 * x * mid;   // i^( n - 1 ) = 2 ( n + 1 ) i^( n + 1 ) + 2 x i^n
                    hi = mid; mid = lo;
                    if ( n - 1 == a - 1 ) want = mid;
                    if ( mid > TF( 1e250 ) ) { hi *= TF( 1e-250 ); mid *= TF( 1e-250 ); want *= TF( 1e-250 ); }
                }
                ia = want / mid * sdot::erfc( x );                // ( mid = i^0 up to the scale )
            }
            out[ k ] = ( a % 2 ? -ia : ia ) * TF( std::sqrt( M_PI ) / 2 );
        }
    }
}

/// `psi[ x_lo .. x_hi ]` by its Taylor series around the middle, `psi^( j )` being `g^( q0 + j )`
template<class TF>
TF taylor( const TF *x, int lo, int hi, int q0 ) {
    const int m = hi - lo;
    const TF c = ( x[ lo ] + x[ hi ] ) / 2;
    TF d[ MAX_TERMS + MAX_NODES ];
    derivatives( c, q0 + m, MAX_TERMS, d );                       // psi^( m + r )( c ), r = 0 ..
    TF h[ MAX_TERMS ];                                            // h_r( y_lo .. y_hi ), by adding the variables one by one
    h[ 0 ] = 1;
    for ( int r = 1; r < MAX_TERMS; ++r ) h[ r ] = 0;
    for ( int i = lo; i <= hi; ++i ) {
        const TF y = x[ i ] - c;
        for ( int r = 1; r < MAX_TERMS; ++r ) h[ r ] += y * h[ r - 1 ];
    }
    TF s = 0, f = 1, prev = 1;                                    // f = 1 / ( m + r )!
    for ( int j = 2; j <= m; ++j ) f /= j;
    for ( int r = 0; r < MAX_TERMS; ++r ) {
        const TF t = d[ r ] * h[ r ] * f, at = t < 0 ? -t : t;
        s += t;
        f /= m + r + 1;
        if ( r > 4 && at + prev <= TF( 1e-18 ) * ( s < 0 ? -s : s ) )
            break;                                                // ( two terms: `h_r` of a symmetric cluster vanishes for odd `r` )
        prev = at;
    }
    return s;
}

/// `G( p, x_0 .. x_m )` ( see the head of the file ); `x` is reordered
template<class TF>
TF G( int p, TF *x, int m ) {
    std::sort( x, x + m + 1 );
    const int q0 = p - m;                                         // psi = g^( q0 )
    TF t[ MAX_NODES ][ MAX_NODES ];                               // t[ lo ][ len ]: psi[ x_lo .. x_lo+len ]
    for ( int len = 0; len <= m; ++len )
        for ( int lo = 0; lo + len <= m; ++lo ) {
            const int hi = lo + len;
            const TF c = ( x[ lo ] + x[ hi ] ) / 2, ac = c < 0 ? -c : c;
            const TF tau = ac > 1 ? 1 / ac : TF( 1 );
            if ( len == 0 ) {
                derivatives( x[ lo ], q0, 1, &t[ lo ][ 0 ] );
            } else if ( x[ hi ] - x[ lo ] <= tau ) {
                t[ lo ][ len ] = taylor( x, lo, hi, q0 );
            } else {
                t[ lo ][ len ] = ( t[ lo + 1 ][ len - 1 ] - t[ lo ][ len - 1 ] ) / ( x[ hi ] - x[ lo ] );
            }
        }
    return t[ 0 ][ m ];
}

} // namespace gauss_dd
} // namespace sdot
