// Extreme eigenvalues of the preconditioned operator from the CG coefficients ( Lanczos: T_jj = 1 / a_j + b_{j-1} / a_{j-1},
// T_{j,j+1} = sqrt( b_j ) / a_j ), by Sturm bisection
#pragma once
#include <cmath>
#include <vector>
#include <algorithm>
inline void lanczos_extremes( const std::vector<double> &al, const std::vector<double> &be, double &lmin, double &lmax ) {
    const int m = int( al.size() );
    if( m == 0 ) { lmin = lmax = 0; return; }
    std::vector<double> d( m ), e( m, 0 );
    for( int j = 0; j < m; ++j ) {
        d[ j ] = 1 / al[ j ] + ( j ? be[ j - 1 ] / al[ j - 1 ] : 0 );
        if( j + 1 < m ) e[ j ] = std::sqrt( std::max( be[ j ], 0.0 ) ) / al[ j ];
    }
    auto count = [&]( double x ) { int c = 0; double q = 1; for( int j = 0; j < m; ++j ) { q = d[ j ] - x - ( j ? e[ j - 1 ] * e[ j - 1 ] / q : 0 ); if( q == 0 ) q = 1e-300; c += q < 0; } return c; };
    double lo = 0, hi = 0; for( int j = 0; j < m; ++j ) hi = std::max( hi, std::abs( d[ j ] ) + 2 * std::abs( e[ j ] ) );
    auto kth = [&]( int k ) { double a = std::min( 0.0, -hi ), b = hi; for( int it = 0; it < 200; ++it ) { const double c = 0.5 * ( a + b ); if( count( c ) > k ) b = c; else a = c; } return 0.5 * ( a + b ); };
    lmin = kth( 0 ); lmax = kth( m - 1 ); ( void )lo;
}
