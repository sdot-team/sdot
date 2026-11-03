// Exact fallback for one cell: clip the domain box by the neighbors sorted by distance, up to the
// security radius ( stop when |k|^2 - 2 |k| R + psi_i - max psi > 0, R = farthest vertex ). Scalar,
// branchy, rare. Faces go to a sparse list ( neighbor flat index, L / ( 2 |k| ) ).
#pragma once
#include <cmath>
#include <vector>
#include <algorithm>

template<class T = double>
struct ExactKernel {
    int RB; // largest ring in the list ( psi_pad is padded by RB )
    std::vector<int> ox, oy;
    std::vector<T> nk;

    ExactKernel( int RB ) : RB( RB ) {
        for( int a = -RB; a <= RB; ++a ) for( int b = -RB; b <= RB; ++b ) if( a || b ) { ox.push_back( a ); oy.push_back( b ); }
        std::vector<int> ord( ox.size() );
        for( size_t i = 0; i < ord.size(); ++i ) ord[ i ] = i;
        std::stable_sort( ord.begin(), ord.end(), [ & ]( int i, int j ) { return ox[ i ] * ox[ i ] + oy[ i ] * oy[ i ] < ox[ j ] * ox[ j ] + oy[ j ] * oy[ j ]; } );
        std::vector<int> x2, y2;
        for( int i : ord ) { x2.push_back( ox[ i ] ); y2.push_back( oy[ i ] ); nk.push_back( std::hypot( T( ox[ i ] ), T( oy[ i ] ) ) ); }
        ox = x2; oy = y2;
    }

    static constexpr int C = 128;

    // the cell as a polygon relative to the seed: n vertices, edge e from vertex e to e+1 carries the
    // offset index pl[ e ] ( -1: box ). Returns false if the list was exhausted before the security radius.
    bool clip( int N, const T *psi_pad, T psi_max, int a, int b, T *px, T *py, int *pl, int &n ) const {
        const int P = N + 2 * RB;
        T qx[ C ], qy[ C ];
        int ql[ C ];
        n = 4;
        px[ 0 ] = -a - T( 0.5 ); py[ 0 ] = -b - T( 0.5 ); pl[ 0 ] = -1;
        px[ 1 ] = N - T( 0.5 ) - a; py[ 1 ] = -b - T( 0.5 ); pl[ 1 ] = -1;
        px[ 2 ] = N - T( 0.5 ) - a; py[ 2 ] = N - T( 0.5 ) - b; pl[ 2 ] = -1;
        px[ 3 ] = -a - T( 0.5 ); py[ 3 ] = N - T( 0.5 ) - b; pl[ 3 ] = -1;
        const T *pi = psi_pad + ( a + RB ) * P + ( b + RB );
        T R2 = 0;
        for( int e = 0; e < n; ++e ) R2 = std::max( R2, px[ e ] * px[ e ] + py[ e ] * py[ e ] );
        bool done = false;
        for( size_t o = 0; o < ox.size(); ++o ) {
            const T R = std::sqrt( R2 ), k = nk[ o ];
            if( k >= R && k * k - 2 * k * R + pi[ 0 ] - psi_max > 0 ) { done = true; break; }
            const int na = a + ox[ o ], nb = b + oy[ o ];
            if( na < 0 || na >= N || nb < 0 || nb >= N ) continue;
            const T kx = ox[ o ], ky = oy[ o ];
            const T c = ( kx * kx + ky * ky + pi[ 0 ] - pi[ ox[ o ] * P + oy[ o ] ] ) / 2;
            T s[ C ];
            bool out = false;
            for( int e = 0; e < n; ++e ) { s[ e ] = kx * px[ e ] + ky * py[ e ] - c; out |= s[ e ] > 0; }
            if( ! out ) continue;
            int m = 0;
            for( int e = 0; e < n; ++e ) {
                int f = e + 1 == n ? 0 : e + 1;
                if( s[ e ] <= 0 ) {
                    qx[ m ] = px[ e ]; qy[ m ] = py[ e ]; ql[ m++ ] = pl[ e ];
                    if( s[ f ] > 0 ) {
                        T t = s[ e ] / ( s[ e ] - s[ f ] );
                        qx[ m ] = px[ e ] + t * ( px[ f ] - px[ e ] ); qy[ m ] = py[ e ] + t * ( py[ f ] - py[ e ] ); ql[ m++ ] = o;
                    }
                } else if( s[ f ] <= 0 ) {
                    T t = s[ e ] / ( s[ e ] - s[ f ] );
                    qx[ m ] = px[ e ] + t * ( px[ f ] - px[ e ] ); qy[ m ] = py[ e ] + t * ( py[ f ] - py[ e ] ); ql[ m++ ] = pl[ e ];
                }
            }
            n = m;
            R2 = 0;
            for( int e = 0; e < n; ++e ) { px[ e ] = qx[ e ]; py[ e ] = qy[ e ]; pl[ e ] = ql[ e ]; R2 = std::max( R2, px[ e ] * px[ e ] + py[ e ] * py[ e ] ); }
            if( n == 0 ) { done = true; break; }
        }
        return done;
    }

    // returns false if the list was exhausted before the security radius ( RB too small )
    bool cell( int N, const T *psi_pad, T psi_max, int a, int b, T &area, std::vector<std::pair<int,T>> &faces ) const {
        T px[ C ], py[ C ];
        int pl[ C ], n;
        const bool done = clip( N, psi_pad, psi_max, a, b, px, py, pl, n );
        T ar = 0;
        for( int e = 0; e < n; ++e ) {
            int f = e + 1 == n ? 0 : e + 1;
            ar += px[ e ] * py[ f ] - py[ e ] * px[ f ];
            if( pl[ e ] >= 0 ) {
                T L = std::hypot( px[ f ] - px[ e ], py[ f ] - py[ e ] );
                if( L > 0 ) faces.push_back( { ( a + ox[ pl[ e ] ] ) * N + b + oy[ pl[ e ] ], L / ( 2 * nk[ pl[ e ] ] ) } );
            }
        }
        area = ar / 2;
        return done;
    }
};
