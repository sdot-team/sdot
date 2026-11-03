// Classic branchy baseline: start from the domain box, clip by the neighbors
// of the same ring-R stencil sorted by distance ( early exit when no vertex is
// outside ), then read area and face lengths. Same inputs and outputs as
// StencilKernel ( hess per stencil offset instead of per direction ).
#pragma once
#include <cmath>
#include <algorithm>
#include <vector>

template<int R, class T = double>
struct ClipKernel {
    static constexpr int nb_offsets = ( 2 * R + 1 ) * ( 2 * R + 1 ) - 1, C = 64;
    int ox[ nb_offsets ], oy[ nb_offsets ];

    ClipKernel() {
        int n = 0;
        for( int a = -R; a <= R; ++a ) for( int b = -R; b <= R; ++b ) if( a || b ) { ox[ n ] = a; oy[ n ] = b; ++n; }
        // sort by distance
        for( int i = 0; i < n; ++i ) for( int j = i + 1; j < n; ++j )
            if( ox[ j ] * ox[ j ] + oy[ j ] * oy[ j ] < ox[ i ] * ox[ i ] + oy[ i ] * oy[ i ] ) { std::swap( ox[ i ], ox[ j ] ); std::swap( oy[ i ], oy[ j ] ); }
    }

    // hess[ i * nb_offsets + o ] = face length / ( 2 |k| )
    void run( int N, const T *psi_pad, T *area, T *hess ) const {
        const int P = N + 2 * R;
        for( int a = 0; a < N; ++a ) {
            for( int b = 0; b < N; ++b ) {
                T px[ C ], py[ C ], qx[ C ], qy[ C ];
                int pl[ C ], ql[ C ], n = 4;
                px[ 0 ] = -a - T( 0.5 ); py[ 0 ] = -b - T( 0.5 ); pl[ 0 ] = -1;
                px[ 1 ] = N - T( 0.5 ) - a; py[ 1 ] = -b - T( 0.5 ); pl[ 1 ] = -1;
                px[ 2 ] = N - T( 0.5 ) - a; py[ 2 ] = N - T( 0.5 ) - b; pl[ 2 ] = -1;
                px[ 3 ] = -a - T( 0.5 ); py[ 3 ] = N - T( 0.5 ) - b; pl[ 3 ] = -1;
                const T *pi = psi_pad + ( a + R ) * P + ( b + R );
                for( int o = 0; o < nb_offsets; ++o ) {
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
                    for( int e = 0; e < n; ++e ) { px[ e ] = qx[ e ]; py[ e ] = qy[ e ]; pl[ e ] = ql[ e ]; }
                }
                T ar = 0;
                T *hs = hess + ( a * N + b ) * nb_offsets;
                for( int o = 0; o < nb_offsets; ++o ) hs[ o ] = 0;
                for( int e = 0; e < n; ++e ) {
                    int f = e + 1 == n ? 0 : e + 1;
                    ar += px[ e ] * py[ f ] - py[ e ] * px[ f ];
                    if( pl[ e ] >= 0 ) {
                        T L = std::hypot( px[ f ] - px[ e ], py[ f ] - py[ e ] );
                        hs[ pl[ e ] ] += L / ( 2 * std::hypot( T( ox[ pl[ e ] ] ), T( oy[ pl[ e ] ] ) ) );
                    }
                }
                area[ a * N + b ] = ar / 2;
            }
        }
    }
};
