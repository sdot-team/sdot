// Micro-benchmark: ring-1 power cells, square lattice ( 8 neighbours, 24 tightening pairs ) vs hexagonal lattice ( 6 neighbours,
// 6 pairs ), same number of cells, same outputs ( area + one hessian weight L / ( 2 |k| ) per direction ), same code style
// ( blocks of B cells along a row, omp simd ). Neither builds vertices: with unit normals n_k sorted by angle, tightened
// offsets ht_k and an angle a between consecutive normals,
//     L_k = ( ht_{k-1} - ht_k cos a ) / sin a + ( ht_{k+1} - ht_k cos a ) / sin a,   area = 1/2 sum_k ht_k L_k.
// The square reference also runs grid_stencil's StencilKernel<1> ( vertices ), for the record.
//
//   hexbench N [ reps ]   ( single thread; the hexagonal lattice has W x R ~ N x N seeds )
// build ( from this folder ):
//   c++ -std=c++20 -O3 -march=native -fno-math-errno -fopenmp-simd -I ../grid_stencil hexbench.cpp -o hexbench
#include "stencil_kernel.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

using T = double;
constexpr int B = 32;
static const T inf = std::numeric_limits<T>::infinity();

double best_ns( long n, int reps, const std::function<void()> &f ) {
    double best = 1e30;
    for( int r = 0; r < reps; ++r ) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        auto t1 = std::chrono::steady_clock::now();
        best = std::min( best, std::chrono::duration<double, std::nano>( t1 - t0 ).count() / double( n ) );
    }
    return best;
}

// ---------------------------------------------------------------- hexagonal, ring 1
// R rows of W seeds, row r at y = r sqrt( 3 ) / 2, odd rows shifted by +1/2. psi padded by one row / column of -inf.
// Directions ( angle 60 k ): 0 = +x, 1 = up right, 2 = up left, 3 = -x, 4 = down left, 5 = down right.
// Only the inner rows ( 1 .. R - 2 ): the first / last rows need the +-y box faces ( 8 directions, as the square ).
struct Hex1 {
    static void run( int W, int R, const T *pp, T *area, T *wgt ) {
        const int P = W + 2;
        const T c = T( 1 ) / std::sqrt( T( 3 ) ); // 1 / ( 2 sin 60 ) * ( 2 cos 60 = 1 )
        for( int r = 1; r < R - 1; ++r ) {
            const int sh = r & 1; // odd rows: neighbours above / below at columns c and c + 1, even rows: c - 1 and c
            const T *pi = pp + ( r + 1 ) * P + 1;
            const T *pu = pi + P - 1 + sh, *pd = pi - P - 1 + sh;
            const T xr = T( W ) - T( 0.25 ) - T( 0.5 ) * sh, xl = T( 0.25 ) + T( 0.5 ) * sh; // box: +x face at W - 1/4 - x, -x at x + 1/4
            for( int b0 = 0; b0 < W; b0 += B ) {
                const int n = std::min( B, W - b0 );
                alignas( 64 ) T ar[ B ], w[ 6 ][ B ];
                #pragma omp simd
                for( int x = 0; x < B; ++x ) {
                    const int i = b0 + ( x < n ? x : 0 );
                    const T p = pi[ i ];
                    T h0 = T( 0.5 ) * ( T( 1 ) + ( p - pi[ i + 1 ] ) );
                    T h1 = T( 0.5 ) * ( T( 1 ) + ( p - pu[ i + 1 ] ) );
                    T h2 = T( 0.5 ) * ( T( 1 ) + ( p - pu[ i ] ) );
                    T h3 = T( 0.5 ) * ( T( 1 ) + ( p - pi[ i - 1 ] ) );
                    T h4 = T( 0.5 ) * ( T( 1 ) + ( p - pd[ i ] ) );
                    T h5 = T( 0.5 ) * ( T( 1 ) + ( p - pd[ i + 1 ] ) );
                    const T bx = T( b0 + x );
                    const T hb0 = xr - bx, hb3 = xl + bx;
                    h0 = hb0 < h0 ? hb0 : h0; h3 = hb3 < h3 ? hb3 : h3;
                    // tightening: n_k = n_{k-1} + n_{k+1}
                    const T s0 = h5 + h1, s1 = h0 + h2, s2 = h1 + h3, s3 = h2 + h4, s4 = h3 + h5, s5 = h4 + h0;
                    const T t0 = s0 < h0 ? s0 : h0, t1 = s1 < h1 ? s1 : h1, t2 = s2 < h2 ? s2 : h2;
                    const T t3 = s3 < h3 ? s3 : h3, t4 = s4 < h4 ? s4 : h4, t5 = s5 < h5 ? s5 : h5;
                    // L_k sin 60 = t_{k-1} + t_{k+1} - t_k ( 2 cos 60 = 1 ); weight = L / 2
                    const T L0 = ( t5 + t1 - t0 ), L1 = ( t0 + t2 - t1 ), L2 = ( t1 + t3 - t2 );
                    const T L3 = ( t2 + t4 - t3 ), L4 = ( t3 + t5 - t4 ), L5 = ( t4 + t0 - t5 );
                    ar[ x ] = c * ( t0 * L0 + t1 * L1 + t2 * L2 + t3 * L3 + t4 * L4 + t5 * L5 );
                    w[ 0 ][ x ] = c * L0; w[ 1 ][ x ] = c * L1; w[ 2 ][ x ] = c * L2;
                    w[ 3 ][ x ] = c * L3; w[ 4 ][ x ] = c * L4; w[ 5 ][ x ] = c * L5;
                }
                const long o = long( r ) * W + b0;
                for( int x = 0; x < n; ++x ) area[ o + x ] = ar[ x ];
                for( int k = 0; k < 6; ++k ) for( int x = 0; x < n; ++x ) wgt[ ( o + x ) * 6 + k ] = w[ k ][ x ];
            }
        }
    }
};

// ---------------------------------------------------------------- square, ring 1, same style
// directions ( angle 45 k ): 0 = +x, 1 = ( 1, 1 ), 2 = +y, ... ; offsets in units of the unit normals
struct Sq1 {
    static void run( int N, const T *pp, T *area, T *wgt ) {
        const int P = N + 2;
        const T r2 = std::sqrt( T( 2 ) ), ir2 = 1 / r2;
        for( int a = 0; a < N; ++a ) {
            const T *pi = pp + ( a + 1 ) * P + 1;
            const T *pu = pi + P, *pd = pi - P; // a + 1, a - 1
            for( int b0 = 0; b0 < N; b0 += B ) {
                const int n = std::min( B, N - b0 );
                alignas( 64 ) T ar[ B ], w[ 8 ][ B ];
                #pragma omp simd
                for( int x = 0; x < B; ++x ) {
                    const int i = b0 + ( x < n ? x : 0 );
                    const T p = pi[ i ];
                    // axis neighbours |k| = 1: h = ( 1 + dp ) / 2; diagonals |k| = sqrt 2: h = ( 2 + dp ) / ( 2 sqrt 2 )
                    T h[ 8 ];
                    h[ 0 ] = T( 0.5 ) * ( T( 1 ) + ( p - pu[ i ] ) );
                    h[ 1 ] = ( T( 2 ) + ( p - pu[ i + 1 ] ) ) * ( T( 0.5 ) * ir2 );
                    h[ 2 ] = T( 0.5 ) * ( T( 1 ) + ( p - pi[ i + 1 ] ) );
                    h[ 3 ] = ( T( 2 ) + ( p - pd[ i + 1 ] ) ) * ( T( 0.5 ) * ir2 );
                    h[ 4 ] = T( 0.5 ) * ( T( 1 ) + ( p - pd[ i ] ) );
                    h[ 5 ] = ( T( 2 ) + ( p - pd[ i - 1 ] ) ) * ( T( 0.5 ) * ir2 );
                    h[ 6 ] = T( 0.5 ) * ( T( 1 ) + ( p - pi[ i - 1 ] ) );
                    h[ 7 ] = ( T( 2 ) + ( p - pu[ i - 1 ] ) ) * ( T( 0.5 ) * ir2 );
                    const T bb = T( b0 + x );
                    const T hb0 = T( N ) - T( 0.5 ) - T( a ), hb4 = T( a ) + T( 0.5 ), hb2 = T( N ) - T( 0.5 ) - bb, hb6 = bb + T( 0.5 );
                    h[ 0 ] = hb0 < h[ 0 ] ? hb0 : h[ 0 ]; h[ 4 ] = hb4 < h[ 4 ] ? hb4 : h[ 4 ];
                    h[ 2 ] = hb2 < h[ 2 ] ? hb2 : h[ 2 ]; h[ 6 ] = hb6 < h[ 6 ] ? hb6 : h[ 6 ];
                    // tightening: the 24 pairs of StencilTables<1>, written with unit normals ( 3 per direction ):
                    //   n_k = ( n_{k-1} + n_{k+1} ) / ( 2 cos 45 ),  n_k = ( n_{k-2} + n_{k+2} ) / ( 2 cos 90 ... ) -> for axis k:
                    //   axis:     n_k = ( n_{k-1} + n_{k+1} ) / sqrt 2, n_k = sqrt 2 n_{k-1} - n_{k-2}, n_k = sqrt 2 n_{k+1} - n_{k+2}
                    // ( pairs with a negative coefficient are not pairs: the 3 positive ones per direction are below )
                    T t[ 8 ];
                    for( int k = 0; k < 8; ++k ) {
                        const T hm1 = h[ ( k + 7 ) & 7 ], hp1 = h[ ( k + 1 ) & 7 ], hm2 = h[ ( k + 6 ) & 7 ], hp2 = h[ ( k + 2 ) & 7 ];
                        const T hm3 = h[ ( k + 5 ) & 7 ], hp3 = h[ ( k + 3 ) & 7 ];
                        // n_k = ( n_{k-1} + n_{k+1} ) ir2, n_k = r2 n_{k-1} + n_{k+2}... ( angles 45: n_{k+-2} orthogonal )
                        T m = h[ k ];
                        T c1 = ( hm1 + hp1 ) * ir2;            m = c1 < m ? c1 : m;
                        T c2 = r2 * hm1 + hp2;                 m = c2 < m ? c2 : m; // n_k = sqrt2 n_{k-1} + n_{k+2} ? see check
                        T c3 = r2 * hp1 + hm2;                 m = c3 < m ? c3 : m;
                        (void)hm3; (void)hp3;
                        t[ k ] = m;
                    }
                    // L_k sin 45 = t_{k-1} + t_{k+1} - 2 cos 45 t_k
                    T a2 = 0;
                    for( int k = 0; k < 8; ++k ) {
                        const T L = ( t[ ( k + 7 ) & 7 ] + t[ ( k + 1 ) & 7 ] - r2 * t[ k ] ) * r2;
                        a2 += t[ k ] * L;
                        w[ k ][ x ] = T( 0.5 ) * L * ( ( k & 1 ) ? ir2 : T( 1 ) );
                    }
                    ar[ x ] = T( 0.5 ) * a2;
                }
                const long o = long( a ) * N + b0;
                for( int x = 0; x < n; ++x ) area[ o + x ] = ar[ x ];
                for( int k = 0; k < 8; ++k ) for( int x = 0; x < n; ++x ) wgt[ ( o + x ) * 8 + k ] = w[ k ][ x ];
            }
        }
    }
};

int main( int argc, char **argv ) {
    if( argc > 3 ) { // check mode: hexbench W R file -> reads psi ( R x W ), writes areas and weights of the inner rows to file.out
        const int W = std::atoi( argv[ 1 ] ), R = std::atoi( argv[ 2 ] );
        std::vector<T> ph( ( W + 2 ) * ( R + 2 ), -inf ), ah( W * R ), wh( W * R * 6 );
        FILE *f = std::fopen( argv[ 3 ], "rb" );
        for( int r = 0; r < R; ++r ) for( int c = 0; c < W; ++c ) if( std::fread( &ph[ ( r + 1 ) * ( W + 2 ) + c + 1 ], sizeof( T ), 1, f ) != 1 ) return 1;
        std::fclose( f );
        Hex1::run( W, R, ph.data(), ah.data(), wh.data() );
        f = std::fopen( ( std::string( argv[ 3 ] ) + ".out" ).c_str(), "wb" );
        std::fwrite( ah.data(), sizeof( T ), ah.size(), f ); std::fwrite( wh.data(), sizeof( T ), wh.size(), f );
        std::fclose( f );
        return 0;
    }
    const int N = argc > 1 ? std::atoi( argv[ 1 ] ) : 512;
    const int reps = argc > 2 ? std::atoi( argv[ 2 ] ) : 7;
    const int W = int( std::lround( N * std::sqrt( std::sqrt( 3.0 ) / 2 ) ) ), R = int( std::lround( double( N ) * N / W ) );
    std::mt19937 rng( 0 );
    std::normal_distribution<T> nd;
    for( double amp : { 0.0, 0.05, 0.1 } ) {
        // square
        std::vector<T> ps( ( N + 2 ) * ( N + 2 ), -inf ), as( N * N ), ws( N * N * 8 ), a0( N * N ), h0( N * N * 8 );
        std::vector<T> ps1( ( N + 2 ) * ( N + 2 ), -inf );
        std::vector<signed char> own( N * N * 8 );
        for( int a = 0; a < N; ++a ) for( int b = 0; b < N; ++b ) ps[ ( a + 1 ) * ( N + 2 ) + b + 1 ] = amp * nd( rng );
        double tsq = best_ns( long( N ) * N, reps, [ & ] { Sq1::run( N, ps.data(), as.data(), ws.data() ); } );
        double tsk = best_ns( long( N ) * N, reps, [ & ] { StencilKernel<1, 32, T>::run( N, ps.data(), a0.data(), h0.data(), own.data() ); } );
        double esq = 0;
        for( int i = 0; i < N * N; ++i ) esq = std::max( esq, std::abs( as[ i ] - a0[ i ] ) );
        // hexagonal
        std::vector<T> ph( ( W + 2 ) * ( R + 2 ), -inf ), ah( W * R ), wh( W * R * 6 );
        for( int r = 0; r < R; ++r ) for( int c = 0; c < W; ++c ) ph[ ( r + 1 ) * ( W + 2 ) + c + 1 ] = amp * nd( rng );
        double thx = best_ns( long( W ) * ( R - 2 ), reps, [ & ] { Hex1::run( W, R, ph.data(), ah.data(), wh.data() ); } );
        double sum = 0;
        for( int i = W; i < W * ( R - 1 ); ++i ) sum += ah[ i ];
        std::printf( "psi noise %.2f | square %dx%d: same-style %6.2f ns/cell, grid_stencil StencilKernel<1> %6.2f ( max|da| %.1e ) "
                     "| hex %dx%d inner rows: %6.2f ns/cell ( area sum %.6f, expected %.6f )\n",
                     amp, N, N, tsq, tsk, esq, W, R, thx, sum, double( W ) * ( R - 2 ) * std::sqrt( 3.0 ) / 2 );
    }
}
