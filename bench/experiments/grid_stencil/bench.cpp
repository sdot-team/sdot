// Micro-benchmark: branch-free fixed stencil vs branchy clipping, seeds on a grid.
//   bench N [ reps ]   ( single thread )
// build ( from this folder ):
//   c++ -std=c++20 -O3 -march=native -fno-math-errno -fopenmp-simd -I ../../../src/sdot/include -I ../../../../loom/src/loom/include bench.cpp -o bench
// tables: micromamba run -n sdot python proto/gen_tables.py stencil_tables.h
#include "stencil_kernel.h"
#include "clip_kernel.h"
#include "sdot_kernel.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <string>
#include <functional>

#ifndef TYPE
#define TYPE double
#endif
#ifndef BLK
#define BLK 32
#endif
using T = TYPE;

template<int R>
std::vector<T> pad( int N, const std::vector<T> &psi ) {
    const int P = N + 2 * R;
    std::vector<T> p( P * P, -std::numeric_limits<T>::infinity() );
    for( int a = 0; a < N; ++a ) for( int b = 0; b < N; ++b ) p[ ( a + R ) * P + b + R ] = psi[ a * N + b ];
    return p;
}

double best_ns_per_cell( int N, int reps, const std::function<void()> &f ) {
    double best = 1e30;
    for( int r = 0; r < reps; ++r ) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        auto t1 = std::chrono::steady_clock::now();
        best = std::min( best, std::chrono::duration<double, std::nano>( t1 - t0 ).count() / ( double( N ) * N ) );
    }
    return best;
}

template<int R>
void one( const char *name, int N, int reps, const std::vector<T> &psi ) {
    using SK = StencilKernel<R, BLK, T>;
    constexpr int D = SK::D;
    ClipKernel<R, T> ck;
    SdotKernel<R, T> sk;
    auto pp = pad<R>( N, psi );
    std::vector<T> a0( N * N ), a1( N * N ), h0( N * N * D ), h1( N * N * ck.nb_offsets ), a2( N * N ), h2( N * N * ck.nb_offsets ), verts( N * N * D * 2 );
    std::vector<unsigned char> flag( N * N );
    std::vector<signed char> own( N * N * D );

    double ts = best_ns_per_cell( N, reps, [ & ] { SK::run( N, pp.data(), a0.data(), h0.data(), own.data() ); } );
    double tc = best_ns_per_cell( N, reps, [ & ] { ck.run( N, pp.data(), a1.data(), h1.data() ); } );
    double tq = best_ns_per_cell( N, reps, [ & ] { sk.run( N, pp.data(), a2.data(), h2.data() ); } );
    int nflag = 0;
    double tv = best_ns_per_cell( N, reps, [ & ] { SK::run( N, pp.data(), a0.data(), h0.data(), own.data(), verts.data() );
                                                    nflag = SK::certify( N, own.data(), verts.data(), flag.data(), T( sizeof( T ) == 4 ? 1e-4 : 1e-9 ) ); } );
    double es = 0;
    for( int i = 0; i < N * N; ++i ) es = std::max( es, double( std::abs( a2[ i ] - a1[ i ] ) ) );
    for( int i = 0; i < N * N * ck.nb_offsets; ++i ) es = std::max( es, double( std::abs( h2[ i ] - h1[ i ] ) ) );

    // agreement: area and hessian weight per neighbor offset
    double ea = 0, eh = 0, tot = 0;
    for( int i = 0; i < N * N; ++i ) {
        ea = std::max( ea, double( std::abs( a0[ i ] - a1[ i ] ) ) );
        tot += a0[ i ];
        T w[ ( 2 * R + 1 ) * ( 2 * R + 1 ) ] = {};
        for( int k = 0; k < D; ++k ) {
            int o = own[ i * D + k ];
            if( o < 0 ) continue;
            int x = StencilTables<R>::off_x[ o ] + R, y = StencilTables<R>::off_y[ o ] + R;
            w[ x * ( 2 * R + 1 ) + y ] += h0[ i * D + k ];
        }
        for( int o = 0; o < ck.nb_offsets; ++o ) {
            int x = ck.ox[ o ] + R, y = ck.oy[ o ] + R;
            eh = std::max( eh, double( std::abs( w[ x * ( 2 * R + 1 ) + y ] - h1[ i * ck.nb_offsets + o ] ) ) );
        }
    }
    std::printf( "%-10s R=%d N=%5d  stencil %6.2f  +certif %6.2f  sdot %6.2f  clip %6.2f ns/cell  sdot/stencil %5.2fx  | flagged %d  max|d| %.1e %.1e sdot-clip %.1e\n",
                 name, R, N, ts, tv, tq, tc, tq / ts, nflag, ea, eh, es );
}

int main( int argc, char **argv ) {
    int N = argc > 1 ? std::atoi( argv[ 1 ] ) : 512;
    int reps = argc > 2 ? std::atoi( argv[ 2 ] ) : 5;
    std::mt19937 rng( 0 );
    std::normal_distribution<T> nd;
    std::vector<std::pair<std::string, std::vector<T>>> cases;
    std::vector<T> psi( N * N );
    cases.push_back( { "constant", psi } );
    for( int a = 0; a < N; ++a ) for( int b = 0; b < N; ++b ) psi[ a * N + b ] = 2.0 * std::sin( 2 * M_PI * a / 64 ) * std::sin( 2 * M_PI * b / 48 );
    cases.push_back( { "smooth", psi } );
    for( auto &v : psi ) v = 0.1 * nd( rng );
    cases.push_back( { "noise0.1", psi } );
    for( auto &[ name, p ] : cases ) {
        one<1>( name.c_str(), N, reps, p );
        one<2>( name.c_str(), N, reps, p );
    }
}
