// Ring 1 / ring 2 in registers ( asimd ): hexagonal lattice ( HexRingReg.h ) vs square grid ( sdot/grid/Ring1Reg.h, Ring2Reg.h ),
// as many cells, single thread.
//   hexreg_bench N [ reps ]                     the bench ( square N x N, hexagonal Wc x R, Wc a multiple of B )
//   hexreg_bench Wc R psi_file                  check mode: writes psi_file.out = ring-2 areas, ring-2 weight sums ( seeds only ),
//                                               ring-1 areas ( inner rows, else 0 ), ring 1 + prediction + ring 2 areas, all R x Wc
// build ( from this folder ):
//   c++ -std=c++20 -O3 -march=native -fno-math-errno -I ../../../src/sdot/include hexreg_bench.cpp -o hexreg_bench
#include "HexRingReg.h"
#include <sdot/grid/Ring1Reg.h>
#include <sdot/grid/Ring2Reg.h>
#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

using T = double;
using U = hexgrid::U;
constexpr int B = 16;
using H = hexgrid::HexRingReg<B,T>;
static const T inf = std::numeric_limits<T>::infinity();
// every output of a call must exist in memory ( else the compiler drops what is not read )
inline void escape( const void *p ) { asm volatile( "" : : "r"( p ) : "memory" ); }

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

struct Hex {
    int Wc, R, P;
    std::vector<T> pp, a2, w2, a1, a12;
    long ncut = 0;
    Hex( int Wc, int R ) : Wc( Wc ), R( R ), P( ( Wc + B - 1 ) / B * B + 4 ), pp( size_t( R + 4 ) * P, -inf ),
        a2( size_t( R ) * Wc ), w2( size_t( R ) * Wc ), a1( size_t( R ) * Wc ), a12( size_t( R ) * Wc ) {}
    T &psi( int r, int c ) { return pp[ size_t( r + 2 ) * P + c + 2 ]; }
    T x( int r, int c ) const { return c + 0.5 * ( r & 1 ); }
    T y( int r ) const { return r * std::sqrt( 3.0 ) / 2; }

    void ring2( bool keep ) {
        H::Out2 out;
        for( int r = 0; r < R; ++r )
            for( int c0 = 0; c0 < Wc; c0 += B ) {
                if( r & 1 ) H::ring2<1>( Wc, R, pp.data(), P, r, c0, out ); else H::ring2<0>( Wc, R, pp.data(), P, r, c0, out );
                escape( &out );
                if( keep )
                    for( int x = 0; x < std::min( B, Wc - c0 ); ++x ) {
                        a2[ size_t( r ) * Wc + c0 + x ] = ( out.inv >> x ) & 1 ? 0 : out.area[ x ];
                        T s = 0;
                        for( int k = 0; k < 12; ++k ) if( ! HEX_BITS || ( ( out.fin[ k ] >> x ) & 1 ) ) s += out.wk[ k ][ x ];
                        w2[ size_t( r ) * Wc + c0 + x ] = s;
                    }
            }
    }
    void ring1( bool keep ) {
        H::Out1 out;
        for( int r = 1; r < R - 1; ++r )
            for( int c0 = 0; c0 < Wc; c0 += B ) {
                if( r & 1 ) H::ring1<1,false>( Wc, R, pp.data(), P, r, c0, out ); else H::ring1<0,false>( Wc, R, pp.data(), P, r, c0, out );
                escape( &out );
                if( keep ) for( int x = 0; x < std::min( B, Wc - c0 ); ++x ) a1[ size_t( r ) * Wc + c0 + x ] = out.area[ x ];
            }
    }
    // ring 1 + prediction, ring 2 on the cut cells ( lists per row ); areas always written ( a12 )
    template<int SH> void row12( int r, H::Out1 &o1, H::Out2 &o2, int *pend ) {
        int np = 0;
        auto flush = [&] {
            H::ring2_list<SH>( Wc, R, pp.data(), P, r, pend, np, o2 );
            escape( &o2 );
            for( int x = 0; x < np; ++x ) a12[ size_t( r ) * Wc + pend[ x ] ] = ( o2.inv >> x ) & 1 ? 0 : o2.area[ x ];
            ncut += np; np = 0;
        };
        for( int c0 = 0; c0 < Wc; c0 += B ) {
            H::ring1<SH,true>( Wc, R, pp.data(), P, r, c0, o1 );
            escape( &o1 );
            const int n = std::min( B, Wc - c0 );
            for( int x = 0; x < n; ++x ) a12[ size_t( r ) * Wc + c0 + x ] = ( o1.inv >> x ) & 1 ? 0 : o1.area[ x ];
            for( U m = o1.cut & ~o1.inv & ( n == 64 ? ~0ull : ( 1ull << n ) - 1 ); m; m &= m - 1 ) {
                pend[ np++ ] = c0 + std::countr_zero( m );
                if( np == B ) flush();
            }
        }
        if( np ) flush();
    }
    // the first / last rows in ring 2 ( the +-y box )
    void ring12() {
        H::Out1 o1; H::Out2 o2;
        alignas( 64 ) int pend[ B ];
        ncut = 0;
        for( int r : { 0, R - 1 } )
            for( int c0 = 0; c0 < Wc; c0 += B ) {
                if( r & 1 ) H::ring2<1>( Wc, R, pp.data(), P, r, c0, o2 ); else H::ring2<0>( Wc, R, pp.data(), P, r, c0, o2 );
                escape( &o2 );
                for( int x = 0; x < std::min( B, Wc - c0 ); ++x ) a12[ size_t( r ) * Wc + c0 + x ] = ( o2.inv >> x ) & 1 ? 0 : o2.area[ x ];
            }
        for( int r = 1; r < R - 1; ++r ) { if( r & 1 ) row12<1>( r, o1, o2, pend ); else row12<0>( r, o1, o2, pend ); }
    }
};

int main( int argc, char **argv ) {
    if( argc > 3 ) {
        const int Wc = std::atoi( argv[ 1 ] ), R = std::atoi( argv[ 2 ] );
        Hex hx( Wc, R );
        FILE *f = std::fopen( argv[ 3 ], "rb" );
        for( int r = 0; r < R; ++r ) for( int c = 0; c < Wc; ++c ) if( std::fread( &hx.psi( r, c ), sizeof( T ), 1, f ) != 1 ) return 1;
        std::fclose( f );
        hx.ring2( true ); hx.ring1( true ); hx.ring12();
        f = std::fopen( ( std::string( argv[ 3 ] ) + ".out" ).c_str(), "wb" );
        for( auto *v : { &hx.a2, &hx.w2, &hx.a1, &hx.a12 } ) std::fwrite( v->data(), sizeof( T ), v->size(), f );
        std::fclose( f );
        std::printf( "cut %ld fallbacks %ld\n", hx.ncut, H::fallbacks );
        return 0;
    }
    const int N = argc > 1 ? std::atoi( argv[ 1 ] ) : 1024;
    const int reps = argc > 2 ? std::atoi( argv[ 2 ] ) : 7;
    const int Wc = int( std::lround( N * std::sqrt( std::sqrt( 3.0 ) / 2 ) / B ) ) * B, R = int( std::lround( double( N ) * N / Wc ) );
    std::mt19937 rng( 0 );
    std::normal_distribution<T> nd;
    std::printf( "square %d x %d, hexagonal %d x %d, B = %d, SIMD width %d; ns/cell, single thread\n", N, N, Wc, R, B, H::W );
    // psi = smooth ( amplitude A, periods 64 x 48 seeds ) + noise
    struct Case { const char *name; double A, noise; };
    for( Case cs : { Case{ "flat", 0, 0 }, Case{ "noise 0.05", 0, 0.05 }, Case{ "noise 0.1", 0, 0.1 }, Case{ "noise 0.3", 0, 0.3 },
                     Case{ "smooth 2", 2, 0 }, Case{ "smooth 20", 20, 0 }, Case{ "smooth 20 + 0.1", 20, 0.1 } } ) {
        auto fpsi = [ & ]( double x, double y ) { return cs.A * std::sin( 2 * M_PI * x / 64 ) * std::sin( 2 * M_PI * y / 48 ) + cs.noise * nd( rng ); };
        std::vector<T> p1( size_t( N + 2 ) * ( N + 2 ), -inf ), p2( size_t( N + 4 ) * ( N + 4 ), -inf );
        for( int a = 0; a < N; ++a ) for( int b = 0; b < N; ++b ) {
            const T v = fpsi( a, b );
            p1[ size_t( a + 1 ) * ( N + 2 ) + b + 1 ] = v; p2[ size_t( a + 2 ) * ( N + 4 ) + b + 2 ] = v;
        }
        const double ts1 = best_ns( long( N ) * N, reps, [ & ] {
            alignas( 64 ) T vx[ 8 ][ B ], vy[ 8 ][ B ], ar[ B ];
            unsigned long long ok[ 8 ];
            for( int a = 0; a < N; ++a ) for( int b0 = 0; b0 < N; b0 += B ) {
                sdot::grid::Ring1Reg<B,T>::block( N, p1.data(), a, b0, vx, vy, ok, ar );
                escape( ar ); escape( vx ); escape( vy ); escape( ok );
            }
        } );
        const double ts1p = best_ns( long( N ) * N, reps, [ & ] {
            alignas( 64 ) T vx[ 8 ][ B ], vy[ 8 ][ B ], ar[ B ];
            unsigned long long ok[ 8 ];
            typename sdot::grid::Ring1Reg<B,T>::Ext ex;
            ex.p2 = p2.data();
            for( int a = 0; a < N; ++a ) for( int b0 = 0; b0 < N; b0 += B ) {
                sdot::grid::Ring1Reg<B,T>::block( N, p1.data(), a, b0, vx, vy, ok, ar, &ex );
                escape( ar ); escape( vx ); escape( vy ); escape( ok ); escape( &ex );
            }
        } );
        const double ts2 = best_ns( long( N ) * N, reps, [ & ] {
            alignas( 64 ) T vx[ 16 ][ B ], vy[ 16 ][ B ], ar[ B ];
            signed char own[ 16 ][ B ];
            int cells[ B ];
            for( int a = 0; a < N; ++a ) for( int b0 = 0; b0 < N; b0 += B ) {
                for( int x = 0; x < B; ++x ) cells[ x ] = a * N + b0 + x;
                sdot::grid::Ring2Reg<B,T>::cells_verts( N, p2.data(), cells, B, vx, vy, own, ar );
                escape( ar ); escape( vx ); escape( vy ); escape( own );
            }
        } );
        Hex hx( Wc, R );
        for( int r = 0; r < R; ++r ) for( int c = 0; c < Wc; ++c ) hx.psi( r, c ) = fpsi( hx.x( r, c ), hx.y( r ) );
        const double th1 = best_ns( long( Wc ) * ( R - 2 ), reps, [ & ] { hx.ring1( false ); } );
        H::fallbacks = 0;
        const double th2 = best_ns( long( Wc ) * R, reps, [ & ] { hx.ring2( false ); } );
        const double fb = 100.0 * H::fallbacks / reps / ( double( Wc ) * R / H::W );
        const double th12 = best_ns( long( Wc ) * R, reps, [ & ] { hx.ring12(); } );
        std::printf( "%-16s | square ring 1 %6.2f  ring 1 + pred %6.2f  ring 2 %6.2f | hex ring 1 %6.2f  ring 2 %6.2f ( fallback %5.2f %% )  ring 1 + pred + ring 2 on cut %6.2f ( cut %5.2f %% )\n",
                     cs.name, ts1, ts1p, ts2, th1, th2, fb, th12, 100.0 * hx.ncut / ( double( Wc ) * ( R - 2 ) ) );
    }
}
