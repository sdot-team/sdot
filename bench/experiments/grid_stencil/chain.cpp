// The whole chain on N = 512 grids: ring-R stencil ( branch-free ) -> vectorized certificate ->
// exact fallback on the flagged cells. Checked against the exact kernel on every cell.
//
// psi: the OT solutions of psi_cases.bin ( N = 32, from proto/export_psi.py ) mirrored 16 x 16
// times; reflecting seeds across the faces of a box reproduces the box-clipped diagram, so every
// tile holds the same cells as the 32 x 32 solve.
//
// build ( from this folder ):
//   c++ -std=c++20 -O3 -march=native -fno-math-errno -fopenmp-simd chain.cpp -o chain
#include "stencil_kernel.h"
#include "exact_kernel.h"
#include <chrono>
#include <cstdio>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

using T = double;
using Clock = std::chrono::steady_clock;

double ms( const std::function<void()> &f, int reps ) {
    double best = 1e30;
    for( int r = 0; r < reps; ++r ) {
        auto t0 = Clock::now();
        f();
        best = std::min( best, std::chrono::duration<double, std::milli>( Clock::now() - t0 ).count() );
    }
    return best;
}

std::vector<T> pad( int N, int R, const std::vector<T> &psi ) {
    const int P = N + 2 * R;
    std::vector<T> p( P * P, -std::numeric_limits<T>::infinity() );
    for( int a = 0; a < N; ++a ) for( int b = 0; b < N; ++b ) p[ ( a + R ) * P + b + R ] = psi[ a * N + b ];
    return p;
}

struct Reference {
    std::vector<T> area;
    std::vector<std::vector<std::pair<int,T>>> faces;
};

template<int R>
void chain( const std::string &name, int N, const std::vector<T> &psi, const ExactKernel<T> &ek, const Reference &ref, double t_ref, int reps ) {
    using SK = StencilKernel<R, 32, T>;
    constexpr int D = SK::D;
    auto pp = pad( N, R, psi ), pe = pad( N, ek.RB, psi );
    const T pmax = *std::max_element( psi.begin(), psi.end() );
    std::vector<T> area( N * N ), hess( N * N * D );
    std::vector<signed char> own( N * N * D );
    std::vector<unsigned char> flag( N * N ), flag_scalar( N * N );
    std::vector<T> verts( N * N * D * 2 );
    typename SK::Planes pl( N );
    std::vector<T> fb_area;
    std::vector<std::vector<std::pair<int,T>>> fb_faces;
    std::vector<int> flagged;

    const T tol = 1e-9;
    int nflag = 0;
    double t_st = ms( [ & ] { SK::run( N, pp.data(), area.data(), hess.data(), own.data(), nullptr, &pl ); }, reps );
    double t_ce = ms( [ & ] { nflag = SK::certify_simd( pl, flag.data(), tol ); }, reps );
    // fused pass on rolling planes ( same flags expected )
    typename SK::Planes rp( N, true );
    std::vector<unsigned char> flag_fused( N * N );
    int nflag_fused = 0;
    double t_fu = ms( [ & ] { nflag_fused = SK::run_certified( N, pp.data(), area.data(), hess.data(), own.data(), rp, flag_fused.data(), tol ); }, reps );
    for( int i = 0; i < N * N; ++i ) nflag_fused += 1000000 * ( flag_fused[ i ] != flag[ i ] );
    flagged.clear();
    for( int i = 0; i < N * N; ++i ) if( flag[ i ] ) flagged.push_back( i );
    int exhausted = 0;
    double t_fb = ms( [ & ] {
        fb_area.assign( flagged.size(), 0 );
        fb_faces.assign( flagged.size(), {} );
        exhausted = 0;
        for( size_t f = 0; f < flagged.size(); ++f )
            exhausted += ! ek.cell( N, pe.data(), pmax, flagged[ f ] / N, flagged[ f ] % N, fb_area[ f ], fb_faces[ f ] );
    }, reps );

    // scalar certificate must agree
    SK::run( N, pp.data(), area.data(), hess.data(), own.data(), verts.data(), &pl );
    SK::certify( N, own.data(), verts.data(), flag_scalar.data(), tol );
    int disagree = 0;
    for( int i = 0; i < N * N; ++i ) disagree += flag[ i ] != flag_scalar[ i ];

    // chain result vs reference
    std::vector<T> res_area = area;
    for( size_t f = 0; f < flagged.size(); ++f ) res_area[ flagged[ f ] ] = fb_area[ f ];
    double ea = 0, eh = 0;
    int wrong = 0;
    for( int i = 0; i < N * N; ++i ) {
        std::map<int,T> got, exp;
        for( auto &[ j, w ] : ref.faces[ i ] ) exp[ j ] += w;
        if( flag[ i ] ) {
            size_t f = std::lower_bound( flagged.begin(), flagged.end(), i ) - flagged.begin();
            for( auto &[ j, w ] : fb_faces[ f ] ) got[ j ] += w;
        } else {
            for( int k = 0; k < D; ++k ) {
                int o = own[ i * D + k ];
                if( o < 0 || hess[ i * D + k ] == 0 ) continue;
                got[ i + StencilTables<R>::off_x[ o ] * N + StencilTables<R>::off_y[ o ] ] += hess[ i * D + k ];
            }
        }
        double e = std::abs( res_area[ i ] - ref.area[ i ] );
        for( auto &[ j, w ] : exp ) e = std::max( e, std::abs( double( w - ( got.count( j ) ? got[ j ] : 0 ) ) ) );
        for( auto &[ j, w ] : got ) if( ! exp.count( j ) ) e = std::max( e, std::abs( double( w ) ) );
        ea = std::max( ea, std::abs( double( res_area[ i ] - ref.area[ i ] ) ) );
        eh = std::max( eh, e );
        wrong += e > 1e-8;
    }

    const double nn = double( N ) * N, tot = t_fu + t_fb;
    std::printf( "%-15s R=%d flagged %5.2f%%  2 passes: stencil %5.1f + certif %5.1f | fused %5.1f + fallback %6.1f = %6.1f ns/cell | exact everywhere %6.1f ( x%5.2f ) | wrong %d  max err %.1e  disagree simd/scalar %d fused %d  exhausted %d\n",
                 name.c_str(), R, 100.0 * nflag / nn, t_st * 1e6 / nn, t_ce * 1e6 / nn, t_fu * 1e6 / nn, t_fb * 1e6 / nn, tot * 1e6 / nn, t_ref * 1e6 / nn, t_ref / tot,
                 wrong, std::max( ea, eh ), disagree, nflag_fused / 1000000, exhausted );
}

int main( int argc, char **argv ) {
    const int reps = argc > 1 ? std::atoi( argv[ 1 ] ) : 3;
    std::vector<std::pair<std::string, std::vector<T>>> cases;
    int N = 0;
    {
        FILE *f = std::fopen( "psi_cases.bin", "rb" );
        int hdr[ 2 ];
        std::fread( hdr, 4, 2, f );
        const int nc = hdr[ 0 ], n = hdr[ 1 ], rep = 16;
        N = n * rep;
        for( int c = 0; c < nc; ++c ) {
            char name[ 33 ] = {};
            std::fread( name, 1, 32, f );
            std::vector<T> psi( n * n );
            std::vector<int> rings( n * n );
            std::fread( psi.data(), 8, n * n, f );
            std::fread( rings.data(), 4, n * n, f );
            std::string s( name );
            s = s.substr( 0, s.find_last_not_of( ' ' ) + 1 );
            if( s.find( "_it1" ) != std::string::npos && s.size() == s.find( "_it1" ) + 4 ) continue; // first iterates are trivially ring 1
            std::vector<T> big( N * N );
            for( int a = 0; a < N; ++a ) for( int b = 0; b < N; ++b ) {
                int ta = a / n, tb = b / n, la = a % n, lb = b % n;
                if( ta % 2 ) la = n - 1 - la;
                if( tb % 2 ) lb = n - 1 - lb;
                big[ a * N + b ] = psi[ la * n + lb ];
            }
            cases.push_back( { s, big } );
        }
    }
    {
        std::mt19937 rng( 0 );
        std::normal_distribution<T> nd;
        std::vector<T> psi( N * N );
        for( int a = 0; a < N; ++a ) for( int b = 0; b < N; ++b ) psi[ a * N + b ] = 2.0 * std::sin( 2 * M_PI * a / 64 ) * std::sin( 2 * M_PI * b / 48 );
        cases.insert( cases.begin(), { "smooth", psi } );
        for( auto &v : psi ) v = 0.1 * nd( rng );
        cases.insert( cases.begin() + 1, { "noise0.1", psi } );
    }

    ExactKernel<T> ek( 24 );
    std::printf( "N = %d, 1 thread, times in ns/cell ( best of %d )\n", N, reps );
    for( auto &[ name, psi ] : cases ) {
        Reference ref;
        ref.area.resize( N * N );
        ref.faces.resize( N * N );
        auto pe = pad( N, ek.RB, psi );
        const T pmax = *std::max_element( psi.begin(), psi.end() );
        double t_ref = ms( [ & ] {
            for( int i = 0; i < N * N; ++i ) { ref.faces[ i ].clear(); ek.cell( N, pe.data(), pmax, i / N, i % N, ref.area[ i ], ref.faces[ i ] ); }
        }, 1 );
        chain<1>( name, N, psi, ek, ref, t_ref, reps );
        chain<2>( name, N, psi, ek, ref, t_ref, reps );
    }
}
