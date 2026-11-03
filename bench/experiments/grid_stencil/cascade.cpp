// Cascade on N = 512 grids ( same psi as chain.cpp ):
//   1. fused ring-1 stencil + certificate on every cell
//   2. ring 2 on the flagged cells, either
//      - by blocks: the blocks of B cells that hold a flagged cell, dilated by 2 rows and one block
//        on each side ( the ring-2 neighbors of every flagged cell ), then the ring-2 certificate
//        on the flagged cells only, or
//      - by cells: ring 2 on the flagged cells only ( gathered ), then a cross-level certificate on
//        the flagged cells and their ring-1 neighbors ( each cell is compared with the planes of the
//        level its neighbor ended at )
//   3. exact fallback on the cells still flagged
// Checked against the exact kernel on every cell.
//
// build ( from this folder ):
//   c++ -std=c++20 -O3 -march=native -fno-math-errno -fopenmp-simd cascade.cpp -o cascade
// needs psi_cases.bin ( proto/export_psi.py )
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
constexpr int B = 32;
using K1 = StencilKernel<1, B, T>;
using K2 = StencilKernel<2, B, T>;

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

struct Cascade {
    int N, NB;
    std::vector<T> p1, p2, pe;
    T pmax;
    // ring 1
    std::vector<T> area1, hess1;
    std::vector<signed char> own1;
    std::vector<unsigned char> flag1;
    K1::Planes pl1;
    // ring 2
    std::vector<T> area2, hess2;
    std::vector<signed char> own2;
    std::vector<unsigned char> flag2, blk, need;
    K2::Planes pl2;
    // exact
    std::vector<int> left;
    std::vector<T> fb_area;
    std::vector<std::vector<std::pair<int,T>>> fb_faces;
    int n1 = 0, n2 = 0, nblk = 0;

    Cascade( int N, const std::vector<T> &psi, int RB ) : N( N ), NB( N / B ), p1( pad( N, 1, psi ) ), p2( pad( N, 2, psi ) ), pe( pad( N, RB, psi ) ),
        pmax( *std::max_element( psi.begin(), psi.end() ) ),
        area1( N * N ), hess1( N * N * K1::D ), own1( N * N * K1::D ), flag1( N * N ), pl1( N ),
        area2( N * N ), hess2( N * N * K2::D ), own2( N * N * K2::D ), flag2( N * N ), blk( N * NB ), need( N * NB ), pl2( N ) { init_dirs(); }

    void stage1( T tol ) { n1 = K1::run_certified( N, p1.data(), area1.data(), hess1.data(), own1.data(), pl1, flag1.data(), tol ); }

    void stage2( T tol ) {
        // flagged blocks, then dilation by 2 rows and 1 block
        for( int a = 0; a < N; ++a )
            for( int bb = 0; bb < NB; ++bb ) {
                unsigned char f = 0;
                for( int x = 0; x < B; ++x ) f |= flag1[ a * N + bb * B + x ];
                blk[ a * NB + bb ] = f;
            }
        nblk = 0;
        for( int a = 0; a < N; ++a )
            for( int bb = 0; bb < NB; ++bb ) {
                unsigned char f = 0;
                for( int da = -2; da <= 2; ++da )
                    for( int db = -1; db <= 1; ++db ) {
                        const int a2 = a + da, b2 = bb + db;
                        if( a2 >= 0 && a2 < N && b2 >= 0 && b2 < NB ) f |= blk[ a2 * NB + b2 ];
                    }
                need[ a * NB + bb ] = f;
                if( f ) {
                    K2::row_block( N, p2.data(), a, bb * B, B, area2.data() + a * N, hess2.data() + a * N * K2::D, own2.data() + a * N * K2::D, nullptr, &pl2 );
                    ++nblk;
                }
            }
        K2::CertTables ct;
        n2 = 0;
        for( int a = 0; a < N; ++a )
            for( int bb = 0; bb < NB; ++bb )
                if( blk[ a * NB + bb ] ) {
                    K2::certify_row( pl2, ct, a, flag2.data(), tol, bb * B, bb * B + B );
                    for( int x = 0; x < B; ++x ) {
                        const int i = a * N + bb * B + x;
                        flag2[ i ] &= flag1[ i ];
                        n2 += flag2[ i ];
                    }
                }
    }

    // ---- cell granularity ----------------------------------------------------------------------
    std::vector<int> list1, check;
    std::vector<unsigned char> mark;
    int dir1[ 25 ], dir2[ 25 ]; // direction vector ( x + 2 ) * 5 + y + 2 -> index, -1 if absent

    void init_dirs() {
        for( int i = 0; i < 25; ++i ) dir1[ i ] = dir2[ i ] = -1;
        for( int k = 0; k < K1::D; ++k ) dir1[ int( StencilTables<1>::dir_x[ k ] + 2 ) * 5 + int( StencilTables<1>::dir_y[ k ] + 2 ) ] = k;
        for( int k = 0; k < K2::D; ++k ) dir2[ int( StencilTables<2>::dir_x[ k ] + 2 ) * 5 + int( StencilTables<2>::dir_y[ k ] + 2 ) ] = k;
    }

    // cross-level certificate of cell i
    bool cell_ok( int i, T tol ) const {
        const bool l2 = flag1[ i ];
        const int D = l2 ? K2::D : K1::D;
        const T *vx = l2 ? pl2.vx.data() : pl1.vx.data(), *vy = l2 ? pl2.vy.data() : pl1.vy.data();
        const signed char *ow = l2 ? pl2.ow.data() : pl1.ow.data();
        const int PP = l2 ? pl2.plane() : pl1.plane(), ii = ( l2 ? pl2.row( i / N ) + i % N + 2 : pl1.row( i / N ) + i % N + 1 );
        for( int k = 0; k < D; ++k ) {
            const int km = ( k + D - 1 ) % D, o = ow[ k * PP + ii ];
            const T ex = vx[ k * PP + ii ] - vx[ km * PP + ii ], ey = vy[ k * PP + ii ] - vy[ km * PP + ii ];
            if( o < 0 || std::abs( ex ) + std::abs( ey ) <= tol ) continue;
            const int dx = l2 ? StencilTables<2>::off_x[ o ] : StencilTables<1>::off_x[ o ];
            const int dy = l2 ? StencilTables<2>::off_y[ o ] : StencilTables<1>::off_y[ o ];
            const int j = i + dx * N + dy;
            const bool m2 = flag1[ j ];
            const int ndx = -int( l2 ? StencilTables<2>::dir_x[ k ] : StencilTables<1>::dir_x[ k ] );
            const int ndy = -int( l2 ? StencilTables<2>::dir_y[ k ] : StencilTables<1>::dir_y[ k ] );
            const int q = ( m2 ? dir2 : dir1 )[ ( ndx + 2 ) * 5 + ndy + 2 ];
            if( q < 0 ) return false;
            const int Dj = m2 ? K2::D : K1::D, qm = ( q + Dj - 1 ) % Dj;
            const T *wx = m2 ? pl2.vx.data() : pl1.vx.data(), *wy = m2 ? pl2.vy.data() : pl1.vy.data();
            const signed char *wo = m2 ? pl2.ow.data() : pl1.ow.data();
            const int PQ = m2 ? pl2.plane() : pl1.plane(), jj = ( m2 ? pl2.row( j / N ) + j % N + 2 : pl1.row( j / N ) + j % N + 1 );
            const int o2 = wo[ q * PQ + jj ];
            if( o2 < 0 ) return false;
            if( ( m2 ? StencilTables<2>::off_x[ o2 ] : StencilTables<1>::off_x[ o2 ] ) != -dx ) return false;
            if( ( m2 ? StencilTables<2>::off_y[ o2 ] : StencilTables<1>::off_y[ o2 ] ) != -dy ) return false;
            if( std::abs( vx[ km * PP + ii ] - wx[ q * PQ + jj ] - dx ) > tol || std::abs( vy[ km * PP + ii ] - wy[ q * PQ + jj ] - dy ) > tol ) return false;
            if( std::abs( vx[ k * PP + ii ] - wx[ qm * PQ + jj ] - dx ) > tol || std::abs( vy[ k * PP + ii ] - wy[ qm * PQ + jj ] - dy ) > tol ) return false;
        }
        return true;
    }

    void stage2_cells( T tol ) {
        list1.clear();
        for( int i = 0; i < N * N; ++i ) if( flag1[ i ] ) list1.push_back( i );
        for( size_t c = 0; c < list1.size(); c += B )
            K2::cells_block( N, p2.data(), list1.data() + c, int( std::min<size_t>( B, list1.size() - c ) ), area2.data(), hess2.data(), own2.data(), &pl2 );
        // cells to check: the flagged ones and their ring-1 neighbors
        mark.assign( N * N, 0 );
        check.clear();
        for( int i : list1 )
            for( int da = -1; da <= 1; ++da )
                for( int db = -1; db <= 1; ++db ) {
                    const int a = i / N + da, b = i % N + db;
                    if( a < 0 || a >= N || b < 0 || b >= N || mark[ a * N + b ] ) continue;
                    mark[ a * N + b ] = 1;
                    check.push_back( a * N + b );
                }
        std::fill( flag2.begin(), flag2.end(), 0 );
        n2 = 0;
        for( int i : check )
            if( ! cell_ok( i, tol ) ) { flag2[ i ] = 1; ++n2; }
        // every cell whose final result is not a stencil one goes to `left`, sorted
        nblk = check.size();
    }

    void stage3_cells( const ExactKernel<T> &ek ) {
        left.clear();
        for( int i : check ) if( flag2[ i ] ) left.push_back( i );
        std::sort( left.begin(), left.end() );
        fb_area.assign( left.size(), 0 );
        fb_faces.assign( left.size(), {} );
        for( size_t f = 0; f < left.size(); ++f )
            ek.cell( N, pe.data(), pmax, left[ f ] / N, left[ f ] % N, fb_area[ f ], fb_faces[ f ] );
    }

    void stage3( const ExactKernel<T> &ek ) {
        left.clear();
        for( int a = 0; a < N; ++a )
            for( int bb = 0; bb < NB; ++bb )
                if( blk[ a * NB + bb ] )
                    for( int x = 0; x < B; ++x ) {
                        const int i = a * N + bb * B + x;
                        if( flag2[ i ] ) left.push_back( i );
                    }
        fb_area.assign( left.size(), 0 );
        fb_faces.assign( left.size(), {} );
        for( size_t f = 0; f < left.size(); ++f )
            ek.cell( N, pe.data(), pmax, left[ f ] / N, left[ f ] % N, fb_area[ f ], fb_faces[ f ] );
    }

    void reset_flags2() {
        // flag2 is only written on flagged blocks: clear it so that a rerun starts clean
        std::fill( flag2.begin(), flag2.end(), 0 );
    }

    // final result of cell i
    void result( int i, T &area, std::map<int,T> &faces ) const {
        faces.clear();
        if( flag2[ i ] ) {
            size_t f = std::lower_bound( left.begin(), left.end(), i ) - left.begin();
            area = fb_area[ f ];
            for( auto &[ j, w ] : fb_faces[ f ] ) faces[ j ] += w;
        } else if( ! flag1[ i ] ) {
            area = area1[ i ];
            for( int k = 0; k < K1::D; ++k ) {
                int o = own1[ i * K1::D + k ];
                if( o >= 0 && hess1[ i * K1::D + k ] != 0 ) faces[ i + StencilTables<1>::off_x[ o ] * N + StencilTables<1>::off_y[ o ] ] += hess1[ i * K1::D + k ];
            }
        } else {
            area = area2[ i ];
            for( int k = 0; k < K2::D; ++k ) {
                int o = own2[ i * K2::D + k ];
                if( o >= 0 && hess2[ i * K2::D + k ] != 0 ) faces[ i + StencilTables<2>::off_x[ o ] * N + StencilTables<2>::off_y[ o ] ] += hess2[ i * K2::D + k ];
            }
        }
    }
};

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
            if( s.size() == s.find( "_it1" ) + 4 ) continue; // first iterates are trivially ring 1
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

    const T tol = 1e-9;
    ExactKernel<T> ek( 24 );
    std::printf( "N = %d, 1 thread, ns/cell ( best of %d ); %%: flagged after ring 1 / cells touched by stage 2 ( blocks: recomputed, cells: checked ) / sent to exact\n", N, reps );
    for( auto &[ name, psi ] : cases ) {
        Cascade cs( N, psi, ek.RB );

        // exact reference
        std::vector<T> ref_area( N * N );
        std::vector<std::vector<std::pair<int,T>>> ref_faces( N * N );
        double t_ref = ms( [ & ] {
            for( int i = 0; i < N * N; ++i ) { ref_faces[ i ].clear(); ek.cell( N, cs.pe.data(), cs.pmax, i / N, i % N, ref_area[ i ], ref_faces[ i ] ); }
        }, 1 );

        double t1 = ms( [ & ] { cs.stage1( tol ); }, reps );

        // previous chain: exact fallback straight after ring 1
        double t_direct = ms( [ & ] {
            T ar;
            std::vector<std::pair<int,T>> fc;
            for( int i = 0; i < N * N; ++i ) if( cs.flag1[ i ] ) { fc.clear(); ek.cell( N, cs.pe.data(), cs.pmax, i / N, i % N, ar, fc ); }
        }, reps );

        auto verify = [ & ]( int &wrong, double &err ) {
            wrong = 0; err = 0;
            std::map<int,T> got;
            for( int i = 0; i < N * N; ++i ) {
                T ar;
                cs.result( i, ar, got );
                std::map<int,T> exp;
                for( auto &[ j, w ] : ref_faces[ i ] ) exp[ j ] += w;
                double e = std::abs( double( ar - ref_area[ i ] ) );
                for( auto &[ j, w ] : exp ) e = std::max( e, std::abs( double( w - ( got.count( j ) ? got[ j ] : 0 ) ) ) );
                for( auto &[ j, w ] : got ) if( ! exp.count( j ) ) e = std::max( e, std::abs( double( w ) ) );
                err = std::max( err, e );
                wrong += e > 1e-8;
            }
        };
        const double nn = double( N ) * N, old = t1 + t_direct;

        for( int by_cells = 0; by_cells < 2; ++by_cells ) {
            double t2, t3;
            if( by_cells ) {
                t2 = ms( [ & ] { cs.stage2_cells( tol ); }, reps );
                t3 = ms( [ & ] { cs.stage3_cells( ek ); }, reps );
            } else {
                t2 = ms( [ & ] { cs.reset_flags2(); cs.stage2( tol ); }, reps );
                t3 = ms( [ & ] { cs.stage3( ek ); }, reps );
            }
            int wrong;
            double err;
            verify( wrong, err );
            const double tot = t1 + t2 + t3;
            std::printf( "%-15s %-6s %5.1f%% / %5.1f%% / %5.1f%%  ring1 %5.1f + ring2 %6.1f + exact %6.1f = %6.1f | ring1+exact %6.1f | exact everywhere %6.1f | x%5.2f vs ring1+exact, x%5.2f vs exact | wrong %d err %.1e\n",
                         name.c_str(), by_cells ? "cells" : "blocks", 100.0 * cs.n1 / nn, 100.0 * ( by_cells ? cs.nblk : cs.nblk * B ) / nn, 100.0 * cs.n2 / nn,
                         t1 * 1e6 / nn, t2 * 1e6 / nn, t3 * 1e6 / nn, tot * 1e6 / nn, old * 1e6 / nn, t_ref * 1e6 / nn, old / tot, t_ref / tot, wrong, err );
        }
    }
}
