#pragma once

// THE SOLVER OF THE DIRACS OF A 2D GRID ( `SumOfDiracsOnAGrid`, `SdotPlanNd._build_grid` ): damped Newton on the weights of
// N x N seeds at the centres of the square cells of a grid, N a power of two, the domain being the box of the grid.
// A port of `bench/experiments/grid_stencil/newton.cpp`. No positions are stored: everything is in INDEX UNITS ( seed
// `( a, b )` at `( a, b )`, the domain `[ -1/2, N - 1/2 ]^2`, `psi = w / h^2` ), physical units only at the edges.
//
//   geometry   ring-1 stencil + certificate ( fused, `StencilKernel.h` ), ring 2 on the flagged cells, cross-level
//              certificate, exact clipping on what is left ( `ExactKernel.h` )
//   integrals  the uniform density in SIMD ( the area, the lengths of the faces ); any other density cell by cell: the cell is
//              rebuilt from its neighbours as a sdot cell ( `ProviderList` ) and integrated by `measure_and_facets`
//   linear     H = dm / dpsi, matrix-free: per cell 8 weights toward the ring-1 neighbours in SoA planes, plus a short list of
//              extra ( i, j, w ) for the faces of the cells that came from ring 2 or from the exact path. PCG in row
//              passes, preconditioned by the weighted semi-coarsened tent blurs ( `TensorPrecond` )
//   start      coarse to fine: the solve on n0, 2 n0, ..., N, each level from the cubic interpolation of the previous psi
//   damping    sharp targets ( cells a hundredth of a pixel wide beside cells of ten pixels ): the steps limited by H + lambda L,
//              the line search on the residual, the cells a start or a step left empty repaired one by one ( `newton` )
//   threads    those of the CPU queue ( `Par` ): contiguous slices of rows or of cells, the sums combined in thread order

#include "StencilKernel.h"
#include "ExactKernel.h"
#include "GridDensities.h"
#include "../sdotplan/Sweep.h"
#include "../sdotplan/Bounds.h"
#include "../sdotplan/WarmRepair.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <numeric>
#include <vector>

namespace sdot {
namespace grid {

using T = double;
constexpr int B = 32;
using K1 = StencilKernel<1, B, T>;
using K2 = StencilKernel<2, B, T>;
using T1 = StencilTables<1>;
using T2 = StencilTables<2>;

inline double now_ms() { return std::chrono::duration<double, std::milli>( std::chrono::steady_clock::now().time_since_epoch() ).count(); }

// ================================================================================================
// threads
// ================================================================================================
/// CONTIGUOUS SLICES of `[ 0, n )`, one per thread of the CPU queue; the partial results combined in thread order, so that the
/// result depends on the number of threads only. Serial below `min_work` items of work ( a launch costs a few microseconds ).
struct Par {
    static constexpr SI min_work = SI( 1 ) << 15;
    const CpuQueue *queue = nullptr;
    int nt = 1;

    Par() = default;
    explicit Par( const CpuQueue &q ) : queue( &q ), nt( std::max( q.nb_workers(), 1 ) ) {}

    bool on( SI work ) const { return queue && nt > 1 && work >= min_work; }

    /// `f( b, e, t )` on the slices of `[ 0, n )`; `work`: what the whole loop touches
    template<class F>
    void slices( int n, SI work, F &&f ) const {
        if( ! on( work ) || n < 2 ) { if( n > 0 ) f( 0, n, 0 ); return; }
        queue->run_threads( nt, [&]( int t ) {
            const int b = int( SI( n ) * t / nt ), e = int( SI( n ) * ( t + 1 ) / nt );
            if( b < e ) f( b, e, t );
        } );
    }

    /// `f( b, e )` returns the part of a slice, `comb( acc, part )` adds it in
    template<class R, class F, class C>
    R reduce( int n, SI work, R init, F &&f, C &&comb ) const {
        if( ! on( work ) || n < 2 ) { if( n > 0 ) comb( init, f( 0, n ) ); return init; }
        std::vector<R> parts( nt, init );
        std::vector<char> used( nt, 0 );
        slices( n, work, [&]( int b, int e, int t ) { parts[ t ] = f( b, e ); used[ t ] = 1; } );
        for( int t = 0; t < nt; ++t ) if( used[ t ] ) comb( init, parts[ t ] );
        return init;
    }
    template<class R, class F>
    R sum( int n, SI work, F &&f ) const { return reduce( n, work, R( 0 ), FORWARD( f ), []( R &a, const R &b ) { a += b; } ); }
};

// ================================================================================================
// the seeds as sdot sees them, for the cells rebuilt from a list of neighbours ( any density )
// ================================================================================================
struct GridListPd {
    using TF = double;
    static constexpr int  ct_dim = 2;
    static constexpr bool on_cpu = true;
    static constexpr bool has_weights = true;

    int N;
    double ox, oy, h;                                    ///< the corner of the grid and the side of a cell ( physical )
    const T *psi;                                        ///< index units, N * N
    const int *ranks = nullptr;                          ///< the neighbours of the cell being built
    int nb = 0;

    SI   nb_seeds() const { return SI( N ) * N; }
    SI   user_id( SI k ) const { return k; }
    SI   seed_of( SI k ) const { return k; }
    auto point( SI k ) const {
        const T a = T( k / N ) + T( 0.5 ), b = T( k % N ) + T( 0.5 );
        return Vector<TF,2>::with_func( [&]( PI d ) { return d == 0 ? ox + h * a : oy + h * b; } );
    }
    TF   weight( SI k ) const { return h * h * psi[ k ]; }

    template<class TK>
    auto provider( SI k0 ) const { return sdotplan::ProviderList<GridListPd,TK>( *this, k0, ranks, nb ); }
};

// ================================================================================================
// geometry + integrals
// ================================================================================================
/// `Integ`: a density of `GridDensities.h` ( SIMD tiers + a scalar path ), or `DistCells`, the cells of sdot ( any density ). One copy of it per thread ( its
/// scratch ); the threads take contiguous slices of rows or of cells.
template<class Integ>
struct Evaluator {
    int N, P1;                                           // P1 = N + 2: padded planes of the hessian weights
    Par par;
    std::vector<Integ> integ;                            // one per thread
    ExactKernel<T> ek{ 24 };
    // buffers
    std::vector<T> p1, p2, pe;
    std::vector<unsigned char> flag1, flag2, mark;
    K1::Planes pl1;
    K2::Planes pl2;                                      // ( only the cells of `list1` are written, and read )
    std::vector<std::vector<T>> row_th;                  // what `row_block` writes beside the planes, one row per thread ( unused )
    std::vector<int> list1, check;
    std::vector<unsigned char> dbad;                     // a ring-1 cell that the SIMD rules of the density could not take
    std::vector<int> fine;                               // ( those of the first tier, for the second )
    std::vector<std::vector<int>> fine_th;
    // outputs
    std::vector<T> mass;                                 // N * N, index units
    NoInitVector<T> W;                                   // 8 planes of P1 * P1: weight toward the ring-1 neighbour in direction k
    struct Extra { int i, j; T w; };
    std::vector<Extra> extras;                           // faces toward the seeds out of ring 1
    std::vector<std::vector<Extra>> ex_th;               // ( the same, per thread, before they are gathered )
    // stats of the last call
    int n_flag1 = 0, n_exact = 0;
    T pmax = 0;                                          // the largest psi, for the exact path ( `pe`: psi padded for it )
    bool exhausted = false;                              // the exact path ran out of its list ( a cell wider than its 24 rings ), or
                                                         // a cell that sdot could not build ( `DistCells::failed` )
    double t_geom = 0, t_int = 0;

    Evaluator( int N, const Integ &integ, Par par ) : N( N ), P1( N + 2 ), par( par ), integ( par.nt, integ ),
        flag1( N * N ), flag2( N * N ), mark( N * N ), pl1( N, false, false ), pl2( N, false, false ), row_th( par.nt ),
        fine_th( par.nt ), mass( N * N ), W( 8 * ( N + 2 ) * ( N + 2 ) ), ex_th( par.nt ) {
        // ( the padding of the ring-1 planes is read: their initial values, in parallel; the interior of `W` is written by each `run` )
        par.slices( pl1.slots, SI( pl1.slots ) * pl1.P * K1::D * 17, [&]( int s0, int s1, int ) { pl1.fill( s0, s1 ); } );
    }

    void pad( int R, const std::vector<T> &psi, std::vector<T> &p ) const {
        const int P = N + 2 * R;
        const T inf = -std::numeric_limits<T>::infinity();
        p.resize( size_t( P ) * P );
        par.slices( P, SI( P ) * P, [&]( int r0, int r1, int ) {
            for( int r = r0; r < r1; ++r ) {
                T *row = &p[ size_t( r ) * P ];
                const int a = r - R;
                if( a < 0 || a >= N ) { std::fill( row, row + P, inf ); continue; }
                std::fill( row, row + R, inf );
                std::memcpy( row + R, &psi[ size_t( a ) * N ], N * sizeof( T ) );
                std::fill( row + R + N, row + P, inf );
            }
        } );
    }

    // the ring-1 direction of an offset ( -1: out of ring 1 )
    static int dir1( int dx, int dy ) {
        for( int k = 0; k < 8; ++k ) if( T1::off_x[ k ] == dx && T1::off_y[ k ] == dy ) return k;
        return -1;
    }

    // cross-level certificate ( cascade.cpp )
    bool cell_ok( int i, T tol ) const {
        static const auto dirs = [] {
            std::array<int, 50> r;
            r.fill( -1 );
            for( int k = 0; k < K1::D; ++k ) r[ int( T1::dir_x[ k ] + 2 ) * 5 + int( T1::dir_y[ k ] + 2 ) ] = k;
            for( int k = 0; k < K2::D; ++k ) r[ 25 + int( T2::dir_x[ k ] + 2 ) * 5 + int( T2::dir_y[ k ] + 2 ) ] = k;
            return r;
        }();
        const bool l2 = flag1[ i ];
        const int D = l2 ? K2::D : K1::D;
        const T *vx = l2 ? pl2.vx.data() : pl1.vx.data(), *vy = l2 ? pl2.vy.data() : pl1.vy.data();
        const signed char *ow = l2 ? pl2.ow.data() : pl1.ow.data();
        const int PP = l2 ? pl2.plane() : pl1.plane(), ii = l2 ? pl2.row( i / N ) + i % N + 2 : pl1.row( i / N ) + i % N + 1;
        for( int k = 0; k < D; ++k ) {
            const int km = ( k + D - 1 ) % D, o = ow[ k * PP + ii ];
            const T ex = vx[ k * PP + ii ] - vx[ km * PP + ii ], ey = vy[ k * PP + ii ] - vy[ km * PP + ii ];
            if( o < 0 || std::abs( ex ) + std::abs( ey ) <= tol ) continue;
            const int dx = l2 ? T2::off_x[ o ] : T1::off_x[ o ], dy = l2 ? T2::off_y[ o ] : T1::off_y[ o ];
            const int j = i + dx * N + dy;
            const bool m2 = flag1[ j ];
            const int ndx = -int( l2 ? T2::dir_x[ k ] : T1::dir_x[ k ] ), ndy = -int( l2 ? T2::dir_y[ k ] : T1::dir_y[ k ] );
            const int q = dirs[ ( m2 ? 25 : 0 ) + ( ndx + 2 ) * 5 + ndy + 2 ];
            if( q < 0 ) return false;
            const int Dj = m2 ? K2::D : K1::D, qm = ( q + Dj - 1 ) % Dj;
            const T *wx = m2 ? pl2.vx.data() : pl1.vx.data(), *wy = m2 ? pl2.vy.data() : pl1.vy.data();
            const signed char *wo = m2 ? pl2.ow.data() : pl1.ow.data();
            const int PQ = m2 ? pl2.plane() : pl1.plane(), jj = m2 ? pl2.row( j / N ) + j % N + 2 : pl1.row( j / N ) + j % N + 1;
            const int o2 = wo[ q * PQ + jj ];
            if( o2 < 0 ) return false;
            if( ( m2 ? T2::off_x[ o2 ] : T1::off_x[ o2 ] ) != -dx || ( m2 ? T2::off_y[ o2 ] : T1::off_y[ o2 ] ) != -dy ) return false;
            if( std::abs( vx[ km * PP + ii ] - wx[ q * PQ + jj ] - dx ) > tol || std::abs( vy[ km * PP + ii ] - wy[ q * PQ + jj ] - dy ) > tol ) return false;
            if( std::abs( vx[ k * PP + ii ] - wx[ qm * PQ + jj ] - dx ) > tol || std::abs( vy[ k * PP + ii ] - wy[ qm * PQ + jj ] - dy ) > tol ) return false;
        }
        return true;
    }

    // the weight of the face toward `j` ( index units ): in the planes for a ring-1 neighbour, in `ex` otherwise ( the row of
    // `i` only: the threads write disjoint rows )
    void put_face( int i, int j, T w, std::vector<Extra> &ex ) {
        const int k = dir1( j / N - i / N, j % N - i % N );
        if( k >= 0 ) W[ k * P1 * P1 + ( i / N + 1 ) * P1 + i % N + 1 ] = w;
        else ex.push_back( { i, j, w } );
    }
    void clear_faces( int i ) {
        for( int k = 0; k < 8; ++k ) W[ k * P1 * P1 + ( i / N + 1 ) * P1 + i % N + 1 ] = 0;
    }
    // the faces of the threads, in thread order
    void gather_extras() {
        extras.clear();
        for( auto &e : ex_th ) { extras.insert( extras.end(), e.begin(), e.end() ); e.clear(); }
    }

    // THE SAME WEIGHT SEEN FROM BOTH SIDES: a face on a discontinuity of the density ( the edge of a pixel ) is integrated by each
    // cell with its own side's value, and CG needs a symmetric H -- the mean of the two
    void symmetrize() {
        const int PP = P1 * P1;
        // ( for a direction k, each pair ( i, i + sh ) is visited once: the rows of the threads touch disjoint entries )
        par.slices( N, SI( N ) * N * 8, [&]( int a0, int a1, int ) {
            for( int k = 0; k < 4; ++k ) {               // ( direction k + 4 is the opposite of k: T1::off )
                const int ko = k + 4, sh = T1::off_x[ k ] * P1 + T1::off_y[ k ];
                for( int a = a0; a < a1; ++a )
                    for( int b = 0; b < N; ++b ) {
                        const int a2 = a + T1::off_x[ k ], b2 = b + T1::off_y[ k ];
                        if( a2 < 0 || a2 >= N || b2 < 0 || b2 >= N ) continue;
                        const int i = ( a + 1 ) * P1 + b + 1;
                        T &wi = W[ k * PP + i ], &wj = W[ ko * PP + i + sh ];
                        wi = wj = ( wi + wj ) / 2;
                    }
            }
        } );
        if( extras.empty() ) return;
        for( auto &e : extras ) if( e.i > e.j ) std::swap( e.i, e.j );
        std::sort( extras.begin(), extras.end(), []( const Extra &x, const Extra &y ) { return x.i != y.i ? x.i < y.i : x.j < y.j; } );
        std::vector<Extra> merged;
        for( size_t q = 0; q < extras.size(); ) {
            size_t r = q;
            T s = 0;
            for( ; r < extras.size() && extras[ r ].i == extras[ q ].i && extras[ r ].j == extras[ q ].j; ++r ) s += extras[ r ].w;
            const T w = s / 2;                           // ( a face seen from one side only counts half: the other saw nothing )
            merged.push_back( { extras[ q ].i, extras[ q ].j, w } );
            merged.push_back( { extras[ q ].j, extras[ q ].i, w } );
            q = r;
        }
        extras.swap( merged );
    }

    // THE POLYGON OF CELL i ( relative to the seed, index units ): vertex e, edge e from vertex e to e+1 toward `nbr[ e ]`
    // ( -1: the box ). From the ring-1 planes, the ring-2 planes or the exact path, as the certificates decided. `exh`: set when
    // the exact path ran out of its list.
    int polygon( int i, T *px, T *py, int *nbr, bool &exh ) const {
        if( flag1[ i ] && flag2[ i ] ) {
            int pl[ ExactKernel<T>::C ], n;
            T qx[ ExactKernel<T>::C ], qy[ ExactKernel<T>::C ];
            if( ! ek.clip( N, pe.data(), pmax, i / N, i % N, qx, qy, pl, n ) ) exh = true;
            for( int e = 0; e < n; ++e ) {
                px[ e ] = qx[ e ]; py[ e ] = qy[ e ];
                nbr[ e ] = pl[ e ] < 0 ? -1 : i + ek.ox[ pl[ e ] ] * N + ek.oy[ pl[ e ] ];
            }
            return n;
        }
        const bool l2 = flag1[ i ];
        const int D = l2 ? K2::D : K1::D, PQ = l2 ? pl2.plane() : pl1.plane(), ii = l2 ? pl2.row( i / N ) + i % N + 2 : pl1.row( i / N ) + i % N + 1;
        const T *vx = l2 ? pl2.vx.data() : pl1.vx.data(), *vy = l2 ? pl2.vy.data() : pl1.vy.data();
        const signed char *ow = l2 ? pl2.ow.data() : pl1.ow.data();
        // vertex e = plane vertex ( e - 1 ), so that edge e ( vertex e -> e+1 ) is direction e
        for( int e = 0; e < D; ++e ) {
            px[ e ] = vx[ ( ( e + D - 1 ) % D ) * PQ + ii ];
            py[ e ] = vy[ ( ( e + D - 1 ) % D ) * PQ + ii ];
            const int o = ow[ e * PQ + ii ];
            nbr[ e ] = o < 0 ? -1 : i + ( l2 ? T2::off_x[ o ] * N + T2::off_y[ o ] : T1::off_x[ o ] * N + T1::off_y[ o ] );
        }
        return D;
    }

    void geometry( const std::vector<T> &psi ) {
        const T tol = 1e-9;
        const SI nn = SI( N ) * N;
        // ring 1 on every cell, then its certificate ( a row needs the rows a - 1 .. a + 1: two launches )
        pad( 1, psi, p1 );
        par.slices( N, nn * 16, [&]( int a0, int a1, int t ) {
            auto &rb = row_th[ t ];
            rb.resize( size_t( N ) * ( 1 + 2 * K1::D ) );
            T *area = rb.data(), *hess = area + N;
            signed char *own = reinterpret_cast<signed char *>( hess + N * K1::D );
            for( int a = a0; a < a1; ++a )
                for( int b0 = 0; b0 < N; b0 += B )
                    K1::row_block( N, p1.data(), a, b0, B, area, hess, own, nullptr, &pl1 );
        } );
        n_flag1 = par.template sum<int>( N, nn * 8, [&]( int a0, int a1 ) {
            typename K1::CertTables ct;
            int c = 0;
            for( int a = a0; a < a1; ++a ) c += K1::certify_row( pl1, ct, a, flag1.data(), tol );
            return c;
        } );
        n_exact = 0;
        std::fill( flag2.begin(), flag2.end(), 0 );
        check.clear();
        if( n_flag1 ) {
            pad( 2, psi, p2 );
            list1.clear();
            for( int i = 0; i < N * N; ++i ) if( flag1[ i ] ) list1.push_back( i );
            const int nc = int( ( list1.size() + B - 1 ) / B );
            par.slices( nc, SI( list1.size() ) * 64, [&]( int c0, int c1, int ) {
                for( int c = c0; c < c1; ++c )
                    K2::cells_block( N, p2.data(), list1.data() + c * B, int( std::min<size_t>( B, list1.size() - c * B ) ), nullptr, nullptr, nullptr, &pl2 );
            } );
            for( int i : list1 )
                for( int da = -1; da <= 1; ++da )
                    for( int db = -1; db <= 1; ++db ) {
                        const int a = i / N + da, b = i % N + db;
                        if( a < 0 || a >= N || b < 0 || b >= N || mark[ a * N + b ] ) continue;
                        mark[ a * N + b ] = 1;
                        check.push_back( a * N + b );
                    }
            n_exact = par.template sum<int>( int( check.size() ), SI( check.size() ) * 32, [&]( int q0, int q1 ) {
                int c = 0;
                for( int q = q0; q < q1; ++q ) { const int i = check[ q ]; mark[ i ] = 0; flag2[ i ] = ! cell_ok( i, tol ); c += flag2[ i ] && flag1[ i ]; }
                return c;
            } );
            // ( a ring-1 cell that fails the cross-level check takes the exact path too )
            for( int i : check ) if( flag2[ i ] && ! flag1[ i ] ) { flag1[ i ] = 1; ++n_exact; }
        }
        if( n_exact ) {
            pad( ek.RB, psi, pe );
            pmax = *std::max_element( psi.begin(), psi.end() );
        }
    }

    void run( const std::vector<T> &psi ) {
        double t0 = now_ms();
        exhausted = false;
        geometry( psi );
        double t1 = now_ms();
        const SI nn = SI( N ) * N;
        std::vector<char> exh( par.nt, 0 );
        if constexpr ( Integ::simd ) {
            // ---- the ring-1 cells in SIMD ( two tiers ), the others and those the SIMD rules could not take by the scalar path
            const int PP = pl1.plane();
            dbad.resize( nn );
            auto store = [&]( int i, const T *face, int x, T ms ) {
                mass[ i ] = ms;
                const int ii = pl1.row( i / N ) + i % N + 1;
                for( int k = 0; k < 8; ++k )
                    W[ k * P1 * P1 + ( i / N + 1 ) * P1 + i % N + 1 ] = pl1.ow[ k * PP + ii ] >= 0 ? face[ k * B + x ] * T( 0.5 ) * T( T1::dir_inv_norm[ k ] ) : T( 0 );
            };
            if constexpr ( Integ::has_simd ) {
                par.slices( N, nn * 64, [&]( int a0, int a1, int t ) {
                    auto &fl = fine_th[ t ];
                    fl.clear();
                    for( int a = a0; a < a1; ++a ) {
                        for( int b0 = 0; b0 < N; b0 += B ) {
                            alignas( 64 ) T vx[ 8 ][ B ], vy[ 8 ][ B ], face[ 8 ][ B ], ms[ B ], sa[ B ], sb[ B ];
                            alignas( 64 ) unsigned char bad[ B ] = {};
                            const int i0 = pl1.row( a ) + b0 + 1;
                            for( int k = 0; k < 8; ++k )
                                for( int x = 0; x < B; ++x ) { vx[ k ][ x ] = pl1.vx[ k * PP + i0 + x ]; vy[ k ][ x ] = pl1.vy[ k * PP + i0 + x ]; }
                            for( int x = 0; x < B; ++x ) { sa[ x ] = a; sb[ x ] = b0 + x; }
                            integ[ t ].template cells<8,B>( vx, vy, sa, sb, ms, face, bad );
                            for( int x = 0; x < B; ++x ) mass[ a * N + b0 + x ] = ms[ x ];
                            for( int k = 0; k < 8; ++k ) {
                                const T inv = T( 0.5 ) * T( T1::dir_inv_norm[ k ] );
                                const signed char *ow = pl1.ow.data() + k * PP + i0;
                                T *w = W.data() + k * P1 * P1 + ( a + 1 ) * P1 + b0 + 1;
                                for( int x = 0; x < B; ++x ) w[ x ] = ow[ x ] >= 0 ? face[ k ][ x ] * inv : T( 0 );
                            }
                            for( int x = 0; x < B; ++x ) {
                                const int i = a * N + b0 + x;
                                dbad[ i ] = 0;
                                if( bad[ x ] && ! flag1[ i ] ) fl.push_back( i );
                            }
                        }
                    }
                } );
                // the second tier, on the cells the first one could not take
                fine.clear();
                for( auto &fl : fine_th ) fine.insert( fine.end(), fl.begin(), fl.end() );
                const int nc = int( ( fine.size() + B - 1 ) / B );
                par.slices( nc, SI( fine.size() ) * 512, [&]( int c0, int c1, int t ) {
                    for( int c = c0; c < c1; ++c ) {
                        const int n = int( std::min<size_t>( B, fine.size() - size_t( c ) * B ) );
                        alignas( 64 ) T vx[ 8 ][ B ], vy[ 8 ][ B ], face[ 8 ][ B ], ms[ B ], sa[ B ], sb[ B ];
                        alignas( 64 ) unsigned char bad[ B ] = {};
                        for( int x = 0; x < B; ++x ) {
                            const int i = fine[ size_t( c ) * B + ( x < n ? x : n - 1 ) ], ii = pl1.row( i / N ) + i % N + 1;
                            sa[ x ] = i / N; sb[ x ] = i % N;
                            for( int k = 0; k < 8; ++k ) { vx[ k ][ x ] = pl1.vx[ k * PP + ii ]; vy[ k ][ x ] = pl1.vy[ k * PP + ii ]; }
                        }
                        integ[ t ].template cells_fine<8,B>( vx, vy, sa, sb, ms, face, bad );
                        for( int x = 0; x < n; ++x ) {
                            const int i = fine[ size_t( c ) * B + x ];
                            dbad[ i ] = bad[ x ];
                            if( ! bad[ x ] ) store( i, &face[ 0 ][ 0 ], x, ms[ x ] );
                        }
                    }
                } );
            } else
                std::fill( dbad.begin(), dbad.end(), 1 );
            // the scalar path
            par.slices( N * N, Integ::has_simd ? nn : nn * 256, [&]( int i0, int i1, int t ) {
                bool ex = false;
                for( int i = i0; i < i1; ++i ) {
                    if( ! flag1[ i ] && ! dbad[ i ] ) continue;
                    T px[ ExactKernel<T>::C ], py[ ExactKernel<T>::C ], face[ ExactKernel<T>::C ];
                    int nbr[ ExactKernel<T>::C ];
                    const int n = polygon( i, px, py, nbr, ex );
                    integ[ t ].polygon( n, px, py, T( i / N ), T( i % N ), mass[ i ], face );
                    clear_faces( i );
                    for( int e = 0; e < n; ++e ) {
                        const int j = nbr[ e ];
                        if( j < 0 || face[ e ] == 0 ) continue;
                        put_face( i, j, face[ e ] / ( 2 * std::hypot( T( j / N - i / N ), T( j % N - i % N ) ) ), ex_th[ t ] );
                    }
                }
                exh[ t ] = ex;
            } );
            gather_extras();
            if constexpr ( ! Integ::exact )
                symmetrize();
        } else {
            // ---- every cell through sdot's cells, rebuilt from its neighbours
            for( auto &ig : integ ) ig.bind( psi );
            par.slices( N * N, nn * 1024, [&]( int i0, int i1, int t ) {
                T px[ ExactKernel<T>::C ], py[ ExactKernel<T>::C ];
                int nbr[ ExactKernel<T>::C ], ranks[ ExactKernel<T>::C ];
                bool ex = false;
                for( int i = i0; i < i1; ++i ) {
                    const int n = polygon( i, px, py, nbr, ex );
                    int nb = 0;
                    for( int e = 0; e < n; ++e ) if( nbr[ e ] >= 0 ) ranks[ nb++ ] = nbr[ e ];
                    clear_faces( i );
                    // ( an empty cell: no neighbours would mean the whole domain to the rebuilt cell )
                    if( n == 0 ) { mass[ i ] = 0; continue; }
                    integ[ t ].cell( i, ranks, nb, mass[ i ], [&]( int j, T w ) { put_face( i, j, w, ex_th[ t ] ); } );
                }
                exh[ t ] = ex;
            } );
            gather_extras();
            symmetrize();
            for( auto &ig : integ ) {                    // ( a cell sdot could not build: like a trial out of the exact path's reach )
                exhausted = exhausted || ig.failed;
                ig.failed = false;
            }
        }
        for( char e : exh ) exhausted = exhausted || e;
        double t2 = now_ms();
        t_geom += t1 - t0;
        t_int += t2 - t1;
    }

    // barycenters ( index units, relative to the seed ) and the second moments about the seed, at the last `run`
    void moments( const std::vector<T> &psi, std::vector<T> &bx, std::vector<T> &by, std::vector<T> &m2 ) {
        bx.assign( N * N, 0 ); by.assign( N * N, 0 ); m2.assign( N * N, 0 );
        if constexpr ( ! Integ::simd ) for( auto &ig : integ ) ig.bind( psi );
        par.slices( N * N, SI( N ) * N * 1024, [&]( int i0, int i1, int t ) {
            T px[ ExactKernel<T>::C ], py[ ExactKernel<T>::C ];
            int nbr[ ExactKernel<T>::C ], ranks[ ExactKernel<T>::C ];
            bool ex = false;
            for( int i = i0; i < i1; ++i ) {
                const int n = polygon( i, px, py, nbr, ex );
                if( n == 0 ) continue;
                if constexpr ( Integ::simd ) {
                    T m, mx, my, s2;
                    integ[ t ].moments( n, px, py, T( i / N ), T( i % N ), m, mx, my, s2 );
                    bx[ i ] = m > 0 ? mx / m : 0; by[ i ] = m > 0 ? my / m : 0; m2[ i ] = s2;
                } else {
                    int nb = 0;
                    for( int e = 0; e < n; ++e ) if( nbr[ e ] >= 0 ) ranks[ nb++ ] = nbr[ e ];
                    integ[ t ].moments( i, ranks, nb, bx[ i ], by[ i ], m2[ i ] );
                }
            }
        } );
    }
};

/// ANY DENSITY: the cell of `i` rebuilt by sdot from its neighbours ( the certified stencil gave them ), integrated by
/// `measure_and_facets` / `integrate_moments_into`. Index units out: masses times `S = N^2 / mass of the domain`.
template<class Dom,class Dist,class TK>
struct DistCells {
    static constexpr bool simd = false;
    using Local = typename Dom::template Local<TK>;
    static constexpr int nbc = diagram::nb_work_cells<Dist>();

    const Dom &dom;
    const Dist &dist;
    GridListPd pd;
    double S;                                            ///< index units per unit of mass ( `N^2`: the masses sum to 1 )
    SI cap = 32, words = 0;
    std::vector<std::int32_t> scratch;
    bool failed = false;                                 ///< a cell that sdot could not build or integrate

    DistCells( const Dom &dom, const Dist &dist, int N, double ox, double oy, double h ) : dom( dom ), dist( dist ), pd{ N, ox, oy, h, nullptr }, S( double( N ) * N ) {
        resize();
    }
    void resize() {
        words = diagram::words_for<Local,double>( cap, nbc, false );
        scratch.assign( size_t( words ) + 16, 0 );
    }
    void bind( const std::vector<T> &psi ) { pd.psi = psi.data(); }

    template<class F>
    void cell( int i, const int *ranks, int nb, T &mass, F &&face ) {
        pd.ranks = ranks;
        pd.nb = nb;
        for ( ;; ) {
            Carver cv{ scratch.data(), words };
            Local c, piece;
            c.attach( cv, cap );
            if constexpr ( nbc > 1 ) piece.attach( cv, cap );
            else                     piece = c;
            double m = 0;
            bool ok = sdotplan::measure_and_facets( pd, SI( i ), c, piece, dom, dist, m,
                                                    [&]( SI j, double cij ) { face( int( j ), T( S * pd.h * pd.h * cij ) ); }, true );
            if ( ok ) { mass = T( S * m ); return; }
            if ( cap > 4096 ) {                          // ( not a question of room )
                failed = true;
                mass = 0;
                return;
            }
            cap *= 2;
            resize();
        }
    }

    void moments( int i, const int *ranks, int nb, T &bx, T &by, T &m2 ) {
        pd.ranks = ranks;
        pd.nb = nb;
        for ( ;; ) {
            Carver cv{ scratch.data(), words };
            Local c, piece;
            c.attach( cv, cap );
            if constexpr ( nbc > 1 ) piece.attach( cv, cap );
            else                     piece = c;
            double m = 0, s2 = 0, mx[ 2 ] = { 0, 0 };
            auto first = [&]( int d ) -> double & { return mx[ d ]; };
            if ( diagram::make_cell( pd, c, SI( i ), dom ) && diagram::integrate_moments_into<double>( m, first, s2, c, piece, dist ) ) {
                // relative to the seed, in index units; the second moment about the seed, in index units ( `S` mass, squared index lengths )
                const auto p = pd.point( i );
                const double h = pd.h;
                bx = m > 0 ? T( ( mx[ 0 ] / m - p[ 0 ] ) / h ) : T( 0 );
                by = m > 0 ? T( ( mx[ 1 ] / m - p[ 1 ] ) / h ) : T( 0 );
                m2 = T( S * ( s2 - 2 * ( p[ 0 ] * mx[ 0 ] + p[ 1 ] * mx[ 1 ] ) + m * ( p[ 0 ] * p[ 0 ] + p[ 1 ] * p[ 1 ] ) ) / ( h * h ) );
                return;
            }
            cap *= 2;
            resize();
        }
    }
};

// ================================================================================================
// matrix-free operator ( vectors padded to ( N + 2 )^2, zero border ) and the tensor blur preconditioner
// ================================================================================================
struct Operator {
    int N, P;
    const NoInitVector<T> *W;
    std::vector<std::pair<int,int>> ex_ij;
    std::vector<T> ex_w;
    int off[ 8 ];

    template<class Ev>
    void bind( const Ev &ev ) {
        N = ev.N; P = N + 2; W = &ev.W;
        ex_ij.clear(); ex_w.clear();
        for( auto &e : ev.extras ) {
            ex_ij.push_back( { ( e.i / N + 1 ) * P + e.i % N + 1, ( e.j / N + 1 ) * P + e.j % N + 1 } );
            ex_w.push_back( e.w );
        }
        for( int k = 0; k < 8; ++k ) off[ k ] = T1::off_x[ k ] * P + T1::off_y[ k ];
    }
};

/// the 1D tents along a row
struct Tents {
    // t ( m ) = R fr ( 2 m )
    static void restrict_row( const T *fr, int m, T *t ) {
        t[ 0 ] = fr[ 0 ] + T( 0.75 ) * fr[ 1 ] + ( m > 1 ? T( 0.25 ) * fr[ 2 ] : T( 0 ) );
        #pragma omp simd
        for( int B = 1; B < m - 1; ++B ) t[ B ] = T( 0.75 ) * ( fr[ 2 * B ] + fr[ 2 * B + 1 ] ) + T( 0.25 ) * ( fr[ 2 * B - 1 ] + fr[ 2 * B + 2 ] );
        if( m > 1 ) t[ m - 1 ] = T( 0.75 ) * fr[ 2 * m - 2 ] + fr[ 2 * m - 1 ] + T( 0.25 ) * fr[ 2 * m - 3 ];
    }
    // fr ( 2 m ) += P t ( m )
    static void prolong_row_add( const T *t, int m, T *fr ) {
        const int n = 2 * m;
        fr[ 0 ] += t[ 0 ];
        fr[ n - 1 ] += t[ m - 1 ];
        #pragma omp simd
        for( int B = 1; B < m; ++B ) fr[ 2 * B ] += T( 0.75 ) * t[ B ] + T( 0.25 ) * t[ B - 1 ];
        #pragma omp simd
        for( int B = 0; B < m - 1; ++B ) fr[ 2 * B + 1 ] += T( 0.75 ) * t[ B ] + T( 0.25 ) * t[ B + 1 ];
    }
};

/// THE WEIGHTED SEMI-COARSENED TENT BLURS: z = r / diag + sum over the subspaces ( la, lb ) of P D R r, D from the
/// conductances of the faces, each subspace weighted by the balance of its two tent energies ( newton.cpp, `tensor` )
struct TensorPrecond {
    int N = 0, P = 0, L = 0, band = 1 << 20;
    bool axes = true;
    T c = T( 2 ) / 3, pw = T( 0.5 );
    Par par;
    std::vector<T> d0, id0;
    std::vector<std::vector<T>> g, Di;                   // ( L + 1 )^2 grids, index la * ( L + 1 ) + lb
    std::vector<T> Da, Db;

    static T balance( T ea, T eb, T pw ) {
        if( pw == 0 ) return 1;
        const T lo = std::min( ea, eb ), hi = std::max( ea, eb );
        if( hi == 0 ) return 0;
        return pw == 1 ? lo / hi : pw == T( 0.5 ) ? std::sqrt( lo / hi ) : std::pow( lo / hi, pw );
    }
    bool kept( int la, int lb ) const { return ( la || lb ) && ( std::abs( la - lb ) <= band || ( axes && ( la == 0 || lb == 0 ) ) ); }
    int gi( int la, int lb ) const { return la * ( L + 1 ) + lb; }

    void alloc( int n ) {
        N = n;
        L = 0;
        while( ( N >> ( L + 1 ) ) >= 2 ) ++L;
        g.assign( ( L + 1 ) * ( L + 1 ), {} ); Di.assign( ( L + 1 ) * ( L + 1 ), {} );
        for( int la = 0; la <= L; ++la )
            for( int lb = 0; lb <= L; ++lb ) {
                g[ gi( la, lb ) ].resize( ( N >> la ) * ( N >> lb ) );
                if( kept( la, lb ) ) Di[ gi( la, lb ) ].resize( ( N >> la ) * ( N >> lb ) );
            }
    }
    void restrict_tree( int from_la = 1 ) {
        for( int la = from_la; la <= L; ++la ) restrict_a( g[ gi( la - 1, 0 ) ].data(), N >> ( la - 1 ), N, g[ gi( la, 0 ) ].data() );
        for( int la = 0; la <= L; ++la )
            for( int lb = la == 0 && from_la == 2 ? 2 : 1; lb <= L; ++lb ) restrict_b( g[ gi( la, lb - 1 ) ].data(), N >> la, N >> ( lb - 1 ), g[ gi( la, lb ) ].data() );
    }

    // scaling from conductances: ka( i ) = sum_k w_k | oa_k |, kb( i ) = sum_k w_k | ob_k |, Ea = c R ka / ha^2, Eb = c R kb / hb^2
    void build( const Operator &op ) {
        if( N != op.N || g.empty() ) alloc( op.N );
        P = op.P;
        const int PP = P * P;
        d0.assign( P * P, 0 );
        Da.assign( N * N, 0 ); Db.assign( N * N, 0 );
        par.slices( N, SI( N ) * N * 8, [&]( int a0, int a1, int ) {
            for( int a = a0; a < a1; ++a ) {
                T *dr = &d0[ ( a + 1 ) * P + 1 ], *ra = &Da[ a * N ], *rb = &Db[ a * N ];
                for( int k = 0; k < 8; ++k ) {
                    const T *wr = &( *op.W )[ k * PP + ( a + 1 ) * P + 1 ];
                    const bool fa = T1::off_x[ k ], fb = T1::off_y[ k ];
                    #pragma omp simd
                    for( int b = 0; b < N; ++b ) { dr[ b ] += wr[ b ]; ra[ b ] += fa ? wr[ b ] : T( 0 ); rb[ b ] += fb ? wr[ b ] : T( 0 ); }
                }
            }
        } );
        for( size_t e = 0; e < op.ex_w.size(); ++e ) {
            const int i = op.ex_ij[ e ].first, j = op.ex_ij[ e ].second, f = ( i / P - 1 ) * N + i % P - 1;
            d0[ i ] += op.ex_w[ e ];
            Da[ f ] += std::abs( j / P - i / P ) * op.ex_w[ e ];
            Db[ f ] += std::abs( j % P - i % P ) * op.ex_w[ e ];
        }
        id0.resize( d0.size() );
        par.slices( P, SI( P ) * P, [&]( int r0, int r1, int ) {
            #pragma omp simd
            for( size_t i = size_t( r0 ) * P; i < size_t( r1 ) * P; ++i ) id0[ i ] = d0[ i ] > 0 ? 1 / d0[ i ] : 0;
        } );
        for( int pass = 0; pass < 2; ++pass ) {
            std::copy( ( pass ? Db : Da ).begin(), ( pass ? Db : Da ).end(), g[ 0 ].begin() );
            restrict_tree();
            for( int la = 0; la <= L; ++la )
                for( int lb = 0; lb <= L; ++lb ) {
                    if( ! kept( la, lb ) ) continue;
                    T *D = Di[ gi( la, lb ) ].data();
                    const T *v = g[ gi( la, lb ) ].data();
                    const int n = ( N >> la ) * ( N >> lb );
                    const T sa = c / T( 1 << la ) / T( 1 << la ), sb = c / T( 1 << lb ) / T( 1 << lb );
                    par.slices( n, n, [&]( int q0, int q1, int ) {
                        if( ! pass ) {
                            #pragma omp simd
                            for( int q = q0; q < q1; ++q ) D[ q ] = sa * v[ q ];
                        } else {
                            for( int q = q0; q < q1; ++q ) {
                                const T ea = D[ q ], eb = sb * v[ q ], d = ea + eb;
                                D[ q ] = d > 0 ? balance( ea, eb, pw ) / d : 0;
                            }
                        }
                    } );
                }
        }
    }

    // ( the rows of the coarse grid in slices: each writes its own rows )
    void restrict_a( const T *f, int na, int nb, T *c ) const {
        const int m = na / 2;
        par.slices( m, SI( na ) * nb, [&]( int A0, int A1, int ) {
            for( int A = A0; A < A1; ++A ) {
                const T w0 = A == 0 ? T( 1 ) : T( 0.75 ), w1 = A == m - 1 ? T( 1 ) : T( 0.75 ), wm = A >= 1 ? T( 0.25 ) : T( 0 ), wp = A <= m - 2 ? T( 0.25 ) : T( 0 );
                const T *r0 = f + 2 * A * nb, *r1 = r0 + nb, *rm = A >= 1 ? r0 - nb : r0, *rp = A <= m - 2 ? r1 + nb : r1;
                T *cr = c + A * nb;
                #pragma omp simd
                for( int b = 0; b < nb; ++b ) cr[ b ] = w0 * r0[ b ] + w1 * r1[ b ] + wm * rm[ b ] + wp * rp[ b ];
            }
        } );
    }
    void restrict_b( const T *f, int na, int nb, T *c ) const {
        par.slices( na, SI( na ) * nb, [&]( int a0, int a1, int ) {
            for( int a = a0; a < a1; ++a ) Tents::restrict_row( f + a * nb, nb / 2, c + a * ( nb / 2 ) );
        } );
    }
    void prolong_a_add( const T *c, int ma, int nb, T *f ) const {
        par.slices( ma, SI( ma ) * nb * 2, [&]( int A0, int A1, int ) {
            for( int A = A0; A < A1; ++A ) {
                const T *c0 = c + A * nb, *cm = A >= 1 ? c0 - nb : c0, *cp = A <= ma - 2 ? c0 + nb : c0;
                const T e0 = A == 0 ? T( 1 ) : T( 0.75 ), em = A == 0 ? T( 0 ) : T( 0.25 ), e1 = A == ma - 1 ? T( 1 ) : T( 0.75 ), ep = A == ma - 1 ? T( 0 ) : T( 0.25 );
                T *t0 = f + 2 * A * nb, *t1 = t0 + nb;
                #pragma omp simd
                for( int b = 0; b < nb; ++b ) { t0[ b ] += e0 * c0[ b ] + em * cm[ b ]; t1[ b ] += e1 * c0[ b ] + ep * cp[ b ]; }
            }
        } );
    }
    void prolong_b_add( const T *c, int na, int mb, T *f ) const {
        par.slices( na, SI( na ) * mb * 2, [&]( int a0, int a1, int ) {
            for( int a = a0; a < a1; ++a ) Tents::prolong_row_add( c + a * mb, mb, f + a * 2 * mb );
        } );
    }

    // ---- hooks of pcg ----
    // row a of the new residual: copy, its b restriction ( 0, 1 ) -- rows independent
    void row_b( int a, const T *rv ) {
        std::memcpy( g[ 0 ].data() + a * N, rv, N * sizeof( T ) );
        Tents::restrict_row( rv, N / 2, g[ gi( 0, 1 ) ].data() + a * ( N / 2 ) );
    }
    // the rest of the tree from the ( 0, 0 ) and ( 0, 1 ) grids, scaling, prolongation down to ( 1, 0 ) and ( 0, 1 )
    void coarse() {
        restrict_a( g[ 0 ].data(), N, N, g[ gi( 1, 0 ) ].data() );
        restrict_tree( 2 );
        for( int q = 1; q < ( L + 1 ) * ( L + 1 ); ++q ) {
            T *v = g[ q ].data();
            const int n = int( g[ q ].size() );
            if( Di[ q ].empty() ) { std::fill( v, v + n, T( 0 ) ); continue; }
            const T *d = Di[ q ].data();
            par.slices( n, n, [&]( int c0, int c1, int ) {
                #pragma omp simd
                for( int c2 = c0; c2 < c1; ++c2 ) v[ c2 ] *= d[ c2 ];
            } );
        }
        for( int la = 0; la <= L; ++la )
            for( int lb = L; lb >= ( la ? 1 : 2 ); --lb ) prolong_b_add( g[ gi( la, lb ) ].data(), N >> la, N >> lb, g[ gi( la, lb - 1 ) ].data() );
        for( int la = L; la >= 2; --la ) prolong_a_add( g[ gi( la, 0 ) ].data(), N >> la, N, g[ gi( la - 1, 0 ) ].data() );
    }
    // row a of z ( already r / diag ): add the ( 0, 1 ) and ( 1, 0 ) parts
    void row_c( int a, T *zr ) const {
        Tents::prolong_row_add( g[ gi( 0, 1 ) ].data() + a * ( N / 2 ), N / 2, zr );
        const int m = N / 2, A = a / 2, W = N;
        const T *c0 = g[ gi( 1, 0 ) ].data() + A * W;
        T e, en;
        const T *cn;
        if( a % 2 == 0 ) { e = A == 0 ? T( 1 ) : T( 0.75 ); en = A == 0 ? T( 0 ) : T( 0.25 ); cn = A >= 1 ? c0 - W : c0; }
        else { e = A == m - 1 ? T( 1 ) : T( 0.75 ); en = A == m - 1 ? T( 0 ) : T( 0.25 ); cn = A <= m - 2 ? c0 + W : c0; }
        #pragma omp simd
        for( int b = 0; b < N; ++b ) zr[ b ] += e * c0[ b ] + en * cn[ b ];
    }
};

/// PCG in row passes per iteration, each on slices of rows ( p update; matvec; x / r update + Jacobi + row half of the
/// restriction; the coarse tree; row half of the prolongation + sums, the mean projection folded in ). H is singular along
/// the constants: the iterates are kept at zero mean. Returns the number of iterations.
inline int pcg_fused( const Operator &op, TensorPrecond &M, const std::vector<T> &b, std::vector<T> &x, T rtol, int maxit, const Par &par ) {
    const int N = op.N, P = op.P, PP = P * P;
    const SI nn = SI( N ) * N;
    std::vector<T> r = b, z( PP, 0 ), p( PP, 0 ), q( PP, 0 );
    std::fill( x.begin(), x.end(), 0 );
    const T *W = op.W->data(), *id0 = M.id0.data();
    int off[ 8 ];
    for( int k = 0; k < 8; ++k ) off[ k ] = op.off[ k ];
    using T3 = std::array<T,3>;
    auto add3 = []( T3 &a, const T3 &c ) { a[ 0 ] += c[ 0 ]; a[ 1 ] += c[ 1 ]; a[ 2 ] += c[ 2 ]; };

    auto pass_c = [ & ]( T &s, T &rz ) {
        const T3 sums = par.reduce( N, nn * 4, T3{ 0, 0, 0 }, [&]( int a0, int a1 ) {
            T sz = 0, rzr = 0, sr = 0;
            for( int a = a0; a < a1; ++a ) {
                const int row = ( a + 1 ) * P + 1;
                T *zr = z.data() + row;
                const T *rr = r.data() + row;
                M.row_c( a, zr );
                #pragma omp simd reduction( +: sz, rzr, sr )
                for( int x = 0; x < N; ++x ) { sz += zr[ x ]; rzr += rr[ x ] * zr[ x ]; sr += rr[ x ]; }
            }
            return T3{ sz, rzr, sr };
        }, add3 );
        s = sums[ 0 ] / ( T( N ) * N );
        rz = sums[ 1 ] - s * sums[ 2 ];
    };

    T nb = par.template sum<T>( P, SI( PP ), [&]( int r0, int r1 ) {
        T s = 0;
        for( size_t i = size_t( r0 ) * P; i < size_t( r1 ) * P; ++i ) s += b[ i ] * b[ i ];
        return s;
    } );
    nb = std::sqrt( nb );
    if( nb == 0 ) return 0;
    par.slices( N, nn * 3, [&]( int a0, int a1, int ) {
        for( int a = a0; a < a1; ++a ) {
            const int row = ( a + 1 ) * P + 1;
            #pragma omp simd
            for( int x = 0; x < N; ++x ) z[ row + x ] = r[ row + x ] * id0[ row + x ];
            M.row_b( a, r.data() + row );
        }
    } );
    M.coarse();
    T s, rz;
    pass_c( s, rz );
    T beta = 0;
    int it = 0;
    for( ; it < maxit; ++it ) {
        // pass A: the new direction, then the matvec ( a row reads the rows a - 1 .. a + 1 of p: two launches )
        par.slices( N, nn * 3, [&]( int a0, int a1, int ) {
            for( int a = a0; a < a1; ++a ) {
                const int row = ( a + 1 ) * P + 1;
                #pragma omp simd
                for( int x = 0; x < N; ++x ) p[ row + x ] = z[ row + x ] - s + beta * p[ row + x ];
            }
        } );
        T pq = par.template sum<T>( N, nn * 18, [&]( int a0, int a1 ) {
            T pq = 0;
            const T *pv = p.data();
            T *qv = q.data();
            for( int a = a0; a < a1; ++a ) {
                const int r0 = ( a + 1 ) * P + 1;
                #pragma omp simd reduction( +: pq )
                for( int x = 0; x < N; ++x ) {
                    const int i = r0 + x;
                    const T vi = pv[ i ];
                    T acc = 0;
                    for( int k = 0; k < 8; ++k ) acc += W[ k * PP + i ] * ( vi - pv[ i + off[ k ] ] );
                    qv[ i ] = acc;
                    pq += vi * acc;
                }
            }
            return pq;
        } );
        for( size_t e = 0; e < op.ex_w.size(); ++e ) {
            const int i = op.ex_ij[ e ].first, j = op.ex_ij[ e ].second;
            const T d = op.ex_w[ e ] * ( p[ i ] - p[ j ] );
            q[ i ] += d;
            pq += p[ i ] * d;
        }
        if( ! ( pq > 0 ) ) break;                        // ( the rounding of a system solved as far as it can be )
        // pass B
        const T alpha = rz / pq;
        const T rr = par.template sum<T>( N, nn * 7, [&]( int a0, int a1 ) {
            T rr = 0;
            for( int a = a0; a < a1; ++a ) {
                const int row = ( a + 1 ) * P + 1;
                T *xr = x.data() + row, *rv = r.data() + row, *zr = z.data() + row;
                const T *pr = p.data() + row, *qr = q.data() + row, *dr = id0 + row;
                #pragma omp simd reduction( +: rr )
                for( int i = 0; i < N; ++i ) {
                    xr[ i ] += alpha * pr[ i ];
                    const T ri = rv[ i ] - alpha * qr[ i ];
                    rv[ i ] = ri;
                    rr += ri * ri;
                    zr[ i ] = ri * dr[ i ];
                }
                M.row_b( a, rv );
            }
            return rr;
        } );
        if( std::sqrt( rr ) <= rtol * nb ) { ++it; break; }
        // coarse levels + pass C
        M.coarse();
        T rz2;
        pass_c( s, rz2 );
        beta = rz2 / rz;
        rz = rz2;
    }
    return it;
}

// ================================================================================================
// Newton, coarse to fine
// ================================================================================================
struct Options {
    double tol_abs = 0, tol_rel = 1e-9;                  ///< stop when max | m - t | <= tol_abs ( mass units ) or max | m - t | / t <= tol_rel
    double ctol = 1e-2;                                  ///< the relative tolerance of the coarse levels
    int    n0 = 64;                                      ///< the coarsest level ( `N` itself: no coarse to fine )
    int    max_iter = 50;
    double cg_rtol = 1e-2;
    double step_limit = 2;                               ///< the largest move of a bisector of the grid in a step ( index units ), see `newton`
};

struct Stats {
    int status = 0, nb_iter = 0, nb_diag = 0, nb_backtracks = 0, lin_nb_iter = 0, nb_levels = 0;
    double domain_mass = 1;
    double residual = 0, residual0 = -1, t_diag = 0, t_lin = 0, t_total = 0;
    bool exhausted = false;
};

/// psi on the n x n grid -> psi on the 2n x 2n grid, by separable cubic Lagrange interpolation ( exact on quadratics: the
/// |x|^2 part of psi gives no parity pattern ). Index units: x_f = 2 x_c + 1/2, and psi_f = 4 psi_c
inline std::vector<T> refine_psi( int n, const std::vector<T> &pc, const Par &par ) {
    const int N = 2 * n;
    std::vector<int> j0( N );
    std::vector<std::array<T,4>> w( N );
    for( int a = 0; a < N; ++a ) {
        const T u = ( a - T( 0.5 ) ) / 2;
        j0[ a ] = std::clamp( int( std::floor( u ) ) - 1, 0, n - 4 );
        for( int k = 0; k < 4; ++k ) {
            T l = 1;
            for( int m = 0; m < 4; ++m ) if( m != k ) l *= ( u - j0[ a ] - m ) / T( k - m );
            w[ a ][ k ] = l;
        }
    }
    std::vector<T> t( N * n ), pf( N * N );
    par.slices( N, SI( N ) * n * 4, [&]( int a0, int a1, int ) {
        for( int a = a0; a < a1; ++a ) for( int b = 0; b < n; ++b ) {
            T v = 0;
            for( int k = 0; k < 4; ++k ) v += w[ a ][ k ] * pc[ ( j0[ a ] + k ) * n + b ];
            t[ a * n + b ] = v;
        }
    } );
    par.slices( N, SI( N ) * N * 4, [&]( int a0, int a1, int ) {
        for( int a = a0; a < a1; ++a ) for( int b = 0; b < N; ++b ) {
            T v = 0;
            for( int k = 0; k < 4; ++k ) v += w[ b ][ k ] * t[ a * n + j0[ b ] + k ];
            pf[ a * N + b ] = 4 * v;
        }
    } );
    return pf;
}

/// PROTOTYPE: the P1 nodes of the target, read from a file, blurred ( mean 1 )
struct BlurredNodes {
    int nv = 0;
    std::vector<T> v0;
    std::map<double,std::vector<T>> cache;
    bool load( const char *path ) {
        FILE *f = fopen( path, "rb" );
        if( ! f ) return false;
        fseek( f, 0, SEEK_END ); const long sz = ftell( f ); fseek( f, 0, SEEK_SET );
        v0.resize( sz / sizeof( T ) );
        const size_t rd = fread( v0.data(), sizeof( T ), v0.size(), f ); fclose( f );
        if( rd != v0.size() ) return false;
        nv = int( std::lround( std::sqrt( double( v0.size() ) ) ) );
        T m = 0;
        for( int a = 0; a < nv; ++a ) for( int b = 0; b < nv; ++b ) m += v0[ a * nv + b ] * ( a == 0 || a == nv - 1 ? T( 0.5 ) : T( 1 ) ) * ( b == 0 || b == nv - 1 ? T( 0.5 ) : T( 1 ) );
        m /= T( nv - 1 ) * ( nv - 1 );
        for( T &v : v0 ) v /= m;
        return true;
    }
    const std::vector<T> &at( double s ) {
        if( s <= 0 ) return v0;
        auto it = cache.find( s );
        if( it != cache.end() ) return it->second;
        const int r = int( std::ceil( 4 * s ) );
        std::vector<T> g( 2 * r + 1 ), t( v0.size() ), o( v0.size() );
        if( getenv( "SDOT_HOP_MAX" ) ) {
            const int q = std::max( 1, int( std::ceil( 2 * s ) ) );
            for( int a = 0; a < nv; ++a ) for( int b = 0; b < nv; ++b ) {
                T u = 0;
                for( int k = std::max( -q, -b ); k <= std::min( q, nv - 1 - b ); ++k ) u = std::max( u, v0[ a * nv + b + k ] );
                t[ a * nv + b ] = u;
            }
            for( int a = 0; a < nv; ++a ) for( int b = 0; b < nv; ++b ) {
                T u = 0;
                for( int k = std::max( -q, -a ); k <= std::min( q, nv - 1 - a ); ++k ) u = std::max( u, t[ ( a + k ) * nv + b ] );
                o[ a * nv + b ] = u;
            }
            return cache[ s ] = std::move( o );
        }
        for( int k = -r; k <= r; ++k ) g[ k + r ] = std::exp( -0.5 * k * k / ( s * s ) );
        for( int a = 0; a < nv; ++a ) for( int b = 0; b < nv; ++b ) {
            T u = 0, w = 0;
            for( int k = std::max( -r, -b ); k <= std::min( r, nv - 1 - b ); ++k ) { u += g[ k + r ] * v0[ a * nv + b + k ]; w += g[ k + r ]; }
            t[ a * nv + b ] = u / w;
        }
        for( int a = 0; a < nv; ++a ) for( int b = 0; b < nv; ++b ) {
            T u = 0, w = 0;
            for( int k = std::max( -r, -a ); k <= std::min( r, nv - 1 - a ); ++k ) { u += g[ k + r ] * t[ ( a + k ) * nv + b ]; w += g[ k + r ]; }
            o[ a * nv + b ] = u / w;
        }
        return cache[ s ] = std::move( o );
    }
    T sample( const std::vector<T> &v, T x, T y ) const {
        x = std::clamp( x, T( 0 ), T( nv - 1 ) ); y = std::clamp( y, T( 0 ), T( nv - 1 ) );
        const int a = std::min( int( x ), nv - 2 ), b = std::min( int( y ), nv - 2 );
        const T u = x - a, w = y - b;
        return ( 1 - u ) * ( ( 1 - w ) * v[ a * nv + b ] + w * v[ a * nv + b + 1 ] ) + u * ( ( 1 - w ) * v[ ( a + 1 ) * nv + b ] + w * v[ ( a + 1 ) * nv + b + 1 ] );
    }
};

/// DAMPED NEWTON on one level. `target`: the masses in index units ( they sum to the mass of the domain in index units ).
/// `psi`: start on input, solution on output. `S`: index units per unit of mass ( for `tol_abs` ).
template<class Integ>
void newton( Evaluator<Integ> &ev, std::vector<T> target, std::vector<T> &psi, double tol_rel, double tol_abs, double S,
             const Options &o, Stats &st ) {
    const int N = ev.N;
    const Par &par = ev.par;
    const SI nn = SI( N ) * N;
    Operator op;
    TensorPrecond M;
    M.par = par;
    const int P = N + 2, PP = P * P;
    std::vector<T> rhs( PP ), dx( PP ), trial( N * N );
    // the residual of masses `m`: the largest relative and absolute errors, the squared L2 norm, the smallest mass
    struct Err { T rel = 0, abs = 0, l2 = 0, mmin = std::numeric_limits<T>::max(); };
    auto err_of = [ & ]( const std::vector<T> &m ) {
        Err r = par.reduce( N, nn, Err{}, [&]( int a0, int a1 ) {
            Err e;
            for( SI i = SI( a0 ) * N; i < SI( a1 ) * N; ++i ) {
                const T d = std::abs( m[ i ] - target[ i ] );
                e.abs = std::max( e.abs, d );
                e.rel = std::max( e.rel, target[ i ] > 0 ? d / target[ i ] : d > 0 ? T( 1e300 ) : T( 0 ) );
                e.l2 += d * d;
                e.mmin = std::min( e.mmin, m[ i ] );
            }
            return e;
        }, []( Err &a, const Err &b ) { a.rel = std::max( a.rel, b.rel ); a.abs = std::max( a.abs, b.abs ); a.l2 += b.l2; a.mmin = std::min( a.mmin, b.mmin ); } );
        r.l2 = std::sqrt( r.l2 );
        return r;
    };
    auto done = [ & ]( const Err &e ) { return e.rel <= tol_rel || e.abs <= tol_abs * S; };

    ev.run( psi );
    ++st.nb_diag;
    // the targets scaled to what the domain holds ( a density with mass outside of it: a gaussian on a box ), as
    // `sdotplan/Solve.h` does -- the cells tile the domain whatever psi
    {
        T sm = 0, st_ = 0;
        for( int i = 0; i < N * N; ++i ) { sm += ev.mass[ i ]; st_ += target[ i ]; }
        for( auto &t : target ) t *= sm / st_;
        st.domain_mass = double( sm / S );
    }
    // THE CELLS THAT A START OR A STEP LEFT ( NEARLY ) EMPTY: an empty cell has no row in H, and on a sharp target the cells of small
    // mass are a fraction of a pixel wide, so that a step good for all the others ( the residual divided by 30 ) still crosses a
    // few of them. Each one searches its own psi, `sdotplan/WarmRepair.h`, one diagram per round. On return `ev` is at `p`.
    std::vector<T> rlo, rhi, rstep, rtouched;
    auto repair = [ & ]( std::vector<T> &p ) {
        rlo.resize( nn ); rhi.resize( nn ); rstep.resize( nn ); rtouched.resize( nn );
        int nb_small = 0;
        for( int i = 0; i < N * N; ++i ) {
            sdotplan::warm_repair_init( rlo[ i ], rhi[ i ], rstep[ i ], rtouched[ i ], 1.0, false );
            nb_small += ev.mass[ i ] < sdotplan::WarmRepair::LO * target[ i ];
        }
        if( ! sdotplan::WarmRepair::worth( nb_small, double( N ) * N ) ) return;
        for( int round = 0; round < sdotplan::WarmRepair::MAX_ROUNDS; ++round ) {
            bool moved = false;
            for( int i = 0; i < N * N; ++i ) moved |= sdotplan::warm_repair_cell( ev.mass[ i ], target[ i ], p[ i ], rlo[ i ], rhi[ i ], rstep[ i ], rtouched[ i ], 1.0 );
            if( ! moved ) break;
            ev.run( p );
            ++st.nb_diag;
        }
    };
    // ( the line search on the L2 norm of the residual: the one an ( inexact ) Newton step reduces as predicted -- the largest
    // relative error can sit on a small cell that a CG at 1e-2 leaves out )
    const T min_t = *std::min_element( target.begin(), target.end() );
    Err err = err_of( ev.mass );
    if( st.residual0 < 0 ) st.residual0 = err.abs / S;
    if( err.mmin < sdotplan::WarmRepair::LO * min_t ) {
        repair( psi );
        err = err_of( ev.mass );
        if( ! ( err.mmin > 0 ) ) {                       // ( too many cells, or a search that failed: the last resort is the Voronoi, as in `sdotplan/Solve.h` )
            std::fill( psi.begin(), psi.end(), T( 0 ) );
            ev.run( psi );
            ++st.nb_diag;
            err = err_of( ev.mass );
        }
    }
    const T eps0 = T( 0.5 ) * std::min( err.mmin, T( sdotplan::WarmRepair::LO ) * min_t );
    bool stuck = false;
    bool limited = true;
    BlurredNodes bn;
    const bool hsig = getenv( "SDOT_HSIGMA_FILE" ) && bn.load( getenv( "SDOT_HSIGMA_FILE" ) );
    T sigma = 0;                                         // in cells of this level
    // the faces of the diagram at `ev`, their density read on the blurred image
    auto blurred_faces = [ & ]( T s ) {
        const T sc = T( bn.nv - 1 ) / N;
        const std::vector<T> &v = bn.at( s * sc );
        par.slices( N * N, nn * 64, [&]( int i0, int i1, int t ) {
            T px[ ExactKernel<T>::C ], py[ ExactKernel<T>::C ];
            int nbr[ ExactKernel<T>::C ];
            bool ex = false;
            for( int i = i0; i < i1; ++i ) {
                const int n = ev.polygon( i, px, py, nbr, ex );
                ev.clear_faces( i );
                for( int e = 0; e < n; ++e ) {
                    const int j = nbr[ e ], f = e + 1 == n ? 0 : e + 1;
                    if( j < 0 ) continue;
                    const T len = std::hypot( px[ f ] - px[ e ], py[ f ] - py[ e ] );
                    if( len == 0 ) continue;
                    const int m = std::clamp( int( std::ceil( len * sc ) ), 1, 64 );
                    T r = 0;
                    for( int q = 0; q < m; ++q ) {
                        const T u = ( q + T( 0.5 ) ) / m;
                        r += bn.sample( v, ( T( i / N ) + T( 0.5 ) + px[ e ] + u * ( px[ f ] - px[ e ] ) ) * sc, ( T( i % N ) + T( 0.5 ) + py[ e ] + u * ( py[ f ] - py[ e ] ) ) * sc );
                    }
                    ev.put_face( i, j, r / m * len / ( 2 * std::hypot( T( j / N - i / N ), T( j % N - i % N ) ) ), ev.ex_th[ t ] );
                }
            }
        } );
        ev.gather_extras();
        ev.symmetrize();
    };
    T lambda = 0;
    for( int it = 1; it <= o.max_iter && ! done( err ) && ! ev.exhausted; ++it ) {
        // the linear system H dx = target - m
        const double tl = now_ms();
        op.bind( ev );
        M.build( op );
        std::fill( rhs.begin(), rhs.end(), 0 );
        T sb = par.template sum<T>( N, nn, [&]( int a0, int a1 ) {
            T s = 0;
            for( int a = a0; a < a1; ++a ) for( int b = 0; b < N; ++b ) s += rhs[ ( a + 1 ) * P + b + 1 ] = target[ a * N + b ] - ev.mass[ a * N + b ];
            return s;
        } );
        // ( H is singular along the constants: drop the mean of the rhs, the rounding of the sums of the masses )
        sb /= T( N ) * N;
        par.slices( N, nn, [&]( int a0, int a1, int ) {
            for( int a = a0; a < a1; ++a ) for( int b = 0; b < N; ++b ) rhs[ ( a + 1 ) * P + b + 1 ] -= sb;
        } );
        // THE STEP LIMIT. A face that sits where the density is low has a small weight in H, however dense the density one pixel
        // further: a sharp edge of the target gives steps of 100 in psi where a move of one pixel would do, and a line search
        // along such a step crawls. H + lambda L ( L the laplacian of the grid ) stiffens these faces, and these only: lambda
        // grows until no bisector of the grid moves by more than `step_limit`, and goes back to 0 near the solution
        const T lam0 = 1e-3;
        auto stiffen = [ & ]( T d ) {
            par.slices( N, nn, [&]( int a0, int a1, int ) {
                for( int k = 0; k < 8; ++k ) {
                    if( T1::off_x[ k ] && T1::off_y[ k ] ) continue;
                    T *w = ev.W.data() + SI( k ) * PP;
                    for( int a = a0; a < a1; ++a ) for( int b = 0; b < N; ++b ) {
                        const int a2 = a + T1::off_x[ k ], b2 = b + T1::off_y[ k ];
                        if( a2 >= 0 && a2 < N && b2 >= 0 && b2 < N ) w[ ( a + 1 ) * P + b + 1 ] += d;
                    }
                }
            } );
        };
        auto bisector_move = [ & ]() {
            return par.reduce( N, nn, T( 0 ), [&]( int a0, int a1 ) {
                T j = 0;
                for( int a = a0; a < a1; ++a ) for( int b = 0; b < N; ++b ) {
                    const T v = dx[ ( a + 1 ) * P + b + 1 ];
                    if( a + 1 < N ) j = std::max( j, std::abs( v - dx[ ( a + 2 ) * P + b + 1 ] ) );
                    if( b + 1 < N ) j = std::max( j, std::abs( v - dx[ ( a + 1 ) * P + b + 2 ] ) );
                }
                return j / 2;
            }, []( T &a, const T &b ) { a = std::max( a, b ); } );
        };
        if( getenv( "SDOT_CHECK_FACES" ) && hsig ) {
            auto sum = [ & ]() { T a = 0; for( SI k = 0; k < 8; ++k ) for( int r = 0; r < N; ++r ) for( int b = 0; b < N; ++b ) a += ev.W[ k * PP + ( r + 1 ) * P + b + 1 ]; for( auto &e : ev.extras ) a += e.w; return a; };
            const T s0 = sum();
            std::vector<T> w0( ev.W.begin(), ev.W.end() );
            auto x0 = ev.extras;
            blurred_faces( 0 );
            T dm = 0; for( size_t q = 0; q < w0.size(); ++q ) dm = std::max( dm, std::abs( w0[ q ] - ev.W[ q ] ) );
            fprintf( stderr, "CHECK N %d sum exact %.8g sampled %.8g max diff %.3g extras %zu %zu\n", N, s0, sum(), dm, x0.size(), ev.extras.size() );
            std::copy( w0.begin(), w0.end(), ev.W.begin() ); ev.extras = x0;
        }
        lambda = lambda / 4 < lam0 || ! limited ? 0 : lambda / 4;
        { const T dn = getenv( "SDOT_SIG_DOWN" ) ? atof( getenv( "SDOT_SIG_DOWN" ) ) : 2; sigma = sigma / dn < T( 0.25 ) || ! limited ? 0 : sigma / dn; }
        const T sig_fixed = getenv( "SDOT_SIG_FIXED" ) ? atof( getenv( "SDOT_SIG_FIXED" ) ) : 0;
        if( hsig && sig_fixed > 0 ) sigma = limited && err.rel > 1 ? sig_fixed : 0;
        if( hsig ) lambda = 0;
        if( lambda > 0 ) { stiffen( lambda ); op.bind( ev ); M.build( op ); }
        if( sigma > 0 ) { blurred_faces( sigma ); op.bind( ev ); M.build( op ); }
        int cg_its = pcg_fused( op, M, rhs, dx, o.cg_rtol, 20000, par );
        for( int round = 0; limited && ! ( hsig && sig_fixed > 0 ) && round < 16 && bisector_move() > ( getenv( "SDOT_STEP_LIMIT" ) ? atof( getenv( "SDOT_STEP_LIMIT" ) ) : o.step_limit ); ++round ) {
            if( hsig ) {
                if( sigma >= 32 ) break;
                sigma = sigma > 0 ? 2 * sigma : T( 0.5 );
                blurred_faces( sigma );
            } else {
            const T d = lambda > 0 ? 3 * lambda : lam0;
            lambda += d;
            stiffen( d );
            }
            op.bind( ev );
            M.build( op );
            cg_its += pcg_fused( op, M, rhs, dx, o.cg_rtol, 20000, par );
        }
        st.lin_nb_iter += cg_its;
        st.t_lin += ( now_ms() - tl ) / 1e3;
        // damping ( a limited step is not a Newton step: nothing says that the norm of the residual decreases at the rate of `tau`, nor
        // for a small `tau`. Newton itself takes over when it does not )
        const T slope = lambda > 0 || sigma > 0 ? T( 0.02 ) : T( 0.5 ), tau_min = lambda > 0 || sigma > 0 ? T( 0.1 ) : T( 1e-6 );
        T tau = 1;
        while( true ) {
            par.slices( N, nn, [&]( int a0, int a1, int ) {
                for( int a = a0; a < a1; ++a ) for( int b = 0; b < N; ++b ) trial[ a * N + b ] = psi[ a * N + b ] + tau * dx[ ( a + 1 ) * P + b + 1 ];
            } );
            ev.run( trial );
            ++st.nb_diag;
            Err e2 = err_of( ev.mass );
            // ( a trial whose cells outgrow the exact path's rings is rejected like one that empties a cell )
            if( ! ev.exhausted && e2.l2 <= ( 1 - slope * tau ) * err.l2 ) {
                if( e2.mmin <= eps0 && ! getenv( "SDOT_NO_TRIAL_REPAIR" ) ) {
                    repair( trial );
                    e2 = err_of( ev.mass );
                }
                if( ! ev.exhausted && e2.mmin > eps0 && e2.l2 <= ( 1 - slope * tau ) * err.l2 ) { err = e2; break; }
            }
            tau /= 2;
            ++st.nb_backtracks;
            if( tau < tau_min ) { stuck = true; break; }
        }
        // ( no step: the last accepted weights stay, with their outputs )
        if( stuck ) {
            ev.run( psi );
            ++st.nb_diag;
            if( ! limited ) break;
            limited = stuck = false;
            continue;
        }
        limited = true;
        psi.swap( trial );
        ++st.nb_iter;
    }
    st.residual = err.abs / S;
    st.exhausted = st.exhausted || ev.exhausted;
    st.status = ev.exhausted || stuck ? 6 : done( err ) ? 1 : 2;
    st.t_diag = ( ev.t_geom + ev.t_int ) / 1e3;
}

/// THE SOLVE: `make_integ( n, h_n, blur )` gives the integrals of level `n` ( side `h_n` ), `blur` the width of the
/// continuation ( `GaussCells`, the others ignore it ); `nu` the masses of the diracs ( C order,
/// they sum to 1 ), `w0` a start ( physical weights ) or null. Writes the weights ( `w[ 0 ] = 0` ), the masses of the cells,
/// their barycenters and the cost, all physical. In index units inside: `S = n^2` per unit of mass at level `n`. The threads
/// are those of `queue`.
template<class MakeInteg>
Stats solve( const CpuQueue &queue, int N, double ox, double oy, double h, MakeInteg &&make_integ, const double *nu, const double *w0,
             const Options &o, double *weights, double *masses, double *bary, double &cost ) {
    Stats st;
    const double t0 = now_ms();
    const Par par( queue );
    const int n0 = w0 ? N : std::min( std::max( o.n0, 32 ), N );
    std::vector<T> psi;
    if( w0 ) {
        psi.resize( size_t( N ) * N );
        for( int i = 0; i < N * N; ++i ) psi[ i ] = w0[ i ] / ( h * h );
    } else
        psi.assign( size_t( n0 ) * n0, T( 0 ) );
    // the masses of each level: sums over the blocks of the finest
    std::vector<T> nu_l( nu, nu + size_t( N ) * N );
    std::vector<std::vector<T>> nus{ nu_l };
    for( int n = N; n > n0; n /= 2 ) {
        const auto &f = nus.back();
        std::vector<T> c( size_t( n / 2 ) * ( n / 2 ) );
        for( int a = 0; a < n / 2; ++a ) for( int b = 0; b < n / 2; ++b )
            c[ a * ( n / 2 ) + b ] = f[ ( 2 * a ) * n + 2 * b ] + f[ ( 2 * a ) * n + 2 * b + 1 ] + f[ ( 2 * a + 1 ) * n + 2 * b ] + f[ ( 2 * a + 1 ) * n + 2 * b + 1 ];
        nus.push_back( std::move( c ) );
    }
    // THE WIDTH CONTINUATION ( gaussians, no start given ): on the coarsest level, the density blurred by a gaussian of width
    // `b` ( from half the grid down by halves ), then the density itself -- far tails would give Newton cells of no mass
    using Integ0 = DECAYED_TYPE_OF( make_integ( n0, h, 0.0 ) );
    if constexpr ( requires( const Integ0 &ig ) { ig.smallest_sigma; } ) {
        if( ! w0 ) {
            const double hl = h * N / n0, S = double( n0 ) * n0, smin = make_integ( n0, hl, 0.0 ).smallest_sigma;
            std::vector<T> target( nus.back().size() );
            for( size_t i = 0; i < target.size(); ++i ) target[ i ] = nus.back()[ i ] * S;
            for( double b = h * N / 2; b > smin / 4; b /= 2 ) {
                Evaluator<Integ0> ev( n0, make_integ( n0, hl, b ), par );
                Stats sl;
                newton( ev, target, psi, std::max( o.tol_rel, o.ctol ), 0.0, S, o, sl );
                st.nb_iter += sl.nb_iter; st.nb_diag += sl.nb_diag; st.nb_backtracks += sl.nb_backtracks; st.lin_nb_iter += sl.lin_nb_iter;
                st.t_diag += sl.t_diag; st.t_lin += sl.t_lin; ++st.nb_levels;
                if( st.residual0 < 0 ) st.residual0 = sl.residual0;
            }
        }
    }
    for( int n = n0, l = int( nus.size() ) - 1; n <= N; n *= 2, --l ) {
        if( n > n0 ) psi = refine_psi( n / 2, psi, par );
        const double S = double( n ) * n, hl = h * N / n;
        std::vector<T> target( nus[ l ].size() );
        for( size_t i = 0; i < target.size(); ++i ) target[ i ] = nus[ l ][ i ] * S;
        auto integ = make_integ( n, hl, 0.0 );
        Evaluator<DECAYED_TYPE_OF( integ )> ev( n, integ, par );
        Stats sl;
        const bool fine = n == N;
        newton( ev, target, psi, fine ? o.tol_rel : std::max( o.tol_rel, o.ctol ), fine ? o.tol_abs : 0.0, S, o, sl );
        st.nb_iter += sl.nb_iter; st.nb_diag += sl.nb_diag; st.nb_backtracks += sl.nb_backtracks; st.lin_nb_iter += sl.lin_nb_iter;
        st.t_diag += sl.t_diag; st.t_lin += sl.t_lin; ++st.nb_levels;
        if( st.residual0 < 0 ) st.residual0 = sl.residual0;
        st.exhausted = st.exhausted || sl.exhausted;
        if( ! fine ) continue;                           // ( a level that failed still gives a start )
        st.status = sl.status;
        st.residual = sl.residual;
        st.domain_mass = sl.domain_mass;
        // the outputs, physical
        std::vector<T> bx, by, m2;
        ev.moments( psi, bx, by, m2 );
        cost = par.template sum<double>( N, SI( N ) * N, [&]( int a0, int a1 ) {
            double c = 0;
            for( int i = a0 * N; i < a1 * N; ++i ) {
                weights[ i ] = h * h * ( psi[ i ] - psi[ 0 ] );
                masses[ i ] = ev.mass[ i ] / S;
                bary[ 2 * i + 0 ] = ox + h * ( T( i / N ) + T( 0.5 ) + bx[ i ] );
                bary[ 2 * i + 1 ] = oy + h * ( T( i % N ) + T( 0.5 ) + by[ i ] );
                c += h * h * m2[ i ] / S;
            }
            return c;
        } );
    }
    st.t_total = ( now_ms() - t0 ) / 1e3;
    return st;
}

} // namespace grid
} // namespace sdot

