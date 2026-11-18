#pragma once

// THE SOLVER OF THE DIRACS OF A 2D GRID ( `SumOfDiracsOnAGrid`, `SdotPlanNd._build_grid` ): damped Newton on the weights of
// N x N seeds at the centres of the square cells of a grid, N a power of two, the domain being the box of the grid.
// A port of `bench/experiments/grid_stencil/newton.cpp`. No positions are stored: everything is in INDEX UNITS ( seed
// `( a, b )` at `( a, b )`, the domain `[ -1/2, N - 1/2 ]^2`, `psi = w / h^2` ), physical units only at the edges.
//
//   geometry   ring-1 stencil ( `StencilKernel.h` ) + the ZONE CERTIFICATE: the ring-1 cells contain the true ones, so the sum of
//              their areas on a zone exceeds the zone exactly where a cell is wrong. Rounds: the cells that touch a bad zone
//              rebuilt with more seeds ( ring 2 in SIMD, then the seeds within 4 sqrt 2, 8 sqrt 2, .. by `ExactKernel.h` ), filling
//              a table of the halves of the bad zones, until none is bad.
//              ( `use_zones = false`: the former local certificates, ring 2 on the flagged cells, exact clipping with psi_max )
//   integrals  the uniform density in SIMD ( the area, the lengths of the faces ); any other density cell by cell: the cell is
//              rebuilt from its neighbours as a sdot cell ( `ProviderList` ) and integrated by `measure_and_facets`
//   linear     H = dm / dpsi, matrix-free: per cell 8 weights toward the ring-1 neighbours in SoA planes, plus a short list of
//              extra ( i, j, w ) for the faces of the cells that came from ring 2 or from the exact path. PCG in row
//              passes, preconditioned by the weighted semi-coarsened tent blurs ( `TensorPrecond` )
//   start      coarse to fine: the solve on n0, 2 n0, ..., N, each level from the cubic interpolation of the previous psi
//   damping    sharp targets ( cells a hundredth of a pixel wide beside cells of ten pixels ): the line search on the residual,
//              `tau` brought under the limit of the cells that a trial left under the floor, the cells that remain empty repaired
//              one by one ( `newton` )
//   threads    those of the CPU queue ( `Par` ): contiguous slices of rows or of cells, the sums combined in thread order

#include "StencilKernel.h"
#include "ExactKernel.h"
#include "GridDensities.h"
#include "../cell/Engine2Reg.h"
#include "../sdotplan/Sweep.h"
#include "../sdotplan/Bounds.h"
#include "../sdotplan/WarmRepair.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <cstdint>

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
    struct Extra { int i, j; T w; unsigned char l = 0; };   // ( l: the fused zones, the level of the cell that wrote it )
    std::vector<Extra> extras;                           // faces toward the seeds out of ring 1
    std::vector<std::vector<Extra>> ex_th;               // ( the same, per thread, before they are gathered )
    // stats of the last call
    int n_flag1 = 0, n_exact = 0;
    T pmax = 0;                                          // the largest psi, for the exact path ( `pe`: psi padded for it )
    bool exhausted = false;                              // the exact path ran out of its list ( a cell wider than all the rings ), or
                                                         // a cell that sdot could not build ( `DistCells::failed` )
    double t_geom = 0, t_int = 0;
    // ZONE CERTIFICATE ( `use_zones`, see `geometry_zones` )
    bool use_zones = true;
    int zone0 = 8;                                       // side of the zones of the ring-1 table, in pixels
    T zone_min = T( 1 );                                 // the zones of the next tables halve down to this side
    bool zone_predict = true;                            // ring 2 before the first table on the cells the local certificate flags
    bool use_reg = true;                                 // the rounds past ring 2 in the register kernel of the cells
    bool use_tree = false;                               // past ring 2: the exact cell at once, by a mipmap of affine majorants
                                                         // ( measured 1.6 - 3.3x slower than doubling the radius: 110 - 270 blocks per cell )
    bool fused_zones = true;                             // the zones in ONE pass with the integration ( `run_zones` ): no plane, no polygon
                                                         // kept, a level and a box per cell; false: `geometry_zones`, then the integration
    std::vector<unsigned char> lvl;                      // ( fused ) the level of each cell: 0, 1 ring 1, 2.. the seeds within sqrt( 2 ) 2^( lvl - 1 ), 255 exact
    std::vector<float> zbx;                              // ( fused ) the box of each cell, floats rounded outward ( lo > hi: empty )
    ExactKernel<T> ek1{ 1 };                             // ( fused ) ring 1 as a polygon, on `p1`
    bool fused_last = false;                             // the last diagram came from `run_zones` ( `polygon` builds the cells from `lvl` )
    bool use_prev = false;                               // ring 1 = the 3 x 3 stencil and the cuts past it of the previous call ( `prev_*` )
    bool rebuild_tree = false;                           // the cells of the bad zones of the first table to their exact cell at once, by
                                                         // the mipmap of affine majorants ( `tree_clip` ): one table, no rounds
    std::vector<int> prev_off, prev_nb;                  // per cell, the neighbours past ring 1 of the last call ( `use_prev` )
    std::vector<int> ek_idx;                             // ( the index in `ek` of an offset, ( 2 RB + 1 )^2 )
    int ek_idx_RB = -1;
    int ek_index( int dx, int dy ) {
        const int R = ek.RB, S = 2 * R + 1;
        if( ek_idx_RB != R ) {
            ek_idx.assign( size_t( S ) * S, -1 );
            for( int q = 0; q < int( ek.ox.size() ); ++q ) ek_idx[ size_t( ek.ox[ q ] + R ) * S + ek.oy[ q ] + R ] = q;
            ek_idx_RB = R;
        }
        return std::abs( dx ) > R || std::abs( dy ) > R ? -1 : ek_idx[ size_t( dx + R ) * S + dy + R ];
    }
    struct Maj { T gx, gy, c; };                         // psi_j - |x_j|^2 <= gx a_j + gy b_j + c on a block ( index coordinates )
    std::vector<std::vector<Maj>> pmip;                  // affine majorants of psi over blocks of side mip_ls << l ( l = 0: leaves )
    int mip_ls = 4;
    struct Poly { int n = 0, nm = 0; bool exh = false, fin = false, glob = false; T R = 0; std::vector<T> x, y; std::vector<int> pl; };   // ( pl: offsets of `ek`, -1: box -- or the neighbours themselves if `glob`; the offsets [ 0, nm ) cut it, those within R; fin: exact )
    std::vector<int> slot;                               // N * N: the polygon of a rebuilt cell in `polys` ( -1: the ring-1 planes )
    std::vector<Poly> polys;                             // ( the first `nb_polys` are in use )
    int nb_polys = 0;
    std::vector<T> area1, area2;                         // N * N: the areas of the ring-1 cells, of the ring-2 cells rebuilt
    std::vector<int> round_cells, round_bad;             // per round ( 0: ring 1 ): the cells built, the bad zones of its table
    int n_rounds = 0;
    double t_zone[ 4 ] = {};                             // ( ms, cumulated: ring 1, its table, rebuilds, tables )

    Evaluator( int N, const Integ &integ, Par par ) : N( N ), P1( N + 2 ), par( par ), integ( par.nt, integ ),
        flag1( N * N ), flag2( N * N ), mark( N * N ), pl1( 1, false, false ), pl2( 1, false, false ), row_th( par.nt ),
        fine_th( par.nt ), mass( N * N ), W( 8 * ( N + 2 ) * ( N + 2 ) ), ex_th( par.nt ) {
    }
    /// the planes of every cell, for the paths that keep them ( `geometry_zones`, `geometry_cert`: not the fused zones )
    void ensure_planes() {
        if( pl1.N == N ) return;
        pl1 = typename K1::Planes( N, false, false );
        pl2 = typename K2::Planes( N, false, false );
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
        if( fused_last ) return level_polygon( i, lvl[ i ], px, py, nbr, exh );
        if( use_zones && flag1[ i ] ) {
            const Poly &q = polys[ slot[ i ] ];
            for( int e = 0; e < q.n; ++e ) { px[ e ] = q.x[ e ]; py[ e ] = q.y[ e ]; nbr[ e ] = q.glob || q.pl[ e ] < 0 ? q.pl[ e ] : i + ek.ox[ q.pl[ e ] ] * N + ek.oy[ q.pl[ e ] ]; }
            exh = exh || q.exh;
            return q.n;
        }
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

    // THE MASS OF CELL i AT THE WEIGHTS `p`, ITS EDGES FROZEN: only the neighbours `ranks` cut it ( thread `t`'s scratch )
    T frozen_mass( int t, int i, const int *ranks, int nb, const std::vector<T> &p ) {
        T m = 0;
        if constexpr ( Integ::simd ) {
            constexpr int C = ExactKernel<T>::C;
            T px[ C ], py[ C ], qx[ C ], qy[ C ], face[ C ];
            const T a = T( i / N ), b = T( i % N ), lx = T( -0.5 ) - a, hx = T( N ) - T( 0.5 ) - a, ly = T( -0.5 ) - b, hy = T( N ) - T( 0.5 ) - b;
            px[ 0 ] = lx; py[ 0 ] = ly; px[ 1 ] = hx; py[ 1 ] = ly; px[ 2 ] = hx; py[ 2 ] = hy; px[ 3 ] = lx; py[ 3 ] = hy;
            int n = 4;
            for( int e = 0; e < nb && n; ++e ) {
                const T dx = T( ranks[ e ] / N ) - a, dy = T( ranks[ e ] % N ) - b, c = ( dx * dx + dy * dy + p[ i ] - p[ ranks[ e ] ] ) / 2;
                int q = 0;
                for( int v = 0; v < n && q + 2 < C; ++v ) {
                    const int w = v + 1 == n ? 0 : v + 1;
                    const T s0 = dx * px[ v ] + dy * py[ v ] - c, s1 = dx * px[ w ] + dy * py[ w ] - c;
                    if( s0 <= 0 ) { qx[ q ] = px[ v ]; qy[ q ] = py[ v ]; ++q; }
                    if( ( s0 < 0 && s1 > 0 ) || ( s0 > 0 && s1 < 0 ) ) {
                        const T u = s0 / ( s0 - s1 );
                        qx[ q ] = px[ v ] + u * ( px[ w ] - px[ v ] ); qy[ q ] = py[ v ] + u * ( py[ w ] - py[ v ] ); ++q;
                    }
                }
                std::copy( qx, qx + q, px ); std::copy( qy, qy + q, py );
                n = q;
            }
            if( n >= 3 ) integ[ t ].polygon( n, px, py, a, b, m, face );
        } else {
            integ[ t ].bind( p );
            integ[ t ].cell( i, ranks, nb, m, []( int, T ) {} );
            integ[ t ].failed = false;
        }
        return m;
    }

    void geometry( const std::vector<T> &psi ) {
        fused_last = false;
        if( use_zones ) geometry_zones( psi );
        else geometry_cert( psi );
    }

    // THE MIPMAP OF AFFINE MAJORANTS of f = psi - |x|^2 ( concave where psi'' <= 2: nearly flat on smooth targets, so the plane
    // stays tight on large blocks ): on a leaf ( `mip_ls` x `mip_ls` seeds ), the least-squares slope and the offset
    // that puts the plane above every seed; on a group of 2 x 2 blocks, the mean slope of the children and an offset above
    // their planes on their boxes ( a corner of each )
    void build_mip( const std::vector<T> &psi ) {
        mip_ls = std::min( 4, N );
        pmip.clear();
        for( int side = mip_ls; side <= N; side *= 2 ) {
            const int m = N / side;
            std::vector<Maj> v( size_t( m ) * m );
            if( side == mip_ls ) {
                T su2 = 0;
                for( int c = 0; c < side; ++c ) su2 += ( c - T( side - 1 ) / 2 ) * ( c - T( side - 1 ) / 2 );
                su2 *= side;
                for( int u = 0; u < m; ++u )
                    for( int w = 0; w < m; ++w ) {
                        T sx = 0, sy = 0;
                        for( int c = 0; c < side; ++c ) for( int d = 0; d < side; ++d ) {
                            const T ja = T( u * side + c ), jb = T( w * side + d ), f = psi[ size_t( u * side + c ) * N + w * side + d ] - ja * ja - jb * jb;
                            sx += f * ( c - T( side - 1 ) / 2 ); sy += f * ( d - T( side - 1 ) / 2 );
                        }
                        Maj &q = v[ size_t( u ) * m + w ];
                        q.gx = side > 1 ? sx / su2 : 0; q.gy = side > 1 ? sy / su2 : 0;
                        q.c = -std::numeric_limits<T>::infinity();
                        for( int c = 0; c < side; ++c ) for( int d = 0; d < side; ++d ) {
                            const int ja = u * side + c, jb = w * side + d;
                            q.c = std::max( q.c, psi[ size_t( ja ) * N + jb ] - T( ja ) * ja - T( jb ) * jb - q.gx * ja - q.gy * jb );
                        }
                    }
            } else {
                const auto &q = pmip.back();
                const int mq = 2 * m, hs = side / 2;
                for( int u = 0; u < m; ++u )
                    for( int w = 0; w < m; ++w ) {
                        Maj &r = v[ size_t( u ) * m + w ];
                        r.gx = r.gy = 0;
                        for( int c = 0; c < 4; ++c ) { const Maj &k = q[ size_t( 2 * u + c / 2 ) * mq + 2 * w + c % 2 ]; r.gx += k.gx / 4; r.gy += k.gy / 4; }
                        r.c = -std::numeric_limits<T>::infinity();
                        for( int c = 0; c < 4; ++c ) {
                            const int cu = 2 * u + c / 2, cw = 2 * w + c % 2;
                            const Maj &k = q[ size_t( cu ) * mq + cw ];
                            // max over the child's box of ( g_k - g_r ) . x + c_k: at a corner
                            const T dgx = k.gx - r.gx, dgy = k.gy - r.gy;
                            const T xa = dgx > 0 ? T( cu * hs + hs - 1 ) : T( cu * hs ), xb = dgy > 0 ? T( cw * hs + hs - 1 ) : T( cw * hs );
                            r.c = std::max( r.c, dgx * xa + dgy * xb + k.c );
                        }
                        r.c += T( 1e-12 ) * ( T( 1 ) + std::abs( r.c ) );
                    }
            }
            pmip.push_back( std::move( v ) );
        }
    }

    /// THE EXACT CELL i from a polygon that contains it ( px, py, nbr: neighbours, -1 box ), cut by every seed of a block of
    /// the mipmap that can still cut it ( the majorant of the block against each vertex ). Near blocks first. The seeds with |dx|, |dy| <= `skip` already cut it.
    /// False if the polygon outgrows the arrays.
    bool tree_clip( int i, const std::vector<T> &psi, T *px, T *py, int *nbr, int &n, int skip ) const {
        constexpr int C = ExactKernel<T>::C;
        const int a = i / N, b = i % N, L = int( pmip.size() );
        const T pi = psi[ i ];
        T qx[ C ], qy[ C ], sv[ C ]; int qn[ C ];
        struct It { int l, u, v; };
        It st[ 4 * 64 ];
        int sp = 0;
        st[ sp++ ] = { L - 1, 0, 0 };
        while( sp && n ) {
            const It it = st[ --sp ];
            const int side = mip_ls << it.l, m = N / side;
            const Maj &M = pmip[ it.l ][ size_t( it.u ) * m + it.v ];
            // seed j cuts vertex P ( absolute ) off when psi_j - |x_j - P|^2 > psi_i - |P - x_i|^2; with f_j <= g . x_j + c,
            // psi_j - |x_j - P|^2 <= ( g + 2 P ) . x_j + c - |P|^2: affine in x_j, its maximum over the box is at a corner
            const T X0 = T( it.u * side ), X1 = T( it.u * side + side - 1 ), Y0 = T( it.v * side ), Y1 = T( it.v * side + side - 1 );
            bool keep = false;
            for( int e = 0; e < n && ! keep; ++e ) {
                const T Px = px[ e ] + a, Py = py[ e ] + b, hx = M.gx + 2 * Px, hy = M.gy + 2 * Py;
                const T val = hx * ( hx > 0 ? X1 : X0 ) + hy * ( hy > 0 ? Y1 : Y0 ) + M.c - Px * Px - Py * Py;
                const T r2 = px[ e ] * px[ e ] + py[ e ] * py[ e ];
                keep = val > pi - r2 - T( 1e-11 ) * ( T( 1 ) + std::abs( M.c ) + Px * Px + Py * Py + std::abs( pi ) );
            }
            if( ! keep ) continue;
            if( it.l ) {
                // the four children, the nearest to the seed popped first
                It ch[ 4 ]; T dd[ 4 ];
                const int hs = side / 2;
                for( int c = 0; c < 4; ++c ) {
                    ch[ c ] = { it.l - 1, 2 * it.u + c / 2, 2 * it.v + c % 2 };
                    const T cx0 = T( ch[ c ].u * hs - a ), cy0 = T( ch[ c ].v * hs - b );
                    const T dx = std::clamp( T( 0 ), cx0, cx0 + hs - 1 ), dy = std::clamp( T( 0 ), cy0, cy0 + hs - 1 );
                    dd[ c ] = dx * dx + dy * dy;
                }
                int ord[ 4 ] = { 0, 1, 2, 3 };
                std::sort( ord, ord + 4, [&]( int p, int q ) { return dd[ p ] > dd[ q ]; } );
                for( int c = 0; c < 4; ++c ) st[ sp++ ] = ch[ ord[ c ] ];
                continue;
            }
            for( int c = 0; c < side && n; ++c )
                for( int d = 0; d < side && n; ++d ) {
                    const int ja = it.u * side + c, jb = it.v * side + d, kx = ja - a, ky = jb - b;
                    if( std::abs( kx ) <= skip && std::abs( ky ) <= skip ) continue;
                    const T fx = T( kx ), fy = T( ky ), cc = ( fx * fx + fy * fy + pi - psi[ size_t( ja ) * N + jb ] ) / 2;
                    bool out = false;
                    for( int e = 0; e < n; ++e ) { sv[ e ] = fx * px[ e ] + fy * py[ e ] - cc; out |= sv[ e ] > 0; }
                    if( ! out ) continue;
                    if( n + 1 >= C ) return false;
                    int q = 0;
                    for( int e = 0; e < n; ++e ) {
                        const int f = e + 1 == n ? 0 : e + 1;
                        if( sv[ e ] <= 0 ) {
                            qx[ q ] = px[ e ]; qy[ q ] = py[ e ]; qn[ q++ ] = nbr[ e ];
                            if( sv[ f ] > 0 ) { const T t = sv[ e ] / ( sv[ e ] - sv[ f ] ); qx[ q ] = px[ e ] + t * ( px[ f ] - px[ e ] ); qy[ q ] = py[ e ] + t * ( py[ f ] - py[ e ] ); qn[ q++ ] = ja * N + jb; }
                        } else if( sv[ f ] <= 0 ) { const T t = sv[ e ] / ( sv[ e ] - sv[ f ] ); qx[ q ] = px[ e ] + t * ( px[ f ] - px[ e ] ); qy[ q ] = py[ e ] + t * ( py[ f ] - py[ e ] ); qn[ q++ ] = nbr[ e ]; }
                    }
                    n = q;
                    for( int e = 0; e < n; ++e ) { px[ e ] = qx[ e ]; py[ e ] = qy[ e ]; nbr[ e ] = qn[ e ]; }
                }
        }
        return true;
    }

    // THE SEEDS [ o, o1 ) OF `ek` as cuts for the register kernel of the cells ( `cell/Engine2Reg.h` ): no radius of security
    struct RingProvider {
        const T *pi; int P, N, a, b; const int *ox, *oy; int o, o1;
        template<class S>
        bool next( const S &, NothingLocal &, Plane<T,2> &p ) {
            while( o < o1 ) {
                const int q = o++, na = a + ox[ q ], nb = b + oy[ q ];
                if( na < 0 || na >= N || nb < 0 || nb >= N ) continue;
                const T kx = T( ox[ q ] ), ky = T( oy[ q ] );
                p.dir[ 0 ] = kx; p.dir[ 1 ] = ky; p.off = ( kx * kx + ky * ky + pi[ 0 ] - pi[ ox[ q ] * P + oy[ q ] ] ) / 2; p.id = q;
                return true;
            }
            return false;
        }
    };
    /// the polygon ( px, py, pl, n ) of cell i cut further by the seeds [ o0, o1 ) of `ek`, in the registers ( `pe` padded by
    /// `ek.RB` ). False when it does not fit ( the caller goes to `ExactKernel` )
    bool reg_clip( int i, T *px, T *py, int *pl, int &n, int o0, int o1 ) const {
        constexpr int C = ExactKernel<T>::C;
        constexpr SI words = Local2<T>::words_for( C );
        alignas( 64 ) std::int32_t buf[ words + 16 ];
        Carver cv{ buf, words + 16 };
        Local2<T> c;
        if( ! c.attach( cv, C ) || n > C ) return false;
        for( int e = 0; e < n; ++e ) { c.vx[ e ] = px[ e ]; c.vy[ e ] = py[ e ]; c.cid[ e ] = pl[ e ]; }
        c.nb = n;
        if( n == 0 ) return true;
        const int a = i / N, b = i % N, P = N + 2 * ek.RB;
        RingProvider f{ pe.data() + size_t( a + ek.RB ) * P + ( b + ek.RB ), P, N, a, b, ek.ox.data(), ek.oy.data(), o0, o1 };
        NothingLocal loc;
        if( run_2d_registers( c, f, loc ) ) return false;
        n = c.nb;
        for( int e = 0; e < n; ++e ) { px[ e ] = c.vx[ e ]; py[ e ] = c.vy[ e ]; pl[ e ] = c.cid[ e ]; }
        return true;
    }

    // the area of the polygon ( px, py, n ) inside the box [ x0, x1 ] x [ y0, y1 ] ( same frame )
    static T rect_area( const T *px, const T *py, int n, T x0, T x1, T y0, T y1 ) {
        constexpr int C = ExactKernel<T>::C + 8;
        T ax[ C ], ay[ C ], bx[ C ], by[ C ];
        int m = std::min( n, C - 8 );
        // ( in a frame centred on the box: the pieces of a far cell lose no digits )
        const T cx = ( x0 + x1 ) / 2, cy = ( y0 + y1 ) / 2;
        x0 -= cx; x1 -= cx; y0 -= cy; y1 -= cy;
        for( int e = 0; e < m; ++e ) { ax[ e ] = px[ e ] - cx; ay[ e ] = py[ e ] - cy; }
        auto cut = [&]( T *sx, T *sy, int k, T *dx, T *dy, T kx, T ky, T c ) {   // keep kx x + ky y <= c
            int q = 0;
            for( int e = 0; e < k; ++e ) {
                const int f = e + 1 == k ? 0 : e + 1;
                const T s0 = kx * sx[ e ] + ky * sy[ e ] - c, s1 = kx * sx[ f ] + ky * sy[ f ] - c;
                if( s0 <= 0 ) { dx[ q ] = sx[ e ]; dy[ q ] = sy[ e ]; ++q; }
                if( ( s0 <= 0 ) != ( s1 <= 0 ) ) { const T u = s0 / ( s0 - s1 ); dx[ q ] = sx[ e ] + u * ( sx[ f ] - sx[ e ] ); dy[ q ] = sy[ e ] + u * ( sy[ f ] - sy[ e ] ); ++q; }
            }
            return q;
        };
        m = cut( ax, ay, m, bx, by, -1, 0, -x0 );
        m = cut( bx, by, m, ax, ay, 1, 0, x1 );
        m = cut( ax, ay, m, bx, by, 0, -1, -y0 );
        m = cut( bx, by, m, ax, ay, 0, 1, y1 );
        T s = 0;
        for( int e = 0; e < m; ++e ) { const int f = e + 1 == m ? 0 : e + 1; s += ax[ e ] * ay[ f ] - ay[ e ] * ax[ f ]; }
        return s / 2;
    }

    // THE ZONE CERTIFICATE. A cell built from fewer seeds contains the true one, so on a zone Z the excess
    // E_Z = sum_i |C_i n Z| - |Z| is >= 0 ( rounding aside ), and E_Z = 0 proves every cell exact in Z: no radius of security, no
    // psi_max. The tables come with the cells: ring 1 on every cell fills a table of `zone0`-pixel zones; the cells that touch a
    // bad zone are rebuilt with more seeds ( ring 2 in SIMD, then the seeds within 4 sqrt 2, 8 sqrt 2, .. ) and fill a table of
    // the halves of the bad zones ( every cell touching such a zone was rebuilt: it touches the bad parent ); and so on until no
    // zone is bad. Past the list of `ek`, the exact path ( psi_max ). A cell rebuilt but out of the bad zones is exact: the zones
    // it touches are clean, at this round or at an earlier one ( where its larger cell was exact already ).
    void geometry_zones( const std::vector<T> &psi ) {
        ensure_planes();
        const SI nn = SI( N ) * N;
        constexpr int C = ExactKernel<T>::C;
        const T inf = std::numeric_limits<T>::infinity();
        double tz = now_ms();
        auto lap = [&]( int k ) { const double t = now_ms(); t_zone[ k ] += t - tz; tz = t; };

        // ---- ring 1 on every cell, with its areas
        pad( 1, psi, p1 );
        area1.resize( nn );
        par.slices( N, nn * 16, [&]( int a0, int a1, int t ) {
            auto &rb = row_th[ t ];
            rb.resize( size_t( N ) * ( 1 + 2 * K1::D ) );
            T *hess = rb.data() + N;
            signed char *own = reinterpret_cast<signed char *>( hess + N * K1::D );
            for( int a = a0; a < a1; ++a )
                for( int b0 = 0; b0 < N; b0 += B )
                    K1::row_block( N, p1.data(), a, b0, B, area1.data() + size_t( a ) * N, hess, own, nullptr, &pl1 );
        } );
        std::fill( flag1.begin(), flag1.end(), 0 );
        std::fill( flag2.begin(), flag2.end(), 0 );
        slot.assign( nn, -1 );
        nb_polys = 0; n_flag1 = n_exact = n_rounds = 0;
        round_cells.clear(); round_bad.clear();
        lap( 0 );

        // ---- tools
        const T lo = T( -0.5 ), hi = T( N ) - T( 0.5 );
        auto bbox = [&]( const T *px, const T *py, int n, int i, T *bb ) {
            bb[ 0 ] = bb[ 2 ] = inf; bb[ 1 ] = bb[ 3 ] = -inf;
            for( int e = 0; e < n; ++e ) { bb[ 0 ] = std::min( bb[ 0 ], px[ e ] ); bb[ 1 ] = std::max( bb[ 1 ], px[ e ] ); bb[ 2 ] = std::min( bb[ 2 ], py[ e ] ); bb[ 3 ] = std::max( bb[ 3 ], py[ e ] ); }
            bb[ 0 ] += i / N; bb[ 1 ] += i / N; bb[ 2 ] += i % N; bb[ 3 ] += i % N;
        };
        // the rounding of the pieces of a zone ( side s, area az ): each cell loses ~eps |v| s, |v| its reach from the centre of the
        // zone; the rest of the tolerance is what no mass cares about
        auto reach = [&]( const T *bb, T cx, T cy ) { return std::max( std::max( std::abs( bb[ 0 ] - cx ), std::abs( bb[ 1 ] - cx ) ), std::max( std::abs( bb[ 2 ] - cy ), std::abs( bb[ 3 ] - cy ) ) ); };
        auto ztol = [&]( T az, T sr ) { return T( 1e-14 ) * ( az + sr ); };   // ( 1e-12 let a cell 5e-9 too large pass in a zone of long cells )
        // a stencil cell given inverted by the planes ( an edge of negative length ) is empty: its seeds leave it no room
        auto inverted = [&]( const T *px, const T *py, int n, const double *dx, const double *dy ) {
            T sc = 1;
            for( int e = 0; e < n; ++e ) sc = std::max( sc, std::abs( px[ e ] ) + std::abs( py[ e ] ) );
            for( int e = 0; e < n; ++e ) {
                const int f = e + 1 == n ? 0 : e + 1;
                if( T( -dy[ e ] ) * ( px[ f ] - px[ e ] ) + T( dx[ e ] ) * ( py[ f ] - py[ e ] ) < -T( 1e-9 ) * sc ) return true;
            }
            return false;
        };
        auto shoelace = [&]( const T *px, const T *py, int n ) { T s = 0; for( int e = 0; e < n; ++e ) { const int f = e + 1 == n ? 0 : e + 1; s += px[ e ] * py[ f ] - py[ e ] * px[ f ]; } return s / 2; };
        auto take = [&]( int i ) -> Poly & {             // ( the polygon of a rebuilt cell )
            if( slot[ i ] < 0 ) {
                slot[ i ] = nb_polys++;
                if( nb_polys > int( polys.size() ) ) polys.emplace_back();
                flag1[ i ] = 1;
            }
            return polys[ slot[ i ] ];
        };
        auto put = [&]( Poly &p, const T *px, const T *py, const int *pl, int n ) { p.n = n; p.x.assign( px, px + n ); p.y.assign( py, py + n ); p.pl.assign( pl, pl + n ); };
        // THE TABLE of a round: zones of side s ( a grid from the corner of the domain, clipped to it ), those of `zid( u, v ) >= 0`;
        // `get( i, px, py, ar )`: the polygon of cell i ( relative to its seed ) and its area. What goes back: the bad zones, and
        // the cells of `cells` whose box meets one of them
        std::vector<T> bxs( 4 * size_t( nn ) );          // ( the boxes of the cells, absolute; +inf: empty )
        struct Table { std::vector<char> bad; std::vector<int> next; int nb_bad = 0; };
        auto table = [&]( T s, int nzs, const std::vector<int> &cells, int nzones, auto &&zid, auto &&zuv, auto &&get ) {
            Table tb;
            std::vector<std::vector<T>> Et( par.nt );
            auto zr = [&]( int u, int v, T *r ) { r[ 0 ] = lo + u * s; r[ 1 ] = std::min( hi, lo + ( u + 1 ) * s ); r[ 2 ] = lo + v * s; r[ 3 ] = std::min( hi, lo + ( v + 1 ) * s ); };
            auto zof = [&]( T x ) { return std::clamp( int( std::floor( ( x - lo ) / s ) ), 0, nzs - 1 ); };
            par.slices( int( cells.size() ), SI( cells.size() ) * 64, [&]( int q0, int q1, int t ) {
                auto &E = Et[ t ];
                E.assign( 2 * size_t( nzones ), T( 0 ) );
                T px[ C ], py[ C ], bb[ 4 ], r[ 4 ], ar;
                for( int q = q0; q < q1; ++q ) {
                    const int i = cells[ q ], n = get( i, px, py, ar );
                    if( n < 3 ) { bxs[ 4 * size_t( i ) ] = inf; continue; }
                    bbox( px, py, n, i, bb );
                    std::copy( bb, bb + 4, &bxs[ 4 * size_t( i ) ] );
                    const int u0 = zof( bb[ 0 ] ), u1 = zof( bb[ 1 ] ), v0 = zof( bb[ 2 ] ), v1 = zof( bb[ 3 ] );
                    for( int u = u0; u <= u1; ++u )
                        for( int v = v0; v <= v1; ++v ) {
                            const int z = zid( u, v );
                            if( z < 0 ) continue;
                            zr( u, v, r );
                            const T pc = u0 == u1 && v0 == v1 ? ar : rect_area( px, py, n, r[ 0 ] - i / N, r[ 1 ] - i / N, r[ 2 ] - i % N, r[ 3 ] - i % N );
                            E[ 2 * z ] += pc;
                            E[ 2 * z + 1 ] += ( reach( bb, ( r[ 0 ] + r[ 1 ] ) / 2, ( r[ 2 ] + r[ 3 ] ) / 2 ) + s ) * s;
                        }
                }
            } );
            std::vector<T> E( 2 * size_t( nzones ), T( 0 ) );
            for( auto &e : Et ) if( e.size() ) for( size_t z = 0; z < E.size(); ++z ) E[ z ] += e[ z ];
            tb.bad.assign( nzones, 0 );
            for( int z = 0; z < nzones; ++z ) {
                T r[ 4 ]; int u, v; zuv( z, u, v ); zr( u, v, r );
                const T az = ( r[ 1 ] - r[ 0 ] ) * ( r[ 3 ] - r[ 2 ] );
                tb.bad[ z ] = E[ 2 * z ] - az > ztol( az, E[ 2 * z + 1 ] );
                tb.nb_bad += tb.bad[ z ];
            }
            if( ! tb.nb_bad ) return tb;
            std::vector<std::vector<int>> nx( par.nt );
            par.slices( int( cells.size() ), SI( cells.size() ) * 16, [&]( int q0, int q1, int t ) {
                for( int q = q0; q < q1; ++q ) {
                    const int i = cells[ q ];
                    const T *bb = &bxs[ 4 * size_t( i ) ];
                    if( bb[ 0 ] == inf ) continue;
                    bool hit = false;
                    for( int u = zof( bb[ 0 ] ); u <= zof( bb[ 1 ] ) && ! hit; ++u )
                        for( int v = zof( bb[ 2 ] ); v <= zof( bb[ 3 ] ) && ! hit; ++v ) { const int z = zid( u, v ); hit = z >= 0 && tb.bad[ z ]; }
                    if( hit ) nx[ t ].push_back( i );
                }
            } );
            for( auto &x : nx ) tb.next.insert( tb.next.end(), x.begin(), x.end() );
            return tb;
        };

        // ---- the cuts of the previous call past ring 1 ( `use_prev` ): the cells that had some start from their ring-1 polygon
        // cut by them ( a superset of the cell still: the zones check it )
        if( use_prev && prev_off.size() == size_t( nn ) + 1 && prev_off[ nn ] ) {
            std::vector<int> xc;
            for( int i = 0; i < nn; ++i ) if( prev_off[ i + 1 ] > prev_off[ i ] ) xc.push_back( i );
            for( int i : xc ) take( i );
            ek_index( 0, 0 );                            // ( the table, before the threads )
            const int PP = pl1.plane(), nm1 = ek.nb_within( std::sqrt( T( 2 ) ) );
            par.slices( int( xc.size() ), SI( xc.size() ) * 64, [&]( int q0, int q1, int ) {
                T px[ C ], py[ C ], qx[ C ], qy[ C ], sv[ C ]; int pl[ C ], ql[ C ];
                const int S = 2 * ek.RB + 1;
                for( int q = q0; q < q1; ++q ) {
                    const int i = xc[ q ], ii = pl1.row( i / N ) + i % N + 1;
                    int n = K1::D;
                    for( int e = 0; e < K1::D; ++e ) {
                        px[ e ] = pl1.vx[ ( ( e + K1::D - 1 ) % K1::D ) * PP + ii ];
                        py[ e ] = pl1.vy[ ( ( e + K1::D - 1 ) % K1::D ) * PP + ii ];
                        const int o = pl1.ow[ e * PP + ii ];
                        pl[ e ] = o < 0 ? -1 : ek_idx[ size_t( T1::off_x[ o ] + ek.RB ) * S + T1::off_y[ o ] + ek.RB ];
                    }
                    Poly &p = polys[ slot[ i ] ];
                    p.exh = p.fin = p.glob = false; p.nm = nm1; p.R = std::sqrt( T( 2 ) );
                    if( inverted( px, py, K1::D, T1::dir_x, T1::dir_y ) ) { put( p, px, py, pl, 0 ); continue; }
                    for( int r = prev_off[ i ]; r < prev_off[ i + 1 ] && n; ++r ) {
                        const int j = prev_nb[ r ], dx = j / N - i / N, dy = j % N - i % N;
                        if( std::abs( dx ) > ek.RB || std::abs( dy ) > ek.RB ) continue;
                        const int o = ek_idx[ size_t( dx + ek.RB ) * S + dy + ek.RB ];
                        const T fx = T( dx ), fy = T( dy ), cc = ( fx * fx + fy * fy + psi[ i ] - psi[ j ] ) / 2;
                        bool out = false;
                        for( int e = 0; e < n; ++e ) { sv[ e ] = fx * px[ e ] + fy * py[ e ] - cc; out |= sv[ e ] > 0; }
                        if( ! out || n + 1 >= C ) continue;
                        int m = 0;
                        for( int e = 0; e < n; ++e ) {
                            const int f = e + 1 == n ? 0 : e + 1;
                            if( sv[ e ] <= 0 ) {
                                qx[ m ] = px[ e ]; qy[ m ] = py[ e ]; ql[ m++ ] = pl[ e ];
                                if( sv[ f ] > 0 ) { const T t = sv[ e ] / ( sv[ e ] - sv[ f ] ); qx[ m ] = px[ e ] + t * ( px[ f ] - px[ e ] ); qy[ m ] = py[ e ] + t * ( py[ f ] - py[ e ] ); ql[ m++ ] = o; }
                            } else if( sv[ f ] <= 0 ) { const T t = sv[ e ] / ( sv[ e ] - sv[ f ] ); qx[ m ] = px[ e ] + t * ( px[ f ] - px[ e ] ); qy[ m ] = py[ e ] + t * ( py[ f ] - py[ e ] ); ql[ m++ ] = pl[ e ]; }
                        }
                        n = m;
                        for( int e = 0; e < n; ++e ) { px[ e ] = qx[ e ]; py[ e ] = qy[ e ]; pl[ e ] = ql[ e ]; }
                    }
                    put( p, px, py, pl, n );
                }
            } );
        }

        // ---- ring 2 in SIMD on a list of cells, its planes read back as polygons ( offsets of T2 turned into those of `ek` )
        const T R2s = 2 * std::sqrt( T( 2 ) );           // ( the disc of radius 2 sqrt 2 is the square ring 2 )
        auto ring2_cells = [&]( const std::vector<int> &cl2 ) {
            if( cl2.empty() ) return;
            for( int i : cl2 ) take( i );
            pad( 2, psi, p2 );
            area2.resize( nn );
            const int nm2 = ek.nb_within( R2s );
            int to_ek[ T2::nb_offsets ];
            for( int o = 0; o < T2::nb_offsets; ++o )
                for( int q = 0; q < nm2; ++q ) if( ek.ox[ q ] == T2::off_x[ o ] && ek.oy[ q ] == T2::off_y[ o ] ) to_ek[ o ] = q;
            const int nc = int( ( cl2.size() + B - 1 ) / B );
            par.slices( nc, SI( cl2.size() ) * 64, [&]( int c0, int c1, int ) {
                T px[ K2::D ], py[ K2::D ]; int pl[ K2::D ];
                for( int c = c0; c < c1; ++c ) {
                    const int *cl = cl2.data() + size_t( c ) * B, n = int( std::min<size_t>( B, cl2.size() - size_t( c ) * B ) );
                    K2::cells_block( N, p2.data(), cl, n, area2.data(), nullptr, nullptr, &pl2 );
                    const int PQ = pl2.plane();
                    for( int x = 0; x < n; ++x ) {
                        const int i = cl[ x ], ii = pl2.row( i / N ) + i % N + 2;
                        for( int e = 0; e < K2::D; ++e ) {
                            px[ e ] = pl2.vx[ ( ( e + K2::D - 1 ) % K2::D ) * PQ + ii ];
                            py[ e ] = pl2.vy[ ( ( e + K2::D - 1 ) % K2::D ) * PQ + ii ];
                            const int o = pl2.ow[ e * PQ + ii ];
                            pl[ e ] = o < 0 ? -1 : to_ek[ o ];
                        }
                        Poly &p = polys[ slot[ i ] ];
                        p.exh = p.fin = p.glob = false; p.nm = nm2; p.R = R2s;
                        if( inverted( px, py, K2::D, T2::dir_x, T2::dir_y ) ) { put( p, px, py, pl, 0 ); continue; }
                        // ( the vertices of the empty directions repeat: vertex e goes when e + 1 is the same point )
                        int m = 0;
                        for( int e = 0; e < K2::D; ++e ) {
                            const int f = e + 1 == K2::D ? 0 : e + 1;
                            if( px[ f ] == px[ e ] && py[ f ] == py[ e ] && m + ( K2::D - e ) > 3 ) continue;
                            px[ m ] = px[ e ]; py[ m ] = py[ e ]; pl[ m ] = pl[ e ]; ++m;
                        }
                        put( p, px, py, pl, m );
                    }
                }
            } );
        };
        // ---- a guess, not a proof: ring 2 on the cells whose ring-1 faces do not match their neighbours' ( the local
        // certificate ); the tables check everything anyway
        if( zone_predict ) {
            std::vector<std::vector<int>> pt( par.nt );
            par.slices( N, nn * 8, [&]( int a0, int a1, int t ) {
                typename K1::CertTables ct;
                for( int a = a0; a < a1; ++a ) {
                    K1::certify_row( pl1, ct, a, flag2.data(), T( 1e-9 ) );
                    for( int b = 0; b < N; ++b ) if( flag2[ a * N + b ] ) pt[ t ].push_back( a * N + b );
                }
            } );
            std::fill( flag2.begin(), flag2.end(), 0 );
            std::vector<int> pr;
            for( auto &x : pt ) for( int i : x ) if( slot[ i ] < 0 ) pr.push_back( i );   // ( not those cut by the previous cuts )
            ring2_cells( pr );
            round_cells.push_back( int( pr.size() ) );
        } else
            round_cells.push_back( 0 );
        lap( 2 );

        // ---- round 0: the table of ring 1, by blocks of B cells of a row read from the planes ( contiguous ): their boxes, the
        // inverted ones ( empty for good: a polygon of 0 vertices ); a cell inside one zone adds its area, the others their pieces
        T s = T( zone0 );
        int nzs = int( std::ceil( T( N ) / s ) );
        std::vector<char> emp( nn, 0 );
        Table tb;
        {
            const int PP = pl1.plane(), nz2 = nzs * nzs;
            std::vector<std::vector<T>> Et( par.nt );
            auto zof = [&]( T x ) { return std::clamp( int( std::floor( ( x - lo ) / s ) ), 0, nzs - 1 ); };
            par.slices( N, nn * 16, [&]( int a0, int a1, int t ) {
                auto &E = Et[ t ];
                E.assign( 2 * size_t( nz2 ), T( 0 ) );
                for( int a = a0; a < a1; ++a )
                    for( int b0 = 0; b0 < N; b0 += B ) {
                        const int i0 = pl1.row( a ) + b0 + 1, n = std::min( B, N - b0 );
                        alignas( 64 ) T x0[ B ], x1[ B ], y0[ B ], y1[ B ], sc[ B ];
                        alignas( 64 ) unsigned char inv[ B ];
                        #pragma omp simd
                        for( int x = 0; x < B; ++x ) { x0[ x ] = y0[ x ] = inf; x1[ x ] = y1[ x ] = -inf; sc[ x ] = 1; inv[ x ] = 0; }
                        for( int k = 0; k < K1::D; ++k ) {
                            const T *vx = pl1.vx.data() + k * PP + i0, *vy = pl1.vy.data() + k * PP + i0;
                            #pragma omp simd
                            for( int x = 0; x < B; ++x ) {
                                x0[ x ] = vx[ x ] < x0[ x ] ? vx[ x ] : x0[ x ]; x1[ x ] = vx[ x ] > x1[ x ] ? vx[ x ] : x1[ x ];   // ( not fmin / fmax: libm calls with gcc )
                                y0[ x ] = vy[ x ] < y0[ x ] ? vy[ x ] : y0[ x ]; y1[ x ] = vy[ x ] > y1[ x ] ? vy[ x ] : y1[ x ];
                                const T a1 = std::abs( vx[ x ] ) + std::abs( vy[ x ] );
                                sc[ x ] = a1 > sc[ x ] ? a1 : sc[ x ];
                            }
                        }
                        // edge k from plane vertex k - 1 to plane vertex k, along ( -dir_y, dir_x )
                        for( int k = 0; k < K1::D; ++k ) {
                            const int km = ( k + K1::D - 1 ) % K1::D;
                            const T *vx = pl1.vx.data() + k * PP + i0, *vy = pl1.vy.data() + k * PP + i0, *wx = pl1.vx.data() + km * PP + i0, *wy = pl1.vy.data() + km * PP + i0;
                            const T tx = -T( T1::dir_y[ k ] ), ty = T( T1::dir_x[ k ] );
                            #pragma omp simd
                            for( int x = 0; x < B; ++x ) inv[ x ] |= tx * ( vx[ x ] - wx[ x ] ) + ty * ( vy[ x ] - wy[ x ] ) < -T( 1e-9 ) * sc[ x ];
                        }
                        for( int x = 0; x < n; ++x ) {
                            const int b = b0 + x, i = a * N + b;
                            T *bb = &bxs[ 4 * size_t( i ) ];
                            if( slot[ i ] >= 0 ) {               // ( rebuilt in ring 2 )
                                const Poly &p = polys[ slot[ i ] ];
                                if( p.n < 3 ) { bb[ 0 ] = inf; continue; }
                                bbox( p.x.data(), p.y.data(), p.n, i, bb );
                                const int u0 = zof( bb[ 0 ] ), u1 = zof( bb[ 1 ] ), v0 = zof( bb[ 2 ] ), v1 = zof( bb[ 3 ] );
                                for( int u = u0; u <= u1; ++u )
                                    for( int v = v0; v <= v1; ++v ) {
                                        const T r0 = lo + u * s, r1 = std::min( hi, lo + ( u + 1 ) * s ), r2 = lo + v * s, r3 = std::min( hi, lo + ( v + 1 ) * s );
                                        const int z = u * nzs + v;
                                        E[ 2 * z ] += u0 == u1 && v0 == v1 ? shoelace( p.x.data(), p.y.data(), p.n ) : rect_area( p.x.data(), p.y.data(), p.n, r0 - a, r1 - a, r2 - b, r3 - b );
                                        E[ 2 * z + 1 ] += ( reach( bb, ( r0 + r1 ) / 2, ( r2 + r3 ) / 2 ) + s ) * s;
                                    }
                                continue;
                            }
                            bb[ 0 ] = x0[ x ] + a; bb[ 1 ] = x1[ x ] + a; bb[ 2 ] = y0[ x ] + b; bb[ 3 ] = y1[ x ] + b;
                            if( inv[ x ] ) { emp[ i ] = 1; continue; }
                            const int u0 = zof( bb[ 0 ] ), u1 = zof( bb[ 1 ] ), v0 = zof( bb[ 2 ] ), v1 = zof( bb[ 3 ] );
                            if( u0 == u1 && v0 == v1 ) {
                                const int z = u0 * nzs + v0;
                                E[ 2 * z ] += area1[ i ];
                                E[ 2 * z + 1 ] += ( reach( bb, lo + ( u0 + T( 0.5 ) ) * s, lo + ( v0 + T( 0.5 ) ) * s ) + s ) * s;
                                continue;
                            }
                            T px[ K1::D ], py[ K1::D ];
                            for( int e = 0; e < K1::D; ++e ) { px[ e ] = pl1.vx[ ( ( e + K1::D - 1 ) % K1::D ) * PP + i0 + x ]; py[ e ] = pl1.vy[ ( ( e + K1::D - 1 ) % K1::D ) * PP + i0 + x ]; }
                            for( int u = u0; u <= u1; ++u )
                                for( int v = v0; v <= v1; ++v ) {
                                    const T r0 = lo + u * s, r1 = std::min( hi, lo + ( u + 1 ) * s ), r2 = lo + v * s, r3 = std::min( hi, lo + ( v + 1 ) * s );
                                    const int z = u * nzs + v;
                                    E[ 2 * z ] += rect_area( px, py, K1::D, r0 - a, r1 - a, r2 - b, r3 - b );
                                    E[ 2 * z + 1 ] += ( reach( bb, ( r0 + r1 ) / 2, ( r2 + r3 ) / 2 ) + s ) * s;
                                }
                        }
                    }
            } );
            std::vector<T> E( 2 * size_t( nz2 ), T( 0 ) );
            for( auto &e : Et ) if( e.size() ) for( size_t z = 0; z < E.size(); ++z ) E[ z ] += e[ z ];
            tb.bad.assign( nz2, 0 );
            // ( a prefix sum of the bad zones: does a box meet one, in O( 1 ) )
            std::vector<int> ps( size_t( nzs + 1 ) * ( nzs + 1 ), 0 );
            for( int u = 0; u < nzs; ++u )
                for( int v = 0; v < nzs; ++v ) {
                    const int z = u * nzs + v;
                    const T r0 = lo + u * s, r1 = std::min( hi, lo + ( u + 1 ) * s ), r2 = lo + v * s, r3 = std::min( hi, lo + ( v + 1 ) * s ), az = ( r1 - r0 ) * ( r3 - r2 );
                    tb.bad[ z ] = E[ 2 * z ] - az > ztol( az, E[ 2 * z + 1 ] );
                    tb.nb_bad += tb.bad[ z ];
                    ps[ size_t( u + 1 ) * ( nzs + 1 ) + v + 1 ] = tb.bad[ z ] + ps[ size_t( u ) * ( nzs + 1 ) + v + 1 ] + ps[ size_t( u + 1 ) * ( nzs + 1 ) + v ] - ps[ size_t( u ) * ( nzs + 1 ) + v ];
                }
            for( int i = 0; i < nn; ++i ) if( emp[ i ] ) { take( i ).n = 0; polys[ slot[ i ] ].exh = false; }
            if( tb.nb_bad ) {
                std::vector<std::vector<int>> nx( par.nt );
                par.slices( N, nn * 4, [&]( int a0, int a1, int t ) {
                    for( int i = a0 * N; i < a1 * N; ++i ) {
                        if( emp[ i ] ) continue;
                        const T *bb = &bxs[ 4 * size_t( i ) ];
                        if( bb[ 0 ] == inf ) continue;
                        const int u0 = zof( bb[ 0 ] ), u1 = zof( bb[ 1 ] ) + 1, v0 = zof( bb[ 2 ] ), v1 = zof( bb[ 3 ] ) + 1;
                        if( ps[ size_t( u1 ) * ( nzs + 1 ) + v1 ] - ps[ size_t( u0 ) * ( nzs + 1 ) + v1 ] - ps[ size_t( u1 ) * ( nzs + 1 ) + v0 ] + ps[ size_t( u0 ) * ( nzs + 1 ) + v0 ] ) nx[ t ].push_back( i );
                    }
                } );
                for( auto &x : nx ) tb.next.insert( tb.next.end(), x.begin(), x.end() );
            }
        }
        std::vector<int> cells, zgrid;
        round_bad.push_back( tb.nb_bad );
        lap( 1 );

        // ---- the next rounds
        std::vector<std::pair<int,int>> bad_uv;          // the bad zones of the last table, at side s
        for( int z = 0; z < int( tb.bad.size() ); ++z ) if( tb.bad[ z ] ) bad_uv.push_back( { z / nzs, z % nzs } );
        bool padded = false;
        if( rebuild_tree && tb.nb_bad ) {
            // every cell of the bad zones to its exact cell, by the mipmap: the bad zones then hold exact cells only, no more table
            build_mip( psi );
            const std::vector<int> &cl = tb.next;
            std::vector<char> had( cl.size() );
            for( size_t q = 0; q < cl.size(); ++q ) { had[ q ] = slot[ cl[ q ] ] >= 0; take( cl[ q ] ); }
            const int PP = pl1.plane();
            par.slices( int( cl.size() ), SI( cl.size() ) * 1024, [&]( int q0, int q1, int ) {
                T px[ C ], py[ C ]; int nb[ C ], n;
                for( int q = q0; q < q1; ++q ) {
                    const int i = cl[ q ];
                    Poly &p = polys[ slot[ i ] ];
                    if( had[ q ] ) {
                        n = p.n; std::copy( p.x.begin(), p.x.end(), px ); std::copy( p.y.begin(), p.y.end(), py );
                        for( int e = 0; e < n; ++e ) nb[ e ] = p.glob || p.pl[ e ] < 0 ? p.pl[ e ] : i + ek.ox[ p.pl[ e ] ] * N + ek.oy[ p.pl[ e ] ];
                    } else {
                        const int ii = pl1.row( i / N ) + i % N + 1;
                        n = K1::D;
                        for( int e = 0; e < K1::D; ++e ) {
                            px[ e ] = pl1.vx[ ( ( e + K1::D - 1 ) % K1::D ) * PP + ii ];
                            py[ e ] = pl1.vy[ ( ( e + K1::D - 1 ) % K1::D ) * PP + ii ];
                            const int o = pl1.ow[ e * PP + ii ];
                            nb[ e ] = o < 0 ? -1 : i + T1::off_x[ o ] * N + T1::off_y[ o ];
                        }
                    }
                    p.exh = n && ! tree_clip( i, psi, px, py, nb, n, 1 );   // ( every seed of the 3 x 3 stencil cut it already )
                    put( p, px, py, nb, n );
                    p.glob = true; p.fin = true;
                }
            } );
            n_rounds = 1;
            round_cells.push_back( int( cl.size() ) ); round_bad.push_back( 0 );
            lap( 2 );
            tb.nb_bad = 0;
        }
        for( int k = 1; tb.nb_bad; ++k ) {
            // the cells of the bad zones: ring 2 for those still in ring 1, the others from their polygon with the seeds within twice
            // their radius ( past the list of `ek`: the exact path, once )
            std::vector<int> fresh, more;
            for( int i : tb.next ) {
                if( slot[ i ] < 0 ) fresh.push_back( i );
                else if( ! polys[ slot[ i ] ].fin ) more.push_back( i );
            }
            if( fresh.empty() && more.empty() ) break;
            ring2_cells( fresh );
            if( more.size() && ! padded ) { pad( ek.RB, psi, pe ); pmax = *std::max_element( psi.begin(), psi.end() ); if( use_tree ) build_mip( psi ); padded = true; }
            par.slices( int( more.size() ), SI( more.size() ) * 4096, [&]( int q0, int q1, int ) {
                T px[ C ], py[ C ]; int pl[ C ], n;
                for( int q = q0; q < q1; ++q ) {
                    const int i = more[ q ];
                    Poly &p = polys[ slot[ i ] ];
                    if( use_tree && ! p.glob ) {             // ( from the ring-2 polygon, its neighbours made global )
                        n = p.n; std::copy( p.x.begin(), p.x.end(), px ); std::copy( p.y.begin(), p.y.end(), py );
                        int nb[ C ];
                        for( int e = 0; e < n; ++e ) nb[ e ] = p.pl[ e ] < 0 ? -1 : i + ek.ox[ p.pl[ e ] ] * N + ek.oy[ p.pl[ e ] ];
                        const int skip = p.R <= R2s * T( 1.0001 ) ? 2 : -1;
                        if( tree_clip( i, psi, px, py, nb, n, skip ) ) {
                            put( p, px, py, nb, n );
                            p.glob = true; p.fin = true; p.exh = false;
                            continue;
                        }
                    }
                    const T R = 2 * p.R;
                    const bool last = R > T( ek.RB );
                    const int nm = last ? -1 : ek.nb_within( R );
                    n = p.n; std::copy( p.x.begin(), p.x.end(), px ); std::copy( p.y.begin(), p.y.end(), py ); std::copy( p.pl.begin(), p.pl.end(), pl );
                    if( ! last && use_reg && reg_clip( i, px, py, pl, n, p.nm, nm ) ) p.exh = false;
                    else {
                        n = p.n; std::copy( p.x.begin(), p.x.end(), px ); std::copy( p.y.begin(), p.y.end(), py ); std::copy( p.pl.begin(), p.pl.end(), pl );
                        p.exh = ! ek.clip_range( N, pe.data(), last ? pmax : inf, i / N, i % N, px, py, pl, n, p.nm, nm );
                    }
                    put( p, px, py, pl, n );
                    p.nm = last ? int( ek.ox.size() ) : nm; p.R = R; p.fin = last;
                }
            } );
            cells.clear();
            cells.insert( cells.end(), fresh.begin(), fresh.end() );
            cells.insert( cells.end(), more.begin(), more.end() );
            n_rounds = k;
            lap( 2 );
            // the table of the halves of the bad zones ( or of the bad zones, at `zone_min` )
            const bool split = s / 2 >= zone_min;
            std::vector<std::pair<int,int>> zs;
            for( auto [ u, v ] : bad_uv )
                for( int cu = 0; cu <= int( split ); ++cu )
                    for( int cv = 0; cv <= int( split ); ++cv ) {
                        const int uu = split ? 2 * u + cu : u, vv = split ? 2 * v + cv : v;
                        if( lo + uu * ( split ? s / 2 : s ) >= hi || lo + vv * ( split ? s / 2 : s ) >= hi ) continue;
                        zs.push_back( { uu, vv } );
                    }
            if( split ) { s /= 2; nzs = int( std::ceil( T( N ) / s ) ); }
            zgrid.assign( size_t( nzs ) * nzs, -1 );     // ( the zones of the table, by their place in the grid of side s )
            for( int z = 0; z < int( zs.size() ); ++z ) zgrid[ size_t( zs[ z ].first ) * nzs + zs[ z ].second ] = z;
            auto rebuilt = [&]( int i, T *px, T *py, T &ar ) {
                const Poly &p = polys[ slot[ i ] ];
                std::copy( p.x.begin(), p.x.end(), px ); std::copy( p.y.begin(), p.y.end(), py );
                ar = p.n >= 3 ? shoelace( px, py, p.n ) : T( 0 );
                return p.n;
            };
            tb = table( s, nzs, cells, int( zs.size() ),
                        [&]( int u, int v ) { return zgrid[ size_t( u ) * nzs + v ]; },
                        [&]( int z, int &u, int &v ) { u = zs[ z ].first; v = zs[ z ].second; }, rebuilt );
            round_cells.push_back( int( cells.size() ) ); round_bad.push_back( tb.nb_bad );
            bad_uv.clear();
            for( int z = 0; z < int( zs.size() ); ++z ) if( tb.bad[ z ] ) bad_uv.push_back( zs[ z ] );
            lap( 3 );
        }
        n_flag1 = nb_polys;
        for( int q = 0; q < nb_polys; ++q ) n_exact += polys[ q ].fin;

        // the cuts past ring 1, for the next call ( the cells of ring 1 have none )
        if( use_prev ) {
            prev_off.assign( size_t( nn ) + 1, 0 );
            std::vector<std::pair<int,int>> pj;
            for( int i = 0; i < nn; ++i ) {
                if( slot[ i ] < 0 ) continue;
                const Poly &p = polys[ slot[ i ] ];
                for( int e = 0; e < p.n; ++e ) {
                    if( p.pl[ e ] < 0 ) continue;
                    const int j = p.glob ? p.pl[ e ] : i + ek.ox[ p.pl[ e ] ] * N + ek.oy[ p.pl[ e ] ];
                    if( std::abs( j / N - i / N ) > 1 || std::abs( j % N - i % N ) > 1 ) pj.push_back( { i, j } );
                }
            }
            std::sort( pj.begin(), pj.end() );
            pj.erase( std::unique( pj.begin(), pj.end() ), pj.end() );
            prev_nb.resize( pj.size() );
            for( size_t r = 0; r < pj.size(); ++r ) { ++prev_off[ pj[ r ].first + 1 ]; prev_nb[ r ] = pj[ r ].second; }
            for( int i = 0; i < nn; ++i ) prev_off[ i + 1 ] += prev_off[ i ];
        }
    }

    void geometry_cert( const std::vector<T> &psi ) {
        ensure_planes();
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

    /// THE CELL OF SEED i AT LEVEL lv ( `run_zones` ), relative to its seed, edge e toward `nbr[ e ]` ( -1: the box ): 0 or 1 ring 1
    /// ( `p1` ), 2.. the seeds within sqrt( 2 ) 2^( lv - 1 ), 255 the exact clipping with psi_max ( `pe`, `pmax` )
    int level_polygon( int i, int lv, T *px, T *py, int *nbr, bool &exh ) const {
        int pl[ ExactKernel<T>::C ], n;
        const int a = i / N, b = i % N;
        if( lv <= 1 ) {
            ek1.clip( N, p1.data(), std::numeric_limits<T>::infinity(), a, b, px, py, pl, n, 8 );
            for( int e = 0; e < n; ++e ) nbr[ e ] = pl[ e ] < 0 ? -1 : i + ek1.ox[ pl[ e ] ] * N + ek1.oy[ pl[ e ] ];
            return n;
        }
        const bool last = lv == 255;
        const int nm = last ? -1 : ek.nb_within( std::sqrt( T( 2 ) ) * T( 1 << std::min( lv - 1, 20 ) ) );
        // ( from the ring-1 cell: its 8 cuts are the first 8 offsets of `ek` too, both sorted by distance the same way )
        ek1.clip( N, p1.data(), std::numeric_limits<T>::infinity(), a, b, px, py, pl, n, 8 );
        if( ! ek.clip_range( N, pe.data(), last ? pmax : std::numeric_limits<T>::infinity(), a, b, px, py, pl, n, 8, nm ) ) exh = true;
        for( int e = 0; e < n; ++e ) nbr[ e ] = pl[ e ] < 0 ? -1 : i + ek.ox[ pl[ e ] ] * N + ek.oy[ pl[ e ] ];
        return n;
    }
    int next_level( int lv ) const {
        if( lv == 255 ) return 255;
        const int nl = std::max( lv, 1 ) + 1;
        return std::sqrt( T( 2 ) ) * T( 1 << std::min( nl - 1, 20 ) ) > T( ek.RB ) ? 255 : nl;
    }

    /// THE DIAGRAM OF `psi` BY THE ZONE CERTIFICATE, FUSED ( `gpu/GridNewton2D.cuh::run_zones` ): each block of B cells of a row built
    /// in ring 1 ( a rolling buffer of planes per thread ), integrated in SIMD and added to the first table at once; then the cells of
    /// the bad zones rebuilt a level further ( `level_polygon` ), integrated and added to the table of the halves of the bad zones,
    /// until no zone is bad. What a cell keeps: its level and its box.
    void run_zones( const std::vector<T> &psi ) {
        const double t0 = now_ms();
        const SI nn = SI( N ) * N;
        constexpr int C = ExactKernel<T>::C;
        const T inf = std::numeric_limits<T>::infinity(), lo = T( -0.5 ), hi = T( N ) - T( 0.5 );
        pad( 1, psi, p1 );
        if( zone_predict ) pad( 2, psi, p2 );
        lvl.assign( nn, 0 );
        fused_last = true;
        zbx.resize( 4 * size_t( nn ) );
        n_flag1 = n_exact = n_rounds = 0;
        round_cells.clear(); round_bad.clear();
        for( auto &e : ex_th ) e.clear();
        double tz = now_ms();
        auto lap = [&]( int k ) { const double t = now_ms(); t_zone[ k ] += t - tz; tz = t; };
        const T sn = T( 1e-13 );                         // ( a cell past a zone line by rounding only is in one zone, see the card's )
        auto ztol = [&]( T az, T sr ) { return T( 1e-14 ) * ( az + sr ); };
        auto put_box = [&]( int i, const T *bb ) {
            float *z = &zbx[ 4 * size_t( i ) ];
            z[ 0 ] = std::nextafter( float( bb[ 0 ] + sn ), -INFINITY ); z[ 1 ] = std::nextafter( float( bb[ 1 ] - sn ), INFINITY );
            z[ 2 ] = std::nextafter( float( bb[ 2 ] + sn ), -INFINITY ); z[ 3 ] = std::nextafter( float( bb[ 3 ] - sn ), INFINITY );
        };
        auto empty_box = [&]( int i ) { zbx[ 4 * size_t( i ) ] = 1; zbx[ 4 * size_t( i ) + 1 ] = 0; };
        // the pieces of a polygon cell ( absolute box bb ) in the zones of side s that `in( u, v )` counts
        auto pieces = [&]( int i, const T *px, const T *py, int n, T ar, const T *bb, T s, int nzs, auto &&in, std::vector<T> &E ) {
            const T pa = std::abs( psi[ i ] );           // ( the planes' rounding: edges moved by ~ eps |psi|, see the card's `cell_pieces` )
            auto zof = [&]( T x ) { return std::clamp( int( std::floor( ( x - lo ) / s ) ), 0, nzs - 1 ); };
            const int a = i / N, b = i % N;
            const int u0 = zof( bb[ 0 ] + sn ), u1 = zof( bb[ 1 ] - sn ), v0 = zof( bb[ 2 ] + sn ), v1 = zof( bb[ 3 ] - sn );
            for( int u = u0; u <= u1; ++u )
                for( int v = v0; v <= v1; ++v ) {
                    if( ! in( u, v ) ) continue;
                    const T r0 = lo + u * s, r1 = std::min( hi, lo + ( u + 1 ) * s ), r2 = lo + v * s, r3 = std::min( hi, lo + ( v + 1 ) * s );
                    const size_t z = size_t( u ) * nzs + v;
                    E[ 2 * z ] += u0 == u1 && v0 == v1 ? ar : rect_area( px, py, n, r0 - a, r1 - a, r2 - b, r3 - b );
                    const T cx = ( r0 + r1 ) / 2, cy = ( r2 + r3 ) / 2;
                    E[ 2 * z + 1 ] += ( std::max( std::max( std::abs( bb[ 0 ] - cx ), std::abs( bb[ 1 ] - cx ) ), std::max( std::abs( bb[ 2 ] - cy ), std::abs( bb[ 3 ] - cy ) ) ) + pa + s ) * s;
                }
        };
        // a polygon cell integrated ( its faces out of ring 1 tagged with its level ) and added to a table
        auto integrate_poly = [&]( int t, int i, const T *px, const T *py, const int *nbr, int n, int lv ) {
            T face[ C ];
            clear_faces( i );
            if( n < 3 ) { mass[ i ] = 0; return; }
            integ[ t ].polygon( n, px, py, T( i / N ), T( i % N ), mass[ i ], face );
            for( int e = 0; e < n; ++e ) {
                const int j = nbr[ e ];
                if( j < 0 || face[ e ] == 0 ) continue;
                const T w = face[ e ] / ( 2 * std::hypot( T( j / N - i / N ), T( j % N - i % N ) ) );
                const int k = dir1( j / N - i / N, j % N - i % N );
                if( k >= 0 ) W[ k * P1 * P1 + ( i / N + 1 ) * P1 + i % N + 1 ] = w;
                else ex_th[ t ].push_back( { i, j, w, ( unsigned char ) lv } );
            }
        };
        auto box_of = [&]( int i, const T *px, const T *py, int n, T *bb, T &ar ) {
            bb[ 0 ] = bb[ 2 ] = inf; bb[ 1 ] = bb[ 3 ] = -inf; ar = 0;
            for( int e = 0; e < n; ++e ) {
                const int f = e + 1 == n ? 0 : e + 1;
                bb[ 0 ] = std::min( bb[ 0 ], px[ e ] ); bb[ 1 ] = std::max( bb[ 1 ], px[ e ] ); bb[ 2 ] = std::min( bb[ 2 ], py[ e ] ); bb[ 3 ] = std::max( bb[ 3 ], py[ e ] );
                ar += px[ e ] * py[ f ] - py[ e ] * px[ f ];
            }
            ar /= 2;
            bb[ 0 ] += i / N; bb[ 1 ] += i / N; bb[ 2 ] += i % N; bb[ 3 ] += i % N;
        };

        // the predicted cells, B at a time: ring 2 in SIMD ( `K2::cells_verts`, nothing stored ), integrated in SIMD, into a table
        auto ring2_block = [&]( int t, const int *cl, int n, T s, int nzs, std::vector<T> &E ) {
            constexpr int D2 = K2::D;
            alignas( 64 ) T vx[ D2 ][ B ], vy[ D2 ][ B ], face[ D2 ][ B ], ms[ B ], ar[ B ], sa[ B ], sb[ B ];
            alignas( 64 ) signed char own[ D2 ][ B ];
            alignas( 64 ) unsigned char bad[ B ] = {};
            K2::cells_verts( N, p2.data(), cl, n, vx, vy, own, ar );
            for( int x = 0; x < B; ++x ) { const int i = cl[ x < n ? x : n - 1 ]; sa[ x ] = i / N; sb[ x ] = i % N; }
            if constexpr ( Integ::has_simd ) integ[ t ].template cells<D2,B>( vx, vy, sa, sb, ms, face, bad );
            else for( int x = 0; x < B; ++x ) bad[ x ] = 1;
            auto all = []( int, int ) { return true; };
            for( int x = 0; x < n; ++x ) {
                const int i = cl[ x ];
                lvl[ i ] = 2;
                // the polygon ( vertex e = vertex e - 1 of the kernel, edge e toward the offset own[ e ] ), inverted: empty
                T px[ D2 ], py[ D2 ], sc = 1; int nbr[ D2 ];
                for( int e = 0; e < D2; ++e ) {
                    px[ e ] = vx[ ( e + D2 - 1 ) % D2 ][ x ]; py[ e ] = vy[ ( e + D2 - 1 ) % D2 ][ x ];
                    sc = std::max( sc, std::abs( px[ e ] ) + std::abs( py[ e ] ) );
                    const int o = own[ e ][ x ];
                    nbr[ e ] = o < 0 ? -1 : i + T2::off_x[ o ] * N + T2::off_y[ o ];
                }
                bool inv = false;
                for( int e = 0; e < D2; ++e ) { const int f = e + 1 == D2 ? 0 : e + 1; inv |= -T( T2::dir_y[ e ] ) * ( px[ f ] - px[ e ] ) + T( T2::dir_x[ e ] ) * ( py[ f ] - py[ e ] ) < -T( 1e-9 ) * sc; }
                if( inv ) { mass[ i ] = 0; clear_faces( i ); empty_box( i ); continue; }
                if( bad[ x ] ) integrate_poly( t, i, px, py, nbr, D2, 2 );
                else {
                    mass[ i ] = ms[ x ];
                    clear_faces( i );
                    for( int e = 0; e < D2; ++e ) {
                        const int o = own[ e ][ x ];
                        if( o < 0 || face[ e ][ x ] == 0 ) continue;
                        const T w = face[ e ][ x ] * T( 0.5 ) * T( T2::dir_inv_norm[ e ] ) / T( T2::off_m[ o ] );
                        const int k = dir1( T2::off_x[ o ], T2::off_y[ o ] );
                        if( k >= 0 ) W[ k * P1 * P1 + ( i / N + 1 ) * P1 + i % N + 1 ] = w;
                        else ex_th[ t ].push_back( { i, nbr[ e ], w, ( unsigned char ) 2 } );
                    }
                }
                T bb[ 4 ], a2;
                box_of( i, px, py, D2, bb, a2 );
                put_box( i, bb );
                pieces( i, px, py, D2, ar[ x ], bb, s, nzs, all, E );
            }
        };

        // ---- ring 1, every cell: built, integrated, into the first table
        T s = T( zone0 );
        int nzs = int( std::ceil( T( N ) / s ) );
        std::vector<std::vector<T>> Et( par.nt );
        std::vector<std::vector<int>> fine_l( par.nt );
        std::vector<int> n_pred( par.nt, 0 );

        par.slices( N, nn * 64, [&]( int a0, int a1, int t ) {
            auto &E = Et[ t ];
            E.assign( 2 * size_t( nzs ) * nzs, T( 0 ) );
            auto &fl = fine_l[ t ];
            fl.clear();
            alignas( 64 ) int pr[ B ];                   // ( the predicted cells waiting for a block of ring 2 )
            int npr = 0;
            n_pred[ t ] = 0;
            typename K1::Planes rp( N, true, false );    // ( the rows of this thread only, in a ring )
            rp.fill( 0, rp.slots );
            const int PP = rp.plane();
            auto &rb = row_th[ t ];
            rb.resize( size_t( N ) * ( 1 + 2 * K1::D ) );
            T *area = rb.data(), *hess = area + N;
            signed char *own = reinterpret_cast<signed char *>( hess + N * K1::D );
            auto all = []( int, int ) { return true; };
            for( int a = a0; a < a1; ++a )
                for( int b0 = 0; b0 < N; b0 += B ) {
                    const int n = std::min( B, N - b0 );
                    K1::row_block( N, p1.data(), a, b0, B, area, hess, own, nullptr, &rp );
                    const int i0 = rp.row( a ) + b0 + 1;
                    alignas( 64 ) T vx[ 8 ][ B ], vy[ 8 ][ B ], face[ 8 ][ B ], ms[ B ], sa[ B ], sb[ B ];
                    alignas( 64 ) unsigned char bad[ B ] = {};
                    for( int k = 0; k < 8; ++k )
                        for( int x = 0; x < B; ++x ) { vx[ k ][ x ] = rp.vx[ k * PP + i0 + x ]; vy[ k ][ x ] = rp.vy[ k * PP + i0 + x ]; }
                    for( int x = 0; x < B; ++x ) { sa[ x ] = a; sb[ x ] = b0 + x; }
                    if constexpr ( Integ::has_simd ) integ[ t ].template cells<8,B>( vx, vy, sa, sb, ms, face, bad );
                    else for( int x = 0; x < B; ++x ) bad[ x ] = 1;
                    // THE PREDICTION, in SIMD over the block: a seed of ring 2 cuts the cell iff 2 h( k ) > |k|^2 + psi_i - psi_j, h( k ) =
                    // max_v k . v its support ( `p2`: psi padded by -inf, a seed out of the grid cuts nothing ); level 2 at once ( below )
                    alignas( 64 ) unsigned char cut2[ B ] = {};
                    if( zone_predict ) {
                        const int P2 = N + 4;
                        const T *pi = p2.data() + size_t( a + 2 ) * P2 + b0 + 2;
                        for( int dx = -2; dx <= 2; ++dx )
                            for( int dy = -2; dy <= 2; ++dy ) {
                                if( dx > -2 && dx < 2 && dy > -2 && dy < 2 ) continue;
                                const T *pj = pi + dx * P2 + dy, k2 = T( dx * dx + dy * dy );
                                #pragma omp simd
                                for( int x = 0; x < B; ++x ) {
                                    T h = dx * vx[ 0 ][ x ] + dy * vy[ 0 ][ x ];
                                    for( int k = 1; k < 8; ++k ) { const T v = dx * vx[ k ][ x ] + dy * vy[ k ][ x ]; h = v > h ? v : h; }   // ( not fmax: gcc )
                                    cut2[ x ] |= 2 * h > k2 + pi[ x ] - pj[ x ];
                                }
                            }
                    }
                    for( int x = 0; x < n; ++x ) {
                        const int i = a * N + b0 + x;
                        // the polygon ( vertex e = plane vertex e - 1 ), inverted: empty
                        T px[ 8 ], py[ 8 ], sc = 1;
                        for( int e = 0; e < 8; ++e ) { px[ e ] = vx[ ( e + 7 ) & 7 ][ x ]; py[ e ] = vy[ ( e + 7 ) & 7 ][ x ]; sc = std::max( sc, std::abs( px[ e ] ) + std::abs( py[ e ] ) ); }
                        bool inv = false;
                        for( int e = 0; e < 8; ++e ) { const int f = ( e + 1 ) & 7; inv |= -T( T1::dir_y[ e ] ) * ( px[ f ] - px[ e ] ) + T( T1::dir_x[ e ] ) * ( py[ f ] - py[ e ] ) < -T( 1e-9 ) * sc; }
                        // THE PREDICTION: a seed of ring 2 cuts the cell iff 2 h( k ) > |k|^2 + psi_i - psi_j, h( k ) = max_v k . v ( its
                        // support ): level 2 at once, into this table ( below )
                        if( ! inv && cut2[ x ] ) {
                            pr[ npr++ ] = i; ++n_pred[ t ];
                            if( npr == B ) { ring2_block( t, pr, B, s, nzs, E ); npr = 0; }
                            continue;
                        }
                        if( inv ) { mass[ i ] = 0; clear_faces( i ); empty_box( i ); continue; }
                        if( bad[ x ] ) fl.push_back( i );
                        else {
                            mass[ i ] = ms[ x ];
                            const signed char *ow = rp.ow.data() + i0 + x;
                            for( int k = 0; k < 8; ++k )
                                W[ k * P1 * P1 + ( a + 1 ) * P1 + b0 + x + 1 ] = ow[ k * PP ] >= 0 ? face[ k ][ x ] * T( 0.5 ) * T( T1::dir_inv_norm[ k ] ) : T( 0 );
                        }
                        T bb[ 4 ], ar;
                        box_of( i, px, py, 8, bb, ar );
                        put_box( i, bb );
                        pieces( i, px, py, 8, area[ b0 + x ], bb, s, nzs, all, E );
                    }
                }
            if( npr ) ring2_block( t, pr, npr, s, nzs, E );
        } );
        // ( the cells the SIMD rules could not take: the scalar rule on their ring-1 polygon; psi padded for the exact kernel once
        // there are cells past ring 1, which `level_polygon` rebuilds later on )
        bool padded = false;
        for( int c : n_pred ) if( c && ! padded ) { pad( ek.RB, psi, pe ); pmax = *std::max_element( psi.begin(), psi.end() ); padded = true; }
        par.slices( par.nt, nn, [&]( int t0_, int t1_, int ) {
            for( int t = t0_; t < t1_; ++t )
                for( int i : fine_l[ t ] ) {
                    T px[ C ], py[ C ]; int nbr[ C ]; bool e = false;
                    const int n = level_polygon( i, 1, px, py, nbr, e );
                    integrate_poly( t, i, px, py, nbr, n, 1 );
                }
        } );
        std::vector<T> E( 2 * size_t( nzs ) * nzs, T( 0 ) );
        for( auto &e : Et ) if( e.size() ) for( size_t z = 0; z < E.size(); ++z ) E[ z ] += e[ z ];
        std::vector<char> bad( size_t( nzs ) * nzs, 0 ), prev;
        int nbad = 0;
        auto judge = [&]( T s, int nzs, auto &&in ) {
            nbad = 0;
            bad.assign( size_t( nzs ) * nzs, 0 );
            for( int u = 0; u < nzs; ++u )
                for( int v = 0; v < nzs; ++v ) {
                    if( ! in( u, v ) ) continue;
                    const size_t z = size_t( u ) * nzs + v;
                    const T r0 = lo + u * s, r1 = std::min( hi, lo + ( u + 1 ) * s ), r2 = lo + v * s, r3 = std::min( hi, lo + ( v + 1 ) * s ), az = ( r1 - r0 ) * ( r3 - r2 );
                    bad[ z ] = E[ 2 * z ] - az > ztol( az, E[ 2 * z + 1 ] );
                    nbad += bad[ z ];
                }
        };
        judge( s, nzs, []( int, int ) { return true; } );
        round_bad.push_back( nbad ); round_cells.push_back( 0 );
        lap( 0 );

        // ---- the rounds
        std::vector<std::vector<int>> rl_th( par.nt );
        for( int k = 1; nbad && k < 64; ++k ) {
            // the cells whose box meets a bad zone
            par.slices( N, nn, [&]( int a0, int a1, int t ) {
                auto &rl = rl_th[ t ];
                rl.clear();
                auto zof = [&]( float x ) { return std::clamp( int( std::floor( ( T( x ) - lo ) / s ) ), 0, nzs - 1 ); };
                for( int i = a0 * N; i < a1 * N; ++i ) {
                    const float *z = &zbx[ 4 * size_t( i ) ];
                    if( z[ 0 ] > z[ 1 ] ) continue;
                    bool hit = false;
                    for( int u = zof( z[ 0 ] ); u <= zof( z[ 1 ] ) && ! hit; ++u )
                        for( int v = zof( z[ 2 ] ); v <= zof( z[ 3 ] ) && ! hit; ++v ) hit = bad[ size_t( u ) * nzs + v ];
                    if( hit ) rl.push_back( i );
                }
            } );
            std::vector<int> rl;
            for( auto &x : rl_th ) rl.insert( rl.end(), x.begin(), x.end() );
            if( rl.empty() ) break;
            if( ! padded ) { pad( ek.RB, psi, pe ); pmax = *std::max_element( psi.begin(), psi.end() ); padded = true; }
            // the next table: the halves of the bad zones
            const bool split = s / 2 >= zone_min;
            const T s2 = split ? s / 2 : s;
            const int nz2 = int( std::ceil( T( N ) / s2 ) ), ratio = split ? 2 : 1;
            prev.swap( bad );
            const int pnzs = nzs;
            auto in2 = [&]( int u, int v ) { return bool( prev[ size_t( u / ratio ) * pnzs + v / ratio ] ); };
            std::vector<int> live( par.nt, 0 ), exh( par.nt, 0 );
            par.slices( int( rl.size() ), SI( rl.size() ) * 4096, [&]( int q0, int q1, int t ) {
                auto &E = Et[ t ];
                E.assign( 2 * size_t( nz2 ) * nz2, T( 0 ) );
                T px[ C ], py[ C ]; int nbr[ C ];
                for( int q = q0; q < q1; ++q ) {
                    const int i = rl[ q ], nl = next_level( lvl[ i ] );
                    live[ t ] += lvl[ i ] != 255;
                    lvl[ i ] = ( unsigned char ) nl;
                    bool e = false;
                    const int n = level_polygon( i, nl, px, py, nbr, e );
                    exh[ t ] |= e;
                    integrate_poly( t, i, px, py, nbr, n, nl );
                    if( n < 3 ) { empty_box( i ); continue; }
                    T bb[ 4 ], ar;
                    box_of( i, px, py, n, bb, ar );
                    put_box( i, bb );
                    pieces( i, px, py, n, ar, bb, s2, nz2, in2, E );
                }
            } );
            E.assign( 2 * size_t( nz2 ) * nz2, T( 0 ) );
            for( auto &e : Et ) if( e.size() == E.size() ) for( size_t z = 0; z < E.size(); ++z ) E[ z ] += e[ z ];
            s = s2; nzs = nz2;
            judge( s, nzs, in2 );
            n_rounds = k;
            round_cells.push_back( int( rl.size() ) ); round_bad.push_back( nbad );
            int nlive = 0; for( int x : live ) nlive += x;
            bool ex = false; for( int x : exh ) ex = ex || x;
            if( ex ) exhausted = true;
            if( ! nlive ) break;
        }
        lap( 2 );
        for( int i = 0; i < nn; ++i ) { n_flag1 += lvl[ i ] > 1; n_exact += lvl[ i ] == 255; }
        // the faces out of ring 1: those of the last level of each cell
        for( auto &x : ex_th ) x.erase( std::remove_if( x.begin(), x.end(), [&]( const Extra &e ) { return e.l != std::max<unsigned char>( lvl[ e.i ], 1 ); } ), x.end() );
        gather_extras();
        if constexpr ( ! Integ::exact )
            symmetrize();
        t_geom += now_ms() - t0;
        if( exhausted && ek.RB < N ) {                   // ( a cell past the rings: twice as many )
            ek = ExactKernel<T>( std::min( 2 * ek.RB, N ) );
            exhausted = false;
            run_zones( psi );
        }
    }

    void run( const std::vector<T> &psi ) {
        if constexpr ( Integ::simd ) if( use_zones && fused_zones ) { run_zones( psi ); return; }
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
        bool out_of_rings = false;
        for( char e : exh ) out_of_rings = out_of_rings || e;
        exhausted = exhausted || out_of_rings;
        double t2 = now_ms();
        t_geom += t1 - t0;
        t_int += t2 - t1;
        // ( sharp masses: a cell may reach far beyond its rings, over seeds of almost no mass )
        if( out_of_rings && ek.RB < N ) {
            ek = ExactKernel<T>( std::min( 2 * ek.RB, N ) );
            run( psi );
        }
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

/// DAMPED NEWTON on one level. `target`: the masses in index units ( they sum to the mass of the domain in index units ).
/// `psi`: start on input, solution on output. `S`: index units per unit of mass ( for `tol_abs` ).
template<class Integ>
void newton( Evaluator<Integ> &ev, std::vector<T> target, std::vector<T> &psi, double tol_rel, double tol_abs, double S,
             const Options &o, Stats &st, double domain = 0 ) {
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
    // `sdotplan/Solve.h` does -- the cells tile the domain whatever psi, but the finer levels take it from the coarsest: a start
    // interpolated across very sharp masses can have cells that a far seed should cut and that the certificates of the rings let through
    {
        T sm = 0, st_ = 0;
        for( int i = 0; i < N * N; ++i ) { sm += ev.mass[ i ]; st_ += target[ i ]; }
        if( domain > 0 ) sm = T( domain * S );
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
    // THE LIMIT OF THE STEP ( `sdotplan/Bounds.h` ): the trial at `tau` left cells under the floor. Each one, its edges frozen as in
    // the accepted diagram ( `ev` is put back on `psi` for that ), bisected on `tau`: what goes back is the largest `tau` that
    // keeps them all above the floor, 0 when they do not fall with their edges frozen. An edge may still appear before: the trial checks
    std::vector<int> bad, bad_nb, bad_of, cand, fell;
    std::vector<char> below;
    auto tau_limit = [ & ]( T tau ) {
        bad.clear();
        for( int i = 0; i < N * N; ++i ) if( ev.mass[ i ] <= eps0 ) bad.push_back( i );
        ev.geometry( psi );
        bad_nb.clear(); bad_of.assign( 1, 0 ); cand.clear();
        for( SI q = 0; q < SI( bad.size() ); ++q ) {
            T px[ ExactKernel<T>::C ], py[ ExactKernel<T>::C ];
            int nbr[ ExactKernel<T>::C ];
            bool ex = false;
            const int n = ev.polygon( bad[ q ], px, py, nbr, ex );
            for( int e = 0; e < n; ++e ) if( nbr[ e ] >= 0 ) bad_nb.push_back( nbr[ e ] );
            bad_of.push_back( int( bad_nb.size() ) );
            if( n ) cand.push_back( int( q ) );
        }
        auto falling = [ & ]( T s ) {
            par.slices( N, nn, [&]( int a0, int a1, int ) {
                for( int a = a0; a < a1; ++a ) for( int b = 0; b < N; ++b ) trial[ a * N + b ] = psi[ a * N + b ] + s * dx[ ( a + 1 ) * P + b + 1 ];
            } );
            below.assign( cand.size(), 0 );
            par.slices( int( cand.size() ), SI( cand.size() ) * 1024, [&]( int c0, int c1, int t ) {
                for( int c = c0; c < c1; ++c ) {
                    const int q = cand[ c ];
                    below[ c ] = ev.frozen_mass( t, bad[ q ], bad_nb.data() + bad_of[ q ], bad_of[ q + 1 ] - bad_of[ q ], trial ) <= eps0;
                }
            } );
            fell.clear();
            for( size_t c = 0; c < cand.size(); ++c ) if( below[ c ] ) fell.push_back( cand[ c ] );
        };
        falling( tau );
        if( fell.empty() ) return T( 0 );
        cand.swap( fell );
        T lo = 0, hi = tau;
        for( int r = 0; r < 48 && hi > T( 1e-10 ) && ! ( lo > 0 && hi - lo <= T( 0.05 ) * hi ); ++r ) {
            const T mid = ( lo + hi ) / 2;
            falling( mid );
            if( fell.empty() ) lo = mid;
            else { hi = mid; cand.swap( fell ); }
        }
        return lo;
    };
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
        const int cg_its = pcg_fused( op, M, rhs, dx, o.cg_rtol, 20000, par );
        st.lin_nb_iter += cg_its;
        st.t_lin += ( now_ms() - tl ) / 1e3;
        // damping
        const T slope = T( 0.5 );
        T tau_min = T( 1e-6 );
        T tau = 1;
        int limit_rounds = 0;
        while( true ) {
            par.slices( N, nn, [&]( int a0, int a1, int ) {
                for( int a = a0; a < a1; ++a ) for( int b = 0; b < N; ++b ) trial[ a * N + b ] = psi[ a * N + b ] + tau * dx[ ( a + 1 ) * P + b + 1 ];
            } );
            ev.run( trial );
            ++st.nb_diag;
            Err e2 = err_of( ev.mass );
            if( ! ev.exhausted && e2.mmin <= eps0 && limit_rounds < 8 ) {
                ++limit_rounds;
                const T al = tau_limit( tau );
                if( al > 0 ) {
                    tau = T( 0.9 ) * al;
                    tau_min = std::min( tau_min, tau / 64 );
                    continue;
                }
                limit_rounds = 8;                        // ( no limit: back on the trial, that `tau_limit` moved, and the damping alone )
                continue;
            }
            // ( a trial whose cells outgrow the exact path's rings is rejected like one that empties a cell )
            if( ! ev.exhausted && e2.l2 <= ( 1 - slope * tau ) * err.l2 ) {
                if( e2.mmin <= eps0 ) {
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
            break;
        }
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
    double domain = 0;                                   // ( what the domain holds, from the coarsest level )
    for( int n = n0, l = int( nus.size() ) - 1; n <= N; n *= 2, --l ) {
        if( n > n0 ) psi = refine_psi( n / 2, psi, par );
        const double S = double( n ) * n, hl = h * N / n;
        std::vector<T> target( nus[ l ].size() );
        for( size_t i = 0; i < target.size(); ++i ) target[ i ] = nus[ l ][ i ] * S;
        auto integ = make_integ( n, hl, 0.0 );
        Evaluator<DECAYED_TYPE_OF( integ )> ev( n, integ, par );
        Stats sl;
        const bool fine = n == N;
        newton( ev, target, psi, fine ? o.tol_rel : std::max( o.tol_rel, o.ctol ), fine ? o.tol_abs : 0.0, S, o, sl, domain );
        domain = sl.domain_mass;
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

