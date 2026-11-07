#pragma once

// =====================================================================================
// THE DIRACS OF A 2D GRID SOLVED ON THE CARD ( `SumOfDiracsOnAGrid`, `SdotPlanNd._build_grid_card` ): the dedicated solver of
// `grid/GridNewton2D.h`, step for step, in ONE ffi call whose handler drives the loops on the call's stream. N x N seeds at the
// centres of the square cells of a grid, N a power of two, the domain the box of the grid; everything in INDEX UNITS ( seed
// `( a, b )` at `( a, b )`, the domain `[ -1/2, N - 1/2 ]^2`, `psi = w / h^2` ), physical units only at the edges.
//
//   geometry   ring-1 stencil, one thread per cell, into SoA planes ( `ring1_kernel`, the tables of `grid/StencilTables.h` folded
//              at compile time ), its certificate ( `certify1_kernel` ); on the flagged cells ( a list ): ring 2, the
//              cross-level certificate on their 3 x 3 neighbourhoods, the exact clipping on what is left ( `exact_kernel`, the
//              port of `grid/ExactKernel.h` )
//   integrals  the density on the edges of each cell, the card's closed forms ( `Density2D.cuh` ): a constant ( shoelace and
//              lengths ), an `Image` on a regular grid ( Green on its rows ), isotropic gaussians ( polar corners, `erf`s ) --
//              the CPU cuts the cells by the pixels and runs quadratures, both exact to rounding ( or to 1e-14 )
//   linear     H = dm / dpsi, matrix-free: 8 planes of weights toward the ring-1 neighbours, plus the faces toward the seeds out of
//              ring 1 ( ring-2 and exact cells: a short list, merged on the host as the CPU does ). PCG preconditioned by the
//              weighted semi-coarsened tent blurs ( `TensorPrecond` of the CPU ): the tree of the ( la, lb ) grids made by one
//              launch per level along the first axis and ONE launch for all the levels along the second ( a block per row )
//   start      coarse to fine: the solve on n0, 2 n0, ..., N, each level from the cubic interpolation of the previous psi; the
//              gaussians first on the coarsest level through their widening ( the CPU's width continuation )
//   damping    sharp targets, as the CPU: the line search on the residual, `tau` brought under the limit of the cells that a trial
//              left under the floor ( `tau_limit` ), the cells that remain empty repaired one by one ( `repair` )
//
// The host reads back SCALARS where it decides: the counts of the flagged cells ( one read when there is none ), the residual of
// each diagram, the residual of each CG iteration. The reductions are those of `Reduce.cuh` ( a fixed tree: the same number at
// every run ); the lists of flagged cells are made by atomics ( their order changes, nothing computed from them does: the faces
// out of ring 1 are sorted on the host ).
// =====================================================================================

#include "Reduce.cuh"
#include "../grid/StencilTables.h"
#include "../sdotplan/Report.h"
#include "../sdotplan/WarmRepair.h"
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace sdot::gpu2d::grid {

namespace sp = sdot::sdotplan;

constexpr int    C    = 128;                             ///< the vertices of a cell of the exact path ( `ExactKernel::C` )
constexpr int    RB   = 24;                              ///< the rings of the exact path's offsets at the start ( they grow on demand, `Solver::grow_rings` )
constexpr int    MAXL = 15;                              ///< the levels of the preconditioner's trees ( N up to 2^16 )
constexpr double TOL  = 1e-9;                            ///< the certificates' tolerance ( index units, the CPU's )

// ================================================================================================
// the tables of the stencils
// ================================================================================================
/// `grid/StencilTables.h` as a value, plus what the certificates derive from it ( `opp`, `back`, `dir_of`, `dir1` )
template<int R>
struct Tab {
    using Tb = sdot::grid::StencilTables<R>;
    static constexpr int D = Tb::D, NO = Tb::nb_offsets, NP = Tb::nb_pairs;
    double dir_x[ D ] = {}, dir_y[ D ] = {}, dir_inv_norm[ D ] = {}, off_m[ NO ] = {}, off_sqn[ NO ] = {};
    double pair_lam[ NP ] = {}, pair_mu[ NP ] = {}, vinv[ 4 * D ] = {};
    int    off_dir[ NO ] = {}, off_x[ NO ] = {}, off_y[ NO ] = {}, pair_beg[ D + 1 ] = {}, pair_j[ NP ] = {}, pair_l[ NP ] = {};
    int    opp[ D ] = {};                                ///< the direction opposite to k
    int    back[ NO ] = {};                              ///< the offset opposite to o
    int    dir_of[ 25 ] = {};                            ///< `( dx + 2 ) * 5 + dy + 2` -> the direction of that normal ( -1: none )
    int    dir1[ 9 ] = {};                               ///< `( dx + 1 ) * 3 + dy + 1` -> the ring-1 direction of that offset ( -1: none )
    int    dir_px = 0, dir_mx = 0, dir_py = 0, dir_my = 0;
};

/// ( a constant expression wherever it is called: the device's ring 1 folds it, the host copies it into `c_tab*` )
template<int R>
constexpr __host__ __device__ Tab<R> make_tab() {
    using Tb = sdot::grid::StencilTables<R>;
    using T1 = sdot::grid::StencilTables<1>;
    Tab<R> t;
    for ( int k = 0; k < t.D; ++k ) {
        t.dir_x[ k ] = Tb::dir_x[ k ]; t.dir_y[ k ] = Tb::dir_y[ k ]; t.dir_inv_norm[ k ] = Tb::dir_inv_norm[ k ];
        for ( int q = 0; q < 4; ++q ) t.vinv[ 4 * k + q ] = Tb::vinv[ 4 * k + q ];
    }
    for ( int k = 0; k <= t.D; ++k ) t.pair_beg[ k ] = Tb::pair_beg[ k ];
    for ( int o = 0; o < t.NO; ++o ) {
        t.off_dir[ o ] = Tb::off_dir[ o ]; t.off_x[ o ] = Tb::off_x[ o ]; t.off_y[ o ] = Tb::off_y[ o ];
        t.off_m[ o ] = Tb::off_m[ o ]; t.off_sqn[ o ] = Tb::off_sqn[ o ];
    }
    for ( int p = 0; p < t.NP; ++p ) {
        t.pair_j[ p ] = Tb::pair_j[ p ]; t.pair_l[ p ] = Tb::pair_l[ p ]; t.pair_lam[ p ] = Tb::pair_lam[ p ]; t.pair_mu[ p ] = Tb::pair_mu[ p ];
    }
    for ( int k = 0; k < t.D; ++k )
        for ( int l = 0; l < t.D; ++l )
            if ( Tb::dir_x[ l ] == -Tb::dir_x[ k ] && Tb::dir_y[ l ] == -Tb::dir_y[ k ] ) t.opp[ k ] = l;
    for ( int o = 0; o < t.NO; ++o )
        for ( int p = 0; p < t.NO; ++p )
            if ( Tb::off_x[ p ] == -Tb::off_x[ o ] && Tb::off_y[ p ] == -Tb::off_y[ o ] ) t.back[ o ] = p;
    for ( int q = 0; q < 25; ++q ) t.dir_of[ q ] = -1;
    for ( int k = 0; k < t.D; ++k ) t.dir_of[ int( Tb::dir_x[ k ] + 2 ) * 5 + int( Tb::dir_y[ k ] + 2 ) ] = k;
    for ( int q = 0; q < 9; ++q ) t.dir1[ q ] = -1;
    for ( int k = 0; k < T1::D; ++k ) t.dir1[ ( T1::off_x[ k ] + 1 ) * 3 + T1::off_y[ k ] + 1 ] = k;
    t.dir_px = Tb::dir_px; t.dir_mx = Tb::dir_mx; t.dir_py = Tb::dir_py; t.dir_my = Tb::dir_my;
    return t;
}

/// the tables read at RUNTIME indices ( the rare paths: ring 2, the cross-level certificate, the faces of the flagged cells ),
/// copied at the start of each solve
static __constant__ Tab<1> c_tab1;
static __constant__ Tab<2> c_tab2;

inline double grid_now() {
    using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count();
}

__device__ __forceinline__ double inf_d() { return __longlong_as_double( 0x7ff0000000000000LL ); }

/// THE RING-R CELL of seed `( a, b )` ( `StencilKernel::row_block`, one lane ): the offsets merged per direction ( the argmin kept as
/// owner, -1 the box ), tightened by the pairs, one vertex per pair of consecutive directions ( vertex k ends edge k ). Ring 1:
/// `t` a constant expression and every loop unrolled ( registers ); ring 2: the tables in constant memory.
template<int R,class Tb>
__device__ __forceinline__ void stencil( const Tb &t, int N, const double *psi, int a, int b, double *vx, double *vy, signed char *ow ) {
    constexpr int D = Tab<R>::D, NO = Tab<R>::NO, UR = R == 1 ? 64 : 1;
    const double inf = inf_d();
    double h[ D ], ht[ D ];
#pragma unroll
    for ( int d = 0; d < D; ++d ) { h[ d ] = inf; ow[ d ] = -1; }
    const double pi = psi[ SI( a ) * N + b ];
#pragma unroll UR
    for ( int o = 0; o < NO; ++o ) {
        const int na = a + t.off_x[ o ], nb = b + t.off_y[ o ];
        if ( na < 0 || na >= N || nb < 0 || nb >= N )
            continue;                                    // ( the CPU's padding: -inf, an offset of +inf )
        const int d = t.off_dir[ o ];
        const double hk = ( t.off_sqn[ o ] + pi - psi[ SI( na ) * N + nb ] ) * ( 0.5 / t.off_m[ o ] );
        if ( hk < h[ d ] ) { h[ d ] = hk; ow[ d ] = ( signed char ) o; }
    }
    // the box
    const double hpx = N - 0.5 - a, hmx = a + 0.5, hpy = N - 0.5 - b, hmy = b + 0.5;
    if ( hpx <= h[ t.dir_px ] ) { h[ t.dir_px ] = hpx; ow[ t.dir_px ] = -1; }
    if ( hmx <= h[ t.dir_mx ] ) { h[ t.dir_mx ] = hmx; ow[ t.dir_mx ] = -1; }
    if ( hpy <= h[ t.dir_py ] ) { h[ t.dir_py ] = hpy; ow[ t.dir_py ] = -1; }
    if ( hmy <= h[ t.dir_my ] ) { h[ t.dir_my ] = hmy; ow[ t.dir_my ] = -1; }
    // the tightening ( one pass on the raw offsets is exact: optimal LP bases have 2 elements )
#pragma unroll
    for ( int k = 0; k < D; ++k ) {
        double v = h[ k ];
#pragma unroll UR
        for ( int p = t.pair_beg[ k ]; p < t.pair_beg[ k + 1 ]; ++p )
            v = fmin( v, t.pair_lam[ p ] * h[ t.pair_j[ p ] ] + t.pair_mu[ p ] * h[ t.pair_l[ p ] ] );
        ht[ k ] = v;
    }
#pragma unroll
    for ( int k = 0; k < D; ++k ) {
        const int k1 = ( k + 1 ) % D;
        vx[ k ] = t.vinv[ 4 * k + 0 ] * ht[ k ] + t.vinv[ 4 * k + 1 ] * ht[ k1 ];
        vy[ k ] = t.vinv[ 4 * k + 2 ] * ht[ k ] + t.vinv[ 4 * k + 3 ] * ht[ k1 ];
    }
}

/// `f( std::integral_constant<int,I>() )` for I in [ 0, M ): the index a constant expression in `f`
template<int M,class F>
__device__ __forceinline__ void unroll( F &&f ) {
    [&]<int... I>( std::integer_sequence<int,I...> ) { ( f( std::integral_constant<int,I>() ), ... ); }( std::make_integer_sequence<int,M>() );
}

/// the pair of the tightening that `p` is in ( a constant expression )
template<int R>
constexpr __host__ __device__ int pair_dir( int p ) {
    const Tab<R> t = make_tab<R>();
    int k = 0;
    while ( t.pair_beg[ k + 1 ] <= p ) ++k;
    return k;
}

/// `stencil< 1 >` with every entry of the tables a constant of the code ( no table in memory: the cells' hot loop )
__device__ __forceinline__ void stencil1( int N, const double *psi, int a, int b, double *vx, double *vy, signed char *ow ) {
    constexpr Tab<1> T = make_tab<1>();
    const double inf = inf_d();
    double h[ 8 ], ht[ 8 ];
#pragma unroll
    for ( int d = 0; d < 8; ++d ) { h[ d ] = inf; ow[ d ] = -1; }
    const double pi = psi[ SI( a ) * N + b ];
    unroll<8>( [&]( auto O ) {
        constexpr int o = decltype( O )::value, d = T.off_dir[ o ], dx = T.off_x[ o ], dy = T.off_y[ o ];
        constexpr double c = T.off_sqn[ o ], sc = 0.5 / T.off_m[ o ];
        const int na = a + dx, nb = b + dy;
        if ( na < 0 || na >= N || nb < 0 || nb >= N )
            return;
        const double hk = ( c + pi - psi[ SI( na ) * N + nb ] ) * sc;
        if ( hk < h[ d ] ) { h[ d ] = hk; ow[ d ] = ( signed char ) o; }
    } );
    const double hpx = N - 0.5 - a, hmx = a + 0.5, hpy = N - 0.5 - b, hmy = b + 0.5;
    if ( hpx <= h[ T.dir_px ] ) { h[ T.dir_px ] = hpx; ow[ T.dir_px ] = -1; }
    if ( hmx <= h[ T.dir_mx ] ) { h[ T.dir_mx ] = hmx; ow[ T.dir_mx ] = -1; }
    if ( hpy <= h[ T.dir_py ] ) { h[ T.dir_py ] = hpy; ow[ T.dir_py ] = -1; }
    if ( hmy <= h[ T.dir_my ] ) { h[ T.dir_my ] = hmy; ow[ T.dir_my ] = -1; }
#pragma unroll
    for ( int k = 0; k < 8; ++k ) ht[ k ] = h[ k ];
    unroll<Tab<1>::NP>( [&]( auto P ) {
        constexpr int p = decltype( P )::value, k = pair_dir<1>( p ), j = T.pair_j[ p ], l = T.pair_l[ p ];
        constexpr double lam = T.pair_lam[ p ], mu = T.pair_mu[ p ];
        ht[ k ] = fmin( ht[ k ], lam * h[ j ] + mu * h[ l ] );
    } );
    unroll<8>( [&]( auto K ) {
        constexpr int k = decltype( K )::value, k1 = ( k + 1 ) % 8;
        constexpr double m0 = T.vinv[ 4 * k ], m1 = T.vinv[ 4 * k + 1 ], m2 = T.vinv[ 4 * k + 2 ], m3 = T.vinv[ 4 * k + 3 ];
        vx[ k ] = m0 * ht[ k ] + m1 * ht[ k1 ];
        vy[ k ] = m2 * ht[ k ] + m3 * ht[ k1 ];
    } );
}

// ================================================================================================
// the geometry
// ================================================================================================
/// the ring-1 planes: `vx[ k * nn + i ]`, vertex k of cell i ( relative to its seed ), `ow` the owner of edge k
__global__ void __launch_bounds__( BLOCK ) ring1_kernel( int N, const double *psi, double *vx, double *vy, signed char *ow ) {
    const SI nn = SI( N ) * N, i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= nn )
        return;
    double px[ 8 ], py[ 8 ];
    signed char o[ 8 ];
    stencil1( N, psi, int( i / N ), int( i % N ), px, py, o );
#pragma unroll
    for ( int k = 0; k < 8; ++k ) { vx[ k * nn + i ] = px[ k ]; vy[ k * nn + i ] = py[ k ]; ow[ k * nn + i ] = o[ k ]; }
}

/// what the passes of one diagram count ( zeroed at its start, read back by the host )
struct Counts {
    unsigned n1;                                         ///< the flagged cells ( `list1` )
    unsigned nchk;                                       ///< the cells of their neighbourhoods ( `check` )
    unsigned nex;                                        ///< the cells of the exact path ( `exl` )
    unsigned nx;                                         ///< the faces out of ring 1 ( `extras` )
    unsigned exhausted;                                  ///< a cell of the exact path wider than its rings, or with too many vertices
    unsigned nbad;                                       ///< the cells under the floor ( `tau_limit` )
    unsigned nfell;                                      ///< those that fall at the `tau` tried
    unsigned pad[ 1 ];
};

/// THE RING-1 CERTIFICATE ( `StencilKernel::certify_row` ): every positive edge k of cell i ( owner j ) must be edge opp( k ) of
/// cell j, owned by i, with the same endpoints; a cell that fails goes to `list1`
__global__ void __launch_bounds__( BLOCK ) certify1_kernel( int N, const double *vx, const double *vy, const signed char *ow,
                                                            unsigned char *flag1, int *list1, Counts *cnt ) {
    const SI nn = SI( N ) * N, i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= nn )
        return;
    const Tab<1> &T = c_tab1;                            // ( constant memory: a constexpr local here was copied to local memory )
    bool bad = false;
#pragma unroll
    for ( int k = 0; k < 8; ++k ) {
        const int km = ( k + 7 ) % 8, q = T.opp[ k ], qm = ( q + 7 ) % 8;
        if ( ow[ k * nn + i ] < 0 )                      // ( ring 1: the owner of edge k is offset k, or the box )
            continue;
        const double ivx = vx[ k * nn + i ], ivy = vy[ k * nn + i ], imx = vx[ km * nn + i ], imy = vy[ km * nn + i ];
        if ( fabs( ivx - imx ) + fabs( ivy - imy ) <= TOL )
            continue;
        const int dx = T.off_x[ k ], dy = T.off_y[ k ];
        const SI j = i + SI( dx ) * N + dy;
        const bool same = ow[ q * nn + j ] == T.back[ k ]
            && fabs( imx - vx[ q * nn + j ] - dx ) <= TOL && fabs( imy - vy[ q * nn + j ] - dy ) <= TOL
            && fabs( ivx - vx[ qm * nn + j ] - dx ) <= TOL && fabs( ivy - vy[ qm * nn + j ] - dy ) <= TOL;
        bad |= ! same;
    }
    flag1[ i ] = bad;
    if ( bad )
        list1[ atomicAdd( &cnt->n1, 1u ) ] = int( i );
}

/// the ring-2 planes of the flagged cells, at their place in `list1` ( `pos2` )
struct Planes2 {
    double      *vx = nullptr, *vy = nullptr;
    signed char *ow = nullptr;
    SI           cap = 0;
};

__global__ void __launch_bounds__( BLOCK ) ring2_kernel( int N, const double *psi, const int *list1, SI n1, Planes2 p2, int *pos2 ) {
    const SI p = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( p >= n1 )
        return;
    const int i = list1[ p ];
    double px[ 16 ], py[ 16 ];
    signed char o[ 16 ];
    stencil<2>( c_tab2, N, psi, i / N, i % N, px, py, o );
    for ( int k = 0; k < 16; ++k ) { p2.vx[ k * p2.cap + p ] = px[ k ]; p2.vy[ k * p2.cap + p ] = py[ k ]; p2.ow[ k * p2.cap + p ] = o[ k ]; }
    pos2[ i ] = int( p );
}

/// the 3 x 3 neighbourhoods of the flagged cells, each cell once ( `mark` ), into `check`
__global__ void __launch_bounds__( BLOCK ) neighbourhood_kernel( int N, const int *list1, SI n1, unsigned *mark, int *check, Counts *cnt ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q >= 9 * n1 )
        return;
    const int i = list1[ q / 9 ], d = int( q % 9 ), a = i / N + d / 3 - 1, b = i % N + d % 3 - 1;
    if ( a < 0 || a >= N || b < 0 || b >= N )
        return;
    const int j = a * N + b;
    if ( atomicExch( &mark[ j ], 1u ) == 0u )
        check[ atomicAdd( &cnt->nchk, 1u ) ] = j;
}

/// what a cell's planes are, ring 1 or ring 2 as `flag1` says
struct RingView {
    const double *vx, *vy;
    const signed char *ow;
    SI plane, at;                                        ///< `vx[ k * plane + at ]`
    bool l2;
};

__device__ __forceinline__ RingView ring_of( SI i, SI nn, const unsigned char *flag1, const double *vx1, const double *vy1, const signed char *ow1,
                                             const Planes2 &p2, const int *pos2 ) {
    if ( flag1[ i ] ) return RingView{ p2.vx, p2.vy, p2.ow, p2.cap, SI( pos2[ i ] ), true };
    return RingView{ vx1, vy1, ow1, nn, i, false };
}

/// THE CROSS-LEVEL CERTIFICATE of cell i ( `Evaluator::cell_ok` ): its positive edges seen from their owners, whatever ring each
/// side was built with
__device__ bool cell_ok( int N, SI i, const unsigned char *flag1, const double *vx1, const double *vy1, const signed char *ow1,
                         const Planes2 &p2, const int *pos2 ) {
    const SI nn = SI( N ) * N;
    const RingView ri = ring_of( i, nn, flag1, vx1, vy1, ow1, p2, pos2 );
    const int D = ri.l2 ? 16 : 8;
    for ( int k = 0; k < D; ++k ) {
        const int km = ( k + D - 1 ) % D, o = ri.ow[ k * ri.plane + ri.at ];
        const double ex = ri.vx[ k * ri.plane + ri.at ] - ri.vx[ km * ri.plane + ri.at ], ey = ri.vy[ k * ri.plane + ri.at ] - ri.vy[ km * ri.plane + ri.at ];
        if ( o < 0 || fabs( ex ) + fabs( ey ) <= TOL )
            continue;
        const int dx = ri.l2 ? c_tab2.off_x[ o ] : c_tab1.off_x[ o ], dy = ri.l2 ? c_tab2.off_y[ o ] : c_tab1.off_y[ o ];
        const SI j = i + SI( dx ) * N + dy;
        const RingView rj = ring_of( j, nn, flag1, vx1, vy1, ow1, p2, pos2 );
        const int ndx = -int( ri.l2 ? c_tab2.dir_x[ k ] : c_tab1.dir_x[ k ] ), ndy = -int( ri.l2 ? c_tab2.dir_y[ k ] : c_tab1.dir_y[ k ] );
        const int q = rj.l2 ? c_tab2.dir_of[ ( ndx + 2 ) * 5 + ndy + 2 ] : c_tab1.dir_of[ ( ndx + 2 ) * 5 + ndy + 2 ];
        if ( q < 0 )
            return false;
        const int Dj = rj.l2 ? 16 : 8, qm = ( q + Dj - 1 ) % Dj;
        const int o2 = rj.ow[ q * rj.plane + rj.at ];
        if ( o2 < 0 )
            return false;
        if ( ( rj.l2 ? c_tab2.off_x[ o2 ] : c_tab1.off_x[ o2 ] ) != -dx || ( rj.l2 ? c_tab2.off_y[ o2 ] : c_tab1.off_y[ o2 ] ) != -dy )
            return false;
        if ( fabs( ri.vx[ km * ri.plane + ri.at ] - rj.vx[ q * rj.plane + rj.at ] - dx ) > TOL || fabs( ri.vy[ km * ri.plane + ri.at ] - rj.vy[ q * rj.plane + rj.at ] - dy ) > TOL )
            return false;
        if ( fabs( ri.vx[ k * ri.plane + ri.at ] - rj.vx[ qm * rj.plane + rj.at ] - dx ) > TOL || fabs( ri.vy[ k * ri.plane + ri.at ] - rj.vy[ qm * rj.plane + rj.at ] - dy ) > TOL )
            return false;
    }
    return true;
}

__global__ void __launch_bounds__( BLOCK ) cell_ok_kernel( int N, const int *check, SI nchk, const unsigned char *flag1, const double *vx1, const double *vy1,
                                                           const signed char *ow1, Planes2 p2, const int *pos2, unsigned char *flag2 ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q >= nchk )
        return;
    const int i = check[ q ];
    flag2[ i ] = ! cell_ok( N, i, flag1, vx1, vy1, ow1, p2, pos2 );
}

/// a ring-1 cell that fails the cross-level check takes the exact path too ( into `list1` ); the marks cleared
__global__ void __launch_bounds__( BLOCK ) promote_kernel( const int *check, SI nchk, unsigned *mark, const unsigned char *flag2, unsigned char *flag1,
                                                           int *list1, Counts *cnt ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q >= nchk )
        return;
    const int i = check[ q ];
    mark[ i ] = 0u;
    if ( flag2[ i ] && ! flag1[ i ] ) {
        flag1[ i ] = 1;
        list1[ atomicAdd( &cnt->n1, 1u ) ] = i;
    }
}

/// the cells of the exact path ( flagged twice ), into `exl` ( `expos`: their place there )
__global__ void __launch_bounds__( BLOCK ) exact_list_kernel( const int *list1, const Counts *cnt, const unsigned char *flag2, int *exl, int *expos, Counts *cnt_out ) {
    const SI n1 = cnt->n1;
    for ( SI p = SI( blockIdx.x ) * BLOCK + threadIdx.x; p < n1; p += SI( gridDim.x ) * BLOCK ) {
        const int i = list1[ p ];
        if ( flag2[ i ] ) {
            const unsigned e = atomicAdd( &cnt_out->nex, 1u );
            exl[ e ] = i;
            expos[ i ] = int( e );
        }
    }
}

/// the polygons of the exact path: `px[ p * C + e ]`, edge e toward `nb[ p * C + e ]` ( -1: the box ), `n[ p ]` vertices
struct ExactPolys {
    double *px = nullptr, *py = nullptr;
    int    *nb = nullptr, *n = nullptr;
    SI      cap = 0;
};

/// the offsets of the exact path sorted by distance ( `ExactKernel`'s list )
struct ExactOffsets {
    const int    *ox = nullptr, *oy = nullptr;
    const double *nk = nullptr;
    int           nb = 0;
};

/// THE EXACT CELL ( `ExactKernel::clip` ): the box clipped by the neighbours by increasing distance, up to the security radius
__global__ void __launch_bounds__( BLOCK ) exact_kernel( int N, const double *psi, const double *pmax_, const int *exl, SI nex, ExactOffsets eo,
                                                         ExactPolys ep, Counts *cnt ) {
    const SI p = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( p >= nex )
        return;
    const int i = exl[ p ], a = i / N, b = i % N;
    const double pmax = *pmax_;
    double px[ C ], py[ C ], qx[ C ], qy[ C ], s[ C ];
    int pl[ C ], ql[ C ];
    int n = 4;
    px[ 0 ] = -a - 0.5;    py[ 0 ] = -b - 0.5;    pl[ 0 ] = -1;
    px[ 1 ] = N - 0.5 - a; py[ 1 ] = -b - 0.5;    pl[ 1 ] = -1;
    px[ 2 ] = N - 0.5 - a; py[ 2 ] = N - 0.5 - b; pl[ 2 ] = -1;
    px[ 3 ] = -a - 0.5;    py[ 3 ] = N - 0.5 - b; pl[ 3 ] = -1;
    const double pi = psi[ i ];
    double R2 = 0;
    for ( int e = 0; e < n; ++e ) R2 = fmax( R2, px[ e ] * px[ e ] + py[ e ] * py[ e ] );
    bool done = false;
    for ( int o = 0; o < eo.nb; ++o ) {
        const double R = sqrt( R2 ), k = eo.nk[ o ];
        if ( k >= R && k * k - 2 * k * R + pi - pmax > 0 ) { done = true; break; }
        if ( n + 1 >= C ) break;                         // ( a cut adds one vertex at most )
        const int na = a + eo.ox[ o ], nb = b + eo.oy[ o ];
        if ( na < 0 || na >= N || nb < 0 || nb >= N )
            continue;
        const double kx = eo.ox[ o ], ky = eo.oy[ o ];
        const double c = ( kx * kx + ky * ky + pi - psi[ SI( na ) * N + nb ] ) / 2;
        bool out = false;
        for ( int e = 0; e < n; ++e ) { s[ e ] = kx * px[ e ] + ky * py[ e ] - c; out |= s[ e ] > 0; }
        if ( ! out )
            continue;
        int m = 0;
        for ( int e = 0; e < n && m < C - 1; ++e ) {
            const int f = e + 1 == n ? 0 : e + 1;
            if ( s[ e ] <= 0 ) {
                qx[ m ] = px[ e ]; qy[ m ] = py[ e ]; ql[ m++ ] = pl[ e ];
                if ( s[ f ] > 0 ) {
                    const double t = s[ e ] / ( s[ e ] - s[ f ] );
                    qx[ m ] = px[ e ] + t * ( px[ f ] - px[ e ] ); qy[ m ] = py[ e ] + t * ( py[ f ] - py[ e ] ); ql[ m++ ] = o;
                }
            } else if ( s[ f ] <= 0 ) {
                const double t = s[ e ] / ( s[ e ] - s[ f ] );
                qx[ m ] = px[ e ] + t * ( px[ f ] - px[ e ] ); qy[ m ] = py[ e ] + t * ( py[ f ] - py[ e ] ); ql[ m++ ] = pl[ e ];
            }
        }
        n = m;
        R2 = 0;
        for ( int e = 0; e < n; ++e ) { px[ e ] = qx[ e ]; py[ e ] = qy[ e ]; pl[ e ] = ql[ e ]; R2 = fmax( R2, px[ e ] * px[ e ] + py[ e ] * py[ e ] ); }
        if ( n == 0 ) { done = true; break; }
    }
    if ( ! done )
        atomicOr( &cnt->exhausted, 1u );
    ep.n[ p ] = n;
    for ( int e = 0; e < n; ++e ) {
        ep.px[ p * C + e ] = px[ e ];
        ep.py[ p * C + e ] = py[ e ];
        ep.nb[ p * C + e ] = pl[ e ] < 0 ? -1 : i + eo.ox[ pl[ e ] ] * N + eo.oy[ pl[ e ] ];
    }
}

/// THE POLYGON OF A FLAGGED CELL ( relative to its seed ): vertex e, edge e from vertex e to e+1 toward `nbr[ e ]` ( -1: the box );
/// the exact path's or the ring-2 planes', as the certificates decided ( `Evaluator::polygon` )
__device__ __forceinline__ int flagged_polygon( int N, int i, const unsigned char *flag2, const int *pos2, const int *expos, const Planes2 &p2,
                                                const ExactPolys &ep, double *px, double *py, int *nbr ) {
    if ( flag2[ i ] ) {
        const SI p = expos[ i ];
        const int n = ep.n[ p ];
        for ( int e = 0; e < n; ++e ) { px[ e ] = ep.px[ p * C + e ]; py[ e ] = ep.py[ p * C + e ]; nbr[ e ] = ep.nb[ p * C + e ]; }
        return n;
    }
    const SI p = pos2[ i ];
    for ( int e = 0; e < 16; ++e ) {
        const int v = ( e + 15 ) % 16;                   // ( vertex e = plane vertex e - 1: edge e is direction e )
        px[ e ] = p2.vx[ v * p2.cap + p ];
        py[ e ] = p2.vy[ v * p2.cap + p ];
        const int o = p2.ow[ e * p2.cap + p ];
        nbr[ e ] = o < 0 ? -1 : i + c_tab2.off_x[ o ] * N + c_tab2.off_y[ o ];
    }
    return 16;
}

// ================================================================================================
// the densities, in the index units of a level ( `rho_idx( u ) = n^2 h_n^2 rho( x )`, `x = o + h_n ( u + 1/2 )` )
// ================================================================================================
struct GUnit  { static constexpr bool exact = true; };
struct GImage { static constexpr bool exact = false; DensImage im; };
struct GGauss { static constexpr bool exact = false; DensGauss g; };

struct GImage1 { static constexpr bool exact = false; DensImage1 im; };

/// the prefixes of a row ( `DensImage1::p`, `Density2D.cuh::image1_prefix_row` )
__global__ void __launch_bounds__( BLOCK ) grid_image1_prefix( SI nx, SI ny, const double *kx, const double *c, double *p ) {
    const SI j = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( j < ny )
        image1_prefix_row( nx, ny, j, kx, c, p );
}

/// ONE EDGE of the image ( `Density2D.cuh::image_edge` ), but an edge ALONG a line of the pixels takes the pixels on the side of its
/// cell ( the left of the walk: the cells are counterclockwise ) for `int rho ds`, as the CPU does -- `floor` takes the right or the
/// upper one, and the two faces of a pixel's edge would then both see the same pixel ( the mean the laplacian makes of them would
/// not be the CPU's ). The mass is the same either way ( `F0` is continuous across a column; a row's edge has `dY = 0` ).
template<bool MOM>
__device__ __forceinline__ double grid_image_edge( const DensImage &im, ImageRefs &ref, double X0, double Y0, double X1, double Y1, DensSums &acc ) {
    if ( ! ref.set ) {
        const SI i = image_col( im, X0 ), j = image_row( im, Y0 ), o = j * ( im.nx + 1 ) + i;
        ref.r0 = im.s0[ o ];
        if constexpr ( MOM ) { ref.r1 = im.s1[ o ]; ref.r2 = im.s2[ o ]; }
        ref.set = true;
    }
    const double dX = X1 - X0, dY = Y1 - Y0;
    const double L = sqrt( dX * dX + dY * dY );
    if ( ! ( L > 0 ) )
        return 0;
    const double INF = 1e300;
    const double u0 = X0 / im.hx, w0 = Y0 / im.hy, du = dX / im.hx, dw = dY / im.hy;
    SI i = image_col( im, X0 ), j = image_row( im, Y0 );
    if ( du == 0 && dw > 0 && u0 == floor( u0 ) ) { i = SI( u0 ) - 1; i = i < 0 ? 0 : ( i >= im.nx ? im.nx - 1 : i ); }
    if ( dw == 0 && du < 0 && w0 == floor( w0 ) ) { j = SI( w0 ) - 1; j = j < 0 ? 0 : ( j >= im.ny ? im.ny - 1 : j ); }
    const SI si = du > 0 ? 1 : -1, sj = dw > 0 ? 1 : -1;
    double tx = du == 0 ? INF : ( double( du > 0 ? i + 1 : i ) - u0 ) / du;
    double ty = dw == 0 ? INF : ( double( dw > 0 ? j + 1 : j ) - w0 ) / dw;
    const double ax = du == 0 ? INF : fabs( 1 / du ), ay = dw == 0 ? INF : fabs( 1 / dw );
    tx = tx < 0 ? 0 : tx;
    ty = ty < 0 ? 0 : ty;
    double tp = 0, Xp = X0, line = 0;
    const SI guard = im.nx + im.ny + 4;
    for ( SI g = 0; g < guard; ++g ) {
        const bool by_x = tx < ty;
        double tn = by_x ? tx : ty;
        const bool last = ! ( tn < 1 );
        if ( last ) tn = 1;
        const double Xc = X0 + dX * tn;
        const SI o = j * ( im.nx + 1 ) + i;
        const double v = im.v[ j * im.nx + i ], dt = tn - tp, bx = double( i ) * im.hx;
        acc.m += dt * dY * ( im.s0[ o ] - ref.r0 + v * 0.5 * ( ( Xp - bx ) + ( Xc - bx ) ) );
        line += dt * L * v;
        if constexpr ( MOM ) {
            const double g1 = 0.21132486540518711775, g2 = 0.78867513459481288225;
            const double dYp = dt * dY;
            for ( int q = 0; q < 2; ++q ) {
                const double tau = tp + dt * ( q ? g2 : g1 );
                const double X = X0 + dX * tau, Y = Y0 + dY * tau, xi = X - bx;
                const double F0 = im.s0[ o ] - ref.r0 + v * xi;
                const double F1 = im.s1[ o ] - ref.r1 + v * xi * 0.5 * ( X + bx );
                const double F2 = im.s2[ o ] - ref.r2 + v * xi * ( X * X + X * bx + bx * bx ) / 3;
                acc.mx += 0.5 * dYp * F1;
                acc.my += 0.5 * dYp * Y * F0;
                acc.m2 += 0.5 * dYp * ( F2 + Y * Y * F0 );
            }
        }
        if ( last )
            break;
        if ( by_x ) { i += si; i = i < 0 ? 0 : ( i >= im.nx ? im.nx - 1 : i ); tx += ax; }
        else        { j += sj; j = j < 0 ? 0 : ( j >= im.ny ? im.ny - 1 : j ); ty += ay; }
        tp = tn;
        Xp = Xc;
    }
    return line;
}

/// THE INTEGRALS OF A CELL of seed `( sa, sb )`: `edges( f )` visits its edges `f( e, ax, ay, bx, by )` in order ( seed's frame,
/// counterclockwise ), `face( e, int rho ds )` gets each edge's line integral; returns the mass and, for `MOM`, the moments about
/// the seed ( `mx`, `my`, `m2` )
template<bool MOM,class Edges,class Face>
__device__ __forceinline__ DensSums cell_sums( const GUnit &, double, double, const Edges &edges, Face &&face ) {
    DensSums r;
    edges( [&]( int e, double x0, double y0, double x1, double y1 ) {
        const double c = x0 * y1 - x1 * y0;
        r.m += c / 2;
        if constexpr ( MOM ) {
            r.mx += c * ( x0 + x1 ) / 6;
            r.my += c * ( y0 + y1 ) / 6;
            r.m2 += c * ( x0 * x0 + x0 * x1 + x1 * x1 + y0 * y0 + y0 * y1 + y1 * y1 ) / 12;
        }
        face( e, sqrt( ( x1 - x0 ) * ( x1 - x0 ) + ( y1 - y0 ) * ( y1 - y0 ) ) );
    } );
    return r;
}

template<bool MOM,class Edges,class Face>
__device__ __forceinline__ DensSums cell_sums( const GImage &d, double sa, double sb, const Edges &edges, Face &&face ) {
    DensSums acc;
    ImageRefs ref;
    double a2 = 0;
    const double Ox = sa - d.im.x0, Oy = sb - d.im.y0;
    edges( [&]( int e, double ax, double ay, double bx, double by ) {
        a2 += ax * by - bx * ay;
        face( e, grid_image_edge<MOM>( d.im, ref, Ox + ax, Oy + ay, Ox + bx, Oy + by, acc ) );
    } );
    return image_finish( d.im, acc, a2 < 0 ? -1.0 : 1.0, sa, sb );
}

template<bool MOM,class Edges,class Face>
__device__ __forceinline__ DensSums cell_sums( const GImage1 &d, double sa, double sb, const Edges &edges, Face &&face ) {
    DensSums acc;
    ImageRefs ref;
    double a2 = 0;
    const double Ox = sa - d.im.x0, Oy = sb - d.im.y0;
    edges( [&]( int e, double ax, double ay, double bx, double by ) {
        a2 += ax * by - bx * ay;
        face( e, image1_edge<MOM>( d.im, ref, Ox + ax, Oy + ay, Ox + bx, Oy + by, acc ) );
    } );
    // about the image's corner -> about the seed ( `image_finish` )
    const double sg = a2 < 0 ? -1.0 : 1.0, mx = sg * acc.mx, my = sg * acc.my, m2 = sg * acc.m2;
    DensSums r;
    r.m = sg * acc.m;
    r.mx = mx - Ox * r.m;
    r.my = my - Oy * r.m;
    r.m2 = m2 - 2 * ( Ox * mx + Oy * my ) + ( Ox * Ox + Oy * Oy ) * r.m;
    return r;
}

template<bool MOM,class Edges,class Face>
__device__ __forceinline__ DensSums cell_sums( const GGauss &d, double sa, double sb, const Edges &edges, Face &&face ) {
    DensSums acc;
    double a2 = 0;
    edges( [&]( int e, double ax, double ay, double bx, double by ) {
        a2 += ax * by - bx * ay;
        face( e, gauss_lines<MOM>( d.g, sa, sb, ax, ay, bx, by, acc ) );
    } );
    const double sg = a2 < 0 ? -1.0 : 1.0;
    DensSums r;
    r.mx = sg * acc.mx; r.my = sg * acc.my; r.m2 = sg * acc.m2;
    gauss_cell<MOM>( d.g, sa, sb, [&]( auto &&f ) { edges( [&]( int, double ax, double ay, double bx, double by ) { f( ax, ay, bx, by ); } ); }, r );
    return r;
}

/// the ring-1 polygon of cell i from the planes: vertex e = plane vertex e - 1, so that edge e is direction e
struct Ring1Poly {
    double px[ 8 ], py[ 8 ];
    signed char ow[ 8 ];
    __device__ __forceinline__ void load( SI nn, SI i, const double *vx, const double *vy, const signed char *ow1 ) {
#pragma unroll
        for ( int e = 0; e < 8; ++e ) {
            px[ e ] = vx[ ( ( e + 7 ) % 8 ) * nn + i ];
            py[ e ] = vy[ ( ( e + 7 ) % 8 ) * nn + i ];
            ow[ e ] = ow1[ e * nn + i ];
        }
    }
    template<class F>
    __device__ __forceinline__ void operator()( F &&f ) const {
#pragma unroll
        for ( int e = 0; e < 8; ++e ) f( e, px[ e ], py[ e ], px[ ( e + 1 ) % 8 ], py[ ( e + 1 ) % 8 ] );
    }
};

/// a polygon of `n` vertices in local memory ( the flagged cells )
struct AnyPoly {
    const double *px, *py;
    int n;
    template<class F>
    __device__ __forceinline__ void operator()( F &&f ) const {
        for ( int e = 0; e < n; ++e ) { const int g = e + 1 == n ? 0 : e + 1; f( e, px[ e ], py[ e ], px[ g ], py[ g ] ); }
    }
};

/// the faces out of ring 1 ( ring-2 and exact cells ), `( i, j, w )`
struct Extras {
    int    *i = nullptr, *j = nullptr;
    double *w = nullptr;
    SI      cap = 0;
};

/// THE RING-1 CELLS ( not flagged ): the mass, the weights toward the 8 neighbours ( `L / ( 2 |k| )` )
template<class Dn>
__global__ void __launch_bounds__( BLOCK ) integrate1_kernel( int N, Dn dens, const double *vx, const double *vy, const signed char *ow,
                                                              const unsigned char *flag1, double *mass, double *W ) {
    const SI nn = SI( N ) * N, i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= nn || flag1[ i ] )
        return;
    constexpr Tab<1> T = make_tab<1>();
    Ring1Poly poly;
    poly.load( nn, i, vx, vy, ow );
    double w[ 8 ];
    const DensSums r = cell_sums<false>( dens, double( i / N ), double( i % N ), poly, [&]( int e, double line ) { w[ e ] = line; } );
    mass[ i ] = r.m;
#pragma unroll
    for ( int k = 0; k < 8; ++k ) W[ k * nn + i ] = poly.ow[ k ] >= 0 ? w[ k ] * 0.5 * T.dir_inv_norm[ k ] : 0.0;
}

/// THE FLAGGED CELLS ( ring 2 or exact ): the mass, the weights toward the ring-1 neighbours in the planes, the others to `extras`
template<class Dn>
__global__ void __launch_bounds__( BLOCK ) integrate_flagged_kernel( int N, Dn dens, const int *list1, SI n1, const unsigned char *flag2, const int *pos2,
                                                                     const int *expos, Planes2 p2, ExactPolys ep, double *mass, double *W, Extras ex, Counts *cnt ) {
    const SI p = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( p >= n1 )
        return;
    const SI nn = SI( N ) * N;
    const int i = list1[ p ], a = i / N, b = i % N;
    double px[ C ], py[ C ];
    int nbr[ C ];
    const int n = flagged_polygon( N, i, flag2, pos2, expos, p2, ep, px, py, nbr );
    for ( int k = 0; k < 8; ++k ) W[ k * nn + i ] = 0;
    const DensSums r = cell_sums<false>( dens, double( a ), double( b ), AnyPoly{ px, py, n }, [&]( int e, double line ) {
        const int j = nbr[ e ];
        if ( j < 0 || line == 0 )
            return;
        const int da = j / N - a, db = j % N - b;
        const double w = line / ( 2 * sqrt( double( da * da + db * db ) ) );
        const int k = ( da >= -1 && da <= 1 && db >= -1 && db <= 1 ) ? c_tab1.dir1[ ( da + 1 ) * 3 + db + 1 ] : -1;
        if ( k >= 0 ) {
            W[ k * nn + i ] = w;
            return;
        }
        const unsigned q = atomicAdd( &cnt->nx, 1u );
        if ( q < ex.cap ) { ex.i[ q ] = i; ex.j[ q ] = j; ex.w[ q ] = w; }
    } );
    mass[ i ] = r.m;
}

/// THE SAME WEIGHT SEEN FROM BOTH SIDES ( `Evaluator::symmetrize` ): the mean of the two, for the densities whose faces are not
/// exact ( a face on the edge of a pixel )
__global__ void __launch_bounds__( BLOCK ) symmetrize_kernel( int N, double *W ) {
    const SI nn = SI( N ) * N, i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= nn )
        return;
    // ( direction k + 4 is the opposite of k: each pair ( i, i + off_k ), k < 4, is visited once )
    for ( int k = 0; k < 4; ++k ) {
        const int a2 = int( i / N ) + c_tab1.off_x[ k ], b2 = int( i % N ) + c_tab1.off_y[ k ];
        if ( a2 < 0 || a2 >= N || b2 < 0 || b2 >= N )
            continue;
        const SI j = SI( a2 ) * N + b2;
        double &wi = W[ k * nn + i ], &wj = W[ ( k + 4 ) * nn + j ];
        wi = wj = ( wi + wj ) / 2;
    }
}

/// the moments of the cells ( relative to the seed, index units ): `bx`, `by` the barycentre, `m2` the second moment about the seed
template<class Dn>
__global__ void __launch_bounds__( BLOCK ) moments1_kernel( int N, Dn dens, const double *vx, const double *vy, const signed char *ow,
                                                            const unsigned char *flag1, double *bx, double *by, double *m2 ) {
    const SI nn = SI( N ) * N, i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= nn || flag1[ i ] )
        return;
    Ring1Poly poly;
    poly.load( nn, i, vx, vy, ow );
    const DensSums r = cell_sums<true>( dens, double( i / N ), double( i % N ), poly, []( int, double ) {} );
    bx[ i ] = r.m > 0 ? r.mx / r.m : 0;
    by[ i ] = r.m > 0 ? r.my / r.m : 0;
    m2[ i ] = r.m2;
}

template<class Dn>
__global__ void __launch_bounds__( BLOCK ) moments_flagged_kernel( int N, Dn dens, const int *list1, SI n1, const unsigned char *flag2, const int *pos2,
                                                                   const int *expos, Planes2 p2, ExactPolys ep, double *bx, double *by, double *m2 ) {
    const SI p = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( p >= n1 )
        return;
    const int i = list1[ p ];
    double px[ C ], py[ C ];
    int nbr[ C ];
    const int n = flagged_polygon( N, i, flag2, pos2, expos, p2, ep, px, py, nbr );
    const DensSums r = cell_sums<true>( dens, double( i / N ), double( i % N ), AnyPoly{ px, py, n }, []( int, double ) {} );
    bx[ i ] = r.m > 0 ? r.mx / r.m : 0;
    by[ i ] = r.m > 0 ? r.my / r.m : 0;
    m2[ i ] = r.m2;
}

// ================================================================================================
// the image of a level: its prefix sums ( `DensityHost2D.cuh::image_prefix`, the input read where it is, C order x the slowest )
// ================================================================================================
template<class TF>
__global__ void __launch_bounds__( BLOCK ) grid_image_values( SI nx, SI ny, Strided<TF,1> prm, SI first, double scale, double *v ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q < nx * ny ) {
        const SI i = q / ny, j = q % ny;                 // ( the input: x the slowest; `DensImage`: rows of y )
        v[ j * nx + i ] = scale * double( prm( first + q ) );
    }
}

template<class TF>
__global__ void __launch_bounds__( BLOCK ) grid_scaled_values( SI n, Strided<TF,1> prm, SI first, double scale, double *v ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q < n ) v[ q ] = scale * double( prm( first + q ) );
}

__global__ void __launch_bounds__( BLOCK ) grid_image_prefix( SI nx, SI ny, double hx, const double *v, double *s0, double *s1, double *s2 ) {
    const SI j = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( j >= ny )
        return;
    double a0 = 0, a1 = 0, a2 = 0;
    const double h2 = hx * hx, h3 = h2 * hx;
    for ( SI i = 0; i < nx; ++i ) {
        const SI o = j * ( nx + 1 ) + i;
        const double vv = v[ j * nx + i ], di = double( i );
        s0[ o ] = a0; s1[ o ] = a1; s2[ o ] = a2;
        a0 += vv * hx;
        a1 += vv * h2 * ( 2 * di + 1 ) * 0.5;
        a2 += vv * h3 * ( 3 * di * di + 3 * di + 1 ) / 3;
    }
    const SI o = j * ( nx + 1 ) + nx;
    s0[ o ] = a0; s1[ o ] = a1; s2[ o ] = a2;
}

// ================================================================================================
// the preconditioner: the weighted semi-coarsened tent blurs ( `GridNewton2D.h::TensorPrecond` )
// ================================================================================================
/// the ( la, lb ) grids of a level, `( n >> la ) x ( n >> lb )` each, la-major in one buffer
struct TreeLayout {
    int n = 0, L = 0;
    SI  off[ MAXL + 2 ][ MAXL + 1 ] = {};                ///< `off[ L + 1 ][ 0 ]`: the total
    SI  rows = 0;                                        ///< the rows of the ( la, 0 ) grids, all la

    static TreeLayout of( int n ) {
        TreeLayout t;
        t.n = n;
        while ( ( n >> ( t.L + 1 ) ) >= 2 ) ++t.L;
        SI o = 0;
        for ( int la = 0; la <= t.L; ++la ) {
            for ( int lb = 0; lb <= t.L; ++lb ) { t.off[ la ][ lb ] = o; o += SI( n >> la ) * ( n >> lb ); }
            t.rows += n >> la;
        }
        t.off[ t.L + 1 ][ 0 ] = o;
        return t;
    }
    __host__ __device__ SI total() const { return off[ L + 1 ][ 0 ]; }
    static SI total_for( int n ) { return of( n ).total(); }
};

/// t( B ) of R f ( a tent restriction to m values )
__device__ __forceinline__ double restrict_at( const double *f, int m, int B, SI stride ) {
    const double w0 = B == 0 ? 1.0 : 0.75, w1 = B == m - 1 ? 1.0 : 0.75, wm = B >= 1 ? 0.25 : 0.0, wp = B <= m - 2 ? 0.25 : 0.0;
    double r = w0 * f[ ( 2 * B ) * stride ] + w1 * f[ ( 2 * B + 1 ) * stride ];
    if ( B >= 1 ) r += wm * f[ ( 2 * B - 1 ) * stride ];
    if ( B <= m - 2 ) r += wp * f[ ( 2 * B + 2 ) * stride ];
    return r;
}

/// ( P t )( X ), t of m values, X < 2 m
__device__ __forceinline__ double prolong_at( const double *t, int m, int X, SI stride ) {
    const int B = X / 2;
    if ( X % 2 == 0 ) return ( B == 0 ? 1.0 : 0.75 ) * t[ B * stride ] + ( B == 0 ? 0.0 : 0.25 * t[ ( B - 1 ) * stride ] );
    return ( B == m - 1 ? 1.0 : 0.75 ) * t[ B * stride ] + ( B == m - 1 ? 0.0 : 0.25 * t[ ( B + 1 ) * stride ] );
}

/// ( la, 0 ) from ( la - 1, 0 ): along the first axis
__global__ void __launch_bounds__( BLOCK ) restrict_a_kernel( TreeLayout t, double *g, int la ) {
    const int m = t.n >> la, nb = t.n;
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q >= SI( m ) * nb )
        return;
    const int A = int( q / nb ), b = int( q % nb );
    g[ t.off[ la ][ 0 ] + q ] = restrict_at( g + t.off[ la - 1 ][ 0 ] + b, m, A, nb );
}

/// ( la - 1, 0 ) += P ( la, 0 )
__global__ void __launch_bounds__( BLOCK ) prolong_a_kernel( TreeLayout t, double *g, int la ) {
    const int m = t.n >> la, nb = t.n;
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q >= SI( 2 * m ) * nb )
        return;
    const int X = int( q / nb ), b = int( q % nb );
    g[ t.off[ la - 1 ][ 0 ] + q ] += prolong_at( g + t.off[ la ][ 0 ] + b, m, X, nb );
}

/// the row r of all the ( la, 0 ) grids: `la`, and its row `a` there
__device__ __forceinline__ void row_of( const TreeLayout &t, SI r, int &la, int &a ) {
    la = 0;
    while ( r >= SI( t.n >> la ) ) { r -= t.n >> la; ++la; }
    a = int( r );
}

/// ONE BLOCK PER ROW of the ( la, 0 ) grids: the restrictions along the second axis ( la, 1 ), ..., ( la, L ) of that row, then,
/// given `D`, every grid of the row but ( 0, 0 ) scaled
__global__ void __launch_bounds__( BLOCK ) pyramid_kernel( TreeLayout t, double *g, const double *D ) {
    int la, a;
    row_of( t, blockIdx.x, la, a );
    for ( int lb = 1; lb <= t.L; ++lb ) {
        const int nb = t.n >> ( lb - 1 ), m = nb / 2;
        const double *f = g + t.off[ la ][ lb - 1 ] + SI( a ) * nb;
        double *c = g + t.off[ la ][ lb ] + SI( a ) * m;
        for ( int B = threadIdx.x; B < m; B += blockDim.x ) c[ B ] = restrict_at( f, m, B, 1 );
        __syncthreads();
    }
    if ( D )
        for ( int lb = la ? 0 : 1; lb <= t.L; ++lb ) {
            const int m = t.n >> lb;
            const SI o = t.off[ la ][ lb ] + SI( a ) * m;
            for ( int B = threadIdx.x; B < m; B += blockDim.x ) g[ o + B ] *= D[ o + B ];
        }
}

/// ONE BLOCK PER ROW of the ( la, 0 ) grids: ( la, lb - 1 ) += P ( la, lb ) from lb = L down, into ( la, 0 ) -- into ( 0, 1 ) for
/// la = 0 ( ( 0, 0 ) is the residual itself: `final_kernel` adds the last step to z )
__global__ void __launch_bounds__( BLOCK ) prolong_b_kernel( TreeLayout t, double *g ) {
    int la, a;
    row_of( t, blockIdx.x, la, a );
    for ( int lb = t.L; lb >= ( la ? 1 : 2 ); --lb ) {
        const int m = t.n >> lb;
        const double *c = g + t.off[ la ][ lb ] + SI( a ) * m;
        double *f = g + t.off[ la ][ lb - 1 ] + SI( a ) * 2 * m;
        for ( int X = threadIdx.x; X < 2 * m; X += blockDim.x ) f[ X ] += prolong_at( c, m, X, 1 );
        __syncthreads();
    }
}

/// z += P_b ( 0, 1 ) + P_a ( 1, 0 )
__global__ void __launch_bounds__( BLOCK ) final_kernel( TreeLayout t, const double *g, double *z ) {
    const int n = t.n, m = n / 2;
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q >= SI( n ) * n )
        return;
    const int a = int( q / n ), b = int( q % n );
    z[ q ] += prolong_at( g + t.off[ 0 ][ 1 ] + SI( a ) * m, m, b, 1 ) + prolong_at( g + t.off[ 1 ][ 0 ] + b, m, a, n );
}

/// the conductances of the faces: `d0` ( the diagonal ), `Da`, `Db` ( weighted by the offsets along each axis )
__global__ void __launch_bounds__( BLOCK ) conductances_kernel( int N, const double *W, double *d0, double *Da, double *Db ) {
    const SI nn = SI( N ) * N, i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= nn )
        return;
    constexpr Tab<1> T = make_tab<1>();
    double d = 0, da = 0, db = 0;
#pragma unroll
    for ( int k = 0; k < 8; ++k ) {
        const double w = W[ k * nn + i ];
        d += w;
        da += T.off_x[ k ] ? w : 0.0;
        db += T.off_y[ k ] ? w : 0.0;
    }
    d0[ i ] = d; Da[ i ] = da; Db[ i ] = db;
}

/// the faces out of ring 1, by row of the operator: `rows[ r ]`, its entries `[ beg[ r ], beg[ r + 1 ] )` ( `cols`, `w` )
struct ExRows {
    const int    *rows = nullptr, *beg = nullptr, *cols = nullptr;
    const double *w = nullptr;
    SI            nr = 0;
};

__global__ void __launch_bounds__( BLOCK ) ex_conductances_kernel( int N, ExRows ex, double *d0, double *Da, double *Db ) {
    const SI r = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( r >= ex.nr )
        return;
    const int i = ex.rows[ r ];
    double d = 0, da = 0, db = 0;
    for ( int e = ex.beg[ r ]; e < ex.beg[ r + 1 ]; ++e ) {
        const int j = ex.cols[ e ];
        d += ex.w[ e ];
        da += abs( j / N - i / N ) * ex.w[ e ];
        db += abs( j % N - i % N ) * ex.w[ e ];
    }
    d0[ i ] += d; Da[ i ] += da; Db[ i ] += db;
}

__global__ void __launch_bounds__( BLOCK ) inverse_kernel( SI n, const double *d0, double *id0 ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) id0[ i ] = d0[ i ] > 0 ? 1 / d0[ i ] : 0;
}

/// the scaling of each grid but ( 0, 0 ): `balance( Ea, Eb ) / ( Ea + Eb )`, `Ea = c R ka / 4^la`, `Eb = c R kb / 4^lb`, the balance
/// `sqrt( min / max )` ( `TensorPrecond::build`, `pw = 1/2`, every subspace kept )
__global__ void __launch_bounds__( BLOCK ) scaling_kernel( TreeLayout t, const double *ta, const double *tb, double *D ) {
    const SI q = t.off[ 0 ][ 1 ] + SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q >= t.total() )
        return;
    int la = 0;
    while ( q >= t.off[ la + 1 ][ 0 ] ) ++la;
    int lb = t.L;
    while ( q < t.off[ la ][ lb ] ) --lb;
    const double c = 2.0 / 3.0, ea = c / double( 1 << la ) / double( 1 << la ) * ta[ q ], eb = c / double( 1 << lb ) / double( 1 << lb ) * tb[ q ];
    const double d = ea + eb, lo = fmin( ea, eb ), hi = fmax( ea, eb );
    D[ q ] = d > 0 ? ( hi == 0 ? 0.0 : sqrt( lo / hi ) ) / d : 0.0;
}

// ================================================================================================
// CG on the card
// ================================================================================================
/// the scalars of the iteration, kept on the card ( the host reads `rr` only )
struct CgScalars {
    double rz, beta, s, alpha;
};

/// x = 0, r = b, p = 0, z = r / diag, g( 0, 0 ) = r; `| b |^2`
struct CgInitFn {
    const double *b, *id0;
    double *x, *r, *z, *p, *g00;
    __device__ void operator()( SI i, Sum1 &acc ) const {
        const double v = b[ i ];
        x[ i ] = 0; r[ i ] = v; p[ i ] = 0; z[ i ] = v * id0[ i ]; g00[ i ] = v;
        acc.s += v * v;
    }
};

/// q = H p ( the ring-1 planes ), `p . q`
struct MatvecFn {
    int N;
    const double *W, *p;
    double *q;
    __device__ void operator()( SI i, Sum1 &acc ) const {
        constexpr Tab<1> T = make_tab<1>();
        const SI nn = SI( N ) * N;
        const int a = int( i / N ), b = int( i % N );
        const double vi = p[ i ];
        double s = 0;
#pragma unroll
        for ( int k = 0; k < 8; ++k ) {
            const int a2 = a + T.off_x[ k ], b2 = b + T.off_y[ k ];
            const SI j = ( a2 < 0 || a2 >= N || b2 < 0 || b2 >= N ) ? i : SI( a2 ) * N + b2;
            s += W[ k * nn + i ] * ( vi - p[ j ] );
        }
        q[ i ] = s;
        acc.s += vi * s;
    }
};

/// q += the faces out of ring 1, `p . ( that part )`
struct MatvecExFn {
    ExRows ex;
    const double *p;
    double *q;
    __device__ void operator()( SI r, Sum1 &acc ) const {
        const int i = ex.rows[ r ];
        double d = 0;
        for ( int e = ex.beg[ r ]; e < ex.beg[ r + 1 ]; ++e ) d += ex.w[ e ] * ( p[ i ] - p[ ex.cols[ e ] ] );
        q[ i ] += d;
        acc.s += p[ i ] * d;
    }
};

/// x += alpha p, r -= alpha q, z = r / diag, g( 0, 0 ) = r; `| r |^2`
struct CgUpdateFn {
    const CgScalars *cs;
    const double *p, *q, *id0;
    double *x, *r, *z, *g00;
    __device__ void operator()( SI i, Sum1 &acc ) const {
        const double al = cs->alpha;
        x[ i ] += al * p[ i ];
        const double ri = r[ i ] - al * q[ i ];
        r[ i ] = ri; z[ i ] = ri * id0[ i ]; g00[ i ] = ri;
        acc.s += ri * ri;
    }
};

/// `sum z`, `r . z`, `sum r` ( the mean projection folded in )
struct CgPassCFn {
    const double *r, *z;
    __device__ void operator()( SI i, Sum3 &acc ) const { acc.s0 += z[ i ]; acc.s1 += r[ i ] * z[ i ]; acc.s2 += r[ i ]; }
};

__global__ void cg_alpha_kernel( CgScalars *cs, const Sum1 *pq, const Sum1 *pq_ex ) {
    cs->alpha = cs->rz / ( pq->s + ( pq_ex ? pq_ex->s : 0.0 ) );
}

__global__ void cg_beta_kernel( CgScalars *cs, const Sum3 *c, double nn, int first ) {
    const double s = c->s0 / nn, rz2 = c->s1 - s * c->s2;
    cs->s = s;
    cs->beta = first ? 0.0 : rz2 / cs->rz;
    cs->rz = rz2;
}

__global__ void __launch_bounds__( BLOCK ) cg_direction_kernel( SI n, const CgScalars *cs, const double *z, double *p ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) p[ i ] = z[ i ] - cs->s + cs->beta * p[ i ];
}

// ================================================================================================
// small kernels of Newton
// ================================================================================================
/// the residual of the masses: the largest relative and absolute errors, the L2 norm ( squared ), the smallest mass
struct Err {
    double rel, abs, l2, mmin;
    __host__ __device__ static Err identity() { return { 0.0, 0.0, 0.0, 1e300 }; }
    __device__ void combine( const Err &o ) { rel = fmax( rel, o.rel ); abs = fmax( abs, o.abs ); l2 += o.l2; mmin = fmin( mmin, o.mmin ); }
};

struct ErrFn {
    const double *m, *t;
    __device__ void operator()( SI i, Err &e ) const {
        const double d = fabs( m[ i ] - t[ i ] );
        e.abs = fmax( e.abs, d );
        e.rel = fmax( e.rel, t[ i ] > 0 ? d / t[ i ] : d > 0 ? 1e300 : 0.0 );
        e.l2 += d * d;
        e.mmin = fmin( e.mmin, m[ i ] );
    }
};

/// a maximum, from -inf
struct MaxV {
    double m;
    __host__ __device__ static MaxV identity() { return { -1e308 }; }
    __device__ void combine( const MaxV &o ) { m = fmax( m, o.m ); }
};
struct MaxOfFn {
    const double *x;
    __device__ void operator()( SI i, MaxV &acc ) const { acc.m = fmax( acc.m, x[ i ] ); }
};

struct MinOfFn {
    const double *x;
    __device__ void operator()( SI i, Min1 &acc ) const { acc.m = fmin( acc.m, x[ i ] ); }
};

__global__ void copy_max( const MaxV *m, double *out ) { *out = m->m; }

template<class TF>
__global__ void __launch_bounds__( BLOCK ) write_values( SI n, const double *src, StridedOut<TF,1> dst ) {
    const SI k = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( k < n ) dst( k ) = TF( src[ k ] );
}

struct Sum2Fn {
    const double *x, *y;
    __device__ void operator()( SI i, Sum2 &acc ) const { acc.s0 += x[ i ]; acc.s1 += y[ i ]; }
};

/// b = t - m, `sum b`
struct RhsFn {
    const double *t, *m;
    double *b;
    __device__ void operator()( SI i, Sum1 &acc ) const { const double v = t[ i ] - m[ i ]; b[ i ] = v; acc.s += v; }
};

__global__ void __launch_bounds__( BLOCK ) sub_mean_kernel( SI n, double *b, const Sum1 *s ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) b[ i ] -= s->s / double( n );
}

/// the cells that a trial left under the floor, in any order ( nothing computed from them depends on it )
__global__ void __launch_bounds__( BLOCK ) under_floor_kernel( SI n, const double *mass, double eps0, int *bad, Counts *cnt ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n && mass[ i ] <= eps0 )
        bad[ atomicAdd( &cnt->nbad, 1u ) ] = int( i );
}

/// the neighbours of the cells `bad` in the diagram that is there ( `C` places each ); `active`: those with a polygon
__global__ void __launch_bounds__( BLOCK ) frozen_faces_kernel( int N, const int *bad, SI nb, const unsigned char *flag1, const unsigned char *flag2, const signed char *ow1,
                                                                const int *pos2, const int *expos, Planes2 p2, ExactPolys ep, int *faces, int *nfaces, unsigned char *active ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q >= nb )
        return;
    const SI nn = SI( N ) * N;
    const int i = bad[ q ];
    int m = 0;
    bool any = true;
    if ( flag1[ i ] ) {
        double px[ C ], py[ C ];
        int nbr[ C ];
        const int n = flagged_polygon( N, i, flag2, pos2, expos, p2, ep, px, py, nbr );
        any = n > 0;
        for ( int e = 0; e < n; ++e ) if ( nbr[ e ] >= 0 ) faces[ q * C + m++ ] = nbr[ e ];
    } else
        for ( int e = 0; e < 8; ++e ) {
            const int o = ow1[ e * nn + i ];
            if ( o >= 0 ) faces[ q * C + m++ ] = i + c_tab1.off_x[ o ] * N + c_tab1.off_y[ o ];
        }
    nfaces[ q ] = m;
    active[ q ] = any;
}

/// THE MASSES OF THE CELLS `bad` AT `psi + s x`, THEIR EDGES FROZEN ( `Evaluator::frozen_mass` ): `below` those of `active` that fall under the floor
template<class Dn>
__global__ void __launch_bounds__( BLOCK ) frozen_mass_kernel( int N, Dn dens, const int *bad, SI nb, const int *faces, const int *nfaces, const double *psi, const double *x,
                                                               double s, double eps0, const unsigned char *active, unsigned char *below, Counts *cnt ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q >= nb )
        return;
    below[ q ] = 0;
    if ( ! active[ q ] )
        return;
    const int i = bad[ q ], a = i / N, b = i % N;
    double px[ C ], py[ C ], qx[ C ], qy[ C ];
    px[ 0 ] = -a - 0.5;    py[ 0 ] = -b - 0.5;
    px[ 1 ] = N - 0.5 - a; py[ 1 ] = -b - 0.5;
    px[ 2 ] = N - 0.5 - a; py[ 2 ] = N - 0.5 - b;
    px[ 3 ] = -a - 0.5;    py[ 3 ] = N - 0.5 - b;
    int n = 4;
    const double pi = psi[ i ] + s * x[ i ];
    for ( int f = 0; f < nfaces[ q ] && n; ++f ) {
        const int j = faces[ q * C + f ];
        const double kx = j / N - a, ky = j % N - b, c = ( kx * kx + ky * ky + pi - psi[ j ] - s * x[ j ] ) / 2;
        int m = 0;
        for ( int v = 0; v < n && m + 2 < C; ++v ) {
            const int w = v + 1 == n ? 0 : v + 1;
            const double s0 = kx * px[ v ] + ky * py[ v ] - c, s1 = kx * px[ w ] + ky * py[ w ] - c;
            if ( s0 <= 0 ) { qx[ m ] = px[ v ]; qy[ m ] = py[ v ]; ++m; }
            if ( ( s0 < 0 && s1 > 0 ) || ( s0 > 0 && s1 < 0 ) ) {
                const double u = s0 / ( s0 - s1 );
                qx[ m ] = px[ v ] + u * ( px[ w ] - px[ v ] ); qy[ m ] = py[ v ] + u * ( py[ w ] - py[ v ] ); ++m;
            }
        }
        for ( int v = 0; v < m; ++v ) { px[ v ] = qx[ v ]; py[ v ] = qy[ v ]; }
        n = m;
    }
    const double mass = n >= 3 ? cell_sums<false>( dens, double( a ), double( b ), AnyPoly{ px, py, n }, [&]( int, double ) {} ).m : 0.0;
    if ( mass <= eps0 ) {
        below[ q ] = 1;
        atomicAdd( &cnt->nfell, 1u );
    }
}

/// the cells at or under the floor
struct FloorCountFn {
    const double *m;
    double eps0;
    __device__ void operator()( SI i, Sum1 &acc ) const { acc.s += m[ i ] <= eps0 ? 1.0 : 0.0; }
};

/// the state of the repair of each cell ( `sdotplan/WarmRepair.h`, in squared spacings: index units )
struct RepairState { double *lo, *hi, *step, *touched; };

__global__ void __launch_bounds__( BLOCK ) repair_init_kernel( SI n, RepairState rs ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n )
        sp::warm_repair_init( rs.lo[ i ], rs.hi[ i ], rs.step[ i ], rs.touched[ i ], 1.0, false );
}

__global__ void __launch_bounds__( BLOCK ) repair_round_kernel( SI n, const double *mass, const double *target, double *ps, RepairState rs, double *moved ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n )
        moved[ i ] = sp::warm_repair_cell( mass[ i ], target[ i ], ps[ i ], rs.lo[ i ], rs.hi[ i ], rs.step[ i ], rs.touched[ i ], 1.0 ) ? 1.0 : 0.0;
}

/// the cells under `LO` of their target
struct SmallFn {
    const double *m, *t;
    __device__ void operator()( SI i, Sum1 &acc ) const { acc.s += m[ i ] < sp::WarmRepair::LO * t[ i ] ? 1.0 : 0.0; }
};

__global__ void __launch_bounds__( BLOCK ) axpy_kernel( SI n, const double *x, double t, const double *d, double *y ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) y[ i ] = x[ i ] + t * d[ i ];
}

__global__ void __launch_bounds__( BLOCK ) scale_kernel( SI n, const double *x, double f, double *y ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) y[ i ] = f * x[ i ];
}

/// the masses of the next coarser level: the sums of the 2 x 2 blocks
__global__ void __launch_bounds__( BLOCK ) coarsen_kernel( int n, const double *f, double *c ) {
    const int m = n / 2;
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q >= SI( m ) * m )
        return;
    const SI a = q / m, b = q % m;
    c[ q ] = f[ ( 2 * a ) * n + 2 * b ] + f[ ( 2 * a ) * n + 2 * b + 1 ] + f[ ( 2 * a + 1 ) * n + 2 * b ] + f[ ( 2 * a + 1 ) * n + 2 * b + 1 ];
}

/// psi on the n x n grid -> psi on the 2n x 2n grid, separable cubic Lagrange ( `refine_psi` ): `x_f = 2 x_c + 1/2`, `psi_f = 4 psi_c`
__device__ __forceinline__ void cubic_weights( int n, int a, int &j0, double ( &w )[ 4 ] ) {
    const double u = ( a - 0.5 ) / 2;
    j0 = int( floor( u ) ) - 1;
    j0 = j0 < 0 ? 0 : ( j0 > n - 4 ? n - 4 : j0 );
#pragma unroll
    for ( int k = 0; k < 4; ++k ) {
        double l = 1;
#pragma unroll
        for ( int m = 0; m < 4; ++m ) if ( m != k ) l *= ( u - j0 - m ) / double( k - m );
        w[ k ] = l;
    }
}

__global__ void __launch_bounds__( BLOCK ) refine_kernel( int n, const double *pc, double *pf ) {
    const int N = 2 * n;
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q >= SI( N ) * N )
        return;
    const int a = int( q / N ), b = int( q % N );
    int ja, jb;
    double wa[ 4 ], wb[ 4 ];
    cubic_weights( n, a, ja, wa );
    cubic_weights( n, b, jb, wb );
    double v = 0;
#pragma unroll
    for ( int l = 0; l < 4; ++l ) {
        double t = 0;
#pragma unroll
        for ( int k = 0; k < 4; ++k ) t += wa[ k ] * pc[ SI( ja + k ) * n + jb + l ];
        v += wb[ l ] * t;
    }
    pf[ q ] = 4 * v;
}

template<class TF>
__global__ void __launch_bounds__( BLOCK ) read_input( SI n, Strided<TF,1> src, double f, double *dst ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i < n ) dst[ i ] = f * double( src( i ) );
}

/// the outputs, physical: the weights ( `w[ 0 ] = 0` ), the masses, the barycentres; the cost's terms into `c`
template<class TF>
__global__ void __launch_bounds__( BLOCK ) outputs_kernel( int N, double ox, double oy, double h, double S, const double *psi, const double *mass,
                                                           const double *bx, const double *by, const double *m2, StridedOut<TF,1> w, StridedOut<TF,1> m,
                                                           StridedOut<TF,2> bary, double *c ) {
    const SI i = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( i >= SI( N ) * N )
        return;
    w( i ) = TF( h * h * ( psi[ i ] - psi[ 0 ] ) );
    m( i ) = TF( mass[ i ] / S );
    bary( i, 0 ) = TF( ox + h * ( double( i / N ) + 0.5 + bx[ i ] ) );
    bary( i, 1 ) = TF( oy + h * ( double( i % N ) + 0.5 + by[ i ] ) );
    c[ i ] = h * h * m2[ i ] / S;
}

template<class TF>
__global__ void write_sum( const Sum1 *s, TF *dst ) { *dst = TF( s->s ); }

// ================================================================================================
// the solver
// ================================================================================================
struct Options {
    double tol_abs = 0, tol_rel = 1e-9;                  ///< stop when max | m - t | <= tol_abs ( mass units ) or max | m - t | / t <= tol_rel
    double ctol = 1e-2;                                  ///< the relative tolerance of the coarse levels
    int    n0 = 64;                                      ///< the coarsest level
    int    max_iter = 50;
    double cg_rtol = 1e-2;
    bool   trace = false;                                ///< the phases' times on stdout
};

struct Stats {
    int status = 0, nb_iter = 0, nb_diag = 0, nb_backtracks = 0, lin_nb_iter = 0, nb_levels = 0;
    double domain_mass = 1;
    double residual = 0, residual0 = -1, t_diag = 0, t_lin = 0, t_total = 0;
    bool exhausted = false;
    bool failed = false;                                 ///< the pool refused memory
};

/// the input density of the solve, physical ( from the call's `dens_prm` ): its kind and what the host needs of it
struct DensityIn {
    int kind = 0;                                        ///< 0 uniform, 1 image, 2 gaussians, 3 image of degree 1
    // the image: `nx x ny` pixels from ( x0, y0 ), of side ( hx, hy ); its values in the input from `first`
    SI nx = 0, ny = 0, first = 0;
    double x0 = 0, y0 = 0, hx = 1, hy = 1;
    // the gaussians: ( cx, cy, sigma, mass ) each
    std::vector<double> g;
    // the image of degree 1: its knots ( physical ), its coefficients in the input from `first`
    std::vector<double> kx, ky;
};

/// THE CARD'S SOLVER of one call: the buffers, taken once for the finest level and used by every level
template<class Alloc>
struct Solver {
    const CudaQueue &queue;
    Alloc &allocator;
    int Nf;                                              ///< the finest level
    SI nnf;
    bool ok = true;

    // the vectors of a level ( n x n, the finest's room )
    double *psi, *trial, *target, *mass, *dx, *rhs, *x, *r, *z, *p, *q, *id0, *d0;
    // geometry
    double *vx1, *vy1;
    signed char *ow1;
    unsigned char *flag1, *flag2;
    unsigned *mark;
    int *pos2, *expos, *list1, *check, *exl;
    Planes2 p2;
    ExactPolys ep;
    double *W;
    Extras xs;
    Counts *cnt;
    Counts hc{};
    // the faces out of ring 1, merged ( `ExRows` ) -- the host's copy and the card's
    ExRows exr;
    int *exr_buf = nullptr;
    double *exr_w = nullptr;
    SI exr_cap = 0;
    // the exact path
    ExactOffsets eo;
    int rb = RB;                                         ///< the rings of `eo`
    double *pmax;
    // the cells under the floor ( `tau_limit` )
    int *bad = nullptr, *faces = nullptr, *nfaces = nullptr;
    unsigned char *active = nullptr, *below = nullptr;
    SI bad_cap = 0;
    // the preconditioner's trees
    double *g, *tb, *Di;
    // the reductions
    void *partials;
    Sum1 *s1a, *s1b;
    Sum2 *s2;
    Sum3 *s3;
    Err *err;
    MaxV *mx;
    Min1 *mn;
    CgScalars *cs;
    // the density of the current level
    int kind = 0;
    GImage gim{};
    GGauss gga{};
    GImage1 gi1{};
    double *i1_c = nullptr, *i1_k = nullptr, *i1_p = nullptr;
    double *img_v = nullptr, *img_s0 = nullptr, *img_s1 = nullptr, *img_s2 = nullptr, *gauss_d = nullptr;
    // timings
    double t_geom = 0, t_int = 0;

    Solver( const CudaQueue &queue, Alloc &allocator, int N ) : queue( queue ), allocator( allocator ), Nf( N ), nnf( SI( N ) * N ) {}

    template<class T>
    T *grab( SI count ) {
        if ( ! ok ) return nullptr;
        const SI bytes = ( SI( sizeof( T ) ) * std::max<SI>( count, 1 ) + 255 ) & ~SI( 255 );
        if ( measuring ) { arena_bytes += bytes; return reinterpret_cast<T *>( std::uintptr_t( 256 ) ); }
        if ( arena ) { T *res = reinterpret_cast<T *>( arena + arena_used ); arena_used += bytes; return res; }
        T *res = static_cast<T *>( take( allocator, bytes ) );
        if ( ! res ) ok = false;
        return res;
    }
    // ONE request to the pool for every buffer of fixed size ( `prepare` runs its requests twice: sized, then carved ) -- XLA's
    // allocator was seen to abort on freeing the fifty pieces of a call ( `bfc_allocator.cc`, once, not reproduced )
    bool measuring = false;
    unsigned char *arena = nullptr;
    SI arena_bytes = 0, arena_used = 0;
    std::vector<double *> nus;                           ///< the masses of the levels, finest first


    bool prepare( int n0, const DensityIn &din ) {
        const SI nn = nnf;
        measuring = true;
        buffers( n0, din );
        measuring = false;
        arena = static_cast<unsigned char *>( take( allocator, arena_bytes ) );
        if ( ! arena ) return ok = false;
        buffers( n0, din );
        arena = nullptr;
        zero_fill( queue, mark, SI( sizeof( unsigned ) ) * nn );
        zero_fill( queue, flag1, nn );
        zero_fill( queue, flag2, nn );
        // the tables of the rare paths, and the exact path's offsets ( sorted by distance, `ExactKernel` )
        const Tab<1> t1 = make_tab<1>();
        const Tab<2> t2 = make_tab<2>();
        cuda_check( cudaMemcpyToSymbolAsync( c_tab1, &t1, sizeof( t1 ), 0, cudaMemcpyHostToDevice, queue.stream ), "copy of the ring-1 tables" );
        cuda_check( cudaMemcpyToSymbolAsync( c_tab2, &t2, sizeof( t2 ), 0, cudaMemcpyHostToDevice, queue.stream ), "copy of the ring-2 tables" );
        load_rings();
        return true;
    }

    /// the exact path's offsets of `rb` rings, sorted by distance ( `ExactKernel` )
    void load_rings() {
        std::vector<int> ox, oy;
        for ( int a = -rb; a <= rb; ++a ) for ( int b = -rb; b <= rb; ++b ) if ( a || b ) { ox.push_back( a ); oy.push_back( b ); }
        std::vector<int> ord( ox.size() );
        for ( size_t i = 0; i < ord.size(); ++i ) ord[ i ] = int( i );
        std::stable_sort( ord.begin(), ord.end(), [&]( int i, int j ) { return ox[ i ] * ox[ i ] + oy[ i ] * oy[ i ] < ox[ j ] * ox[ j ] + oy[ j ] * oy[ j ]; } );
        std::vector<int> sx, sy;
        std::vector<double> nk;
        for ( int i : ord ) { sx.push_back( ox[ i ] ); sy.push_back( oy[ i ] ); nk.push_back( std::hypot( double( ox[ i ] ), double( oy[ i ] ) ) ); }
        upload( const_cast<int *>( eo.ox ), sx.data(), SI( sx.size() ) ); upload( const_cast<int *>( eo.oy ), sy.data(), SI( sy.size() ) );
        upload( const_cast<double *>( eo.nk ), nk.data(), SI( nk.size() ) );
    }

    /// cells that outgrow the rings ( thin parts of the masses: they reach across seeds of almost no mass ): twice as many rings
    bool grow_rings( int n ) {
        rb = std::min( 2 * rb, n );
        const SI ne = SI( 2 * rb + 1 ) * ( 2 * rb + 1 ) - 1;
        eo = ExactOffsets{ grab<int>( ne ), grab<int>( ne ), grab<double>( ne ), int( ne ) };
        if ( ! ok ) return false;
        load_rings();
        return true;
    }

    /// every buffer of fixed size ( twice: `measuring`, then carved from `arena` )
    void buffers( int n0, const DensityIn &din ) {
        const SI nn = nnf;
        p2 = Planes2{}; ep = ExactPolys{}; xs = Extras{};
        for ( double **v : { &psi, &trial, &target, &mass, &dx, &rhs, &x, &r, &z, &p, &q, &id0, &d0 } ) *v = grab<double>( nn );
        vx1 = grab<double>( 8 * nn ); vy1 = grab<double>( 8 * nn ); ow1 = grab<signed char>( 8 * nn );
        W = grab<double>( 8 * nn );
        flag1 = grab<unsigned char>( nn ); flag2 = grab<unsigned char>( nn ); mark = grab<unsigned>( nn );
        pos2 = grab<int>( nn ); expos = grab<int>( nn ); list1 = grab<int>( nn ); check = grab<int>( nn ); exl = grab<int>( nn );
        const SI tot = TreeLayout::total_for( Nf );
        g = grab<double>( tot ); tb = grab<double>( tot ); Di = grab<double>( tot );
        partials = grab<unsigned char>( SI( RED_GRID ) * 64 );
        s1a = grab<Sum1>( 1 ); s1b = grab<Sum1>( 1 ); s2 = grab<Sum2>( 1 ); s3 = grab<Sum3>( 1 ); err = grab<Err>( 1 ); mx = grab<MaxV>( 1 ); mn = grab<Min1>( 1 );
        cs = grab<CgScalars>( 1 ); cnt = grab<Counts>( 1 ); pmax = grab<double>( 1 );
        const SI ne = SI( 2 * RB + 1 ) * ( 2 * RB + 1 ) - 1;
        eo = ExactOffsets{ grab<int>( ne ), grab<int>( ne ), grab<double>( ne ), int( ne ) };
        // the lists of the flagged cells: room for a sixteenth of the cells in ring 2, grown when needed
        grow_ring2( std::max<SI>( nn / 16, 1024 ) ); grow_exact( 1024 ); grow_extras( 16 * 1024 );
        // the masses of the levels; the density
        nus.clear();
        for ( int n = Nf; n >= n0; n /= 2 ) nus.push_back( grab<double>( SI( n ) * n ) );
        if ( din.kind == 1 ) {
            const SI np = din.nx * din.ny, nt = ( din.nx + 1 ) * din.ny;
            img_v = grab<double>( np ); img_s0 = grab<double>( nt ); img_s1 = grab<double>( nt ); img_s2 = grab<double>( nt );
        }
        if ( din.kind == 2 ) gauss_d = grab<double>( SI( din.g.size() ) );
        if ( din.kind == 3 ) {
            i1_c = grab<double>( 4 * din.nx * din.ny ); i1_k = grab<double>( din.nx + din.ny + 2 ); i1_p = grab<double>( 6 * ( din.nx + 1 ) * din.ny );
        }
        exr_cap = 0;
    }

    template<class T>
    void upload( T *dst, const T *src, SI n ) {
        if ( n > 0 ) cuda_check( cudaMemcpyAsync( dst, src, sizeof( T ) * n, cudaMemcpyHostToDevice, queue.stream ), "copy to the card ( grid solver )" );
        cuda_check( cudaStreamSynchronize( queue.stream ), "sync ( grid solver upload )" );
    }
    template<class T>
    void down( T *dst, const T *src, SI n ) { if ( n > 0 ) read_back( queue, dst, src, n ); }

    // ( the lists of the flagged cells: `false` when the pool has no room, nothing changed and the solve going on -- `run` rejects the diagram )
    bool room() { const bool r = ok; ok = true; return r; }
    bool grow_ring2( SI cap ) {
        if ( cap <= p2.cap ) return true;
        Planes2 g = p2;
        g.vx = grab<double>( 16 * cap ); g.vy = grab<double>( 16 * cap ); g.ow = grab<signed char>( 16 * cap );
        g.cap = cap;
        if ( ! room() ) return false;
        p2 = g;
        return true;
    }
    bool grow_exact( SI cap ) {
        if ( cap <= ep.cap ) return true;
        ExactPolys g = ep;
        g.px = grab<double>( C * cap ); g.py = grab<double>( C * cap ); g.nb = grab<int>( C * cap ); g.n = grab<int>( cap );
        g.cap = cap;
        if ( ! room() ) return false;
        ep = g;
        return true;
    }
    bool grow_bad( SI cap ) {
        if ( cap <= bad_cap ) return true;
        bad = grab<int>( cap ); faces = grab<int>( C * cap ); nfaces = grab<int>( cap );
        active = grab<unsigned char>( cap ); below = grab<unsigned char>( cap );
        bad_cap = cap;
        return ok;
    }
    bool grow_extras( SI cap ) {
        if ( cap <= xs.cap ) return true;
        Extras g = xs;
        g.i = grab<int>( cap ); g.j = grab<int>( cap ); g.w = grab<double>( cap );
        g.cap = cap;
        if ( ! room() ) return false;
        xs = g;
        return true;
    }

    template<class Acc,class F>
    void red( SI n, const F &f, Acc *out ) { reduce( queue, n, f, static_cast<Acc *>( partials ), out ); }

    // ---- the density of a level -----------------------------------------------------------------------------------
    /// the image of level n ( side `hl` ): its frame in index units, its values `n^2 h_n^2 rho`, its prefix sums
    template<class TFP>
    bool image_level( const DensityIn &din, Strided<TFP,1> prm, int n, double ox, double oy, double hl ) {
        const SI np = din.nx * din.ny, nt = ( din.nx + 1 ) * din.ny;
        ( void ) nt;
        const double Sh2 = double( n ) * n * hl * hl, hx = din.hx / hl, hy = din.hy / hl;
        launch_kernel( queue, &grid_image_values<TFP>, blocks_for( np ), BLOCK, 0, din.nx, din.ny, prm, din.first, Sh2, img_v );
        launch_kernel( queue, &grid_image_prefix, blocks_for( din.ny ), BLOCK, 0, din.nx, din.ny, hx, ( const double * ) img_v, img_s0, img_s1, img_s2 );
        gim.im = DensImage{ ( din.x0 - ox ) / hl - 0.5, ( din.y0 - oy ) / hl - 0.5, hx, hy, din.nx, din.ny, img_v, img_s0, img_s1, img_s2 };
        kind = 1;
        return true;
    }
    /// the gaussians of level n, widened by `blur` ( the continuation ): centres, widths in index units, masses `n^2 m`
    bool gauss_level( const DensityIn &din, int n, double ox, double oy, double hl, double blur ) {
        const SI k = SI( din.g.size() / 4 );
        std::vector<double> h( 4 * k );
        for ( SI q = 0; q < k; ++q ) {
            const double *pr = din.g.data() + 4 * q, s = std::sqrt( pr[ 2 ] * pr[ 2 ] + blur * blur );
            h[ q ] = ( pr[ 0 ] - ox ) / hl - 0.5;
            h[ k + q ] = ( pr[ 1 ] - oy ) / hl - 0.5;
            h[ 2 * k + q ] = s / hl;
            h[ 3 * k + q ] = double( n ) * n * pr[ 3 ];
        }
        upload( gauss_d, h.data(), 4 * k );
        gga.g = DensGauss{ k, gauss_d, gauss_d + k, gauss_d + 2 * k, gauss_d + 3 * k };
        kind = 2;
        return true;
    }
    /// the image of degree 1 on level n: its knots in index units from its corner, its coefficients `n^2 h_n^2 rho`, the prefixes
    template<class TFP>
    bool image1_level( const DensityIn &din, Strided<TFP,1> prm, int n, double ox, double oy, double hl ) {
        const SI nx = din.nx, ny = din.ny;
        const double x0 = ( din.kx[ 0 ] - ox ) / hl - 0.5, y0 = ( din.ky[ 0 ] - oy ) / hl - 0.5;
        std::vector<double> k( nx + ny + 2 );
        for ( SI q = 0; q <= nx; ++q ) k[ q ] = ( din.kx[ q ] - din.kx[ 0 ] ) / hl;
        for ( SI q = 0; q <= ny; ++q ) k[ nx + 1 + q ] = ( din.ky[ q ] - din.ky[ 0 ] ) / hl;
        upload( i1_k, k.data(), nx + ny + 2 );
        launch_kernel( queue, &grid_scaled_values<TFP>, blocks_for( 4 * nx * ny ), BLOCK, 0, 4 * nx * ny, prm, din.first, double( n ) * n * hl * hl, i1_c );
        launch_kernel( queue, &grid_image1_prefix, blocks_for( ny ), BLOCK, 0, nx, ny, ( const double * ) i1_k, ( const double * ) i1_c, i1_p );
        gi1.im = DensImage1{ nx, ny, x0, y0, i1_k, i1_k + nx + 1, i1_c, i1_p };
        kind = 3;
        return true;
    }

    template<class F>
    void with_density( F &&f ) {
        if ( kind == 3 ) f( gi1 );
        else if ( kind == 1 ) f( gim );
        else if ( kind == 2 ) f( gga );
        else f( GUnit{} );
    }

    // ---- one diagram ----------------------------------------------------------------------------------------------
    bool exhausted = false;
    SI n1 = 0;                                           ///< the flagged cells of the last diagram

    /// THE DIAGRAM OF `ps` on level n ( `Evaluator::run` ): the geometry, the masses, the weights of H
    bool run( int n, const double *ps, bool integrate = true ) {
        const SI nn = SI( n ) * n;
        const double t0 = grid_now();
        exhausted = false;
        launch_kernel( queue, &ring1_kernel, blocks_for( nn ), BLOCK, 0, n, ps, vx1, vy1, ow1 );
        zero_fill( queue, cnt, sizeof( Counts ) );
        launch_kernel( queue, &certify1_kernel, blocks_for( nn ), BLOCK, 0, n, ( const double * ) vx1, ( const double * ) vy1, ( const signed char * ) ow1,
                       flag1, list1, cnt );
        down( &hc, cnt, 1 );
        n1 = hc.n1;
        zero_fill( queue, flag2, nn );
        SI nex = 0;
        // ( a diagram whose flagged cells do not fit in the pool counts as exhausted: the trial is rejected like one out of the exact
        // path's reach, rather than the call refused. Doubled when there is room, else what this diagram needs )
        auto no_room = [&]() { exhausted = true; n1 = 0; return true; };
        if ( n1 ) {
            if ( n1 > p2.cap && ! grow_ring2( std::min( std::max( n1, 2 * p2.cap ), nnf ) ) && ! grow_ring2( n1 ) ) return no_room();
            launch_kernel( queue, &ring2_kernel, blocks_for( n1 ), BLOCK, 0, n, ps, ( const int * ) list1, n1, p2, pos2 );
            launch_kernel( queue, &neighbourhood_kernel, blocks_for( 9 * n1 ), BLOCK, 0, n, ( const int * ) list1, n1, mark, check, cnt );
            down( &hc, cnt, 1 );
            const SI nchk = hc.nchk;
            launch_kernel( queue, &cell_ok_kernel, blocks_for( nchk ), BLOCK, 0, n, ( const int * ) check, nchk, ( const unsigned char * ) flag1,
                           ( const double * ) vx1, ( const double * ) vy1, ( const signed char * ) ow1, p2, ( const int * ) pos2, flag2 );
            launch_kernel( queue, &promote_kernel, blocks_for( nchk ), BLOCK, 0, ( const int * ) check, nchk, mark, ( const unsigned char * ) flag2, flag1, list1, cnt );
            launch_kernel( queue, &exact_list_kernel, std::min<int>( blocks_for( nn ), 1024 ), BLOCK, 0, ( const int * ) list1, ( const Counts * ) cnt,
                           ( const unsigned char * ) flag2, exl, expos, cnt );
            down( &hc, cnt, 1 );
            n1 = hc.n1;
            nex = hc.nex;
            if ( nex ) {
                if ( nex > ep.cap && ! grow_exact( std::min( std::max( nex, 2 * ep.cap ), nnf ) ) && ! grow_exact( nex ) ) return no_room();
                red( nn, MaxOfFn{ ps }, mx );
                launch_kernel( queue, &copy_max, 1, 1, 0, ( const MaxV * ) mx, pmax );
                launch_kernel( queue, &exact_kernel, blocks_for( nex ), BLOCK, 0, n, ps, ( const double * ) pmax, ( const int * ) exl, nex, eo, ep, cnt );
                down( &hc, cnt, 1 );
                if ( hc.exhausted && rb < n )
                    return grow_rings( n ) && run( n, ps, integrate );
            }
            // ( a flagged cell has at most 16 faces in ring 2, `C` exact )
            const SI nxm = 16 * ( n1 - nex ) + SI( C ) * nex;
            if ( nxm > xs.cap && ! grow_extras( std::max( nxm, 2 * xs.cap ) ) && ! grow_extras( nxm ) ) return no_room();
        }
        if ( ! integrate )
            return true;
        const double t1 = grid_now();
        with_density( [&]( const auto &dens ) {
            using Dn = std::decay_t<decltype( dens )>;
            launch_kernel( queue, &integrate1_kernel<Dn>, blocks_for( nn ), BLOCK, 0, n, dens, ( const double * ) vx1, ( const double * ) vy1,
                           ( const signed char * ) ow1, ( const unsigned char * ) flag1, mass, W );
            if ( n1 )
                launch_kernel( queue, &integrate_flagged_kernel<Dn>, blocks_for( n1 ), BLOCK, 0, n, dens, ( const int * ) list1, n1, ( const unsigned char * ) flag2,
                               ( const int * ) pos2, ( const int * ) expos, p2, ep, mass, W, xs, cnt );
            if ( ! Dn::exact )
                launch_kernel( queue, &symmetrize_kernel, blocks_for( nn ), BLOCK, 0, n, W );
        } );
        exr.nr = 0;
        if ( n1 ) {
            down( &hc, cnt, 1 );
            exhausted = hc.exhausted != 0;
            if ( hc.nx > xs.cap ) { std::fprintf( stderr, "sdot grid solver ( card ): %u faces out of ring 1 for %lld places\n", hc.nx, ( long long ) xs.cap ); exhausted = true; }
            if ( hc.nx && ! merge_extras( n, std::min<SI>( hc.nx, xs.cap ) ) ) return false;
        }
        cuda_check( cudaStreamSynchronize( queue.stream ), "sync ( grid diagram )" );   // ( the timing; the residual reads it back next )
        t_geom += t1 - t0;
        t_int += grid_now() - t1;
        return true;
    }

    /// THE FACES OUT OF RING 1, on the host ( `Evaluator::gather_extras` + `symmetrize` ): sorted by ( i, j ); for a density whose
    /// faces are not exact, each pair once per side at the mean of what both saw ( a face seen from one side only counts half );
    /// then by row for the card
    bool merge_extras( int n, SI nx ) {
        std::vector<int> hi( nx ), hj( nx );
        std::vector<double> hw( nx );
        down( hi.data(), xs.i, nx ); down( hj.data(), xs.j, nx ); down( hw.data(), xs.w, nx );
        struct E { int i, j; double w; };
        std::vector<E> e( nx );
        for ( SI q = 0; q < nx; ++q ) e[ q ] = { hi[ q ], hj[ q ], hw[ q ] };
        const bool exact = kind == 0;
        if ( ! exact ) for ( auto &x : e ) if ( x.i > x.j ) std::swap( x.i, x.j );
        std::sort( e.begin(), e.end(), []( const E &a, const E &b ) { return a.i != b.i ? a.i < b.i : a.j != b.j ? a.j < b.j : a.w < b.w; } );
        if ( ! exact ) {
            std::vector<E> merged;
            for ( size_t q = 0; q < e.size(); ) {
                size_t r = q;
                double s = 0;
                for ( ; r < e.size() && e[ r ].i == e[ q ].i && e[ r ].j == e[ q ].j; ++r ) s += e[ r ].w;
                merged.push_back( { e[ q ].i, e[ q ].j, s / 2 } );
                merged.push_back( { e[ q ].j, e[ q ].i, s / 2 } );
                q = r;
            }
            std::sort( merged.begin(), merged.end(), []( const E &a, const E &b ) { return a.i != b.i ? a.i < b.i : a.j < b.j; } );
            e.swap( merged );
        }
        std::vector<int> rows, beg, cols;
        std::vector<double> w;
        for ( size_t q = 0; q < e.size(); ++q ) {
            if ( rows.empty() || rows.back() != e[ q ].i ) { rows.push_back( e[ q ].i ); beg.push_back( int( q ) ); }
            cols.push_back( e[ q ].j );
            w.push_back( e[ q ].w );
        }
        beg.push_back( int( e.size() ) );
        const SI ne = SI( e.size() ), nr = SI( rows.size() );
        if ( 2 * nr + 1 + ne > exr_cap ) {
            exr_cap = 2 * ( 2 * nr + 1 + ne );
            exr_buf = grab<int>( exr_cap );
            exr_w = grab<double>( exr_cap );
            if ( ! ok ) return false;
        }
        std::vector<int> buf;
        buf.insert( buf.end(), rows.begin(), rows.end() );
        buf.insert( buf.end(), beg.begin(), beg.end() );
        buf.insert( buf.end(), cols.begin(), cols.end() );
        upload( exr_buf, buf.data(), SI( buf.size() ) );
        upload( exr_w, w.data(), ne );
        exr = ExRows{ exr_buf, exr_buf + nr, exr_buf + 2 * nr + 1, exr_w, nr };
        ( void ) n;
        return true;
    }

    // ---- the linear system ------------------------------------------------------------------------------------------
    TreeLayout tl;

    /// the preconditioner of the last diagram's H
    void build_precond( int n ) {
        const SI nn = SI( n ) * n;
        tl = TreeLayout::of( n );
        launch_kernel( queue, &conductances_kernel, blocks_for( nn ), BLOCK, 0, n, ( const double * ) W, d0, g, tb );
        if ( exr.nr )
            launch_kernel( queue, &ex_conductances_kernel, blocks_for( exr.nr ), BLOCK, 0, n, exr, d0, g, tb );
        launch_kernel( queue, &inverse_kernel, blocks_for( nn ), BLOCK, 0, nn, ( const double * ) d0, id0 );
        for ( double *t : { g, tb } ) {
            for ( int la = 1; la <= tl.L; ++la )
                launch_kernel( queue, &restrict_a_kernel, blocks_for( SI( n >> la ) * n ), BLOCK, 0, tl, t, la );
            launch_kernel( queue, &pyramid_kernel, int( tl.rows ), BLOCK, 0, tl, t, ( const double * ) nullptr );
        }
        launch_kernel( queue, &scaling_kernel, blocks_for( tl.total() - tl.off[ 0 ][ 1 ] ), BLOCK, 0, tl, ( const double * ) g, ( const double * ) tb, Di );
    }

    /// z += the coarse part of M r ( `g( 0, 0 )` holds r )
    void apply_precond( int n ) {
        for ( int la = 1; la <= tl.L; ++la )
            launch_kernel( queue, &restrict_a_kernel, blocks_for( SI( n >> la ) * n ), BLOCK, 0, tl, g, la );
        launch_kernel( queue, &pyramid_kernel, int( tl.rows ), BLOCK, 0, tl, g, ( const double * ) Di );
        launch_kernel( queue, &prolong_b_kernel, int( tl.rows ), BLOCK, 0, tl, g );
        for ( int la = tl.L; la >= 2; --la )
            launch_kernel( queue, &prolong_a_kernel, blocks_for( SI( n >> ( la - 1 ) ) * n ), BLOCK, 0, tl, g, la );
        launch_kernel( queue, &final_kernel, blocks_for( SI( n ) * n ), BLOCK, 0, tl, ( const double * ) g, z );
    }

    /// PCG on `H x = b` ( `pcg_fused` ): the iterates at zero mean, stopped on `| r | <= rtol | b |`. Returns the iterations.
    int pcg( int n, const double *b, double rtol, int maxit ) {
        const SI nn = SI( n ) * n;
        Sum1 h1;
        red( nn, CgInitFn{ b, id0, x, r, z, p, g }, s1a );
        down( &h1, s1a, 1 );
        const double nb = std::sqrt( h1.s );
        if ( nb == 0 ) return 0;
        apply_precond( n );
        red( nn, CgPassCFn{ r, z }, s3 );
        launch_kernel( queue, &cg_beta_kernel, 1, 1, 0, cs, ( const Sum3 * ) s3, double( nn ), 1 );
        int it = 0;
        for ( ; it < maxit; ++it ) {
            launch_kernel( queue, &cg_direction_kernel, blocks_for( nn ), BLOCK, 0, nn, ( const CgScalars * ) cs, ( const double * ) z, p );
            red( nn, MatvecFn{ n, W, p, q }, s1a );
            if ( exr.nr ) red( exr.nr, MatvecExFn{ exr, p, q }, s1b );
            launch_kernel( queue, &cg_alpha_kernel, 1, 1, 0, cs, ( const Sum1 * ) s1a, ( const Sum1 * ) ( exr.nr ? s1b : nullptr ) );
            red( nn, CgUpdateFn{ cs, p, q, id0, x, r, z, g }, s1a );
            down( &h1, s1a, 1 );
            if ( std::sqrt( h1.s ) <= rtol * nb ) { ++it; break; }
            apply_precond( n );
            red( nn, CgPassCFn{ r, z }, s3 );
            launch_kernel( queue, &cg_beta_kernel, 1, 1, 0, cs, ( const Sum3 * ) s3, double( nn ), 0 );
        }
        return it;
    }

    // ---- Newton ----------------------------------------------------------------------------------------------------
    Err err_of( SI nn, const double *t ) {
        Err e;
        red( nn, ErrFn{ mass, t }, err );
        down( &e, err, 1 );
        e.l2 = std::sqrt( e.l2 );
        return e;
    }

    /// THE CELLS THAT A START OR A STEP LEFT ( NEARLY ) EMPTY ( `newton::repair` of the CPU ): each one searches its own psi, one
    /// diagram per round ( the state in the CG's vectors, free between two solves ). On return the diagram is that of `ps`
    bool repair( int n, double *ps, Stats &st ) {
        const SI nn = SI( n ) * n;
        red( nn, SmallFn{ mass, target }, s1a );
        Sum1 small;
        down( &small, s1a, 1 );
        if ( ! sp::WarmRepair::worth( small.s, double( nn ) ) )
            return true;
        RepairState rs{ r, z, p, q };
        launch_kernel( queue, &repair_init_kernel, blocks_for( nn ), BLOCK, 0, nn, rs );
        for ( int round = 0; round < sp::WarmRepair::MAX_ROUNDS; ++round ) {
            launch_kernel( queue, &repair_round_kernel, blocks_for( nn ), BLOCK, 0, nn, ( const double * ) mass, ( const double * ) target, ps, rs, rhs );
            red( nn, SumOf{ rhs }, s1a );
            Sum1 moved;
            down( &moved, s1a, 1 );
            if ( moved.s == 0 )
                break;
            if ( ! run( n, ps ) ) return false;
            ++st.nb_diag;
            if ( exhausted )
                break;
        }
        return true;
    }

    /// THE LIMIT OF THE STEP ( `newton::tau_limit` of the CPU, `sdotplan/Bounds.h` ): the trial at `tau` left cells under the floor.
    /// Each one, its edges frozen as in the accepted diagram ( put back for that ), bisected on `tau`: `res` the largest `tau` that
    /// keeps them all above the floor, 0 when they do not fall with their edges frozen. An edge may still appear before: the trial checks
    bool tau_limit( int n, double tau, double eps0, double &res ) {
        const SI nn = SI( n ) * n;
        res = 0;
        zero_fill( queue, cnt, sizeof( Counts ) );
        if ( ! grow_bad( std::max<SI>( nnf / 32, 1024 ) ) ) return false;
        // ( as many as the room: a step that empties more cells than that is left to the damping alone )
        red( nn, FloorCountFn{ mass, eps0 }, s1a );
        Sum1 nbs;
        down( &nbs, s1a, 1 );
        const SI nb = SI( nbs.s );
        if ( nb > bad_cap )
            return true;
        launch_kernel( queue, &under_floor_kernel, blocks_for( nn ), BLOCK, 0, nn, ( const double * ) mass, eps0, bad, cnt );
        if ( ! run( n, psi, false ) ) return false;
        if ( exhausted )
            return true;
        launch_kernel( queue, &frozen_faces_kernel, blocks_for( nb ), BLOCK, 0, n, ( const int * ) bad, nb, ( const unsigned char * ) flag1, ( const unsigned char * ) flag2,
                       ( const signed char * ) ow1, ( const int * ) pos2, ( const int * ) expos, p2, ep, faces, nfaces, active );
        auto falling = [&]( double s ) {
            zero_fill( queue, cnt, sizeof( Counts ) );
            with_density( [&]( const auto &dens ) {
                using Dn = std::decay_t<decltype( dens )>;
                launch_kernel( queue, &frozen_mass_kernel<Dn>, blocks_for( nb ), BLOCK, 0, n, dens, ( const int * ) bad, nb, ( const int * ) faces, ( const int * ) nfaces,
                               ( const double * ) psi, ( const double * ) x, s, eps0, ( const unsigned char * ) active, below, cnt );
            } );
            down( &hc, cnt, 1 );
            return hc.nfell != 0;
        };
        if ( ! falling( tau ) )
            return true;
        std::swap( active, below );
        double lo = 0, hi = tau;
        for ( int round = 0; round < 48 && hi > 1e-10 && ! ( lo > 0 && hi - lo <= 0.05 * hi ); ++round ) {
            const double mid = ( lo + hi ) / 2;
            if ( ! falling( mid ) ) lo = mid;
            else { hi = mid; std::swap( active, below ); }
        }
        res = lo;
        return true;
    }

    /// DAMPED NEWTON on level n ( `GridNewton2D.h::newton` ): `target` the masses in index units, `psi` the start and the solution.
    /// `domain`: what the domain holds, from the coarsest level ( a start interpolated across very sharp masses can have cells that a
    /// far seed should cut and that the certificates of the rings let through )
    void newton( int n, double tol_rel, double tol_abs, double S, const Options &o, Stats &st, double domain = 0 ) {
        const SI nn = SI( n ) * n;
        auto done = [&]( const Err &e ) { return e.rel <= tol_rel || e.abs <= tol_abs * S; };
        if ( ! run( n, psi ) ) { st.failed = true; return; }
        ++st.nb_diag;
        // the targets scaled to what the domain holds ( a density with mass outside of it ), as `sdotplan/Solve.h` does
        {
            Sum2 h2;
            red( nn, Sum2Fn{ mass, target }, s2 );
            down( &h2, s2, 1 );
            const double sm = domain > 0 ? domain * S : h2.s0;
            launch_kernel( queue, &scale_kernel, blocks_for( nn ), BLOCK, 0, nn, ( const double * ) target, sm / h2.s1, target );
            st.domain_mass = sm / S;
        }
        Err e = err_of( nn, target );
        if ( st.residual0 < 0 ) st.residual0 = e.abs / S;
        double min_t;
        {
            Min1 h1;
            red( nn, MinOfFn{ target }, mn );
            down( &h1, mn, 1 );
            min_t = h1.m;
        }
        if ( e.mmin < sp::WarmRepair::LO * min_t ) {
            if ( ! repair( n, psi, st ) ) { st.failed = true; return; }
            e = err_of( nn, target );
            if ( ! ( e.mmin > 0 ) ) {                    // ( too many cells, or a search that failed: the last resort is the Voronoi )
                zero_fill( queue, psi, SI( sizeof( double ) ) * nn );
                if ( ! run( n, psi ) ) { st.failed = true; return; }
                ++st.nb_diag;
                e = err_of( nn, target );
            }
        }
        const double eps0 = 0.5 * std::min( e.mmin, sp::WarmRepair::LO * min_t );
        bool stuck = false;
        for ( int it = 1; it <= o.max_iter && ! done( e ) && ! exhausted; ++it ) {
            const double tl0 = grid_now();
            build_precond( n );
            red( nn, RhsFn{ target, mass, rhs }, s1b );
            launch_kernel( queue, &sub_mean_kernel, blocks_for( nn ), BLOCK, 0, nn, rhs, ( const Sum1 * ) s1b );
            st.lin_nb_iter += pcg( n, rhs, o.cg_rtol, 20000 );
            st.t_lin += grid_now() - tl0;
            // the damping
            double tau = 1, tau_min = 1e-6;
            int limit_rounds = 0;
            while ( true ) {
                launch_kernel( queue, &axpy_kernel, blocks_for( nn ), BLOCK, 0, nn, ( const double * ) psi, tau, ( const double * ) x, trial );
                if ( ! run( n, trial ) ) { st.failed = true; return; }
                ++st.nb_diag;
                Err e2 = err_of( nn, target );
                if ( ! exhausted && e2.mmin <= eps0 && limit_rounds < 8 ) {
                    ++limit_rounds;
                    double al;
                    if ( ! tau_limit( n, tau, eps0, al ) ) { st.failed = true; return; }
                    if ( al > 0 ) {
                        tau = 0.9 * al;
                        tau_min = std::min( tau_min, tau / 64 );
                    } else
                        limit_rounds = 8;                // ( no limit: back on the trial and the damping alone )
                    continue;
                }
                // ( a trial whose cells outgrow the exact path is rejected like one that empties a cell )
                if ( ! exhausted && e2.l2 <= ( 1 - tau / 2 ) * e.l2 ) {
                    if ( e2.mmin <= eps0 ) {
                        if ( ! repair( n, trial, st ) ) { st.failed = true; return; }
                        e2 = err_of( nn, target );
                    }
                    if ( ! exhausted && e2.mmin > eps0 && e2.l2 <= ( 1 - tau / 2 ) * e.l2 ) { e = e2; break; }
                }
                tau /= 2;
                ++st.nb_backtracks;
                if ( tau < tau_min ) { stuck = true; break; }
            }
            // ( no step: the last accepted weights stay, with their outputs )
            if ( stuck ) {
                if ( ! run( n, psi ) ) { st.failed = true; return; }
                ++st.nb_diag;
                break;
            }
            std::swap( psi, trial );
            ++st.nb_iter;
        }
        st.residual = e.abs / S;
        st.exhausted = st.exhausted || exhausted;
        st.status = exhausted || stuck ? 6 : done( e ) ? 1 : 2;
    }

    /// the moments of the last diagram ( level n ) into `dx` ( bx ), `rhs` ( by ), `trial` ( m2 )
    void moments( int n ) {
        const SI nn = SI( n ) * n;
        with_density( [&]( const auto &dens ) {
            using Dn = std::decay_t<decltype( dens )>;
            launch_kernel( queue, &moments1_kernel<Dn>, blocks_for( nn ), BLOCK, 0, n, dens, ( const double * ) vx1, ( const double * ) vy1,
                           ( const signed char * ) ow1, ( const unsigned char * ) flag1, dx, rhs, trial );
            if ( n1 )
                launch_kernel( queue, &moments_flagged_kernel<Dn>, blocks_for( n1 ), BLOCK, 0, n, dens, ( const int * ) list1, n1, ( const unsigned char * ) flag2,
                               ( const int * ) pos2, ( const int * ) expos, p2, ep, dx, rhs, trial );
        } );
    }

};

/// THE SOLVE ( `GridNewton2D.h::solve` ): `nu` the masses of the diracs ( C order ), `w0` a start ( physical weights ) or none,
/// the outputs physical. Returns the stats.
template<class Alloc,class TFN,class TFW,class TFP,class TFO>
Stats solve( const CudaQueue &queue, Alloc &allocator, int N, double ox, double oy, double h, const DensityIn &din, Strided<TFP,1> prm,
             Strided<TFN,1> nu_in, Strided<TFW,1> w0_in, bool has_w0, const Options &o,
             StridedOut<TFO,1> weights, StridedOut<TFO,1> masses, StridedOut<TFO,2> bary, TFO *cost ) {
    Stats st;
    const double t0 = grid_now();
    Solver<Alloc> sv( queue, allocator, N );
    const int n0 = has_w0 ? N : std::min( std::max( o.n0, 32 ), N );
    if ( ! sv.prepare( n0, din ) ) { st.failed = true; return st; }
    auto trace = [&]( const char *what ) {
        if ( ! o.trace ) return;
        cuda_check( cudaStreamSynchronize( queue.stream ), "sync ( trace )" );
        std::printf( "  grid card: %-28s %8.4f s ( diagrams %.4f, linear %.4f )\n", what, grid_now() - t0, sv.t_geom + sv.t_int, st.t_lin );
        std::fflush( stdout );
    };
    trace( "buffers" );
    const SI nnf = SI( N ) * N;
    // the masses of each level: sums over the blocks of the finest ( normalized to 1 ), coarsest last
    const std::vector<double *> &nus = sv.nus;
    {
        double *f = nus[ 0 ];
        launch_kernel( queue, &read_input<TFN>, blocks_for( nnf ), BLOCK, 0, nnf, nu_in, 1.0, f );
        Sum1 h1;
        sv.red( nnf, SumOf{ f }, sv.s1a );
        sv.down( &h1, sv.s1a, 1 );
        launch_kernel( queue, &scale_kernel, blocks_for( nnf ), BLOCK, 0, nnf, ( const double * ) f, 1 / h1.s, f );
        for ( int n = N, l = 0; n > n0; n /= 2, ++l )
            launch_kernel( queue, &coarsen_kernel, blocks_for( SI( n / 2 ) * ( n / 2 ) ), BLOCK, 0, n, ( const double * ) nus[ l ], nus[ l + 1 ] );
    }
    trace( "masses of the levels" );
    if ( has_w0 )
        launch_kernel( queue, &read_input<TFW>, blocks_for( nnf ), BLOCK, 0, nnf, w0_in, 1 / ( h * h ), sv.psi );
    else
        zero_fill( queue, sv.psi, SI( sizeof( double ) ) * n0 * n0 );
    auto density = [&]( int n, double hl, double blur ) {
        if ( din.kind == 1 ) return sv.image_level( din, prm, n, ox, oy, hl );
        if ( din.kind == 2 ) return sv.gauss_level( din, n, ox, oy, hl, blur );
        if ( din.kind == 3 ) return sv.image1_level( din, prm, n, ox, oy, hl );
        sv.kind = 0;
        return true;
    };
    auto add = [&]( const Stats &sl ) {
        st.nb_iter += sl.nb_iter; st.nb_diag += sl.nb_diag; st.nb_backtracks += sl.nb_backtracks; st.lin_nb_iter += sl.lin_nb_iter;
        st.t_lin += sl.t_lin; ++st.nb_levels;
        if ( st.residual0 < 0 ) st.residual0 = sl.residual0;
        st.failed = st.failed || sl.failed;
    };
    // THE WIDTH CONTINUATION ( gaussians, no start given ): on the coarsest level, the density widened by `b` ( from half the grid
    // down by halves ), then the density itself
    if ( din.kind == 2 && ! has_w0 ) {
        double smin = din.g[ 2 ];
        for ( size_t q = 0; q < din.g.size() / 4; ++q ) smin = std::min( smin, din.g[ 4 * q + 2 ] );
        const double hl = h * N / n0, S = double( n0 ) * n0;
        for ( double b = h * N / 2; b > smin / 4; b /= 2 ) {
            if ( ! density( n0, hl, b ) ) { st.failed = true; break; }
            launch_kernel( queue, &scale_kernel, blocks_for( SI( n0 ) * n0 ), BLOCK, 0, SI( n0 ) * n0, ( const double * ) nus.back(), S, sv.target );
            Stats sl;
            sv.newton( n0, std::max( o.tol_rel, o.ctol ), 0.0, S, o, sl );
            add( sl );
            trace( "width continuation stage" );
            if ( sl.failed ) break;
        }
    }
    double domain = 0;                                   // ( what the domain holds, from the coarsest level )
    for ( int n = n0, l = int( nus.size() ) - 1; n <= N && ! st.failed; n *= 2, --l ) {
        if ( n > n0 ) {
            launch_kernel( queue, &refine_kernel, blocks_for( SI( n ) * n ), BLOCK, 0, n / 2, ( const double * ) sv.psi, sv.trial );
            std::swap( sv.psi, sv.trial );
        }
        const double S = double( n ) * n, hl = h * N / n;
        if ( ! density( n, hl, 0.0 ) ) { st.failed = true; break; }
        launch_kernel( queue, &scale_kernel, blocks_for( SI( n ) * n ), BLOCK, 0, SI( n ) * n, ( const double * ) nus[ l ], S, sv.target );
        Stats sl;
        const bool fine = n == N;
        sv.newton( n, fine ? o.tol_rel : std::max( o.tol_rel, o.ctol ), fine ? o.tol_abs : 0.0, S, o, sl, domain );
        domain = sl.domain_mass;
        add( sl );
        if ( o.trace ) { char w[ 64 ]; std::snprintf( w, 64, "level %d ( %d its )", n, sl.nb_iter ); trace( w ); }
        st.exhausted = st.exhausted || sl.exhausted;
        if ( sl.failed ) break;
        if ( ! fine ) continue;                          // ( a level that failed still gives a start )
        st.status = sl.status;
        st.residual = sl.residual;
        st.domain_mass = sl.domain_mass;
        // the outputs, physical
        sv.moments( N );
        launch_kernel( queue, &outputs_kernel<TFO>, blocks_for( nnf ), BLOCK, 0, N, ox, oy, h, S, ( const double * ) sv.psi, ( const double * ) sv.mass,
                       ( const double * ) sv.dx, ( const double * ) sv.rhs, ( const double * ) sv.trial, weights, masses, bary, sv.q );
        sv.red( nnf, SumOf{ sv.q }, sv.s1a );
        launch_kernel( queue, &write_sum<TFO>, 1, 1, 0, ( const Sum1 * ) sv.s1a, cost );
        trace( "moments and outputs" );
    }
    st.t_diag = sv.t_geom + sv.t_int;
    if ( st.failed ) st.status = 6;
    cuda_check( cudaStreamSynchronize( queue.stream ), "sync ( grid solver )" );
    st.t_total = grid_now() - t0;
    return st;
}

/// THE HANDLER'S BODY ( `SdotPlanNd._build_grid_card` ): the options and the density read back, the solve, the stats written
template<class Alloc>
void solve_call( const CudaQueue &queue, Alloc &allocator, const auto &nu, const auto &w0, const auto &opts_in, const auto &dens_prm,
                 auto &&weights, auto &&masses, auto &&bary, auto &&cost, auto &&stats ) {
    using TFO = std::remove_const_t<typename std::decay_t<decltype( weights )>::TF>;
    if constexpr ( requires { allocator.why; } )
        allocator.why = "It was the card's grid solve of " + std::to_string( SI( nu.shape( 0 ) ) ) + " diracs ( `SdotPlanNd._build_grid_card`, "
                        "`gpu/GridNewton2D.cuh` ). Solve a smaller grid, give XLA more of the card ( XLA_PYTHON_CLIENT_MEM_FRACTION ), or solve "
                        "on the CPU ( LOOM_DEVICE=cpu )";
    // the options ( `SdotPlanNd._build_grid_card` ): N, ox, oy, h, tol_abs, tol_rel, ctol, n0, max_iter, cg_rtol, has_w0, kind, the
    // image's x0, y0, hx, hy, nx, ny, trace
    const SI no = SI( opts_in.shape( 0 ) ), np = SI( dens_prm.shape( 0 ) );
    std::vector<double> op( no );
    {
        double *tmp = static_cast<double *>( take( allocator, SI( sizeof( double ) ) * std::max<SI>( no, 1 ) ) );
        if ( ! tmp ) return;
        using TO = std::remove_const_t<typename std::decay_t<decltype( opts_in )>::TF>;
        launch_kernel( queue, &read_input<TO>, 1, BLOCK, 0, no, strided( opts_in ), 1.0, tmp );
        read_back( queue, op.data(), ( const double * ) tmp, no );
    }
    const int N = int( op[ 0 ] );
    Options o;
    o.trace = no > 18 && op[ 18 ] != 0;
    o.tol_abs = op[ 4 ]; o.tol_rel = op[ 5 ]; o.ctol = op[ 6 ]; o.n0 = int( op[ 7 ] ); o.max_iter = int( op[ 8 ] ); o.cg_rtol = op[ 9 ];
    DensityIn din;
    din.kind = int( op[ 11 ] );
    if ( din.kind == 1 ) {
        din.x0 = op[ 12 ]; din.y0 = op[ 13 ]; din.hx = op[ 14 ]; din.hy = op[ 15 ]; din.nx = SI( op[ 16 ] ); din.ny = SI( op[ 17 ] );
        din.first = 0;
    } else if ( din.kind == 3 ) {
        // the image of degree 1: `prm` = its knots ( read back, small ), then its coefficients ( read where they are )
        din.nx = SI( op[ 16 ] ); din.ny = SI( op[ 17 ] );
        const SI nk = din.nx + din.ny + 2;
        double *tmp = static_cast<double *>( take( allocator, SI( sizeof( double ) ) * nk ) );
        if ( ! tmp ) return;
        using TP = std::remove_const_t<typename std::decay_t<decltype( dens_prm )>::TF>;
        launch_kernel( queue, &read_input<TP>, blocks_for( nk ), BLOCK, 0, nk, strided( dens_prm ), 1.0, tmp );
        std::vector<double> h( nk );
        read_back( queue, h.data(), ( const double * ) tmp, nk );
        din.kx.assign( h.begin(), h.begin() + din.nx + 1 );
        din.ky.assign( h.begin() + din.nx + 1, h.end() );
        din.first = nk;
    } else if ( din.kind == 2 ) {
        // the gaussians: small, read back
        double *tmp = static_cast<double *>( take( allocator, SI( sizeof( double ) ) * std::max<SI>( np, 1 ) ) );
        if ( ! tmp ) return;
        using TP = std::remove_const_t<typename std::decay_t<decltype( dens_prm )>::TF>;
        launch_kernel( queue, &read_input<TP>, blocks_for( np ), BLOCK, 0, np, strided( dens_prm ), 1.0, tmp );
        std::vector<double> h( np );
        read_back( queue, h.data(), ( const double * ) tmp, np );
        din.g.assign( h.begin() + 1, h.begin() + 1 + 4 * SI( h[ 0 ] ) );   // ( `prm`: k, then cx, cy, sigma, mass for each )
    }
    const Stats st = solve( queue, allocator, N, op[ 1 ], op[ 2 ], op[ 3 ], din, strided( dens_prm ), strided( nu ), strided( w0 ), op[ 10 ] != 0, o,
                            strided_out<TFO,1>( weights ), strided_out<TFO,1>( masses ), strided_out<TFO,2>( bary ),
                            const_cast<TFO *>( cost.data().raw ) );
    if ( st.failed && ! st.nb_diag )
        return;                                          // ( the pool said no: loom reports it )
    std::vector<double> s( sp::NB_STATS, 0.0 );
    s[ sp::STATUS ] = st.status;
    s[ sp::RESIDUAL ] = st.residual;
    s[ sp::RESIDUAL0 ] = st.residual0;
    s[ sp::NB_ITER ] = st.nb_iter;
    s[ sp::NB_DIAG ] = st.nb_diag;
    s[ sp::NB_BACKTRACKS ] = st.nb_backtracks;
    s[ sp::T_DIAG ] = st.t_diag;
    s[ sp::T_LIN ] = st.t_lin;
    s[ sp::DOMAIN_MASS ] = st.domain_mass;
    s[ sp::LIN_NB_ITER ] = st.lin_nb_iter;
    s[ sp::T_TOTAL ] = st.t_total;
    s[ sp::NB_CONTINUATION_STEPS ] = st.nb_levels;
    s[ sp::RESIDUAL_FULL ] = st.residual;
    if constexpr ( requires { allocator.taken; } )
        s[ sp::SCRATCH_BYTES ] = double( allocator.taken );
    double *tmp = static_cast<double *>( take( allocator, SI( sizeof( double ) ) * sp::NB_STATS ) );
    if ( ! tmp ) return;
    cuda_check( cudaMemcpyAsync( tmp, s.data(), sizeof( double ) * sp::NB_STATS, cudaMemcpyHostToDevice, queue.stream ), "copy of the stats" );
    launch_kernel( queue, &write_values<TFO>, 1, BLOCK, 0, SI( sp::NB_STATS ), ( const double * ) tmp, strided_out<TFO,1>( stats ) );
    cuda_check( cudaStreamSynchronize( queue.stream ), "sync ( stats )" );
}

} // namespace sdot::gpu2d::grid
