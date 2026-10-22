#pragma once

// =====================================================================================
// THE GRID METHOD of `ContinuousOtPlan1d` ( `method = "grid"` ): tents against an image, the tents' density taken as
// UNIFORM inside each cell of a fine grid ( `scale` cells per image bin ) -- the sort-free transport of
// `sdot.projected1d`, for tents of any half-width.
//
// Per chunk of views ( never every view at once ):
//   1. the target on the fine grid: each bin's mass split evenly between its cells, the cumulated masses in closed form;
//   2. DEPOSIT: the EXACT mass of the tents in each fine cell, from their kinks ( three per tent, O( 1 ) whatever the
//      cells it covers, see below ); the mass that leaves the grid stays in its border cells;
//   3. the GRID TRANSPORT per view ( `Projected1d.h`: `grid_row` on the host, `grid_kernel` a block per view on the
//      card ): W2^2, exact for the gridded density, and the differences dg of dL/dmu between neighbour cells;
//   4. GATHER, per tent: dW/ds_i = ( m_i / w_i ) sum_e f( t_e ) dg_e and dW/dw_i = ( m_i / w_i ) sum_e t_e f( t_e ) dg_e
//      ( by parts, f the tent's profile, t_e = ( e - s_i ) / w_i ), in O( 1 ) from prefix sums of dg, then the gradients ( adjoint, or the summed cost's in the forward ), or the barycenters
//      b_i = s_i - dW/ds_i / ( 2 m_i ) and the Q_i of `ot1d::TentMoments` from dW/dw_i. No potentials.
// Positions are relative to the grid's start. The image's bins must be uniform ( its frame, no knots ).
// =====================================================================================

#include "Projected1d.h"

namespace sdot {

namespace ot1d_grid {

/// the geometry of the fine grid of an item: start, cell width, number of cells
struct Grid1d {
    double s_min, h;
    SI     M;
};

/// the grid of an item, from its image ( uniform bins ) and `scale`
HD inline Grid1d grid_of( const auto &item, SI nbin, int scale ) {
    Grid1d g;
    item.dst_dist.with_defaults( [&]( auto &&img ) {
        const double sc = double( img.frame( 0, 0 ) );
        g.s_min = double( img.origin( 0 ) ) + sc * double( img.knots( 0 ) );
        g.h = sc * ( double( img.knots( nbin ) ) - double( img.knots( 0 ) ) ) / double( nbin * scale );
    } );
    g.M = nbin * scale;
    return g;
}

/// the target on the grid of item: nu ( cell masses, total 1 ) and Cn ( cumulated, M + 1 ), cells [ j0, j1 )
HD inline void target_cells( const auto &item, SI nbin, int scale, SI j0, SI j1, double *nu, double *Cn, double *cdf_bins ) {
    item.dst_dist.with_defaults( [&]( auto &&img ) {
        const double sc = double( img.frame( 0, 0 ) );
        for ( SI j = j0; j < j1; ++j ) {
            const SI b = j / scale, r = j - b * scale;
            const double bm = sc * ( double( img.knots( b + 1 ) ) - double( img.knots( b ) ) ) * double( img.values[ b ] ) / cdf_bins[ nbin ];
            nu[ j ] = bm / scale;
            Cn[ j ] = cdf_bins[ b ] / cdf_bins[ nbin ] + bm * double( r ) / scale;
        }
        if ( j1 == nbin * scale )
            Cn[ j1 ] = 1;
    } );
}

/// the bins' cumulated masses of an item ( nbin + 1 )
HD inline void bin_cdf( const auto &item, SI nbin, double *cdf ) {
    item.dst_dist.with_defaults( [&]( auto &&img ) {
        const double sc = double( img.frame( 0, 0 ) );
        double c = 0;
        for ( SI j = 0; j < nbin; ++j ) {
            cdf[ j ] = c;
            c += sc * ( double( img.knots( j + 1 ) ) - double( img.knots( j ) ) ) * double( img.values[ j ] );
        }
        cdf[ nbin ] = c;
    } );
}

// ── the deposit by KINKS, the gather by PREFIX SUMS: O( 1 ) per tent and view, whatever the cells per tent ───────────
//
// A tent is linear between its three kinks ( s - w, s, s + w; slope jumps c = k, -2k, k, k = m / w^2 ): the density is
// sum_k c_k ( x - x_k )_+. A kink at offset delta inside cell j adds c ( h - delta )^2 / 2 to the mass of cell j, c ( h - delta )
// to the density at its right edge, c to the slope after it: three sums per cell ( `A`, `B`, `Q` ), then one recurrence
// per view gives the exact cell masses. Kinks left of the grid go to cell 0 ( a negative delta: the mass that leaves the
// grid stays in its border cell ), the last cell takes the rest of the mass ( kinks right of it are not needed ).
// The gather: f( t_e ) and t_e f( t_e ) are polynomials of the edge index e on each half of the tent, so the sums over
// its edges come from the per-view prefix sums of dg_e, ( e - c ) dg_e, ( e - c )^2 dg_e ( c = M / 2 ).

/// the three kinks of a tent into the sums of its cells
HD inline void deposit_kinks( const Grid1d &g, double s, double w, double m, double *A, double *B, double *Q, bool atomic ) {
    const double k = m / ( w * w );
    for ( int q = 0; q < 3; ++q ) {
        const double x = s + double( q - 1 ) * w, c = q == 1 ? -2 * k : k;
        const double u = ( x - g.s_min ) / g.h;
        SI j = SI( floor( u ) );
        if ( j >= g.M - 1 )
            continue;
        double delta;
        if ( j < 0 ) { j = 0; delta = x - g.s_min; } else delta = x - ( g.s_min + double( j ) * g.h );
        const double r = g.h - delta;
        const double a = c, b = c * r, qq = 0.5 * c * r * r;
#if defined( __CUDA_ARCH__ )
        if ( atomic ) { atomicAdd( A + j, a ); atomicAdd( B + j, b ); atomicAdd( Q + j, qq ); continue; }
#endif
        A[ j ] += a; B[ j ] += b; Q[ j ] += qq;
    }
}

/// the cell masses ( total 1 ) from the kinks' sums, plus `D` ( masses deposited directly, or null )
HD inline void masses_from_kinks( const Grid1d &g, const double *A, const double *B, const double *Q, double *mu, const double *D = nullptr ) {
    const double h = g.h;
    double d = 0, slope = 0, acc = 0;
    for ( SI j = 0; j + 1 < g.M; ++j ) {
        const double mj = d * h + 0.5 * slope * h * h + Q[ j ] + ( D ? D[ j ] : 0.0 );
        mu[ j ] = mj > 0 ? mj : 0.0;
        acc += mu[ j ];
        d += slope * h + B[ j ];
        slope += A[ j ];
    }
    mu[ g.M - 1 ] = acc < 1 ? 1 - acc : 0.0;
}

/// the prefix sums of dg_e ( e - c )^p, p = 0, 1, 2, over the edges e < E ( E = 0 .. M + 1 )
HD inline void edge_moments( SI M, const double *dg, double *P0, double *P1, double *P2 ) {
    const double c = double( M / 2 );
    double s0 = 0, s1 = 0, s2 = 0;
    for ( SI e = 0; e <= M; ++e ) {
        P0[ e ] = s0; P1[ e ] = s1; P2[ e ] = s2;
        const double x = double( e ) - c, d = dg[ e ];
        s0 += d; s1 += d * x; s2 += d * x * x;
    }
    P0[ M + 1 ] = s0; P1[ M + 1 ] = s1; P2[ M + 1 ] = s2;
}

/// dW/ds and dW/dw of a tent from the prefix sums ( `edge_moments` )
HD inline void gather_prefix( const Grid1d &g, double s, double w, double m, const double *P0, const double *P1, const double *P2,
                              double &ds, double &dw ) {
    const double iw = 1 / w, c = double( g.M / 2 );
    const double t1 = g.h * iw, t0 = ( g.s_min + c * g.h - s ) * iw;     // t_e = t0 + t1 ( e - c )
    // the edges of each half, raw ( in double: a far tent stays representable ) and intersected with the inner edges
    // 1 .. M - 1 ( the two end ones carry no difference )
    auto edge = [&]( double x ) { const double e = x < -1 ? -1.0 : ( x > double( g.M + 1 ) ? double( g.M + 1 ) : x ); return SI( e ); };
    const double ul = ( s - w - g.s_min ) / g.h, um = ( s - g.s_min ) / g.h, uh = ( s + w - g.s_min ) / g.h;
    const SI el = edge( ceil( ul ) ), em = edge( floor( um ) ), eh = edge( floor( uh ) );
    auto sums = [&]( SI a, SI b, double &S0, double &S1, double &S2 ) {
        a = a < 1 ? SI( 1 ) : a;
        b = b > g.M - 1 ? g.M - 1 : b;
        if ( b < a ) { S0 = S1 = S2 = 0; return; }
        S0 = P0[ b + 1 ] - P0[ a ]; S1 = P1[ b + 1 ] - P1[ a ]; S2 = P2[ b + 1 ] - P2[ a ];
    };
    double L0, L1, L2, R0, R1, R2;
    sums( el, em, L0, L1, L2 );
    sums( em + 1, eh, R0, R1, R2 );
    // left: f = 1 + t, t f = t + t^2; right: f = 1 - t, t f = t - t^2
    const double fs = ( 1 + t0 ) * L0 + t1 * L1 + ( 1 - t0 ) * R0 - t1 * R1;
    const double tt0 = t0 * t0, tt1 = 2 * t0 * t1, tt2 = t1 * t1;
    const double tfs = ( t0 * L0 + t1 * L1 ) + ( tt0 * L0 + tt1 * L1 + tt2 * L2 ) + ( t0 * R0 + t1 * R1 ) - ( tt0 * R0 + tt1 * R1 + tt2 * R2 );
    ds = fs * m * iw;
    dw = tfs * m * iw;
}

/// the per-tent results of a view: the outputs ( MODE 0 ), the adjoint's gradient times g ( 1 ), the summed cost's
/// ( 2 ) -- the gradients accumulated by the caller
HD inline ot1d::TentMoments moments_from( double s, double w, double m, double ds, double dw ) {
    ot1d::TentMoments r;
    r.b = s - ds / ( 2 * m );
    r.Q = w * w / 6 - dw * w / ( 2 * m );
    r.phi = 0;
    return r;
}

} // namespace ot1d_grid

/// host, the grid method: the views split between the threads, each with its own rows
template<class Grad,class SG = std::nullptr_t>
void continuous_ot_1d_grid( const CpuQueue &queue, const auto &batch, auto &&plan, Grad &&grad_plan, auto &allocator, int scale, SG &&sg = nullptr ) {
    using namespace ot1d_grid;
    constexpr bool bwd = ! std::is_same_v<std::decay_t<Grad>,std::nullptr_t>;
    constexpr bool with_sg = ! std::is_same_v<std::decay_t<SG>,std::nullptr_t>;
    if constexpr ( with_sg ) {
        sg.points.fill_with( queue, 0 );
        sg.half_widths.fill_with( queue, 0 );
    }
    const SI nb_items = SI( batch.size() );
    if ( nb_items == 0 )
        return;
    const auto item0 = plan( batch[ 0 ] );
    const SI n = item0.src_dist.nb_tents(), nbin = SI( item0.dst_dist.values.size() ), M = nbin * scale;
    double tot = 0;
    for ( SI i = 0; i < n; ++i )
        tot += double( item0.src_dist.mass( i ) );
    const int nt = int( std::min<SI>( std::max( queue.nb_workers(), 1 ), nb_items ) );
    const SI nd = 4 * M + 2 + nbin + 1 + 3 * M + 3 * ( M + 2 );
    double *D = allocator.template raw<double>( SI( nt ) * nd );
    if ( ! D )
        return;
    auto body = [&]( int th ) {
        double *mu = D + SI( th ) * nd, *nu = mu + M, *Cn = nu + M, *dg = Cn + M + 1, *cdf = dg + M + 1;
        double *A = cdf + nbin + 1, *B = A + M, *Q = B + M, *P0 = Q + M, *P1 = P0 + M + 2, *P2 = P1 + M + 2;
        for ( SI v = nb_items * th / nt; v < nb_items * ( th + 1 ) / nt; ++v ) {
            auto item = plan( batch[ v ] );
            const auto &src = item.src_dist;
            const Grid1d g = grid_of( item, nbin, scale );
            bin_cdf( item, nbin, cdf );
            target_cells( item, nbin, scale, 0, M, nu, Cn, cdf );
            for ( SI j = 0; j < 3 * M; ++j )
                A[ j ] = 0;
            for ( SI i = 0; i < n; ++i )
                deposit_kinks( g, double( src.position( i ) ), double( src.half_width( i ) ), double( src.mass( i ) ) / tot, A, B, Q, false );
            masses_from_kinks( g, A, B, Q, mu );
            ot1d_coop::Arr<double> amu{ mu }, anu{ nu }, aCn{ Cn }, adg{ dg };
            const double cost = projected1d::grid_row( amu, anu, aCn, adg, M, g.h );
            edge_moments( M, dg, P0, P1, P2 );
            double gv = 1;
            if constexpr ( bwd )
                gv = double( grad_plan( batch[ v ] ).cost );
            for ( SI i = 0; i < n; ++i ) {
                const double s = double( src.position( i ) ), w = double( src.half_width( i ) ), m = double( src.mass( i ) ) / tot;
                double ds, dw;
                gather_prefix( g, s, w, m, P0, P1, P2, ds, dw );
                if constexpr ( bwd ) {
                    src.add_position_grad( grad_plan( batch[ v ] ).src_dist, i, gv * ds );
                    src.add_half_width_grad( grad_plan( batch[ v ] ).src_dist, i, gv * dw );
                } else {
                    item.write_tent( i, moments_from( s, w, m, ds, dw ) );
                    if constexpr ( with_sg ) {
                        src.add_position_grad( sg, i, ds );
                        src.add_half_width_grad( sg, i, dw );
                    }
                }
            }
            if constexpr ( ! bwd )
                item.write_cost( cost );
        }
    };
    if ( nt <= 1 )
        body( 0 );
    else
        queue.run_threads( nt, body );
}

#if defined( __CUDACC__ )

namespace ot1d_grid {

constexpr int GR_BLOCK = 256;
constexpr SI  SHARED_HIST_BYTES = 48 * 1024;   // a view's kink sums in shared memory up to this ( the default limit )

inline int gr_blocks( SI n ) { return int( ( n + GR_BLOCK - 1 ) / GR_BLOCK ); }

/// what `projected1d::grid_kernel` reads: rows of raw buffers ( the problem index unused, one row per view )
template<class T>
struct Rows3 {
    T *p; SI stride;
    __host__ __device__ T &operator()( SI, SI a, SI i ) const { return p[ a * stride + i ]; }
};
template<class T>
struct Rows2 {
    T *p;
    __host__ __device__ T &operator()( SI, SI a ) const { return p[ a ]; }
};

/// per view of the chunk: the bins' cdf, the target cells, the grid; the kinks' sums cleared
template<class Plan,class Batch>
__global__ void __launch_bounds__( GR_BLOCK ) targets_kernel( Plan plan, Batch batch, SI v0, SI V, SI nbin, int scale, double *cdf, double *nu, double *Cn,
                                                           double *K, Grid1d *grids, int *wide ) {
    const SI v = SI( blockIdx.x );
    const SI M = nbin * scale;
    const auto item = plan( batch[ v0 + v ] );
    double *cv = cdf + v * ( nbin + 1 );
    if ( threadIdx.x == 0 ) {
        bin_cdf( item, nbin, cv );
        grids[ v ] = grid_of( item, nbin, scale );
    }
    __syncthreads();
    const SI per = ( M + GR_BLOCK - 1 ) / GR_BLOCK, j0 = std::min( M, SI( threadIdx.x ) * per ), j1 = std::min( M, j0 + per );
    target_cells( item, nbin, scale, j0, j1, nu + v * M, Cn + v * ( M + 1 ), cv );
    for ( SI j = threadIdx.x; j < 4 * M; j += GR_BLOCK )
        K[ v * 4 * M + j ] = 0;
    if ( v == 0 && threadIdx.x == 0 && wide )
        *wide = 0;
}

/// thread ( v, i ): the kinks of tent i of view v into the sums of its cells ( global atomics )
template<class Plan,class Batch>
__global__ void __launch_bounds__( GR_BLOCK ) kinks_kernel( Plan plan, Batch batch, SI v0, SI V, SI n, SI M, const double *tot, const Grid1d *grids, double *K,
                                                         float narrow ) {
    const SI q = SI( blockIdx.x ) * GR_BLOCK + threadIdx.x;
    if ( q >= V * n )
        return;
    const SI v = q / n, i = q - v * n;
    const auto src = plan( batch[ v0 + v ] ).src_dist;
    if ( double( src.half_width( i ) ) <= double( narrow ) * grids[ v ].h )
        return;     // deposited directly
    double *Kv = K + v * 4 * M;
    deposit_kinks( grids[ v ], double( src.position( i ) ), double( src.half_width( i ) ), double( src.mass( i ) ) / *tot, Kv, Kv + M, Kv + 2 * M, true );
}

// ── FLOAT32 POINTS: the per-tent work in float ( a card does float far faster than double: 32 x on a 2080 Ti ) ────────
// The kinks are placed and summed per block in float ( shared memory, native atomics ), then added to the double sums
// of the view, whose recurrence, as the grid transport, stays in double. The gather loops over the tent's edges in
// float on a float copy of dg ( the prefix-sum form cancels too much in float ). The precision is the points' float:
// positions to ~ 1e-7 of the detector, as `projected1d`'s float path.

/// `deposit_kinks` in float, but the slope jumps `A` in double: the recurrence integrates them twice ( slope, then
/// density, then mass ), and their float roundings came out at 1e-4 of the masses. The kinks are placed RELATIVE TO THE CELL OF THE TENT'S CENTER: the center is rounded
/// once ( a shift of the whole tent, harmless ), and the offsets of the three kinks are small numbers, exact to far
/// better than the grid -- rounded separately from the grid's start, the three kinks would not cancel, and the
/// density left behind each tent would pile up along the view.
__device__ __forceinline__ void deposit_kinks_f( float s_min, float h, SI M, float s, float w, float m, double *A, float *B, float *Q ) {
    const float k = m / ( w * w ), ih = 1 / h;
    const float xc = s - s_min;
    const SI jc = SI( floorf( xc * ih ) );
    const float uc = xc - float( jc ) * h;               // the center's offset in its cell
    for ( int q = 0; q < 3; ++q ) {
        const float o = uc + float( q - 1 ) * w, c = q == 1 ? -2 * k : k;
        const SI dj = SI( floorf( o * ih ) );
        SI j = jc + dj;
        if ( j >= M - 1 )
            continue;
        float delta = o - float( dj ) * h;
        if ( j < 0 ) { delta += float( j ) * h; j = 0; }
        const float r = h - delta;
        atomicAdd( A + j, double( c ) );
        atomicAdd( B + j, c * r );
        atomicAdd( Q + j, 0.5f * c * r * r );
    }
}

/// a tent is NARROW when it covers few cells ( w <= NARROW_CELLS h ): its exact cell masses are then deposited
/// directly ( positive: float sums are safe, one atomic per cell ), cheaper than its three kinks' nine sums. The
/// kernels take the threshold as `narrow` ( 0 when the direct histogram does not fit in shared memory: all wide )
constexpr float NARROW_CELLS = 4;

/// the tent's primitive on [ -1, 1 ] ( 0 / 1 outside )
__device__ __forceinline__ float tent_G_f( float t ) {
    const float u = t < -1 ? -1.0f : ( t > 1 ? 1.0f : t );
    return 0.5f + u - u * ( u < 0 ? -u : u ) * 0.5f;
}

/// the exact masses of a narrow tent in its cells ( float, relative to its center's cell ); the grid's end edges at
/// -inf / +inf ( the mass that leaves the grid stays in its border cells )
__device__ __forceinline__ void deposit_direct_f( float s_min, float h, SI M, float s, float w, float m, float *D ) {
    const float iw = 1 / w, ih = 1 / h, xc = s - s_min;
    const SI jc = SI( floorf( xc * ih ) );
    const float uc = xc - float( jc ) * h;
    SI ja = jc + SI( floorf( ( uc - w ) * ih ) ), jb = jc + SI( ceilf( ( uc + w ) * ih ) );
    ja = ja < 0 ? SI( 0 ) : ( ja > M - 1 ? M - 1 : ja );
    jb = jb < ja + 1 ? ja + 1 : ( jb > M ? M : jb );
    auto G_at = [&]( SI e ) { return e <= 0 ? 0.0f : ( e >= M ? 1.0f : tent_G_f( ( float( e - jc ) * h - uc ) * iw ) ); };
    float G0 = G_at( ja );
    for ( SI j = ja; j < jb; ++j ) {
        const float G1 = G_at( j + 1 ), c = m * ( G1 - G0 );
        if ( c != 0 )
            atomicAdd( D + j, c );
        G0 = G1;
    }
}

/// dW/ds and dW/dw of a tent in float, by its edges ( `dgf`: dg in float ), t relative to the center's cell as above
__device__ __forceinline__ void gather_direct_f( float s_min, float h, SI M, float s, float w, float m, const float *dgf, float &ds, float &dw ) {
    const float iw = 1 / w, ih = 1 / h, xc = s - s_min;
    const SI jc = SI( floorf( xc * ih ) );
    const float uc = xc - float( jc ) * h;
    const SI span = SI( ceilf( w * ih ) ) + 1;
    SI a = jc - span + 1, b = jc + span;
    a = a < 1 ? SI( 1 ) : a;
    b = b > M - 1 ? M - 1 : b;
    float a_s = 0, a_w = 0;
    for ( SI e = a; e <= b; ++e ) {
        const float t = ( float( e - jc ) * h - uc ) * iw, at = t < 0 ? -t : t;
        if ( at < 1 ) {
            const float fv = 1 - at, d = dgf[ e ];
            a_s += d * fv;
            a_w += d * t * fv;
        }
    }
    ds = a_s * m * iw;
    dw = a_w * m * iw;
}

/// block ( v, tile ): the kinks of a tile of view v into sums in shared memory ( `Acc`: double, or float for float
/// points ), then one global atomic per touched cell ( global atomics alone serialize on the few cells of a row )
template<class Acc,class Plan,class Batch>
__global__ void __launch_bounds__( GR_BLOCK ) kinks_shared_kernel( Plan plan, Batch batch, SI v0, SI V, SI n, SI M, SI tiles, const double *tot,
                                                                const Grid1d *grids, double *K, const int *wide, float narrow ) {
    extern __shared__ __align__( 8 ) unsigned char sk_raw[];
    if ( wide && ! *wide )      // float points: only the wide tents come here, and there are none
        return;
    const SI v = SI( blockIdx.x ) / tiles, tile = SI( blockIdx.x ) - v * tiles;
    // the sums of the cells: A in double, B and Q in `Acc` ( see `deposit_kinks_f` )
    double *sA = reinterpret_cast<double *>( sk_raw );
    Acc *sB = reinterpret_cast<Acc *>( sA + M ), *sQ = sB + M;
    for ( SI j = threadIdx.x; j < M; j += GR_BLOCK ) {
        sA[ j ] = 0;
        sB[ j ] = 0;
        sQ[ j ] = 0;
    }
    __syncthreads();
    const auto src = plan( batch[ v0 + v ] ).src_dist;
    const Grid1d g = grids[ v ];
    const double it = 1 / *tot;
    const SI i0 = n * tile / tiles, i1 = n * ( tile + 1 ) / tiles;
    if constexpr ( std::is_same_v<Acc,float> ) {
        const float smf = float( g.s_min ), hf = float( g.h ), itf = float( it );
        for ( SI i = i0 + threadIdx.x; i < i1; i += GR_BLOCK ) {
            const float w = float( src.half_width( i ) );
            if ( w > narrow * hf )
                deposit_kinks_f( smf, hf, M, float( src.position( i ) ), w, float( src.mass( i ) ) * itf, sA, sB, sQ );
        }
    } else {
        for ( SI i = i0 + threadIdx.x; i < i1; i += GR_BLOCK )
            deposit_kinks( g, double( src.position( i ) ), double( src.half_width( i ) ), double( src.mass( i ) ) * it, sA, sB, sQ, true );
    }
    __syncthreads();
    double *Kv = K + v * 4 * M;
    for ( SI j = threadIdx.x; j < M; j += GR_BLOCK ) {
        if ( sA[ j ] != 0 ) atomicAdd( Kv + j, sA[ j ] );
        if ( sB[ j ] != 0 ) atomicAdd( Kv + M + j, double( sB[ j ] ) );
        if ( sQ[ j ] != 0 ) atomicAdd( Kv + 2 * M + j, double( sQ[ j ] ) );
    }
}

/// block ( v, tile ), float points: the narrow tents of a tile of view v into a histogram of masses in shared memory
/// ( float ), then one global atomic per touched cell into the view's direct masses ( double )
template<class Plan,class Batch>
__global__ void __launch_bounds__( GR_BLOCK ) direct_shared_kernel( Plan plan, Batch batch, SI v0, SI V, SI n, SI M, SI tiles, const double *tot,
                                                                 const Grid1d *grids, double *K, float narrow ) {
    extern __shared__ __align__( 8 ) unsigned char sd_raw[];
    float *sD = reinterpret_cast<float *>( sd_raw );
    const SI v = SI( blockIdx.x ) / tiles, tile = SI( blockIdx.x ) - v * tiles;
    for ( SI j = threadIdx.x; j < M; j += GR_BLOCK )
        sD[ j ] = 0;
    __syncthreads();
    const auto src = plan( batch[ v0 + v ] ).src_dist;
    const Grid1d g = grids[ v ];
    const float smf = float( g.s_min ), hf = float( g.h ), itf = float( 1 / *tot );
    const SI i0 = n * tile / tiles, i1 = n * ( tile + 1 ) / tiles;
    for ( SI i = i0 + threadIdx.x; i < i1; i += GR_BLOCK ) {
        const float w = float( src.half_width( i ) );
        if ( w <= narrow * hf )
            deposit_direct_f( smf, hf, M, float( src.position( i ) ), w, float( src.mass( i ) ) * itf, sD );
    }
    __syncthreads();
    double *Dv = K + v * 4 * M + 3 * M;
    for ( SI j = threadIdx.x; j < M; j += GR_BLOCK )
        if ( sD[ j ] != 0 )
            atomicAdd( Dv + j, double( sD[ j ] ) );
}

/// thread i: is there a wide tent ( for the grid of the chunk's first view )? `wide` cleared by `targets_kernel`
template<class Plan,class Batch>
__global__ void __launch_bounds__( GR_BLOCK ) wide_kernel( Plan plan, Batch batch, SI v0, SI n, const Grid1d *grids, int *wide, float narrow ) {
    const SI i = SI( blockIdx.x ) * GR_BLOCK + threadIdx.x;
    if ( i >= n )
        return;
    if ( float( plan( batch[ v0 ] ).src_dist.half_width( i ) ) > narrow * float( grids[ 0 ].h ) )
        *wide = 1;
}

/// thread v: the cell masses of view v from its kinks' sums and its direct masses
template<int = 0>
__global__ void __launch_bounds__( GR_BLOCK ) masses_kernel( SI V, SI M, const Grid1d *grids, const double *K, double *mu ) {
    const SI v = SI( blockIdx.x ) * GR_BLOCK + threadIdx.x;
    if ( v >= V )
        return;
    const double *Kv = K + v * 4 * M;
    masses_from_kinks( grids[ v ], Kv, Kv + M, Kv + 2 * M, mu + v * M, Kv + 3 * M );
}

/// thread v: the prefix sums of the edge differences of view v ( `edge_moments` )
template<int = 0>
__global__ void __launch_bounds__( GR_BLOCK ) moments_kernel( SI V, SI M, const double *dg, double *P ) {
    const SI v = SI( blockIdx.x ) * GR_BLOCK + threadIdx.x;
    if ( v >= V )
        return;
    double *Pv = P + v * 3 * ( M + 2 );
    edge_moments( M, dg + v * ( M + 1 ), Pv, Pv + M + 2, Pv + 2 * ( M + 2 ) );
}

/// the grid transport ran in grid units ( h = 1 ): its loss and its dg back to the view's units ( x h^2 ); `dgf`: a
/// float copy of dg ( float points ), or null
template<int = 0>
__global__ void __launch_bounds__( GR_BLOCK ) scale_kernel( SI V, SI M, const Grid1d *grids, double *dg, double *loss, float *dgf ) {
    const SI q = SI( blockIdx.x ) * GR_BLOCK + threadIdx.x;
    if ( q < V * ( M + 1 ) ) {
        const double h = grids[ q / ( M + 1 ) ].h;
        dg[ q ] *= h * h;
        if ( dgf )
            dgf[ q ] = float( dg[ q ] );
    }
    if ( q < V ) {
        const double h = grids[ q ].h;
        loss[ q ] *= h * h;
    }
}

/// per tent i, over the views of the chunk: dW/ds, dW/dw, then as `ot1d_card::tents_kernel` ( MODE 0 / 1 / 2 )
template<int MODE,class Plan,class Grad,class Batch>
__global__ void __launch_bounds__( GR_BLOCK ) gather_kernel( Plan plan, Grad grad_plan, Batch batch, SI v0, SI V, SI n, SI M, const double *tot,
                                                          const Grid1d *grids, const double *P, const float *dgf, const double *loss ) {
    const SI i = SI( blockIdx.x ) * GR_BLOCK + threadIdx.x;
    if ( i >= n )
        return;
    using Src = std::decay_t<decltype( plan( batch[ 0 ] ).src_dist )>;
    constexpr SI D = Src::ct_proj;
    double gp[ D ], gw = 0;
    for ( SI d = 0; d < D; ++d )
        gp[ d ] = 0;
    const auto src0 = plan( batch[ v0 ] ).src_dist;
    static_assert( ot1d_card::shared_half_width_v<std::decay_t<decltype( src0 )>>, "the card's kernels need half-widths shared by the items" );
    const double m = double( src0.mass( i ) ) / *tot, w = double( src0.half_width( i ) );
    for ( SI v = 0; v < V; ++v ) {
        auto item = plan( batch[ v0 + v ] );
        const auto &src = item.src_dist;
        const double s = double( src.position( i ) );
        double ds, dw;
        if constexpr ( sizeof( typename Src::TF ) == 4 ) {
            float dsf, dwf;
            gather_direct_f( float( grids[ v ].s_min ), float( grids[ v ].h ), M, float( src.position( i ) ), float( w ), float( m ),
                             dgf + v * ( M + 1 ), dsf, dwf );
            ds = dsf;
            dw = dwf;
        } else {
            const double *Pv = P + v * 3 * ( M + 2 );
            gather_prefix( grids[ v ], s, w, m, Pv, Pv + M + 2, Pv + 2 * ( M + 2 ), ds, dw );
        }
        if constexpr ( MODE != 0 ) {
            double g = 1;
            if constexpr ( MODE == 1 )
                g = double( grad_plan( batch[ v0 + v ] ).cost );
            for ( SI d = 0; d < D; ++d )
                gp[ d ] += g * ds * double( src.normal_component( d ) );
            gw += g * dw;
        }
        if constexpr ( MODE != 1 ) {
            item.write_tent( i, moments_from( s, w, m, ds, dw ) );
            if ( i == 0 )
                item.write_cost( loss[ v ] );
        }
    }
    if constexpr ( MODE != 0 ) {
        auto gsrc = [&] {
            if constexpr ( MODE == 1 )
                return grad_plan( batch[ v0 ] ).src_dist;
            else
                return grad_plan;
        }();
        for ( SI d = 0; d < D; ++d )
            src0.add_point_grad( gsrc, i, d, gp[ d ] );
        src0.add_half_width_grad( gsrc, i, gw );
    }
}

/// float points: per tent i, over the views of the chunk, as `gather_kernel` -- the point read once, the views'
/// normals, grids and cotangents staged in shared memory by tiles ( no aggregate built per view and per tent )
template<int MODE,class Plan,class Grad,class Batch>
__global__ void __launch_bounds__( GR_BLOCK ) gather_f_kernel( Plan plan, Grad grad_plan, Batch batch, SI v0, SI V, SI n, SI M, const double *tot,
                                                            const Grid1d *grids, const float *dgf, const double *loss ) {
    using Src = std::decay_t<decltype( plan( batch[ 0 ] ).src_dist )>;
    constexpr SI D = Src::ct_proj, TILE = GR_BLOCK;
    constexpr bool with_bary = DECAYED_TYPE_OF( plan( batch[ 0 ] ).barycenters )::is_valid;
    __shared__ float sh_n[ TILE * D ], sh_s0[ TILE ], sh_h[ TILE ], sh_g[ TILE ];
    const SI i = SI( blockIdx.x ) * GR_BLOCK + threadIdx.x;
    const bool act = i < n;
    const auto src0 = plan( batch[ v0 ] ).src_dist;
    static_assert( ot1d_card::shared_half_width_v<std::decay_t<decltype( src0 )>>, "the card's kernels need half-widths shared by the items" );
    float p[ D ], w = 1, m = 0;
    if ( act ) {
        for ( SI d = 0; d < D; ++d )
            p[ d ] = float( src0.point( i, d ) );
        w = float( src0.half_width( i ) );
        m = float( double( src0.mass( i ) ) / *tot );
    }
    double gp[ D ], gw = 0;
    for ( SI d = 0; d < D; ++d )
        gp[ d ] = 0;
    for ( SI t0 = 0; t0 < V; t0 += TILE ) {
        const SI nt = V - t0 < TILE ? V - t0 : TILE;
        __syncthreads();
        if ( SI( threadIdx.x ) < nt ) {
            const SI v = t0 + threadIdx.x;
            auto item = plan( batch[ v0 + v ] );
            for ( SI d = 0; d < D; ++d )
                sh_n[ threadIdx.x * D + d ] = float( item.src_dist.normal_component( d ) );
            sh_s0[ threadIdx.x ] = float( grids[ v ].s_min );
            sh_h[ threadIdx.x ] = float( grids[ v ].h );
            if constexpr ( MODE == 1 )
                sh_g[ threadIdx.x ] = float( grad_plan( batch[ v0 + v ] ).cost );
            if constexpr ( MODE != 1 )
                if ( blockIdx.x == 0 )
                    item.write_cost( loss[ v ] );
        }
        __syncthreads();
        if ( ! act )
            continue;
        for ( SI u = 0; u < nt; ++u ) {
            float s = 0;
            for ( SI d = 0; d < D; ++d )
                s += p[ d ] * sh_n[ u * D + d ];
            float ds, dw;
            gather_direct_f( sh_s0[ u ], sh_h[ u ], M, s, w, m, dgf + ( t0 + u ) * ( M + 1 ), ds, dw );
            if constexpr ( MODE != 0 ) {
                const float g = MODE == 1 ? sh_g[ u ] : 1.0f;
                for ( SI d = 0; d < D; ++d )
                    gp[ d ] += double( g * ds * sh_n[ u * D + d ] );
                gw += double( g * dw );
            }
            if constexpr ( MODE != 1 && with_bary )
                plan( batch[ v0 + t0 + u ] ).write_tent( i, moments_from( s, w, m, ds, dw ) );
        }
    }
    if constexpr ( MODE != 0 ) {
        if ( ! act )
            return;
        auto gsrc = [&] {
            if constexpr ( MODE == 1 )
                return grad_plan( batch[ v0 ] ).src_dist;
            else
                return grad_plan;
        }();
        for ( SI d = 0; d < D; ++d )
            src0.add_point_grad( gsrc, i, d, gp[ d ] );
        src0.add_half_width_grad( gsrc, i, gw );
    }
}

} // namespace ot1d_grid

/// card, the grid method: the views by chunks ( a memory budget, `SDOT_OT1D_BUDGET_MB` ), each by four launches
template<class Queue,class Grad,class SG = std::nullptr_t> requires ( ! std::is_same_v<Queue,CpuQueue> )
void continuous_ot_1d_grid( const Queue &queue, const auto &batch, auto &&plan, Grad &&grad_plan, auto &allocator, int scale, SG &&sg = nullptr ) {
    using namespace ot1d_grid;
    constexpr bool bwd = ! std::is_same_v<std::decay_t<Grad>,std::nullptr_t>;
    constexpr bool with_sg = ! std::is_same_v<std::decay_t<SG>,std::nullptr_t>;
    if constexpr ( with_sg ) {
        sg.points.fill_with( queue, 0 );
        sg.half_widths.fill_with( queue, 0 );
    }
    const SI nb_items = SI( batch.size() );
    if ( nb_items == 0 )
        return;
    const auto item0 = plan( batch[ 0 ] );
    const SI n = item0.src_dist.nb_tents(), nbin = SI( item0.dst_dist.values.size() ), M = nbin * scale;
    if ( n == 0 )
        return;
    const SI per_view = ( 4 * M + 2 + nbin + 2 + 4 * M + 3 * ( M + 2 ) ) * SI( sizeof( double ) ) + SI( sizeof( Grid1d ) );
    SI budget = SI( 1 ) << 30;
    if ( const char *e = std::getenv( "SDOT_OT1D_BUDGET_MB" ) )
        budget = SI( std::atol( e ) ) << 20;
    const SI V = std::max<SI>( 1, std::min<SI>( nb_items, budget / per_view ) );

    double *tot   = allocator.template raw<double>( 1 );
    double *cdf   = allocator.template raw<double>( V * ( nbin + 1 ) );
    double *nu    = allocator.template raw<double>( V * M );
    double *Cn    = allocator.template raw<double>( V * ( M + 1 ) );
    double *mu    = allocator.template raw<double>( V * M );
    double *dg    = allocator.template raw<double>( V * ( M + 1 ) );
    double *loss  = allocator.template raw<double>( V );
    double *K     = allocator.template raw<double>( V * 4 * M );     // per view: the kinks' A, B, Q, the direct masses
    int    *wide  = allocator.template raw<int>( 1 );
    using TFp = typename std::decay_t<decltype( item0.src_dist )>::TF;
    constexpr bool f32 = sizeof( TFp ) == 4;    // float points: the per-tent work in float ( see above )
    using Acc = std::conditional_t<f32,float,double>;
    double *P     = f32 ? nullptr : allocator.template raw<double>( V * 3 * ( M + 2 ) );
    float  *dgf   = f32 ? allocator.template raw<float>( V * ( M + 1 ) ) : nullptr;
    Grid1d *grids = allocator.template raw<Grid1d>( V );
    if ( allocator.failed )
        return;
    // float points: the narrow tents deposited directly, when their histogram fits in shared memory
    const float narrow = f32 && M * SI( sizeof( float ) ) <= SHARED_HIST_BYTES ? NARROW_CELLS : 0.0f;

    const auto kplan  = kernel_form( queue, MutList(), plan );
    const auto kbatch = kernel_form( queue, MutList(), batch );
    using PlanT = std::decay_t<decltype( kplan )>;
    using BatchT = std::decay_t<decltype( kbatch )>;
    launch_kernel( queue, &ot1d_card::total_mass_kernel<PlanT,BatchT>, 1, ot1d_card::OT_BLOCK, 0, kplan, kbatch, tot );
    for ( SI v0 = 0; v0 < nb_items; v0 += V ) {
        const SI Vc = std::min<SI>( V, nb_items - v0 );
        launch_kernel( queue, &targets_kernel<PlanT,BatchT>, int( Vc ), GR_BLOCK, 0, kplan, kbatch, v0, Vc, nbin, scale, cdf, nu, Cn, K, grids,
                       narrow > 0 ? wide : nullptr );
        // tiles per view: enough blocks for the card, a few hundred tents per thread at most
        const SI tiles = std::max<SI>( 1, std::min<SI>( ( n + 4 * GR_BLOCK - 1 ) / ( 4 * GR_BLOCK ), ( 8192 + Vc - 1 ) / Vc ) );
        if ( narrow > 0 ) {
            launch_kernel( queue, &wide_kernel<PlanT,BatchT>, gr_blocks( n ), GR_BLOCK, 0, kplan, kbatch, v0, n, ( const Grid1d * )grids, wide, narrow );
            launch_kernel( queue, &direct_shared_kernel<PlanT,BatchT>, int( Vc * tiles ), GR_BLOCK, int( M * sizeof( float ) ), kplan, kbatch, v0, Vc, n, M,
                           tiles, ( const double * )tot, ( const Grid1d * )grids, K, narrow );
        }
        const SI shared = M * SI( sizeof( double ) + 2 * sizeof( Acc ) );
        if ( shared <= SHARED_HIST_BYTES )
            launch_kernel( queue, &kinks_shared_kernel<Acc,PlanT,BatchT>, int( Vc * tiles ), GR_BLOCK, int( shared ), kplan, kbatch, v0, Vc, n, M,
                           tiles, ( const double * )tot, ( const Grid1d * )grids, K, ( const int * )( narrow > 0 ? wide : nullptr ), narrow );
        else
            launch_kernel( queue, &kinks_kernel<PlanT,BatchT>, gr_blocks( Vc * n ), GR_BLOCK, 0, kplan, kbatch, v0, Vc, n, M, ( const double * )tot,
                           ( const Grid1d * )grids, K, narrow );
        launch_kernel( queue, &masses_kernel<>, gr_blocks( Vc ), GR_BLOCK, 0, Vc, M, ( const Grid1d * )grids, ( const double * )K, mu );
        // the grid transport in grid units ( h = 1 ), then scaled by h^2 in the gather ( the grids may differ per view )
        Rows3<double> MU{ mu, M }, NU{ nu, M }, CN{ Cn, M + 1 }, DG{ dg, M + 1 };
        Rows2<double> LO{ loss };
        launch_kernel( queue, &projected1d::grid_kernel<Rows3<double>,Rows3<double>,Rows3<double>,Rows2<double>,Rows3<double>>,
                       int( Vc ), projected1d::GRID_BLOCK, 0, MU, NU, CN, LO, DG, Vc, M, 1.0 );
        launch_kernel( queue, &scale_kernel<>, gr_blocks( Vc * ( M + 2 ) ), GR_BLOCK, 0, Vc, M, ( const Grid1d * )grids, dg, loss, dgf );
        if constexpr ( ! f32 )
            launch_kernel( queue, &moments_kernel<>, gr_blocks( Vc ), GR_BLOCK, 0, Vc, M, ( const double * )dg, P );
        // the gather: in float for float points ( `gather_f_kernel` ), by prefix sums otherwise
        auto gather = [&]<int MODE>( const auto &kgrad ) {
            using GradT = std::decay_t<decltype( kgrad )>;
            if constexpr ( f32 )
                launch_kernel( queue, &gather_f_kernel<MODE,PlanT,GradT,BatchT>, gr_blocks( n ), GR_BLOCK, 0, kplan, kgrad, kbatch, v0, Vc, n, M,
                               ( const double * )tot, ( const Grid1d * )grids, ( const float * )dgf, ( const double * )loss );
            else
                launch_kernel( queue, &gather_kernel<MODE,PlanT,GradT,BatchT>, gr_blocks( n ), GR_BLOCK, 0, kplan, kgrad, kbatch, v0, Vc, n, M,
                               ( const double * )tot, ( const Grid1d * )grids, ( const double * )P, ( const float * )dgf, ( const double * )loss );
        };
        if constexpr ( bwd )
            gather.template operator()<1>( kernel_form( queue, MutList(), grad_plan ) );
        else if constexpr ( with_sg )
            gather.template operator()<2>( ot1d_coop::SumGrad{ kernel_form( queue, MutList(), sg.points ), kernel_form( queue, MutList(), sg.half_widths ) } );
        else
            gather.template operator()<0>( kplan );
    }
}

#endif

}
