#pragma once

// =====================================================================================
// THE FUSED PASSES OVER THE OBJECTS of `sdot.projected1d` ( see its docstring for the loss ):
//
//   * DEPOSIT: hist( a, j ) += the exact mass, divided by N, that object i projected on angle a puts in the fine cell j;
//   * GATHER:  ( dL/dpoints, dL/dradii ) from g( a, j ) = dL/dhist( a, j ).
//
// The projection s = p.n_a is recomputed in both passes ( never an [ angles, N ] array ). Same formulas, same rounding
// order as the pure jax version: an object covers the cells j0 .. j0 + L - 1, j0 = floor( ( s - w - s_min ) / h ) clamped to
// [ 0, M - L ], and the grid's two extreme edges are sent to -inf / +inf ( the mass of an object that leaves the grid stays
// in its border cells ).
//
// The profiles ( SHAPE ): 0 = tent of half-width h ( a dirac seen by the grid ), 1 = dome ( 1 - t^2 )^K. `params`:
// [ s_min, h, max_w, inv_n, c_K, a_0 .. a_K ], G( t ) = 1/2 + sum_j a_j t^( 2j + 1 ), f( t ) = c_K ( 1 - t^2 )^K.
// K = 0 stands for the HALF power: f( t ) = c sqrt( 1 - t^2 ), c = 2 / pi, the projection of a disc of density 1 ( an
// ellipse as well ), G( t ) = 1/2 + ( t sqrt( 1 - t^2 ) + asin( t ) ) / pi.
//
// CPU: the deposit splits the ANGLES between the threads ( each owns its rows: no atomics ), objects by blocks so that a
// block stays in cache across the angles of a thread; the gather splits the OBJECTS. CUDA: one thread per object and per
// tile of angles, atomics into the histogram ( global memory: a row of 16k cells does not fit in shared memory ), the
// gradient accumulated in registers over the tile.
// =====================================================================================

#include <loom/support/common_macros.h>
#include <loom/support/kernels/CpuQueue.h>

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>

#if defined( __CUDACC__ )
#define SDOT_P1D_HD __host__ __device__ __forceinline__
#else
#define SDOT_P1D_HD inline
#endif

namespace sdot {
namespace projected1d {

/// a strided view ( byte strides, as `TensorView` keeps them ), readable on the host and on the card
template<class T,int N>
struct View {
    char *p = nullptr;
    SI    s[ N ] = {};

    SDOT_P1D_HD T &operator()( SI i ) const { return *reinterpret_cast<T *>( p + i * s[ 0 ] ); }
    SDOT_P1D_HD T &operator()( SI i, SI j ) const { return *reinterpret_cast<T *>( p + i * s[ 0 ] + j * s[ 1 ] ); }
    SDOT_P1D_HD T &operator()( SI i, SI j, SI k ) const { return *reinterpret_cast<T *>( p + i * s[ 0 ] + j * s[ 1 ] + k * s[ 2 ] ); }

    /// the slice at index `b` of the first axis ( the batch )
    SDOT_P1D_HD View<T,N-1> operator[]( SI b ) const {
        View<T,N-1> r;
        r.p = p + b * s[ 0 ];
        for ( int d = 1; d < N; ++d )
            r.s[ d - 1 ] = s[ d ];
        return r;
    }
};

template<class V>
auto view( const V &v ) {
    using T = std::remove_const_t<typename std::decay_t<V>::TF>;
    constexpr int N = std::decay_t<V>::ct_rank;
    View<T,N> res;
    res.p = reinterpret_cast<char *>( const_cast<T *>( v.data().raw ) );
    [&]<int... I>( std::integer_sequence<int,I...> ) {
        ( ( res.s[ I ] = SI( v._strides[ Ct<int,I>() ] ) ), ... );
    }( std::make_integer_sequence<int,N>() );
    return res;
}

constexpr int MAX_K = 8;

/// what every thread needs: the geometry and the profile
template<class TF>
struct Params {
    TF  s_min, h, max_w, inv_n, c, a[ MAX_K + 1 ];
    SI  M, A, N;

    template<int SHAPE,int K>
    SDOT_P1D_HD TF G( TF t ) const {
        const TF u = t < TF( -1 ) ? TF( -1 ) : ( t > TF( 1 ) ? TF( 1 ) : t );
        if constexpr ( SHAPE == 0 ) {
            return TF( 0.5 ) + u - u * ( u < 0 ? -u : u ) / 2;
        } else if constexpr ( K == 0 ) {
            return TF( 0.5 ) + c / 2 * ( u * TF( sqrt( 1 - u * u ) ) + TF( asin( u ) ) );
        } else {
            const TF u2 = u * u;
            TF s = TF( 0.5 ), p = u;
            for ( int j = 0; j <= K; ++j ) {
                s = s + a[ j ] * p;
                p = p * u2;
            }
            return s;
        }
    }

    template<int SHAPE,int K>
    SDOT_P1D_HD TF f( TF t ) const {
        if constexpr ( SHAPE == 0 ) {
            const TF v = TF( 1 ) - ( t < 0 ? -t : t );
            return v > 0 ? v : TF( 0 );
        } else {
            const TF v = TF( 1 ) - t * t;
            if ( ! ( v > 0 ) )
                return TF( 0 );
            TF r = c;
            if constexpr ( K == 0 )
                return r * TF( sqrt( v ) );
            for ( int j = 0; j < K; ++j )
                r = r * v;
            return r;
        }
    }

    /// the half-width of an object
    template<int SHAPE>
    SDOT_P1D_HD TF half_width( TF r ) const {
        if constexpr ( SHAPE == 0 ) return h;
        else return r < max_w ? r : max_w;
    }

    /// the first cell j0 and the edge parameters t_l, l = 0 .. L, of an object at s of half-width w
    template<int L>
    SDOT_P1D_HD SI window( TF s, TF w, TF *t ) const {
        SI j0 = SI( floor( ( s - w - s_min ) / h ) );
        j0 = j0 < 0 ? 0 : ( j0 > M - L ? M - L : j0 );
        for ( int l = 0; l <= L; ++l ) {
            const SI e = j0 + l;
            t[ l ] = e == 0 ? TF( -2 ) : ( e == M ? TF( 2 ) : ( s_min + TF( e ) * h - s ) / w );
        }
        return j0;
    }
};

template<class TF,class VP>
SDOT_P1D_HD Params<TF> read_params( const VP &params, SI M, SI A, SI N, int K ) {
    Params<TF> pr;
    pr.s_min = TF( params( 0 ) ); pr.h = TF( params( 1 ) ); pr.max_w = TF( params( 2 ) ); pr.inv_n = TF( params( 3 ) );
    pr.c = TF( params( 4 ) );
    for ( int j = 0; j <= MAX_K; ++j )
        pr.a[ j ] = j <= K ? TF( params( 5 + j ) ) : TF( 0 );
    pr.M = M; pr.A = A; pr.N = N;
    return pr;
}

// ── one object, one angle ─────────────────────────────────────────────────────

template<int SHAPE,int K,int L,class TF,class Hist>
SDOT_P1D_HD void deposit_one( const Params<TF> &pr, const Hist &hist, SI a, TF s, TF w, TF mass, bool atomic ) {
    TF t[ L + 1 ], Gv[ L + 1 ];
    const SI j0 = pr.template window<L>( s, w, t );
    for ( int l = 0; l <= L; ++l )
        Gv[ l ] = pr.template G<SHAPE,K>( t[ l ] );
    for ( int l = 0; l < L; ++l ) {
        const TF m = ( Gv[ l + 1 ] - Gv[ l ] ) * mass;
        if ( m != TF( 0 ) ) {
#if defined( __CUDA_ARCH__ )
            if ( atomic ) { atomicAdd( &hist( a, j0 + l ), m ); continue; }
#endif
            hist( a, j0 + l ) += m;
        }
    }
}

/// ( d/ds, d/dw ) of sum_l g( a, j0 + l ) m_l, without the factor inv_n / w, from the DIFFERENCES of g between
/// neighbour cells, dg( a, e ) = g_e - g_e-1 on the edges e = 0 .. M ( `grid` writes them, computed in double: g is
/// a potential, large against its differences, which are what the gradient is made of ). Summing by parts,
/// sum_l g_l ( f_l - f_l+1 ) = sum_e f( t_e ) dg_e over the window's edges ( f vanishes on its two end edges ).
template<int SHAPE,int K,int L,class TF,class DG>
SDOT_P1D_HD void gather_one( const Params<TF> &pr, const DG &dg, SI a, TF s, TF w, TF &ds, TF &dw ) {
    TF t[ L + 1 ];
    const SI j0 = pr.template window<L>( s, w, t );
    TF a_s = 0, a_w = 0;
    for ( int l = 0; l <= L; ++l ) {
        const TF d = TF( dg( a, j0 + l ) );
        const TF fv = pr.template f<SHAPE,K>( t[ l ] );
        a_s += d * fv;
        if ( t[ l ] >= TF( -1 ) && t[ l ] <= TF( 1 ) )
            a_w += d * t[ l ] * fv;
    }
    ds = a_s;
    dw = a_w;
}

// ── the grid transport, one ( problem, angle ) row ─────────────────────────────
//
// W2^2 between mu ( cell masses, normalized here ) and nu ( cell masses of total 1, Cn their cumulated sums ), both
// uniform inside the cells of width h, and its derivative with respect to the masses mu. One sweep from the last cell
// down, in double:
//   * within cell i the transport map T( s ) = Q_nu( F_mu( s ) ) is piecewise linear ( cut where F_mu crosses a Cn );
//   * L = sum int ( s - T )^2 dmu, and with I0_i = 2 int_cell ( s - T ) ds, I1_i = ( 2 / h ) int_cell ( s - T )( s - e_i ) ds,
//     dL/dmu_i = - sum_{j>i} I0_j - I1_i ( the cell average of the Kantorovich potential, up to a constant ); an EMPTY
//     cell gets the limit of that formula ( T constant over it ), not a zero;
//   * the normalization mu' = mu / S: g_i = ( G_i - sum_j G_j mu'_j ) / S.
// Positions are relative to the grid's start ( the loss does not depend on the origin ).

/// the contributions of cell i ( normalized mass m on [ Cm_lo, Cm_hi ], at e = i h ) to the loss, I0 and I1; `k` is
/// the target cell where the sweep stands ( Cn( k ) <= Cm_lo ), moved forward
template<class VN,class VC>
SDOT_P1D_HD void cell_terms( double m, double Cm_lo, double Cm_hi, double e, SI &k, const VN &nu, const VC &Cn, SI M, double h,
                             double &I0, double &I1, double &loss ) {
    I0 = 0;
    I1 = 0;
    if ( ! ( m > 0 ) ) {
        while ( k < M - 1 && double( Cn( k + 1 ) ) <= Cm_lo )
            ++k;
        const double nk = double( nu( k ) );
        const double Tc = double( k ) * h + ( nk > 0 ? ( Cm_lo - double( Cn( k ) ) ) * h / nk : 0.0 );
        I0 = 2 * h * ( e + h / 2 - Tc );
        I1 = 2 * h * h / 3 + ( e - Tc ) * h;
        return;
    }
    double tl = Cm_lo;
    while ( true ) {
        while ( k < M - 1 && double( Cn( k + 1 ) ) <= tl )
            ++k;
        const double ck = double( Cn( k ) );
        const double cn = double( Cn( k + 1 ) );
        const double th = k == M - 1 || cn > Cm_hi ? Cm_hi : cn;
        const double nk = double( nu( k ) );
        if ( nk > 0 && th > tl ) {
            const double s0 = e + ( tl - Cm_lo ) * h / m, s1 = e + ( th - Cm_lo ) * h / m;
            const double T0 = double( k ) * h + ( tl - ck ) * h / nk, T1 = double( k ) * h + ( th - ck ) * h / nk;
            const double d0 = s0 - T0, d1 = s1 - T1, u0 = s0 - e, u1 = s1 - e;
            loss += ( th - tl ) * ( d0 * d0 + d0 * d1 + d1 * d1 ) / 3;
            I0 += ( s1 - s0 ) * ( d0 + d1 );
            I1 += ( 2 / h ) * ( s1 - s0 ) * ( 2 * d0 * u0 + d0 * u1 + d1 * u0 + 2 * d1 * u1 ) / 6;
        }
        if ( th >= Cm_hi || k == M - 1 )
            break;
        tl = th;
    }
}

/// the first target cell of a sweep starting at the cumulated mass c: the last k with Cn( k ) <= c
template<class VC>
SDOT_P1D_HD SI start_cell( const VC &Cn, SI M, double c ) {
    SI lo = 0, hi = M - 1;
    while ( lo < hi ) {
        const SI mid = ( lo + hi + 1 ) / 2;
        if ( double( Cn( mid ) ) <= c ) lo = mid; else hi = mid - 1;
    }
    return lo;
}

/// one row, sequentially ( the CPU's ): returns the loss, writes dg [ M + 1 ]
template<class TM,class VN,class VC,class TD>
double grid_row( const TM &mu, const VN &nu, const VC &Cn, TD &dg, SI M, double h ) {
    double S = 0;
    for ( SI i = 0; i < M; ++i )
        S += double( mu( i ) );
    const double inv_S = S > 0 ? 1 / S : 0;
    double loss = 0, total = 0, I0, I1;
    SI k = 0;
    double Cm = 0;
    for ( SI i = 0; i < M; ++i ) {
        const double m = double( mu( i ) ) * inv_S, hi = i == M - 1 ? 1.0 : Cm + m;
        cell_terms( m, Cm, hi, double( i ) * h, k, nu, Cn, M, h, I0, I1, loss );
        total += I0;
        Cm = hi;
    }
    // g_i = - sum_{j>i} I0_j - I1_i; dg_e = ( g_e - g_e-1 ) / S ( the normalization's constant cancels )
    double prefix = 0, G_prev = 0, dummy = 0;
    k = 0;
    Cm = 0;
    dg( 0 ) = 0;
    dg( M ) = 0;
    for ( SI i = 0; i < M; ++i ) {
        const double m = double( mu( i ) ) * inv_S, hi = i == M - 1 ? 1.0 : Cm + m;
        cell_terms( m, Cm, hi, double( i ) * h, k, nu, Cn, M, h, I0, I1, dummy );
        prefix += I0;
        const double G = - ( total - prefix ) - I1;
        if ( i > 0 )
            dg( i ) = ( G - G_prev ) * inv_S;
        G_prev = G;
        Cm = hi;
    }
    return loss;
}

/// a row of a View<T,3> as a 1D accessor
template<class V>
struct Row {
    V  v;
    SI b, a;
    SDOT_P1D_HD auto &operator()( SI i ) const { return v( b, a, i ); }
};

// ── CPU ───────────────────────────────────────────────────────────────────────

template<class F>
void par_ranges( const CpuQueue &queue, SI n, SI min_per_thread, F &&f ) {
    const int nt = int( std::min<SI>( std::max( queue.nb_workers(), 1 ), std::max<SI>( n / std::max<SI>( min_per_thread, 1 ), 1 ) ) );
    if ( nt <= 1 ) {
        f( SI( 0 ), n );
        return;
    }
    queue.run_threads( nt, [&]( int t ) { f( n * t / nt, n * ( t + 1 ) / nt ); } );
}

// The tensors carry a BATCH axis in front ( K independent problems of the same geometry, N objects each ):
// points [ K, N, 2 ], radii [ K, N ], hist and g [ K, A, M ], grad_points [ K, N, 2 ], grad_radii [ K, N ].

template<int SHAPE,int K,int L>
void deposit( const CpuQueue &queue, const auto &points, const auto &radii, const auto &normals, const auto &params, auto &&hist ) {
    using TF = std::remove_const_t<typename std::decay_t<decltype( points )>::TF>;
    const auto P = view( points );
    const auto R = view( radii );
    const auto Nr = view( normals );
    const auto H = view( hist );
    const SI NB = points.shape( 0 ), N = points.shape( 1 ), A = normals.shape( 0 ), M = hist.shape( 2 );
    const Params<TF> pr = read_params<TF>( view( params ), M, A, N, K );
    constexpr SI OBJ_BLOCK = 2048;
    // each thread owns a range of ( problem, angle ) rows: no atomics
    par_ranges( queue, NB * A, 1, [&]( SI r0, SI r1 ) {
        for ( SI r = r0; r < r1; ) {
            const SI b = r / A, a0 = r % A, a1 = std::min( A, a0 + ( r1 - r ) );
            const auto Pb = P[ b ];
            const auto Rb = R[ b ];
            const auto Hb = H[ b ];
            for ( SI i0 = 0; i0 < N; i0 += OBJ_BLOCK ) {
                const SI i1 = std::min( N, i0 + OBJ_BLOCK );
                for ( SI a = a0; a < a1; ++a ) {
                    const TF nx = TF( Nr( a, 0 ) ), ny = TF( Nr( a, 1 ) );
                    for ( SI i = i0; i < i1; ++i ) {
                        const TF s = TF( Pb( i, 0 ) ) * nx + TF( Pb( i, 1 ) ) * ny;
                        deposit_one<SHAPE,K,L>( pr, Hb, a, s, pr.template half_width<SHAPE>( TF( Rb( i ) ) ), pr.inv_n, false );
                    }
                }
            }
            r += a1 - a0;
        }
    } );
}

template<int SHAPE,int K,int L>
void gather( const CpuQueue &queue, const auto &points, const auto &radii, const auto &normals, const auto &params,
             const auto &g, auto &&grad_points, auto &&grad_radii ) {
    using TF = std::remove_const_t<typename std::decay_t<decltype( points )>::TF>;
    const auto P = view( points );
    const auto R = view( radii );
    const auto Nr = view( normals );
    const auto Gr = view( g );
    const auto GP = view( grad_points );
    const auto GR = view( grad_radii );
    const SI NB = points.shape( 0 ), N = points.shape( 1 ), A = normals.shape( 0 ), M = g.shape( 2 );
    const Params<TF> pr = read_params<TF>( view( params ), M, A, N, K );
    par_ranges( queue, NB * N, 1024, [&]( SI q0, SI q1 ) {
        for ( SI q = q0; q < q1; ++q ) {
            const SI b = q / N, i = q % N;
            const TF px = TF( P( b, i, 0 ) ), py = TF( P( b, i, 1 ) ), r = TF( R( b, i ) );
            const TF w = pr.template half_width<SHAPE>( r );
            const auto Gb = Gr[ b ];
            TF gx = 0, gy = 0, gw = 0;
            for ( SI a = 0; a < A; ++a ) {
                const TF nx = TF( Nr( a, 0 ) ), ny = TF( Nr( a, 1 ) );
                TF ds, dw;
                gather_one<SHAPE,K,L>( pr, Gb, a, px * nx + py * ny, w, ds, dw );
                gx += ds * nx;
                gy += ds * ny;
                gw += dw;
            }
            const TF sc = pr.inv_n / w;
            GP( b, i, 0 ) = gx * sc;
            GP( b, i, 1 ) = gy * sc;
            GR( b, i ) = ( SHAPE == 0 || ! ( r < pr.max_w ) ) ? TF( 0 ) : gw * sc;
        }
    } );
}

/// the grid transport of every ( problem, angle ) row: loss [ K, A ] and dg [ K, A, M + 1 ], the differences between
/// neighbour cells of dL/dmu ( see `gather_one` ). In GRID UNITS ( h = 1 ): the caller multiplies both by h^2 -- the
/// kernels then need no scalar from the host ( the params live on the card ).
void grid( const CpuQueue &queue, const auto &mu, const auto &nu, const auto &Cn, const auto &params, auto &&loss, auto &&dg ) {
    const auto MU = view( mu );
    const auto NU = view( nu );
    const auto CN = view( Cn );
    const auto LO = view( loss );
    const auto DG = view( dg );
    const SI NB = mu.shape( 0 ), A = mu.shape( 1 ), M = mu.shape( 2 );
    par_ranges( queue, NB * A, 1, [&]( SI r0, SI r1 ) {
        for ( SI r = r0; r < r1; ++r ) {
            const SI b = r / A, a = r % A;
            Row<decltype( MU )> rm{ MU, b, a };
            Row<decltype( NU )> rn{ NU, b, a };
            Row<decltype( CN )> rc{ CN, b, a };
            Row<decltype( DG )> rd{ DG, b, a };
            LO( b, a ) = grid_row( rm, rn, rc, rd, M, 1.0 );
        }
    } );
}

// ── ELLIPSES ──────────────────────────────────────────────────────────────────
//
// The same passes for objects that are ELLIPSES with a mass of their own: the image of the unit dome by a matrix M, given
// as C = M M^T = ( c00, c11, c01 ) per object. Along an angle of normal n its projection is the dome of half-width
// w = sqrt( n.C n ), kept in [ min_w, max_w ] ( L, the cells an object may cover, is a static size ).
//
//   covs [ K, N, 3 ], masses [ K, N ]; `params` as above, with MIN_W in place of inv_n ( the masses say it );
//   deposit: hist( a, j ) += mass_i x the exact share of object i in cell j;
//   gather:  ( dL/dpoints, dL/dcovs, dL/dmasses ) from g( a, j ) = dL/dhist( a, j ) ITSELF ( not its differences: the
//            derivative with respect to a mass is a sum of g, not of differences; in double ).

/// the half-width along ( nx, ny ), and whether it is strictly inside its bounds ( its derivative then exists )
template<class TF>
SDOT_P1D_HD TF ellipse_width( TF c00, TF c11, TF c01, TF nx, TF ny, TF min_w, TF max_w, bool &inside ) {
    const TF w2 = c00 * nx * nx + c11 * ny * ny + 2 * c01 * nx * ny;
    const TF w = w2 > 0 ? TF( sqrt( w2 ) ) : TF( 0 );
    inside = w > min_w && w < max_w;
    return w < min_w ? min_w : ( w > max_w ? max_w : w );
}

/// for one object and one angle: ( d/ds, d/dw ) of sum_l g( a, j0 + l ) m_l without the factor mass / w ( as `gather_one`,
/// the differences of g taken here ), and d/dmass = sum_l g( a, j0 + l ) x the share of cell l
template<int K,int L,class TF,class VG>
SDOT_P1D_HD void gather_ellipse_one( const Params<TF> &pr, const VG &g, SI a, TF s, TF w, TF &ds, TF &dw, TF &dm ) {
    TF t[ L + 1 ];
    const SI j0 = pr.template window<L>( s, w, t );
    TF a_s = 0, a_w = 0, a_m = 0, G_prev = pr.template G<1,K>( t[ 0 ] );
    for ( int l = 0; l <= L; ++l ) {
        const SI e = j0 + l;
        const TF d = ( e == 0 || e == pr.M ) ? TF( 0 ) : TF( g( a, e ) ) - TF( g( a, e - 1 ) );
        const TF fv = pr.template f<1,K>( t[ l ] );
        a_s += d * fv;
        if ( t[ l ] >= TF( -1 ) && t[ l ] <= TF( 1 ) )
            a_w += d * t[ l ] * fv;
        if ( l < L ) {
            const TF G_next = pr.template G<1,K>( t[ l + 1 ] );
            a_m += TF( g( a, e ) ) * ( G_next - G_prev );
            G_prev = G_next;
        }
    }
    ds = a_s;
    dw = a_w;
    dm = a_m;
}

/// what one object gathers over the angles a0 .. a1: ( dL/dpx, dL/dpy, dL/dc00, dL/dc11, dL/dc01, dL/dmass ) into `out`
template<int K,int L,class TF,class VN,class VG>
SDOT_P1D_HD void gather_ellipse( const Params<TF> &pr, const VN &Nr, const VG &g, SI a0, SI a1, TF px, TF py, TF c00, TF c11, TF c01, TF mass, TF *out ) {
    for ( int c = 0; c < 6; ++c )
        out[ c ] = 0;
    for ( SI a = a0; a < a1; ++a ) {
        const TF nx = TF( Nr( a, 0 ) ), ny = TF( Nr( a, 1 ) );
        bool inside;
        const TF w = ellipse_width( c00, c11, c01, nx, ny, pr.inv_n, pr.max_w, inside );
        TF ds, dw, dm;
        gather_ellipse_one<K,L>( pr, g, a, px * nx + py * ny, w, ds, dw, dm );
        const TF sc = mass / w;
        out[ 0 ] += ds * sc * nx;
        out[ 1 ] += ds * sc * ny;
        if ( inside ) {                          // dw/dC = n n^T / ( 2 w )
            const TF k = dw * sc / ( 2 * w );
            out[ 2 ] += k * nx * nx;
            out[ 3 ] += k * ny * ny;
            out[ 4 ] += k * 2 * nx * ny;
        }
        out[ 5 ] += dm;
    }
}

template<int K,int L>
void deposit_ellipses( const CpuQueue &queue, const auto &points, const auto &covs, const auto &masses, const auto &normals, const auto &params, auto &&hist ) {
    using TF = std::remove_const_t<typename std::decay_t<decltype( points )>::TF>;
    const auto P = view( points );
    const auto C = view( covs );
    const auto Ms = view( masses );
    const auto Nr = view( normals );
    const auto H = view( hist );
    const SI NB = points.shape( 0 ), N = points.shape( 1 ), A = normals.shape( 0 ), M = hist.shape( 2 );
    const Params<TF> pr = read_params<TF>( view( params ), M, A, N, K );
    constexpr SI OBJ_BLOCK = 2048;
    par_ranges( queue, NB * A, 1, [&]( SI r0, SI r1 ) {
        for ( SI r = r0; r < r1; ) {
            const SI b = r / A, a0 = r % A, a1 = std::min( A, a0 + ( r1 - r ) );
            const auto Hb = H[ b ];
            for ( SI i0 = 0; i0 < N; i0 += OBJ_BLOCK ) {
                const SI i1 = std::min( N, i0 + OBJ_BLOCK );
                for ( SI a = a0; a < a1; ++a ) {
                    const TF nx = TF( Nr( a, 0 ) ), ny = TF( Nr( a, 1 ) );
                    for ( SI i = i0; i < i1; ++i ) {
                        bool inside;
                        const TF w = ellipse_width( TF( C( b, i, 0 ) ), TF( C( b, i, 1 ) ), TF( C( b, i, 2 ) ), nx, ny, pr.inv_n, pr.max_w, inside );
                        deposit_one<1,K,L>( pr, Hb, a, TF( P( b, i, 0 ) ) * nx + TF( P( b, i, 1 ) ) * ny, w, TF( Ms( b, i ) ), false );
                    }
                }
            }
            r += a1 - a0;
        }
    } );
}

template<int K,int L>
void gather_ellipses( const CpuQueue &queue, const auto &points, const auto &covs, const auto &masses, const auto &normals, const auto &params,
                      const auto &g, auto &&grad_points, auto &&grad_covs, auto &&grad_masses ) {
    using TF = std::remove_const_t<typename std::decay_t<decltype( points )>::TF>;
    const auto P = view( points );
    const auto C = view( covs );
    const auto Ms = view( masses );
    const auto Nr = view( normals );
    const auto Gr = view( g );
    const auto GP = view( grad_points );
    const auto GC = view( grad_covs );
    const auto GM = view( grad_masses );
    const SI NB = points.shape( 0 ), N = points.shape( 1 ), A = normals.shape( 0 ), M = g.shape( 2 );
    const Params<TF> pr = read_params<TF>( view( params ), M, A, N, K );
    par_ranges( queue, NB * N, 1024, [&]( SI q0, SI q1 ) {
        for ( SI q = q0; q < q1; ++q ) {
            const SI b = q / N, i = q % N;
            TF out[ 6 ];
            gather_ellipse<K,L>( pr, Nr, Gr[ b ], SI( 0 ), A, TF( P( b, i, 0 ) ), TF( P( b, i, 1 ) ), TF( C( b, i, 0 ) ), TF( C( b, i, 1 ) ), TF( C( b, i, 2 ) ), TF( Ms( b, i ) ), out );
            GP( b, i, 0 ) = out[ 0 ]; GP( b, i, 1 ) = out[ 1 ];
            GC( b, i, 0 ) = out[ 2 ]; GC( b, i, 1 ) = out[ 3 ]; GC( b, i, 2 ) = out[ 4 ];
            GM( b, i ) = out[ 5 ];
        }
    } );
}

// ── POLYGONS ──────────────────────────────────────────────────────────────────
//
// The same passes for objects that are POLYGONS of V vertices ( counter-clockwise, simple; density `weight` inside ):
// along an angle the projection is piecewise linear, and the mass on the side s <= e of a cell edge is the sum over the
// sides of the polygon of a clipped trapezoid ( Green: area = - the integral of u ds along the border, u the coordinate
// across; the cut itself, where ds = 0, adds nothing ). No sort of the vertices.
//
//   verts [ K, N, 2 V ] ( x0, y0, x1, y1, .. ), weights [ K, N ]; `params`: [ s_min, h, .. ] ( the rest is not read ). A
//   polygon must not cover more than L cells;
//   deposit: hist( a, j ) += weight_i x the area of polygon i in cell j;
//   gather:  ( dL/dverts, dL/dweights ) from g( a, j ) = dL/dhist( a, j ) itself.

/// the integral of u ds along the part s <= e of the segment ( s0, u0 ) -> ( s1, u1 ); if `d`: its derivatives with
/// respect to ( s0, u0, s1, u1 )
template<class TF>
SDOT_P1D_HD TF clipped_trapezoid( TF s0, TF u0, TF s1, TF u1, TF e, TF *d ) {
    const bool sw = s0 > s1;                     // the reversed segment gives the opposite
    if ( sw ) {
        const TF ts = s0, tu = u0;
        s0 = s1; u0 = u1; s1 = ts; u1 = tu;
    }
    TF X = 0, ds0 = 0, du0 = 0, ds1 = 0, du1 = 0;
    if ( e >= s1 ) {
        X = ( s1 - s0 ) * ( u0 + u1 ) / 2;
        ds0 = - ( u0 + u1 ) / 2; ds1 = - ds0;
        du0 = du1 = ( s1 - s0 ) / 2;
    } else if ( e > s0 ) {
        const TF D = s1 - s0, q = e - s0, r = q / D, ue = u0 + ( u1 - u0 ) * r;
        X = q * ( u0 + ue ) / 2;
        du0 = q * ( 2 - r ) / 2;
        du1 = q * r / 2;
        ds0 = - ( u0 + ue ) / 2 + q * ( u1 - u0 ) * ( r - 1 ) / ( 2 * D );
        ds1 = - q * ( u1 - u0 ) * r / ( 2 * D );
    }
    if ( d ) {
        const TF sg = sw ? TF( -1 ) : TF( 1 );
        d[ sw ? 2 : 0 ] = sg * ds0; d[ sw ? 3 : 1 ] = sg * du0;
        d[ sw ? 0 : 2 ] = sg * ds1; d[ sw ? 1 : 3 ] = sg * du1;
    }
    return sw ? - X : X;
}

/// a polygon seen along ( nx, ny ): sv, uv [ V ] its vertices along and across the detector, e [ L + 1 ] the edges of
/// the cells it may cover ( the grid's two extreme edges sent to infinity ); returns the first cell
template<int V,int L,class TF>
SDOT_P1D_HD SI polygon_window( const Params<TF> &pr, const TF *xy, TF nx, TF ny, TF *sv, TF *uv, TF *e ) {
    TF lo = 0;
    for ( int v = 0; v < V; ++v ) {
        sv[ v ] = xy[ 2 * v ] * nx + xy[ 2 * v + 1 ] * ny;
        uv[ v ] = xy[ 2 * v + 1 ] * nx - xy[ 2 * v ] * ny;
        lo = v == 0 || sv[ v ] < lo ? sv[ v ] : lo;
    }
    SI j0 = SI( floor( ( lo - pr.s_min ) / pr.h ) );
    j0 = j0 < 0 ? 0 : ( j0 > pr.M - L ? pr.M - L : j0 );
    for ( int l = 0; l <= L; ++l ) {
        const SI c = j0 + l;
        e[ l ] = c == 0 ? TF( -1e30 ) : ( c == pr.M ? TF( 1e30 ) : pr.s_min + TF( c ) * pr.h );
    }
    return j0;
}

template<int V,int L,class TF,class Hist>
SDOT_P1D_HD void deposit_polygon_one( const Params<TF> &pr, const Hist &hist, SI a, const TF *xy, TF nx, TF ny, TF weight, bool atomic ) {
    TF sv[ V ], uv[ V ], e[ L + 1 ], A[ L + 1 ];
    const SI j0 = polygon_window<V,L>( pr, xy, nx, ny, sv, uv, e );
    for ( int l = 0; l <= L; ++l )
        A[ l ] = 0;
    for ( int v = 0; v < V; ++v ) {
        const int w = v + 1 < V ? v + 1 : 0;
        for ( int l = 0; l <= L; ++l )
            A[ l ] -= clipped_trapezoid<TF>( sv[ v ], uv[ v ], sv[ w ], uv[ w ], e[ l ], nullptr );
    }
    for ( int l = 0; l < L; ++l ) {
        const TF m = ( A[ l + 1 ] - A[ l ] ) * weight;
        if ( m != TF( 0 ) ) {
#if defined( __CUDA_ARCH__ )
            if ( atomic ) { atomicAdd( &hist( a, j0 + l ), m ); continue; }
#endif
            hist( a, j0 + l ) += m;
        }
    }
}

/// what one polygon gathers over the angles a0 .. a1: ( dL/dverts [ 2 V ], dL/dweight ) into `out`
template<int V,int L,class TF,class VN,class VG>
SDOT_P1D_HD void gather_polygon( const Params<TF> &pr, const VN &Nr, const VG &g, SI a0, SI a1, const TF *xy, TF weight, TF *out ) {
    for ( int c = 0; c <= 2 * V; ++c )
        out[ c ] = 0;
    for ( SI a = a0; a < a1; ++a ) {
        const TF nx = TF( Nr( a, 0 ) ), ny = TF( Nr( a, 1 ) );
        TF sv[ V ], uv[ V ], e[ L + 1 ], gs[ V ], gu[ V ], d[ 4 ];
        const SI j0 = polygon_window<V,L>( pr, xy, nx, ny, sv, uv, e );
        for ( int v = 0; v < V; ++v )
            gs[ v ] = gu[ v ] = 0;
        TF gw = 0;
        for ( int l = 0; l <= L; ++l ) {         // sum_l g_l ( A_l+1 - A_l ) = sum over the edges of c_l A_l
            const TF c = ( l < L ? TF( g( a, j0 + l ) ) : TF( 0 ) ) - ( l > 0 ? TF( g( a, j0 + l - 1 ) ) : TF( 0 ) );
            for ( int v = 0; v < V; ++v ) {
                const int w = v + 1 < V ? v + 1 : 0;
                gw += c * clipped_trapezoid<TF>( sv[ v ], uv[ v ], sv[ w ], uv[ w ], e[ l ], d );
                gs[ v ] += c * d[ 0 ]; gu[ v ] += c * d[ 1 ];
                gs[ w ] += c * d[ 2 ]; gu[ w ] += c * d[ 3 ];
            }
        }
        for ( int v = 0; v < V; ++v ) {
            out[ 2 * v     ] += weight * ( gs[ v ] * nx - gu[ v ] * ny );
            out[ 2 * v + 1 ] += weight * ( gs[ v ] * ny + gu[ v ] * nx );
        }
        out[ 2 * V ] += gw;
    }
}

template<int V,int L>
void deposit_polygons( const CpuQueue &queue, const auto &verts, const auto &weights, const auto &normals, const auto &params, auto &&hist ) {
    using TF = std::remove_const_t<typename std::decay_t<decltype( verts )>::TF>;
    const auto P = view( verts );
    const auto W = view( weights );
    const auto Nr = view( normals );
    const auto H = view( hist );
    const SI NB = verts.shape( 0 ), N = verts.shape( 1 ), A = normals.shape( 0 ), M = hist.shape( 2 );
    const Params<TF> pr = read_params<TF>( view( params ), M, A, N, 0 );
    par_ranges( queue, NB * A, 1, [&]( SI r0, SI r1 ) {
        for ( SI r = r0; r < r1; ++r ) {
            const SI b = r / A, a = r % A;
            const auto Hb = H[ b ];
            const TF nx = TF( Nr( a, 0 ) ), ny = TF( Nr( a, 1 ) );
            for ( SI i = 0; i < N; ++i ) {
                TF xy[ 2 * V ];
                for ( int c = 0; c < 2 * V; ++c )
                    xy[ c ] = TF( P( b, i, c ) );
                deposit_polygon_one<V,L>( pr, Hb, a, xy, nx, ny, TF( W( b, i ) ), false );
            }
        }
    } );
}

template<int V,int L>
void gather_polygons( const CpuQueue &queue, const auto &verts, const auto &weights, const auto &normals, const auto &params,
                      const auto &g, auto &&grad_verts, auto &&grad_weights ) {
    using TF = std::remove_const_t<typename std::decay_t<decltype( verts )>::TF>;
    const auto P = view( verts );
    const auto W = view( weights );
    const auto Nr = view( normals );
    const auto Gr = view( g );
    const auto GP = view( grad_verts );
    const auto GW = view( grad_weights );
    const SI NB = verts.shape( 0 ), N = verts.shape( 1 ), A = normals.shape( 0 ), M = g.shape( 2 );
    const Params<TF> pr = read_params<TF>( view( params ), M, A, N, 0 );
    par_ranges( queue, NB * N, 1024, [&]( SI q0, SI q1 ) {
        for ( SI q = q0; q < q1; ++q ) {
            const SI b = q / N, i = q % N;
            TF xy[ 2 * V ], out[ 2 * V + 1 ];
            for ( int c = 0; c < 2 * V; ++c )
                xy[ c ] = TF( P( b, i, c ) );
            gather_polygon<V,L>( pr, Nr, Gr[ b ], SI( 0 ), A, xy, TF( W( b, i ) ), out );
            for ( int c = 0; c < 2 * V; ++c )
                GP( b, i, c ) = out[ c ];
            GW( b, i ) = out[ 2 * V ];
        }
    } );
}

// ── CUDA ──────────────────────────────────────────────────────────────────────
#if defined( __CUDACC__ )

constexpr int BLOCK = 256;

/// the angle tiles: enough blocks to fill the card ( ~ 8 per SM x 68 SMs ), never more tiles than angles
inline SI angle_tiles( SI nb_obj_blocks, SI A ) {
    const SI want = ( 4096 + nb_obj_blocks - 1 ) / nb_obj_blocks;
    return std::max<SI>( 1, std::min<SI>( A, want ) );
}

/// block -> ( problem, block of objects, tile of angles )
struct BlockPos {
    SI b, ob, tile;
    __device__ BlockPos( SI nb_obj_blocks, SI tiles ) {
        SI q = SI( blockIdx.x );
        ob = q % nb_obj_blocks; q /= nb_obj_blocks;
        tile = q % tiles;
        b = q / tiles;
    }
};

template<int SHAPE,int K,int L,class TF,class VP,class VR,class VN,class VH>
__global__ void __launch_bounds__( BLOCK ) deposit_kernel( View<TF,1> PP, SI M, SI A, SI N, VP P, VR R, VN Nr, VH H, SI nb_obj_blocks, SI tiles ) {
    const Params<TF> pr = read_params<TF>( PP, M, A, N, K );
    const BlockPos bp( nb_obj_blocks, tiles );
    const SI i = bp.ob * BLOCK + threadIdx.x;
    if ( i >= pr.N )
        return;
    const TF px = TF( P( bp.b, i, 0 ) ), py = TF( P( bp.b, i, 1 ) );
    const TF w = pr.template half_width<SHAPE>( TF( R( bp.b, i ) ) );
    const auto Hb = H[ bp.b ];
    const SI a0 = pr.A * bp.tile / tiles, a1 = pr.A * ( bp.tile + 1 ) / tiles;
    for ( SI a = a0; a < a1; ++a ) {
        const TF s = px * TF( Nr( a, 0 ) ) + py * TF( Nr( a, 1 ) );
        deposit_one<SHAPE,K,L>( pr, Hb, a, s, w, pr.inv_n, true );
    }
}

template<int SHAPE,int K,int L,class TF,class VP,class VR,class VN,class VG,class VGP,class VGR>
__global__ void __launch_bounds__( BLOCK ) gather_kernel( View<TF,1> PP, SI M, SI A, SI N, VP P, VR R, VN Nr, VG Gr, VGP GP, VGR GR, SI nb_obj_blocks, SI tiles ) {
    const Params<TF> pr = read_params<TF>( PP, M, A, N, K );
    const BlockPos bp( nb_obj_blocks, tiles );
    const SI i = bp.ob * BLOCK + threadIdx.x;
    if ( i >= pr.N )
        return;
    const SI b = bp.b;
    const TF px = TF( P( b, i, 0 ) ), py = TF( P( b, i, 1 ) ), r = TF( R( b, i ) );
    const TF w = pr.template half_width<SHAPE>( r );
    const auto Gb = Gr[ b ];
    const SI a0 = pr.A * bp.tile / tiles, a1 = pr.A * ( bp.tile + 1 ) / tiles;
    TF gx = 0, gy = 0, gw = 0;
    for ( SI a = a0; a < a1; ++a ) {
        const TF nx = TF( Nr( a, 0 ) ), ny = TF( Nr( a, 1 ) );
        TF ds, dw;
        gather_one<SHAPE,K,L>( pr, Gb, a, px * nx + py * ny, w, ds, dw );
        gx += ds * nx;
        gy += ds * ny;
        gw += dw;
    }
    const TF sc = pr.inv_n / w;
    if ( tiles == 1 ) {
        GP( b, i, 0 ) = gx * sc;
        GP( b, i, 1 ) = gy * sc;
        GR( b, i ) = ( SHAPE == 0 || ! ( r < pr.max_w ) ) ? TF( 0 ) : gw * sc;
    } else {
        atomicAdd( &GP( b, i, 0 ), gx * sc );
        atomicAdd( &GP( b, i, 1 ), gy * sc );
        if ( SHAPE != 0 && r < pr.max_w )
            atomicAdd( &GR( b, i ), gw * sc );
    }
}

template<int SHAPE,int K,int L,class Queue> requires ( ! std::is_same_v<Queue,CpuQueue> )
void deposit( const Queue &queue, const auto &points, const auto &radii, const auto &normals, const auto &params, auto &&hist ) {
    using TF = std::remove_const_t<typename std::decay_t<decltype( points )>::TF>;
    const SI NB = points.shape( 0 ), N = points.shape( 1 ), A = normals.shape( 0 ), M = hist.shape( 2 );
    static_assert( std::is_same_v<std::remove_const_t<typename std::decay_t<decltype( params )>::TF>,TF>, "params: the points' float" );
    const SI nb = ( N + BLOCK - 1 ) / BLOCK, tiles = angle_tiles( nb * NB, A );
    const auto P = view( points );
    const auto R = view( radii );
    const auto Nr = view( normals );
    const auto H = view( hist );
    const auto PP = view( params );
    launch_kernel( queue, &deposit_kernel<SHAPE,K,L,TF,decltype( P ),decltype( R ),decltype( Nr ),decltype( H )>,
                   int( NB * nb * tiles ), BLOCK, 0, PP, M, A, N, P, R, Nr, H, nb, tiles );
}

template<int SHAPE,int K,int L,class Queue> requires ( ! std::is_same_v<Queue,CpuQueue> )
void gather( const Queue &queue, const auto &points, const auto &radii, const auto &normals, const auto &params,
             const auto &g, auto &&grad_points, auto &&grad_radii ) {
    using TF = std::remove_const_t<typename std::decay_t<decltype( points )>::TF>;
    const SI NB = points.shape( 0 ), N = points.shape( 1 ), A = normals.shape( 0 ), M = g.shape( 2 );
    const SI nb = ( N + BLOCK - 1 ) / BLOCK, tiles = angle_tiles( nb * NB, A );
    const auto P = view( points );
    const auto R = view( radii );
    const auto Nr = view( normals );
    const auto Gr = view( g );
    const auto GP = view( grad_points );
    const auto GR = view( grad_radii );
    const auto PP = view( params );
    launch_kernel( queue, &gather_kernel<SHAPE,K,L,TF,decltype( P ),decltype( R ),decltype( Nr ),decltype( Gr ),decltype( GP ),decltype( GR )>,
                   int( NB * nb * tiles ), BLOCK, 0, PP, M, A, N, P, R, Nr, Gr, GP, GR, nb, tiles );
}

constexpr int GRID_BLOCK = 256;

/// one block per ( problem, angle ) row: the cells split in GRID_BLOCK chunks, the cumulated masses and the suffix
/// sums of I0 by scans in shared memory, two sweeps of its chunk per thread ( the first for the sums )
template<class VM,class VN,class VC,class VL,class VD>
__global__ void __launch_bounds__( GRID_BLOCK ) grid_kernel( VM MU, VN NU, VC CN, VL LO, VD DG, SI A, SI M, double h ) {
    __shared__ double sh_a[ GRID_BLOCK ], sh_b[ GRID_BLOCK ], sh_g[ GRID_BLOCK ];
    __shared__ double sh_S, sh_loss;
    const SI r = SI( blockIdx.x ), b = r / A, a = r % A;
    const int t = threadIdx.x;
    const SI chunk = ( M + GRID_BLOCK - 1 ) / GRID_BLOCK, i0 = std::min( M, SI( t ) * chunk ), i1 = std::min( M, i0 + chunk );
    Row<VM> mu{ MU, b, a };
    Row<VN> nu{ NU, b, a };
    Row<VC> Cn{ CN, b, a };
    Row<VD> dg{ DG, b, a };

    double part = 0;
    for ( SI i = i0; i < i1; ++i )
        part += double( mu( i ) );
    sh_a[ t ] = part;
    __syncthreads();
    if ( t == 0 ) {
        double acc = 0;
        for ( int u = 0; u < GRID_BLOCK; ++u ) {
            const double v = sh_a[ u ];
            sh_a[ u ] = acc;
            acc += v;
        }
        sh_S = acc;
    }
    __syncthreads();
    const double inv_S = sh_S > 0 ? 1 / sh_S : 0;
    const double Cm0 = sh_a[ t ] * inv_S;

    // first sweep: this chunk's I0 and loss
    const SI k0 = start_cell( Cn, M, Cm0 );
    double I0, I1, loss = 0, tot = 0, Cm = Cm0;
    SI k = k0;
    for ( SI i = i0; i < i1; ++i ) {
        const double m = double( mu( i ) ) * inv_S, hi = i == M - 1 ? 1.0 : Cm + m;
        cell_terms( m, Cm, hi, double( i ) * h, k, nu, Cn, M, h, I0, I1, loss );
        tot += I0;
        Cm = hi;
    }
    __syncthreads();
    sh_a[ t ] = tot;
    sh_b[ t ] = loss;
    __syncthreads();
    if ( t == 0 ) {
        double acc = 0, ls = 0;
        for ( int u = GRID_BLOCK - 1; u >= 0; --u ) {
            const double v = sh_a[ u ];
            sh_a[ u ] = acc;                   // the I0 of the chunks AFTER u
            acc += v;
            ls += sh_b[ u ];
        }
        sh_loss = ls;
    }
    __syncthreads();
    const double after = sh_a[ t ];

    // second sweep: G_i = - ( after + the I0 of this chunk after i ) - I1_i, and its differences
    double prefix = 0, G_prev = 0, G_first = 0, dummy = 0;
    k = k0;
    Cm = Cm0;
    for ( SI i = i0; i < i1; ++i ) {
        const double m = double( mu( i ) ) * inv_S, hi = i == M - 1 ? 1.0 : Cm + m;
        cell_terms( m, Cm, hi, double( i ) * h, k, nu, Cn, M, h, I0, I1, dummy );
        prefix += I0;
        const double G = - ( after + tot - prefix ) - I1;
        if ( i == i0 )
            G_first = G;
        else
            dg( i ) = ( G - G_prev ) * inv_S;
        G_prev = G;
        Cm = hi;
    }
    sh_g[ t ] = G_prev;
    __syncthreads();
    if ( i0 < i1 ) {
        if ( i0 > 0 )
            dg( i0 ) = ( G_first - sh_g[ t - 1 ] ) * inv_S;
        else
            dg( 0 ) = 0;
    }
    if ( t == 0 ) {
        dg( M ) = 0;
        LO( b, a ) = sh_loss;
    }
}

template<class Queue> requires ( ! std::is_same_v<Queue,CpuQueue> )
void grid( const Queue &queue, const auto &mu, const auto &nu, const auto &Cn, const auto &params, auto &&loss, auto &&dg ) {
    const auto MU = view( mu );
    const auto NU = view( nu );
    const auto CN = view( Cn );
    const auto LO = view( loss );
    const auto DG = view( dg );
    const SI NB = mu.shape( 0 ), A = mu.shape( 1 ), M = mu.shape( 2 );
    launch_kernel( queue, &grid_kernel<decltype( MU ),decltype( NU ),decltype( CN ),decltype( LO ),decltype( DG )>,
                   int( NB * A ), GRID_BLOCK, 0, MU, NU, CN, LO, DG, A, M, 1.0 );   // h = 1: see `grid`
}

// ellipses ( see the CPU section ): the same tiles of angles, atomics wherever several tiles share an object

template<int K,int L,class TF,class VP,class VC,class VM,class VN,class VH>
__global__ void __launch_bounds__( BLOCK ) deposit_ellipses_kernel( View<TF,1> PP, SI M, SI A, SI N, VP P, VC C, VM Ms, VN Nr, VH H, SI nb_obj_blocks, SI tiles ) {
    const Params<TF> pr = read_params<TF>( PP, M, A, N, K );
    const BlockPos bp( nb_obj_blocks, tiles );
    const SI i = bp.ob * BLOCK + threadIdx.x;
    if ( i >= pr.N )
        return;
    const TF px = TF( P( bp.b, i, 0 ) ), py = TF( P( bp.b, i, 1 ) ), mass = TF( Ms( bp.b, i ) );
    const TF c00 = TF( C( bp.b, i, 0 ) ), c11 = TF( C( bp.b, i, 1 ) ), c01 = TF( C( bp.b, i, 2 ) );
    const auto Hb = H[ bp.b ];
    const SI a0 = pr.A * bp.tile / tiles, a1 = pr.A * ( bp.tile + 1 ) / tiles;
    for ( SI a = a0; a < a1; ++a ) {
        const TF nx = TF( Nr( a, 0 ) ), ny = TF( Nr( a, 1 ) );
        bool inside;
        const TF w = ellipse_width( c00, c11, c01, nx, ny, pr.inv_n, pr.max_w, inside );
        deposit_one<1,K,L>( pr, Hb, a, px * nx + py * ny, w, mass, true );
    }
}

template<int K,int L,class TF,class VP,class VC,class VM,class VN,class VG,class VGP,class VGC,class VGM>
__global__ void __launch_bounds__( BLOCK ) gather_ellipses_kernel( View<TF,1> PP, SI M, SI A, SI N, VP P, VC C, VM Ms, VN Nr, VG Gr, VGP GP, VGC GC, VGM GM, SI nb_obj_blocks, SI tiles ) {
    const Params<TF> pr = read_params<TF>( PP, M, A, N, K );
    const BlockPos bp( nb_obj_blocks, tiles );
    const SI i = bp.ob * BLOCK + threadIdx.x;
    if ( i >= pr.N )
        return;
    const SI b = bp.b;
    TF out[ 6 ];
    gather_ellipse<K,L>( pr, Nr, Gr[ b ], pr.A * bp.tile / tiles, pr.A * ( bp.tile + 1 ) / tiles, TF( P( b, i, 0 ) ), TF( P( b, i, 1 ) ),
                         TF( C( b, i, 0 ) ), TF( C( b, i, 1 ) ), TF( C( b, i, 2 ) ), TF( Ms( b, i ) ), out );
    if ( tiles == 1 ) {
        GP( b, i, 0 ) = out[ 0 ]; GP( b, i, 1 ) = out[ 1 ];
        GC( b, i, 0 ) = out[ 2 ]; GC( b, i, 1 ) = out[ 3 ]; GC( b, i, 2 ) = out[ 4 ];
        GM( b, i ) = out[ 5 ];
    } else {
        atomicAdd( &GP( b, i, 0 ), out[ 0 ] ); atomicAdd( &GP( b, i, 1 ), out[ 1 ] );
        atomicAdd( &GC( b, i, 0 ), out[ 2 ] ); atomicAdd( &GC( b, i, 1 ), out[ 3 ] ); atomicAdd( &GC( b, i, 2 ), out[ 4 ] );
        atomicAdd( &GM( b, i ), out[ 5 ] );
    }
}

template<int K,int L,class Queue> requires ( ! std::is_same_v<Queue,CpuQueue> )
void deposit_ellipses( const Queue &queue, const auto &points, const auto &covs, const auto &masses, const auto &normals, const auto &params, auto &&hist ) {
    using TF = std::remove_const_t<typename std::decay_t<decltype( points )>::TF>;
    const SI NB = points.shape( 0 ), N = points.shape( 1 ), A = normals.shape( 0 ), M = hist.shape( 2 );
    const SI nb = ( N + BLOCK - 1 ) / BLOCK, tiles = angle_tiles( nb * NB, A );
    const auto P = view( points );
    const auto C = view( covs );
    const auto Ms = view( masses );
    const auto Nr = view( normals );
    const auto H = view( hist );
    const auto PP = view( params );
    launch_kernel( queue, &deposit_ellipses_kernel<K,L,TF,decltype( P ),decltype( C ),decltype( Ms ),decltype( Nr ),decltype( H )>,
                   int( NB * nb * tiles ), BLOCK, 0, PP, M, A, N, P, C, Ms, Nr, H, nb, tiles );
}

template<int K,int L,class Queue> requires ( ! std::is_same_v<Queue,CpuQueue> )
void gather_ellipses( const Queue &queue, const auto &points, const auto &covs, const auto &masses, const auto &normals, const auto &params,
                      const auto &g, auto &&grad_points, auto &&grad_covs, auto &&grad_masses ) {
    using TF = std::remove_const_t<typename std::decay_t<decltype( points )>::TF>;
    const SI NB = points.shape( 0 ), N = points.shape( 1 ), A = normals.shape( 0 ), M = g.shape( 2 );
    const SI nb = ( N + BLOCK - 1 ) / BLOCK, tiles = angle_tiles( nb * NB, A );
    const auto P = view( points );
    const auto C = view( covs );
    const auto Ms = view( masses );
    const auto Nr = view( normals );
    const auto Gr = view( g );
    const auto GP = view( grad_points );
    const auto GC = view( grad_covs );
    const auto GM = view( grad_masses );
    const auto PP = view( params );
    launch_kernel( queue, &gather_ellipses_kernel<K,L,TF,decltype( P ),decltype( C ),decltype( Ms ),decltype( Nr ),decltype( Gr ),decltype( GP ),decltype( GC ),decltype( GM )>,
                   int( NB * nb * tiles ), BLOCK, 0, PP, M, A, N, P, C, Ms, Nr, Gr, GP, GC, GM, nb, tiles );
}

// polygons ( see the CPU section ), as the ellipses

template<int V,int L,class TF,class VP,class VW,class VN,class VH>
__global__ void __launch_bounds__( BLOCK ) deposit_polygons_kernel( View<TF,1> PP, SI M, SI A, SI N, VP P, VW W, VN Nr, VH H, SI nb_obj_blocks, SI tiles ) {
    const Params<TF> pr = read_params<TF>( PP, M, A, N, 0 );
    const BlockPos bp( nb_obj_blocks, tiles );
    const SI i = bp.ob * BLOCK + threadIdx.x;
    if ( i >= pr.N )
        return;
    TF xy[ 2 * V ];
    for ( int c = 0; c < 2 * V; ++c )
        xy[ c ] = TF( P( bp.b, i, c ) );
    const TF weight = TF( W( bp.b, i ) );
    const auto Hb = H[ bp.b ];
    const SI a0 = pr.A * bp.tile / tiles, a1 = pr.A * ( bp.tile + 1 ) / tiles;
    for ( SI a = a0; a < a1; ++a )
        deposit_polygon_one<V,L>( pr, Hb, a, xy, TF( Nr( a, 0 ) ), TF( Nr( a, 1 ) ), weight, true );
}

template<int V,int L,class TF,class VP,class VW,class VN,class VG,class VGP,class VGW>
__global__ void __launch_bounds__( BLOCK ) gather_polygons_kernel( View<TF,1> PP, SI M, SI A, SI N, VP P, VW W, VN Nr, VG Gr, VGP GP, VGW GW, SI nb_obj_blocks, SI tiles ) {
    const Params<TF> pr = read_params<TF>( PP, M, A, N, 0 );
    const BlockPos bp( nb_obj_blocks, tiles );
    const SI i = bp.ob * BLOCK + threadIdx.x;
    if ( i >= pr.N )
        return;
    const SI b = bp.b;
    TF xy[ 2 * V ], out[ 2 * V + 1 ];
    for ( int c = 0; c < 2 * V; ++c )
        xy[ c ] = TF( P( b, i, c ) );
    gather_polygon<V,L>( pr, Nr, Gr[ b ], pr.A * bp.tile / tiles, pr.A * ( bp.tile + 1 ) / tiles, xy, TF( W( b, i ) ), out );
    for ( int c = 0; c < 2 * V; ++c ) {
        if ( tiles == 1 ) GP( b, i, c ) = out[ c ];
        else atomicAdd( &GP( b, i, c ), out[ c ] );
    }
    if ( tiles == 1 ) GW( b, i ) = out[ 2 * V ];
    else atomicAdd( &GW( b, i ), out[ 2 * V ] );
}

template<int V,int L,class Queue> requires ( ! std::is_same_v<Queue,CpuQueue> )
void deposit_polygons( const Queue &queue, const auto &verts, const auto &weights, const auto &normals, const auto &params, auto &&hist ) {
    using TF = std::remove_const_t<typename std::decay_t<decltype( verts )>::TF>;
    const SI NB = verts.shape( 0 ), N = verts.shape( 1 ), A = normals.shape( 0 ), M = hist.shape( 2 );
    const SI nb = ( N + BLOCK - 1 ) / BLOCK, tiles = angle_tiles( nb * NB, A );
    const auto P = view( verts );
    const auto W = view( weights );
    const auto Nr = view( normals );
    const auto H = view( hist );
    const auto PP = view( params );
    launch_kernel( queue, &deposit_polygons_kernel<V,L,TF,decltype( P ),decltype( W ),decltype( Nr ),decltype( H )>,
                   int( NB * nb * tiles ), BLOCK, 0, PP, M, A, N, P, W, Nr, H, nb, tiles );
}

template<int V,int L,class Queue> requires ( ! std::is_same_v<Queue,CpuQueue> )
void gather_polygons( const Queue &queue, const auto &verts, const auto &weights, const auto &normals, const auto &params,
                      const auto &g, auto &&grad_verts, auto &&grad_weights ) {
    using TF = std::remove_const_t<typename std::decay_t<decltype( verts )>::TF>;
    const SI NB = verts.shape( 0 ), N = verts.shape( 1 ), A = normals.shape( 0 ), M = g.shape( 2 );
    const SI nb = ( N + BLOCK - 1 ) / BLOCK, tiles = angle_tiles( nb * NB, A );
    const auto P = view( verts );
    const auto W = view( weights );
    const auto Nr = view( normals );
    const auto Gr = view( g );
    const auto GP = view( grad_verts );
    const auto GW = view( grad_weights );
    const auto PP = view( params );
    launch_kernel( queue, &gather_polygons_kernel<V,L,TF,decltype( P ),decltype( W ),decltype( Nr ),decltype( Gr ),decltype( GP ),decltype( GW )>,
                   int( NB * nb * tiles ), BLOCK, 0, PP, M, A, N, P, W, Nr, Gr, GP, GW, nb, tiles );
}

#endif

} // namespace projected1d
} // namespace sdot
