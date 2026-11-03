// Branch-free power cells for seeds on a regular 2D grid.
//
// Seed (a, b) sits at (a, b), domain = [-1/2, N-1/2]^2, flat index a * N + b.
// psi is read from a padded array ( N + 2 R ) x ( N + 2 R ) whose padding holds
// -inf, so that missing neighbors give h = +inf without any test.
//
// Per cell: offsets h_k of the ring-R stencil, merged per direction ( argmin
// kept as owner ), tightened to the support function by a fixed list of pairs,
// then one vertex per pair of consecutive directions. Outputs the area and,
// per direction, the off-diagonal hessian weight L / ( 2 |k| ) with its owner.
#pragma once
#include "StencilTables.h"
#include <algorithm>
#include <limits>
#include <memory>
#include <cmath>
#include <utility>
#include <vector>

namespace sdot {
namespace grid {

/// an allocator that leaves its values uninitialized: a buffer of a few hundred megabytes is then neither zeroed by one thread
/// nor committed where it is never written
template<class V>
struct NoInitAlloc : std::allocator<V> {
    template<class U> struct rebind { using other = NoInitAlloc<U>; };
    NoInitAlloc() = default;
    template<class U> NoInitAlloc( const NoInitAlloc<U> & ) {}
    template<class U> void construct( U *p ) noexcept { ::new( static_cast<void *>( p ) ) U; }
    template<class U, class... A> void construct( U *p, A &&...a ) { ::new( static_cast<void *>( p ) ) U( std::forward<A>( a )... ); }
};
template<class V> using NoInitVector = std::vector<V, NoInitAlloc<V>>;

template<int R, int B = 32, class T = double>
struct StencilKernel {
    using Tb = StencilTables<R>;
    static constexpr int D = Tb::D;

    // vertices and owners in SoA planes, one row of P = N + 2 R entries per grid row ( columns padded by R ).
    // Full: every row, plus R padding rows on each side. Rolling: only 2 R + 1 rows in a ring ( slot 0 is
    // the padding row ), enough to certify row a - R right after computing row a ( fused pass ). `init = false`: the values
    // are left uninitialized ( `fill` sets them, or nothing reads what was not computed ).
    struct Planes {
        int N, P, slots;
        bool rolling;
        NoInitVector<T> vx, vy;
        NoInitVector<signed char> ow;
        Planes( int N, bool rolling = false, bool init = true ) : N( N ), P( N + 2 * R ), slots( rolling ? 2 * R + 2 : N + 2 * R ), rolling( rolling ),
            vx( D * slots * P ), vy( D * slots * P ), ow( D * slots * P ) { if( init ) fill( 0, slots ); }
        // the initial values of the slots [ s0, s1 ) of each plane
        void fill( int s0, int s1 ) {
            for( int k = 0; k < D; ++k ) {
                const size_t b = size_t( k ) * slots * P + size_t( s0 ) * P, e = size_t( k ) * slots * P + size_t( s1 ) * P;
                std::fill( vx.begin() + b, vx.begin() + e, T( 0 ) );
                std::fill( vy.begin() + b, vy.begin() + e, T( 0 ) );
                std::fill( ow.begin() + b, ow.begin() + e, -2 );
            }
        }
        int plane() const { return slots * P; }
        int row( int a ) const {
            if( ! rolling ) return ( a + R ) * P;
            return a < 0 || a >= N ? 0 : ( 1 + a % ( 2 * R + 1 ) ) * P;
        }
    };

    // cells ( a, b0 .. b0 + n - 1 ), n <= B
    static void row_block( int N, const T *psi_pad, int a, int b0, int n, T *area, T *hess, signed char *owner, T *verts, Planes *pl = nullptr ) {
        const int P = N + 2 * R;
        const T inf = std::numeric_limits<T>::infinity();
        alignas( 64 ) T h[ D ][ B ], ht[ D ][ B ];
        alignas( 64 ) signed char own[ D ][ B ];
        const T *pi = psi_pad + ( a + R ) * P + ( b0 + R );

        for( int d = 0; d < D; ++d )
            for( int x = 0; x < B; ++x ) { h[ d ][ x ] = inf; own[ d ][ x ] = -1; }

        for( int o = 0; o < Tb::nb_offsets; ++o ) {
            const int d = Tb::off_dir[ o ];
            const T c = T( Tb::off_sqn[ o ] ), s = T( 0.5 ) / T( Tb::off_m[ o ] );
            const T *pj = pi + Tb::off_x[ o ] * P + Tb::off_y[ o ];
            #pragma omp simd
            for( int x = 0; x < B; ++x ) {
                T hk = ( c + pi[ x ] - pj[ x ] ) * s;
                bool lt = hk < h[ d ][ x ];
                h[ d ][ x ] = lt ? hk : h[ d ][ x ];
                own[ d ][ x ] = lt ? (signed char)o : own[ d ][ x ];
            }
        }

        // box
        #pragma omp simd
        for( int x = 0; x < B; ++x ) {
            T bb = T( b0 + x );
            T hpx = T( N ) - T( 0.5 ) - T( a ), hmx = T( a ) + T( 0.5 ), hpy = T( N ) - T( 0.5 ) - bb, hmy = bb + T( 0.5 );
            bool l0 = hpx <= h[ Tb::dir_px ][ x ], l1 = hmx <= h[ Tb::dir_mx ][ x ], l2 = hpy <= h[ Tb::dir_py ][ x ], l3 = hmy <= h[ Tb::dir_my ][ x ];
            h[ Tb::dir_px ][ x ] = l0 ? hpx : h[ Tb::dir_px ][ x ]; own[ Tb::dir_px ][ x ] = l0 ? -1 : own[ Tb::dir_px ][ x ];
            h[ Tb::dir_mx ][ x ] = l1 ? hmx : h[ Tb::dir_mx ][ x ]; own[ Tb::dir_mx ][ x ] = l1 ? -1 : own[ Tb::dir_mx ][ x ];
            h[ Tb::dir_py ][ x ] = l2 ? hpy : h[ Tb::dir_py ][ x ]; own[ Tb::dir_py ][ x ] = l2 ? -1 : own[ Tb::dir_py ][ x ];
            h[ Tb::dir_my ][ x ] = l3 ? hmy : h[ Tb::dir_my ][ x ]; own[ Tb::dir_my ][ x ] = l3 ? -1 : own[ Tb::dir_my ][ x ];
        }

        // tightening ( one pass on the raw offsets is exact: optimal LP bases have 2 elements )
        for( int k = 0; k < D; ++k ) {
            #pragma omp simd
            for( int x = 0; x < B; ++x ) ht[ k ][ x ] = h[ k ][ x ];
            for( int p = Tb::pair_beg[ k ]; p < Tb::pair_beg[ k + 1 ]; ++p ) {
                const int j = Tb::pair_j[ p ], l = Tb::pair_l[ p ];
                const T lam = T( Tb::pair_lam[ p ] ), mu = T( Tb::pair_mu[ p ] );
                #pragma omp simd
                for( int x = 0; x < B; ++x )
                    ht[ k ][ x ] = std::fmin( ht[ k ][ x ], lam * h[ j ][ x ] + mu * h[ l ][ x ] );
            }
        }

        // vertices, area, edge lengths
        alignas( 64 ) T vx[ D ][ B ], vy[ D ][ B ];
        for( int k = 0; k < D; ++k ) {
            const int k1 = ( k + 1 ) % D;
            const double *M = Tb::vinv + 4 * k;
            #pragma omp simd
            for( int x = 0; x < B; ++x ) {
                vx[ k ][ x ] = T( M[ 0 ] ) * ht[ k ][ x ] + T( M[ 1 ] ) * ht[ k1 ][ x ];
                vy[ k ][ x ] = T( M[ 2 ] ) * ht[ k ][ x ] + T( M[ 3 ] ) * ht[ k1 ][ x ];
            }
        }
        alignas( 64 ) T ar[ B ];
        #pragma omp simd
        for( int x = 0; x < B; ++x ) ar[ x ] = 0;
        for( int k = 0; k < D; ++k ) {
            const int km = ( k + D - 1 ) % D;
            // edge k from vertex k - 1 to vertex k, tangent = ( -n_y, n_x ) / |n|
            const T tx = -T( Tb::dir_y[ k ] ), ty = T( Tb::dir_x[ k ] ), inv = T( Tb::dir_inv_norm[ k ] );
            #pragma omp simd
            for( int x = 0; x < B; ++x ) {
                ar[ x ] += vx[ km ][ x ] * vy[ k ][ x ] - vy[ km ][ x ] * vx[ k ][ x ];
                T L = ( tx * ( vx[ k ][ x ] - vx[ km ][ x ] ) + ty * ( vy[ k ][ x ] - vy[ km ][ x ] ) ) * inv;
                // |k_owner| = m |n|, weight = L / ( 2 |k_owner| )
                int o = own[ k ][ x ];
                T m = o >= 0 ? T( Tb::off_m[ o < 0 ? 0 : o ] ) : T( 1 );
                T w = std::fmax( L, T( 0 ) ) * inv * T( 0.5 ) / m;
                hess[ ( b0 + x ) * D + k ] = o >= 0 ? w : T( 0 );
            }
        }
        for( int x = 0; x < n; ++x ) {
            area[ b0 + x ] = T( 0.5 ) * ar[ x ];
            for( int k = 0; k < D; ++k ) owner[ ( b0 + x ) * D + k ] = own[ k ][ x ];
        }
        if( pl ) {
            const int PP = pl->plane(), i0 = pl->row( a ) + b0 + R;
            for( int k = 0; k < D; ++k ) {
                T *px = pl->vx.data() + k * PP + i0, *py = pl->vy.data() + k * PP + i0;
                signed char *po = pl->ow.data() + k * PP + i0;
                #pragma omp simd
                for( int x = 0; x < B; ++x ) { px[ x ] = vx[ k ][ x ]; py[ x ] = vy[ k ][ x ]; po[ x ] = own[ k ][ x ]; }
            }
        }
        if( verts )
            for( int x = 0; x < n; ++x )
                for( int k = 0; k < D; ++k ) {
                    verts[ ( ( b0 + x ) * D + k ) * 2 + 0 ] = vx[ k ][ x ];
                    verts[ ( ( b0 + x ) * D + k ) * 2 + 1 ] = vy[ k ][ x ];
                }
    }

    // same as row_block for an arbitrary list of cells ( flat indices a * N + b, 1 <= n <= B ): psi is
    // gathered, outputs are scattered ( area[ i ], hess[ i * D + k ], owner[ i * D + k ] -- each may be null --, planes )
    static void cells_block( int N, const T *psi_pad, const int *cells, int n, T *area, T *hess, signed char *owner, Planes *pl ) {
        const int P = N + 2 * R;
        const T inf = std::numeric_limits<T>::infinity();
        alignas( 64 ) T h[ D ][ B ], ht[ D ][ B ];
        alignas( 64 ) signed char own[ D ][ B ];
        alignas( 64 ) int base[ B ];
        alignas( 64 ) T ca[ B ], cb[ B ];
        for( int x = 0; x < B; ++x ) {
            const int i = cells[ x < n ? x : n - 1 ];
            ca[ x ] = T( i / N ); cb[ x ] = T( i % N );
            base[ x ] = ( i / N + R ) * P + ( i % N + R );
        }

        for( int d = 0; d < D; ++d )
            for( int x = 0; x < B; ++x ) { h[ d ][ x ] = inf; own[ d ][ x ] = -1; }

        for( int o = 0; o < Tb::nb_offsets; ++o ) {
            const int d = Tb::off_dir[ o ];
            const T c = T( Tb::off_sqn[ o ] ), s = T( 0.5 ) / T( Tb::off_m[ o ] );
            const int sh = Tb::off_x[ o ] * P + Tb::off_y[ o ];
            #pragma omp simd
            for( int x = 0; x < B; ++x ) {
                T hk = ( c + psi_pad[ base[ x ] ] - psi_pad[ base[ x ] + sh ] ) * s;
                bool lt = hk < h[ d ][ x ];
                h[ d ][ x ] = lt ? hk : h[ d ][ x ];
                own[ d ][ x ] = lt ? (signed char)o : own[ d ][ x ];
            }
        }

        // box
        #pragma omp simd
        for( int x = 0; x < B; ++x ) {
            T bb = cb[ x ];
            T hpx = T( N ) - T( 0.5 ) - ca[ x ], hmx = ca[ x ] + T( 0.5 ), hpy = T( N ) - T( 0.5 ) - bb, hmy = bb + T( 0.5 );
            bool l0 = hpx <= h[ Tb::dir_px ][ x ], l1 = hmx <= h[ Tb::dir_mx ][ x ], l2 = hpy <= h[ Tb::dir_py ][ x ], l3 = hmy <= h[ Tb::dir_my ][ x ];
            h[ Tb::dir_px ][ x ] = l0 ? hpx : h[ Tb::dir_px ][ x ]; own[ Tb::dir_px ][ x ] = l0 ? -1 : own[ Tb::dir_px ][ x ];
            h[ Tb::dir_mx ][ x ] = l1 ? hmx : h[ Tb::dir_mx ][ x ]; own[ Tb::dir_mx ][ x ] = l1 ? -1 : own[ Tb::dir_mx ][ x ];
            h[ Tb::dir_py ][ x ] = l2 ? hpy : h[ Tb::dir_py ][ x ]; own[ Tb::dir_py ][ x ] = l2 ? -1 : own[ Tb::dir_py ][ x ];
            h[ Tb::dir_my ][ x ] = l3 ? hmy : h[ Tb::dir_my ][ x ]; own[ Tb::dir_my ][ x ] = l3 ? -1 : own[ Tb::dir_my ][ x ];
        }

        // tightening ( one pass on the raw offsets is exact: optimal LP bases have 2 elements )
        for( int k = 0; k < D; ++k ) {
            #pragma omp simd
            for( int x = 0; x < B; ++x ) ht[ k ][ x ] = h[ k ][ x ];
            for( int p = Tb::pair_beg[ k ]; p < Tb::pair_beg[ k + 1 ]; ++p ) {
                const int j = Tb::pair_j[ p ], l = Tb::pair_l[ p ];
                const T lam = T( Tb::pair_lam[ p ] ), mu = T( Tb::pair_mu[ p ] );
                #pragma omp simd
                for( int x = 0; x < B; ++x )
                    ht[ k ][ x ] = std::fmin( ht[ k ][ x ], lam * h[ j ][ x ] + mu * h[ l ][ x ] );
            }
        }

        // vertices, area, edge lengths
        alignas( 64 ) T vx[ D ][ B ], vy[ D ][ B ];
        for( int k = 0; k < D; ++k ) {
            const int k1 = ( k + 1 ) % D;
            const double *M = Tb::vinv + 4 * k;
            #pragma omp simd
            for( int x = 0; x < B; ++x ) {
                vx[ k ][ x ] = T( M[ 0 ] ) * ht[ k ][ x ] + T( M[ 1 ] ) * ht[ k1 ][ x ];
                vy[ k ][ x ] = T( M[ 2 ] ) * ht[ k ][ x ] + T( M[ 3 ] ) * ht[ k1 ][ x ];
            }
        }
        alignas( 64 ) T ar[ B ], hw[ D ][ B ];
        #pragma omp simd
        for( int x = 0; x < B; ++x ) ar[ x ] = 0;
        for( int k = 0; k < D; ++k ) {
            const int km = ( k + D - 1 ) % D;
            // edge k from vertex k - 1 to vertex k, tangent = ( -n_y, n_x ) / |n|
            const T tx = -T( Tb::dir_y[ k ] ), ty = T( Tb::dir_x[ k ] ), inv = T( Tb::dir_inv_norm[ k ] );
            #pragma omp simd
            for( int x = 0; x < B; ++x ) {
                ar[ x ] += vx[ km ][ x ] * vy[ k ][ x ] - vy[ km ][ x ] * vx[ k ][ x ];
                T L = ( tx * ( vx[ k ][ x ] - vx[ km ][ x ] ) + ty * ( vy[ k ][ x ] - vy[ km ][ x ] ) ) * inv;
                // |k_owner| = m |n|, weight = L / ( 2 |k_owner| )
                int o = own[ k ][ x ];
                T m = o >= 0 ? T( Tb::off_m[ o < 0 ? 0 : o ] ) : T( 1 );
                T w = std::fmax( L, T( 0 ) ) * inv * T( 0.5 ) / m;
                hw[ k ][ x ] = o >= 0 ? w : T( 0 );
            }
        }
        for( int x = 0; x < n; ++x ) {
            const int i = cells[ x ], j = pl->row( i / N ) + i % N + R, PP = pl->plane();
            if( area ) area[ i ] = T( 0.5 ) * ar[ x ];
            for( int k = 0; k < D; ++k ) {
                if( hess ) hess[ i * D + k ] = hw[ k ][ x ];
                if( owner ) owner[ i * D + k ] = own[ k ][ x ];
                pl->vx[ k * PP + j ] = vx[ k ][ x ];
                pl->vy[ k * PP + j ] = vy[ k ][ x ];
                pl->ow[ k * PP + j ] = own[ k ][ x ];
            }
        }
    }

    // all cells ( N % B == 0 ); area[ N * N ], hess[ N * N * D ], owner[ N * N * D ], verts[ N * N * D * 2 ] or null
    static void run( int N, const T *psi_pad, T *area, T *hess, signed char *owner, T *verts = nullptr, Planes *pl = nullptr ) {
        if ( N % B ) throw "N must be a multiple of B";
        for( int a = 0; a < N; ++a )
            for( int b0 = 0; b0 < N; b0 += B )
                row_block( N, psi_pad, a, b0, N - b0 < B ? N - b0 : B, area + a * N, hess + a * N * D, owner + a * N * D, verts ? verts + a * N * D * 2 : nullptr, pl );
    }

    // same certificate as `certify`, vectorized across the cells of a row: for direction k, the owner is
    // one of the few multiples m d_k, so the neighbor's planes are read at a FIXED shift per candidate
    // ( contiguous loads ) and the right candidate is selected by the owner
    struct CertTables {
        int opp[ D ], back[ Tb::nb_offsets ];
        CertTables() {
            for( int k = 0; k < D; ++k )
                for( int l = 0; l < D; ++l )
                    if( Tb::dir_x[ l ] == -Tb::dir_x[ k ] && Tb::dir_y[ l ] == -Tb::dir_y[ k ] ) opp[ k ] = l;
            for( int o = 0; o < Tb::nb_offsets; ++o )
                for( int p = 0; p < Tb::nb_offsets; ++p )
                    if( Tb::off_x[ p ] == -Tb::off_x[ o ] && Tb::off_y[ p ] == -Tb::off_y[ o ] ) back[ o ] = p;
        }
    };

    // blocks [ b_beg, b_end ) of B cells ( default: the whole row )
    static int certify_row( const Planes &pl, const CertTables &ct, int a, unsigned char *flag, T tol, int b_beg = 0, int b_end = -1 ) {
        const int N = pl.N, PP = pl.plane();
        if( b_end < 0 ) b_end = N;
        int count = 0;
        for( int b0 = b_beg; b0 < b_end; b0 += B ) {
            const int i0 = pl.row( a ) + b0 + R;
            alignas( 64 ) unsigned char bad[ B ] = {};
            for( int k = 0; k < D; ++k ) {
                const int km = ( k + D - 1 ) % D, q = ct.opp[ k ], qm = ( q + D - 1 ) % D;
                const T *ivx = pl.vx.data() + k * PP + i0, *ivy = pl.vy.data() + k * PP + i0;
                const T *imx = pl.vx.data() + km * PP + i0, *imy = pl.vy.data() + km * PP + i0;
                const signed char *iow = pl.ow.data() + k * PP + i0;
                alignas( 64 ) unsigned char same[ B ] = {};
                for( int o = 0; o < Tb::nb_offsets; ++o ) {
                    if( Tb::off_dir[ o ] != k ) continue;
                    const int dx = Tb::off_x[ o ], dy = Tb::off_y[ o ], j0 = pl.row( a + dx ) + b0 + R + dy;
                    const T *wqx = pl.vx.data() + q * PP + j0, *wqy = pl.vy.data() + q * PP + j0;
                    const T *wmx = pl.vx.data() + qm * PP + j0, *wmy = pl.vy.data() + qm * PP + j0;
                    const signed char *wow = pl.ow.data() + q * PP + j0;
                    const signed char bo = ct.back[ o ];
                    const T fx = dx, fy = dy;
                    #pragma omp simd
                    for( int x = 0; x < B; ++x ) {
                        bool ok = ( iow[ x ] == o ) & ( wow[ x ] == bo )
                            & ( std::abs( imx[ x ] - wqx[ x ] - fx ) <= tol ) & ( std::abs( imy[ x ] - wqy[ x ] - fy ) <= tol )
                            & ( std::abs( ivx[ x ] - wmx[ x ] - fx ) <= tol ) & ( std::abs( ivy[ x ] - wmy[ x ] - fy ) <= tol );
                        same[ x ] |= ok;
                    }
                }
                #pragma omp simd
                for( int x = 0; x < B; ++x ) {
                    const T ex = ivx[ x ] - imx[ x ], ey = ivy[ x ] - imy[ x ];
                    const bool pos = std::abs( ex ) + std::abs( ey ) > tol;
                    bad[ x ] |= pos & ( iow[ x ] >= 0 ) & ! same[ x ];
                }
            }
            for( int x = 0; x < B; ++x ) { flag[ a * N + b0 + x ] = bad[ x ]; count += bad[ x ]; }
        }
        return count;
    }

    // two passes ( full planes )
    static int certify_simd( const Planes &pl, unsigned char *flag, T tol ) {
        CertTables ct;
        int count = 0;
        for( int a = 0; a < pl.N; ++a ) count += certify_row( pl, ct, a, flag, tol );
        return count;
    }

    // one fused pass on rolling planes: compute row a, certify row a - R while its neighbors are in cache
    static int run_certified( int N, const T *psi_pad, T *area, T *hess, signed char *owner, Planes &pl, unsigned char *flag, T tol ) {
        if ( N % B ) throw "N must be a multiple of B";
        CertTables ct;
        int count = 0;
        for( int a = 0; a < N + R; ++a ) {
            if( a < N )
                for( int b0 = 0; b0 < N; b0 += B )
                    row_block( N, psi_pad, a, b0, B, area + a * N, hess + a * N * D, owner + a * N * D, nullptr, &pl );
            if( a >= R )
                count += certify_row( pl, ct, a - R, flag, tol );
        }
        return count;
    }

    // local certificate: every positive edge k of cell i ( owner j ) must be edge opp( k ) of cell j,
    // owned by i, with the same endpoints. No flag anywhere => exact diagram. Returns the number of flagged cells.
    static int certify( int N, const signed char *owner, const T *verts, unsigned char *flag, T tol ) {
        int opp[ D ], back[ Tb::nb_offsets ];
        for( int k = 0; k < D; ++k )
            for( int l = 0; l < D; ++l )
                if( Tb::dir_x[ l ] == -Tb::dir_x[ k ] && Tb::dir_y[ l ] == -Tb::dir_y[ k ] ) opp[ k ] = l;
        for( int o = 0; o < Tb::nb_offsets; ++o )
            for( int p = 0; p < Tb::nb_offsets; ++p )
                if( Tb::off_x[ p ] == -Tb::off_x[ o ] && Tb::off_y[ p ] == -Tb::off_y[ o ] ) back[ o ] = p;
        for( int i = 0; i < N * N; ++i ) flag[ i ] = 0;
        for( int i = 0; i < N * N; ++i ) {
            const T *v = verts + i * D * 2;
            bool bad = false;
            for( int k = 0; k < D; ++k ) {
                const int km = ( k + D - 1 ) % D, o = owner[ i * D + k ];
                const T ex = v[ 2 * k ] - v[ 2 * km ], ey = v[ 2 * k + 1 ] - v[ 2 * km + 1 ];
                const bool pos = std::abs( ex ) + std::abs( ey ) > tol;
                const int oo = o < 0 ? 0 : o;
                const int dx = Tb::off_x[ oo ], dy = Tb::off_y[ oo ];
                const int j = i + dx * N + dy; // valid when o >= 0 ( padding gives inf, never an owner )
                const int jj = o < 0 ? i : j, q = opp[ k ], qm = ( q + D - 1 ) % D;
                const T *w = verts + jj * D * 2;
                // endpoints in i's frame: w + ( x_j - x_i )
                const bool same = owner[ jj * D + q ] == back[ oo ]
                    && std::abs( v[ 2 * km ] - w[ 2 * q ] - dx ) <= tol && std::abs( v[ 2 * km + 1 ] - w[ 2 * q + 1 ] - dy ) <= tol
                    && std::abs( v[ 2 * k ] - w[ 2 * qm ] - dx ) <= tol && std::abs( v[ 2 * k + 1 ] - w[ 2 * qm + 1 ] - dy ) <= tol;
                bad |= pos && o >= 0 && ! same;
            }
            flag[ i ] = bad;
        }
        int n = 0;
        for( int i = 0; i < N * N; ++i ) n += flag[ i ];
        return n;
    }
};

} // namespace grid
} // namespace sdot
