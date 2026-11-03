// Damped Newton ( KMT ) for semi-discrete OT with the seeds on an N x N grid, single thread.
//
//   geometry   ring-1 stencil + certificate ( fused ), ring 2 on the flagged cells, cross-level
//              certificate, exact clipping on what is left ( cascade.cpp )
//   integrals  Lebesgue ( exact ) or Gauss ( density.h ): mass of each cell and density integral on
//              each face; the hessian weight of face ij is w_ij = int_face rho / ( 2 |x_i - x_j| )
//   linear     H = dm/dpsi ( a weighted graph laplacian ), matrix-free: per cell, 8 weights toward
//              the ring-1 neighbors in SoA planes ( contiguous stencil loads ), plus a short list
//              of extra ( i, j, w ) for the cells that came from ring 2 or from the exact path.
//              PCG with Jacobi, or with the multilevel blur: z = sum_l P_l D_l^-1 P_l^T r, where
//              P_l^T sums 2^l x 2^l blocks ( box blur + decimation ), P_l copies back, and D_l is
//              the Galerkin diagonal ( weight crossing the block boundary ) -- additive / BPX.
//
// build ( from this folder ):
//   c++ -std=c++20 -O3 -march=native -fno-math-errno -fopenmp-simd -fveclib=Darwin_libsystem_m newton.cpp -o newton
// run:  ./newton lebesgue|gauss|mix N [ sigma_rel ] [ precond: tensor ( default )|tensor-unfused|tent|tent-unfused|jacobi|box|agg|agg-v|frames|tensor-diag ] [ cg_rtol ]
//       tensor: weighted semi-coarsened tent blurs ( TensorPrecond ), env TBAND ( |la - lb| kept ), TP ( balance power, 0.5 ),
//       TPLANES=1 ( scaling from exact crossing weights instead of conductance fields )
//       ./newton gauss N sigma_rel precond cg_rtol [ tau ] [ quadrature tol ]
//       ./newton check-gauss N sigma_rel   ( quadrature and hessian checks )
//       env TOL=<newton tolerance> ( default 1e-9; N = 2048 Gauss floors at ~1.5e-9, psi ~ N^2 )
//       options anywhere: n0=<coarsest size> ( coarse to fine: solve on n0, 2 n0, ..., N ), ctol=<tol of the coarse levels>
#include "stencil_kernel.h"
#include "exact_kernel.h"
#include "density.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <tuple>
#include <vector>

using T = double;
using Clock = std::chrono::steady_clock;
constexpr int B = 32;
using K1 = StencilKernel<1, B, T>;
using K2 = StencilKernel<2, B, T>;
using T1 = StencilTables<1>;
using T2 = StencilTables<2>;

static double now_ms() { return std::chrono::duration<double, std::milli>( Clock::now().time_since_epoch() ).count(); }

// ================================================================================================
// geometry + integrals
// ================================================================================================
template<class Density>
struct Evaluator {
    int N, P1; // P1 = N + 2: padded planes of the hessian weights
    const Density &rho;
    ExactKernel<T> ek{ 24 };
    // buffers
    std::vector<T> p1, p2, pe;
    std::vector<T> area1, hess1, area2, hess2;
    std::vector<signed char> own1, own2;
    std::vector<unsigned char> flag1, flag2, mark, dbad;
    K1::Planes pl1;
    K2::Planes pl2;
    std::vector<int> list1, check;
    int dir1[ 25 ], dir2[ 25 ];
    // outputs
    std::vector<T> mass;                 // N * N
    std::vector<T> W;                    // 8 planes of P1 * P1: weight toward the ring-1 neighbor in direction k
    struct Extra { int i, j; T w; };
    std::vector<Extra> extras;           // faces of the cells that did not come from ring 1
    // stats of the last call
    int n_flag1 = 0, n_exact = 0, n_dens = 0, n_fine = 0;
    double t_geom = 0, t_int = 0, t_tier1 = 0, t_tier2 = 0, t_scalar = 0;

    Evaluator( int N, const Density &rho ) : N( N ), P1( N + 2 ), rho( rho ),
        area1( N * N ), hess1( N * N * K1::D ), area2( N * N ), hess2( N * N * K2::D ),
        own1( N * N * K1::D ), own2( N * N * K2::D ), flag1( N * N ), flag2( N * N ), mark( N * N ), dbad( N * N ),
        pl1( N ), pl2( N ), mass( N * N ), W( 8 * ( N + 2 ) * ( N + 2 ) ) {
        for( int i = 0; i < 25; ++i ) dir1[ i ] = dir2[ i ] = -1;
        for( int k = 0; k < K1::D; ++k ) dir1[ int( T1::dir_x[ k ] + 2 ) * 5 + int( T1::dir_y[ k ] + 2 ) ] = k;
        for( int k = 0; k < K2::D; ++k ) dir2[ int( T2::dir_x[ k ] + 2 ) * 5 + int( T2::dir_y[ k ] + 2 ) ] = k;
    }

    static void pad( int N, int R, const std::vector<T> &psi, std::vector<T> &p ) {
        const int P = N + 2 * R;
        p.assign( P * P, -std::numeric_limits<T>::infinity() );
        for( int a = 0; a < N; ++a ) std::memcpy( &p[ ( a + R ) * P + R ], &psi[ a * N ], N * sizeof( T ) );
    }

    // cross-level certificate ( cascade.cpp )
    bool cell_ok( int i, T tol ) const {
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
            const int q = ( m2 ? dir2 : dir1 )[ ( ndx + 2 ) * 5 + ndy + 2 ];
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

    // scalar integration of a polygon; faces go to `extras`
    template<class Label>
    void scalar_cell( int i, int n, const T *px, const T *py, Label &&nbr ) {
        T face[ 160 ];
        rho.polygon( n, px, py, T( i / N ), T( i % N ), mass[ i ], face );
        for( int e = 0; e < n; ++e ) {
            const int j = nbr( e );
            if( j < 0 || face[ e ] == 0 ) continue;
            const T d = std::hypot( T( j / N - i / N ), T( j % N - i % N ) );
            extras.push_back( { i, j, face[ e ] / ( 2 * d ) } );
        }
        for( int k = 0; k < 8; ++k ) W[ k * P1 * P1 + ( i / N + 1 ) * P1 + i % N + 1 ] = 0;
    }

    void run( const std::vector<T> &psi ) {
        const T tol = 1e-9;
        double t0 = now_ms();
        pad( N, 1, psi, p1 );
        n_flag1 = K1::run_certified( N, p1.data(), area1.data(), hess1.data(), own1.data(), pl1, flag1.data(), tol );
        n_exact = 0;
        if( n_flag1 ) {
            pad( N, 2, psi, p2 );
            list1.clear();
            for( int i = 0; i < N * N; ++i ) if( flag1[ i ] ) list1.push_back( i );
            for( size_t c = 0; c < list1.size(); c += B )
                K2::cells_block( N, p2.data(), list1.data() + c, int( std::min<size_t>( B, list1.size() - c ) ), area2.data(), hess2.data(), own2.data(), &pl2 );
            check.clear();
            for( int i : list1 )
                for( int da = -1; da <= 1; ++da )
                    for( int db = -1; db <= 1; ++db ) {
                        const int a = i / N + da, b = i % N + db;
                        if( a < 0 || a >= N || b < 0 || b >= N || mark[ a * N + b ] ) continue;
                        mark[ a * N + b ] = 1;
                        check.push_back( a * N + b );
                    }
            for( int i : check ) { mark[ i ] = 0; flag2[ i ] = ! cell_ok( i, tol ); n_exact += flag2[ i ]; }
        }
        double t1 = now_ms();

        // ---- integrals of the ring-1 cells ( SIMD ), then the others ----
        extras.clear();
        n_dens = 0;
        const int PP = pl1.plane();
        for( int a = 0; a < N; ++a ) {
            for( int b0 = 0; b0 < N; b0 += B ) {
                alignas( 64 ) T vx[ 8 ][ B ], vy[ 8 ][ B ], face[ 8 ][ B ], ms[ B ], sa[ B ], sb[ B ];
                alignas( 64 ) unsigned char bad[ B ] = {};
                const int i0 = pl1.row( a ) + b0 + 1;
                for( int k = 0; k < 8; ++k )
                    for( int x = 0; x < B; ++x ) { vx[ k ][ x ] = pl1.vx[ k * PP + i0 + x ]; vy[ k ][ x ] = pl1.vy[ k * PP + i0 + x ]; }
                for( int x = 0; x < B; ++x ) { sa[ x ] = a; sb[ x ] = b0 + x; }
                rho.template cells<8, B>( vx, vy, sa, sb, ms, face, bad );
                for( int x = 0; x < B; ++x ) mass[ a * N + b0 + x ] = ms[ x ];
                for( int k = 0; k < 8; ++k ) {
                    const T inv = T( 0.5 ) * T( T1::dir_inv_norm[ k ] );
                    const signed char *ow = pl1.ow.data() + k * PP + i0;
                    T *w = W.data() + k * P1 * P1 + ( a + 1 ) * P1 + b0 + 1;
                    for( int x = 0; x < B; ++x ) w[ x ] = ow[ x ] >= 0 ? face[ k ][ x ] * inv : T( 0 );
                }
                for( int x = 0; x < B; ++x ) dbad[ a * N + b0 + x ] = bad[ x ];
            }
        }
        double t1b = now_ms();
        // second SIMD tier for the ring-1 cells that the first one could not take
        list1.clear();
        for( int i = 0; i < N * N; ++i ) if( dbad[ i ] && ! flag1[ i ] && ! flag2[ i ] ) list1.push_back( i );
        n_fine = list1.size();
        for( size_t c = 0; c < list1.size(); c += B ) {
            const int n = int( std::min<size_t>( B, list1.size() - c ) );
            alignas( 64 ) T vx[ 8 ][ B ], vy[ 8 ][ B ], face[ 8 ][ B ], ms[ B ], sa[ B ], sb[ B ];
            alignas( 64 ) unsigned char bad[ B ] = {};
            for( int x = 0; x < B; ++x ) {
                const int i = list1[ c + ( x < n ? x : n - 1 ) ], ii = pl1.row( i / N ) + i % N + 1;
                sa[ x ] = i / N; sb[ x ] = i % N;
                for( int k = 0; k < 8; ++k ) { vx[ k ][ x ] = pl1.vx[ k * PP + ii ]; vy[ k ][ x ] = pl1.vy[ k * PP + ii ]; }
            }
            rho.template cells_fine<8, B>( vx, vy, sa, sb, ms, face, bad );
            for( int x = 0; x < n; ++x ) {
                const int i = list1[ c + x ], ii = pl1.row( i / N ) + i % N + 1;
                dbad[ i ] = bad[ x ];
                if( bad[ x ] ) continue;
                mass[ i ] = ms[ x ];
                for( int k = 0; k < 8; ++k ) W[ k * P1 * P1 + ( i / N + 1 ) * P1 + i % N + 1 ] = pl1.ow[ k * PP + ii ] >= 0 ? face[ k ][ x ] * T( 0.5 ) * T( T1::dir_inv_norm[ k ] ) : T( 0 );
            }
        }
        double t1c = now_ms();
        // cells that need the scalar path
        for( int i = 0; i < N * N; ++i ) {
            if( flag1[ i ] && flag2[ i ] ) continue; // exact below
            if( ! flag1[ i ] && ! flag2[ i ] && ! dbad[ i ] ) continue;
            if( ! flag1[ i ] && flag2[ i ] ) continue;
            const bool l2 = flag1[ i ];
            const int D = l2 ? K2::D : K1::D, PQ = l2 ? pl2.plane() : PP, ii = l2 ? pl2.row( i / N ) + i % N + 2 : pl1.row( i / N ) + i % N + 1;
            const T *vx = l2 ? pl2.vx.data() : pl1.vx.data(), *vy = l2 ? pl2.vy.data() : pl1.vy.data();
            const signed char *ow = l2 ? pl2.ow.data() : pl1.ow.data();
            // polygon: vertex e = plane vertex ( e - 1 ), so that edge e ( vertex e -> e+1 ) is direction e
            T px[ 16 ], py[ 16 ];
            for( int e = 0; e < D; ++e ) { px[ e ] = vx[ ( ( e + D - 1 ) % D ) * PQ + ii ]; py[ e ] = vy[ ( ( e + D - 1 ) % D ) * PQ + ii ]; }
            ++n_dens;
            scalar_cell( i, D, px, py, [ & ]( int e ) {
                const int o = ow[ e * PQ + ii ];
                return o < 0 ? -1 : i + ( l2 ? T2::off_x[ o ] * N + T2::off_y[ o ] : T1::off_x[ o ] * N + T1::off_y[ o ] );
            } );
        }
        if( n_exact ) {
            pad( N, ek.RB, psi, pe );
            const T pmax = *std::max_element( psi.begin(), psi.end() );
            for( int i : check ) {
                if( ! flag2[ i ] ) continue;
                T px[ ExactKernel<T>::C ], py[ ExactKernel<T>::C ];
                int pl[ ExactKernel<T>::C ], n;
                ek.clip( N, pe.data(), pmax, i / N, i % N, px, py, pl, n );
                scalar_cell( i, n, px, py, [ & ]( int e ) { return pl[ e ] < 0 ? -1 : i + ek.ox[ pl[ e ] ] * N + ek.oy[ pl[ e ] ]; } );
            }
        }
        for( int i : check ) flag2[ i ] = 0;
        double t2 = now_ms();
        t_geom = t1 - t0;
        t_int = t2 - t1;
        t_tier1 = t1b - t1; t_tier2 = t1c - t1b; t_scalar = t2 - t1c;
    }
};

// ================================================================================================
// matrix-free operator and preconditioners ( vectors padded to ( N + 2 )^2, zero border )
// ================================================================================================
struct Operator {
    int N, P;
    const std::vector<T> *W;
    const void *extras_ptr;
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

    // y = H v, returns dot( v, y )
    T apply( const T *v, T *y ) const {
        const int PP = P * P;
        T dot = 0;
        for( int a = 0; a < N; ++a ) {
            const int r = ( a + 1 ) * P + 1;
            #pragma omp simd reduction( +: dot )
            for( int x = 0; x < N; ++x ) {
                const int i = r + x;
                const T vi = v[ i ];
                T s = 0;
                for( int k = 0; k < 8; ++k ) s += ( *W )[ k * PP + i ] * ( vi - v[ i + off[ k ] ] );
                y[ i ] = s;
                dot += vi * s;
            }
        }
        for( size_t e = 0; e < ex_w.size(); ++e ) {
            const T d = ex_w[ e ] * ( v[ ex_ij[ e ].first ] - v[ ex_ij[ e ].second ] );
            y[ ex_ij[ e ].first ] += d;
            dot += v[ ex_ij[ e ].first ] * d;
        }
        return dot;
    }

    void diag( std::vector<T> &d ) const {
        const int PP = P * P;
        d.assign( PP, 0 );
        for( int a = 0; a < N; ++a )
            for( int b = 0; b < N; ++b ) {
                const int i = ( a + 1 ) * P + b + 1;
                for( int k = 0; k < 8; ++k ) d[ i ] += ( *W )[ k * PP + i ];
            }
        for( size_t e = 0; e < ex_w.size(); ++e ) d[ ex_ij[ e ].first ] += ex_w[ e ];
    }
};

struct Precond {
    // 0: Jacobi, 1: box ( 2 x 2 block sums, copy back, D_l = Galerkin diagonal of the aggregation ),
    // 2: tent ( bilinear prolongation, restriction = its transpose [ 1 3 3 1 ] / 4, D_l = c D_agg / 2^l:
    //    the Galerkin diagonal of the aggregation grows like 2^l, the bilinear one is scale invariant )
    int mode;
    bool multilevel;
    T c_tent = T( 2 ) / 3;
    std::vector<T> r0, tmp;
    int N, P, L;
    std::vector<T> d0;                       // padded fine diagonal
    std::vector<std::vector<T>> Dl, rl;      // levels 1..L ( n_l x n_l, n_l = N >> l ), unpadded
    std::vector<std::vector<T>> Wl[ 2 ];     // coarse 8-direction weights while building

    std::vector<T> id0;                      // padded inverse of d0 ( 0 where d0 = 0 )
    void build( const Operator &op ) {
        N = op.N; P = op.P;
        op.diag( d0 );
        id0.resize( d0.size() );
        for( size_t i = 0; i < d0.size(); ++i ) id0[ i ] = d0[ i ] > 0 ? 1 / d0[ i ] : 0;
        if( ! multilevel ) return;
        L = 0;
        while( ( N >> ( L + 1 ) ) >= 2 ) ++L;
        Dl.assign( L + 1, {} ); rl.assign( L + 1, {} );
        // level-0 weights ( unpadded ), 8 directions
        std::vector<T> w( 8 * N * N );
        for( int k = 0; k < 8; ++k )
            for( int a = 0; a < N; ++a )
                for( int b = 0; b < N; ++b ) w[ k * N * N + a * N + b ] = ( *op.W )[ k * P * P + ( a + 1 ) * P + b + 1 ];
        int dirk[ 9 ];
        for( int k = 0; k < 8; ++k ) dirk[ ( T1::off_x[ k ] + 1 ) * 3 + T1::off_y[ k ] + 1 ] = k;
        int n = N;
        for( int l = 1; l <= L; ++l ) {
            const int m = n / 2;
            std::vector<T> wc( 8 * m * m, 0 );
            for( int k = 0; k < 8; ++k )
                for( int a = 0; a < n; ++a )
                    for( int b = 0; b < n; ++b ) {
                        const T x = w[ k * n * n + a * n + b ];
                        if( x == 0 ) continue;
                        const int ja = a + T1::off_x[ k ], jb = b + T1::off_y[ k ];
                        const int ca = a / 2, cb = b / 2, da = ja / 2 - ca, db = jb / 2 - cb;
                        if( da == 0 && db == 0 ) continue;
                        wc[ dirk[ ( da + 1 ) * 3 + db + 1 ] * m * m + ca * m + cb ] += x;
                    }
            Dl[ l ].assign( m * m, 0 );
            for( int k = 0; k < 8; ++k )
                for( int c = 0; c < m * m; ++c ) Dl[ l ][ c ] += wc[ k * m * m + c ];
            rl[ l ].assign( m * m, 0 );
            w.swap( wc );
            n = m;
        }
        // extras: the crossing part at every level
        for( size_t e = 0; e < op.ex_w.size(); ++e ) {
            const int i = op.ex_ij[ e ].first, j = op.ex_ij[ e ].second;
            int ia = i / P - 1, ib = i % P - 1, ja = j / P - 1, jb = j % P - 1;
            for( int l = 1; l <= L; ++l ) {
                ia /= 2; ib /= 2; ja /= 2; jb /= 2;
                if( ia != ja || ib != jb ) Dl[ l ][ ia * ( N >> l ) + ib ] += op.ex_w[ e ];
            }
        }
        for( int l = 1; l <= L; ++l ) for( auto &x : Dl[ l ] ) x = x > 0 ? ( mode == 2 ? T( 1 << l ) / c_tent : T( 1 ) ) / x : 0;
        r0.assign( N * N, 0 );
        tmp.assign( N * N, 0 );
    }

    // tent transfer along one axis: f has n entries ( stride sf ), c has m = n / 2 ( stride sc )
    static void restrict1( const T *f, int sf, T *c, int sc, int m ) {
        for( int B = 0; B < m; ++B ) {
            T v = ( B == 0 ? T( 1 ) : T( 0.75 ) ) * f[ 2 * B * sf ] + ( B == m - 1 ? T( 1 ) : T( 0.75 ) ) * f[ ( 2 * B + 1 ) * sf ];
            if( B >= 1 ) v += T( 0.25 ) * f[ ( 2 * B - 1 ) * sf ];
            if( B <= m - 2 ) v += T( 0.25 ) * f[ ( 2 * B + 2 ) * sf ];
            c[ B * sc ] = v;
        }
    }
    static void prolong1_add( const T *c, int sc, T *f, int sf, int m ) {
        for( int B = 0; B < m; ++B ) {
            f[ 2 * B * sf ] += B == 0 ? c[ 0 ] : T( 0.75 ) * c[ B * sc ] + T( 0.25 ) * c[ ( B - 1 ) * sc ];
            f[ ( 2 * B + 1 ) * sf ] += B == m - 1 ? c[ B * sc ] : T( 0.75 ) * c[ B * sc ] + T( 0.25 ) * c[ ( B + 1 ) * sc ];
        }
    }
    // one row of the restriction: t ( m ) = R fr ( 2 m ), along b
    static void restrict_row( const T *fr, int m, T *t ) {
        t[ 0 ] = fr[ 0 ] + T( 0.75 ) * fr[ 1 ] + ( m > 1 ? T( 0.25 ) * fr[ 2 ] : T( 0 ) );
        #pragma omp simd
        for( int B = 1; B < m - 1; ++B ) t[ B ] = T( 0.75 ) * ( fr[ 2 * B ] + fr[ 2 * B + 1 ] ) + T( 0.25 ) * ( fr[ 2 * B - 1 ] + fr[ 2 * B + 2 ] );
        if( m > 1 ) t[ m - 1 ] = T( 0.75 ) * fr[ 2 * m - 2 ] + fr[ 2 * m - 1 ] + T( 0.25 ) * fr[ 2 * m - 3 ];
    }
    // one row of the prolongation: fr ( 2 m ) += P t ( m ), along b
    static void prolong_row_add( const T *t, int m, T *fr ) {
        const int n = 2 * m;
        fr[ 0 ] += t[ 0 ];
        fr[ n - 1 ] += t[ m - 1 ];
        #pragma omp simd
        for( int B = 1; B < m; ++B ) fr[ 2 * B ] += T( 0.75 ) * t[ B ] + T( 0.25 ) * t[ B - 1 ];
        #pragma omp simd
        for( int B = 0; B < m - 1; ++B ) fr[ 2 * B + 1 ] += T( 0.75 ) * t[ B ] + T( 0.25 ) * t[ B + 1 ];
    }
    // c ( m x m ) = R f ( n x n ): rows first ( contiguous along b ), then combinations of whole rows
    void restrict2( const T *f, int n, T *c ) {
        const int m = n / 2;
        for( int a = 0; a < n; ++a ) restrict_row( f + a * n, m, tmp.data() + a * m );
        restrict_cols( m, c );
    }
    // second half of restrict2: c ( m x m ) from the row-restricted rows in tmp ( 2 m x m )
    void restrict_cols( int m, T *c ) {
        for( int A = 0; A < m; ++A ) {
            const T w0 = A == 0 ? T( 1 ) : T( 0.75 ), w1 = A == m - 1 ? T( 1 ) : T( 0.75 ), wm = A >= 1 ? T( 0.25 ) : T( 0 ), wp = A <= m - 2 ? T( 0.25 ) : T( 0 );
            const T *r0 = tmp.data() + 2 * A * m, *r1 = r0 + m, *rm = A >= 1 ? r0 - m : r0, *rp = A <= m - 2 ? r1 + m : r1;
            T *cr = c + A * m;
            #pragma omp simd
            for( int b = 0; b < m; ++b ) cr[ b ] = w0 * r0[ b ] + w1 * r1[ b ] + wm * rm[ b ] + wp * rp[ b ];
        }
    }
    // f ( n x n ) += P c ( m x m )
    void prolong2_add( const T *c, int m, T *f ) {
        prolong_cols( c, m );
        for( int a = 0; a < 2 * m; ++a ) prolong_row_add( tmp.data() + a * m, m, f + a * 2 * m );
    }
    // first half of prolong2_add: along a, tmp ( 2 m x m ) = P c ( m x m )
    void prolong_cols( const T *c, int m ) {
        for( int A = 0; A < m; ++A ) {
            const T *c0 = c + A * m, *cm = A >= 1 ? c0 - m : c0, *cp = A <= m - 2 ? c0 + m : c0;
            const T e0 = A == 0 ? T( 1 ) : T( 0.75 ), em = A == 0 ? T( 0 ) : T( 0.25 ), e1 = A == m - 1 ? T( 1 ) : T( 0.75 ), ep = A == m - 1 ? T( 0 ) : T( 0.25 );
            T *t0 = tmp.data() + 2 * A * m, *t1 = t0 + m;
            #pragma omp simd
            for( int b = 0; b < m; ++b ) { t0[ b ] = e0 * c0[ b ] + em * cm[ b ]; t1[ b ] = e1 * c0[ b ] + ep * cp[ b ]; }
        }
    }

    // tent, fused with the CG passes: the caller has put the row-restricted fine residual in tmp
    // ( N x N / 2, see restrict_row ); on return tmp holds the column-prolongated level-1
    // correction, to be finished row by row with prolong_row_add
    void coarse_tent() {
        restrict_cols( N / 2, rl[ 1 ].data() );
        for( int l = 2; l <= L; ++l ) restrict2( rl[ l - 1 ].data(), N >> ( l - 1 ), rl[ l ].data() );
        for( int l = L; l >= 1; --l ) {
            const int m = N >> l;
            T *rv = rl[ l ].data();
            const T *dv = Dl[ l ].data();
            #pragma omp simd
            for( int c = 0; c < m * m; ++c ) rv[ c ] *= dv[ c ];
            if( l < L ) prolong2_add( rl[ l + 1 ].data(), m / 2, rv );
        }
        prolong_cols( rl[ 1 ].data(), N / 2 );
    }

    // hooks of pcg_fused ( tent ): row half of the restriction, coarse levels, row half of the prolongation
    void row_b( int a, const T *rv ) { restrict_row( rv, N / 2, tmp.data() + a * ( N / 2 ) ); }
    void coarse() { coarse_tent(); }
    void row_c( int a, T *zr ) const { prolong_row_add( tmp.data() + a * ( N / 2 ), N / 2, zr ); }

    void apply( const T *r, T *z ) {
        // level 0: Jacobi
        for( int a = 0; a < N; ++a ) {
            const int row = ( a + 1 ) * P + 1;
            #pragma omp simd
            for( int x = 0; x < N; ++x ) z[ row + x ] = d0[ row + x ] > 0 ? r[ row + x ] / d0[ row + x ] : 0;
        }
        if( ! multilevel ) return;
        if( mode == 2 ) {
            for( int a = 0; a < N; ++a ) std::memcpy( &r0[ a * N ], &r[ ( a + 1 ) * P + 1 ], N * sizeof( T ) );
            restrict2( r0.data(), N, rl[ 1 ].data() );
            for( int l = 2; l <= L; ++l ) restrict2( rl[ l - 1 ].data(), N >> ( l - 1 ), rl[ l ].data() );
            for( int l = L; l >= 1; --l ) {
                const int m = N >> l;
                for( int c = 0; c < m * m; ++c ) rl[ l ][ c ] *= Dl[ l ][ c ];
                if( l < L ) prolong2_add( rl[ l + 1 ].data(), m / 2, rl[ l ].data() );
            }
            std::fill( r0.begin(), r0.end(), 0 );
            prolong2_add( rl[ 1 ].data(), N / 2, r0.data() );
            for( int a = 0; a < N; ++a ) {
                const int row = ( a + 1 ) * P + 1;
                #pragma omp simd
                for( int x = 0; x < N; ++x ) z[ row + x ] += r0[ a * N + x ];
            }
            return;
        }
        // restriction: 2 x 2 block sums ( box blur + decimation )
        {
            const int m = N / 2;
            for( int a = 0; a < m; ++a )
                for( int b = 0; b < m; ++b ) {
                    const int i = ( 2 * a + 1 ) * P + 2 * b + 1;
                    rl[ 1 ][ a * m + b ] = r[ i ] + r[ i + 1 ] + r[ i + P ] + r[ i + P + 1 ];
                }
        }
        for( int l = 2; l <= L; ++l ) {
            const int n = N >> ( l - 1 ), m = n / 2;
            const T *f = rl[ l - 1 ].data();
            T *c = rl[ l ].data();
            for( int a = 0; a < m; ++a )
                for( int b = 0; b < m; ++b )
                    c[ a * m + b ] = f[ 2 * a * n + 2 * b ] + f[ 2 * a * n + 2 * b + 1 ] + f[ ( 2 * a + 1 ) * n + 2 * b ] + f[ ( 2 * a + 1 ) * n + 2 * b + 1 ];
        }
        // e_l = r_l / D_l, prolongation by copy, accumulated from the coarsest level ( in place in rl )
        for( int l = L; l >= 1; --l ) {
            const int m = N >> l;
            for( int c = 0; c < m * m; ++c ) rl[ l ][ c ] *= Dl[ l ][ c ];
            if( l < L ) {
                const T *cc = rl[ l + 1 ].data();
                const int h = m / 2;
                for( int a = 0; a < m; ++a )
                    for( int b = 0; b < m; ++b ) rl[ l ][ a * m + b ] += cc[ ( a / 2 ) * h + b / 2 ];
            }
        }
        const int m = N / 2;
        for( int a = 0; a < N; ++a ) {
            const int row = ( a + 1 ) * P + 1;
            const T *c = rl[ 1 ].data() + ( a / 2 ) * m;
            #pragma omp simd
            for( int x = 0; x < N; ++x ) z[ row + x ] += c[ x / 2 ];
        }
    }
};

// Tensor ( semi-coarsened ) multilevel blur: z = r / diag + sum over ( la, lb ) of P D^-1 P^T r, where
// the subspace ( la, lb ) blurs with a tent of width 2^la along a and 2^lb along b ( grid
// ( N >> la ) x ( N >> lb ) ). The isotropic tent is the diagonal la = lb; the other subspaces hold
// the modes that are smooth along one axis only ( strips of elongated cells ). Transfers are the
// separable 1D tents of Precond, along one axis at a time: every restricted grid comes from its
// neighbor ( la - 1, 0 ) or ( la, lb - 1 ), so the whole set costs ~ 4 N^2 per sweep.
// D( la, lb ) = c ( Da / 2^la + Db / 2^lb ), Da ( Db ) the weight leaving the 2^la x 2^lb block
// across its a ( b ) faces: the tent energy of a face scales as its length / the width across it.
// Kept subspaces: |la - lb| <= band, plus la = 0 or lb = 0 when `axes`.
struct TensorPrecond {
    int N, P, L, band = 1 << 20;
    bool axes = true;
    T c = T( 2 ) / 3;
    std::vector<T> d0, id0;
    std::vector<std::vector<T>> g, Di;  // ( L + 1 )^2 grids, index la * ( L + 1 ) + lb
    std::vector<T> wa, w, wc, Da, Db;       // build buffers
    int nkept = 0;
    double t_restrict = 0, t_scale = 0, t_prolong = 0;

    // weight of a block: ( min / max of its two tent energies )^pw. A block whose shape matches the
    // local anisotropy has balanced energies; the mismatched shapes of the same family fade out, so a
    // smooth mode is not counted by every aspect ratio ( lambda_max of the additive sum stays low )
    T pw = 0;
    static T balance( T ea, T eb, T pw ) {
        if( pw == 0 ) return 1;
        const T lo = std::min( ea, eb ), hi = std::max( ea, eb );
        if( hi == 0 ) return 0;
        return pw == 1 ? lo / hi : pw == T( 0.5 ) ? std::sqrt( lo / hi ) : std::pow( lo / hi, pw );
    }
    bool kept( int la, int lb ) const { return ( la || lb ) && ( std::abs( la - lb ) <= band || ( axes && ( la == 0 || lb == 0 ) ) ); }
    int gi( int la, int lb ) const { return la * ( L + 1 ) + lb; }

    // 8-direction weights of an na x nb grid, coarsened along one axis. For a fine index of parity p
    // along that axis, direction k lands in the coarse direction of offset floor( ( p + o_k ) / 2 ),
    // fixed per ( k, p ): plain strided sums ( weights toward the outside of the grid are 0 )
    static void coarsen( const std::vector<T> &w, int na, int nb, int axis, std::vector<T> &wc ) {
        int dirk[ 9 ];
        for( int k = 0; k < 8; ++k ) dirk[ ( T1::off_x[ k ] + 1 ) * 3 + T1::off_y[ k ] + 1 ] = k;
        const int ma = axis == 0 ? na / 2 : na, mb = axis == 1 ? nb / 2 : nb, mm = ma * mb;
        wc.assign( 8 * mm, 0 );
        for( int k = 0; k < 8; ++k )
            for( int p = 0; p < 2; ++p ) {
                const int o = axis == 0 ? T1::off_x[ k ] : T1::off_y[ k ], dc = ( p + o + 2 ) / 2 - 1;
                const int da = axis == 0 ? dc : T1::off_x[ k ], db = axis == 1 ? dc : T1::off_y[ k ];
                if( da == 0 && db == 0 ) continue;
                T *dst = wc.data() + dirk[ ( da + 1 ) * 3 + db + 1 ] * mm;
                const T *src = w.data() + k * na * nb;
                if( axis == 0 ) {
                    for( int A = 0; A < ma; ++A ) {
                        const T *sr = src + ( 2 * A + p ) * nb;
                        T *dr = dst + A * mb;
                        #pragma omp simd
                        for( int b = 0; b < mb; ++b ) dr[ b ] += sr[ b ];
                    }
                } else {
                    for( int a = 0; a < ma; ++a ) {
                        const T *sr = src + a * nb + p;
                        T *dr = dst + a * mb;
                        #pragma omp simd
                        for( int B = 0; B < mb; ++B ) dr[ B ] += sr[ 2 * B ];
                    }
                }
            }
    }

    double t_setup = 0;
    bool planes = false; // scaling from the coarsened weight planes ( exact crossing weights ), else from conductance fields
    void build( const Operator &op ) { if( planes ) build_planes( op ); else build_fields( op ); }

    void alloc( int n ) {
        N = n;
        L = 0;
        while( ( N >> ( L + 1 ) ) >= 2 ) ++L;
        g.resize( ( L + 1 ) * ( L + 1 ) ); Di.resize( ( L + 1 ) * ( L + 1 ) );
        nkept = 0;
        for( int la = 0; la <= L; ++la )
            for( int lb = 0; lb <= L; ++lb ) {
                g[ gi( la, lb ) ].resize( ( N >> la ) * ( N >> lb ) );
                if( kept( la, lb ) ) { Di[ gi( la, lb ) ].resize( ( N >> la ) * ( N >> lb ) ); ++nkept; } else Di[ gi( la, lb ) ].clear();
            }
    }
    // whole restriction tree of the unpadded field in g[ 0 ]
    void restrict_tree( int from_la = 1 ) {
        for( int la = from_la; la <= L; ++la ) restrict_a( g[ gi( la - 1, 0 ) ].data(), N >> ( la - 1 ), N, g[ gi( la, 0 ) ].data() );
        for( int la = 0; la <= L; ++la )
            for( int lb = la == 0 && from_la == 2 ? 2 : 1; lb <= L; ++lb ) restrict_b( g[ gi( la, lb - 1 ) ].data(), N >> la, N >> ( lb - 1 ), g[ gi( la, lb ) ].data() );
    }

    // scaling from conductances: ka( i ) = sum_k w_k | oa_k |, kb( i ) = sum_k w_k | ob_k |. The weight
    // crossing the a faces of a block ~ sum_block ka / ha, and the tent restriction sums ~ the block
    // ( its weights add up to ha hb ): Ea = c R ka / ha^2, Eb = c R kb / hb^2. Two restrictions.
    void build_fields( const Operator &op ) {
        const double t0 = now_ms();
        if( N != op.N || g.empty() ) alloc( op.N );
        P = op.P;
        const int PP = P * P;
        d0.assign( P * P, 0 );
        Da.assign( N * N, 0 ); Db.assign( N * N, 0 );
        for( int a = 0; a < N; ++a ) {
            T *dr = &d0[ ( a + 1 ) * P + 1 ], *ra = &Da[ a * N ], *rb = &Db[ a * N ];
            for( int k = 0; k < 8; ++k ) {
                const T *wr = &( *op.W )[ k * PP + ( a + 1 ) * P + 1 ];
                const bool fa = T1::off_x[ k ], fb = T1::off_y[ k ];
                #pragma omp simd
                for( int b = 0; b < N; ++b ) { dr[ b ] += wr[ b ]; ra[ b ] += fa ? wr[ b ] : T( 0 ); rb[ b ] += fb ? wr[ b ] : T( 0 ); }
            }
        }
        for( size_t e = 0; e < op.ex_w.size(); ++e ) {
            const int i = op.ex_ij[ e ].first, j = op.ex_ij[ e ].second, f = ( i / P - 1 ) * N + i % P - 1;
            d0[ i ] += op.ex_w[ e ];
            Da[ f ] += std::abs( j / P - i / P ) * op.ex_w[ e ];
            Db[ f ] += std::abs( j % P - i % P ) * op.ex_w[ e ];
        }
        id0.resize( d0.size() );
        #pragma omp simd
        for( size_t i = 0; i < d0.size(); ++i ) id0[ i ] = d0[ i ] > 0 ? 1 / d0[ i ] : 0;
        for( int pass = 0; pass < 2; ++pass ) {
            std::copy( ( pass ? Db : Da ).begin(), ( pass ? Db : Da ).end(), g[ 0 ].begin() );
            restrict_tree();
            for( int la = 0; la <= L; ++la )
                for( int lb = 0; lb <= L; ++lb ) {
                    if( ! kept( la, lb ) ) continue;
                    T *D = Di[ gi( la, lb ) ].data();
                    const T *v = g[ gi( la, lb ) ].data();
                    const int n = ( N >> la ) * ( N >> lb );
                    if( ! pass ) {
                        const T sa = c / T( 1 << la ) / T( 1 << la );
                        #pragma omp simd
                        for( int q = 0; q < n; ++q ) D[ q ] = sa * v[ q ];
                    } else {
                        const T sb = c / T( 1 << lb ) / T( 1 << lb );
                        for( int q = 0; q < n; ++q ) {
                            const T ea = D[ q ], eb = sb * v[ q ], d = ea + eb;
                            D[ q ] = d > 0 ? balance( ea, eb, pw ) / d : 0;
                        }
                    }
                }
        }
        t_setup += now_ms() - t0;
    }

    void build_planes( const Operator &op ) {
        const double t0 = now_ms();
        N = op.N; P = op.P;
        L = 0;
        while( ( N >> ( L + 1 ) ) >= 2 ) ++L;
        g.resize( ( L + 1 ) * ( L + 1 ) ); Di.resize( ( L + 1 ) * ( L + 1 ) );
        wa.resize( 8 * N * N );
        for( int k = 0; k < 8; ++k )
            for( int a = 0; a < N; ++a ) std::memcpy( &wa[ ( k * N + a ) * N ], &( *op.W )[ k * P * P + ( a + 1 ) * P + 1 ], N * sizeof( T ) );
        // fine diagonal: row sums of the planes, plus the extras
        d0.assign( P * P, 0 );
        for( int a = 0; a < N; ++a ) {
            T *dr = &d0[ ( a + 1 ) * P + 1 ];
            for( int k = 0; k < 8; ++k ) {
                const T *wr = &wa[ ( k * N + a ) * N ];
                #pragma omp simd
                for( int b = 0; b < N; ++b ) dr[ b ] += wr[ b ];
            }
        }
        for( size_t e = 0; e < op.ex_w.size(); ++e ) d0[ op.ex_ij[ e ].first ] += op.ex_w[ e ];
        id0.resize( d0.size() );
        #pragma omp simd
        for( size_t i = 0; i < d0.size(); ++i ) id0[ i ] = d0[ i ] > 0 ? 1 / d0[ i ] : 0;
        nkept = 0;
        for( int la = 0; la <= L; ++la ) {
            if( la ) { coarsen( wa, N >> ( la - 1 ), N, 0, wc ); wa.swap( wc ); }
            for( int lb = 0; lb <= L; ++lb ) {
                const int na = N >> la, nb = N >> lb;
                if( lb == 1 ) coarsen( wa, na, nb * 2, 1, w );
                else if( lb ) { coarsen( w, na, nb * 2, 1, wc ); w.swap( wc ); }
                const std::vector<T> &wl = lb ? w : wa;
                g[ gi( la, lb ) ].resize( na * nb );
                auto &D = Di[ gi( la, lb ) ];
                if( ! kept( la, lb ) ) { D.clear(); continue; }
                ++nkept;
                Da.assign( na * nb, 0 ); Db.assign( na * nb, 0 );
                for( int k = 0; k < 8; ++k ) {
                    const T *wk = wl.data() + k * na * nb;
                    if( T1::off_x[ k ] ) {
                        #pragma omp simd
                        for( int c = 0; c < na * nb; ++c ) Da[ c ] += wk[ c ];
                    }
                    if( T1::off_y[ k ] ) {
                        #pragma omp simd
                        for( int c = 0; c < na * nb; ++c ) Db[ c ] += wk[ c ];
                    }
                }
                for( size_t e = 0; e < op.ex_w.size(); ++e ) {
                    const int i = op.ex_ij[ e ].first, j = op.ex_ij[ e ].second;
                    const int ia = ( i / P - 1 ) >> la, ib = ( i % P - 1 ) >> lb, ja = ( j / P - 1 ) >> la, jb = ( j % P - 1 ) >> lb;
                    if( ia != ja ) Da[ ia * nb + ib ] += op.ex_w[ e ];
                    if( ib != jb ) Db[ ia * nb + ib ] += op.ex_w[ e ];
                }
                D.resize( na * nb );
                const T sa = c / T( 1 << la ), sb = c / T( 1 << lb );
                for( int q = 0; q < na * nb; ++q ) {
                    const T ea = sa * Da[ q ], eb = sb * Db[ q ], d = ea + eb;
                    D[ q ] = d > 0 ? balance( ea, eb, pw ) / d : 0;
                }
            }
        }
        t_setup += now_ms() - t0;
    }

    // 1D tents ( see Precond::restrict_row / prolong_row_add ) along a ( whole rows ) or b ( in rows )
    static void restrict_a( const T *f, int na, int nb, T *c ) {
        const int m = na / 2;
        for( int A = 0; A < m; ++A ) {
            const T w0 = A == 0 ? T( 1 ) : T( 0.75 ), w1 = A == m - 1 ? T( 1 ) : T( 0.75 ), wm = A >= 1 ? T( 0.25 ) : T( 0 ), wp = A <= m - 2 ? T( 0.25 ) : T( 0 );
            const T *r0 = f + 2 * A * nb, *r1 = r0 + nb, *rm = A >= 1 ? r0 - nb : r0, *rp = A <= m - 2 ? r1 + nb : r1;
            T *cr = c + A * nb;
            #pragma omp simd
            for( int b = 0; b < nb; ++b ) cr[ b ] = w0 * r0[ b ] + w1 * r1[ b ] + wm * rm[ b ] + wp * rp[ b ];
        }
    }
    static void restrict_b( const T *f, int na, int nb, T *c ) {
        for( int a = 0; a < na; ++a ) Precond::restrict_row( f + a * nb, nb / 2, c + a * ( nb / 2 ) );
    }
    static void prolong_a_add( const T *c, int ma, int nb, T *f ) {
        for( int A = 0; A < ma; ++A ) {
            const T *c0 = c + A * nb, *cm = A >= 1 ? c0 - nb : c0, *cp = A <= ma - 2 ? c0 + nb : c0;
            const T e0 = A == 0 ? T( 1 ) : T( 0.75 ), em = A == 0 ? T( 0 ) : T( 0.25 ), e1 = A == ma - 1 ? T( 1 ) : T( 0.75 ), ep = A == ma - 1 ? T( 0 ) : T( 0.25 );
            T *t0 = f + 2 * A * nb, *t1 = t0 + nb;
            #pragma omp simd
            for( int b = 0; b < nb; ++b ) { t0[ b ] += e0 * c0[ b ] + em * cm[ b ]; t1[ b ] += e1 * c0[ b ] + ep * cp[ b ]; }
        }
    }
    static void prolong_b_add( const T *c, int na, int mb, T *f ) {
        for( int a = 0; a < na; ++a ) Precond::prolong_row_add( c + a * mb, mb, f + a * 2 * mb );
    }

    // ---- hooks of pcg_fused ----
    // pass B, row a of the new residual: copy, its b restriction ( 0, 1 ), and the a restriction
    // ( 1, 0 ) of the coarse row whose 4 fine rows are now complete
    void row_b( int a, const T *rv ) {
        T *g00 = g[ 0 ].data();
        std::memcpy( g00 + a * N, rv, N * sizeof( T ) );
        Precond::restrict_row( rv, N / 2, g[ gi( 0, 1 ) ].data() + a * ( N / 2 ) );
        const int m = N / 2;
        auto crow = [ & ]( int A ) {
            const T w0 = A == 0 ? T( 1 ) : T( 0.75 ), w1 = A == m - 1 ? T( 1 ) : T( 0.75 ), wm = A >= 1 ? T( 0.25 ) : T( 0 ), wp = A <= m - 2 ? T( 0.25 ) : T( 0 );
            const T *r0 = g00 + 2 * A * N, *r1 = r0 + N, *rm = A >= 1 ? r0 - N : r0, *rp = A <= m - 2 ? r1 + N : r1;
            T *cr = g[ gi( 1, 0 ) ].data() + A * N;
            #pragma omp simd
            for( int b = 0; b < N; ++b ) cr[ b ] = w0 * r0[ b ] + w1 * r1[ b ] + wm * rm[ b ] + wp * rp[ b ];
        };
        if( a >= 2 && a % 2 == 0 ) crow( a / 2 - 1 );
        if( a == N - 1 ) crow( m - 1 );
    }
    // the rest of the tree, scaling, prolongation down to ( 1, 0 ) and ( 0, 1 )
    void coarse() {
        double t0 = now_ms();
        restrict_tree( 2 );
        double t1 = now_ms();
        for( int q = 1; q < ( L + 1 ) * ( L + 1 ); ++q ) {
            T *v = g[ q ].data();
            const int n = int( g[ q ].size() );
            if( Di[ q ].empty() ) { std::fill( v, v + n, T( 0 ) ); continue; }
            const T *d = Di[ q ].data();
            #pragma omp simd
            for( int c2 = 0; c2 < n; ++c2 ) v[ c2 ] *= d[ c2 ];
        }
        double t2 = now_ms();
        for( int la = 0; la <= L; ++la )
            for( int lb = L; lb >= ( la ? 1 : 2 ); --lb ) prolong_b_add( g[ gi( la, lb ) ].data(), N >> la, N >> lb, g[ gi( la, lb - 1 ) ].data() );
        for( int la = L; la >= 2; --la ) prolong_a_add( g[ gi( la, 0 ) ].data(), N >> la, N, g[ gi( la - 1, 0 ) ].data() );
        double t3 = now_ms();
        t_restrict += t1 - t0; t_scale += t2 - t1; t_prolong += t3 - t2;
    }
    // pass C, row a of z ( already r / diag ): add the ( 0, 1 ) and ( 1, 0 ) parts
    void row_c( int a, T *zr ) const {
        Precond::prolong_row_add( g[ gi( 0, 1 ) ].data() + a * ( N / 2 ), N / 2, zr );
        const int m = N / 2, A = a / 2, W = N;
        const T *c0 = g[ gi( 1, 0 ) ].data() + A * W;
        T e, en;
        const T *cn;
        if( a % 2 == 0 ) { e = A == 0 ? T( 1 ) : T( 0.75 ); en = A == 0 ? T( 0 ) : T( 0.25 ); cn = A >= 1 ? c0 - W : c0; }
        else { e = A == m - 1 ? T( 1 ) : T( 0.75 ); en = A == m - 1 ? T( 0 ) : T( 0.25 ); cn = A <= m - 2 ? c0 + W : c0; }
        #pragma omp simd
        for( int b = 0; b < N; ++b ) zr[ b ] += e * c0[ b ] + en * cn[ b ];
    }

    void apply( const T *r, T *z ) {
        double t0 = now_ms();
        T *g00 = g[ 0 ].data();
        for( int a = 0; a < N; ++a ) std::memcpy( g00 + a * N, &r[ ( a + 1 ) * P + 1 ], N * sizeof( T ) );
        for( int la = 1; la <= L; ++la ) restrict_a( g[ gi( la - 1, 0 ) ].data(), N >> ( la - 1 ), N, g[ gi( la, 0 ) ].data() );
        for( int la = 0; la <= L; ++la )
            for( int lb = 1; lb <= L; ++lb ) restrict_b( g[ gi( la, lb - 1 ) ].data(), N >> la, N >> ( lb - 1 ), g[ gi( la, lb ) ].data() );
        double t1 = now_ms();
        for( int q = 1; q < ( L + 1 ) * ( L + 1 ); ++q ) {
            T *v = g[ q ].data();
            const int n = int( g[ q ].size() );
            if( Di[ q ].empty() ) { std::fill( v, v + n, T( 0 ) ); continue; }
            const T *d = Di[ q ].data();
            #pragma omp simd
            for( int c = 0; c < n; ++c ) v[ c ] *= d[ c ];
        }
        std::fill( g00, g00 + N * N, T( 0 ) );
        double t2 = now_ms();
        for( int la = 0; la <= L; ++la )
            for( int lb = L; lb >= 1; --lb ) prolong_b_add( g[ gi( la, lb ) ].data(), N >> la, N >> lb, g[ gi( la, lb - 1 ) ].data() );
        for( int la = L; la >= 1; --la ) prolong_a_add( g[ gi( la, 0 ) ].data(), N >> la, N, g[ gi( la - 1, 0 ) ].data() );
        for( int a = 0; a < N; ++a ) {
            const int row = ( a + 1 ) * P + 1;
            #pragma omp simd
            for( int x = 0; x < N; ++x ) z[ row + x ] = r[ row + x ] * id0[ row + x ] + g00[ a * N + x ];
        }
        double t3 = now_ms();
        t_restrict += t1 - t0; t_scale += t2 - t1; t_prolong += t3 - t2;
    }
};

// Tensor blurs in several frames: the axis frame ( a, b ), and the sheared frames ( a, c ) with
// c = b - s a ( s = +-1 ), where moving along a at fixed c follows the diagonal ( 1, s ). In a sheared
// frame the subspaces ( la, lc ) with la > lc blur along the diagonal, i.e. hold the modes of strips
// of cells elongated at 45 / 135 degrees. A sheared frame is a ( N x 2N ) array, zero outside the
// grid. Same transfers as TensorPrecond ( 1D tents, one axis at a time ); D( la, lc ) = c ( Da / 2^la
// + Dc / 2^lc ) with the weights crossing the faces of the block in frame coordinates, computed by
// brute force here ( experiment: iteration counts first ).
// Kept: axis frame |la - lb| <= band; sheared frames 1 <= la - lc <= dband.
struct FramePrecond {
    struct Frame {
        int s, na, nb, La, Lb;            // base grid na x nb, levels 0..La along a, 0..Lb along b
        std::vector<int> fine;            // base index -> padded fine index, -1 outside
        std::vector<std::vector<T>> g, Di;
        int gi( int la, int lb ) const { return la * ( Lb + 1 ) + lb; }
    };
    int N, P, band = 1, dband = 2;
    T c = T( 2 ) / 3;
    std::vector<T> d0, id0;
    std::vector<Frame> fr;
    int nkept = 0;
    double t_restrict = 0, t_scale = 0, t_prolong = 0, t_setup = 0;

    T pw = 0;
    static T balance( T ea, T eb, T pw ) { return TensorPrecond::balance( ea, eb, pw ); }
    bool kept( const Frame &f, int la, int lb ) const {
        if( ! la && ! lb ) return false;
        if( f.s == 0 ) return std::abs( la - lb ) <= band;
        return la - lb >= 1 && la - lb <= dband;
    }

    void build( const Operator &op ) {
        const double t0 = now_ms();
        N = op.N; P = op.P;
        op.diag( d0 );
        id0.resize( d0.size() );
        for( size_t i = 0; i < d0.size(); ++i ) id0[ i ] = d0[ i ] > 0 ? 1 / d0[ i ] : 0;
        int L = 0;
        while( ( N >> ( L + 1 ) ) >= 2 ) ++L;
        if( fr.empty() ) {
            for( int s : { 0, 1, -1 } ) {
                if( s && ! dband ) continue;
                Frame f;
                f.s = s; f.na = N; f.nb = s ? 2 * N : N; f.La = L; f.Lb = s ? L + 1 : L;
                f.fine.assign( f.na * f.nb, -1 );
                for( int a = 0; a < N; ++a )
                    for( int b = 0; b < N; ++b ) f.fine[ a * f.nb + cof( s, a, b ) ] = ( a + 1 ) * P + b + 1;
                f.g.resize( ( f.La + 1 ) * ( f.Lb + 1 ) ); f.Di.resize( f.g.size() );
                for( int la = 0; la <= f.La; ++la )
                    for( int lb = 0; lb <= f.Lb; ++lb ) f.g[ f.gi( la, lb ) ].assign( ( f.na >> la ) * ( f.nb >> lb ), 0 );
                fr.push_back( std::move( f ) );
            }
        }
        nkept = 0;
        const int PP = P * P;
        for( auto &f : fr ) {
            for( int la = 0; la <= f.La; ++la )
                for( int lb = 0; lb <= f.Lb; ++lb ) {
                    auto &D = f.Di[ f.gi( la, lb ) ];
                    D.clear();
                    if( ! kept( f, la, lb ) ) continue;
                    ++nkept;
                    const int ma = f.na >> la, mb = f.nb >> lb;
                    std::vector<T> Da( ma * mb, 0 ), Db( ma * mb, 0 );
                    auto add = [ & ]( int a, int b, int ja, int jb, T w ) {
                        const int fb = cof( f.s, a, b ), gb = cof( f.s, ja, jb );
                        const int ia = a >> la, ib = fb >> lb, ka = ja >> la, kb = gb >> lb;
                        if( ia != ka ) Da[ ia * mb + ib ] += w;
                        if( ib != kb ) Db[ ia * mb + ib ] += w;
                    };
                    for( int a = 0; a < N; ++a )
                        for( int b = 0; b < N; ++b ) {
                            const int i = ( a + 1 ) * P + b + 1;
                            for( int k = 0; k < 8; ++k ) {
                                const T w = ( *op.W )[ k * PP + i ];
                                if( w != 0 ) add( a, b, a + T1::off_x[ k ], b + T1::off_y[ k ], w );
                            }
                        }
                    for( size_t e = 0; e < op.ex_w.size(); ++e ) {
                        const int i = op.ex_ij[ e ].first, j = op.ex_ij[ e ].second;
                        add( i / P - 1, i % P - 1, j / P - 1, j % P - 1, op.ex_w[ e ] );
                    }
                    D.resize( ma * mb );
                    const T sa = c / T( 1 << la ), sb = c / T( 1 << lb );
                    for( int q = 0; q < ma * mb; ++q ) {
                        const T ea = sa * Da[ q ], eb = sb * Db[ q ], d = ea + eb;
                        D[ q ] = d > 0 ? balance( ea, eb, pw ) / d : 0;
                    }
                }
        }
        t_setup += now_ms() - t0;
    }
    // frame column of the fine cell ( a, b ); sheared: c = b - s a, shifted into [ 0, 2N - 1 )
    int cof( int s, int a, int b ) const { return s == 0 ? b : s > 0 ? b - a + N - 1 : b + a; }

    void apply( const T *r, T *z ) {
        double t0 = now_ms(), tr = 0, ts = 0;
        for( int a = 0; a < N; ++a ) {
            const int row = ( a + 1 ) * P + 1;
            #pragma omp simd
            for( int x = 0; x < N; ++x ) z[ row + x ] = r[ row + x ] * id0[ row + x ];
        }
        for( auto &f : fr ) {
            double u0 = now_ms();
            T *g00 = f.g[ 0 ].data();
            for( int q = 0; q < f.na * f.nb; ++q ) g00[ q ] = f.fine[ q ] >= 0 ? r[ f.fine[ q ] ] : T( 0 );
            for( int la = 1; la <= f.La; ++la ) TensorPrecond::restrict_a( f.g[ f.gi( la - 1, 0 ) ].data(), f.na >> ( la - 1 ), f.nb, f.g[ f.gi( la, 0 ) ].data() );
            for( int la = 0; la <= f.La; ++la )
                for( int lb = 1; lb <= f.Lb; ++lb ) TensorPrecond::restrict_b( f.g[ f.gi( la, lb - 1 ) ].data(), f.na >> la, f.nb >> ( lb - 1 ), f.g[ f.gi( la, lb ) ].data() );
            double u1 = now_ms();
            for( size_t q = 0; q < f.g.size(); ++q ) {
                T *v = f.g[ q ].data();
                const int n = int( f.g[ q ].size() );
                if( f.Di[ q ].empty() ) { std::fill( v, v + n, T( 0 ) ); continue; }
                const T *d = f.Di[ q ].data();
                #pragma omp simd
                for( int c2 = 0; c2 < n; ++c2 ) v[ c2 ] *= d[ c2 ];
            }
            double u2 = now_ms();
            for( int la = 0; la <= f.La; ++la )
                for( int lb = f.Lb; lb >= 1; --lb ) TensorPrecond::prolong_b_add( f.g[ f.gi( la, lb ) ].data(), f.na >> la, f.nb >> lb, f.g[ f.gi( la, lb - 1 ) ].data() );
            for( int la = f.La; la >= 1; --la ) TensorPrecond::prolong_a_add( f.g[ f.gi( la, 0 ) ].data(), f.na >> la, f.nb, f.g[ f.gi( la - 1, 0 ) ].data() );
            for( int q = 0; q < f.na * f.nb; ++q ) if( f.fine[ q ] >= 0 ) z[ f.fine[ q ] ] += g00[ q ];
            tr += u1 - u0; ts += u2 - u1;
        }
        const double t3 = now_ms();
        t_restrict += tr; t_scale += ts; t_prolong += t3 - t0 - tr - ts;
    }
};

// Diagonal blurs, fast version of the sheared frames of FramePrecond, added to a TensorPrecond.
// Frame s = +-1: rows a, columns c = b - s a + cst ( cst = N - 1 for s = 1 ), so moving along a at fixed
// c follows the diagonal ( 1, s ). Grids ( la, lc ) are stored ( N >> la ) x ( 2N >> lc ), but each row
// only touches its valid range [ lo, hi ) ( about half of the width; the rest stays 0 ). Kept
// subspaces: 1 <= la - lc <= dband ( elongated along the diagonal ); the grids with lc < la - dband
// are only intermediates of the restriction chains ( la, 0 ) -> ( la, 1 ) -> ...
// Scaling without weight planes: with the conductances ka( i ) = sum_k w_k | da_k |, kc( i ) =
// sum_k w_k | dc_k | ( frame offsets ), the weight crossing the a faces of a block is ~ sum_block ka / ha,
// and sum_block ~ the tent restriction ( its weights sum to ha hc ). So Ea = c R ka / ha^2,
// Ec = c R kc / hc^2, D^-1 = balance( Ea, Ec )^pw / ( Ea + Ec ): two restrictions per frame.
struct DiagFrames {
    struct Grid {
        int na = 0, W = 0;
        bool kept = false;
        std::vector<T> v, Di;
        std::vector<int> lo, hi;
    };
    struct Frame {
        int s, L;
        std::vector<std::vector<Grid>> g; // g[ la ][ lc ], lc < max( la, 1 )
    };
    int N, P, L, dband = 2;
    T c = T( 2 ) / 3, pw = T( 0.5 );
    std::vector<Frame> fr;
    int nkept = 0;
    double t_setup = 0, t_apply = 0;

    int lo0( int s, int a ) const { return s > 0 ? N - 1 - a : a; }

    void layout() {
        L = 0;
        while( ( N >> ( L + 1 ) ) >= 2 ) ++L;
        fr.clear();
        nkept = 0;
        for( int s : { 1, -1 } ) {
            Frame f;
            f.s = s; f.L = L;
            f.g.resize( L + 1 );
            for( int la = 0; la <= L; ++la ) {
                const int nc = std::max( la, 1 );
                f.g[ la ].resize( nc );
                for( int lc = 0; lc < nc; ++lc ) {
                    Grid &G = f.g[ la ][ lc ];
                    G.na = N >> la; G.W = ( 2 * N ) >> lc;
                    G.kept = la - lc >= 1 && la - lc <= dband;
                    nkept += G.kept;
                    G.v.assign( G.na * G.W, 0 );
                    G.lo.resize( G.na ); G.hi.resize( G.na );
                    if( la == 0 ) {
                        for( int a = 0; a < N; ++a ) { G.lo[ a ] = lo0( s, a ); G.hi[ a ] = G.lo[ a ] + N; }
                    } else if( lc == 0 ) {
                        const Grid &F = f.g[ la - 1 ][ 0 ];
                        for( int A = 0; A < G.na; ++A ) {
                            int l = 1 << 30, h = 0;
                            for( int j = std::max( 0, 2 * A - 1 ); j <= std::min( F.na - 1, 2 * A + 2 ); ++j ) { l = std::min( l, F.lo[ j ] ); h = std::max( h, F.hi[ j ] ); }
                            G.lo[ A ] = l; G.hi[ A ] = h;
                        }
                    } else {
                        const Grid &F = f.g[ la ][ lc - 1 ];
                        for( int A = 0; A < G.na; ++A ) {
                            G.lo[ A ] = std::max( 0, ( F.lo[ A ] - 1 ) >> 1 );
                            G.hi[ A ] = std::min( G.W, ( F.hi[ A ] >> 1 ) + 1 );
                        }
                    }
                    if( G.kept ) G.Di.assign( G.na * G.W, 0 );
                }
            }
            fr.push_back( std::move( f ) );
        }
    }

    // ---- transfers on valid ranges ( values outside are 0 and stay 0 ) ----
    static void restrict_a( const Grid &F, Grid &G ) {
        const int m = G.na, W = G.W;
        for( int A = 0; A < m; ++A ) {
            const T w0 = A == 0 ? T( 1 ) : T( 0.75 ), w1 = A == m - 1 ? T( 1 ) : T( 0.75 ), wm = A >= 1 ? T( 0.25 ) : T( 0 ), wp = A <= m - 2 ? T( 0.25 ) : T( 0 );
            const T *r0 = F.v.data() + 2 * A * W, *r1 = r0 + W, *rm = A >= 1 ? r0 - W : r0, *rp = A <= m - 2 ? r1 + W : r1;
            T *cr = G.v.data() + A * W;
            const int l = G.lo[ A ], h = G.hi[ A ];
            #pragma omp simd
            for( int b = l; b < h; ++b ) cr[ b ] = w0 * r0[ b ] + w1 * r1[ b ] + wm * rm[ b ] + wp * rp[ b ];
        }
    }
    static void prolong_a_add( const Grid &G, Grid &F ) {
        const int m = G.na, W = G.W;
        for( int A = 0; A < m; ++A ) {
            const T *c0 = G.v.data() + A * W, *cm = A >= 1 ? c0 - W : c0, *cp = A <= m - 2 ? c0 + W : c0;
            const T e0 = A == 0 ? T( 1 ) : T( 0.75 ), em = A == 0 ? T( 0 ) : T( 0.25 ), e1 = A == m - 1 ? T( 1 ) : T( 0.75 ), ep = A == m - 1 ? T( 0 ) : T( 0.25 );
            T *t0 = F.v.data() + 2 * A * W, *t1 = t0 + W;
            {
                const int l = F.lo[ 2 * A ], h = F.hi[ 2 * A ];
                #pragma omp simd
                for( int b = l; b < h; ++b ) t0[ b ] += e0 * c0[ b ] + em * cm[ b ];
            }
            {
                const int l = F.lo[ 2 * A + 1 ], h = F.hi[ 2 * A + 1 ];
                #pragma omp simd
                for( int b = l; b < h; ++b ) t1[ b ] += e1 * c0[ b ] + ep * cp[ b ];
            }
        }
    }
    // along c, zero extension: t[ B ] = ( f[ 2B-1 ] + 3 f[ 2B ] + 3 f[ 2B+1 ] + f[ 2B+2 ] ) / 4
    static void restrict_c( const Grid &F, Grid &G ) {
        const int Wf = F.W, Wc = G.W;
        for( int A = 0; A < G.na; ++A ) {
            const T *f = F.v.data() + A * Wf;
            T *t = G.v.data() + A * Wc;
            const int l = G.lo[ A ], h = G.hi[ A ];
            auto get = [ & ]( int i ) { return i >= 0 && i < Wf ? f[ i ] : T( 0 ); };
            const int il = std::max( l, 1 ), ih = std::min( h, Wc - 1 );
            for( int B = l; B < std::min( il, h ); ++B ) t[ B ] = T( 0.25 ) * ( get( 2 * B - 1 ) + get( 2 * B + 2 ) ) + T( 0.75 ) * ( get( 2 * B ) + get( 2 * B + 1 ) );
            #pragma omp simd
            for( int B = il; B < ih; ++B ) t[ B ] = T( 0.25 ) * ( f[ 2 * B - 1 ] + f[ 2 * B + 2 ] ) + T( 0.75 ) * ( f[ 2 * B ] + f[ 2 * B + 1 ] );
            for( int B = std::max( ih, l ); B < h; ++B ) t[ B ] = T( 0.25 ) * ( get( 2 * B - 1 ) + get( 2 * B + 2 ) ) + T( 0.75 ) * ( get( 2 * B ) + get( 2 * B + 1 ) );
        }
    }
    // transpose: f[ 2B ] += 3/4 t[ B ] + 1/4 t[ B-1 ], f[ 2B+1 ] += 3/4 t[ B ] + 1/4 t[ B+1 ], on the valid range of f
    static void prolong_c_add( const Grid &G, Grid &F ) {
        const int Wf = F.W, Wc = G.W;
        for( int A = 0; A < G.na; ++A ) {
            const T *t = G.v.data() + A * Wc;
            T *f = F.v.data() + A * Wf;
            const int l = F.lo[ A ], h = F.hi[ A ];
            auto get = [ & ]( int B ) { return B >= 0 && B < Wc ? t[ B ] : T( 0 ); };
            auto one = [ & ]( int i ) { const int B = i >> 1; f[ i ] += T( 0.75 ) * get( B ) + T( 0.25 ) * get( i & 1 ? B + 1 : B - 1 ); };
            // vector part on whole pairs [ 2 B0, 2 B1 ) with 1 <= B and B + 1 < Wc
            const int B0 = std::max( ( l + 1 ) >> 1, 1 ), B1 = std::min( h >> 1, Wc - 1 );
            if( B0 >= B1 ) { for( int i = l; i < h; ++i ) one( i ); continue; }
            for( int i = l; i < 2 * B0; ++i ) one( i );
            #pragma omp simd
            for( int B = B0; B < B1; ++B ) {
                f[ 2 * B ] += T( 0.75 ) * t[ B ] + T( 0.25 ) * t[ B - 1 ];
                f[ 2 * B + 1 ] += T( 0.75 ) * t[ B ] + T( 0.25 ) * t[ B + 1 ];
            }
            for( int i = 2 * B1; i < h; ++i ) one( i );
        }
    }
    static void zero( Grid &G ) {
        for( int A = 0; A < G.na; ++A ) std::fill( G.v.begin() + A * G.W + G.lo[ A ], G.v.begin() + A * G.W + G.hi[ A ], T( 0 ) );
    }

    // restriction of the base grid ( already filled ) through all chains
    void restrict_all( Frame &f ) {
        for( int la = 1; la <= L; ++la ) restrict_a( f.g[ la - 1 ][ 0 ], f.g[ la ][ 0 ] );
        for( int la = 2; la <= L; ++la )
            for( int lc = 1; lc < la; ++lc ) restrict_c( f.g[ la ][ lc - 1 ], f.g[ la ][ lc ] );
    }
    void fill_base( Frame &f, const T *src, int stride, int off ) {
        Grid &B0 = f.g[ 0 ][ 0 ];
        for( int a = 0; a < N; ++a ) std::memcpy( B0.v.data() + a * B0.W + B0.lo[ a ], src + a * stride + off, N * sizeof( T ) );
    }

    void build( const Operator &op ) {
        const double t0 = now_ms();
        if( N != op.N || fr.empty() ) { N = op.N; layout(); }
        P = op.P;
        const int PP = P * P;
        std::vector<T> ka( N * N ), kc( N * N );
        for( auto &f : fr ) {
            // conductances in frame offsets: da = oa, dc = ob - s oa
            int adc[ 8 ];
            for( int k = 0; k < 8; ++k ) adc[ k ] = std::abs( T1::off_y[ k ] - f.s * T1::off_x[ k ] );
            for( int a = 0; a < N; ++a ) {
                T *ra = &ka[ a * N ], *rc = &kc[ a * N ];
                for( int b = 0; b < N; ++b ) { ra[ b ] = 0; rc[ b ] = 0; }
                for( int k = 0; k < 8; ++k ) {
                    const T *w = &( *op.W )[ k * PP + ( a + 1 ) * P + 1 ];
                    const T fa = std::abs( T1::off_x[ k ] ), fc = adc[ k ];
                    #pragma omp simd
                    for( int b = 0; b < N; ++b ) { ra[ b ] += fa * w[ b ]; rc[ b ] += fc * w[ b ]; }
                }
            }
            for( size_t e = 0; e < op.ex_w.size(); ++e ) {
                const int i = op.ex_ij[ e ].first, j = op.ex_ij[ e ].second;
                const int ia = i / P - 1, ib = i % P - 1, da = j / P - 1 - ia, db = j % P - 1 - ib;
                ka[ ia * N + ib ] += std::abs( da ) * op.ex_w[ e ];
                kc[ ia * N + ib ] += std::abs( db - f.s * da ) * op.ex_w[ e ];
            }
            // Ea into Di, then D^-1
            for( int pass = 0; pass < 2; ++pass ) {
                fill_base( f, pass ? kc.data() : ka.data(), N, 0 );
                restrict_all( f );
                for( int la = 1; la <= L; ++la )
                    for( int lc = 0; lc < la; ++lc ) {
                        Grid &G = f.g[ la ][ lc ];
                        if( ! G.kept ) continue;
                        const T sa = c / T( 1 << la ) / T( 1 << la ), sc = c / T( 1 << lc ) / T( 1 << lc );
                        for( int A = 0; A < G.na; ++A ) {
                            T *d = G.Di.data() + A * G.W;
                            const T *v = G.v.data() + A * G.W;
                            if( ! pass ) {
                                for( int q = G.lo[ A ]; q < G.hi[ A ]; ++q ) d[ q ] = sa * v[ q ];
                            } else {
                                for( int q = G.lo[ A ]; q < G.hi[ A ]; ++q ) {
                                    const T ea = d[ q ], ec = sc * v[ q ], s2 = ea + ec;
                                    d[ q ] = s2 > 0 ? TensorPrecond::balance( ea, ec, pw ) / s2 : 0;
                                }
                            }
                        }
                    }
            }
        }
        t_setup += now_ms() - t0;
    }

    // z += sum over the kept sheared subspaces
    void apply_add( const T *r, T *z ) {
        const double t0 = now_ms();
        for( auto &f : fr ) {
            fill_base( f, r + P + 1, P, 0 );
            restrict_all( f );
            for( int la = 0; la <= L; ++la )
                for( size_t lc = 0; lc < f.g[ la ].size(); ++lc ) {
                    Grid &G = f.g[ la ][ lc ];
                    if( ! G.kept ) { zero( G ); continue; }
                    for( int A = 0; A < G.na; ++A ) {
                        T *v = G.v.data() + A * G.W;
                        const T *d = G.Di.data() + A * G.W;
                        const int l = G.lo[ A ], h = G.hi[ A ];
                        #pragma omp simd
                        for( int q = l; q < h; ++q ) v[ q ] *= d[ q ];
                    }
                }
            for( int la = L; la >= 1; --la ) {
                for( int lc = la - 1; lc >= 1; --lc ) prolong_c_add( f.g[ la ][ lc ], f.g[ la ][ lc - 1 ] );
                prolong_a_add( f.g[ la ][ 0 ], f.g[ la - 1 ][ 0 ] );
            }
            const Grid &B0 = f.g[ 0 ][ 0 ];
            for( int a = 0; a < N; ++a ) {
                const T *src = B0.v.data() + a * B0.W + B0.lo[ a ];
                T *dst = z + ( a + 1 ) * P + 1;
                #pragma omp simd
                for( int b = 0; b < N; ++b ) dst[ b ] += src[ b ];
            }
        }
        t_apply += now_ms() - t0;
    }
};

// axis tensor blur + fast diagonal frames
struct TensorDiag {
    TensorPrecond ax;
    DiagFrames dg;
    void build( const Operator &op ) { ax.build( op ); dg.build( op ); }
    void apply( const T *r, T *z ) { ax.apply( r, z ); dg.apply_add( r, z ); }
};

// dband = 1: the kept subspaces ( k + 1, k ) form one chain, K_k = R_c R_a K_{k-1}, through the
// intermediates I_k = ( k + 1, k - 1 ); K_0 = ( 1, 0 ) is restricted straight from the fine vector
// ( shifted rows, no base copy ) and prolongated straight into z.
struct DiagChain {
    using Grid = DiagFrames::Grid;
    struct Frame { int s; std::vector<Grid> K, I; };   // I[ 0 ] unused
    int N, P, L;
    T c = T( 2 ) / 3, pw = T( 0.5 );
    std::vector<Frame> fr;
    int nkept = 0;
    double t_setup = 0, t_apply = 0;

    int lo0( int s, int a ) const { return s > 0 ? N - 1 - a : a; }
    static void rows_union( const Grid &F, Grid &G ) {
        for( int A = 0; A < G.na; ++A ) {
            int l = 1 << 30, h = 0;
            for( int j = std::max( 0, 2 * A - 1 ); j <= std::min( F.na - 1, 2 * A + 2 ); ++j ) { l = std::min( l, F.lo[ j ] ); h = std::max( h, F.hi[ j ] ); }
            G.lo[ A ] = l; G.hi[ A ] = h;
        }
    }
    void layout() {
        L = 0;
        while( ( N >> ( L + 1 ) ) >= 2 ) ++L;
        fr.clear(); nkept = 0;
        for( int s : { 1, -1 } ) {
            Frame f; f.s = s;
            f.K.resize( L ); f.I.resize( L );
            for( int k = 0; k < L; ++k ) {
                Grid &K = f.K[ k ];
                K.na = N >> ( k + 1 ); K.W = ( 2 * N ) >> k; K.lo.resize( K.na ); K.hi.resize( K.na );
                if( k == 0 ) {
                    for( int A = 0; A < K.na; ++A ) {
                        int l = 1 << 30, h = 0;
                        for( int j = std::max( 0, 2 * A - 1 ); j <= std::min( N - 1, 2 * A + 2 ); ++j ) { l = std::min( l, lo0( s, j ) ); h = std::max( h, lo0( s, j ) + N ); }
                        K.lo[ A ] = l; K.hi[ A ] = h;
                    }
                } else {
                    Grid &I = f.I[ k ];
                    I.na = K.na; I.W = ( 2 * N ) >> ( k - 1 ); I.lo.resize( I.na ); I.hi.resize( I.na );
                    rows_union( f.K[ k - 1 ], I );
                    for( int A = 0; A < K.na; ++A ) { K.lo[ A ] = std::max( 0, ( I.lo[ A ] - 1 ) >> 1 ); K.hi[ A ] = std::min( K.W, ( I.hi[ A ] >> 1 ) + 1 ); }
                    I.v.assign( I.na * I.W, 0 );
                }
                K.v.assign( K.na * K.W, 0 ); K.Di.assign( K.na * K.W, 0 );
                ++nkept;
            }
            fr.push_back( std::move( f ) );
        }
    }
    // K_0 from the fine rows ( row j at src + j * stride, N values, column lo0( j ) in the frame )
    void restrict0( const Frame &f, const T *src, int stride, Grid &K ) const {
        const int m = K.na;
        for( int A = 0; A < m; ++A ) {
            T *cr = K.v.data() + A * K.W;
            std::fill( cr + K.lo[ A ], cr + K.hi[ A ], T( 0 ) );
            const T w0 = A == 0 ? T( 1 ) : T( 0.75 ), w1 = A == m - 1 ? T( 1 ) : T( 0.75 );
            const int js[ 4 ] = { 2 * A - 1, 2 * A, 2 * A + 1, 2 * A + 2 };
            const T ws[ 4 ] = { A >= 1 ? T( 0.25 ) : T( 0 ), w0, w1, A <= m - 2 ? T( 0.25 ) : T( 0 ) };
            for( int q = 0; q < 4; ++q ) {
                if( ws[ q ] == 0 ) continue;
                const T w = ws[ q ], *sr = src + js[ q ] * stride;
                T *d = cr + lo0( f.s, js[ q ] );
                #pragma omp simd
                for( int b = 0; b < N; ++b ) d[ b ] += w * sr[ b ];
            }
        }
    }
    // transpose of restrict0, added into the fine rows
    void prolong0_add( const Frame &f, const Grid &K, T *dst, int stride ) const {
        const int m = K.na;
        for( int A = 0; A < m; ++A ) {
            const T *c0 = K.v.data() + A * K.W, *cm = A >= 1 ? c0 - K.W : c0, *cp = A <= m - 2 ? c0 + K.W : c0;
            const T e0 = A == 0 ? T( 1 ) : T( 0.75 ), em = A == 0 ? T( 0 ) : T( 0.25 ), e1 = A == m - 1 ? T( 1 ) : T( 0.75 ), ep = A == m - 1 ? T( 0 ) : T( 0.25 );
            for( int p = 0; p < 2; ++p ) {
                const int j = 2 * A + p, o = lo0( f.s, j );
                const T e = p ? e1 : e0, en = p ? ep : em;
                const T *cn = p ? cp : cm;
                T *d = dst + j * stride;
                #pragma omp simd
                for( int b = 0; b < N; ++b ) d[ b ] += e * c0[ o + b ] + en * cn[ o + b ];
            }
        }
    }
    void restrict_all( Frame &f, const T *src, int stride ) {
        restrict0( f, src, stride, f.K[ 0 ] );
        for( int k = 1; k < L; ++k ) { DiagFrames::restrict_a( f.K[ k - 1 ], f.I[ k ] ); DiagFrames::restrict_c( f.I[ k ], f.K[ k ] ); }
    }

    void build( const Operator &op ) {
        const double t0 = now_ms();
        if( N != op.N || fr.empty() ) { N = op.N; layout(); }
        P = op.P;
        const int PP = P * P;
        std::vector<T> ka( N * N, 0 ), kc( N * N );
        for( int a = 0; a < N; ++a )
            for( int k = 0; k < 8; ++k ) {
                if( ! T1::off_x[ k ] ) continue;
                const T *w = &( *op.W )[ k * PP + ( a + 1 ) * P + 1 ];
                T *ra = &ka[ a * N ];
                #pragma omp simd
                for( int b = 0; b < N; ++b ) ra[ b ] += w[ b ];
            }
        for( size_t e = 0; e < op.ex_w.size(); ++e ) {
            const int i = op.ex_ij[ e ].first, j = op.ex_ij[ e ].second;
            ka[ ( i / P - 1 ) * N + i % P - 1 ] += std::abs( j / P - i / P ) * op.ex_w[ e ];
        }
        for( auto &f : fr ) {
            std::fill( kc.begin(), kc.end(), T( 0 ) );
            for( int a = 0; a < N; ++a )
                for( int k = 0; k < 8; ++k ) {
                    const T fc = std::abs( T1::off_y[ k ] - f.s * T1::off_x[ k ] );
                    if( fc == 0 ) continue;
                    const T *w = &( *op.W )[ k * PP + ( a + 1 ) * P + 1 ];
                    T *rc = &kc[ a * N ];
                    #pragma omp simd
                    for( int b = 0; b < N; ++b ) rc[ b ] += fc * w[ b ];
                }
            for( size_t e = 0; e < op.ex_w.size(); ++e ) {
                const int i = op.ex_ij[ e ].first, j = op.ex_ij[ e ].second;
                const int da = j / P - i / P, db = j % P - i % P;
                kc[ ( i / P - 1 ) * N + i % P - 1 ] += std::abs( db - f.s * da ) * op.ex_w[ e ];
            }
            for( int pass = 0; pass < 2; ++pass ) {
                restrict_all( f, pass ? kc.data() : ka.data(), N );
                for( int k = 0; k < L; ++k ) {
                    Grid &G = f.K[ k ];
                    const T sa = c / T( 1 << ( k + 1 ) ) / T( 1 << ( k + 1 ) ), sc = c / T( 1 << k ) / T( 1 << k );
                    for( int A = 0; A < G.na; ++A ) {
                        T *d = G.Di.data() + A * G.W;
                        const T *v = G.v.data() + A * G.W;
                        for( int q = G.lo[ A ]; q < G.hi[ A ]; ++q ) {
                            if( ! pass ) { d[ q ] = sa * v[ q ]; continue; }
                            const T ea = d[ q ], ec = sc * v[ q ], s2 = ea + ec;
                            d[ q ] = s2 > 0 ? TensorPrecond::balance( ea, ec, pw ) / s2 : 0;
                        }
                    }
                }
            }
        }
        t_setup += now_ms() - t0;
    }

    void apply_add( const T *r, T *z ) {
        const double t0 = now_ms();
        for( auto &f : fr ) {
            restrict_all( f, r + P + 1, P );
            for( int k = 0; k < L; ++k ) {
                Grid &G = f.K[ k ];
                for( int A = 0; A < G.na; ++A ) {
                    T *v = G.v.data() + A * G.W;
                    const T *d = G.Di.data() + A * G.W;
                    const int l = G.lo[ A ], h = G.hi[ A ];
                    #pragma omp simd
                    for( int q = l; q < h; ++q ) v[ q ] *= d[ q ];
                }
                if( k ) DiagFrames::zero( f.I[ k ] );
            }
            for( int k = L - 1; k >= 1; --k ) { DiagFrames::prolong_c_add( f.K[ k ], f.I[ k ] ); DiagFrames::prolong_a_add( f.I[ k ], f.K[ k - 1 ] ); }
            prolong0_add( f, f.K[ 0 ], z + P + 1, P );
        }
        t_apply += now_ms() - t0;
    }
};

struct TensorChain {
    TensorPrecond ax;
    DiagChain dg;
    void build( const Operator &op ) { ax.build( op ); dg.build( op ); }
    void apply( const T *r, T *z ) { ax.apply( r, z ); dg.apply_add( r, z ); }
};

// Aggregation AMG on the weighted graph laplacian ( in the spirit of AGMG, Notay ): each level pairs
// every node with its strongest still-free neighbor, twice ( aggregates of <= 4 nodes that follow
// the strong couplings, so the anisotropic tail cells are coarsened along their strong direction );
// the coarse operator sums the weights between aggregates ( again a graph laplacian, constants kept
// exactly ). Symmetric Gauss-Seidel smoothing, V-cycle or K-cycle ( 2 flexible CG steps per coarse
// level ), dense Cholesky on the coarsest level.
struct Amg {
    struct Level {
        int n;
        std::vector<int> ptr, col;           // off-diagonal pattern
        std::vector<T> w, d, id;             // weights ( > 0, A_ij = -w ), diagonal and its inverse
        std::vector<int> agg;                // node -> aggregate of the next level
        std::vector<T> r, z, t, v1, v2, w1, w2, x; // work vectors
    };
    std::vector<Level> lv;
    // level 0 on the grid structure: padded vectors, weights read from the operator's planes, the
    // rows of the extras ( cells off ring 1 ) in a small CSR
    const Operator *op0 = nullptr;
    int N0 = 0, P0 = 0;
    std::vector<T> id0p, t0v;                    // padded
    std::vector<int> exptr, excol;               // per interior row, padded column indices
    std::vector<T> exw;
    std::vector<unsigned char> hasex;            // padded
    std::vector<T> chol;                     // coarsest level: dense Cholesky of A + e e^T ( d_mean / n )
    bool kcycle = true;
    int kmin = 1;                            // K-cycle on the levels >= kmin, V below
    int max_coarse = 100;
    double beta_strong = 0.25;

    // one pass of pairwise matching: returns the number of aggregates
    static int pairwise( const Level &L, std::vector<int> &agg, double beta ) {
        agg.assign( L.n, -1 );
        int nc = 0;
        for( int i = 0; i < L.n; ++i ) {
            if( agg[ i ] >= 0 ) continue;
            T smax = 0;
            for( int p = L.ptr[ i ]; p < L.ptr[ i + 1 ]; ++p ) smax = std::max( smax, L.w[ p ] );
            int best = -1;
            T bw = beta * smax;
            for( int p = L.ptr[ i ]; p < L.ptr[ i + 1 ]; ++p ) {
                const int j = L.col[ p ];
                if( agg[ j ] < 0 && j != i && L.w[ p ] >= bw && ( best < 0 || L.w[ p ] > bw ) ) { best = j; bw = L.w[ p ]; }
            }
            agg[ i ] = nc;
            if( best >= 0 ) agg[ best ] = nc;
            ++nc;
        }
        return nc;
    }
    // Galerkin operator of a piecewise-constant prolongation: sum of the weights between aggregates
    static void coarsen( const Level &F, const std::vector<int> &agg, int nc, Level &C ) {
        C.n = nc;
        std::vector<int> cnt( nc + 1, 0 ), mem( F.n );
        for( int i = 0; i < F.n; ++i ) ++cnt[ agg[ i ] + 1 ];
        for( int c = 0; c < nc; ++c ) cnt[ c + 1 ] += cnt[ c ];
        std::vector<int> pos( cnt.begin(), cnt.end() - 1 );
        for( int i = 0; i < F.n; ++i ) mem[ pos[ agg[ i ] ]++ ] = i;
        std::vector<int> mark( nc, -1 );
        C.ptr.assign( 1, 0 ); C.col.clear(); C.w.clear(); C.d.assign( nc, 0 );
        C.ptr.reserve( nc + 1 ); C.col.reserve( F.col.size() ); C.w.reserve( F.col.size() );
        for( int c = 0; c < nc; ++c ) {
            const int beg = int( C.col.size() );
            for( int m = cnt[ c ]; m < cnt[ c + 1 ]; ++m ) {
                const int i = mem[ m ];
                C.d[ c ] += F.d[ i ];
                for( int p = F.ptr[ i ]; p < F.ptr[ i + 1 ]; ++p ) {
                    const int J = agg[ F.col[ p ] ];
                    if( J == c ) { C.d[ c ] -= F.w[ p ]; continue; }
                    if( mark[ J ] < beg ) { mark[ J ] = int( C.col.size() ); C.col.push_back( J ); C.w.push_back( 0 ); }
                    C.w[ mark[ J ] ] += F.w[ p ];
                }
            }
            C.ptr.push_back( int( C.col.size() ) );
        }
    }

    template<class Op>
    void build( const Op &op ) {
        const int N = op.N, P = op.P, PP = P * P;
        op0 = &op; N0 = N; P0 = P;
        {
            std::vector<T> d;
            op.diag( d );
            id0p.assign( PP, 0 );
            for( int i = 0; i < PP; ++i ) id0p[ i ] = d[ i ] > 0 ? 1 / d[ i ] : 0;
            t0v.assign( PP, 0 );
            hasex.assign( PP, 0 );
            std::vector<int> cnt( PP + 1, 0 );
            for( size_t e = 0; e < op.ex_w.size(); ++e ) ++cnt[ op.ex_ij[ e ].first + 1 ];
            for( int i = 0; i < PP; ++i ) cnt[ i + 1 ] += cnt[ i ];
            exptr = cnt;
            excol.assign( op.ex_w.size(), 0 ); exw.assign( op.ex_w.size(), 0 );
            for( size_t e = 0; e < op.ex_w.size(); ++e ) {
                const int i = op.ex_ij[ e ].first;
                excol[ cnt[ i ] ] = op.ex_ij[ e ].second; exw[ cnt[ i ]++ ] = op.ex_w[ e ];
                hasex[ i ] = 1;
            }
        }
        double ts = now_ms();
        if( lv.empty() ) lv.resize( 1 );
        lv.resize( 1 );
        Level &L = lv[ 0 ];
        L.n = N * N;
        // level-0 graph ( CSR, unpadded indices ) from the planes and the extras' rows
        L.ptr.resize( L.n + 1 ); L.col.resize( 8 * L.n + op.ex_w.size() ); L.w.resize( L.col.size() ); L.d.assign( L.n, 0 );
        {
            const T *W = op.W->data();
            int q = 0;
            L.ptr[ 0 ] = 0;
            for( int a = 0; a < N; ++a ) for( int b = 0; b < N; ++b ) {
                const int i = a * N + b, ip = ( a + 1 ) * P + b + 1;
                T d = 0;
                for( int k = 0; k < 8; ++k ) {
                    const T w = W[ k * PP + ip ];
                    L.col[ q ] = i + T1::off_x[ k ] * N + T1::off_y[ k ]; L.w[ q ] = w; q += w != 0; d += w;
                }
                for( int p = exptr[ ip ]; p < exptr[ ip + 1 ]; ++p ) {
                    const int j = excol[ p ];
                    L.col[ q ] = ( j / P - 1 ) * N + j % P - 1; L.w[ q ] = exw[ p ]; d += exw[ p ]; ++q;
                }
                L.d[ i ] = d;
                L.ptr[ i + 1 ] = q;
            }
        }
        double tl0 = now_ms(), t_pw = 0, t_co = 0;
        while( lv.back().n > max_coarse ) {
            Level &F = lv.back();
            std::vector<int> a1, a2;
            Level mid;
            double t0 = now_ms();
            const int n1 = pairwise( F, a1, beta_strong );
            double t1 = now_ms();
            coarsen( F, a1, n1, mid );
            double t2 = now_ms();
            const int n2 = pairwise( mid, a2, beta_strong );
            double t3 = now_ms();
            F.agg.resize( F.n );
            for( int i = 0; i < F.n; ++i ) F.agg[ i ] = a2[ a1[ i ] ];
            Level C;
            coarsen( mid, a2, n2, C ); // = the Galerkin operator of the composed aggregation
            t_pw += ( t1 - t0 ) + ( t3 - t2 ); t_co += ( t2 - t1 ) + ( now_ms() - t3 );
            if( n2 > 0.9 * F.n ) break; // no more coarsening
            lv.push_back( std::move( C ) );
        }
        for( auto &l : lv ) {
            for( auto *v : { &l.r, &l.z, &l.t, &l.v1, &l.v2, &l.w1, &l.w2, &l.x } ) v->assign( l.n, 0 );
            l.id.resize( l.n );
            for( int i = 0; i < l.n; ++i ) l.id[ i ] = l.d[ i ] > 0 ? 1 / l.d[ i ] : 0;
        }
        double tv = now_ms();
        // coarsest: dense Cholesky of A + ( d_mean / n ) e e^T ( nonsingular, same action on mean-zero vectors )
        Level &C = lv.back();
        const int n = C.n;
        T dm = 0;
        for( auto v : C.d ) dm += v;
        dm /= n;
        chol.assign( size_t( n ) * n, dm / n );
        for( int i = 0; i < n; ++i ) {
            chol[ size_t( i ) * n + i ] += C.d[ i ];
            for( int p = C.ptr[ i ]; p < C.ptr[ i + 1 ]; ++p ) chol[ size_t( i ) * n + C.col[ p ] ] -= C.w[ p ];
        }
        for( int j = 0; j < n; ++j ) {
            T s = chol[ size_t( j ) * n + j ];
            for( int k = 0; k < j; ++k ) s -= chol[ size_t( j ) * n + k ] * chol[ size_t( j ) * n + k ];
            const T l = std::sqrt( std::max( s, T( 1e-300 ) ) );
            chol[ size_t( j ) * n + j ] = l;
            for( int i = j + 1; i < n; ++i ) {
                T t = chol[ size_t( i ) * n + j ];
                for( int k = 0; k < j; ++k ) t -= chol[ size_t( i ) * n + k ] * chol[ size_t( j ) * n + k ];
                chol[ size_t( i ) * n + j ] = t / l;
            }
        }
        t_setup[ 0 ] += tl0 - ts; t_setup[ 1 ] += t_pw; t_setup[ 2 ] += t_co; t_setup[ 3 ] += tv - tl0 - t_pw - t_co; t_setup[ 4 ] += now_ms() - tv;
    }
    double t_setup[ 5 ] = {}; // level-0 graph, pairwise, coarsen, vectors, coarsest

    // sum_p w_p v_col_p over a row, 4 accumulators ( no reassociation without fast-math )
    static T row_sum( const Level &L, int i, const T *v ) {
        T s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        int p = L.ptr[ i ];
        const int e = L.ptr[ i + 1 ];
        for( ; p + 4 <= e; p += 4 ) {
            s0 += L.w[ p ] * v[ L.col[ p ] ]; s1 += L.w[ p + 1 ] * v[ L.col[ p + 1 ] ];
            s2 += L.w[ p + 2 ] * v[ L.col[ p + 2 ] ]; s3 += L.w[ p + 3 ] * v[ L.col[ p + 3 ] ];
        }
        for( ; p < e; ++p ) s0 += L.w[ p ] * v[ L.col[ p ] ];
        return ( s0 + s1 ) + ( s2 + s3 );
    }
    static void matvec( const Level &L, const T *v, T *y ) {
        for( int i = 0; i < L.n; ++i ) y[ i ] = L.d[ i ] * v[ i ] - row_sum( L, i, v );
    }
    static void gs( const Level &L, const T *r, T *z, bool forward ) {
        for( int c = 0; c < L.n; ++c ) {
            const int i = forward ? c : L.n - 1 - c;
            z[ i ] = ( r[ i ] + row_sum( L, i, z ) ) * L.id[ i ];
        }
    }
    void coarsest( const T *r, T *z ) const {
        const int n = lv.back().n;
        for( int i = 0; i < n; ++i ) {
            T s = r[ i ];
            for( int k = 0; k < i; ++k ) s -= chol[ size_t( i ) * n + k ] * z[ k ];
            z[ i ] = s / chol[ size_t( i ) * n + i ];
        }
        for( int i = n - 1; i >= 0; --i ) {
            T s = z[ i ];
            for( int k = i + 1; k < n; ++k ) s -= chol[ size_t( k ) * n + i ] * z[ k ];
            z[ i ] = s / chol[ size_t( i ) * n + i ];
        }
    }
    static T dot( int n, const T *u, const T *v ) { T s = 0; for( int i = 0; i < n; ++i ) s += u[ i ] * v[ i ]; return s; }

    // z = B_l r ( symmetric for the V-cycle; the K-cycle makes it nonlinear -> flexible outer CG )
    double t_lvl[ 32 ] = {}, t_coarsest = 0;
    void cycle( int l, const T *r, T *z ) {
        Level &L = lv[ l ];
        if( l + 1 == int( lv.size() ) ) { double t0 = now_ms(); coarsest( r, z ); t_coarsest += now_ms() - t0; return; }
        double t0 = now_ms();
        Level &C = lv[ l + 1 ];
        std::fill( z, z + L.n, 0 );
        gs( L, r, z, true );
        matvec( L, z, L.t.data() );
        std::fill( C.r.begin(), C.r.end(), 0 );
        for( int i = 0; i < L.n; ++i ) C.r[ L.agg[ i ] ] += r[ i ] - L.t[ i ];
        t_lvl[ l ] += now_ms() - t0;
        coarse_solve( l + 1 );
        t0 = now_ms();
        for( int i = 0; i < L.n; ++i ) z[ i ] += C.x[ L.agg[ i ] ];
        // post-smoothing: backward Gauss-Seidel, in place ( no residual needed )
        gs( L, r, z, false );
        t_lvl[ l ] += now_ms() - t0;
    }
    // C.x ~ A_c^-1 C.r: one cycle ( V ), or 2 flexible CG steps preconditioned by the cycle ( K )
    void coarse_solve( int l ) {
        Level &C = lv[ l ];
        const int n = C.n;
        if( ! kcycle || l < kmin || l + 1 == int( lv.size() ) ) { cycle( l, C.r.data(), C.x.data() ); return; }
        cycle( l, C.r.data(), C.v1.data() );
        matvec( C, C.v1.data(), C.w1.data() );
        const T rho1 = dot( n, C.v1.data(), C.w1.data() ), a1 = dot( n, C.v1.data(), C.r.data() );
        if( rho1 <= 0 ) { std::fill( C.x.begin(), C.x.end(), 0 ); return; }
        for( int i = 0; i < n; ++i ) C.t[ i ] = C.r[ i ] - a1 / rho1 * C.w1[ i ];
        const T nr = std::sqrt( dot( n, C.r.data(), C.r.data() ) ), nt = std::sqrt( dot( n, C.t.data(), C.t.data() ) );
        for( int i = 0; i < n; ++i ) C.x[ i ] = a1 / rho1 * C.v1[ i ];
        if( nt <= 0.25 * nr ) return;
        // second step ( v2 orthogonalized against v1 in the A inner product )
        std::vector<T> &rr = C.z; // free at this level: cycle( l ) writes its result in its argument
        rr = C.t;
        cycle( l, rr.data(), C.v2.data() );
        matvec( C, C.v2.data(), C.w2.data() );
        const T g = dot( n, C.v2.data(), C.w1.data() ) / rho1;
        for( int i = 0; i < n; ++i ) { C.v2[ i ] -= g * C.v1[ i ]; C.w2[ i ] -= g * C.w1[ i ]; }
        const T rho2 = dot( n, C.v2.data(), C.w2.data() ), a2 = dot( n, C.v2.data(), rr.data() );
        if( rho2 > 0 ) for( int i = 0; i < n; ++i ) C.x[ i ] += a2 / rho2 * C.v2[ i ];
    }
    // Gauss-Seidel on the padded grid: z_i = ( r_i + sum_k W_k,i z_i+off_k + extras ) / d_i
    void gs0( const T *r, T *z, bool forward ) const {
        const int N = N0, P = P0, PP = P * P;
        const T *W = op0->W->data();
        const int *off = op0->off;
        for( int c = 0; c < N; ++c ) {
            const int a = forward ? c : N - 1 - c, row = ( a + 1 ) * P + 1;
            // the neighbor updated just before ( k = 1: ( 0, -1 ) forward, k = 5: ( 0, +1 ) backward ) is
            // kept in a register and added last, the 7 others as a tree: the loop-carried chain is one
            // FMA, not 8 FMAs through memory
            const int kd = forward ? 1 : 5;
            int ko[ 7 ], n = 0;
            for( int k = 0; k < 8; ++k ) if( k != kd ) ko[ n++ ] = k;
            const int step = forward ? 1 : -1;
            T zp = z[ forward ? row - 1 : row + N ];   // padding: 0
            const T *Wd = W + kd * PP, *W0 = W + ko[ 0 ] * PP, *W1 = W + ko[ 1 ] * PP, *W2 = W + ko[ 2 ] * PP, *W3 = W + ko[ 3 ] * PP;
            const T *W4 = W + ko[ 4 ] * PP, *W5 = W + ko[ 5 ] * PP, *W6 = W + ko[ 6 ] * PP;
            const int o0 = off[ ko[ 0 ] ], o1 = off[ ko[ 1 ] ], o2 = off[ ko[ 2 ] ], o3 = off[ ko[ 3 ] ], o4 = off[ ko[ 4 ] ], o5 = off[ ko[ 5 ] ], o6 = off[ ko[ 6 ] ];
            for( int xx = 0, i = forward ? row : row + N - 1; xx < N; ++xx, i += step ) {
                const T q0 = W0[ i ] * z[ i + o0 ] + W1[ i ] * z[ i + o1 ];
                const T q1 = W2[ i ] * z[ i + o2 ] + W3[ i ] * z[ i + o3 ];
                const T q2 = W4[ i ] * z[ i + o4 ] + W5[ i ] * z[ i + o5 ];
                T q = ( r[ i ] + W6[ i ] * z[ i + o6 ] ) + ( ( q0 + q1 ) + q2 );
                if( hasex[ i ] ) for( int p = exptr[ i ]; p < exptr[ i + 1 ]; ++p ) q += exw[ p ] * z[ excol[ p ] ];
                const T id = id0p[ i ];
                zp = q * id + ( Wd[ i ] * id ) * zp;   // loop-carried: one FMA
                z[ i ] = zp;
            }
        }
    }
    // padded interface: the level-0 part of the cycle on the grid, the rest on the graphs
    void apply( int N, int P, const T *r, T *z ) {
        const int PP = P * P;
        Level &L = lv[ 0 ];
        if( lv.size() == 1 ) {
            for( int a = 0; a < N; ++a ) std::memcpy( &L.r[ a * N ], &r[ ( a + 1 ) * P + 1 ], N * sizeof( T ) );
            coarsest( L.r.data(), L.z.data() );
            for( int a = 0; a < N; ++a ) std::memcpy( &z[ ( a + 1 ) * P + 1 ], &L.z[ a * N ], N * sizeof( T ) );
            return;
        }
        Level &C = lv[ 1 ];
        double t0 = now_ms();
        std::fill( z, z + PP, 0 );
        gs0( r, z, true );
        op0->apply( z, t0v.data() );
        std::fill( C.r.begin(), C.r.end(), 0 );
        for( int a = 0; a < N; ++a ) for( int b = 0; b < N; ++b ) {
            const int i = ( a + 1 ) * P + b + 1;
            C.r[ L.agg[ a * N + b ] ] += r[ i ] - t0v[ i ];
        }
        t_lvl[ 0 ] += now_ms() - t0;
        coarse_solve( 1 );
        t0 = now_ms();
        for( int a = 0; a < N; ++a ) for( int b = 0; b < N; ++b ) z[ ( a + 1 ) * P + b + 1 ] += C.x[ L.agg[ a * N + b ] ];
        gs0( r, z, false );
        t_lvl[ 0 ] += now_ms() - t0;
    }
};

// flexible PCG ( Polak-Ribiere beta ) with the AMG preconditioner; b and x padded
int pcg_amg( const Operator &op, Amg &M, const std::vector<T> &b, std::vector<T> &x, T rtol, int maxit, double &t_mv, double &t_pc ) {
    const int N = op.N, P = op.P, PP = P * P;
    std::vector<T> r = b, z( PP, 0 ), p( PP, 0 ), q( PP, 0 ), rold( PP, 0 );
    std::fill( x.begin(), x.end(), 0 );
    auto dot = [ & ]( const std::vector<T> &u, const std::vector<T> &v ) { T s = 0; for( int i = 0; i < PP; ++i ) s += u[ i ] * v[ i ]; return s; };
    auto precond = [ & ] {
        double t0 = now_ms();
        M.apply( N, P, r.data(), z.data() );
        T s = 0;
        for( int i = 0; i < PP; ++i ) s += z[ i ];
        s /= T( N ) * N;
        for( int a = 0; a < N; ++a ) for( int b2 = 0; b2 < N; ++b2 ) z[ ( a + 1 ) * P + b2 + 1 ] -= s;
        t_pc += now_ms() - t0;
    };
    const T nb = std::sqrt( dot( b, b ) );
    if( nb == 0 ) return 0;
    precond();
    p = z;
    T rz = dot( r, z );
    int it = 0;
    for( ; it < maxit; ++it ) {
        double t0 = now_ms();
        const T pq = op.apply( p.data(), q.data() );
        t_mv += now_ms() - t0;
        const T alpha = rz / pq;
        rold = r;
        T rr = 0;
        for( int i = 0; i < PP; ++i ) { x[ i ] += alpha * p[ i ]; r[ i ] -= alpha * q[ i ]; rr += r[ i ] * r[ i ]; }
        if( std::sqrt( rr ) <= rtol * nb ) { ++it; break; }
        precond();
        T rz2 = 0, zd = 0;
        for( int i = 0; i < PP; ++i ) { rz2 += r[ i ] * z[ i ]; zd += z[ i ] * ( r[ i ] - rold[ i ] ); }
        const T beta = zd / rz;
        rz = rz2;
        for( int i = 0; i < PP; ++i ) p[ i ] = z[ i ] + beta * p[ i ];
    }
    return it;
}

// PCG on the interior; returns the number of iterations. b and x padded, mean of b assumed ~ 0.
template<class Pc>
int pcg( const Operator &op, Pc &M, const std::vector<T> &b, std::vector<T> &x, T rtol, int maxit, double &t_mv, double &t_pc ) {
    const int N = op.N, P = op.P, PP = P * P;
    std::vector<T> r = b, z( PP, 0 ), p( PP, 0 ), q( PP, 0 );
    std::fill( x.begin(), x.end(), 0 );
    auto dot = [ & ]( const std::vector<T> &u, const std::vector<T> &v ) { T s = 0; for( int i = 0; i < PP; ++i ) s += u[ i ] * v[ i ]; return s; };
    auto demean = [ & ]( std::vector<T> &u ) {
        T s = 0;
        for( int i = 0; i < PP; ++i ) s += u[ i ];
        s /= T( N ) * N;
        for( int a = 0; a < N; ++a ) for( int b2 = 0; b2 < N; ++b2 ) u[ ( a + 1 ) * P + b2 + 1 ] -= s;
    };
    const T nb = std::sqrt( dot( b, b ) );
    if( nb == 0 ) return 0;
    double t0 = now_ms();
    M.apply( r.data(), z.data() );
    demean( z );
    t_pc += now_ms() - t0;
    p = z;
    T rz = dot( r, z );
    int it = 0;
    for( ; it < maxit; ++it ) {
        t0 = now_ms();
        const T pq = op.apply( p.data(), q.data() );
        t_mv += now_ms() - t0;
        const T alpha = rz / pq;
        T rr = 0;
        for( int i = 0; i < PP; ++i ) { x[ i ] += alpha * p[ i ]; r[ i ] -= alpha * q[ i ]; rr += r[ i ] * r[ i ]; }
        if( std::sqrt( rr ) <= rtol * nb ) { ++it; break; }
        t0 = now_ms();
        M.apply( r.data(), z.data() );
        demean( z );
        t_pc += now_ms() - t0;
        const T rz2 = dot( r, z ), beta = rz2 / rz;
        rz = rz2;
        for( int i = 0; i < PP; ++i ) p[ i ] = z[ i ] + beta * p[ i ];
    }
    return it;
}

// Same iterates as pcg with the tent preconditioner, in 3 row passes per iteration:
//   A  p = ( z - mean z ) + beta p, one row ahead of q = H p ( so the stencil sees updated rows ), p.q
//   B  x += alpha p, r -= alpha q, |r|^2, z = r / diag, row half of the tent restriction of r
//   C  ( after the coarse levels ) z += row half of the tent prolongation, sum z, r.z, sum r
// The mean projection is folded in: r.( z - s ) = r.z - s sum r.
// Pc: id0 ( padded inverse diagonal ), row_b( a, r row ), coarse(), row_c( a, z row ) -- Precond ( tent )
// or TensorPrecond.
template<class Pc>
int pcg_fused( const Operator &op, Pc &M, const std::vector<T> &b, std::vector<T> &x, T rtol, int maxit, double &t_mv, double &t_pc ) {
    const int N = op.N, P = op.P, PP = P * P;
    std::vector<T> r = b, z( PP, 0 ), p( PP, 0 ), q( PP, 0 );
    std::fill( x.begin(), x.end(), 0 );
    const T *W = op.W->data(), *id0 = M.id0.data();
    int off[ 8 ];
    for( int k = 0; k < 8; ++k ) off[ k ] = op.off[ k ];

    // pass C, returns the mean of z and r.( z - mean )
    auto pass_c = [ & ]( T &s, T &rz ) {
        T sz = 0, rzr = 0, sr = 0;
        for( int a = 0; a < N; ++a ) {
            const int row = ( a + 1 ) * P + 1;
            T *zr = z.data() + row;
            const T *rr = r.data() + row;
            M.row_c( a, zr );
            #pragma omp simd reduction( +: sz, rzr, sr )
            for( int x = 0; x < N; ++x ) { sz += zr[ x ]; rzr += rr[ x ] * zr[ x ]; sr += rr[ x ]; }
        }
        s = sz / ( T( N ) * N );
        rz = rzr - s * sr;
    };

    T nb = 0;
    for( int i = 0; i < PP; ++i ) nb += b[ i ] * b[ i ];
    nb = std::sqrt( nb );
    if( nb == 0 ) return 0;
    double t0 = now_ms();
    for( int a = 0; a < N; ++a ) {
        const int row = ( a + 1 ) * P + 1;
        #pragma omp simd
        for( int x = 0; x < N; ++x ) z[ row + x ] = r[ row + x ] * id0[ row + x ];
        M.row_b( a, r.data() + row );
    }
    M.coarse();
    T s, rz;
    pass_c( s, rz );
    t_pc += now_ms() - t0;
    T beta = 0;
    int it = 0;
    for( ; it < maxit; ++it ) {
        // pass A
        t0 = now_ms();
        auto update_p = [ & ]( int a ) {
            const int row = ( a + 1 ) * P + 1;
            #pragma omp simd
            for( int x = 0; x < N; ++x ) p[ row + x ] = z[ row + x ] - s + beta * p[ row + x ];
        };
        update_p( 0 );
        T pq = 0;
        for( int a = 0; a < N; ++a ) {
            if( a + 1 < N ) update_p( a + 1 );
            const int r0 = ( a + 1 ) * P + 1;
            const T *pv = p.data();
            T *qv = q.data();
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
        for( size_t e = 0; e < op.ex_w.size(); ++e ) {
            const int i = op.ex_ij[ e ].first, j = op.ex_ij[ e ].second;
            const T d = op.ex_w[ e ] * ( p[ i ] - p[ j ] );
            q[ i ] += d;
            pq += p[ i ] * d;
        }
        t_mv += now_ms() - t0;
        // pass B
        t0 = now_ms();
        const T alpha = rz / pq;
        T rr = 0;
        for( int a = 0; a < N; ++a ) {
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
        if( std::sqrt( rr ) <= rtol * nb ) { ++it; t_pc += now_ms() - t0; break; }
        // coarse levels + pass C
        M.coarse();
        T rz2;
        pass_c( s, rz2 );
        beta = rz2 / rz;
        rz = rz2;
        t_pc += now_ms() - t0;
    }
    return it;
}

// ================================================================================================
// Newton
// ================================================================================================
struct NewtonStats {
    double tot = 0, geom = 0, integ = 0, lin = 0;
    int its = 0, evals = 0, cg = 0;
    void operator+=( const NewtonStats &o ) { tot += o.tot; geom += o.geom; integ += o.integ; lin += o.lin; its += o.its; evals += o.evals; cg += o.cg; }
};

// psi: start on input, solution on output
template<class Density>
NewtonStats newton( const char *name, int N, const Density &rho, const std::vector<T> &target, std::vector<T> &psi, int precond, T cg_rtol, T tol ) {
    Evaluator<Density> ev( N, rho );
    Operator op;
    // precond 3: tent through the unfused pcg ( reference for pcg_fused )
    // precond 4 / 5: aggregation AMG, K-cycle / V-cycle
    // precond 6: tensor ( semi-coarsened ) blur, env TBAND / TAXES / TC
    const bool fused = precond == 2, amg = precond == 4 || precond == 5, tens = precond == 6 || precond == 9, tens_fused = precond == 6, frames = precond == 7, tdiag = precond == 8;
    TensorDiag TD;
    TensorChain TCH;
    const bool chain = tdiag && ( ! getenv( "DBAND" ) || std::atoi( getenv( "DBAND" ) ) == 1 ) && ! getenv( "NOCHAIN" );
    TCH.ax.band = getenv( "TBAND" ) ? std::atoi( getenv( "TBAND" ) ) : 99;
    TCH.ax.pw = TCH.dg.pw = getenv( "TP" ) ? std::atof( getenv( "TP" ) ) : 0.5;
    TD.ax.band = 99;
    if( getenv( "TBAND" ) ) TD.ax.band = std::atoi( getenv( "TBAND" ) );
    if( getenv( "DBAND" ) ) TD.dg.dband = std::atoi( getenv( "DBAND" ) );
    if( getenv( "TP" ) ) TD.ax.pw = TD.dg.pw = std::atof( getenv( "TP" ) ); else TD.ax.pw = 0.5;
    if( getenv( "DC" ) ) TD.dg.c = std::atof( getenv( "DC" ) );
    TensorPrecond TM;
    TM.pw = 0.5;
    if( getenv( "TPLANES" ) ) TM.planes = true;
    FramePrecond FM;
    if( getenv( "TBAND" ) ) FM.band = std::atoi( getenv( "TBAND" ) );
    if( getenv( "DBAND" ) ) FM.dband = std::atoi( getenv( "DBAND" ) );
    if( getenv( "TC" ) ) FM.c = std::atof( getenv( "TC" ) );
    if( getenv( "TP" ) ) FM.pw = TM.pw = std::atof( getenv( "TP" ) );
    if( getenv( "TBAND" ) ) TM.band = std::atoi( getenv( "TBAND" ) );
    if( getenv( "TAXES" ) ) TM.axes = std::atoi( getenv( "TAXES" ) );
    if( getenv( "TC" ) ) TM.c = std::atof( getenv( "TC" ) );
    Amg AM;
    AM.kcycle = precond == 4;
    if( precond == 3 ) precond = 2;
    Precond M{ amg || tens || frames || tdiag ? 0 : precond, precond > 0 && ! amg && ! tens && ! frames && ! tdiag };
    const int P = N + 2, PP = P * P;
    std::vector<T> rhs( PP ), dx( PP ), trial( N * N );
    auto err_of = [ & ]( const std::vector<T> &m ) { T e = 0; for( int i = 0; i < N * N; ++i ) e = std::max( e, std::abs( m[ i ] - target[ i ] ) / target[ i ] ); return e; };

    double t_start = now_ms(), t_geom = 0, t_int = 0, t_mv = 0, t_pc = 0, t_lin = 0, t_set = 0, t_i1 = 0, t_i2 = 0, t_is = 0;
    auto acc_int = [ & ] { t_i1 += ev.t_tier1; t_i2 += ev.t_tier2; t_is += ev.t_scalar; };
    ev.run( psi );
    acc_int();
    t_geom += ev.t_geom; t_int += ev.t_int;
    T err = err_of( ev.mass );
    T min_t = *std::min_element( target.begin(), target.end() );
    const T eps0 = 0.5 * std::min( *std::min_element( ev.mass.begin(), ev.mass.end() ), min_t );
    std::printf( "== %s N=%d precond=%s cg_rtol=%.0e\n", name, N, amg ? ( AM.kcycle ? "aggregation amg, K-cycle" : "aggregation amg, V-cycle" ) : tens ? ( tens_fused ? "tensor blur ( fused cg )" : "tensor blur" ) : frames ? "tensor blurs, axis + diagonal frames" : tdiag ? "tensor blur + fast diagonal frames" : precond == 0 ? "jacobi" : precond == 1 ? "box" : fused ? "tent ( fused cg )" : "tent", cg_rtol );
    std::printf( " it  max|m-t|/t  step  evals  ring1-flag%%  exact  tier2  scalar   geom ms  integ ms  CG its  CG ms\n" );
    std::printf( " %2d  %.3e     -      1     %6.3f   %5d %6d %6d   %7.1f  %7.1f\n", 0, err, 100.0 * ev.n_flag1 / ( N * N ), ev.n_exact, ev.n_fine, ev.n_dens, ev.t_geom, ev.t_int );
    int total_cg = 0, total_evals = 1, nits = 0;
    for( int it = 1; it <= 50 && err > tol; ++it, ++nits ) {
        // linear system H dx = target - m
        double tl = now_ms();
        op.bind( ev );
        if( amg ) AM.build( op ); else if( tens ) TM.build( op ); else if( frames ) FM.build( op ); else if( chain ) TCH.build( op ); else if( tdiag ) TD.build( op ); else M.build( op );
        t_set += now_ms() - tl;
        std::fill( rhs.begin(), rhs.end(), 0 );
        for( int a = 0; a < N; ++a ) for( int b = 0; b < N; ++b ) rhs[ ( a + 1 ) * P + b + 1 ] = target[ a * N + b ] - ev.mass[ a * N + b ];
        {
            // H is singular along the constants: drop the mean of the rhs ( sum of the masses - N^2,
            // ~1e-9 from rounding ), else CG stalls once |rhs| reaches ~1e-8
            T sb = 0;
            for( auto v : rhs ) sb += v;
            sb /= T( N ) * N;
            for( int a = 0; a < N; ++a ) for( int b = 0; b < N; ++b ) rhs[ ( a + 1 ) * P + b + 1 ] -= sb;
        }
        double mv = 0, pc = 0;
        const int its = amg ? pcg_amg( op, AM, rhs, dx, cg_rtol, 20000, mv, pc ) : tens ? ( tens_fused ? pcg_fused( op, TM, rhs, dx, cg_rtol, 20000, mv, pc ) : pcg( op, TM, rhs, dx, cg_rtol, 20000, mv, pc ) ) : frames ? pcg( op, FM, rhs, dx, cg_rtol, 20000, mv, pc ) : chain ? pcg( op, TCH, rhs, dx, cg_rtol, 20000, mv, pc ) : tdiag ? pcg( op, TD, rhs, dx, cg_rtol, 20000, mv, pc ) : ( fused ? pcg_fused<Precond> : pcg<Precond> )( op, M, rhs, dx, cg_rtol, 20000, mv, pc );
        const double dtl = now_ms() - tl;
        t_lin += dtl; t_mv += mv; t_pc += pc;
        total_cg += its;
        // damping
        T tau = 1;
        int evals = 0;
        double tg = 0, ti = 0;
        while( true ) {
            for( int a = 0; a < N; ++a ) for( int b = 0; b < N; ++b ) trial[ a * N + b ] = psi[ a * N + b ] + tau * dx[ ( a + 1 ) * P + b + 1 ];
            ev.run( trial );
            acc_int();
            ++evals;
            tg += ev.t_geom; ti += ev.t_int;
            const T e2 = err_of( ev.mass );
            const T mmin = *std::min_element( ev.mass.begin(), ev.mass.end() );
            if( ( mmin > eps0 && e2 <= ( 1 - tau / 2 ) * err ) || tau < 1e-6 ) { err = e2; break; }
            tau /= 2;
        }
        psi = trial;
        total_evals += evals;
        t_geom += tg; t_int += ti;
        std::printf( " %2d  %.3e  %5.3f  %4d     %6.3f   %5d %6d %6d   %7.1f  %7.1f  %6d  %7.1f\n", it, err, tau, evals,
                     100.0 * ev.n_flag1 / ( N * N ), ev.n_exact, ev.n_fine, ev.n_dens, tg, ti, its, dtl );
    }
    const double tot = now_ms() - t_start;
    std::printf( " total %.0f ms: geometry %.0f, integrals %.0f, linear %.0f ( setup %.0f, matvec%s %.0f, precond%s %.0f, rest %.0f ); %d evals, %d CG its\n",
                 tot, t_geom, t_int, t_lin, t_set, fused ? " + pass A" : "", t_mv, fused ? " + passes B, C" : "", t_pc, t_lin - t_set - t_mv - t_pc, total_evals, total_cg );
    if( amg ) {
        std::printf( " amg levels:" );
        for( size_t l = 0; l < AM.lv.size(); ++l ) std::printf( " %d", AM.lv[ l ].n );
        std::printf( " | smoothing+transfer ms per level:" );
        for( size_t l = 0; l + 1 < AM.lv.size(); ++l ) std::printf( " %.0f", AM.t_lvl[ l ] );
        std::printf( " | coarsest %.0f ms | setup: graph %.0f pairwise %.0f coarsen %.0f vectors %.0f cholesky %.0f\n", AM.t_coarsest, AM.t_setup[ 0 ], AM.t_setup[ 1 ], AM.t_setup[ 2 ], AM.t_setup[ 3 ], AM.t_setup[ 4 ] );
    }
    if( chain ) std::printf( " tensor+diag chain: %d + %d subspaces, setup %.0f + %.0f ms, apply %.0f ( %.0f axis ) ms\n", TCH.ax.nkept, TCH.dg.nkept, TCH.ax.t_setup, TCH.dg.t_setup, TCH.ax.t_restrict + TCH.ax.t_scale + TCH.ax.t_prolong + TCH.dg.t_apply, TCH.ax.t_restrict + TCH.ax.t_scale + TCH.ax.t_prolong );
    else if( tdiag ) std::printf( " tensor+diag: %d + %d subspaces, setup %.0f + %.0f ms, apply %.0f ( %.0f axis ) ms\n", TD.ax.nkept, TD.dg.nkept, TD.ax.t_setup, TD.dg.t_setup, TD.ax.t_restrict + TD.ax.t_scale + TD.ax.t_prolong + TD.dg.t_apply, TD.ax.t_restrict + TD.ax.t_scale + TD.ax.t_prolong );
    if( frames ) std::printf( " frames: %d subspaces, setup %.0f ms ( brute force ), restrict %.0f ms, scale %.0f ms, prolong %.0f ms\n", FM.nkept, FM.t_setup, FM.t_restrict, FM.t_scale, FM.t_prolong );
    if( tens ) std::printf( " tensor: %d subspaces, setup %.0f ms, restrict %.0f ms, scale %.0f ms, prolong %.0f ms\n", TM.nkept, TM.t_setup, TM.t_restrict, TM.t_scale, TM.t_prolong );
    std::printf( " integrals: tier 1 %.0f ms, tier 2 %.0f ms, scalar + exact %.0f ms\n", t_i1, t_i2, t_is );
    return { tot, t_geom, t_int, t_lin, nits, total_evals, total_cg };
}

// psi on the n x n grid -> psi on the 2n x 2n grid. Seeds at integer points of the box
// [ -1/2, n - 1/2 ]^2; the fine grid is the coarse one scaled by 2 ( x_f = 2 x_c + 1/2 ), so the
// fine psi is 4 psi_c at u = ( a - 1/2 ) / 2, by separable cubic Lagrange interpolation ( exact on
// quadratics, so the |x|^2 part of psi gives no parity pattern, nor a jump on the border where the
// stencil is one-sided )
std::vector<T> refine_psi( int n, const std::vector<T> &pc ) {
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
    for( int a = 0; a < N; ++a ) for( int b = 0; b < n; ++b ) {
        T v = 0;
        for( int k = 0; k < 4; ++k ) v += w[ a ][ k ] * pc[ ( j0[ a ] + k ) * n + b ];
        t[ a * n + b ] = v;
    }
    for( int a = 0; a < N; ++a ) for( int b = 0; b < N; ++b ) {
        T v = 0;
        for( int k = 0; k < 4; ++k ) v += w[ b ][ k ] * t[ a * n + j0[ b ] + k ];
        pf[ a * N + b ] = 4 * v;
    }
    return pf;
}

// solve on n0, 2 n0, ..., N: each level starts from the refined solution of the previous one,
// intermediate levels stop at ctol. make( n ) returns { density, target, start psi ( used at n0 ) }
template<class Make>
std::vector<T> coarse_to_fine( const char *name, int N, int n0, T ctol, Make make, int precond, T cg_rtol, T tol ) {
    NewtonStats all;
    std::vector<T> psi;
    double t_ref = 0;
    for( int n = n0; n <= N; n *= 2 ) {
        auto [ rho, target, start ] = make( n );
        if( n == n0 ) psi = start;
        else {
            const double t0 = now_ms();
            psi = refine_psi( n / 2, psi );
            t_ref += now_ms() - t0;
        }
        char nm[ 96 ];
        std::snprintf( nm, 96, "%s ( coarse to fine, level %d of %d )", name, n, N );
        all += newton( nm, n, rho, target, psi, precond, cg_rtol, n == N ? tol : ctol );
    }
    std::printf( "== coarse to fine %d -> %d ( ctol %.0e ): total %.0f ms ( refine %.1f ): geometry %.0f, integrals %.0f, linear %.0f; %d Newton its, %d evals, %d CG its\n",
                 n0, N, ctol, all.tot + t_ref, t_ref, all.geom, all.integ, all.lin, all.its, all.evals, all.cg );
    return psi;
}

// weights of the hessian at psi: range, and anisotropy ( x weights / y weights ) per cell
template<class Density>
void weight_stats( int N, const Density &rho, const std::vector<T> &psi ) {
    Evaluator<Density> ev( N, rho );
    ev.run( psi );
    const int P = N + 2, PP = P * P;
    std::vector<T> an, d, ex_x( N * N, 0 ), ex_y( N * N, 0 ), ex_d( N * N, 0 );
    for( auto &e : ev.extras ) {
        const int dx = e.j / N - e.i / N, dy = e.j % N - e.i % N;
        ( dx && dy ? ex_d : dx ? ex_x : ex_y )[ e.i ] += e.w;
    }
    for( int a = 0; a < N; ++a ) for( int b = 0; b < N; ++b ) {
        const int i = ( a + 1 ) * P + b + 1;
        T wx = ex_x[ a * N + b ], wy = ex_y[ a * N + b ], wd = ex_d[ a * N + b ];
        for( int k = 0; k < 8; ++k ) {
            const T w = ev.W[ k * PP + i ];
            if( T1::off_x[ k ] && T1::off_y[ k ] ) wd += w; else if( T1::off_x[ k ] ) wx += w; else wy += w;
        }
        d.push_back( wx + wy + wd );
        if( wx > 0 && wy > 0 ) an.push_back( std::max( wx / wy, wy / wx ) );
    }
    std::sort( an.begin(), an.end() );
    std::sort( d.begin(), d.end() );
    auto pct = [ & ]( const std::vector<T> &v, double p ) { return v[ std::min<size_t>( v.size() - 1, size_t( p * v.size() ) ) ]; };
    std::printf( "N=%d weights: diag min %.2e  p1 %.2e  median %.2e  max %.2e | anisotropy max(wx/wy, wy/wx): median %.1f  p90 %.1f  p99 %.1f  max %.1f\n",
                 N, d[ 0 ], pct( d, 0.01 ), pct( d, 0.5 ), d.back(), pct( an, 0.5 ), pct( an, 0.9 ), pct( an, 0.99 ), an.back() );
    // conductivity tensor K = sum_k w_k d_k d_k^T per cell: eigenvalue ratio and angle of the strong
    // direction ( degrees from the a axis ), dumped as float32 ( ratio, angle, diag ) to WSTATS=<file>
    std::vector<float> dump;
    int hist[ 8 ] = {}, nstrong = 0;
    for( int a = 0; a < N; ++a ) for( int b = 0; b < N; ++b ) {
        const int i = ( a + 1 ) * P + b + 1;
        T kxx = 0, kxy = 0, kyy = 0, dg = 0;
        for( int k = 0; k < 8; ++k ) {
            const T w = ev.W[ k * PP + i ], ox = T1::off_x[ k ], oy = T1::off_y[ k ];
            kxx += w * ox * ox; kxy += w * ox * oy; kyy += w * oy * oy; dg += w;
        }
        const T tr = kxx + kyy, det = kxx * kyy - kxy * kxy, disc = std::sqrt( std::max( T( 0 ), tr * tr / 4 - det ) );
        const T l1 = tr / 2 + disc, l2 = std::max( tr / 2 - disc, T( 1e-300 ) ), ratio = l1 / l2;
        T ang = T( 0.5 ) * std::atan2( 2 * kxy, kxx - kyy ) * T( 180 / M_PI );
        if( ang < 0 ) ang += 180;
        if( ratio > 3 ) { ++hist[ int( ( ang + 11.25 ) / 22.5 ) % 8 ]; ++nstrong; }
        dump.push_back( float( ratio ) ); dump.push_back( float( ang ) ); dump.push_back( float( dg ) );
    }
    std::printf( "cells with tensor anisotropy > 3: %d ( %.2f %% ); strong direction histogram ( 0, 22.5, ..., 157.5 deg ):", nstrong, 100.0 * nstrong / ( N * N ) );
    for( int h = 0; h < 8; ++h ) std::printf( " %d", hist[ h ] );
    std::printf( "\n" );
    const char *f = getenv( "WSTATS" );
    if( f && std::strlen( f ) > 1 ) { FILE *o = std::fopen( f, "wb" ); std::fwrite( dump.data(), 4, dump.size(), o ); std::fclose( o ); }
}

// ================================================================================================
// preconditioner diagnostics: Ritz values of M^-1 H ( Lanczos coefficients of PCG on a random
// rhs ) and the error left after a few iterations ( where the slow modes live )
// ================================================================================================
template<class Pc>
void pcg_spectrum( const char *name, const Operator &op, Pc &M, int its, int dump_its, const char *dump ) {
    const int N = op.N, P = op.P, PP = P * P;
    std::mt19937 rng( 7 );
    std::normal_distribution<T> nd;
    std::vector<T> xs( PP, 0 ), b( PP, 0 );
    for( int a = 0; a < N; ++a ) for( int c = 0; c < N; ++c ) xs[ ( a + 1 ) * P + c + 1 ] = nd( rng );
    op.apply( xs.data(), b.data() );
    std::vector<T> x( PP, 0 ), r = b, z( PP, 0 ), p( PP, 0 ), q( PP, 0 );
    auto dot = [ & ]( const std::vector<T> &u, const std::vector<T> &v ) { T s = 0; for( int i = 0; i < PP; ++i ) s += u[ i ] * v[ i ]; return s; };
    auto demean = [ & ]( std::vector<T> &u ) {
        T s = 0;
        for( int a = 0; a < N; ++a ) for( int c = 0; c < N; ++c ) s += u[ ( a + 1 ) * P + c + 1 ];
        s /= T( N ) * N;
        for( int a = 0; a < N; ++a ) for( int c = 0; c < N; ++c ) u[ ( a + 1 ) * P + c + 1 ] -= s;
    };
    M.apply( r.data(), z.data() ); demean( z );
    p = z;
    T rz = dot( r, z );
    std::vector<T> al, be;
    T rr0 = 0;
    for( int it = 0; it < its; ++it ) {
        const T pq = op.apply( p.data(), q.data() ), alpha = rz / pq;
        T rr = 0;
        for( int i = 0; i < PP; ++i ) { x[ i ] += alpha * p[ i ]; r[ i ] -= alpha * q[ i ]; rr += r[ i ] * r[ i ]; }
        al.push_back( alpha );
        if( it == 0 ) rr0 = rr;
        if( rr < 1e-16 * rr0 ) break; // stop before the Lanczos coefficients lose their meaning
        if( it + 1 == dump_its && dump ) {
            // error x* - x, demeaned, as float32 N x N
            std::vector<T> e( PP );
            for( int i = 0; i < PP; ++i ) e[ i ] = xs[ i ] - x[ i ];
            demean( e );
            std::vector<float> o( N * N );
            for( int a = 0; a < N; ++a ) for( int c = 0; c < N; ++c ) o[ a * N + c ] = float( e[ ( a + 1 ) * P + c + 1 ] );
            FILE *f = std::fopen( dump, "wb" ); std::fwrite( o.data(), 4, o.size(), f ); std::fclose( f );
        }
        M.apply( r.data(), z.data() ); demean( z );
        const T rz2 = dot( r, z ), beta = rz2 / rz;
        rz = rz2;
        be.push_back( beta );
        for( int i = 0; i < PP; ++i ) p[ i ] = z[ i ] + beta * p[ i ];
    }
    // Lanczos tridiagonal, eigenvalues by Jacobi rotations ( small )
    const int k = int( al.size() );
    be.resize( k );
    std::vector<T> A( k * k, 0 );
    for( int j = 0; j < k; ++j ) {
        A[ j * k + j ] = 1 / al[ j ] + ( j ? be[ j - 1 ] / al[ j - 1 ] : 0 );
        if( j + 1 < k ) A[ j * k + j + 1 ] = A[ ( j + 1 ) * k + j ] = std::sqrt( be[ j ] ) / al[ j ];
    }
    for( int sweep = 0; sweep < 100; ++sweep ) {
        T off = 0;
        for( int i = 0; i < k; ++i ) for( int j = i + 1; j < k; ++j ) off += A[ i * k + j ] * A[ i * k + j ];
        if( off < 1e-24 ) break;
        for( int i = 0; i < k; ++i ) for( int j = i + 1; j < k; ++j ) {
            const T aij = A[ i * k + j ];
            if( std::abs( aij ) < 1e-300 ) continue;
            const T th = ( A[ j * k + j ] - A[ i * k + i ] ) / ( 2 * aij ), t = ( th >= 0 ? 1 : -1 ) / ( std::abs( th ) + std::sqrt( th * th + 1 ) );
            const T cs = 1 / std::sqrt( t * t + 1 ), sn = t * cs;
            for( int m = 0; m < k; ++m ) {
                const T u = A[ m * k + i ], v = A[ m * k + j ];
                A[ m * k + i ] = cs * u - sn * v; A[ m * k + j ] = sn * u + cs * v;
            }
            for( int m = 0; m < k; ++m ) {
                const T u = A[ i * k + m ], v = A[ j * k + m ];
                A[ i * k + m ] = cs * u - sn * v; A[ j * k + m ] = sn * u + cs * v;
            }
        }
    }
    std::vector<T> ev( k );
    for( int j = 0; j < k; ++j ) ev[ j ] = A[ j * k + j ];
    std::sort( ev.begin(), ev.end() );
    std::printf( "%-26s %3d its  Ritz: min %.4f %.4f %.4f %.4f | max %.3f %.3f | kappa %.0f\n", name, k, ev[ 0 ], ev[ 1 ], ev[ 2 ], ev[ 3 ], ev[ k - 2 ], ev[ k - 1 ], ev[ k - 1 ] / ev[ 0 ] );
}

template<class Density>
void spectra( int N, const Density &rho, const std::vector<T> &psi, const char *dir ) {
    Evaluator<Density> ev( N, rho );
    ev.run( psi );
    Operator op;
    op.bind( ev );
    char f[ 512 ];
    auto path = [ & ]( const char *tag ) { std::snprintf( f, 512, "%s/err_%s_%d.f32", dir, tag, N ); return f; };
    { Precond M{ 0, false }; M.build( op ); pcg_spectrum( "jacobi", op, M, 200, 0, nullptr ); }
    { Precond M{ 2, true }; M.build( op ); pcg_spectrum( "tent", op, M, 120, 12, path( "tent" ) ); }
    for( int band : { 1, 2, 99 } ) {
        TensorPrecond M; M.band = band; M.axes = false; M.build( op );
        char nm[ 64 ]; std::snprintf( nm, 64, "tensor band %d", band );
        char tg[ 32 ]; std::snprintf( tg, 32, "tensor%d", band );
        pcg_spectrum( nm, op, M, 120, 12, band == 2 ? path( tg ) : nullptr );
    }
    for( T pw : { 0.5, 1.0, 2.0 } )
        for( int band : { 2, 99 } ) {
            TensorPrecond M; M.band = band; M.axes = false; M.pw = pw; M.build( op );
            char nm[ 64 ]; std::snprintf( nm, 64, "tensor band %d pw %.1f", band, pw );
            pcg_spectrum( nm, op, M, 120, 0, nullptr );
        }
    for( T pw : { 1.0, 2.0 } )
        for( int db : { 2, 4 } ) {
            FramePrecond M; M.band = 99; M.dband = db; M.pw = pw; M.build( op );
            char nm[ 64 ]; std::snprintf( nm, 64, "frames 99/%d pw %.1f", db, pw );
            pcg_spectrum( nm, op, M, 120, 0, nullptr );
        }
    for( T pw : { 0.5, 1.0 } )
        for( int db : { 1, 2, 3 } ) {
            TensorDiag M; M.ax.band = 99; M.ax.pw = M.dg.pw = pw; M.dg.dband = db; M.build( op );
            char nm[ 64 ]; std::snprintf( nm, 64, "fast tensor+diag %d pw %.1f", db, pw );
            pcg_spectrum( nm, op, M, 120, 0, nullptr );
        }
    { TensorChain M; M.ax.band = 99; M.ax.pw = 0.5; M.build( op ); pcg_spectrum( "tensor+diag chain", op, M, 120, 0, nullptr ); }
    for( int db : { 1, 2 } ) {
        FramePrecond M; M.band = 1; M.dband = db; M.build( op );
        char nm[ 64 ]; std::snprintf( nm, 64, "frames band 1 dband %d", db );
        pcg_spectrum( nm, op, M, 120, 0, nullptr );
    }
}

// ================================================================================================
// checks: SIMD quadrature vs adaptive, hessian vs finite differences
// ================================================================================================
template<class Density>
void check_hessian( const char *name, int N, const Density &rho, const std::vector<T> &psi ) {
    Evaluator<Density> ev( N, rho ), ev2( N, rho );
    ev.run( psi );
    std::mt19937 rng( 1 );
    T worst = 0;
    const T h = 1e-6;
    for( int t = 0; t < 40; ++t ) {
        const int j = rng() % ( N * N );
        std::vector<T> p2 = psi, p3 = psi;
        p2[ j ] += h; p3[ j ] -= h;
        ev2.run( p2 );
        std::vector<T> mp = ev2.mass;
        ev2.run( p3 );
        // column j: d m_i / d psi_j for the neighbors i of j
        for( int k = 0; k < 8; ++k ) {
            const int ia = j / N + T1::off_x[ k ], ib = j % N + T1::off_y[ k ];
            if( ia < 0 || ia >= N || ib < 0 || ib >= N ) continue;
            const int i = ia * N + ib;
            const T fd = ( mp[ i ] - ev2.mass[ i ] ) / ( 2 * h );
            // -w_ij from i's planes ( direction toward j ) or extras
            T w = 0;
            for( int kk = 0; kk < 8; ++kk )
                if( ia + T1::off_x[ kk ] == j / N && ib + T1::off_y[ kk ] == j % N ) w += ev.W[ kk * ( N + 2 ) * ( N + 2 ) + ( ia + 1 ) * ( N + 2 ) + ib + 1 ];
            for( auto &e : ev.extras ) if( e.i == i && e.j == j ) w += e.w;
            const T e = std::abs( fd + w ) / std::max( T( 1e-3 ), std::abs( w ) );
            if( e > worst ) std::printf( "   off-diag i=%d ( %d %d ) j=%d: fd %.10e  -w %.10e  flags i: %d %d  j: %d %d\n", i, i / N, i % N, j, fd, -w, ev.flag1[ i ], ev.dbad[ i ], ev.flag1[ j ], ev.dbad[ j ] );
            worst = std::max( worst, e );
        }
        const T fdd = ( mp[ j ] - ev2.mass[ j ] ) / ( 2 * h );
        T d = 0;
        for( int kk = 0; kk < 8; ++kk ) d += ev.W[ kk * ( N + 2 ) * ( N + 2 ) + ( j / N + 1 ) * ( N + 2 ) + j % N + 1 ];
        for( auto &e : ev.extras ) if( e.i == j ) d += e.w;
        worst = std::max( worst, std::abs( fdd - d ) / std::max( T( 1e-3 ), d ) );
    }
    std::printf( "%s: hessian vs centered finite differences ( 40 columns ), worst relative error %.2e\n", name, worst );
}

int main( int argc, char **argv ) {
    std::setvbuf( stdout, nullptr, _IOLBF, 0 );
    // options n0=<coarsest size> ( coarse to fine when > 0 ) and ctol=<tolerance of the coarse levels>
    int n0 = 0;
    T ctol = 1e-3;
    {
        int k = 1;
        for( int i = 1; i < argc; ++i ) {
            const std::string a = argv[ i ];
            if( a.rfind( "n0=", 0 ) == 0 ) n0 = std::atoi( a.c_str() + 3 );
            else if( a.rfind( "ctol=", 0 ) == 0 ) ctol = std::atof( a.c_str() + 5 );
            else argv[ k++ ] = argv[ i ];
        }
        argc = k;
    }
    const std::string mode = argc > 1 ? argv[ 1 ] : "lebesgue";
    const int N = argc > 2 ? std::atoi( argv[ 2 ] ) : 512;
    const double srel = argc > 3 ? std::atof( argv[ 3 ] ) : 0.05;
    const std::string pc = argc > 4 ? argv[ 4 ] : "tensor";
    const T cg_rtol = argc > 5 ? std::atof( argv[ 5 ] ) : 1e-4;
    const T tol = getenv( "TOL" ) ? std::atof( getenv( "TOL" ) ) : 1e-9;

    if( mode == "lebesgue" ) {
        // masses: smooth pattern of contrast 10
        std::vector<T> target( N * N );
        T s = 0;
        for( int a = 0; a < N; ++a ) for( int b = 0; b < N; ++b ) {
            const T x = ( a + 0.5 ) / N, y = ( b + 0.5 ) / N;
            target[ a * N + b ] = 1 + 9 * 0.5 * ( 1 + std::sin( 2 * M_PI * 2 * x ) * std::sin( 2 * M_PI * 3 * y ) );
            s += target[ a * N + b ];
        }
        for( auto &t : target ) t *= T( N ) * N / s;
        const int pci = pc == "jacobi" ? 0 : pc == "box" ? 1 : pc == "tent-unfused" ? 3 : pc == "agg" ? 4 : pc == "agg-v" ? 5 : pc == "tensor" ? 6 : pc == "frames" ? 7 : pc == "tensor-diag" ? 8 : pc == "tensor-unfused" ? 9 : pc == "tent" ? 2 : 6;
        if( n0 > 0 ) {
            // coarse targets: 2 x 2 block sums / 4 ( total n^2 at every level )
            coarse_to_fine( "lebesgue, smooth masses", N, n0, ctol, [ & ]( int n ) {
                std::vector<T> t = target;
                for( int m = N; m > n; m /= 2 ) {
                    std::vector<T> c( m * m / 4 );
                    for( int a = 0; a < m / 2; ++a ) for( int b = 0; b < m / 2; ++b )
                        c[ a * m / 2 + b ] = ( t[ 2 * a * m + 2 * b ] + t[ 2 * a * m + 2 * b + 1 ] + t[ ( 2 * a + 1 ) * m + 2 * b ] + t[ ( 2 * a + 1 ) * m + 2 * b + 1 ] ) / 4;
                    t.swap( c );
                }
                return std::tuple{ Lebesgue{ n }, t, std::vector<T>( n * n, 0 ) };
            }, pci, cg_rtol, tol );
        } else {
            Lebesgue rho{ N };
            std::vector<T> psi( N * N, 0 );
            newton( "lebesgue, smooth masses", N, rho, target, psi, pci, cg_rtol, tol );
        }
    }
    if( mode == "gauss" || mode == "check-gauss" ) {
        const T c = ( N - 1 ) / 2.0, sig = srel * N;
        const double tau = argc > 6 ? std::atof( argv[ 6 ] ) : 0.08, qtol = argc > 7 ? std::atof( argv[ 7 ] ) : 1e-13;
        Gauss rho( N, c, c, sig, tau, qtol );
        std::vector<T> target( N * N, 1 );
        // start: the grid contracted around c to +- 2.5 sigma ( psi quadratic => same combinatorics )
        const T sc = 5 * sig / N;
        std::vector<T> psi( N * N );
        for( int a = 0; a < N; ++a ) for( int b = 0; b < N; ++b ) psi[ a * N + b ] = ( 1 - sc ) * ( ( a - c ) * ( a - c ) + ( b - c ) * ( b - c ) );
        const int pci = pc == "jacobi" ? 0 : pc == "box" ? 1 : pc == "tent-unfused" ? 3 : pc == "agg" ? 4 : pc == "agg-v" ? 5 : pc == "tensor" ? 6 : pc == "frames" ? 7 : pc == "tensor-diag" ? 8 : pc == "tensor-unfused" ? 9 : pc == "tent" ? 2 : 6;
        char name[ 64 ];
        std::snprintf( name, 64, "gauss sigma=%.3g N", srel );
        if( mode == "gauss" && n0 > 0 ) {
            auto sol = coarse_to_fine( name, N, n0, ctol, [ & ]( int n ) {
                const T cn = ( n - 1 ) / 2.0, sn = srel * n, scn = 5 * sn / n;
                std::vector<T> st( n * n );
                for( int a = 0; a < n; ++a ) for( int b = 0; b < n; ++b ) st[ a * n + b ] = ( 1 - scn ) * ( ( a - cn ) * ( a - cn ) + ( b - cn ) * ( b - cn ) );
                return std::tuple{ Gauss( n, cn, cn, sn, tau, qtol ), std::vector<T>( n * n, 1 ), st };
            }, pci, cg_rtol, tol );
            if( getenv( "WSTATS" ) ) weight_stats( N, rho, sol );
            if( getenv( "SPECTRA" ) ) spectra( N, rho, sol, getenv( "SPECTRA" ) );
        } else if( mode == "gauss" ) {
            newton( name, N, rho, target, psi, pci, cg_rtol, tol );
        } else {
            // quadrature: SIMD rule ( cells with spread <= tau ) vs adaptive scalar path
            Evaluator<Gauss> ev( N, rho );
            ev.run( psi );
            Gauss tight( N, c, c, sig, 1e30, 1e-16 );
            Evaluator<Gauss> ev2( N, tight );
            ev2.run( psi );
            T worst = 0;
            for( int i = 0; i < N * N; ++i ) worst = std::max( worst, std::abs( ev.mass[ i ] - ev2.mass[ i ] ) );
            Gauss adaptive_all( N, c, c, sig, -1, 1e-16 ); // every cell through the adaptive path
            Evaluator<Gauss> ev3( N, adaptive_all );
            ev3.run( psi );
            T w2 = 0, w3 = 0;
            for( int i = 0; i < N * N; ++i ) { w2 = std::max( w2, std::abs( ev2.mass[ i ] - ev3.mass[ i ] ) ); w3 = std::max( w3, std::abs( ev.mass[ i ] - ev3.mass[ i ] ) ); }
            std::printf( "quadrature at the start point ( target masses 1 ): |simd(tau=%.2g) - adaptive| %.2e, |simd(all) - adaptive| %.2e, dens fallback cells %d\n", rho.tau, w3, w2, ev.n_dens );
            // the start is degenerate ( 4 cells per vertex, mass not differentiable ): check on a perturbed psi
            std::mt19937 rng( 3 );
            std::normal_distribution<T> nd;
            for( auto &p : psi ) p += 0.05 * nd( rng );
            check_hessian( "gauss ( start + noise 0.05 )", N, rho, psi );
        }
    }
    if( mode == "mix" ) {
        // one case for both features: density = 2 gaussians, masses = 2 other gaussians ( unit
        // square coordinates, grid point a at u = ( a + 1/2 ) / n ); start psi = 0 on the coarsest level
        const double tau = argc > 6 ? std::atof( argv[ 6 ] ) : 0.08, qtol = argc > 7 ? std::atof( argv[ 7 ] ) : 1e-13;
        const int pci = pc == "jacobi" ? 0 : pc == "box" ? 1 : pc == "tent-unfused" ? 3 : pc == "agg" ? 4 : pc == "agg-v" ? 5 : pc == "tensor" ? 6 : pc == "frames" ? 7 : pc == "tensor-diag" ? 8 : pc == "tensor-unfused" ? 9 : pc == "tent" ? 2 : 6;
        auto make = [ & ]( int n ) {
            auto G = [ & ]( double ux, double uy, double s ) { return Gauss( n, ux * n - 0.5, uy * n - 0.5, s * n, tau, qtol ); };
            GaussMix rho( n, 0.5, G( 0.3, 0.35, 0.12 ), G( 0.7, 0.65, 0.15 ) );
            std::vector<T> t( n * n );
            T s = 0;
            for( int a = 0; a < n; ++a ) for( int b = 0; b < n; ++b ) {
                const T x = ( a + 0.5 ) / n, y = ( b + 0.5 ) / n;
                auto g = [ & ]( T cx, T cy, T sg ) { return std::exp( -( ( x - cx ) * ( x - cx ) + ( y - cy ) * ( y - cy ) ) / ( 2 * sg * sg ) ); };
                t[ a * n + b ] = 0.6 * g( 0.35, 0.7, 0.2 ) + 0.4 * g( 0.7, 0.3, 0.25 );
                s += t[ a * n + b ];
            }
            for( auto &v : t ) v *= T( n ) * n / s;
            return std::tuple{ rho, t, std::vector<T>( n * n, 0 ) };
        };
        if( n0 > 0 ) {
            auto sol = coarse_to_fine( "mix: 2 gaussians -> 2 gaussian masses", N, n0, ctol, make, pci, cg_rtol, tol );
            if( const char *f = getenv( "PSIOUT" ) ) { FILE *o = std::fopen( f, "wb" ); std::fwrite( sol.data(), sizeof( T ), sol.size(), o ); std::fclose( o ); }
            if( getenv( "WSTATS" ) ) weight_stats( N, std::get<0>( make( N ) ), sol );
            if( getenv( "SPECTRA" ) ) spectra( N, std::get<0>( make( N ) ), sol, getenv( "SPECTRA" ) );
        }
        else {
            auto [ rho, t, psi ] = make( N );
            newton( "mix: 2 gaussians -> 2 gaussian masses", N, rho, t, psi, pci, cg_rtol, tol );
        }
    }
    if( mode == "check-lebesgue" ) {
        std::vector<T> psi( N * N );
        std::mt19937 rng( 2 );
        std::normal_distribution<T> nd;
        for( auto &p : psi ) p = 0.2 * nd( rng );
        Lebesgue rho{ N };
        check_hessian( "lebesgue ( noise 0.2 )", N, rho, psi );
    }
}
