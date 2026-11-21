#pragma once

// ( the lambdas of the unrolled folds: clang leaves them as calls once their body is large -- closures and arrays in memory, Ring1Reg with
// its extras 41 -> 14 ns/cell on an M4 when forced )
#ifndef SDOT_INLINE_LAMBDA
#define SDOT_INLINE_LAMBDA __attribute__( ( always_inline ) )
#endif

#include <asimd/asimd.h>
#include <bit>
#include <limits>
#include <utility>
#include "StencilTables.h"

namespace sdot::grid {

/// RING 1 OF A ROW BLOCK, IN REGISTERS ( 2D, unrolled, asimd ): the cells ( a, b0 .. b0 + B - 1 ) of an N x N grid, lanes along b,
/// what `StencilKernel<1>::row_block` computes ( vertex k between the directions k and k + 1, relative to the seed, edge k from
/// vertex k - 1 to vertex k ), in the same operations, so the same values.
///
/// Every table of `StencilTables<1>` is a template argument here: the offsets, the 24 pairs of the tightening and the vertex matrices
/// fold into the code ( a coefficient 0, 1 or -1 costs nothing ). The psi of a row come in once: the neighbours to the left and to
/// the right of a vector are `ext_lanes` of the vector before, the vector itself and the vector after ( `ROT` ), or unaligned loads
/// ( `! ROT`, to compare ).
///
/// `psi_pad`: psi padded by one cell of -inf ( `Evaluator::pad( 1 )` ), P = N + 2 columns. Outputs: the vertices, `ok[ k ]` the lanes
/// whose edge k has a seed ( bit x: cell b0 + x; not the box, not a seed out of the grid ), the area ( 2 x the signed area / 2 ).
///
/// With an `Ext` ( the zone certificate ), while the vertices are still in registers: the box of each cell, the inversion test and the
/// prediction of ring 2. The cell is convex ( or inverted, then thrown away ), its vertex k between the directions k and k + 1: the
/// support in a direction is at the one or two vertices whose normals bracket it ( `sup_mask` ), no max over the 8 vertices.
template<int B, class T, bool ROT = true>
struct Ring1Reg {
    using Tb = StencilTables<1>;
    static constexpr int D = 8;
    static constexpr int W = asimd::SimdSize<T>::value;
    static_assert( B % W == 0 && B <= 64, "Ring1Reg: B a multiple of the SIMD width, at most 64" );
    static constexpr int NV = B / W;
    using V = asimd::SimdVec<T,W>;
    using U = unsigned long long;

    /// the extra outputs for the zone certificate. `p2`: psi padded by two cells of -inf ( `Evaluator::pad( 2 )` ), N + 4 columns
    struct Ext {
        const T *p2;
        bool predict = true;                              // false: no prediction ( cut2, cuto stay 0: the caller predicts )
        alignas( 64 ) T zb[ 4 ][ B ];                    // the box, absolute: x min, x max, y min, y max
        U inv, cut2;                                      // bit x: the cell is inverted / a seed of ring 2 cuts it
        U cuto[ 16 ];                                     // bit x: the seed o of ring 2 ( `r2x`, `r2y` ) cuts the cell
        /// the seeds of ring 2 that cut the cell x ( bit o )
        unsigned short cuts( int x ) const { unsigned m = 0; for( int o = 0; o < 16; ++o ) m |= unsigned( ( cuto[ o ] >> x ) & 1 ) << o; return m; }
    };

    /// the vertices where the support in the direction ( kx, ky ) is reached ( bit k: vertex k ): the vertex k between the directions
    /// k and k + 1 strictly around it, or both ends of the edge k parallel to it
    static constexpr unsigned sup_mask( int kx, int ky ) {
        unsigned m = 0;
        for( int j = 0; j < D; ++j ) {
            const int n = ( j + 1 ) % D;
            const double cj = Tb::dir_x[ j ] * ky - Tb::dir_y[ j ] * kx, dj = Tb::dir_x[ j ] * kx + Tb::dir_y[ j ] * ky;
            const double cn = kx * Tb::dir_y[ n ] - ky * Tb::dir_x[ n ];
            if( cj == 0 && dj > 0 ) m |= ( 1u << ( ( j + D - 1 ) % D ) ) | ( 1u << j );
            else if( cj > 0 && cn > 0 ) m |= 1u << j;
        }
        return m;
    }
    static constexpr int r2x[ 16 ] = { -2, -2, -2, -2, -2, -1, -1, 0, 0, 1, 1, 2, 2, 2, 2, 2 };
    static constexpr int r2y[ 16 ] = { -2, -1, 0, 1, 2, -2, 2, -2, 2, -2, 2, -2, -1, 0, 1, 2 };

    // max over the vertices of m of kx X + ky Y ( kx, ky folded )
    template<int kx,int ky,unsigned m,int k=0> [[gnu::always_inline]] static V support( const V ( &X )[ D ], const V ( &Y )[ D ] ) {
        constexpr int f = std::countr_zero( m >> k ) + k;
        const V v = mulc<double( kx )>( X[ f ] ) + mulc<double( ky )>( Y[ f ] );
        if constexpr ( ( m >> ( f + 1 ) ) != 0 ) return asimd::max( v, support<kx,ky,m,f+1>( X, Y ) );
        else return v;
    }

    // x times a constant of the tables, folded when it is 0, 1 or -1
    template<double c> [[gnu::always_inline]] static V mulc( const V &x ) {
        if constexpr ( c == 0 ) return V( T( 0 ) );
        else if constexpr ( c == 1 ) return x;
        else if constexpr ( c == -1 ) return V( T( 0 ) ) - x;
        else return V( T( c ) ) * x;
    }

    // ht[ k ] = min( h[ k ], lam h[ j ] + mu h[ l ] over the pairs of k )
    template<int k,int p> [[gnu::always_inline]] static void tighten( const V ( &h )[ D ], V &t ) {
        if constexpr ( p < Tb::pair_beg[ k + 1 ] ) {
            const V c = mulc<Tb::pair_lam[ p ]>( h[ Tb::pair_j[ p ] ] ) + mulc<Tb::pair_mu[ p ]>( h[ Tb::pair_l[ p ] ] );
            t = asimd::min( t, c );
            tighten<k,p+1>( h, t );
        }
    }

    // one vector of W cells: L, C, R the psi of the rows a - 1, a, a + 1 at columns b - 1, b, b + 1 ( lane by lane )
    // ( `q`: with an `Ext`, the rows a - 2 .. a + 2 of `p2` at the column of the first lane of this vector )
    [[gnu::always_inline]] static void cells( int N, int a, int bv, const V ( &L )[ 3 ], const V ( &C )[ 3 ], const V ( &R )[ 3 ],
                                              T ( &vx )[ D ][ B ], T ( &vy )[ D ][ B ], unsigned long long ( &ok )[ D ], T *ar, int j,
                                              Ext *ex = nullptr, const T *const *q = nullptr ) {
        const T inf = std::numeric_limits<T>::infinity();
        const V pi = C[ 1 ], half( T( 0.5 ) );
        // the psi of the neighbour of each direction ( off_x: the row, off_y: the column )
        const V pj[ D ] = { L[ 0 ], L[ 1 ], L[ 2 ], C[ 2 ], R[ 2 ], R[ 1 ], R[ 0 ], C[ 0 ] };
        V h[ D ];
        [&]<int... k>( std::integer_sequence<int,k...> ) SDOT_INLINE_LAMBDA {
            ( ( h[ k ] = ( V( T( Tb::off_sqn[ k ] ) ) + ( pi - pj[ k ] ) ) * half ), ... );
            ( ( ok[ k ] |= ( unsigned long long )( asimd::to_bits( h[ k ] < V( inf ) ) ) << ( j * W ) ), ... );
        }( std::make_integer_sequence<int,D>() );
        // the box ( it wins ties, as in `row_block` )
        const V bb = V::iota( T( bv ) );
        const V hb[ 4 ] = { V( T( N ) - T( 0.5 ) - T( a ) ), V( T( a ) + T( 0.5 ) ), V( T( N ) - T( 0.5 ) ) - bb, bb + V( T( 0.5 ) ) };
        constexpr int bd[ 4 ] = { Tb::dir_px, Tb::dir_mx, Tb::dir_py, Tb::dir_my };
        const unsigned long long lanes = ( ( W == 64 ? ~0ull : ( 1ull << W ) - 1 ) ) << ( j * W );
        for( int q = 0; q < 4; ++q ) {
            ok[ bd[ q ] ] &= ~lanes | ( ( unsigned long long )( asimd::to_bits( h[ bd[ q ] ] < hb[ q ] ) ) << ( j * W ) );
            h[ bd[ q ] ] = asimd::min( h[ bd[ q ] ], hb[ q ] );
        }
        // the tightening, the vertices, the area
        V ht[ D ], X[ D ], Y[ D ];
        [&]<int... k>( std::integer_sequence<int,k...> ) SDOT_INLINE_LAMBDA {
            ( ( ht[ k ] = h[ k ], tighten<k,Tb::pair_beg[ k ]>( h, ht[ k ] ) ), ... );
            ( ( X[ k ] = mulc<Tb::vinv[ 4 * k + 0 ]>( ht[ k ] ) + mulc<Tb::vinv[ 4 * k + 1 ]>( ht[ ( k + 1 ) % D ] ),
                Y[ k ] = mulc<Tb::vinv[ 4 * k + 2 ]>( ht[ k ] ) + mulc<Tb::vinv[ 4 * k + 3 ]>( ht[ ( k + 1 ) % D ] ) ), ... );
        }( std::make_integer_sequence<int,D>() );
        V s( T( 0 ) );
        [&]<int... k>( std::integer_sequence<int,k...> ) SDOT_INLINE_LAMBDA {
            ( ( s = s + ( X[ ( k + D - 1 ) % D ] * Y[ k ] - Y[ ( k + D - 1 ) % D ] * X[ k ] ) ), ... );
            ( ( X[ k ].store_aligned( vx[ k ] + j * W ), Y[ k ].store_aligned( vy[ k ] + j * W ) ), ... );
        }( std::make_integer_sequence<int,D>() );
        ( s * half ).store_aligned( ar + j * W );
        if( ex ) extra( a, bv, pi, X, Y, *ex, q, j );
    }

    // the box, the inversion test and the prediction of ring 2, from the vertices in registers
    [[gnu::always_inline]] static void extra( int a, int bv, const V &pi, const V ( &X )[ D ], const V ( &Y )[ D ], Ext &ex, const T *const *q, int j ) {
        // the box: at the two ends of the edges of the box directions
        const V aa = V( T( a ) ), bb = V::iota( T( bv ) );
        constexpr int mx = Tb::dir_mx, px = Tb::dir_px, my = Tb::dir_my, py = Tb::dir_py;
        const V x0 = asimd::min( X[ ( mx + D - 1 ) % D ], X[ mx ] ), x1 = asimd::max( X[ ( px + D - 1 ) % D ], X[ px ] );
        const V y0 = asimd::min( Y[ ( my + D - 1 ) % D ], Y[ my ] ), y1 = asimd::max( Y[ ( py + D - 1 ) % D ], Y[ py ] );
        ( x0 + aa ).store_aligned( ex.zb[ 0 ] + j * W ); ( x1 + aa ).store_aligned( ex.zb[ 1 ] + j * W );
        ( y0 + bb ).store_aligned( ex.zb[ 2 ] + j * W ); ( y1 + bb ).store_aligned( ex.zb[ 3 ] + j * W );
        // inverted: an edge k going backward, g = n_k^perp . ( v_k - v_( k - 1 ) ) < -1e-9 scale ( the scale from the box )
        const V zero( T( 0 ) );
        const V sc = asimd::max( V( T( 1 ) ), asimd::max( zero - x0, x1 ) + asimd::max( zero - y0, y1 ) );
        V mg = zero;
        [&]<int... k>( std::integer_sequence<int,k...> ) SDOT_INLINE_LAMBDA {
            ( ( mg = asimd::min( mg, mulc<-Tb::dir_y[ k ]>( X[ k ] - X[ ( k + D - 1 ) % D ] ) + mulc<Tb::dir_x[ k ]>( Y[ k ] - Y[ ( k + D - 1 ) % D ] ) ) ), ... );
        }( std::make_integer_sequence<int,D>() );
        ex.inv |= U( asimd::to_bits( mg < V( T( -1e-9 ) ) * sc ) ) << ( j * W );
        if( ! ex.predict ) return;
        // the prediction: a seed j of ring 2 cuts the cell iff 2 h( k ) > |k|^2 + psi_i - psi_j ( exact for ring 2 )
        V c2 = zero, one( T( 1 ) );
        [&]<int... o>( std::integer_sequence<int,o...> ) SDOT_INLINE_LAMBDA {
            ( ( [&]() SDOT_INLINE_LAMBDA {
                constexpr int kx = r2x[ o ], ky = r2y[ o ];
                const V h = support<kx,ky,sup_mask( kx, ky )>( X, Y );
                const V pj = V::load_unaligned( q[ kx + 2 ] + j * W + ky );
                ex.cuto[ o ] |= U( asimd::to_bits( h + h > V( T( kx * kx + ky * ky ) ) + ( pi - pj ) ) ) << ( j * W );
                c2 = asimd::select( h + h > V( T( kx * kx + ky * ky ) ) + ( pi - pj ), one, c2 );
            }() ), ... );
        }( std::make_integer_sequence<int,16>() );
        ex.cut2 |= U( asimd::to_bits( c2 > zero ) ) << ( j * W );
    }

    /// the cells ( a, b0 .. b0 + B - 1 ); b0 + B <= N ( a shorter block: `block_tail` )
    static void block( int N, const T *psi_pad, int a, int b0, T ( &vx )[ D ][ B ], T ( &vy )[ D ][ B ], unsigned long long ( &ok )[ D ], T *ar, Ext *ex = nullptr ) {
        const int P = N + 2;
        const T *r[ 3 ] = { psi_pad + size_t( a ) * P + b0 + 1, psi_pad + size_t( a + 1 ) * P + b0 + 1, psi_pad + size_t( a + 2 ) * P + b0 + 1 };
        for( int k = 0; k < D; ++k ) ok[ k ] = 0;
        if( ex ) {
            const int P2 = N + 4;
            const T *q[ 5 ];
            for( int d = 0; d < 5; ++d ) q[ d ] = ex->p2 + size_t( a + d ) * P2 + b0 + 2;
            ex->inv = ex->cut2 = 0; for( auto &c : ex->cuto ) c = 0;
            rows( N, r, a, b0, vx, vy, ok, ar, ex, q );
        } else if constexpr ( ROT ) {
            // the vectors of the three rows roll: prev, cur, next; only next is loaded ( the halos: one scalar each end )
            V pv[ 3 ], cu[ 3 ], nx[ 3 ];
            for( int d = 0; d < 3; ++d ) { pv[ d ] = V( r[ d ][ -1 ] ); cu[ d ] = V::load_unaligned( r[ d ] ); }
            [&]<int... j>( std::integer_sequence<int,j...> ) SDOT_INLINE_LAMBDA {
                ( ( [&]() SDOT_INLINE_LAMBDA {
                    for( int d = 0; d < 3; ++d ) nx[ d ] = j + 1 < NV ? V::load_unaligned( r[ d ] + ( j + 1 ) * W ) : V( r[ d ][ B ] );
                    V L[ 3 ], R[ 3 ];
                    for( int d = 0; d < 3; ++d ) { L[ d ] = asimd::ext_lanes( pv[ d ], cu[ d ], asimd::N<W-1>() ); R[ d ] = asimd::ext_lanes( cu[ d ], nx[ d ], asimd::N<1>() ); }
                    cells( N, a, b0 + j * W, L, cu, R, vx, vy, ok, ar, j );
                    for( int d = 0; d < 3; ++d ) { pv[ d ] = cu[ d ]; cu[ d ] = nx[ d ]; }
                }() ), ... );
            }( std::make_integer_sequence<int,NV>() );
        } else
            rows( N, r, a, b0, vx, vy, ok, ar );
    }

    /// a block of n < B cells at the end of a row: the three rows copied with their -inf padding ( lanes past n: -inf seeds )
    static void block_tail( int N, const T *psi_pad, int a, int b0, int n, T ( &vx )[ D ][ B ], T ( &vy )[ D ][ B ], unsigned long long ( &ok )[ D ], T *ar, Ext *ex = nullptr ) {
        const T inf = std::numeric_limits<T>::infinity();
        const int P = N + 2, S = B + 2 * W;
        alignas( 64 ) T loc[ 3 * S ];
        for( int q = 0; q < 3 * S; ++q ) loc[ q ] = -inf;
        const T *r[ 3 ];
        for( int d = 0; d < 3; ++d ) {
            for( int x = -1; x <= n; ++x ) loc[ d * S + W + x ] = psi_pad[ size_t( a + d ) * P + b0 + 1 + x ];
            r[ d ] = loc + d * S + W;
        }
        for( int k = 0; k < D; ++k ) ok[ k ] = 0;
        if( ! ex ) { rows( N, r, a, b0, vx, vy, ok, ar ); return; }
        // ( and the five rows of p2, padded the same way )
        const int P2 = N + 4;
        alignas( 64 ) T loc2[ 5 * S ];
        for( int q = 0; q < 5 * S; ++q ) loc2[ q ] = -inf;
        const T *q[ 5 ];
        for( int d = 0; d < 5; ++d ) {
            for( int x = -2; x <= n + 1; ++x ) loc2[ d * S + W + x ] = ex->p2[ size_t( a + d ) * P2 + b0 + 2 + x ];
            q[ d ] = loc2 + d * S + W;
        }
        ex->inv = ex->cut2 = 0; for( auto &c : ex->cuto ) c = 0;
        rows( N, r, a, b0, vx, vy, ok, ar, ex, q );
    }

private:
    // the cells of the block from the three rows ( pointers on the column of the first cell ), unaligned loads
    static void rows( int N, const T *const ( &r )[ 3 ], int a, int b0, T ( &vx )[ D ][ B ], T ( &vy )[ D ][ B ], unsigned long long ( &ok )[ D ], T *ar,
                      Ext *ex = nullptr, const T *const *q = nullptr ) {
        [&]<int... j>( std::integer_sequence<int,j...> ) SDOT_INLINE_LAMBDA {
            ( ( [&]() SDOT_INLINE_LAMBDA {
                V L[ 3 ], C[ 3 ], R[ 3 ];
                for( int d = 0; d < 3; ++d ) { L[ d ] = V::load_unaligned( r[ d ] + j * W - 1 ); C[ d ] = V::load_unaligned( r[ d ] + j * W ); R[ d ] = V::load_unaligned( r[ d ] + j * W + 1 ); }
                cells( N, a, b0 + j * W, L, C, R, vx, vy, ok, ar, j, ex, q );
            }() ), ... );
        }( std::make_integer_sequence<int,NV>() );
    }
};

} // namespace sdot::grid
