#pragma once

#include <asimd/asimd.h>
#include <algorithm>
#include <bit>
#include <limits>
#include <utility>
#include "StencilTables.h"

namespace sdot::grid {

/// RING 2 BY CLIPPING ( 2D, unrolled, asimd ): the ring-1 cells of a batch ( `Ring1Reg`, the cells the prediction sent to ring 2 )
/// cut by the seeds of ring 2 the prediction flagged, and only those, instead of the 336 pairs of the tightening of `StencilKernel<2>`.
/// The output is that of `StencilKernel<2>::cells_verts`: 16 vertex slots ( vertex k between the directions k and k + 1 of
/// `StencilTables<2>`, edge k from vertex k - 1 to vertex k ), the owner of each edge ( an offset of `StencilTables<2>`, -1: the box or
/// nothing ), the area.
///
/// The ring-1 cell in the 16 slots: the directions of ring 1 are the odd ones ( dir k of ring 1 = dir 2 k + 1 ), so the ring-1 vertex k
/// fills the slots 2 k + 1 and 2 k + 2 ( the edge 2 k + 2 between them has length 0 ). A seed o of ring 2, direction k, half-plane
/// o . x <= h: the slots outside form an arc around its support ( the slots k - 1 and k ) and the slot k + 8, opposite, stays inside
/// ( else the cell is empty ). The outside slots before k collapse onto A, where the boundary leaves the half-plane, those from k on
/// onto B, where it comes back: which side a slot is on is known at compile time, A and B come from two sweeps of selects ( last
/// inside, first outside ), one division each.
template<int B, class T>
struct Ring2Clip {
    using T1 = StencilTables<1>;
    using T2 = StencilTables<2>;
    static constexpr int D = 16;
    static constexpr int W = asimd::SimdSize<T>::value;
    static_assert( B % W == 0 && B <= 64, "Ring2Clip: B a multiple of the SIMD width, at most 64" );
    static constexpr int NV = B / W;
    using V = asimd::SimdVec<T,W>;
    using U = unsigned long long;

    /// the cells waiting for ring 2, with their ring-1 polygon ( `Ring1Reg`'s vertices ), the edges of ring 1 with a seed ( bit k ) and
    /// the seeds of ring 2 that cut ( bit o, the order of `r2x`, `r2y` )
    struct Pending {
        alignas( 64 ) T vx[ 8 ][ B ], vy[ 8 ][ B ];
        int cl[ B ];
        unsigned char ok[ B ];
        unsigned short cut[ B ];
        int n = 0;
    };

    static constexpr int r2x[ 16 ] = { -2, -2, -2, -2, -2, -1, -1, 0, 0, 1, 1, 2, 2, 2, 2, 2 };
    static constexpr int r2y[ 16 ] = { -2, -1, 0, 1, 2, -2, 2, -2, 2, -2, 2, -2, -1, 0, 1, 2 };
    static constexpr int off_index( int ox, int oy ) {
        for( int i = 0; i < T2::nb_offsets; ++i ) if( T2::off_x[ i ] == ox && T2::off_y[ i ] == oy ) return i;
        return -1;
    }

    /// the cells of `pd`, `p2`: psi padded by two cells of -inf ( N + 4 columns ). `empty`: bit x, the cell x is empty ( every slot cut )
    static void cells_verts( int N, const T *p2, const Pending &pd, T ( &vx )[ D ][ B ], T ( &vy )[ D ][ B ], signed char ( &own )[ D ][ B ], T *ar, U &empty ) {
        const int P2 = N + 4, n = pd.n;
        alignas( 64 ) T pi[ B ], pj[ 16 ][ B ];
        for( int x = 0; x < B; ++x ) {
            const int c = pd.cl[ x < n ? x : n - 1 ];
            const size_t o = size_t( c / N + 2 ) * P2 + c % N + 2;
            pi[ x ] = p2[ o ];
            for( int q = 0; q < 16; ++q ) pj[ q ][ x ] = p2[ o + r2x[ q ] * P2 + r2y[ q ] ];
            for( int k = 0; k < 8; ++k ) {
                const int xx = x < n ? x : n - 1;
                own[ 2 * k + 1 ][ x ] = ( pd.ok[ xx ] >> k ) & 1 ? off_index( T1::off_x[ k ], T1::off_y[ k ] ) : -1;
                own[ 2 * k + 2 - ( k == 7 ? 16 : 0 ) ][ x ] = -1;
            }
        }
        empty = 0;
        for( int j = 0; j < NV; ++j ) {
            const int o = j * W;
            V X[ D ], Y[ D ];
            for( int k = 0; k < 8; ++k ) {
                X[ 2 * k + 1 ] = X[ ( 2 * k + 2 ) % D ] = V::load_aligned( pd.vx[ k ] + o );
                Y[ 2 * k + 1 ] = Y[ ( 2 * k + 2 ) % D ] = V::load_aligned( pd.vy[ k ] + o );
            }
            unsigned m16 = 0;
            for( int x = o; x < o + W; ++x ) m16 |= x < n ? pd.cut[ x ] : 0;
            const V vpi = V::load_aligned( pi + o );
            U all_out = 0;
            [&]<int... q>( std::integer_sequence<int,q...> ) {
                ( ( ( m16 >> q ) & 1 ? clip<q>( X, Y, vpi, V::load_aligned( pj[ q ] + o ), own, o, all_out ) : void() ), ... );
            }( std::make_integer_sequence<int,16>() );
            empty |= all_out << o;
            V s( T( 0 ) );
            [&]<int... k>( std::integer_sequence<int,k...> ) {
                ( ( s = s + ( X[ ( k + D - 1 ) % D ] * Y[ k ] - Y[ ( k + D - 1 ) % D ] * X[ k ] ) ), ... );
                ( ( X[ k ].store_aligned( vx[ k ] + o ), Y[ k ].store_aligned( vy[ k ] + o ) ), ... );
            }( std::make_integer_sequence<int,D>() );
            ( s * V( T( 0.5 ) ) ).store_aligned( ar + o );
        }
    }

    // the cut by the seed q of ring 2 ( offset ( ox, oy ), direction k ): o . x <= h, h = ( |o|^2 + psi_i - psi_j ) / 2
    template<int q> [[gnu::always_inline]] static void clip( V ( &X )[ D ], V ( &Y )[ D ], const V &pi, const V &pj, signed char ( &own )[ D ][ B ], int o, U &all_out ) {
        constexpr int ox = r2x[ q ], oy = r2y[ q ], oi = off_index( ox, oy ), k = T2::off_dir[ oi ];
        const V h = ( V( T( ox * ox + oy * oy ) ) + ( pi - pj ) ) * V( T( 0.5 ) );
        V d[ D ];
        [&]<int... m>( std::integer_sequence<int,m...> ) {
            ( ( d[ m ] = V( T( ox ) ) * X[ m ] + V( T( oy ) ) * Y[ m ] ), ... );
        }( std::make_integer_sequence<int,D>() );
        // A: on the edge from the last inside slot to the first outside one, among k - 8 .. k - 1
        V ix = X[ ( k + 8 ) % D ], iy = Y[ ( k + 8 ) % D ], id = d[ ( k + 8 ) % D ];
        V ux = X[ ( k + D - 1 ) % D ], uy = Y[ ( k + D - 1 ) % D ], ud = d[ ( k + D - 1 ) % D ];
        [&]<int... r>( std::integer_sequence<int,r...> ) {
            ( ( [&] { constexpr int m = ( k + 9 + r ) % D;               // k - 7 .. k - 1, increasing: the last inside
                ix = asimd::select( d[ m ] > h, ix, X[ m ] ); iy = asimd::select( d[ m ] > h, iy, Y[ m ] ); id = asimd::select( d[ m ] > h, id, d[ m ] ); }() ), ... );
            ( ( [&] { constexpr int m = ( k + D - 1 - r ) % D;           // k - 1 .. k - 7, decreasing: the first outside
                ux = asimd::select( d[ m ] > h, X[ m ], ux ); uy = asimd::select( d[ m ] > h, Y[ m ], uy ); ud = asimd::select( d[ m ] > h, d[ m ], ud ); }() ), ... );
        }( std::make_integer_sequence<int,7>() );
        const V zero( T( 0 ) );
        const V da = ud - id, ta = asimd::select( da > zero, ( h - id ) / da, zero );
        const V ax = ix + ta * ( ux - ix ), ay = iy + ta * ( uy - iy );
        // B: on the edge from the last outside slot to the first inside one, among k .. k + 7
        V jx = X[ ( k + 8 ) % D ], jy = Y[ ( k + 8 ) % D ], jd = d[ ( k + 8 ) % D ];
        V vx_ = X[ k ], vy_ = Y[ k ], vd = d[ k ];
        [&]<int... r>( std::integer_sequence<int,r...> ) {
            ( ( [&] { constexpr int m = ( k + 7 - r ) % D;               // k + 7 .. k + 1, decreasing: the first inside
                jx = asimd::select( d[ m ] > h, jx, X[ m ] ); jy = asimd::select( d[ m ] > h, jy, Y[ m ] ); jd = asimd::select( d[ m ] > h, jd, d[ m ] ); }() ), ... );
            ( ( [&] { constexpr int m = ( k + 1 + r ) % D;               // k + 1 .. k + 7, increasing: the last outside
                vx_ = asimd::select( d[ m ] > h, X[ m ], vx_ ); vy_ = asimd::select( d[ m ] > h, Y[ m ], vy_ ); vd = asimd::select( d[ m ] > h, d[ m ], vd ); }() ), ... );
        }( std::make_integer_sequence<int,7>() );
        const V db = vd - jd, tb = asimd::select( db > zero, ( vd - h ) / db, zero );
        const V bx = vx_ + tb * ( jx - vx_ ), by = vy_ + tb * ( jy - vy_ );
        // the cut edge gets the seed, the cells all outside are empty
        const U lanes = W == 64 ? ~0ull : ( 1ull << W ) - 1;
        const U cut = U( asimd::to_bits( asimd::max( d[ ( k + D - 1 ) % D ], d[ k ] ) > h ) ) & lanes;
        for( U m = cut; m; m &= m - 1 ) own[ k ][ o + std::countr_zero( m ) ] = oi;
        V dmin = d[ 0 ];
        [&]<int... m>( std::integer_sequence<int,m...> ) { ( ( dmin = asimd::min( dmin, d[ m + 1 ] ) ), ... ); }( std::make_integer_sequence<int,D-1>() );
        all_out |= U( asimd::to_bits( dmin > h ) ) & lanes;
        // the outside slots collapse
        [&]<int... r>( std::integer_sequence<int,r...> ) {
            ( ( [&] { constexpr int m = ( k + 8 + r ) % D;               // k - 8 .. k - 1: onto A
                X[ m ] = asimd::select( d[ m ] > h, ax, X[ m ] ); Y[ m ] = asimd::select( d[ m ] > h, ay, Y[ m ] ); }() ), ... );
            ( ( [&] { constexpr int m = ( k + r ) % D;                   // k .. k + 7: onto B
                X[ m ] = asimd::select( d[ m ] > h, bx, X[ m ] ); Y[ m ] = asimd::select( d[ m ] > h, by, Y[ m ] ); }() ), ... );
        }( std::make_integer_sequence<int,8>() );
    }
};

} // namespace sdot::grid
