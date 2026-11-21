#pragma once

// ( the lambdas of the unrolled folds: clang leaves them as calls once their body is large -- closures and arrays in memory, Ring1Reg with
// its extras 41 -> 14 ns/cell on an M4 when forced )
#ifndef SDOT_INLINE_LAMBDA
#define SDOT_INLINE_LAMBDA __attribute__( ( always_inline ) )
#endif

#include <asimd/asimd.h>
#include <atomic>
#include <bit>
#include <limits>
#include <utility>
#include "StencilTables.h"

namespace sdot::grid {

/// RING 2 OF A LIST OF CELLS, IN REGISTERS ( 2D, unrolled, asimd ): what `StencilKernel<2>::cells_verts` computes, in the same
/// operations ( so the same values ), with every table of `StencilTables<2>` folded into the code.
///
/// Each of the 16 directions once: h_k the min over the seeds of that direction ( ( |o|^2 + psi_i - psi_j ) / ( 2 m ), o = m n_k ),
/// the owner of the edge the seed that wins, the box on the 4 axis directions ( it wins ties ); then one pass of tightening on the raw
/// h ( exact: an optimal LP basis has 2 elements ), ht_k = min( h_k, lam h_j + mu h_l over the 21 pairs around k ), the vertices from
/// consecutive ht, the area.
///
/// The cells come as a list ( flat indices a * N + b, 1 <= n <= B, lanes past n repeat the last cell ): psi gathered per lane once.
///
/// ( `GATHER`: the psi by AVX2 gathers rather than copied lane by lane; `TIGHT`: 1 the 21 pairs, 0 no tightening ( to time it apart ),
/// 100 + P: P sweeps on the neighbours only ( ht_k = min( ht_k, lam ht_{k-1} + mu ht_{k+1} ), in place ), 200 + S: the pairs of span
/// <= S only ( the last two are experiments, not exact in general ), 300 + S: the pairs of span <= S then a certificate, the 21
/// pairs for the vectors it rejects, 400: `sweep` then the certificate )
template<int B, class T, bool GATHER = false, int TIGHT = 1>
struct Ring2Reg {
    using Tb = StencilTables<2>;
    static constexpr int D = 16, NO = Tb::nb_offsets;
    static constexpr int W = asimd::SimdSize<T>::value;
    static_assert( B % W == 0 && B <= 64, "Ring2Reg: B a multiple of the SIMD width, at most 64" );
    static constexpr int NV = B / W;
    using V = asimd::SimdVec<T,W>;

    template<double c> [[gnu::always_inline]] static V mulc( const V &x ) {
        if constexpr ( c == 0 ) return V( T( 0 ) );
        else if constexpr ( c == 1 ) return x;
        else if constexpr ( c == -1 ) return V( T( 0 ) ) - x;
        else return V( T( c ) ) * x;
    }

    // the span of the pair p of k: a + b, its directions k - a and k + b
    static constexpr int span( int k, int p ) {
        int a = ( k - Tb::pair_j[ p ] + D ) % D, b = ( Tb::pair_l[ p ] - k + D ) % D;
        if ( a > D / 2 ) { a = ( k - Tb::pair_l[ p ] + D ) % D; b = ( Tb::pair_j[ p ] - k + D ) % D; }
        return a + b;
    }
    // the pair of k made of its two neighbours k - 1, k + 1
    static inline std::atomic<long> fallbacks = 0;      // vectors that went back to the 21 pairs ( TIGHT >= 300 )
    static inline bool diag = false;                    // ( count the lanes below in the fallbacks: failed the certificate, changed by
    static inline std::atomic<long> diag_failed = 0, diag_changed = 0, diag_inverted = 0; // the 21 pairs, inverted with them too )

    static constexpr int npair( int k ) { for( int p = Tb::pair_beg[ k ]; p < Tb::pair_beg[ k + 1 ]; ++p ) if( span( k, p ) == 2 ) return p; return -1; }

    // ht[ k ] = min( h[ k ], lam h[ j ] + mu h[ l ] over the pairs of k ) ( `S` > 0: only the pairs of span <= S )
    template<int S,int k,int p> [[gnu::always_inline]] static void tighten( const V ( &h )[ D ], V &t ) {
        if constexpr ( p < Tb::pair_beg[ k + 1 ] ) {
            if constexpr ( S == 0 || span( k, p ) <= S )
                t = asimd::min( t, mulc<Tb::pair_lam[ p ]>( h[ Tb::pair_j[ p ] ] ) + mulc<Tb::pair_mu[ p ]>( h[ Tb::pair_l[ p ] ] ) );
            tighten<S,k,p+1>( h, t );
        }
    }

    /// ht by a sweep that keeps the last active lines ( Graham on the half-planes sorted by angle ): a stack of the last `SD` lines
    /// pushed, each with its vertex with the line below; line l pops the top while that vertex is outside n_l . x <= h_l ( at most
    /// `SP` pops, by selects ), then is pushed with its vertex with the new top, kept in s_l. 1.5 turn so that the start is forgotten.
    /// Then a backward fill: W the vertex of the next active line, direction j active iff n_j . W >= h_j ( then W = s_j ),
    /// ht_j = min( h_j, n_j . W ). Not exact when the stack is too short: the certificate checks it
    static constexpr int SD = 4, SP = 3;
    [[gnu::always_inline]] static void sweep( const V ( &h )[ D ], V ( &ht )[ D ] ) {
        const V big( T( 1e30 ) ), zero( T( 0 ) ), one( T( 1 ) );
        V hs[ D ];
        for( int k = 0; k < D; ++k ) hs[ k ] = asimd::min( h[ k ], big );
        V ex[ SD ], ey[ SD ], eh[ SD ], evx[ SD ], evy[ SD ], sx[ D ], sy[ D ], cnt = one;
        for( int i = 0; i < SD; ++i ) { ex[ i ] = V( T( Tb::dir_x[ 0 ] ) ); ey[ i ] = V( T( Tb::dir_y[ 0 ] ) ); eh[ i ] = hs[ 0 ]; evx[ i ] = evy[ i ] = zero; }
        [&]<int... t>( std::integer_sequence<int,t...> ) SDOT_INLINE_LAMBDA {
            ( ( [&]() SDOT_INLINE_LAMBDA {
                constexpr int l = ( t + 1 ) % D;
                const V lx = V( T( Tb::dir_x[ l ] ) ), ly = V( T( Tb::dir_y[ l ] ) );
                for( int r = 0; r < SP; ++r ) {
                    const auto pop = asimd::min( lx * evx[ 0 ] + ly * evy[ 0 ] - hs[ l ], cnt - V( T( 1.5 ) ) ) > zero;
                    for( int i = 0; i + 1 < SD; ++i ) {
                        ex[ i ] = asimd::select( pop, ex[ i + 1 ], ex[ i ] ); ey[ i ] = asimd::select( pop, ey[ i + 1 ], ey[ i ] ); eh[ i ] = asimd::select( pop, eh[ i + 1 ], eh[ i ] );
                        evx[ i ] = asimd::select( pop, evx[ i + 1 ], evx[ i ] ); evy[ i ] = asimd::select( pop, evy[ i + 1 ], evy[ i ] );
                    }
                    cnt = asimd::select( pop, cnt - one, cnt );
                }
                const V det = ex[ 0 ] * ly - ey[ 0 ] * lx;
                const V X = ( eh[ 0 ] * ly - ey[ 0 ] * hs[ l ] ) / det, Y = ( ex[ 0 ] * hs[ l ] - lx * eh[ 0 ] ) / det;
                for( int i = SD - 1; i > 0; --i ) { ex[ i ] = ex[ i - 1 ]; ey[ i ] = ey[ i - 1 ]; eh[ i ] = eh[ i - 1 ]; evx[ i ] = evx[ i - 1 ]; evy[ i ] = evy[ i - 1 ]; }
                ex[ 0 ] = lx; ey[ 0 ] = ly; eh[ 0 ] = hs[ l ]; evx[ 0 ] = X; evy[ 0 ] = Y;
                cnt = asimd::min( cnt + one, V( T( SD ) ) );
                sx[ l ] = X; sy[ l ] = Y;
            }() ), ... );
        }( std::make_integer_sequence<int,D+D/2>() );
        V wx = sx[ D / 2 ], wy = sy[ D / 2 ];
        [&]<int... t>( std::integer_sequence<int,t...> ) SDOT_INLINE_LAMBDA {
            ( ( [&]() SDOT_INLINE_LAMBDA {
                constexpr int j = ( D + D / 2 - 1 - t ) % D;
                const V v = mulc<Tb::dir_x[ j ]>( wx ) + mulc<Tb::dir_y[ j ]>( wy );
                const auto act = v - hs[ j ] + V( T( 1e-12 ) ) * ( one + asimd::max( v, zero - v ) ) > zero;
                wx = asimd::select( act, sx[ j ], wx ); wy = asimd::select( act, sy[ j ], wy );
                ht[ j ] = asimd::min( h[ j ], mulc<Tb::dir_x[ j ]>( wx ) + mulc<Tb::dir_y[ j ]>( wy ) );
            }() ), ... );
        }( std::make_integer_sequence<int,D+D/2>() );
    }

    // the seeds of each direction ( 1 or 2 ), in the order of the table ( a later one wins only if strictly smaller, as in `K2` )
    static constexpr int seed( int d, int r ) { for( int q = 0; q < NO; ++q ) if( Tb::off_dir[ q ] == d && r-- == 0 ) return q; return -1; }

    /// `psi_pad`: psi padded by two cells of -inf ( N + 4 columns ). Outputs as `StencilKernel<2>::cells_verts`
    ///
    /// ( registers: the 16 h stay live through the tightening; the owners leave as bit masks at once, decoded at the end, and the psi
    /// come by gathers, so that the 16 ymm of AVX2 hold the h and the running min )
    static void cells_verts( int N, const T *psi_pad, const int *cells, int n, T ( &vx )[ D ][ B ], T ( &vy )[ D ][ B ], signed char ( &own )[ D ][ B ], T *ar ) {
        using VI = asimd::SimdVec<int,W>;
        using U = unsigned long long;
        const int P = N + 4;
        alignas( 64 ) int base[ B ];
        alignas( 64 ) T ca[ B ], cb[ B ], pi[ GATHER ? 1 : B ], pj[ GATHER ? 1 : NO ][ GATHER ? 1 : B ];
        for( int x = 0; x < B; ++x ) {
            const int i = cells[ x < n ? x : n - 1 ];
            ca[ x ] = T( i / N ); cb[ x ] = T( i % N );
            base[ x ] = ( i / N + 2 ) * P + i % N + 2;
            if constexpr ( ! GATHER ) {
                pi[ x ] = psi_pad[ base[ x ] ];
                [&]<int... q>( std::integer_sequence<int,q...> ) SDOT_INLINE_LAMBDA {
                    ( ( pj[ q ][ x ] = psi_pad[ base[ x ] + Tb::off_x[ q ] * P + Tb::off_y[ q ] ] ), ... );
                }( std::make_integer_sequence<int,NO>() );
            }
        }
        U fin[ D ] = {}, sec[ D ] = {};                  // bit x: the direction has a finite h / its second seed wins ( box: fin cleared )
        for( int j = 0; j < NV; ++j ) {
            const int o = j * W;
            const VI ind = VI::load_aligned( base + o );
            const V half( T( 0.5 ) );
            V vpi;
            if constexpr ( GATHER ) vpi = V::gather( psi_pad, ind ); else vpi = V::load_aligned( pi + o );
            auto hq = [&]<int q>() {
                V pjq;
                if constexpr ( GATHER ) pjq = V::gather( psi_pad + Tb::off_x[ q ] * P + Tb::off_y[ q ], ind ); else pjq = V::load_aligned( pj[ q ] + o );
                return ( V( T( Tb::off_sqn[ q ] ) ) + ( vpi - pjq ) ) * V( T( 0.5 / Tb::off_m[ q ] ) );
            };
            // each direction once: the min over its seeds
            V h[ D ];
            [&]<int... d>( std::integer_sequence<int,d...> ) SDOT_INLINE_LAMBDA {
                ( ( [&]() SDOT_INLINE_LAMBDA {
                    constexpr int q0 = seed( d, 0 ), q1 = seed( d, 1 );
                    h[ d ] = hq.template operator()<q0>();
                    if constexpr ( q1 >= 0 ) {
                        const V c = hq.template operator()<q1>();
                        sec[ d ] |= U( asimd::to_bits( h[ d ] > c ) ) << o;
                        h[ d ] = asimd::min( h[ d ], c );
                    }
                    fin[ d ] |= U( asimd::to_bits( V( std::numeric_limits<T>::infinity() ) > h[ d ] ) ) << o;
                }() ), ... );
            }( std::make_integer_sequence<int,D>() );
            // the box ( it wins ties )
            const V a = V::load_aligned( ca + o ), b = V::load_aligned( cb + o ), hi( T( N ) - T( 0.5 ) );
            const V hb[ 4 ] = { hi - a, a + half, hi - b, b + half };
            constexpr int bd[ 4 ] = { Tb::dir_px, Tb::dir_mx, Tb::dir_py, Tb::dir_my };
            for( int q = 0; q < 4; ++q ) {
                fin[ bd[ q ] ] &= ~( U( ( 1ull << W ) - 1 ) << o ) | ( U( asimd::to_bits( hb[ q ] > h[ bd[ q ] ] ) ) << o );
                h[ bd[ q ] ] = asimd::min( h[ bd[ q ] ], hb[ q ] );
            }
            // the tightening, each vertex as soon as its two ht are known ( ht 0 kept for the last one ), the area
            if constexpr ( TIGHT >= 100 && TIGHT < 200 )
                for( int r = 0; r < TIGHT - 100; ++r )
                    [&]<int... k>( std::integer_sequence<int,k...> ) SDOT_INLINE_LAMBDA {
                        ( ( h[ k ] = asimd::min( h[ k ], mulc<Tb::pair_lam[ npair( k ) ]>( h[ Tb::pair_j[ npair( k ) ] ] ) + mulc<Tb::pair_mu[ npair( k ) ]>( h[ Tb::pair_l[ npair( k ) ] ] ) ) ), ... );
                    }( std::make_integer_sequence<int,D>() );
            // the pairs of span <= S, certified by the local convexity on the neighbours ( ht_k <= lam ht_{k-1} + mu ht_{k+1} for every k:
            // then ht is the support function of { n_k . x <= ht_k }, a cell between the true one and { n_k . x <= h_k }, so the true
            // one ); a vector with a lane that fails takes the 21 pairs
            if constexpr ( TIGHT >= 300 ) {
                V ht[ D ];
                if constexpr ( TIGHT == 400 ) sweep( h, ht );
                else [&]<int... k>( std::integer_sequence<int,k...> ) SDOT_INLINE_LAMBDA { ( ( ht[ k ] = h[ k ], tighten<TIGHT-300,k,Tb::pair_beg[ k ]>( h, ht[ k ] ) ), ... ); }( std::make_integer_sequence<int,D>() );
                V viol( T( -1 ) );
                [&]<int... k>( std::integer_sequence<int,k...> ) SDOT_INLINE_LAMBDA {
                    ( ( [&]() SDOT_INLINE_LAMBDA {
                        constexpr int p = npair( k );
                        const V c = mulc<Tb::pair_lam[ p ]>( ht[ Tb::pair_j[ p ] ] ) + mulc<Tb::pair_mu[ p ]>( ht[ Tb::pair_l[ p ] ] );
                        viol = asimd::max( viol, ht[ k ] - c - V( T( 1e-12 ) ) * ( V( T( 1 ) ) + asimd::max( c, V( T( 0 ) ) - c ) ) );
                    }() ), ... );
                }( std::make_integer_sequence<int,D>() );
                if ( const auto fb = asimd::to_bits( viol > V( T( 0 ) ) ) ) {
                    fallbacks.fetch_add( 1, std::memory_order_relaxed );
                    V h4[ D ]; for( int k = 0; k < D; ++k ) h4[ k ] = ht[ k ];
                    [&]<int... k>( std::integer_sequence<int,k...> ) SDOT_INLINE_LAMBDA { ( ( ht[ k ] = h[ k ], tighten<0,k,Tb::pair_beg[ k ]>( h, ht[ k ] ) ), ... ); }( std::make_integer_sequence<int,D>() );
                    if ( diag ) {                        // ( which lanes failed, which changed, which stay inverted with the 21 pairs: empty )
                        V ch( T( -1 ) ), vf( T( -1 ) );
                        [&]<int... k>( std::integer_sequence<int,k...> ) SDOT_INLINE_LAMBDA {
                            ( ( [&]() SDOT_INLINE_LAMBDA {
                                constexpr int p = npair( k );
                                const V c = mulc<Tb::pair_lam[ p ]>( ht[ Tb::pair_j[ p ] ] ) + mulc<Tb::pair_mu[ p ]>( ht[ Tb::pair_l[ p ] ] );
                                vf = asimd::max( vf, ht[ k ] - c - V( T( 1e-12 ) ) * ( V( T( 1 ) ) + asimd::max( c, V( T( 0 ) ) - c ) ) );
                                const V d = h4[ k ] - ht[ k ];
                                ch = asimd::max( ch, d - V( T( 1e-12 ) ) * ( V( T( 1 ) ) + asimd::max( ht[ k ], V( T( 0 ) ) - ht[ k ] ) ) );
                            }() ), ... );
                        }( std::make_integer_sequence<int,D>() );
                        const unsigned m = ( 1u << W ) - 1;
                        diag_failed.fetch_add( std::popcount( unsigned( fb ) & m ), std::memory_order_relaxed );
                        diag_changed.fetch_add( std::popcount( unsigned( asimd::to_bits( ch > V( T( 0 ) ) ) ) & m ), std::memory_order_relaxed );
                        diag_inverted.fetch_add( std::popcount( unsigned( asimd::to_bits( vf > V( T( 0 ) ) ) ) & unsigned( fb ) & m ), std::memory_order_relaxed );
                    }
                }
                for( int k = 0; k < D; ++k ) h[ k ] = ht[ k ];
            }
            V ht0, hp, X0, Y0, Xp, Yp, s( T( 0 ) );
            [&]<int... k>( std::integer_sequence<int,k...> ) SDOT_INLINE_LAMBDA {
                ( ( [&]() SDOT_INLINE_LAMBDA {
                    V t = h[ k ];
                    if constexpr ( TIGHT == 1 ) tighten<0,k,Tb::pair_beg[ k ]>( h, t );
                    if constexpr ( TIGHT >= 200 && TIGHT < 300 ) tighten<TIGHT-200,k,Tb::pair_beg[ k ]>( h, t );
                    if constexpr ( k == 0 ) ht0 = t;
                    else {
                        constexpr int m = k - 1;
                        const V X = mulc<Tb::vinv[ 4 * m + 0 ]>( hp ) + mulc<Tb::vinv[ 4 * m + 1 ]>( t );
                        const V Y = mulc<Tb::vinv[ 4 * m + 2 ]>( hp ) + mulc<Tb::vinv[ 4 * m + 3 ]>( t );
                        X.store_aligned( vx[ m ] + o ); Y.store_aligned( vy[ m ] + o );
                        if constexpr ( m == 0 ) { X0 = X; Y0 = Y; } else s = s + ( Xp * Y - Yp * X );
                        Xp = X; Yp = Y;
                    }
                    hp = t;
                }() ), ... );
            }( std::make_integer_sequence<int,D>() );
            constexpr int m = D - 1;
            const V X = mulc<Tb::vinv[ 4 * m + 0 ]>( hp ) + mulc<Tb::vinv[ 4 * m + 1 ]>( ht0 );
            const V Y = mulc<Tb::vinv[ 4 * m + 2 ]>( hp ) + mulc<Tb::vinv[ 4 * m + 3 ]>( ht0 );
            X.store_aligned( vx[ m ] + o ); Y.store_aligned( vy[ m ] + o );
            s = s + ( Xp * Y - Yp * X ) + ( X * Y0 - Y * X0 );
            ( s * half ).store_aligned( ar + o );
        }
        // the owners: the box or no seed -1, else the seed that won
        [&]<int... d>( std::integer_sequence<int,d...> ) SDOT_INLINE_LAMBDA {
            ( ( [&]() SDOT_INLINE_LAMBDA {
                constexpr int q0 = seed( d, 0 ), q1 = seed( d, 1 );
                for( int x = 0; x < B; ++x )
                    own[ d ][ x ] = ! ( ( fin[ d ] >> x ) & 1 ) ? -1 : q1 >= 0 && ( ( sec[ d ] >> x ) & 1 ) ? q1 : q0;
            }() ), ... );
        }( std::make_integer_sequence<int,D>() );
    }
};

} // namespace sdot::grid
