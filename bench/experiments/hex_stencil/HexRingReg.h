// Power cells of seeds on a HEXAGONAL lattice ( honeycomb ), fixed stencil in registers ( asimd, unrolled ): ring 1 ( 6 neighbours )
// and ring 2 ( 18 neighbours ). The counterpart of `sdot/grid/Ring1Reg.h` / `Ring2Reg.h` for the square grid.
//
// Lattice: R rows of Wc seeds, row r at y = r sqrt( 3 ) / 2, odd rows shifted by + 1 / 2 ( "offset rows" ), seed ( r, c ) at
// x = c + ( r & 1 ) / 2, box [ -1/4, Wc - 1/4 ] x [ -sqrt( 3 ) / 4, R sqrt( 3 ) / 2 - sqrt( 3 ) / 4 ]. psi padded by two rows and
// two columns of -inf ( a missing seed gives h = +inf, no test ), P columns per padded row ( P >= Wc + 4, the lanes past Wc read -inf ).
// Cell i = { y : |y - x_i|^2 - psi_i <= |y - x_j|^2 - psi_j }, lanes along a row ( one parity per row: `SH` ).
//
// Everything in UNIT normals: h = ( |o|^2 + ( psi_i - psi_j ) ) / ( 2 |o| ). Ring 2 has 12 directions at 30 k degrees: the even ones
// carry the 6 ring-1 seeds ( |o| = 1 ) and their doubles ( |o| = 2 ), the odd ones the 6 seeds at sqrt( 3 ); the box faces are
// directions 0, 3, 6, 9 ( +x, +y, -x, -y ). The tightening is one pass on the raw h ( exact: an optimal LP basis has 2 elements ):
//     ht_k = min( h_k, lam h_{k-a} + mu h_{k+b} ),  a, b >= 1, a + b <= 5 ( 10 pairs per direction, 120 in all ),
//     lam = sin( b 30 ) / sin( ( a + b ) 30 ), mu = sin( a 30 ) / sin( ( a + b ) 30 ).
// ( all spans are needed: on OT iterates, spans 2, 4 and 5 inside, span 3 on the box; pairs_hex.py )
// Then no vertex at all: with an angle t between consecutive normals the edge k has the length
//     L_k = ( ht_{k-1} + ht_{k+1} - 2 cos t ht_k ) / sin t,   area = 1/2 sum_k ht_k L_k,
// and the hessian weight of the edge is L_k / ( 2 |o_owner| ).
//
// Ring 1 ( inner rows only: no +-y box ) is the same with 6 directions at 60 degrees and the single pair ( k - 1, k + 1 ). `ring1p`
// adds the exact ring-2 test: the support of the ring-1 cell is t_j in the even direction 2 j and ( t_j + t_{j+1} ) / sqrt 3 ( its
// vertex ) in the odd direction 2 j + 1, and a ring-2 seed cuts the cell iff its h is below that support ( margin biased to "cut" ).
// `ring12` = ring1p on a row, then ring 2 on the cut cells only ( `ring2_list`, psi copied lane by lane ).
#pragma once

#include <asimd/asimd.h>
#include <limits>
#include <utility>

// lambdas are not inlined by default once the body is large ( clang: the captures go through a closure in memory, h and t with them )
#define HEX_AI __attribute__( ( always_inline ) )
// timing only ( wrong results ): bit 1 no tightening, bit 2 no owner bits / factors, bit 4 no weights / box stores, bit 8 no inversion
// 1: `fin` as bit masks ( to_bits, costly on NEON ); 0: no `fin`, the weight of an edge without a seed ( box, none ) is 0
#ifndef HEX_BITS
#define HEX_BITS 0
#endif
// 1: fast tightening + certificate, the 120 pairs on failure; 0: always the 120 pairs
#ifndef HEX_FAST
#define HEX_FAST 1
#endif
#ifndef HEX_SKIP
#define HEX_SKIP 0
#endif

namespace hexgrid {

using U = unsigned long long;

// sin( 30 k degrees ), k = 0 .. 6
constexpr double s30[ 7 ] = { 0.0, 0.5, 0.86602540378443864676, 1.0, 0.86602540378443864676, 0.5, 0.0 };
constexpr double SQ3 = 1.73205080756887729353;

// the 18 seeds of ring 2 ( axial offsets ): 0..5 the ring-1 seeds u_j ( direction 2 j, |o|^2 = 1 ), 6..11 their doubles 2 u_j
// ( direction 2 j, |o|^2 = 4 ), 12..17 v_j = u_j + u_{j+1} ( direction 2 j + 1, |o|^2 = 3 )
constexpr int uq[ 6 ] = { 1, 0, -1, -1, 0, 1 }, ur[ 6 ] = { 0, 1, 1, 0, -1, -1 };
constexpr int vq[ 6 ] = { 1, -1, -2, -1, 1, 2 }, vr[ 6 ] = { 1, 2, 1, -1, -2, -1 };
constexpr int seed_q( int s ) { return s < 6 ? uq[ s ] : s < 12 ? 2 * uq[ s - 6 ] : vq[ s - 12 ]; }
constexpr int seed_r( int s ) { return s < 6 ? ur[ s ] : s < 12 ? 2 * ur[ s - 6 ] : vr[ s - 12 ]; }
constexpr int seed_m2( int s ) { return s < 6 ? 1 : s < 12 ? 4 : 3; }
constexpr double seed_norm( int s ) { return s < 6 ? 1.0 : s < 12 ? 2.0 : SQ3; }

// column offset of the axial offset ( dq, dr ) from a row of parity SH ( offset rows, odd rows shifted right )
constexpr int dcol( int dq, int dr, int SH ) { const int sh2 = ( SH + ( dr & 1 ) ) & 1; return dq + ( dr + SH - sh2 ) / 2; }
// offset in the padded psi of seed s
constexpr int seed_off( int s, int SH, int P ) { return seed_r( s ) * P + dcol( seed_q( s ), seed_r( s ), SH ); }

template<int B, class T>
struct HexRingReg {
    static constexpr int W = asimd::SimdSize<T>::value;
    static_assert( B % W == 0 && B <= 64, "HexRingReg: B a multiple of the SIMD width, at most 64" );
    static constexpr int NV = B / W, D2 = 12, NS = 18;
    using V = asimd::SimdVec<T,W>;
    static constexpr U LANES = W == 64 ? ~0ull : ( 1ull << W ) - 1;

    template<double c> [[gnu::always_inline]] static V mulc( const V &x ) {
        if constexpr ( c == 0 ) return V( T( 0 ) );
        else if constexpr ( c == 1 ) return x;
        else return V( T( c ) ) * x;
    }

    /// the box of a row: -x, +x faces ( x0, x1 ), distances to the +-y faces
    struct Row {
        T x0, x1, hy0, hy1, xs;
        Row( int Wc, int R, int r ) : x0( T( -0.25 ) ), x1( T( Wc ) - T( 0.25 ) ), hy0( T( r ) * T( SQ3 / 2 ) + T( SQ3 / 4 ) ),
            hy1( T( R ) * T( SQ3 / 2 ) - T( SQ3 / 4 ) - T( r ) * T( SQ3 / 2 ) ), xs( T( 0.5 ) * T( r & 1 ) ) {}
    };

    /// the outputs of a block
    /// `zb`: the box of the cell, relative to its seed ( x min, x max, y min, y max: what the zone certificate needs ); `inv` ( bit x:
    /// the cell is empty, an edge of negative length beyond 1e-9 times its extent )
    struct Out2 {
        alignas( 64 ) T area[ B ];
        alignas( 64 ) T wk[ D2 ][ B ];
        alignas( 64 ) T zb[ 4 ][ B ];
        alignas( 64 ) T tt[ D2 ][ B ];               // the tightened offsets ( unit normals at 30 k degrees ): the polygon, for the zone pieces
        alignas( 64 ) T wf[ D2 ][ B ];               // ( scratch: the factor of each weight, 0 without a seed )
        U fin[ D2 ], sec[ D2 ], inv;                 // ( `fin` only with HEX_BITS )
    };
    struct Out1 { alignas( 64 ) T area[ B ]; alignas( 64 ) T wk[ 6 ][ B ]; alignas( 64 ) T zb[ 4 ][ B ]; alignas( 64 ) T tt[ 6 ][ B ]; U fin[ 6 ], cut, inv; };

    // lam x + mu y in as few operations as the constants allow ( one fma, or an add and a mul, or an add )
    static constexpr bool near( double a, double b ) { return a - b < 1e-15 && b - a < 1e-15; }
    template<double lam,double mu> [[gnu::always_inline]] static V comb( const V &x, const V &y ) {
        if constexpr ( near( lam, mu ) ) return mulc<lam>( x + y );
        else if constexpr ( near( mu, 1 ) ) return asimd::fma( V( T( lam ) ), x, y );
        else if constexpr ( near( lam, 1 ) ) return asimd::fma( V( T( mu ) ), y, x );
        else return asimd::fma( V( T( lam ) ), x, V( T( mu ) ) * y );
    }

    // ---------------------------------------------------------------- ring 2
    // ht_k = min( h_k, lam h_{k-a} + mu h_{k+b} ) over a, b >= 1, a + b <= 5
    template<int k,int a=1,int b=1> [[gnu::always_inline]] static void tighten2( const V ( &h )[ D2 ], V &t ) {
        if constexpr ( a + b <= 5 ) {
            constexpr double lam = s30[ b ] / s30[ a + b ], mu = s30[ a ] / s30[ a + b ];
            t = asimd::min( t, comb<lam,mu>( h[ ( k - a + D2 ) % D2 ], h[ ( k + b ) % D2 ] ) );
            if constexpr ( a + b < 5 ) tighten2<k,a,b+1>( h, t );
            else tighten2<k,a+1,1>( h, t );
        }
    }

    static inline long fallbacks = 0; // vectors that failed the certificate of the fast tightening ( HEX_FAST )
    [[gnu::noinline]] static void full_tighten( const V ( &h )[ D2 ], V ( &t )[ D2 ] ) {
        [&]<int... k>( std::integer_sequence<int,k...> ) HEX_AI { ( ( t[ k ] = h[ k ], tighten2<k>( h, t[ k ] ) ), ... ); }( std::make_integer_sequence<int,D2>() );
    }

    /// one vector of W cells: `pj( s )` the psi of seed s, `x` the abscissae; outputs at lane offset o
    template<class PJ> [[gnu::always_inline]] static void core2( const V &pi, const PJ &pj, const V &x, const Row &rw, int o, Out2 &out ) {
        auto hs = [&]<int s>() HEX_AI { return ( V( T( seed_m2( s ) ) ) + ( pi - pj.template operator()<s>() ) ) * V( T( 0.5 / seed_norm( s ) ) ); };
        V h[ D2 ], fac[ D2 / 2 ]; // even directions: weight L / 2 for u, L / 4 for 2 u
        [&]<int... d>( std::integer_sequence<int,d...> ) HEX_AI {
            ( ( [&]() HEX_AI {
                constexpr int q = d / 2;
                if constexpr ( d % 2 == 0 ) {
                    const V a = hs.template operator()<q>(), b = hs.template operator()<6 + q>();
                    if constexpr ( ! ( HEX_SKIP & 2 ) ) {
                        out.sec[ d ] |= U( asimd::to_bits( a > b ) ) << o;
                        fac[ q ] = asimd::select( a > b, V( T( 0.25 ) ), V( T( 0.5 ) ) );
                    } else fac[ q ] = V( T( 0.5 ) );
                    h[ d ] = asimd::min( a, b );
                } else
                    h[ d ] = hs.template operator()<12 + q>();
                if constexpr ( ! ( HEX_SKIP & 2 ) && HEX_BITS ) out.fin[ d ] |= U( asimd::to_bits( V( std::numeric_limits<T>::infinity() ) > h[ d ] ) ) << o;
                if constexpr ( ! HEX_BITS ) { // the factor of the weight, 0 without a seed, kept in the output until L is known
                    const V f = d % 2 == 0 ? fac[ q ] : V( T( 0.5 / SQ3 ) );
                    asimd::select( V( std::numeric_limits<T>::infinity() ) > h[ d ], f, V( T( 0 ) ) ).store_aligned( &out.wf[ d ][ 0 ] + o );
                }
            }() ), ... );
        }( std::make_integer_sequence<int,D2>() );
        // the box ( it wins ties ): +x 0, +y 3, -x 6, -y 9
        const V hb[ 4 ] = { V( rw.x1 ) - x, V( rw.hy1 ), x - V( rw.x0 ), V( rw.hy0 ) };
        constexpr int bd[ 4 ] = { 0, 3, 6, 9 };
        for( int q = 0; q < 4; ++q ) {
            if constexpr ( ! ( HEX_SKIP & 2 ) && HEX_BITS ) out.fin[ bd[ q ] ] &= ~( LANES << o ) | ( U( asimd::to_bits( hb[ q ] > h[ bd[ q ] ] ) ) << o );
            if constexpr ( ! HEX_BITS ) {
                T *w = &out.wf[ bd[ q ] ][ 0 ] + o;
                asimd::select( hb[ q ] > h[ bd[ q ] ], V::load_aligned( w ), V( T( 0 ) ) ).store_aligned( w );
            }
            h[ bd[ q ] ] = asimd::min( h[ bd[ q ] ], hb[ q ] );
        }
        V t[ D2 ];
        if constexpr ( HEX_SKIP & 1 ) { for( int k = 0; k < D2; ++k ) t[ k ] = h[ k ]; }
        else if constexpr ( ! HEX_FAST ) full_tighten( h, t );
        else {
            // fast: even directions by ( k - 2, k + 2 ) and ( k - 1, k + 1 ) on the raw h, odd ones by their tightened neighbours. Each t_k is
            // an upper bound of the support and <= h_k: if every edge has a length >= 0, { n_k . x <= t_k } is a polygon inside the cell
            // whose support is t, so t is the support of the cell ( exact ). A vector with a lane that fails takes the 120 pairs
            const V c( T( 1 / SQ3 ) );
            [&]<int... e>( std::integer_sequence<int,e...> ) HEX_AI {
                ( ( t[ 2 * e ] = asimd::min( asimd::min( h[ 2 * e ], h[ ( 2 * e + 10 ) % D2 ] + h[ ( 2 * e + 2 ) % D2 ] ),
                                            ( h[ ( 2 * e + 11 ) % D2 ] + h[ 2 * e + 1 ] ) * c ) ), ... );
                ( ( t[ 2 * e + 1 ] = asimd::min( h[ 2 * e + 1 ], ( t[ 2 * e ] + t[ ( 2 * e + 2 ) % D2 ] ) * c ) ), ... );
            }( std::make_integer_sequence<int,D2/2>() );
        }
        // the outputs ( L_k = 2 ( t_{k-1} + t_{k+1} ) - 2 sqrt 3 t_k, 30 degrees ), the min of the L
        auto finish = [&]() HEX_AI {
            if constexpr ( ! ( HEX_SKIP & 4 ) ) {
                ( V( T( 0 ) ) - t[ 6 ] ).store_aligned( out.zb[ 0 ] + o ); t[ 0 ].store_aligned( out.zb[ 1 ] + o );
                ( V( T( 0 ) ) - t[ 9 ] ).store_aligned( out.zb[ 2 ] + o ); t[ 3 ].store_aligned( out.zb[ 3 ] + o );
                for( int k = 0; k < D2; ++k ) t[ k ].store_aligned( out.tt[ k ] + o );
            }
            V s( T( 0 ) ), mL( T( 0 ) );
            [&]<int... k>( std::integer_sequence<int,k...> ) HEX_AI {
                ( ( [&]() HEX_AI {
                    const V L = V( T( 2 ) ) * ( t[ ( k + D2 - 1 ) % D2 ] + t[ ( k + 1 ) % D2 ] ) - V( T( 2 * SQ3 ) ) * t[ k ];
                    mL = asimd::min( mL, L );
                    s = s + t[ k ] * L;
                    if constexpr ( HEX_SKIP & 4 ) {}
                    else if constexpr ( ! HEX_BITS ) ( L * V::load_aligned( &out.wf[ k ][ 0 ] + o ) ).store_aligned( &out.wk[ k ][ 0 ] + o );
                    else if constexpr ( k % 2 == 0 ) ( L * fac[ k / 2 ] ).store_aligned( &out.wk[ k ][ 0 ] + o );
                    else ( L * V( T( 0.5 / SQ3 ) ) ).store_aligned( &out.wk[ k ][ 0 ] + o );
                }() ), ... );
            }( std::make_integer_sequence<int,D2>() );
            ( s * V( T( 0.5 ) ) ).store_aligned( out.area + o );
            return mL;
        };
        V mL = finish();
        if constexpr ( HEX_FAST && ! ( HEX_SKIP & 1 ) ) {
            // the certificate ( 2 sin 30 = 1: L here is twice the one of the check ); a NaN ( inf - inf: an unbounded fast cell ) or an
            // infinite extent fails
            const V ext = asimd::max( t[ 0 ] + t[ 6 ], t[ 3 ] + t[ 9 ] );
            const auto good = asimd::to_bits( asimd::ge( mL, V( T( -2e-13 ) ) * ( V( T( 1 ) ) + ext ) ) ) & asimd::to_bits( V( std::numeric_limits<T>::max() ) > ext );
            if( good != ( 1u << W ) - 1 ) [[unlikely]] {
                ++fallbacks;
                full_tighten( h, t );
                mL = finish();
            }
        }
        if constexpr ( ! ( HEX_SKIP & 8 ) ) out.inv |= U( asimd::to_bits( mL < V( T( -1e-9 ) ) * ( V( T( 1 ) ) + t[ 0 ] + t[ 6 ] + t[ 3 ] + t[ 9 ] ) ) ) << o;
    }

    /// the cells ( r, c0 .. c0 + B - 1 ) of a row of parity SH ( = r & 1 )
    template<int SH>
    static void ring2( int Wc, int R, const T *pp, int P, int r, int c0, Out2 &out ) {
        const Row rw( Wc, R, r );
        const T *row = pp + size_t( r + 2 ) * P + 2 + c0;
        for( int k = 0; k < D2; ++k ) out.fin[ k ] = out.sec[ k ] = 0;
        out.inv = 0;
        // one vector at a time ( no unrolling: interleaved vectors spill the 12 h and 12 t )
        #pragma clang loop unroll( disable )
        for( int o = 0; o < B; o += W ) {
            auto pj = [&]<int s>() HEX_AI { return V::load_unaligned( row + seed_off( s, SH, 0 ) + seed_r( s ) * P + o ); };
            core2( V::load_unaligned( row + o ), pj, V::iota( T( c0 + o ) ) + V( rw.xs ), rw, o, out );
        }
    }

    /// the cells ( r, cols[ 0 .. n - 1 ] ) of a row of parity SH, 1 <= n <= B ( lanes past n repeat the last cell ): psi copied lane by lane
    template<int SH>
    static void ring2_list( int Wc, int R, const T *pp, int P, int r, const int *cols, int n, Out2 &out ) {
        const Row rw( Wc, R, r );
        const T *row = pp + size_t( r + 2 ) * P + 2;
        alignas( 64 ) T pi[ B ], pj[ NS ][ B ], xx[ B ];
        for( int x = 0; x < B; ++x ) {
            const int c = cols[ x < n ? x : n - 1 ];
            pi[ x ] = row[ c ]; xx[ x ] = T( c ) + rw.xs;
            [&]<int... s>( std::integer_sequence<int,s...> ) HEX_AI { ( ( pj[ s ][ x ] = row[ c + seed_off( s, SH, 0 ) + seed_r( s ) * P ] ), ... ); }( std::make_integer_sequence<int,NS>() );
        }
        for( int k = 0; k < D2; ++k ) out.fin[ k ] = out.sec[ k ] = 0;
        out.inv = 0;
        #pragma clang loop unroll( disable )
        for( int o = 0; o < B; o += W ) {
            auto pjv = [&]<int s>() HEX_AI { return V::load_aligned( pj[ s ] + o ); };
            core2( V::load_aligned( pi + o ), pjv, V::load_aligned( xx + o ), rw, o, out );
        }
    }

    // ---------------------------------------------------------------- ring 1 ( inner rows: 1 <= r <= R - 2 )
    /// ring 1 of the cells ( r, c0 .. c0 + B - 1 ); PRED: `out.cut` ( bit x: a ring-2 seed cuts the ring-1 cell c0 + x: the cell is
    /// NOT exact, take ring 2 )
    template<int SH,bool PRED>
    static void ring1( int Wc, int R, const T *pp, int P, int r, int c0, Out1 &out ) {
        const Row rw( Wc, R, r );
        const T *row = pp + size_t( r + 2 ) * P + 2 + c0;
        for( int k = 0; k < 6; ++k ) out.fin[ k ] = 0;
        out.cut = out.inv = 0;
        [&]<int... j>( std::integer_sequence<int,j...> ) HEX_AI {
            ( ( [&]() HEX_AI {
                constexpr int o = j * W;
                const V pi = V::load_unaligned( row + o ), half( T( 0.5 ) ), one( T( 1 ) );
                V h[ 6 ];
                [&]<int... d>( std::integer_sequence<int,d...> ) HEX_AI {
                    ( ( h[ d ] = ( one + ( pi - V::load_unaligned( row + seed_off( d, SH, 0 ) + seed_r( d ) * P + o ) ) ) * half ), ... );
                    ( ( out.fin[ d ] |= U( asimd::to_bits( V( std::numeric_limits<T>::infinity() ) > h[ d ] ) ) << o ), ... );
                }( std::make_integer_sequence<int,6>() );
                const V x = V::iota( T( c0 + o ) ) + V( rw.xs );
                const V hb0 = V( rw.x1 ) - x, hb3 = x - V( rw.x0 );
                out.fin[ 0 ] &= ~( LANES << o ) | ( U( asimd::to_bits( hb0 > h[ 0 ] ) ) << o );
                out.fin[ 3 ] &= ~( LANES << o ) | ( U( asimd::to_bits( hb3 > h[ 3 ] ) ) << o );
                h[ 0 ] = asimd::min( h[ 0 ], hb0 ); h[ 3 ] = asimd::min( h[ 3 ], hb3 );
                V t[ 6 ];
                [&]<int... k>( std::integer_sequence<int,k...> ) HEX_AI { ( ( t[ k ] = asimd::min( h[ k ], h[ ( k + 5 ) % 6 ] + h[ ( k + 1 ) % 6 ] ) ), ... ); }( std::make_integer_sequence<int,6>() );
                // L_k sin 60 = t_{k-1} + t_{k+1} - t_k, area = 1/2 sum t_k L_k, weight L_k / 2
                const V c( T( 1 / SQ3 ) );
                // the box: x by the directions 3, 0, y by the vertices 3 | 4 and 1 | 2
                for( int k = 0; k < 6; ++k ) t[ k ].store_aligned( out.tt[ k ] + o );
                ( V( T( 0 ) ) - t[ 3 ] ).store_aligned( out.zb[ 0 ] + o ); t[ 0 ].store_aligned( out.zb[ 1 ] + o );
                ( V( T( 0 ) ) - c * ( t[ 4 ] + t[ 5 ] ) ).store_aligned( out.zb[ 2 ] + o ); ( c * ( t[ 1 ] + t[ 2 ] ) ).store_aligned( out.zb[ 3 ] + o );
                V s( T( 0 ) ), mL( T( 0 ) );
                [&]<int... k>( std::integer_sequence<int,k...> ) HEX_AI {
                    ( ( [&]() HEX_AI {
                        const V L = t[ ( k + 5 ) % 6 ] + t[ ( k + 1 ) % 6 ] - t[ k ];
                        mL = asimd::min( mL, L );
                        s = s + t[ k ] * L;
                        ( c * L ).store_aligned( &out.wk[ k ][ 0 ] + o );
                    }() ), ... );
                }( std::make_integer_sequence<int,6>() );
                ( c * s ).store_aligned( out.area + o );
                out.inv |= U( asimd::to_bits( mL < V( T( -1e-9 ) ) * ( one + t[ 0 ] + t[ 3 ] ) ) ) << o;
                if constexpr ( PRED ) {
                    // seed s of ring 2 cuts iff h_s < support of the ring-1 cell in its direction ( + a margin: rounding flags, never misses )
                    V cut( T( 0 ) );
                    [&]<int... q>( std::integer_sequence<int,q...> ) HEX_AI {
                        ( ( [&]() HEX_AI {
                            const V sup2 = t[ q ], supv = ( t[ q ] + t[ ( q + 1 ) % 6 ] ) * V( T( 1 / SQ3 ) );
                            const V p2 = V::load_unaligned( row + seed_off( 6 + q, SH, 0 ) + seed_r( 6 + q ) * P + o );
                            const V pv = V::load_unaligned( row + seed_off( 12 + q, SH, 0 ) + seed_r( 12 + q ) * P + o );
                            // 2 |o| sup > |o|^2 + psi_i - psi_j, |o| = 2 / sqrt 3
                            const V m2 = V( T( 4 ) ) * sup2 - ( V( T( 4 ) ) + ( pi - p2 ) ) + V( T( 1e-12 ) ) * ( one + asimd::max( sup2, V( T( 0 ) ) - sup2 ) );
                            const V mv = V( T( 2 * SQ3 ) ) * supv - ( V( T( 3 ) ) + ( pi - pv ) ) + V( T( 1e-12 ) ) * ( one + asimd::max( supv, V( T( 0 ) ) - supv ) );
                            cut = asimd::max( cut, asimd::max( m2, mv ) );
                        }() ), ... );
                    }( std::make_integer_sequence<int,6>() );
                    // the +-y faces of the box are not in ring 1: a cell that reaches them takes ring 2 ( support in -y / +y: the
                    // vertices 4 | 5 and 1 | 2, against the distances to the faces )
                    cut = asimd::max( cut, asimd::max( c * ( t[ 4 ] + t[ 5 ] ) - V( rw.hy0 ), c * ( t[ 1 ] + t[ 2 ] ) - V( rw.hy1 ) ) + V( T( 1e-12 ) ) * ( one + V( rw.hy0 ) ) );
                    out.cut |= U( asimd::to_bits( cut > V( T( 0 ) ) ) ) << o;
                }
            }() ), ... );
        }( std::make_integer_sequence<int,NV>() );
    }
};

} // namespace hexgrid
