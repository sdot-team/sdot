#pragma once

// =====================================================================================
// TYPED 1D PIECES: what a 1D distribution emits, in increasing x, for the generic transport `transport_1d`
// ( `transport_1d.h` ). A piece is a density on [ x0, x1 ], given in the local coordinate u = x - x0:
//
//   * `PolyPiece1d<TF,P>`: a polynomial density c_0 + c_1 u + ... + c_P u^P ( P = 0: a constant pixel, P = 1: a sum of
//     overlapping tents between two of their breakpoints ). Its cumulated mass is in closed form; its inverse too for
//     P <= 1, by a safeguarded Newton beyond.
//
// A piece may carry an EVENT: an integer the transport hands back, with its running sums, when the sweep reaches the
// piece's x0 -- how a distribution made of overlapping COMPONENTS ( tents ) recovers per-component integrals from the
// prefix sums at its breakpoints ( `ProjectedSumOfTents` ). -1: none. A piece of zero length or zero mass is legal: it
// only carries its event ( or a gap ).
//
// Header-only, HD: the same code on the host and on the card.
// =====================================================================================

#include <loom/support/common_macros.h>

#include <cmath>

namespace sdot {
namespace ot1d {

#if defined( __CUDACC__ )
#define SDOT_OT1D_HD __host__ __device__ __forceinline__
#else
#define SDOT_OT1D_HD inline
#endif

template<class TF,int P>
struct PolyPiece1d {
    static constexpr int degree = P;

    TF  x0    = 0, x1 = 0;
    TF  c[ P + 1 ] = {};      ///< the density in u = x - x0
    int event = -1;           ///< handed back by the sweep when it reaches x0 ( -1: none )

    SDOT_OT1D_HD TF length() const { return x1 - x0; }

    /// the density at local coordinate u
    SDOT_OT1D_HD TF density( TF u ) const {
        TF r = c[ P ];
        for ( int j = P - 1; j >= 0; --j )
            r = r * u + c[ j ];
        return r;
    }

    /// the mass on [ x0, x0 + u ]
    SDOT_OT1D_HD TF cdf( TF u ) const {
        TF r = c[ P ] / TF( P + 1 );
        for ( int j = P - 1; j >= 0; --j )
            r = r * u + c[ j ] / TF( j + 1 );
        return r * u;
    }

    SDOT_OT1D_HD TF mass() const { return cdf( length() ); }

    /// the local coordinate u where the cumulated mass reaches m ( 0 <= m <= mass() )
    SDOT_OT1D_HD TF inv_cdf( TF m ) const {
        const TF L = length();
        if ( m <= 0 )
            return 0;
        if constexpr ( P == 0 ) {
            return c[ 0 ] > 0 ? ( m / c[ 0 ] < L ? m / c[ 0 ] : L ) : L;
        } else if constexpr ( P == 1 ) {
            // c0 u + c1 u^2 / 2 = m, the root in [ 0, L ]: u = 2 m / ( c0 + sqrt( c0^2 + 2 c1 m ) ) ( no cancellation )
            const TF d = c[ 0 ] * c[ 0 ] + 2 * c[ 1 ] * m;
            const TF s = d > 0 ? sqrt( d ) : TF( 0 );
            const TF den = c[ 0 ] + s;
            if ( !( den > 0 ) )
                return L;
            const TF u = 2 * m / den;
            return u < L ? u : L;
        } else {
            // a safeguarded Newton on [ 0, L ] ( the cdf is increasing: the density is >= 0 )
            TF lo = 0, hi = L, u = L * ( m / mass() );
            for ( int it = 0; it < 60; ++it ) {
                const TF r = cdf( u ) - m;
                if ( r > 0 ) hi = u; else lo = u;
                const TF f = density( u );
                TF v = f > 0 ? u - r / f : TF( 0.5 ) * ( lo + hi );
                if ( !( v > lo && v < hi ) )
                    v = TF( 0.5 ) * ( lo + hi );
                if ( fabs( v - u ) <= TF( 1e-15 ) * ( L > 0 ? L : TF( 1 ) ) )
                    return v;
                u = v;
            }
            return u;
        }
    }
};

} // namespace ot1d
} // namespace sdot
