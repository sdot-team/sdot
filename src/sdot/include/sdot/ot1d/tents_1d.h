#pragma once

// =====================================================================================
// A SUM OF TENTS ON A LINE as a stream of linear pieces, and what its transport gives back per tent.
//
// Tent i: center s_i, half-width w_i, mass m_i, density m_i t_i( y ), t_i = ( 1 - |y - s_i| / w_i )_+ / w_i. Their sum is
// linear between consecutive breakpoints s_i - w_i, s_i, s_i + w_i, where its slope jumps by m_i / w_i^2, -2 m_i / w_i^2,
// m_i / w_i^2. Sorted ( by the caller ), the breakpoints become a `TentStream` of `PolyPiece1d<double,1>`, each carrying
// the id of the breakpoint at its x0 ( 3 i + 0, 1, 2: left, center, right ), so that `transport_1d` reports its sums
// there ( `Sums1d`, stored by the caller in `EventSums` ).
//
// Per tent, from the sums at its three breakpoints ( `tent_moments` ):
//   b_i   = int T t_i                     the "barycenter": d W2^2 / d s_i = 2 m_i ( s_i - b_i ),
//   Q_i   = int T ( y - s_i ) t_i         and d W2^2 / d w_i = ( 2 m_i / w_i ) ( w_i^2 / 6 - Q_i ) ( var t_i = w_i^2 / 6 ),
//   phi_i = the potential at s_i           the derivative with respect to m_i, up to a constant ( O( w^2 ) ).
// t_i is linear on each half, so these are differences of the prefix sums P0, P1, P2 between its breakpoints.
// =====================================================================================

#include "transport_1d.h"

namespace sdot {
namespace ot1d {

/// the stream of a sorted array of breakpoints: `pos( k )`, `dslope( k )`, `id( k )` ( k < nb ), accessors
template<class Pos,class DSlope,class Id>
struct TentStream {
    using Piece = PolyPiece1d<double,1>;

    Pos    pos;
    DSlope dslope;
    Id     id;
    long     nb;
    long     k     = 0;
    double f     = 0;    // the density at breakpoint k
    double slope = 0;    // after breakpoint k

    SDOT_OT1D_HD bool next( Piece &p ) {
        if ( k >= nb )
            return false;
        const double x = double( pos( k ) );
        slope += double( dslope( k ) );
        p.x0 = x;
        p.event = int( id( k ) );
        if ( k + 1 < nb ) {
            const double x1 = double( pos( k + 1 ) );
            p.x1 = x1;
            p.c[ 0 ] = f > 0 ? f : 0.0;
            p.c[ 1 ] = slope;
            f += slope * ( x1 - x );
        } else {
            p.x1 = x;                // the last breakpoint: only its event
            p.c[ 0 ] = 0;
            p.c[ 1 ] = 0;
        }
        ++k;
        return true;
    }
};

template<class Pos,class DSlope,class Id>
SDOT_OT1D_HD TentStream<Pos,DSlope,Id> tent_stream( Pos pos, DSlope dslope, Id id, long nb ) {
    return { pos, dslope, id, nb };
}

/// the stream of `nb` constant pieces, `piece( k )` -> `PolyPiece1d<double,0>`
template<class F>
struct FuncStream0 {
    using Piece = PolyPiece1d<double,0>;
    F  piece;
    long nb;
    long k = 0;     // the next piece
    SDOT_OT1D_HD bool next( Piece &p ) {
        if ( k >= nb )
            return false;
        p = piece( k++ );
        return true;
    }
};

template<class F>
SDOT_OT1D_HD FuncStream0<F> func_stream0( F piece, long nb, long first = 0 ) {
    return { piece, nb, first };
}

/// the stream of the linear pieces k in [ k0, k1 ) given by `piece( k )` -> `PolyPiece1d<double,1>` ( a chunk )
template<class F>
struct FuncStream1 {
    using Piece = PolyPiece1d<double,1>;
    F  piece;
    long k, k1;
    SDOT_OT1D_HD bool next( Piece &p ) {
        if ( k >= k1 )
            return false;
        p = piece( k++ );
        return true;
    }
};

template<class F>
SDOT_OT1D_HD FuncStream1<F> func_stream1( F piece, long k0, long k1 ) {
    return { piece, k0, k1 };
}

/// per tent ( see the header ): b, Q, phi, from the sums at its left / center / right breakpoints
struct TentMoments {
    double b = 0, Q = 0, phi = 0;
};

SDOT_OT1D_HD TentMoments tent_moments( double s, double w, const Sums1d &L, const Sums1d &C, const Sums1d &R ) {
    const double iw2 = 1.0 / ( w * w );
    const double dl0 = C.P0 - L.P0, dl1 = C.P1 - L.P1, dl2 = C.P2 - L.P2;     // on [ s - w, s ]: t = ( y - s + w ) / w^2
    const double dr0 = R.P0 - C.P0, dr1 = R.P1 - C.P1, dr2 = R.P2 - C.P2;     // on [ s, s + w ]: t = ( s + w - y ) / w^2
    TentMoments m;
    m.b = iw2 * ( dl1 - ( s - w ) * dl0 + ( s + w ) * dr0 - dr1 );
    // ( y - s ) t: left ( y^2 - ( 2 s - w ) y + s ( s - w ) ) / w^2, right ( - y^2 + ( 2 s + w ) y - s ( s + w ) ) / w^2
    m.Q = iw2 * ( dl2 - ( 2 * s - w ) * dl1 + s * ( s - w ) * dl0
                - dr2 + ( 2 * s + w ) * dr1 - s * ( s + w ) * dr0 );
    m.phi = C.phi;
    return m;
}

} // namespace ot1d
} // namespace sdot
