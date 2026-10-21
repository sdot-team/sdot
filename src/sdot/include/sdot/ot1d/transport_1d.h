#pragma once

// =====================================================================================
// THE GENERIC 1D TRANSPORT between two distributions given as streams of typed pieces ( `Piece1d.h` ), in increasing
// x, both of mass 1: W2^2 and the running sums a consumer needs to recover per-component quantities.
//
// A stream is any object with a type `Piece` and `bool next( Piece &p )` ( false at its end ); the two may emit
// different piece types.
// The sweep merges the two cumulated masses: on each merged sub-piece [ y0, y1 ] of the SOURCE, the transport map is
// T( y ) = Q_dst( F_src( y ) ), explicit through the pieces' `cdf` / `inv_cdf`, and the integrals are taken by a
// Q-point Gauss-Legendre rule -- EXACT when T is a polynomial: a linear source piece against a constant target piece
// gives a quadratic T, and Q = 3 integrates the cost ( y - T )^2 f ( degree 5 ) and the prefix sums below exactly.
//
// What it accumulates, in double, along the SOURCE ( `Sums` ):
//   cost  = int ( y - T )^2 dmu_src,
//   P0, P1, P2 = int T dy, int y T dy, int y^2 T dy over the source's support ( not weighted by the source density:
//         a consumer weights them itself, e.g. by its tents, with differences at their breakpoints ),
//   phi   = int 2 ( y - T ) dy, the Kantorovich potential of the source up to a constant ( gaps of the source included,
//         T being constant there ).
// `on_event( id, sums )` is called when the sweep reaches the x0 of a source piece whose `event` is >= 0, with the
// sums at that point -- also after the target is exhausted ( rounding ), so that every event is reported.
//
// Header-only, HD.
// =====================================================================================

#include "Piece1d.h"

#include <type_traits>

namespace sdot {
namespace ot1d {

/// the running sums of `transport_1d` ( see the header )
struct Sums1d {
    double cost = 0, P0 = 0, P1 = 0, P2 = 0, phi = 0;
};

template<int Q>
struct GaussLegendre1d;

template<>
struct GaussLegendre1d<3> {
    SDOT_OT1D_HD static double x( int q ) { return q == 0 ? -0.7745966692414834 : ( q == 1 ? 0.0 : 0.7745966692414834 ); }
    SDOT_OT1D_HD static double w( int q ) { return q == 1 ? 0.8888888888888888 : 0.5555555555555556; }
};

template<>
struct GaussLegendre1d<5> {
    SDOT_OT1D_HD static double x( int q ) {
        constexpr double a = 0.9061798459386640, b = 0.5384693101056831;
        return q == 0 ? -a : q == 1 ? -b : q == 2 ? 0.0 : q == 3 ? b : a;
    }
    SDOT_OT1D_HD static double w( int q ) {
        constexpr double a = 0.2369268850561891, b = 0.4786286704993665, c = 0.5688888888888889;
        return q == 0 || q == 4 ? a : q == 1 || q == 3 ? b : c;
    }
};

/// see the header. `Q`: the points of the Gauss-Legendre rule on each merged sub-piece.
template<int Q = 3, class SrcStream, class DstStream, class OnEvent>
SDOT_OT1D_HD Sums1d transport_1d( SrcStream &&src, DstStream &&dst, OnEvent &&on_event ) {
    using SP = typename std::decay_t<SrcStream>::Piece;
    using DP = typename std::decay_t<DstStream>::Piece;
    using GL = GaussLegendre1d<Q>;

    Sums1d S;
    SP ps;
    DP pd;
    bool has_src = src.next( ps ), has_dst = dst.next( pd );
    double sd = 0;                      // mass already taken in the current source piece
    double dd = 0;                      // mass already taken in the current target piece
    bool fired = false;                 // the current source piece's event already reported

    auto target_at = [&]( double level ) -> double {   // T at the current target piece, `level` the mass taken in it
        const double m = double( pd.mass() );
        return double( pd.x0 ) + double( pd.inv_cdf( level < 0 ? 0 : ( level > m ? m : level ) ) );
    };

    while ( has_src ) {
        if ( !fired && ps.event >= 0 )
            on_event( ps.event, S );
        fired = true;

        const double ms = double( ps.mass() );
        if ( !( ms > 0 ) ) {
            // a gap of the source ( or an event marker ): T is constant there, the potential still moves
            if ( has_dst && ps.length() > 0 ) {
                const double T = target_at( dd );
                const double a = double( ps.x0 ) - T, b = double( ps.x1 ) - T;
                S.phi += b * b - a * a;
            }
            has_src = src.next( ps ); sd = 0; fired = false;
            continue;
        }
        if ( !has_dst ) {                // the target is exhausted ( rounding ): finish the events
            has_src = src.next( ps ); sd = 0; fired = false;
            continue;
        }
        const double md = double( pd.mass() );
        if ( !( md - dd > 0 ) ) {        // a target piece used up ( or empty ): the next one
            has_dst = dst.next( pd ); dd = 0;
            continue;
        }

        const double take = ms - sd < md - dd ? ms - sd : md - dd;
        const bool src_done = take >= ms - sd;
        const double u0 = double( ps.inv_cdf( sd ) );
        const double u1 = src_done ? double( ps.length() ) : double( ps.inv_cdf( sd + take ) );

        // the merged sub-piece [ x0 + u0, x0 + u1 ] of the source onto the target piece, from `dd`
        const double h = 0.5 * ( u1 - u0 );
        if ( h > 0 ) {
            const double F0 = double( ps.cdf( u0 ) );
            for ( int q = 0; q < Q; ++q ) {
                const double u = 0.5 * ( u0 + u1 ) + h * GL::x( q );
                const double y = double( ps.x0 ) + u;
                const double f = double( ps.density( u ) );
                const double T = target_at( dd + double( ps.cdf( u ) ) - F0 );
                const double wq = h * GL::w( q );
                const double e = y - T;
                S.cost += wq * e * e * f;
                S.P0 += wq * T;
                S.P1 += wq * y * T;
                S.P2 += wq * y * y * T;
                S.phi += wq * 2 * e;
            }
        }

        sd += take; dd += take;
        if ( src_done ) {
            has_src = src.next( ps ); sd = 0; fired = false;
        }
    }
    return S;
}

} // namespace ot1d
} // namespace sdot
