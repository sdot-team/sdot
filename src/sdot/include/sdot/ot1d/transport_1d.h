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

/// `transport_1d` for a source of LINEAR pieces ( or constant ) against a target of CONSTANT pieces -- the tents
/// against an image. The same sweep and the same quadrature, with what a polynomial T allows: on a merged sub-piece,
/// t = u - u0, the source density is a + b t, and T( t ) = x_d + ( dd + a t + b t^2 / 2 ) / rho is evaluated as a
/// polynomial ( no inverse cdf at the quadrature points ); the sub-piece starts where the previous one ended; 1 / rho
/// is taken once per target piece. One inverse cdf remains, where a target piece ends inside a source piece.
template<int Q, class SrcStream, class DstStream, class OnEvent>
SDOT_OT1D_HD Sums1d transport_1d_lin_const( SrcStream &&src, DstStream &&dst, OnEvent &&on_event, double dd0 ) {
    using SP = typename std::decay_t<SrcStream>::Piece;
    using GL = GaussLegendre1d<Q>;

    Sums1d S;
    typename std::decay_t<SrcStream>::Piece ps;
    typename std::decay_t<DstStream>::Piece pd;
    bool has_src = src.next( ps ), has_dst = dst.next( pd );
    double sd = 0, us = 0;              // mass and local coordinate already taken in the current source piece
    double dd = dd0;                    // mass already taken in the current target piece
    bool fired = false;
    double ms = has_src ? double( ps.mass() ) : 0.0;
    double md = 0, xd = 0, Ld = 0, rd = 0;   // the current target piece: mass, start, length, 1 / density
    auto load_dst = [&]() {
        md = double( pd.mass() ); xd = double( pd.x0 ); Ld = double( pd.x1 ) - xd;
        rd = pd.c[ 0 ] > 0 ? 1.0 / double( pd.c[ 0 ] ) : 0.0;
    };
    if ( has_dst )
        load_dst();
    auto next_src = [&]() {
        has_src = src.next( ps ); sd = 0; us = 0; fired = false;
        if ( has_src )
            ms = double( ps.mass() );
    };
    auto target_at = [&]( double level ) -> double {   // as the generic version's, for a constant piece
        level = level < 0 ? 0 : ( level > md ? md : level );
        if ( !( level > 0 ) )
            return xd;
        if ( !( rd > 0 ) )
            return xd + Ld;
        const double u = level * rd;
        return xd + ( u < Ld ? u : Ld );
    };

    while ( has_src ) {
        if ( !fired && ps.event >= 0 )
            on_event( ps.event, S );
        fired = true;

        if ( !( ms > 0 ) ) {
            if ( has_dst && ps.length() > 0 ) {
                const double T = target_at( dd );
                const double a = double( ps.x0 ) - T, b = double( ps.x1 ) - T;
                S.phi += b * b - a * a;
            }
            next_src();
            continue;
        }
        if ( !has_dst ) {
            next_src();
            continue;
        }
        if ( !( md - dd > 0 ) ) {
            has_dst = dst.next( pd ); dd = 0;
            if ( has_dst )
                load_dst();
            continue;
        }

        const double take = ms - sd < md - dd ? ms - sd : md - dd;
        const bool src_done = take >= ms - sd;
        const double u0 = us;
        const double u1 = src_done ? double( ps.length() ) : double( ps.inv_cdf( sd + take ) );

        const double h = 0.5 * ( u1 - u0 );
        if ( h > 0 ) {
            const double a = double( ps.density( u0 ) );
            double b = 0;
            if constexpr ( SP::degree == 1 )
                b = double( ps.c[ 1 ] );
            const double T0 = xd + dd * rd, T1 = a * rd, T2 = 0.5 * b * rd;
            const double y0 = double( ps.x0 ) + u0;
            for ( int q = 0; q < Q; ++q ) {
                const double t = h + h * GL::x( q );
                const double y = y0 + t;
                const double f = a + b * t;
                const double T = T0 + t * ( T1 + t * T2 );
                const double wq = h * GL::w( q );
                const double e = y - T;
                S.cost += wq * e * e * f;
                S.P0 += wq * T;
                S.P1 += wq * y * T;
                S.P2 += wq * y * y * T;
                S.phi += wq * 2 * e;
            }
        }

        sd += take; dd += take; us = u1;
        if ( src_done )
            next_src();
    }
    return S;
}

/// see the header. `Q`: the points of the Gauss-Legendre rule on each merged sub-piece. `dd0`: the mass already taken in
/// the target's first piece -- a chunk of a parallel sweep starts the target mid-way ( its sums are then local ).
template<int Q = 3, class SrcStream, class DstStream, class OnEvent>
SDOT_OT1D_HD Sums1d transport_1d( SrcStream &&src, DstStream &&dst, OnEvent &&on_event, double dd0 = 0 ) {
    using SP = typename std::decay_t<SrcStream>::Piece;
    using DP = typename std::decay_t<DstStream>::Piece;
    using GL = GaussLegendre1d<Q>;
    if constexpr ( requires { SP::degree; DP::degree; } )
        if constexpr ( SP::degree <= 1 && DP::degree == 0 )
            return transport_1d_lin_const<Q>( src, dst, on_event, dd0 );

    Sums1d S;
    SP ps;
    DP pd;
    bool has_src = src.next( ps ), has_dst = dst.next( pd );
    double sd = 0;                      // mass already taken in the current source piece
    double dd = dd0;                    // mass already taken in the current target piece
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
