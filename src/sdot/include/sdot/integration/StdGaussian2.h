#pragma once

// =====================================================================================
// THE STANDARD NORMAL OF THE PLANE over a triangle and along a segment -- the exact 2D reduction of `SumOfGaussians` ( see
// `SumOfGaussians.h`: corners in polar coordinates, `erf`s along the edges ), as free functions of the STANDARD FRAME
// `y = ( x - c ) / s`: what a gaussian of the distribution and the gaussian factor of a radial integrand ( `RadialExp.h` )
// both integrate, whatever carries them.
// =====================================================================================

#include <loom/support/common_macros.h>
#include <loom/support/containers/Vector.h>
#include <loom/support/math.h>

namespace sdot {

namespace detail {
    // 8-point Gauss-Legendre on [ -1, 1 ], positive half
    LOOM_CONSTANT( double gl8_x[ 4 ] ) = { 0.1834346424956498, 0.5255324099163290,
                                           0.7966664774136267, 0.9602898564975363 };
    LOOM_CONSTANT( double gl8_w[ 4 ] ) = { 0.3626837833783620, 0.3137066458778873,
                                           0.2223810344533745, 0.1012285362903763 };

    // `Phi`, the standard normal cumulative distribution function
    template<class TF> HD TF std_normal_cdf( TF u ) {
        return TF( 0.5 ) * ( 1 + sdot::erf( u * TF( 0.70710678118654752440 ) ) );
    }
}

namespace integration {

/// the SIGNED standard normal measure of the corner `( 0, P, Q )`. `tail_cut`: beyond, the exponential is worth nothing and
/// the tails are exact ( arctangents ); `nb_panels`: at most that many Gauss-Legendre panels over the core.
template<class TF>
HD TF std_wedge_measure( const auto &P, const auto &Q, TF tail_cut, int nb_panels ) {
    const TF two_pi = TF( 6.283185307179586476925286766559 );

    const TF dx = Q[ 0 ] - P[ 0 ], dy = Q[ 1 ] - P[ 1 ];
    const TF L = sdot::sqrt( dx * dx + dy * dy );
    if ( ! ( L > 0 ) )
        return 0;

    // `( n, u )` DIRECT, so that `cross( P, Q ) = p * L`: the sign of `p` is that of the area
    // of the corner, and `t` grows from `P` towards `Q`.
    const TF ux = dx / L, uy = dy / L;
    const TF nx = uy, ny = -ux;
    const TF p = nx * P[ 0 ] + ny * P[ 1 ];
    const TF ap = p < 0 ? -p : p;
    if ( ! ( ap > 0 ) )                     // the origin IS on the line: flat corner
        return 0;

    const TF t0 = ux * P[ 0 ] + uy * P[ 1 ];
    const TF t1 = ux * Q[ 0 ] + uy * Q[ 1 ];

    TF acc = 0;
    if ( ap >= tail_cut ) {
        // the gaussian is worth nothing along the whole line: only the Lorentzian remains
        acc = sdot::atan( t1 / ap ) - sdot::atan( t0 / ap );
    } else {
        // The TAILS first, each bounded by the segment itself: a segment entirely
        // beyond `tail_cut` on one side only has no core at all, and its tail goes from `t0` to
        // `t1`, not from `tail_cut` to `t1`. Clipping "symmetrically" would count `[ tail_cut, t0 ]`
        // too much -- a clearly visible share of angle when the edge is long and grazes the origin.
        if ( t0 < -tail_cut ) {
            const TF e = t1 < -tail_cut ? t1 : -tail_cut;
            acc += sdot::atan( e / ap ) - sdot::atan( t0 / ap );
        }
        if ( t1 > tail_cut ) {
            const TF b = t0 > tail_cut ? t0 : tail_cut;
            acc += sdot::atan( t1 / ap ) - sdot::atan( b / ap );
        }

        const TF c0 = t0 > -tail_cut ? t0 : -tail_cut;
        const TF c1 = t1 <  tail_cut ? t1 :  tail_cut;

        // the core, by composite Gauss-Legendre. The integrand has a scale `>= 1` there (the factor
        // `1 - exp` cancels the Lorentzian peak when `p` is small), so a few panels
        // suffice whatever the configuration.
        // Panels of at most half a unit, up to `nb_panels`: the same panels on the long edges, ONE ( 8 exponentials instead of 32 )
        // on the short edges of a fine diagram -- the card's rule ( `gpu/Density2D.cuh::gauss_wedge_pt` ).
        if ( c1 > c0 ) {
            const int nc = int( sdot::ceil( ( c1 - c0 ) * 2 ) ), np = nc < 1 ? 1 : ( nc > nb_panels ? nb_panels : nc );
            const TF h = ( c1 - c0 ) / ( 2 * np );
            for ( int k = 0; k < np; ++k ) {
                const TF m = c0 + ( 2 * k + 1 ) * h;
                for ( int j = 0; j < 4; ++j ) {
                    for ( int sg = -1; sg <= 1; sg += 2 ) {
                        const TF t = m + sg * h * TF( detail::gl8_x[ j ] );
                        const TF r2 = ap * ap + t * t;
                        acc += h * TF( detail::gl8_w[ j ] ) * ( 1 - sdot::exp( - r2 / 2 ) ) * ap / r2;
                    }
                }
            }
        }
    }

    return ( p < 0 ? -acc : acc ) / two_pi;
}

/// the standard normal measure of the triangle `ys` ( positive, any orientation )
template<class TF>
HD TF std_triangle_measure( const auto &ys, TF tail_cut, int nb_panels ) {
    const TF s = std_wedge_measure<TF>( ys[ 0 ], ys[ 1 ], tail_cut, nb_panels )
               + std_wedge_measure<TF>( ys[ 1 ], ys[ 2 ], tail_cut, nb_panels )
               + std_wedge_measure<TF>( ys[ 2 ], ys[ 0 ], tail_cut, nb_panels );
    // the signed sum carries the ORIENTATION of the triangle; the measure has none.
    return s < 0 ? -s : s;
}

/// an edge `A -> B` of the triangle `A, B, C`: the OUTWARD normal, the signed distance `y . n` ( constant along the edge ),
/// `Int phi ds` and `Int phi lambda_A ds`
template<class TF>
struct StdEdgeInfo { Vector<TF,2> n; TF p; TF j0; TF j1a; };

template<class TF>
HD StdEdgeInfo<TF> std_edge_info( const auto &A, const auto &B, const auto &C ) {
    const TF two_pi = TF( 6.283185307179586476925286766559 );
    const TF sq_2pi = TF( 2.5066282746310005024157652848110 );

    StdEdgeInfo<TF> res{ Vector<TF,2>::zeros(), 0, 0, 0 };

    const TF dx = B[ 0 ] - A[ 0 ], dy = B[ 1 ] - A[ 1 ];
    const TF L2 = dx * dx + dy * dy;
    if ( ! ( L2 > 0 ) )
        return res;
    const TF L = sdot::sqrt( L2 );

    // the OUTWARD normal: the one that points away from the third vertex
    TF nx = dy / L, ny = -dx / L;
    if ( nx * ( C[ 0 ] - A[ 0 ] ) + ny * ( C[ 1 ] - A[ 1 ] ) > 0 ) { nx = -nx; ny = -ny; }
    res.n[ 0 ] = nx;
    res.n[ 1 ] = ny;
    res.p = nx * A[ 0 ] + ny * A[ 1 ];      // constant along the edge

    // `| A + s ( B - A ) |^2 = L^2 ( s - s0 )^2 + p^2`: the foot of the perpendicular, and the
    // distance to the line. That is what pulls out an `exp( -p^2/2 )` as a factor and leaves a
    // 1D gaussian, hence `erf`s.
    const TF s0 = - ( A[ 0 ] * dx + A[ 1 ] * dy ) / L2;
    TF p2 = A[ 0 ] * A[ 0 ] + A[ 1 ] * A[ 1 ] - L2 * s0 * s0;
    if ( p2 < 0 ) p2 = 0;

    const TF u0 = - L * s0, u1 = L * ( 1 - s0 );
    const TF d_phi = detail::std_normal_cdf( u1 ) - detail::std_normal_cdf( u0 );
    const TF e = sdot::exp( - p2 / 2 );

    res.j0  = e * d_phi / sq_2pi;
    res.j1a = ( e / two_pi ) * ( ( 1 - s0 ) * sq_2pi * d_phi
                               - ( sdot::exp( - u0 * u0 / 2 ) - sdot::exp( - u1 * u1 / 2 ) ) / L );
    return res;
}

/// an edge `A -> B` of the triangle `A, B, C`: its OUTWARD normal and `y . n`, then `Int_e q phi ds` ( `j` ) and
/// `Int_e q phi lambda_A ds` ( `ja` ) for `q = 1, y_0, y_1, |y|^2`
template<class TF>
struct StdEdgeMoments { Vector<TF,2> n; TF p; TF j[ 4 ]; TF ja[ 4 ]; };

template<class TF>
HD StdEdgeMoments<TF> std_edge_moments( const auto &A, const auto &B, const auto &C ) {
    const TF two_pi = TF( 6.283185307179586476925286766559 );
    const TF sq_2pi = TF( 2.5066282746310005024157652848110 );

    StdEdgeMoments<TF> res{ Vector<TF,2>::zeros(), 0, { 0, 0, 0, 0 }, { 0, 0, 0, 0 } };

    const TF dx = B[ 0 ] - A[ 0 ], dy = B[ 1 ] - A[ 1 ];
    const TF L2 = dx * dx + dy * dy;
    if ( ! ( L2 > 0 ) )
        return res;
    const TF L = sdot::sqrt( L2 );

    TF nx = dy / L, ny = -dx / L;
    if ( nx * ( C[ 0 ] - A[ 0 ] ) + ny * ( C[ 1 ] - A[ 1 ] ) > 0 ) { nx = -nx; ny = -ny; }
    res.n[ 0 ] = nx;
    res.n[ 1 ] = ny;
    res.p = nx * A[ 0 ] + ny * A[ 1 ];

    // as in `edge_info`: along the edge, `y = F + u t`, `F` the foot of the perpendicular, `u` in `[ u0, u1 ]`, and
    // `phi = exp( -p^2/2 ) exp( -u^2/2 ) / 2pi`. The weights are polynomials of `u` -- `lambda_A = a0 + a1 u`, `y_c = F_c + u t_c`,
    // `|y|^2 = p^2 + u^2` -- so everything is `G_m = Int u^m exp( -u^2/2 ) du`, `m <= 3`: one `erf` and two `exp`s
    const TF s0 = - ( A[ 0 ] * dx + A[ 1 ] * dy ) / L2;
    TF p2 = A[ 0 ] * A[ 0 ] + A[ 1 ] * A[ 1 ] - L2 * s0 * s0;
    if ( p2 < 0 ) p2 = 0;
    const TF u0 = - L * s0, u1 = L * ( 1 - s0 );
    const TF e0 = sdot::exp( - u0 * u0 / 2 ), e1 = sdot::exp( - u1 * u1 / 2 );
    const TF G0 = sq_2pi * ( detail::std_normal_cdf( u1 ) - detail::std_normal_cdf( u0 ) );
    const TF G1 = e0 - e1;
    const TF G2 = G0 + u0 * e0 - u1 * e1;
    const TF G3 = ( u0 * u0 + 2 ) * e0 - ( u1 * u1 + 2 ) * e1;

    const TF a0 = 1 - s0, a1 = - 1 / L;
    const TF H0 = a0 * G0 + a1 * G1, H1 = a0 * G1 + a1 * G2, H2 = a0 * G2 + a1 * G3;
    const TF E = sdot::exp( - p2 / 2 ) / two_pi;
    const TF F[ 2 ] = { A[ 0 ] + s0 * dx, A[ 1 ] + s0 * dy };
    const TF t[ 2 ] = { dx / L, dy / L };

    res.j[ 0 ] = E * G0;
    res.ja[ 0 ] = E * H0;
    for ( int c = 0; c < 2; ++c ) {
        res.j[ 1 + c ] = E * ( F[ c ] * G0 + t[ c ] * G1 );
        res.ja[ 1 + c ] = E * ( F[ c ] * H0 + t[ c ] * H1 );
    }
    res.j[ 3 ] = E * ( p2 * G0 + G2 );
    res.ja[ 3 ] = E * ( p2 * H0 + H2 );
    return res;
}

} // namespace integration
} // namespace sdot
