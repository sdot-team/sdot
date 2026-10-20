#pragma once

// =====================================================================================
// A RADIAL INTEGRAND AROUND EACH SEED: the cell of seed `i` stays its Laguerre polytope, and what is integrated over it is
//
//     rho( x ) exp( ( w_i - |x - p_i|^2 ) / ( 2 eps ) )
//
// -- the density of the Moreau-Yosida regularization of the entropy ( Leclerc, Mérigot, Santambrogio, Stra, *Lagrangian
// discretization of crowd motion and linear diffusion*, 2020, prop. 5.3 ): the measure `sigma` closest to the diracs, for
// `W_2^2 / ( 2 eps ) + int sigma log sigma`, is this integrand on the Laguerre cells whose masses are those of the diracs.
//
// As with the ball of a sink ( `InBall.h` ), it is the INTEGRATION that changes, not the cell: the distribution cuts the cell into
// its pieces of constant density ( `Image`, `Box`, `ConvexHull`, or none ), and each piece is handed over with a density that knows
// the seed ( `RadialExpDensity` ). On a piece, the integrand is a GAUSSIAN centered on the seed, of width `s = sqrt( eps )` and of
// mass `rho exp( w / ( 2 eps ) ) 2 pi eps`: its integrals are the exact 2D reduction of `SumOfGaussians` ( `StdGaussian2.h` ).
//
//     integrate_over_simplex( pts )                       int_T rho g
//     integrate_moment_over_simplex<K>( pts )             int_T p_K rho g ( `diagram/Moments.h` )
//     facet_mass( pc, cut )                               int_facet rho g ( the hessian: the integrand is continuous across a
//                                                         facet, `|x - p_i|^2 - w_i = |x - p_j|^2 - w_j` there )
//     weight_mass( pc )                                   d / d w_i of the integral over the piece: the integral itself / ( 2 eps )
//                                                         ( the diagonal of the hessian, beyond its facets: no gauge any more )
//
// 2D. The derivatives ( positions, weights, the density's values ) are closed forms too, see below.
// =====================================================================================

#include "StdGaussian2.h"

namespace sdot {
namespace integration {

template<class Dens,class TF_,int D>
struct RadialExpDensity {
    using TF = TF_;
    static constexpr int ct_dim = D;
    static constexpr bool is_constant = false;
    static constexpr bool exact_moments = true;          ///< its moments are closed forms ( `diagram/Moments.h` )
    static constexpr bool depends_on_seed = true;
    static_assert( D == 2, "a radial integrand ( `radial_func` ): in 2D only for now" );
    static_assert( Dens::is_constant, "a radial integrand ( `radial_func` ): only for a density constant on each piece ( `Image` of "
                                      "degree 0, `Box`, `ConvexHull`, or none )" );

    static constexpr TF tail_cut  = 8;                   ///< ( those of `SumOfGaussians` )
    static constexpr int nb_panels = 4;

    Dens          base;                                 ///< the density of the piece
    Vector<TF,D>  center;                               ///< the seed
    TF            weight;                               ///< its weight
    TF            eps;

    HD TF sigma() const { return sdot::sqrt( eps ); }

    /// the mass of the gaussian, `rho exp( w / ( 2 eps ) ) 2 pi eps`: `int_T rho g = mass * ( standard normal measure of T )`
    HD TF mass() const { return TF( base.value ) * sdot::exp( weight / ( 2 * eps ) ) * TF( 6.283185307179586476925286766559 ) * eps; }

    HD TF value_at( const auto &x ) const {
        TF r2 = 0;
        for ( int d = 0; d < D; ++d )
            r2 += ( TF( x[ d ] ) - center[ d ] ) * ( TF( x[ d ] ) - center[ d ] );
        return TF( base.value ) * sdot::exp( ( weight - r2 ) / ( 2 * eps ) );
    }

    /// the triangle in the standard frame of the gaussian
    HD auto standard( const auto &pts ) const {
        const TF is = 1 / sigma();
        return Vector<Vector<TF,2>,3>( Function(), [&]( PI k ) {
            return Vector<TF,2>( Function(), [&]( PI d ) { return ( TF( pts[ k ][ d ] ) - center[ d ] ) * is; } );
        } );
    }

    HD TF integrate_over_simplex( const auto &pts ) const {
        return mass() * std_triangle_measure<TF>( standard( pts ), tail_cut, nb_panels );
    }

    /// `K < 2`: `int x_K`, `K == 2`: `int |x|^2`, `K == 3`: `int 1` -- as `SumOfGaussians::integrate_moment_over_simplex`: in the
    /// standard frame, `int y phi = - sum_e n_e J_e` and `int |y|^2 phi = 2 M - sum_e p_e J_e`
    template<int K>
    HD TF integrate_moment_over_simplex( const auto &pts ) const {
        const auto ys = standard( pts );
        const TF M = std_triangle_measure<TF>( ys, tail_cut, nb_panels );
        if constexpr ( K == 3 )
            return mass() * M;
        else {
            TF N[ 2 ] = { 0, 0 }, P = 0;
            for ( int k = 0; k < 3; ++k ) {
                const auto ei = std_edge_info<TF>( ys[ k ], ys[ ( k + 1 ) % 3 ], ys[ ( k + 2 ) % 3 ] );
                N[ 0 ] += ei.n[ 0 ] * ei.j0;
                N[ 1 ] += ei.n[ 1 ] * ei.j0;
                P += ei.p * ei.j0;
            }
            const TF s = sigma(), c[ 2 ] = { center[ 0 ], center[ 1 ] };
            if constexpr ( K < 2 )
                return mass() * ( c[ K ] * M - s * N[ K ] );
            else
                return mass() * ( ( c[ 0 ] * c[ 0 ] + c[ 1 ] * c[ 1 ] ) * M - 2 * s * ( c[ 0 ] * N[ 0 ] + c[ 1 ] * N[ 1 ] ) + s * s * ( 2 * M - P ) );
        }
    }

    HD void integrate_moments_over_simplex( const auto &pts, TF &m, auto &mx, TF &m2 ) const {
        m += integrate_moment_over_simplex<3>( pts );
        mx[ 0 ] += integrate_moment_over_simplex<0>( pts );
        mx[ 1 ] += integrate_moment_over_simplex<1>( pts );
        m2 += integrate_moment_over_simplex<2>( pts );
    }

    /// `int rho g` on the edge `[ v_cut, v_cut+1 ]` of the piece: an `erf`, the distance to its line being constant
    HD TF facet_mass( const auto &pc, int cut ) const {
        const int nb = pc.nb_vertices();
        const int j = cut + 1 < nb ? cut + 1 : 0;
        const TF ax = TF( pc.coord( cut, 0 ) ), ay = TF( pc.coord( cut, 1 ) );
        const TF ex = TF( pc.coord( j, 0 ) ) - ax, ey = TF( pc.coord( j, 1 ) ) - ay;
        const TF L = sdot::sqrt( ex * ex + ey * ey );
        if ( ! ( L > 0 ) )
            return 0;
        const TF ux = ex / L, uy = ey / L;
        const TF px = ax - center[ 0 ], py = ay - center[ 1 ];
        const TF d = - px * uy + py * ux;                // the signed distance from the seed to the line
        const TF t0 = px * ux + py * uy, t1 = t0 + L;
        const TF is2 = TF( 0.70710678118654752440 ) / sigma();
        return TF( base.value ) * sdot::exp( ( weight - d * d ) / ( 2 * eps ) ) * sdot::sqrt( TF( 1.5707963267948966192313216916398 ) * eps )
             * ( sdot::erf( t1 * is2 ) - sdot::erf( t0 * is2 ) );
    }

    /// `d / d w` of the integral over the piece ( convex: a fan from its first vertex )
    HD TF weight_mass( const auto &pc ) const {
        const int n = pc.nb_vertices();
        TF res = 0;
        for ( int i = 1; i + 1 < n; ++i ) {
            const TF pts[ 3 ][ 2 ] = { { TF( pc.coord( 0, 0 ) ), TF( pc.coord( 0, 1 ) ) },
                                       { TF( pc.coord( i, 0 ) ), TF( pc.coord( i, 1 ) ) },
                                       { TF( pc.coord( i + 1, 0 ) ), TF( pc.coord( i + 1, 1 ) ) } };
            res += integrate_over_simplex( pts );
        }
        return res / ( 2 * eps );
    }

    // ---- the derivatives ------------------------------------------------------------------------------------------------
    // As `SumOfGaussians::integrate_moment_over_simplex_bwd`: moving a vertex sweeps its two edges ( Reynolds ), `d I / d P_k =
    // sum_{e ∋ k} n_e int_e p rho g lambda_k ds` -- `erf`s and `exp`s ( `std_edge_moments` ); the center of the gaussian, the seed,
    // follows from the invariance by translation, `d I / d c = int grad p rho g - sum_k d I / d P_k`. Its weight multiplies the
    // integrand by `exp( w / ( 2 eps ) )`: `d I / d w = I / ( 2 eps )`; the density is a factor: `d I / d rho = I / rho`.

    /// the moment `K` over the triangle `pts` ( returned ), divided by `rho` ( `I_rho` ), its derivatives with respect to the
    /// vertices ( `dP` ) and to the center ( `dc` )
    template<int K>
    HD TF moment_grads( const auto &pts, TF &I_rho, TF ( &dP )[ 3 ][ 2 ], TF ( &dc )[ 2 ] ) const {
        const TF s = sigma(), u = sdot::exp( weight / ( 2 * eps ) ) * TF( 6.283185307179586476925286766559 ) * eps, w = TF( base.value ) * u;
        const TF c[ 2 ] = { center[ 0 ], center[ 1 ] };
        const auto ys = standard( pts );

        // the moment's polynomial in the standard frame, `p( c + s y )`, on the basis `1, y_0, y_1, |y|^2`
        TF a[ 4 ] = { 0, 0, 0, 0 };
        if constexpr ( K < 2 ) {
            a[ 0 ] = c[ K ];
            a[ 1 + K ] = s;
        } else if constexpr ( K == 2 ) {
            a[ 0 ] = c[ 0 ] * c[ 0 ] + c[ 1 ] * c[ 1 ];
            a[ 1 ] = 2 * s * c[ 0 ];
            a[ 2 ] = 2 * s * c[ 1 ];
            a[ 3 ] = s * s;
        } else
            a[ 0 ] = 1;

        // the vertices in the standard frame ( `d / d y_k`, per unit mass ), and `N = sum_e n_e J_e`, `Pp = sum_e p_e J_e`
        TF dm[ 3 ][ 2 ] = { { 0, 0 }, { 0, 0 }, { 0, 0 } }, N[ 2 ] = { 0, 0 }, Pp = 0;
        for ( int k = 0; k < 3; ++k ) {
            const int k1 = ( k + 1 ) % 3, k2 = ( k + 2 ) % 3;
            const auto em = std_edge_moments<TF>( ys[ k ], ys[ k1 ], ys[ k2 ] );
            TF ja = 0, jb = 0;
            for ( int q = 0; q < 4; ++q ) {
                ja += a[ q ] * em.ja[ q ];
                jb += a[ q ] * ( em.j[ q ] - em.ja[ q ] );
            }
            for ( int d = 0; d < 2; ++d ) {
                dm[ k  ][ d ] += ja * em.n[ d ];
                dm[ k1 ][ d ] += jb * em.n[ d ];
                N[ d ] += em.n[ d ] * em.j[ 0 ];
            }
            Pp += em.p * em.j[ 0 ];
        }
        const TF M = std_triangle_measure<TF>( ys, tail_cut, nb_panels );

        // `I / w` and `int grad p rho g / w` ( G ), from `int y phi = - N`, `int |y|^2 phi = 2 M - Pp`
        TF Iw = 0, G[ 2 ] = { 0, 0 };
        if constexpr ( K < 2 ) {
            Iw = c[ K ] * M - s * N[ K ];
            G[ K ] = M;
        } else if constexpr ( K == 2 ) {
            Iw = ( c[ 0 ] * c[ 0 ] + c[ 1 ] * c[ 1 ] ) * M - 2 * s * ( c[ 0 ] * N[ 0 ] + c[ 1 ] * N[ 1 ] ) + s * s * ( 2 * M - Pp );
            for ( int d = 0; d < 2; ++d )
                G[ d ] = 2 * ( c[ d ] * M - s * N[ d ] );
        } else
            Iw = M;

        for ( int d = 0; d < 2; ++d ) {
            dc[ d ] = w * G[ d ];
            for ( int k = 0; k < 3; ++k ) {
                dP[ k ][ d ] = w * dm[ k ][ d ] / s;
                dc[ d ] -= dP[ k ][ d ];
            }
        }
        I_rho = u * Iw;
        return w * Iw;
    }

    template<int K>
    HD void integrate_moment_over_simplex_bwd( const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const {
        TF I_rho, dP[ 3 ][ 2 ], dc[ 2 ];
        moment_grads<K>( pts, I_rho, dP, dc );
        for ( int k = 0; k < 3; ++k )
            for ( int d = 0; d < 2; ++d )
                grad_pts[ k ][ d ] += g * dP[ k ][ d ];
        base.add_value_grad( grad_dist, g * I_rho );
    }

    HD void integrate_over_simplex_bwd( const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const {
        integrate_moment_over_simplex_bwd<3>( pts, g, grad_pts, grad_dist );
    }

    /// the seed's share of the moment `K` over the piece `pc` ( a fan from its first vertex ): its position is the center of the
    /// gaussian, its weight scales it
    template<int K>
    HD void seed_bwd_moment( const auto &pc, TF g, TF *gp, TF &gw ) const {
        const int n = pc.nb_vertices();
        for ( int i = 1; i + 1 < n; ++i ) {
            const TF pts[ 3 ][ 2 ] = { { TF( pc.coord( 0, 0 ) ), TF( pc.coord( 0, 1 ) ) },
                                       { TF( pc.coord( i, 0 ) ), TF( pc.coord( i, 1 ) ) },
                                       { TF( pc.coord( i + 1, 0 ) ), TF( pc.coord( i + 1, 1 ) ) } };
            TF I_rho, dP[ 3 ][ 2 ], dc[ 2 ];
            const TF I = moment_grads<K>( pts, I_rho, dP, dc );
            gp[ 0 ] += g * dc[ 0 ];
            gp[ 1 ] += g * dc[ 1 ];
            gw += g * I / ( 2 * eps );
        }
    }

    HD void seed_bwd( const auto &pc, TF g, TF *gp, TF &gw ) const { seed_bwd_moment<3>( pc, g, gp, gw ); }
};

/// a distribution whose pieces are integrated against the radial factor of the seed `center` ( see above )
template<class Dist,class TF,int D>
struct RadialExpDist {
    static constexpr bool cuts_pieces = [] { if constexpr ( requires { Dist::cuts_pieces; } ) return bool( Dist::cuts_pieces ); else return false; }();

    Dist          dist;
    Vector<TF,D>  center;
    TF            weight;
    TF            eps;

    HD void for_each_piece( const auto &cell, auto &&ws, auto &&func ) const {
        dist.for_each_piece( cell, ws, [&]( const auto &pc, const auto &dens ) {
            func( pc, RadialExpDensity<DECAYED_TYPE_OF( dens ),TF,D>{ dens, center, weight, eps } );
        } );
    }
};

template<class TF,int D>
HD auto radial_exp( const auto &dist, const Vector<TF,D> &center, TF weight, TF eps ) {
    return RadialExpDist<DECAYED_TYPE_OF( dist ),TF,D>{ dist, center, weight, eps };
}

} // namespace integration
} // namespace sdot
