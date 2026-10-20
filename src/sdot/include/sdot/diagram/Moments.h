#pragma once

// =====================================================================================
// A MOMENT OF A DENSITY AS A DENSITY: `x_k rho( x )`, `|x|^2 rho( x )` or `rho( x )` itself, so that what the
// diagram knows how to do with a density -- integrate it over a cell, and send the cotangent back onto the
// seeds, the weights and the density's own parameters ( `measures`, `measures_bwd` ) -- it does for a MOMENT, without
// a line of adjoint written for it.
//
//     sdot::diagram::moment_dist<K>( dist )        K < D : int x_K rho     K == D : int |x|^2 rho     K == D + 1 : int rho
//
//     K == D + 2 : the SPHERE of partial transport inside the cell, where the density is not zero ( `integration/InBall.h`; zero
//                  elsewhere ) -- a length in 2D, an area in 3D: what a surface energy counts ( `PowerDiagram.sphere_areas` )
//
// A density that knows its moments and their adjoints in closed form says so ( `exact_moments`: `MeshElementDensity`, `SumOfGaussians`
// in 2D and 3D, `InBallDensity` ) and integrates them itself. Elsewhere the weight is a polynomial, so `rho( x ) p( x )` has the
// value, the gradient and the parameters' derivative that the product rule says, and the integral over a piece is the ADAPTIVE
// QUADRATURE of `PointwiseDensity` -- exact up to degree 2 ( constant pieces ), refined beyond. Either way what is differentiated
// is what is computed.
//
// What it is for: the transport COST and the BARYCENTERS, differentiated by `SdotPlanNd` ( `_attach_derivatives` ).
// =====================================================================================

#include "../PointwiseDensity.h"
#include <loom/support/containers/Matrix.h>

namespace sdot {
namespace diagram {

/// a density that integrates its moments and their adjoints EXACTLY says so ( `integration/InBall.h` ) -- a marker rather than a
/// `requires` on its member templates, which nvcc does not always see
template<class Dens>
HD constexpr bool exact_moments() {
    if constexpr ( requires { Dens::exact_moments; } ) return bool( Dens::exact_moments );
    else return false;
}

/// the density of one piece, multiplied by the polynomial `K` ( see above ), in dimension `D`
template<class Dens,int D,int K>
struct MomentDensity {
    using TF = typename Dens::TF;
    static constexpr int ct_dim = D;
    static constexpr bool is_constant = false;

    Dens dens;

    /// the density itself: `PointwiseDensity` wraps the one that evaluates
    HD decltype( auto ) inner() const {
        if constexpr ( requires { dens.dens; } ) return ( dens.dens );
        else                                      return ( dens );
    }

    HD static TF poly( const auto &x ) {
        if constexpr ( K < D )           return TF( x[ K ] );
        else if constexpr ( K == D )     return norm_2_p2( x );
        else if constexpr ( K == D + 1 ) return TF( 1 );
        else                             return TF( 0 );   // `D + 2`, the sphere: none on a piece that no ball cuts
    }

    HD static auto poly_grad( const auto &x ) {
        return Vector<TF,D>::with_func( [&]( PI c ) {
            if constexpr ( K < D )       return TF( c == PI( K ) );
            else if constexpr ( K == D ) return TF( 2 ) * TF( x[ c ] );
            else                         return TF( 0 );
        } );
    }

    HD TF rho_at( const auto &x ) const {
        if constexpr ( Dens::is_constant ) return TF( dens.value );
        else                               return TF( inner().value_at( x ) );
    }

    HD TF value_at( const auto &x ) const { return poly( x ) * rho_at( x ); }

    HD auto gradient_at( const auto &x ) const {
        const TF r = rho_at( x ), p = poly( x );
        const auto pg = poly_grad( x );
        if constexpr ( Dens::is_constant ) {
            return Vector<TF,D>::with_func( [&]( PI c ) { return r * pg[ c ]; } );
        } else {
            const auto rg = inner().gradient_at( x );
            return Vector<TF,D>::with_func( [&]( PI c ) { return p * TF( rg[ c ] ) + r * pg[ c ]; } );
        }
    }

    HD void add_value_grad_at( auto &&grad_dist, const auto &x, TF g ) const {
        if constexpr ( Dens::is_constant ) dens.add_value_grad( grad_dist, g * poly( x ) );
        else                               inner().add_value_grad_at( grad_dist, x, g * poly( x ) );
    }

    HD TF integrate_over_simplex( const auto &pts ) const {
        // a density that knows its moments EXACTLY says so; the others go through the quadrature
        if constexpr ( exact_moments<Dens>() )
            return dens.template integrate_moment_over_simplex<K>( pts );
        else
            return PointwiseDensity<MomentDensity>{ *this }.integrate_over_simplex( pts );
    }

    HD void integrate_over_simplex_bwd( const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const {
        // ... and its adjoint too ( `integration/InBall.h` )
        if constexpr ( exact_moments<Dens>() )
            dens.template integrate_moment_over_simplex_bwd<K>( pts, g, grad_pts, grad_dist );
        else
            PointwiseDensity<MomentDensity>{ *this }.integrate_over_simplex_bwd( pts, g, grad_pts, grad_dist );
    }

    /// an integrand that depends on the seed of the cell ( the ball of partial transport ): its share, for this moment
    static constexpr bool depends_on_seed = [] { if constexpr ( requires { Dens::depends_on_seed; } ) return bool( Dens::depends_on_seed ); else return false; }();

    HD void seed_bwd( const auto &pc, TF g, TF *gp, TF &gw ) const {
        if constexpr ( depends_on_seed )
            dens.template seed_bwd_moment<K>( pc, g, gp, gw );
    }

    /// what the derivative with respect to the FRAME of the density needs of a simplex ( `AffineAcc`, `diagram/Ops.h` ), for a
    /// density constant on it: `J = Int p rho` on the diagonal of `X`, `S_ab = Int d_a p ( x - o )_b rho` in `X`, `s = Int grad p rho`
    /// in `y` -- the polynomial of the moment stays in `x` when the density moves. Closed forms: `Int x = |S| m`,
    /// `Int x_a x_b = |S| ( sum_i P_ia P_ib + sum_i P_ia sum_i P_ib ) / ( ( D + 1 )( D + 2 ) )`.
    /// ( a density that is not constant on the piece gives them itself: `affine_terms_moment<K>`, `PolynomialPixel` )
    static constexpr bool has_affine_terms = Dens::is_constant || [] { if constexpr ( requires { Dens::has_affine_terms; } ) return bool( Dens::has_affine_terms ); else return false; }();
    HD void affine_terms( const auto &pts, TF g, auto &acc ) const {
        if constexpr ( Dens::is_constant )
            affine_terms_of_constant( pts, g, acc );
        else
            dens.template affine_terms_moment<K>( pts, g, acc );
    }
    HD void affine_terms_of_constant( const auto &pts, TF g, auto &acc ) const {
        Matrix<TF,D> M;
        for ( int r = 0; r < D; ++r )
            for ( int c = 0; c < D; ++c )
                M( r, c ) = TF( pts[ c + 1 ][ r ] ) - TF( pts[ 0 ][ r ] );
        TF vol = M.determinant();
        vol = vol < 0 ? -vol : vol;
        for ( int i = 2; i <= D; ++i )
            vol /= i;
        const TF v = g * TF( dens.value );
        TF sx[ D ], ix[ D ], ixx[ D ][ D ];
        for ( int a = 0; a < D; ++a ) {
            sx[ a ] = 0;
            for ( int i = 0; i <= D; ++i )
                sx[ a ] += TF( pts[ i ][ a ] );
            ix[ a ] = vol * sx[ a ] / ( D + 1 );
        }
        for ( int a = 0; a < D; ++a )
            for ( int b = 0; b < D; ++b ) {
                TF pp = 0;
                for ( int i = 0; i <= D; ++i )
                    pp += TF( pts[ i ][ a ] ) * TF( pts[ i ][ b ] );
                ixx[ a ][ b ] = vol * ( pp + sx[ a ] * sx[ b ] ) / ( ( D + 1 ) * ( D + 2 ) );
            }
        TF J = 0;
        if constexpr ( K < D ) {
            J = ix[ K ];
            acc.y[ K ] += v * vol;
            for ( int b = 0; b < D; ++b )
                acc.X[ K ][ b ] += v * ( ix[ b ] - acc.o[ b ] * vol );
        } else if constexpr ( K == D ) {
            for ( int a = 0; a < D; ++a ) {
                J += ixx[ a ][ a ];
                acc.y[ a ] += 2 * v * ix[ a ];
                for ( int b = 0; b < D; ++b )
                    acc.X[ a ][ b ] += 2 * v * ( ixx[ a ][ b ] - acc.o[ b ] * ix[ a ] );
            }
        } else if constexpr ( K == D + 1 )
            J = vol;
        for ( int a = 0; a < D; ++a )
            acc.X[ a ][ a ] += v * J;
    }
};

/// a distribution whose pieces carry the moment `K` of their density
template<class Dist,int K>
struct MomentDist {
    static constexpr int moment_order = K;               ///< ( what `cell_dist` reads to put an integrand UNDER the moment )
    static constexpr bool cuts_pieces = [] { if constexpr ( requires { Dist::cuts_pieces; } ) return bool( Dist::cuts_pieces ); else return false; }();

    Dist dist;

    HD void for_each_piece( const auto &cell, auto &&ws, auto &&func ) const {
        dist.for_each_piece( cell, ws, [&]( const auto &pc, const auto &dens ) {
            func( pc, MomentDensity<std::decay_t<decltype( dens )>, std::decay_t<decltype( pc )>::ct_dim, K>{ dens } );
        } );
    }

    // the derivative with respect to the frame of the density ( `AffineAcc`, `diagram/Ops.h` ): that of the distribution
    static constexpr bool is_affine = [] { if constexpr ( requires { Dist::is_affine; } ) return bool( Dist::is_affine ); else return false; }();
    HD bool affine_wanted( const auto &grad_dist ) const { return dist.affine_wanted( grad_dist ); }
    HD void affine_origin( auto *o ) const { dist.affine_origin( o ); }
    HD void add_affine_grad( auto &&grad_dist, const auto &acc ) const { dist.add_affine_grad( grad_dist, acc ); }
};

template<int K>
HD auto moment_dist( const auto &dist ) {
    return MomentDist<std::decay_t<decltype( dist )>, K>{ dist };
}

} // namespace diagram
} // namespace sdot
