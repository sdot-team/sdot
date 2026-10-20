#pragma once

#include <loom/support/common_macros.h>
#include <loom/support/containers/Vector.h>
#include <loom/support/containers/Matrix.h>
#include <loom/support/atomic_add.h>
#include <loom/support/math.h>
#include "integration/SimplexRule.h"

namespace sdot {

// THE DENSITY OF ONE PIXEL of an image of degree `P > 0` ( `Image( degree = P )` ), as the piece contract wants it
// ( `distributions/Distribution.py`, the case `is_constant == false` ): the piece is `cell INTERSECT pixel`, the density a
// tensor-product polynomial of degree `P` along each axis, written in the BERNSTEIN basis of the pixel,
//
//     rho( x ) = sum_alpha c_alpha prod_a B_{ alpha_a, P }( t_a ),        t_a = ( n_a . x - off_a ) / h_a  in [ 0, 1 ],
//
// `n_a` the normal of axis `a` ( `Image::_for_each_piece` ), `off_a` the low knot of the pixel shifted by the origin, `h_a` its
// width in grid units, `B_{ i, P }( t ) = C( P, i ) t^i ( 1 - t )^{ P - i }`. The coefficients are flattened in C order ( the
// first axis the slowest ). `c >= 0` gives `rho >= 0`, and the mass of the pixel is its volume times the mean of its `c`.
//
// `rho` is a polynomial of total degree `D P` in `x`, so everything is EXACT with the rule of `integration/SimplexRule.h`: the
// integral on a simplex, the moments ( `exact_moments` ), the facets ( the hessian, `facet_mass` ), and the adjoints, which are
// those of what is computed. The derivative with respect to the frame and the origin of the image ( `AffineAcc`, `diagram/Ops.h` )
// needs the same integrals with the moment's polynomial: `affine_terms`.
template<class TF_,int D,int P,class Idx>
struct PolynomialPixel {
    using TF = TF_;
    static constexpr int  ct_dim         = D;
    static constexpr bool is_constant    = false;
    static constexpr bool exact_moments  = true;
    static constexpr bool has_affine_terms = true;          ///< ( the mass: `affine_terms`; a moment: `affine_terms_moment` )
    static constexpr int  nb_coeffs      = [] { int r = 1; for ( int a = 0; a < D; ++a ) r *= P + 1; return r; }();
    static constexpr int  degree         = D * P;            ///< the total degree in `x`

    TF                    c[ nb_coeffs ];                    ///< the Bernstein coefficients of the pixel
    Vector<Vector<TF,D>,D> nrm;
    TF                    off[ D ], inv_h[ D ];
    Idx                   idx;                               ///< the pixel, to send the cotangents of its coefficients

    // ---- evaluation --------------------------------------------------------------------------------------------------------

    HD static constexpr TF binomial( int i ) { TF r = 1; for ( int k = 1; k <= i; ++k ) r = r * TF( P - k + 1 ) / TF( k ); return r; }

    /// `B_{ i, P }( t )` for `i <= P`, and with `db` their derivatives
    HD static void bernstein( TF t, TF *b, TF *db ) {
        TF tp[ P + 1 ], up[ P + 1 ];
        tp[ 0 ] = up[ 0 ] = 1;
        for ( int i = 1; i <= P; ++i ) {
            tp[ i ] = tp[ i - 1 ] * t;
            up[ i ] = up[ i - 1 ] * ( 1 - t );
        }
        for ( int i = 0; i <= P; ++i )
            b[ i ] = binomial( i ) * tp[ i ] * up[ P - i ];
        if ( db ) {
            // `B'_{ i, P } = P ( B_{ i - 1, P - 1 } - B_{ i, P - 1 } )`
            auto low = [&]( int i ) -> TF {
                if ( i < 0 || i > P - 1 ) return 0;
                TF r = 1;
                for ( int k = 1; k <= i; ++k ) r = r * TF( P - k ) / TF( k );
                return r * tp[ i ] * up[ P - 1 - i ];
            };
            for ( int i = 0; i <= P; ++i )
                db[ i ] = TF( P ) * ( low( i - 1 ) - low( i ) );
        }
    }

    HD TF t_of( int a, const auto &x ) const {
        TF s = - off[ a ];
        for ( int k = 0; k < D; ++k )
            s += nrm[ a ][ k ] * TF( x[ k ] );
        return s * inv_h[ a ];
    }

    /// the basis on each axis at `x` ( and their derivatives in `t` )
    HD void basis( const auto &x, TF ( &b )[ D ][ P + 1 ], TF ( *db )[ P + 1 ] ) const {
        for ( int a = 0; a < D; ++a )
            bernstein( t_of( a, x ), b[ a ], db ? db[ a ] : nullptr );
    }

    /// `func( flat, alpha )` for each multi-index, `alpha[ 0 ]` the slowest
    HD static void for_each_alpha( auto &&func ) {
        int al[ D ];
        for ( int a = 0; a < D; ++a )
            al[ a ] = 0;
        for ( int flat = 0; flat < nb_coeffs; ++flat ) {
            func( flat, al );
            for ( int a = D - 1; a >= 0; --a ) {
                if ( ++al[ a ] <= P ) break;
                al[ a ] = 0;
            }
        }
    }

    HD TF value_at( const auto &x ) const {
        TF b[ D ][ P + 1 ];
        basis( x, b, nullptr );
        TF s = 0;
        for_each_alpha( [&]( int flat, const int *al ) {
            TF w = c[ flat ];
            for ( int a = 0; a < D; ++a )
                w *= b[ a ][ al[ a ] ];
            s += w;
        } );
        return s;
    }

    HD Vector<TF,D> gradient_at( const auto &x ) const {
        TF b[ D ][ P + 1 ], db[ D ][ P + 1 ];
        basis( x, b, db );
        TF dt[ D ];
        for ( int a = 0; a < D; ++a )
            dt[ a ] = 0;
        for_each_alpha( [&]( int flat, const int *al ) {
            for ( int a = 0; a < D; ++a ) {
                TF w = c[ flat ];
                for ( int e = 0; e < D; ++e )
                    w *= e == a ? db[ e ][ al[ e ] ] : b[ e ][ al[ e ] ];
                dt[ a ] += w;
            }
        } );
        return Vector<TF,D>::with_func( [&]( PI k ) {
            TF s = 0;
            for ( int a = 0; a < D; ++a )
                s += dt[ a ] * inv_h[ a ] * nrm[ a ][ k ];
            return s;
        } );
    }

    /// `d rho( x ) / d c_alpha = prod_a B_{ alpha_a }( t_a )`, times `g`, into the coefficients of the pixel
    HD void add_value_grad_at( auto &&grad_dist, const auto &x, TF g ) const {
        if constexpr ( requires { grad_dist.coeffs; } ) {
            if constexpr ( DECAYED_TYPE_OF( grad_dist.coeffs )::is_valid ) {
                TF b[ D ][ P + 1 ];
                basis( x, b, nullptr );
                for_each_alpha( [&]( int flat, const int *al ) {
                    TF w = g;
                    for ( int a = 0; a < D; ++a )
                        w *= b[ a ][ al[ a ] ];
                    atomic_add( grad_dist.coeffs( idx.with_appended_value( SI( flat ) ) ).ref(), w );
                } );
            }
        }
    }

    // ---- the rule on a simplex of the cell ---------------------------------------------------------------------------------

    HD static TF factorial( int n ) { TF r = 1; for ( int i = 2; i <= n; ++i ) r *= i; return r; }

    HD static auto edge_matrix( const auto &Q ) {
        return Matrix<TF,D>::with_func( [&]( auto r, auto k ) { return TF( Q[ k + 1 ][ r ] ) - TF( Q[ 0 ][ r ] ); } );
    }

    /// `func( x, bary, w )` on the nodes of the rule exact up to `deg`, `w` including the volume of the simplex
    HD static void for_each_node( const auto &Q, int deg, auto &&func ) {
        const TF det = edge_matrix( Q ).determinant();
        const TF vol = ( det < 0 ? -det : det ) / factorial( D );
        integration::for_each_simplex_node<D,TF>( deg, [&]( const TF *bary, TF w ) {
            Vector<TF,D> x;
            for ( int k = 0; k < D; ++k ) {
                x[ k ] = 0;
                for ( int q = 0; q <= D; ++q )
                    x[ k ] += bary[ q ] * TF( Q[ q ][ k ] );
            }
            func( x, bary, vol * w );
        } );
    }

    /// the polynomial of moment `K` ( `diagram/Moments.h`: `K < D` `x_K`, `K == D` `|x|^2`, `K == D + 1` one ), and its gradient
    template<int K> HD static TF poly( const auto &x ) {
        if constexpr ( K < D ) return TF( x[ K ] );
        else if constexpr ( K == D ) { TF s = 0; for ( int k = 0; k < D; ++k ) s += TF( x[ k ] ) * TF( x[ k ] ); return s; }
        else return TF( 1 );
    }
    template<int K> HD static TF poly_grad( const auto &x, int k ) {
        if constexpr ( K < D ) return TF( k == K );
        else if constexpr ( K == D ) return 2 * TF( x[ k ] );
        else return TF( 0 );
    }
    template<int K> static constexpr int poly_degree = K < D ? 1 : K == D ? 2 : 0;

    template<int K>
    HD TF integrate_moment_over_simplex( const auto &Q ) const {
        TF s = 0;
        for_each_node( Q, degree + poly_degree<K>, [&]( const auto &x, const TF *, TF w ) { s += w * poly<K>( x ) * value_at( x ); } );
        return s;
    }

    /// the adjoint: the volume moves with the vertices ( cofactors of the determinant ), a node is a fixed barycentric combination
    /// of them ( the gradient of the integrand, spread with its coordinates ), and the coefficients receive the basis
    template<int K>
    HD void integrate_moment_over_simplex_bwd( const auto &Q, TF g, auto &&grad_pts, auto &&grad_dist ) const {
        const auto M = edge_matrix( Q );
        const TF det = M.determinant();
        const TF vol = ( det < 0 ? -det : det ) / factorial( D );
        if ( ! ( vol > 0 ) )
            return;
        TF mean = 0;                                     // `int / vol`
        for_each_node( Q, degree + poly_degree<K>, [&]( const auto &x, const TF *bary, TF w ) {
            const TF p = poly<K>( x ), r = value_at( x );
            mean += w / vol * p * r;
            const auto gr = gradient_at( x );
            for ( int q = 0; q <= D; ++q )
                for ( int k = 0; k < D; ++k )
                    grad_pts[ q ][ k ] += g * w * bary[ q ] * ( p * gr[ k ] + r * poly_grad<K>( x, k ) );
            add_value_grad_at( grad_dist, x, g * w * p );
        } );

        const TF gv = ( det < 0 ? -g : g ) * mean / factorial( D );
        for ( PI r = 0; r < PI( D ); ++r ) {
            TF row_sum = 0;
            for ( PI k = 0; k < PI( D ); ++k ) {
                const TF minor = M.without_row_and_col( r, k ).determinant();
                const TF cof = ( ( r + k ) % 2 ? -minor : minor ) * gv;
                grad_pts[ k + 1 ][ r ] += cof;
                row_sum += cof;
            }
            grad_pts[ 0 ][ r ] -= row_sum;
        }
    }

    HD TF   integrate_over_simplex    ( const auto &Q ) const { return integrate_moment_over_simplex<D + 1>( Q ); }
    HD void integrate_over_simplex_bwd( const auto &Q, TF g, auto &&grad_pts, auto &&grad_dist ) const {
        integrate_moment_over_simplex_bwd<D + 1>( Q, g, grad_pts, grad_dist );
    }

    /// `int rho`, `int x rho`, `int |x|^2 rho` over the simplex, ACCUMULATED
    HD void integrate_moments_over_simplex( const auto &Q, TF &m, auto &mx, TF &m2 ) const {
        for_each_node( Q, degree + 2, [&]( const auto &x, const TF *, TF w ) {
            const TF r = w * value_at( x );
            m += r;
            TF n2 = 0;
            for ( int k = 0; k < D; ++k ) {
                mx[ k ] += r * TF( x[ k ] );
                n2 += TF( x[ k ] ) * TF( x[ k ] );
            }
            m2 += r * n2;
        } );
    }

    /// what the derivative with respect to the frame needs of a simplex ( `AffineAcc`, `diagram/Ops.h` ): `J = int p rho` on the
    /// diagonal of `X`, `int d_a p ( x - o )_b rho` in `X`, `int grad p rho` in `y`
    template<int K>
    HD void affine_terms_moment( const auto &Q, TF g, auto &acc ) const {
        for_each_node( Q, degree + poly_degree<K>, [&]( const auto &x, const TF *, TF w ) {
            const TF r = g * w * value_at( x ), J = r * poly<K>( x );
            for ( int a = 0; a < D; ++a ) {
                acc.X[ a ][ a ] += J;
                if constexpr ( K <= D ) {
                    const TF ga = r * poly_grad<K>( x, a );
                    acc.y[ a ] += ga;
                    for ( int b = 0; b < D; ++b )
                        acc.X[ a ][ b ] += ga * ( TF( x[ b ] ) - acc.o[ b ] );
                }
            }
        } );
    }
    HD void affine_terms( const auto &Q, TF g, auto &acc ) const { affine_terms_moment<D + 1>( Q, g, acc ); }

    // ---- a facet of a piece ( the hessian ) --------------------------------------------------------------------------------

    HD TF facet_mass( const auto &pc, int cut ) const requires ( D == 2 || D == 3 ) {
        TF res = 0;
        if constexpr ( D == 2 ) {
            const int nb = pc.nb_vertices();
            const int j = cut + 1 < nb ? cut + 1 : 0;
            const TF ax = TF( pc.coord( cut, 0 ) ), ay = TF( pc.coord( cut, 1 ) );
            const TF bx = TF( pc.coord( j, 0 ) ), by = TF( pc.coord( j, 1 ) );
            const TF L = sdot::sqrt( ( bx - ax ) * ( bx - ax ) + ( by - ay ) * ( by - ay ) );
            integration::for_each_simplex_node<1,TF>( degree, [&]( const TF *bary, TF w ) {
                const TF x[ 2 ] = { bary[ 0 ] * ax + bary[ 1 ] * bx, bary[ 0 ] * ay + bary[ 1 ] * by };
                res += L * w * value_at( x );
            } );
        } else {
            pc.for_each_facet_triangle( cut, [&]( int o, int a, int b ) {
                TF p[ 3 ][ 3 ];
                for ( int k = 0; k < 3; ++k ) {
                    p[ 0 ][ k ] = TF( pc.coord( o, k ) );
                    p[ 1 ][ k ] = TF( pc.coord( a, k ) );
                    p[ 2 ][ k ] = TF( pc.coord( b, k ) );
                }
                const TF ux = p[ 1 ][ 0 ] - p[ 0 ][ 0 ], uy = p[ 1 ][ 1 ] - p[ 0 ][ 1 ], uz = p[ 1 ][ 2 ] - p[ 0 ][ 2 ];
                const TF vx = p[ 2 ][ 0 ] - p[ 0 ][ 0 ], vy = p[ 2 ][ 1 ] - p[ 0 ][ 1 ], vz = p[ 2 ][ 2 ] - p[ 0 ][ 2 ];
                const TF cx = uy * vz - uz * vy, cy = uz * vx - ux * vz, cz = ux * vy - uy * vx;
                const TF area = sdot::sqrt( cx * cx + cy * cy + cz * cz ) / 2;
                integration::for_each_simplex_node<2,TF>( degree, [&]( const TF *bary, TF w ) {
                    TF x[ 3 ];
                    for ( int k = 0; k < 3; ++k )
                        x[ k ] = bary[ 0 ] * p[ 0 ][ k ] + bary[ 1 ] * p[ 1 ][ k ] + bary[ 2 ] * p[ 2 ][ k ];
                    res += area * w * value_at( x );
                } );
            } );
        }
        return res;
    }
};

} // namespace sdot
