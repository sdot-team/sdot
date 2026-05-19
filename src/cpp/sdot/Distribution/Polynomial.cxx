#pragma once

#include "../support/SimpleSquareMatrix.h"
#include "../Cell/integral.h"
#include "Polynomial.h"
#include <cmath>

namespace sdot {

#define UP template<int order, int dim, typename Arch, typename TF>
#define DP Polynomial<order,dim,Arch,TF>

// Integrates P(x) = Σ_β coeffs[c] x^β over a simplex defined by dim+1 points.
//
// Change of variables x = v0 + M*t, t ∈ standard d-simplex T_d:
//   x_j(t) = simplex[0][j] + Σ_{i<dim} M(j,i) * t_i
//
// For each Q_k coefficient c with multi-index β = q_pow:
//   ∫_{simplex} x^β dx = |det M| * ∫_{T_d} Π_j (v0_j + Σ_i M(j,i) t_i)^{β_j} dt
//
// The t-polynomial is expanded iteratively (multiply by each linear factor);
// then integrated term-by-term via the Dirichlet formula:
//   ∫_{T_d} t^n dt = Π_k n_k! / (d + Σ n_k)!
//
// t-monomials t^n are encoded as a flat index:  n_0 + stride*n_1 + stride^2*n_2 + ...
// with stride = order*dim+1  (max power of any single t_i is order*dim).
template<int order,int dim,class TF,class Arch>
struct Integral<Polynomial<order,dim,Arch,TF>,Simplex<dim,dim+1,TF,Arch>> {
    static auto integral( const Polynomial<order,dim,Arch,TF> &pol, const Simplex<dim,dim+1,TF,Arch> &simplex ) {
        // M(j, i) = j-th component of i-th edge vector
        auto M = Matrix<TF,dim,Arch>::with_func( [&]( PI row, PI col ) {
            return simplex.pts[ col + 1 ][ row ] - simplex.pts[ 0 ][ row ];
        } );
        const TF jac = std::abs( M.determinant() );

        constexpr int stride    = order * dim + 1;
        constexpr int poly_size = pow_rec( stride, PI( dim ) );

        PI  q_pow[ dim >= 1 ? dim : 1 ] = {};
        TF  result = 0;

        for ( PI c = 0;; ++c ) {
            // build the t-polynomial = Π_j x_j(t)^{q_pow[j]}
            TF poly[ poly_size ] = {};
            poly[ 0 ] = TF( 1 );

            for ( PI j = 0; j < PI( dim ); ++j ) {
                const TF v0j = simplex.pts[ 0 ][ j ];
                for ( PI p = 0; p < q_pow[ j ]; ++p ) {
                    // multiply poly by the linear form (v0j + Σ_k M(j,k) t_k)
                    TF np[ poly_size ] = {};
                    for ( int idx = 0; idx < poly_size; ++idx ) {
                        if ( poly[ idx ] == TF( 0 ) ) continue;
                        np[ idx ] += poly[ idx ] * v0j;
                        int stride_k = 1, rem = idx;
                        for ( PI k = 0; k < PI( dim ); ++k, stride_k *= stride ) {
                            // increment n_k by 1 if in range
                            if ( rem % stride + 1 < stride )
                                np[ idx + stride_k ] += poly[ idx ] * M( j, k );
                            rem /= stride;
                        }
                    }
                    for ( int i = 0; i < poly_size; ++i )
                        poly[ i ] = np[ i ];
                }
            }

            // integrate using Dirichlet: ∫_{T_d} t^n dt = Π_k n_k! / (d + Σ n_k)!
            TF integ = TF( 0 );
            for ( int idx = 0; idx < poly_size; ++idx ) {
                if ( poly[ idx ] == TF( 0 ) ) continue;
                TF term  = poly[ idx ];
                PI total = 0;
                int rem  = idx;
                for ( PI k = 0; k < PI( dim ); ++k ) {
                    PI nk = rem % stride;
                    rem /= stride;
                    term  *= factorial( TF( nk ) );
                    total += nk;
                }
                integ += term / factorial( TF( PI( dim ) + total ) );
            }
            result += pol.coeffs[ c ] * integ;

            // advance Q_k multi-index (last axis fastest)
            PI d = dim;
            while ( d-- ) {
                if ( ++q_pow[ d ] <= PI( order ) ) break;
                q_pow[ d ] = 0;
                if ( d == 0 ) goto done;
            }
        }
        done:
        return jac * result;
    }
};

// Boundary simplex: (dim-1)-simplex with dim vertices embedded in R^dim.
// Ex: line segment in 2D, triangle in 3D.
//
// Same expansion as above but with sdim = dim-1 t-variables.
// Jacobian = sqrt(det(G)) with G = M^T M  (Gram, (dim-1)×(dim-1)):
//   G(r,c) = Σ_j (pts[r+1][j]-pts[0][j]) * (pts[c+1][j]-pts[0][j])
// For dim==2: sqrt(det(G)) = length of segment.
// For dim==3: sqrt(det(G)) = |e0 × e1| = 2 * area of triangle.
template<int order,int dim,class TF,class Arch>
struct Integral<Polynomial<order,dim,Arch,TF>,Simplex<dim,dim,TF,Arch>> {
    static auto integral( const Polynomial<order,dim,Arch,TF> &pol, const Simplex<dim,dim,TF,Arch> &simplex ) {
        constexpr int sdim = dim - 1;  // parameter-space dimension

        auto G = Matrix<TF,Arch,sdim>::with_func( [&]( PI r, PI c ) {
            TF dot = 0;
            for ( PI j = 0; j < PI( dim ); ++j )
                dot += ( simplex.pts[ r + 1 ][ j ] - simplex.pts[ 0 ][ j ] )
                     * ( simplex.pts[ c + 1 ][ j ] - simplex.pts[ 0 ][ j ] );
            return dot;
        } );
        const TF jac = std::sqrt( G.determinant() );

        constexpr int stride    = order * sdim + 1;
        constexpr int poly_size = pow_rec( stride, PI( sdim ) );

        PI  q_pow[ dim >= 1 ? dim : 1 ] = {};
        TF  result = 0;

        for ( PI c = 0;; ++c ) {
            TF poly[ poly_size >= 1 ? poly_size : 1 ] = {};
            poly[ 0 ] = TF( 1 );

            for ( PI j = 0; j < PI( dim ); ++j ) {
                const TF v0j = simplex.pts[ 0 ][ j ];
                for ( PI p = 0; p < q_pow[ j ]; ++p ) {
                    TF np[ poly_size >= 1 ? poly_size : 1 ] = {};
                    for ( int idx = 0; idx < poly_size; ++idx ) {
                        if ( poly[ idx ] == TF( 0 ) ) continue;
                        np[ idx ] += poly[ idx ] * v0j;
                        int stride_k = 1, rem = idx;
                        for ( PI k = 0; k < PI( sdim ); ++k, stride_k *= stride ) {
                            if ( rem % stride + 1 < stride )
                                np[ idx + stride_k ] += poly[ idx ] * ( simplex.pts[ k + 1 ][ j ] - simplex.pts[ 0 ][ j ] );
                            rem /= stride;
                        }
                    }
                    for ( int i = 0; i < poly_size; ++i )
                        poly[ i ] = np[ i ];
                }
            }

            TF integ = TF( 0 );
            for ( int idx = 0; idx < poly_size; ++idx ) {
                if ( poly[ idx ] == TF( 0 ) ) continue;
                TF term  = poly[ idx ];
                PI total = 0;
                int rem  = idx;
                for ( PI k = 0; k < PI( sdim ); ++k ) {
                    PI nk = rem % stride;
                    rem /= stride;
                    term  *= factorial( TF( nk ) );
                    total += nk;
                }
                integ += term / factorial( TF( PI( sdim ) + total ) );
            }
            result += pol.coeffs[ c ] * integ;

            PI d = dim;
            while ( d-- ) {
                if ( ++q_pow[ d ] <= PI( order ) ) break;
                q_pow[ d ] = 0;
                if ( d == 0 ) goto done;
            }
        }
        done:
        return jac * result;
    }
};

#undef UP
#undef DP

} // namespace sdot
