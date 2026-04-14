#pragma once

#include "geometry/SimpleSquareMatrix.h"
#include "support/Tensor.h"
#include "support/P.h"

namespace sdot {

template<class TF>
struct StdKnots {
    TF operator[]( PI i ) const {
        return TF( i );
    }
};

template<class TF,class Arch>
auto default_frame( PI dim ) {
    Tensor<TF,2,Arch> res( Shape(), { dim + 1, dim } );
    for( PI i = 0; i < dim; ++i ) res( 0, i ) = 0;
    for( PI r = 0; r < dim; ++r )
        for( PI i = 0; i < dim; ++i )
            res( r + 1, i ) = ( r == i );
    return res;
}

// C0 1D case -------------------------------------------------------------------------------------------------------------
template<class TF,class Arch>
void _spline_grid_to_polynomial_coeffs_forward( CtInt<0> /* batch_version */, CtInt<0> /* continuity */, const TensorView<TF,2,Arch> &poly_coeffs, const auto &values, const auto &frame, const auto &knots ) {
    for( PI i = 0, n = values.size() - 1; i < n; ++i ) {
        const TF k0 = frame( 0, 0 ) + frame( 1, 0 ) * knots[ 0 ][ i + 0 ];
        const TF k1 = frame( 0, 0 ) + frame( 1, 0 ) * knots[ 0 ][ i + 1 ];
        const TF v0 = values[ i + 0 ];
        const TF v1 = values[ i + 1 ];
        if ( const TF h = k1 - k0 ) {
            const TF c1 = ( v1 - v0 ) / h;
            poly_coeffs( i, 0 ) = v0 - c1 * k0;
            poly_coeffs( i, 1 ) = c1;
        } else {
            poly_coeffs( i, 0 ) = ( v0 + v1 ) / 2;
            poly_coeffs( i, 1 ) = 0;
        }
    }
}

template<class TF,class Arch>
void _spline_grid_to_polynomial_coeffs_backward( CtInt<0> /* batch_version */, CtInt<0> /* continuity */, const auto &grad_values, const auto &grad_frame, const auto &grad_knots, const TensorView<const TF,2,Arch> &poly_coeffs, const auto &grad_poly_coeffs, const auto &values, const auto &frame, const auto &knots ) {
    for( PI i = 0, n = values.size() - 1; i < n; ++i ) {
        const TF k0   = frame( 0, 0 ) + frame( 1, 0 ) * knots[ 0 ][ i + 0 ];
        const TF k1   = frame( 0, 0 ) + frame( 1, 0 ) * knots[ 0 ][ i + 1 ];
        const TF h    = k1 - k0;
        const TF dv   = values[ i + 1 ] - values[ i ];
        const TF ih   = 1 / h;
        const TF ih2  = ih * ih;
        const TF g_c0 = grad_poly_coeffs( i, 0 );
        const TF g_c1 = grad_poly_coeffs( i, 1 );

        grad_values[ i + 0 ] += g_c0 * ( 1 + k0 * ih ) - g_c1 * ih;
        grad_values[ i + 1 ] += g_c1 * ih - g_c0 * k0 * ih;

        const TF grad_k0 = dv * ih2 * ( g_c1 - g_c0 * k1 );
        const TF grad_k1 = dv * ih2 * ( g_c0 * k0 - g_c1 );

        grad_frame( 0, 0 ) += grad_k0 + grad_k1;
        grad_frame( 1, 0 ) += grad_k0 * knots[ 0 ][ i + 0 ] + grad_k1 * knots[ 0 ][ i + 1 ];
        grad_knots[ 0 ][ i + 0 ] += grad_k0 * frame( 1, 0 );
        grad_knots[ 0 ][ i + 1 ] += grad_k1 * frame( 1, 0 );
    }
}

// C0 nD case -------------------------------------------------------------------------------------------------------------
template<class TF,int dp1,class Arch>
void _spline_grid_to_polynomial_coeffs_forward( CtInt<0> /* batch_version */, CtInt<0> /* continuity */, const TensorView<TF,dp1,Arch> &poly_coeffs, const auto &values, const auto &frame, const auto &knots ) {
    constexpr PI ct_dim = dp1 > 0 ? dp1 - 1 : -1;
    constexpr PI ct_nc = ct_dim > 0 ? 1 << ct_dim : -1;
    const PI dim = poly_coeffs.rank() - 1;
    const PI nc = 1 << dim;

    using SSM = SimpleSquareMatrix<TF,ct_nc,Arch>;

    // TODO: for regular frames, use the inverse matrix
    values.for_each_index( [&]( const auto &idx ) {
        // physical positions of the NC = 2^DIM cell corners
        // (may coincide for degenerate/null-width cells — handled by zero-pivot in solve_ge)
        Point<Point<TF,ct_dim>,ct_nc> kc( nc, dim ); // TODO: rt dim
        for ( PI c = 0; c < nc; ++c ) {
            for ( PI d = 0; d < dim; ++d ) {
                kc[ c ][ d ] = frame( 0, d );
                for ( PI r = 0; r < dim; ++r )
                    kc[ c ][ d ] += frame( r + 1, d ) * knots[ r ][ idx[ r ] + ( ( c >> r ) & 1 ) ];
            }
        }

        // Vandermonde: V[ c ][ cpt ] = prod_{d: (cpt>>(dim-1-d))&1} kc[c][d]
        // Column index cpt follows Q_k lex ordering (dim-0 is MSB, dim-1 is LSB),
        // matching PolynomialGrid's expected coefficient layout.
        const auto V = SSM::with_func( nc, [&]( PI c, PI cpt ) -> TF {
            TF vcpt = 1;
            for ( PI d = 0; d < dim; ++d )
                if ( ( cpt >> ( dim - 1 - d ) ) & 1 )
                    vcpt *= kc[ c ][ d ];
            return vcpt;
        } );

        // RHS: value at each corner
        typename SSM::Vec b( nc );
        PointFactory<PI,ct_dim,Arch> pi( dim );
        for ( PI c = 0; c < nc; ++c ) {
            auto off = pi.with_func( [&]( PI d ) { return ( c >> d ) & 1; } );
            b[ c ] = values( idx + off );
        }

        // solve V @ coeffs = b (zero pivot → coeff = 0, handles degenerate cells)
        const auto coeffs = V.solve_ge( b );
        for ( PI p = 0; p < nc; ++p )
            poly_coeffs( idx, p ) = coeffs[ p ];
    }, 1 );
}

template<class TF,int dp1,class Arch>
void _spline_grid_to_polynomial_coeffs_backward( CtInt<0> /* batch_version */, CtInt<0> /* continuity */, const auto &grad_values, const auto &grad_frame, const auto &grad_knots, const TensorView<const TF,dp1,Arch> &poly_coeffs, const auto &grad_poly_coeffs, const auto &values, const auto &frame, const auto &knots ) {
    constexpr PI ct_dim = dp1 > 0 ? dp1 - 1 : -1;
    constexpr PI ct_nc = ct_dim > 0 ? 1 << ct_dim : -1;
    const PI dim = poly_coeffs.rank() - 1;
    const PI nc = 1 << dim;

    using SSM = SimpleSquareMatrix<TF,ct_nc,Arch>;

    values.for_each_index( [&]( const auto &idx ) {
        // recompute kc (same as forward)
        Point<Point<TF,ct_dim>,ct_nc> kc; // TODO: rt dim
        for ( PI c = 0; c < nc; ++c ) {
            for ( PI d = 0; d < dim; ++d ) {
                kc[ c ][ d ] = frame( 0, d );
                for ( PI r = 0; r < dim; ++r )
                    kc[ c ][ d ] += frame( r + 1, d ) * knots[ r ][ idx[ r ] + ( ( c >> r ) & 1 ) ];
            }
        }

        // V[c][cpt] = prod_{d: (cpt>>(dim-1-d))&1} kc[c][d]  (lex ordering, needed for grad_kc below)
        const auto V = SSM::with_func( nc, [&]( PI c, PI cpt ) -> TF {{
            TF vcpt = 1;
            for ( PI d = 0; d < dim; ++d )
                if ( ( cpt >> ( dim - 1 - d ) ) & 1 )
                    vcpt *= kc[ c ][ d ];
            return vcpt;
        }} );

        // load stored coefficients and incoming gradient
        typename SSM::Vec coeffs( nc ), grad_c( nc );
        for ( PI p = 0; p < nc; ++p ) {{
            coeffs[ p ] = poly_coeffs( idx, p );
            grad_c[ p ] = grad_poly_coeffs( idx, p );
        }}

        // solve V^T @ alpha = grad_c  →  alpha = gradient w.r.t. corner values
        // VT[r][c] = V[c][r] = prod_{d: (r>>(dim-1-d))&1} kc[c][d]  (lex ordering)
        const auto VT = SSM::with_func( nc, [&]( PI r, PI c ) -> TF {
            TF vcp = 1;
            for ( PI d = 0; d < dim; ++d )
                if ( ( r >> ( dim - 1 - d ) ) & 1 )
                    vcp *= kc[ c ][ d ];
            return vcp;
        } );
        const auto alpha = VT.solve_ge( grad_c );

        // grad_values at each corner += alpha[c]
        PointFactory<PI,ct_dim,Arch> pi( dim );
        for ( PI c = 0; c < nc; ++c ) {
            auto off = pi.with_func( [&]( PI d ) { return ( c >> d ) & 1; } );
            grad_values( idx + off ) += alpha[ c ];
        }

        // grad_kc[c][d]: dL/dV[c][cpt] = -alpha[c]*coeffs[cpt],
        // dV[c][cpt]/dkc[c][d] = V[c][cpt^(1<<(dim-1-d))] when (cpt>>(dim-1-d))&1
        // propagate directly to grad_frame and grad_knots without a temporary grad_kc array
        for ( PI c = 0; c < nc; ++c ) {
            for ( PI d = 0; d < dim; ++d ) {
                TF gkcd = 0;
                const PI dbit = dim - 1 - d;
                for ( PI p = 0; p < nc; ++p )
                    if ( ( p >> dbit ) & 1 )
                        gkcd -= alpha[ c ] * coeffs[ p ] * V( c, p ^ ( 1 << dbit ) );
                grad_frame( 0, d ) += gkcd;
                for ( PI r = 0; r < dim; ++r ) {
                    grad_frame( r + 1, d ) += gkcd * knots[ r ][ idx[ r ] + ( ( c >> r ) & 1 ) ];
                    grad_knots[ r ][ idx[ r ] + ( ( c >> r ) & 1 ) ] += gkcd * frame( r + 1, d );
                }
            }
        }
    }, 1 );
}

// Cn nD case -------------------------------------------------------------------------------------------------------------
template<int continuity,class TF,int dp1,class Arch>
void _spline_grid_to_polynomial_coeffs_forward( CtInt<0> /* batch_version */,  CtInt<continuity>, const TensorView<TF,dp1,Arch> &poly_coeffs, const auto &values, const auto &frame, const auto &knots ) {
    // TODO
}

template<int continuity,class TF,int dp1,class Arch>
void _spline_grid_to_polynomial_coeffs_backward( CtInt<0> /* batch_version */, CtInt<continuity>, const auto &grad_values, const auto &grad_frame, const auto &grad_knots, const TensorView<const TF,dp1,Arch> &poly_coeffs, const auto &grad_poly_coeffs, const auto &values, const auto &frame, const auto &knots ) {
    // TODO
}

// generic case : ensure non empty inputs -------------------------------------------------------------------------------------------------------------
template<int batch_version,int continuity,class TF,int p_rank,class Arch>
void spline_grid_to_polynomial_coeffs_forward( CtInt<batch_version>, CtInt<continuity>, const TensorView<TF,p_rank,Arch> &poly_coeffs, const auto &values, const auto &frame, const auto &knots ) {
    const PI dim = values.rank() - batch_version;
    if ( frame.empty() )
        return spline_grid_to_polynomial_coeffs_forward( CtInt<batch_version>(), CtInt<continuity>(), poly_coeffs, values, default_frame<TF,Arch>( dim ), knots );
    if ( knots.empty() )
        return spline_grid_to_polynomial_coeffs_forward( CtInt<batch_version>(), CtInt<continuity>(), poly_coeffs, values, frame, std::vector<StdKnots<TF>>( dim ) );
    _spline_grid_to_polynomial_coeffs_forward( CtInt<batch_version>(), CtInt<continuity>(), poly_coeffs, values, frame, knots );
}

template<int batch_version,int continuity,class TF,int p_rank,class Arch>
void spline_grid_to_polynomial_coeffs_backward( CtInt<batch_version>, CtInt<continuity>, const auto &grad_values, const auto &grad_frame, const auto &grad_knots,
                const TensorView<const TF,p_rank,Arch> &poly_coeffs, const auto &grad_poly_coeffs,
                const auto &values, const auto &frame, const auto &knots ) {
    const PI dim = values.rank() - batch_version;
    if ( frame.empty() )
        return spline_grid_to_polynomial_coeffs_backward( CtInt<batch_version>(), CtInt<continuity>(), grad_values, grad_frame, grad_knots, poly_coeffs, grad_poly_coeffs, values, default_frame<TF,Arch>( dim ), knots );
    if ( knots.empty() )
        return spline_grid_to_polynomial_coeffs_backward( CtInt<batch_version>(), CtInt<continuity>(), grad_values, grad_frame, grad_knots, poly_coeffs, grad_poly_coeffs, values, frame, std::vector<StdKnots<TF>>( dim ) );
    return _spline_grid_to_polynomial_coeffs_backward( CtInt<batch_version>(), CtInt<continuity>(), grad_values, grad_frame, grad_knots, poly_coeffs, grad_poly_coeffs, values, frame, knots );
}

} // namespace sdot


