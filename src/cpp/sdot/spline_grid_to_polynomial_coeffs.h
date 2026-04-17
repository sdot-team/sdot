#pragma once

#include "support/SimpleSquareMatrix.h"
#include "support/TridiagonalSolver.h"
#include "support/Tensor.h"
#include "support/Spline1d.h"
#include "support/P.h"
#include <algorithm>
#include <vector>

namespace sdot {

template<class TF>
struct StdKnots {
    TF operator[]( PI i ) const {
        return TF( i );
    }
};

template<class TF,class Arch>
auto default_frame( PI dim ) {
    Tensor<TF,2,Arch> res( Shape(), { Values(), dim + 1, dim } );
    for( PI i = 0; i < dim; ++i )
        res( 0, i ) = 0;
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
        DsVec<DsVec<TF,ct_dim>,ct_nc> kc( Size(), nc, Size(), dim );
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
        typename SSM::Vec b( Size(), nc );
        DsVecFactory<PI,ct_dim,Arch> pi( dim );
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
        DsVec<DsVec<TF,ct_dim>,ct_nc> kc( Size(), nc, Size(), dim );
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
        typename SSM::Vec coeffs( Size(), nc ), grad_c( Size(), nc );
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
        DsVecFactory<PI,ct_dim,Arch> pi( dim );
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


// helper: compute physical knots per axis for a diagonal frame
template<class TF,class Arch>
static auto _diagonal_frame_knots( PI dim, const auto &frame, const auto &knots, const auto &values ) {
    std::vector<std::vector<TF>> x( dim );
    for ( PI d = 0; d < dim; ++d ) {
        x[ d ].resize( values.size( d ) );
        for ( PI i = 0; i < values.size( d ); ++i )
            x[ d ][ i ] = frame( 0, d ) + frame( d + 1, d ) * knots[ d ][ i ];
    }
    return x;
}

// helper: cross-section index shape = tensor shape minus axis d and the trailing coeff axis
template<class SizesVec>
static auto _cs_shape( const SizesVec &shape, PI d ) {
    // shape has rank dim+1; remove axis d (< dim) then the coeff axis now at dim-1
    return shape.without_index( d ).without_index( shape.size() - 2 );
}

// helper: build a full (dim+1)-length index from a cross-section index cs,
//         a position i along axis d, and a trailing coefficient index p.
//         cs has length dim-1 (all axes except d, in order).
template<class Arch>
static auto _full_idx( PI dim, PI d, const auto &cs, PI i, PI p ) {
    DsVec<PI,-1,Arch> idx( Size(), dim + 1, PI( 0 ) );
    for ( PI j = 0;   j < d;   ++j ) idx[ j ] = cs[ j ];
    idx[ d ] = i;
    for ( PI j = d+1; j < dim; ++j ) idx[ j ] = cs[ j - 1 ];
    idx[ dim ] = p;
    return idx;
}

// Cn nD case -------------------------------------------------------------------------------------------------------------
template<class TF,int dp1,class Arch>
void _spline_grid_to_polynomial_coeffs_forward( CtInt<0> /* batch_version */, CtInt<1>, const TensorView<TF,dp1,Arch> &poly_coeffs, const auto &values, const auto &frame, const auto &knots ) {
    const PI dim = values.rank();

    for ( PI r = 0; r < dim; ++r )
        for ( PI d = 0; d < dim; ++d )
            if ( r != d && frame( r + 1, d ) != 0 )
                throw std::runtime_error( "frame must be diagonal for SplineGrid with continuity >= 1" );

    const auto x = _diagonal_frame_knots<TF,Arch>( dim, frame, knots, values );

    // Working tensor: shape (N0,..,N_{dim-1}, nb_coeffs). Starts at (N0,..,N_{dim-1}, 1).
    // unsqueeze() adds a trailing size-1 axis; the Tensor ctor copies correctly (contiguous or strided).
    Tensor<TF,-1,Arch> cur( values.unsqueeze() );

    for ( PI d = 0; d < dim; ++d ) {
        const PI N       = cur.size( d );
        const PI Ni      = N - 1;
        const PI nb_prev = cur.size( dim );

        auto nxt_shape   = cur.sizes();
        nxt_shape[ d ]   = Ni;
        nxt_shape[ dim ] = 4 * nb_prev;
        Tensor<TF,-1,Arch> nxt( Shape(), nxt_shape );

        TensorView<TF,-1,Arch> cs_view( nullptr, _cs_shape( nxt_shape, d ) );
        std::vector<TF> v_line( N );
        std::vector<typename Spline1d<TF>::Coeffs> c_line( Ni );
        auto in_idx  = DsVec<PI,-1,Arch>( Size(), dim + 1, PI( 0 ) );
        auto out_idx = DsVec<PI,-1,Arch>( Size(), dim + 1, PI( 0 ) );

        cs_view.for_each_index( [&]( const auto &cs ) {
            for ( PI j = 0;   j < d;   ++j ) { in_idx[j] = cs[j];   out_idx[j] = cs[j]; }
            for ( PI j = d+1; j < dim; ++j ) { in_idx[j] = cs[j-1]; out_idx[j] = cs[j-1]; }

            for ( PI p = 0; p < nb_prev; ++p ) {
                in_idx[ dim ] = p;
                for ( PI i = 0; i < N;  ++i ) { in_idx[ d ] = i; v_line[ i ] = cur( in_idx ); }

                Spline1d<TF>::values_to_coeffs( c_line, v_line, x[ d ] );

                for ( PI i = 0; i < Ni; ++i ) {
                    out_idx[ d ] = i;
                    const TF cq[4] = { c_line[i].c0, c_line[i].c1, c_line[i].c2, c_line[i].c3 };
                    for ( PI q = 0; q < 4; ++q ) { out_idx[ dim ] = p * 4 + q; nxt( out_idx ) = cq[ q ]; }
                }
            }
        } );

        cur = std::move( nxt );
    }

    cur.for_each_index( [&]( const auto &idx ) {
        poly_coeffs( idx ) = cur( idx );
    } );
}

template<int continuity,class TF,int dp1,class Arch>
void _spline_grid_to_polynomial_coeffs_backward( CtInt<0> /* batch_version */, CtInt<continuity>, const auto &grad_values, const auto &grad_frame, const auto &grad_knots, const TensorView<const TF,dp1,Arch> &poly_coeffs, const auto &grad_poly_coeffs, const auto &values, const auto &frame, const auto &knots ) {
    if constexpr ( continuity == 1 ) {
        const PI dim = values.rank();

        for ( PI r = 0; r < dim; ++r )
            for ( PI d = 0; d < dim; ++d )
                if ( r != d && frame( r + 1, d ) != 0 )
                    throw std::runtime_error( "frame must be diagonal for SplineGrid with continuity >= 1" );

        const auto x = _diagonal_frame_knots<TF,Arch>( dim, frame, knots, values );

        // Re-run forward, storing every intermediate tensor.
        // Tensor is not default-constructible: use reserve+emplace_back to avoid default construction.
        std::vector<Tensor<TF,-1,Arch>> fwd;
        fwd.reserve( dim + 1 );
        fwd.emplace_back( values.unsqueeze() ); // fwd[0]: values with a trailing size-1 coeff axis

        for ( PI d = 0; d < dim; ++d ) {
            const PI N       = fwd[ d ].size( d );
            const PI nb_prev = fwd[ d ].size( dim );
            auto nxt_shape   = fwd[ d ].sizes();
            nxt_shape[ d ]   = N - 1;
            nxt_shape[ dim ] = 4 * nb_prev;
            fwd.emplace_back( Shape(), nxt_shape ); // fwd[d+1], no realloc (reserved)

            TensorView<TF,-1,Arch> cs_view( nullptr, _cs_shape( nxt_shape, d ) );
            std::vector<TF> v_line( N );
            std::vector<typename Spline1d<TF>::Coeffs> c_line( N - 1 );
            auto in_idx  = DsVec<PI,-1,Arch>( Size(), dim + 1, PI( 0 ) );
            auto out_idx = DsVec<PI,-1,Arch>( Size(), dim + 1, PI( 0 ) );

            cs_view.for_each_index( [&]( const auto &cs ) {
                for ( PI j = 0;   j < d;   ++j ) { in_idx[j] = cs[j];   out_idx[j] = cs[j]; }
                for ( PI j = d+1; j < dim; ++j ) { in_idx[j] = cs[j-1]; out_idx[j] = cs[j-1]; }
                for ( PI p = 0; p < nb_prev; ++p ) {
                    in_idx[ dim ] = p;
                    for ( PI i = 0; i < N; ++i ) { in_idx[ d ] = i; v_line[ i ] = fwd[ d ]( in_idx ); }
                    Spline1d<TF>::values_to_coeffs( c_line, v_line, x[ d ] );
                    for ( PI i = 0; i < N - 1; ++i ) {
                        out_idx[ d ] = i;
                        const TF cq[4] = { c_line[i].c0, c_line[i].c1, c_line[i].c2, c_line[i].c3 };
                        for ( PI q = 0; q < 4; ++q ) { out_idx[ dim ] = p * 4 + q; fwd[ d + 1 ]( out_idx ) = cq[ q ]; }
                    }
                }
            } );
        }

        // Backward sweep — Tensor ctor from TensorView copies correctly for any strides
        Tensor<TF,-1,Arch> g_cur( grad_poly_coeffs );

        for ( SI d = SI( dim ) - 1; d >= 0; --d ) {
            const PI N       = fwd[ d ].size( d );
            const PI Ni      = N - 1;
            const PI nb_prev = fwd[ d ].size( dim );

            Tensor<TF,-1,Arch> g_prev( Shape(), fwd[ d ].sizes() );
            std::fill( g_prev.data(), g_prev.data() + g_prev.size(), TF( 0 ) );

            TensorView<TF,-1,Arch> cs_view( nullptr, _cs_shape( fwd[ d + 1 ].sizes(), d ) );
            std::vector<TF> v_line( N ), g_v_line( N ), g_x_line( N );
            std::vector<typename Spline1d<TF>::Coeffs> g_c_line( Ni );
            auto in_idx  = DsVec<PI,-1,Arch>( Size(), dim + 1, PI( 0 ) );
            auto out_idx = DsVec<PI,-1,Arch>( Size(), dim + 1, PI( 0 ) );

            cs_view.for_each_index( [&]( const auto &cs ) {
                for ( PI j = 0;     j < PI(d); ++j ) { in_idx[j] = cs[j];   out_idx[j] = cs[j]; }
                for ( PI j = PI(d)+1; j < dim; ++j ) { in_idx[j] = cs[j-1]; out_idx[j] = cs[j-1]; }

                for ( PI p = 0; p < nb_prev; ++p ) {
                    in_idx[ dim ] = p;
                    for ( PI i = 0; i < N;  ++i ) { in_idx[ d ] = i; v_line[ i ] = fwd[ d ].view()( in_idx ); }

                    for ( PI i = 0; i < Ni; ++i ) {
                        out_idx[ d ] = i;
                        out_idx[ dim ] = p*4+0; g_c_line[i].c0 = g_cur( out_idx );
                        out_idx[ dim ] = p*4+1; g_c_line[i].c1 = g_cur( out_idx );
                        out_idx[ dim ] = p*4+2; g_c_line[i].c2 = g_cur( out_idx );
                        out_idx[ dim ] = p*4+3; g_c_line[i].c3 = g_cur( out_idx );
                    }

                    std::fill( g_v_line.begin(), g_v_line.end(), TF( 0 ) );
                    std::fill( g_x_line.begin(), g_x_line.end(), TF( 0 ) );
                    Spline1d<TF>::values_to_coeffs_backward( g_v_line, g_x_line, g_c_line, v_line, x[ d ] );

                    in_idx[ dim ] = p;
                    for ( PI i = 0; i < N; ++i ) {
                        in_idx[ d ] = i;
                        g_prev( in_idx ) += g_v_line[ i ];
                        grad_frame( 0,     d ) += g_x_line[ i ];
                        grad_frame( d + 1, d ) += g_x_line[ i ] * knots[ d ][ i ];
                        grad_knots[ d ][ i ]   += g_x_line[ i ] * frame( d + 1, d );
                    }
                }
            } );

            g_cur = std::move( g_prev );
        }

        // g_cur has shape (N0,..,N_{dim-1}, 1); propagate to grad_values
        values.for_each_index( [&]( const auto &idx ) {
            grad_values( idx ) += g_cur( idx.with_pushed_value( 0 ) );
        } );
    } else {
        throw std::runtime_error( "Only continuity=1 is supported for now" );
    }
}

// generic case : ensure non empty inputs -------------------------------------------------------------------------------------------------------------
template<int batch_version,int continuity,class TF,int p_rank,class Arch>
void spline_grid_to_polynomial_coeffs_forward( CtInt<batch_version>, CtInt<continuity>, const TensorView<TF,p_rank,Arch> &poly_coeffs, const auto &values, const auto &frame, const auto &knots ) {
    const PI dim = values.rank() - batch_version;
    if ( frame.is_invalid() )
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
    if ( frame.is_invalid() )
        return spline_grid_to_polynomial_coeffs_backward( CtInt<batch_version>(), CtInt<continuity>(), grad_values, grad_frame, grad_knots, poly_coeffs, grad_poly_coeffs, values, default_frame<TF,Arch>( dim ), knots );
    if ( knots.empty() )
        return spline_grid_to_polynomial_coeffs_backward( CtInt<batch_version>(), CtInt<continuity>(), grad_values, grad_frame, grad_knots, poly_coeffs, grad_poly_coeffs, values, frame, std::vector<StdKnots<TF>>( dim ) );
    return _spline_grid_to_polynomial_coeffs_backward( CtInt<batch_version>(), CtInt<continuity>(), grad_values, grad_frame, grad_knots, poly_coeffs, grad_poly_coeffs, values, frame, knots );
}

} // namespace sdot


