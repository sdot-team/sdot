#pragma once

#include "PolynomialGrid.h"

namespace sdot {

#define UTP template<class TF,int ct_dim,int order,class Arch>
#define DTP PolynomialGrid<TF,ct_dim,order,Arch>

UTP DTP::PolynomialGrid( Values values, Bounds bounds, const Knots &knots, bool normalize ) : normalize( normalize ), values( values ), bounds( bounds ), knots( knots ) {
    ASSERT_EQ( values.rank(), knots.size() + 1 );
    ASSERT_EQ( values.size( knots.size() ), nb_coeffs );

    // coeff_values
    coeff_values = 1;
    if ( normalize ) {
        TF measure = 0;
        for_each_in_range( Pi::zeros( dim() ), grid_shape(), [&]( const auto &index ) {
            measure += piece_integral( index, polynomial( index ) );
        } );
        coeff_values = 1 / measure;
    }
}

UTP auto DTP::grid_shape() const {
    return values.sizes().without_index( dim() );
}

UTP DTP::Polynomial DTP::polynomial( Pi position ) const {
    Polynomial res;
    for( PI d = 0; d < nb_coeffs; ++d )
        res.coeffs[ d ] = coeff_values * values( position, d );
    return res;
}

UTP TF DTP::piece_integral( Pi index, const Polynomial &pol ) const {
    if ( has_skew_or_rotation() )
        TODO;

    // Q_k basis: coefficient c corresponds to multi-index (p0,p1,...,p_{d-1})
    // in lex order, each p_d in 0..order.
    // ∫_{cell} x^p dx = ∏_d ∫_{a_d}^{b_d} x_d^{p_d} dx_d  = ∏_d m_{p_d}[d]
    // where m_k[d] = (b_d^{k+1} - a_d^{k+1}) / (k+1)

    const PI n = dim();

    // Per-axis moments m[d][k] = ∫_{a_d}^{b_d} x^k dx,  k = 0..order
    DsVec<TF, ct_dim * ( order + 1 ), Arch> m( Size(), ct_dim * ( order + 1 ) );
    for( PI d = 0; d < n; ++d ) {
        const TF a = knot( d, index[ d ]     );
        const TF b = knot( d, index[ d ] + 1 );
        TF ak = a, bk = b;  // a^{k+1}, b^{k+1} starting at k=0 -> a^1, b^1
        for ( PI k = 0; k <= PI( order ); ++k ) {
            m[ d * ( order + 1 ) + k ] = ( bk - ak ) / TF( k + 1 );
            ak *= a;
            bk *= b;
        }
    }

    // Iterate over all Q_k multi-indices in lex order
    TF res = 0;
    // use a stack-based counter
    PI powers[ ct_dim >= 0 ? ct_dim : 1 ] = {};
    for ( PI c = 0;; ++c ) {
        // ∫ x^powers = ∏_d m[d][powers[d]]
        TF integ = 1;
        for ( PI d = 0; d < n; ++d )
            integ *= m[ d * ( order + 1 ) + powers[ d ] ];
        res += pol.coeffs[ c ] * integ;

        // advance multi-index (lex, last axis fastest)
        PI d = n;
        while ( d-- ) {
            if ( ++powers[ d ] <= PI( order ) )
                break;
            powers[ d ] = 0;

            if ( d == 0 )
                goto done;
        }
    }
    done:

    return res;
}

UTP TF DTP::knot( PI d, PI index ) const {
    if ( ! bounds.empty() )
        TODO;
    return TF( index ) / values.size( d );
}

UTP auto DTP::simplex_facet_integral( std::span<const Pt> vertices, const Polynomial &value ) {
    using namespace std;

    constexpr int ct_m = ( ct_dim >= 1 ? ct_dim - 1 : -1 );
    const PI m = vertices.size() - 1; // facet dimension  = ct_dim - 1
    const PI n = m + 1; // ambient dimension = ct_dim

    // Gram matrix G = Jᵀ·J  (m×m),  J[:,k] = vertices[k+1] - vertices[0]
    // G[a][b] = dot( edge_a, edge_b )
    // measure A = sqrt(det(G)) / m!
    auto G = SimpleSquareMatrix<TF,ct_m,Arch>::with_func( m, [&]( PI a, PI b ) {
        TF s = 0;
        for ( PI i = 0; i < n; ++i )
            s += ( vertices[ a + 1 ][ i ] - vertices[ 0 ][ i ] )
               * ( vertices[ b + 1 ][ i ] - vertices[ 0 ][ i ] );
        return s;
    } );
    const TF A = sqrt( G.determinant() ) / factorial( m );

    TF res = 0;

    // Centroid in ambient R^n
    Pt centroid( Size(), n );
    if constexpr ( order >= 1 ) {
        for ( const auto &p : vertices )
            centroid += p;
        centroid /= TF( m + 1 );
    }

    // Q_k multi-index iteration (same structure as simplex_integral)
    PI powers[ ct_dim >= 0 ? ct_dim : 1 ] = {};
    for ( PI c = 0;; ++c ) {
        PI total = 0;
        for ( PI d = 0; d < n; ++d )
            total += powers[ d ];

        TF integ = 0;
        if ( total == 0 ) {
            integ = A;
        } else if ( total == 1 ) {
            for ( PI d = 0; d < n; ++d ) {
                if ( powers[ d ] == 1 ) {
                    integ = A * centroid[ d ];
                    break;
                }
            }
        } else if ( total == 2 ) {
            PI i = n, j = n;
            for ( PI d = 0; d < n; ++d ) {
                if ( powers[ d ] >= 1 ) {
                    if ( i == n )
                        i = d;
                    else
                        j = d;
                }
                if ( powers[ d ] == 2 )
                    j = d;
            }
            if ( j == n )
                j = i;
            const TF Si = TF( m + 1 ) * centroid[ i ];
            const TF Sj = TF( m + 1 ) * centroid[ j ];
            TF Tij = 0;
            for ( const auto &p : vertices )
                Tij += p[ i ] * p[ j ];
            integ = A / TF( m + 1 ) / TF( m + 2 ) * ( Si * Sj + Tij );
        } else {
            TODO;
        }

        res += value.coeffs[ c ] * integ;

        PI d = n;
        while ( d-- ) {
            if ( ++powers[ d ] <= PI( order ) )
                break;
            powers[ d ] = 0;
            if ( d == 0 )
                goto done;
        }
    }
    done:

    return res;
}

UTP auto DTP::simplex_integral( std::span<const Pt> vertices, const Polynomial &value ) {
    const PI n = vertices.size() - 1;

    // Volume via Jacobian determinant
    auto J = SimpleSquareMatrix<TF,ct_dim,Arch>::with_func( n, [&]( PI r, PI c ) { return vertices[ r + 1 ][ c ] - vertices[ 0 ][ c ]; } );
    const TF V = J.determinant() / factorial( n );

    // Centroid for order-1 moments
    Pt centroid( Size(), n );
    if constexpr ( order >= 1 ) {
        for ( const auto &p : vertices )
            centroid += p;
        centroid /= TF( n + 1 );
    }

    // Q_k basis: iterate multi-indices (p0,...,p_{d-1}), each in 0..order, lex order
    // For a simplex, exact formulas exist only for the P_k basis.
    // Here we use the Q_k monomials evaluated via the known simplex integrals:
    //   ∫ x^p dV  (component-wise powers, not multi-linear in general)
    // For order <= 1 this matches the P_k formula exactly.
    // For order >= 2, cross terms x_i^a * x_j^b (a+b > 2) require higher-order simplex formulas.
    TF res = 0;
    PI c = 0;
    PI powers[ ct_dim >= 0 ? ct_dim : 1 ] = {};
    const PI dim_n = n;
    for ( ;; ++c ) {
        // Compute ∫_{simplex} ∏_d x_d^{powers[d]} dV
        // Sum of all powers
        PI total = 0;
        for ( PI d = 0; d < dim_n; ++d ) total += powers[ d ];

        TF integ = 0;
        if ( total == 0 ) {
            integ = V;
        } else if ( total == 1 ) {
            // ∫ x_i = V * centroid[i]
            for ( PI d = 0; d < dim_n; ++d )
                if ( powers[ d ] == 1 ) { integ = V * centroid[ d ]; break; }
        } else if ( total == 2 ) {
            // ∫ x_i x_j = V/(n+1)/(n+2) * [ S_i*S_j + T_ij ]
            // (same formula for i==j and i!=j)
            PI i = dim_n, j = dim_n;
            for ( PI d = 0; d < dim_n; ++d ) {
                if ( powers[ d ] >= 1 ) { if ( i == dim_n ) i = d; else j = d; }
                if ( powers[ d ] == 2 ) { j = d; }  // x_i^2: both slots are d
            }
            if ( j == dim_n ) j = i;  // pure square
            const TF Si = TF( n + 1 ) * centroid[ i ];
            const TF Sj = TF( n + 1 ) * centroid[ j ];
            TF Tij = 0;
            for ( const auto &p : vertices ) Tij += p[ i ] * p[ j ];
            integ = V / TF( n + 1 ) / TF( n + 2 ) * ( Si * Sj + Tij );
        } else {
            TODO;
        }

        res += value.coeffs[ c ] * integ;

        // advance multi-index
        PI d = dim_n;
        while ( d-- ) {
            if ( ++powers[ d ] <= PI( order ) ) break;
            powers[ d ] = 0;
            if ( d == 0 ) goto done;
        }
    }
    done:

    return res;
}

UTP auto DTP::facet_integral( auto facet, const Polynomial &value ) {
    if ( order == 0 )
        return value.coeffs[ 0 ] * facet.measure();

    TF res = 0;
    facet.for_each_simplex( [&]( const auto &vertices ) {
        res += simplex_facet_integral( vertices, value );
    } );
    return res;
}

UTP auto DTP::integral( auto cell, const Polynomial &value ) {
    if ( order == 0 )
        return value.coeffs[ 0 ] * cell.measure();

    TF res = 0;
    cell.for_each_simplex( [&]( const auto &vertices ) {
        res += simplex_integral( vertices, value );
    } );
    return res;
}

UTP void DTP::_for_each_piece( const auto &cell, const auto &func, auto beg, auto end, auto &cur, PI d ) const {
    if ( d == dim() ) {
        func( cell, polynomial( cur ) );
        return;
    }

    for( PI v = beg[ d ]; v < end[ d ]; ++v ) {
        auto new_cell = cell;

        new_cell.cut( Pt::value_at( dim(), d, - 1 ), - knot( d, v + 0 ), { dim() } );
        new_cell.cut( Pt::value_at( dim(), d, + 1 ), + knot( d, v + 1 ), { dim() } );

        cur[ d ] = v;
        _for_each_piece( new_cell, func, beg, end, cur, d + 1 );
    }
}


UTP void DTP::for_each_piece( const auto &cell, auto &&func ) const {
    using namespace std;
    if ( ! bounds.empty() )
        TODO;

    // FP boundaries
    const Pt mi = cell.min_pos( Pt::ones( dim() ) );
    const Pt ma = cell.max_pos( Pt::zeros( dim() ) );
    for( PI d = 0; d < dim(); ++d )
        if ( ma[ d ] <= mi[ d ] )
            return;

    // I boundaries
    sdot::DsVecFactory<PI,ct_dim,Arch> pif( dim() );
    auto b = pif.with_func( [&]( PI d ) { return floor( mi[ d ] * values.size( d ) ); } );
    auto e = pif.with_func( [&]( PI d ) { return ceil( ma[ d ] * values.size( d ) ); } );

    // traversal
    auto c = b;
    _for_each_piece( cell, func, b, e, c, 0 );
}

UTP T_U auto DTP::englobing_cell( PI /* dim */, typename Cell<U,ct_dim,Arch>::CellInfo cell_info, typename Cell<U,ct_dim,Arch>::CutInfo cut_info ) const -> Cell<U,ct_dim,Arch>{
    if ( ! bounds.empty() )
        TODO;
    return Cell<U,ct_dim,Arch>::axis_aligned_hypercube( Pt::zeros( dim() ), Pt::ones( dim() ), cell_info, cut_info );
}



// UTP PI DTP::nb_values() const { return values.size(); }
// UTP TF DTP::min_x    () const { return bounds.empty() ? TF( 0 ) : bounds( 0, 0 ); }
// UTP TF DTP::max_x    () const { return bounds.empty() ? TF( 1 ) : bounds( 1, 0 ); }
// UTP TF DTP::knot     ( PI i ) const {
//     return _knots.empty()
//         ? min_x() + ( max_x() - min_x() ) * i / ( nb_values() - 1 )
//         : _knots[ i ];
// }

// UTP typename DTP::Cursor DTP::first_cursor() const {
//     const TF x0 = knot( 0 );
//     const TF x1 = knot( 1 );
//     const TF y0 = coeff_values * values[ 0 ];
//     const TF y1 = coeff_values * values[ 1 ];
//     return Cursor{ .mass = ( x1 - x0 ) * ( y0 + y1 ) / 2, .x0 = x0, .x1 = x1, .y0 = y0, .y1 = y1, .i1 = 1 };
// }

// UTP typename DTP::Cursor DTP::last_cursor() const {
//     const PI i0 = values.size() - 2;
//     const PI i1 = values.size() - 1;
//     const TF x0 = knot( i0 );
//     const TF x1 = knot( i1 );
//     const TF y0 = coeff_values * values[ i0 ];
//     const TF y1 = coeff_values * values[ i1 ];
//     return Cursor{ .mass = ( x1 - x0 ) * ( y0 + y1 ) / 2, .x0 = x0, .x1 = x1, .y0 = y0, .y1 = y1, .i1 = i1 };
// }

// UTP TF DTP::Piece::w2_dist( TF ref_x ) const {
//     return ( x0 - x1 ) * (
//         + 4 * ref_x * ( x0 * ( 2 * y0 + y1 ) + x1 * ( y0 + 2 * y1 ) )
//         - 6 * ref_x * ref_x * ( y0 + y1 )
//         - x0 * x0 * ( 3 * y0 + y1 )
//         - x1 * x1 * ( y0 + 3 * y1 )
//         - 2 * x0 * x1 * ( y0 + y1 )
//     ) / 12;
// }

// UTP TF DTP::Piece::moment() const {
//     if ( x0 == x1 )
//         return TF( 0 );
//     const TF b = ( y1 - y0 ) / ( x1 - x0 );
//     const TF a = y0 - b * x0;
//     return ( std::pow( x1, 2 ) - std::pow( x0, 2 ) ) * a / 2
//          + ( std::pow( x1, 3 ) - std::pow( x0, 3 ) ) * b / 3;
// }

// UTP void DTP::take_some_mass( Cursor &c, TF mass_to_take, auto &&func ) const {
//     auto interp = [&]( TF x, TF x0, TF x1, TF y0, TF y1 ) {
//         return y0 + ( y1 - y0 ) * ( x - x0 ) / ( x1 - x0 );
//     };
//     auto emit = [&]( TF x0, TF x1, TF y0, TF y1 ) {
//         func( Piece{ .i1 = c.i1, .k0 = knot( c.i1 - 1 ), .k1 = c.x1, .x0 = x0, .x1 = x1, .y0 = y0, .y1 = y1 } );
//     };

//     // enough mass in current cell?
//     if ( mass_to_take <= c.mass ) {
//         const TF xn = c.x0 + solve_quadratic<TF>( ( c.y0 - c.y1 ) / ( c.x0 - c.x1 ), c.y0, -mass_to_take, 0 );
//         const TF yn = interp( xn, c.x0, c.x1, c.y0, c.y1 );
//         emit( c.x0, xn, c.y0, yn );
//         c.mass -= mass_to_take;
//         c.x0 = xn;
//         c.y0 = yn;
//         return;
//     }

//     // consume current cell entirely, then advance
//     mass_to_take -= c.mass;
//     emit( c.x0, c.x1, c.y0, c.y1 );

//     while ( true ) {
//         ++c.i1;
//         if ( c.i1 == values.size() ) { c.mass = 0; c.x0 = c.x1; c.y0 = c.y1; --c.i1; return; }

//         c.x0 = c.x1;
//         c.x1 = knot( c.i1 );
//         c.y0 = c.y1;
//         c.y1 = coeff_values * values[ c.i1 ];
//         c.mass = ( c.x1 - c.x0 ) * ( c.y0 + c.y1 ) / 2;

//         if ( mass_to_take <= c.mass )
//             break;

//         mass_to_take -= c.mass;
//         func( Piece{ .i1 = c.i1, .k0 = c.x0, .k1 = c.x1, .x0 = c.x0, .x1 = c.x1, .y0 = c.y0, .y1 = c.y1 } );
//     }

//     // partial consumption of new cell
//     const TF xn = c.x0 + solve_quadratic<TF>( ( c.y0 - c.y1 ) / ( c.x0 - c.x1 ), c.y0, -mass_to_take, 0 );
//     const TF yn = interp( xn, c.x0, c.x1, c.y0, c.y1 );
//     func( Piece{ .i1 = c.i1, .k0 = c.x0, .k1 = c.x1, .x0 = c.x0, .x1 = xn, .y0 = c.y0, .y1 = yn } );
//     c.mass -= mass_to_take;
//     c.x0 = xn;
//     c.y0 = yn;
// }

// UTP void DTP::accumulate_gradients_dist( const Piece &p, TF g_dist, TF ref_x, TF potential, TensorView<TF,1,Cpu> grad_values ) const {
//     // Integrand: phi*(x) * phi_k(x),  phi*(x) = (x-ref_x)^2 - potential
//     // phi_{i1-1}(x) = (k1-x)/(k1-k0),  phi_{i1}(x) = (x-k0)/(k1-k0)
//     // Simpson's rule is exact (degree-2 × degree-1 = degree-3 polynomial)
//     const TF dk = p.k1 - p.k0;
//     if ( dk == 0 ) return;
//     const TF xm = ( p.x0 + p.x1 ) / 2;
//     const TF dx = ( p.x1 - p.x0 ) / 6;
//     const TF c = g_dist * coeff_values / dk;

//     auto phi_star = [&]( TF x ) { return ( x - ref_x ) * ( x - ref_x ) - potential; };

//     grad_values[ p.i1 - 1 ] += c * dx * (
//         phi_star( p.x0 ) * ( p.k1 - p.x0 ) +
//         4 * phi_star( xm ) * ( p.k1 - xm  ) +
//         phi_star( p.x1 ) * ( p.k1 - p.x1 )
//     );
//     grad_values[ p.i1 ] += c * dx * (
//         phi_star( p.x0 ) * ( p.x0 - p.k0 ) +
//         4 * phi_star( xm ) * ( xm   - p.k0 ) +
//         phi_star( p.x1 ) * ( p.x1 - p.k0 )
//     );
// }

// UTP void DTP::apply_normalization_correction( TF g_dist, TF phi_avg, TensorView<TF,1,Cpu> grad_values ) const {
//     for( PI k = 0; k < grad_values.size(); ++k ) {
//         TF A_k = 0;
//         if ( k > 0 )                      A_k += ( knot( k ) - knot( k - 1 ) ) / 2;
//         if ( k + 1 < grad_values.size() ) A_k += ( knot( k + 1 ) - knot( k ) ) / 2;
//         grad_values[ k ] -= g_dist * coeff_values * A_k * phi_avg;
//     }
// }

#undef UTP
#undef DTP

} // namespace sdot
