#pragma once

#include "PolynomialGrid.h"

namespace sdot {

#define UTP template<class TF,int ct_dim,int order,class Arch>
#define DTP PolynomialGrid<TF,ct_dim,order,Arch>

UTP DTP::PolynomialGrid( Values values, Bounds bounds, const Knots &knots, bool normalize ) : normalize( normalize ), values( values ), bounds( bounds ), knots( knots ), pif( values.rank() ), pf( values.rank() ) {
    ASSERT( values.rank() == knots.size() + 1 );
    ASSERT( values.size( knots.size() ) == nb_coeffs );

    // coeff_values
    coeff_values = 1;
    if ( normalize ) {
        TF measure = 0;
        for_each_in_range( pif.zeros(), grid_shape(), [&]( const auto &index ) {
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
        res.coeffs[ d ] = values( position, d );
    return res;
}

UTP TF DTP::piece_integral( Pi index, const Polynomial &pol ) const {
    if ( has_skew_or_rotation() )
        TODO;

    // bounds and moments per axis
    //   m0[d] = ∫ 1  dx  on [a,b] = b - a
    //   m1[d] = ∫ x  dx  on [a,b] = (b² - a²) / 2
    //   m2[d] = ∫ x² dx  on [a,b] = (b³ - a³) / 3
    auto a  = pf.with_func( [&]( PI d ) { return knot( d, index[ d ]     ); } );
    auto b  = pf.with_func( [&]( PI d ) { return knot( d, index[ d ] + 1 ); } );
    auto m0 = pf.with_func( [&]( PI d ) { return b[ d ] - a[ d ]; } );
    auto m1 = pf.with_func( [&]( PI d ) { return ( b[ d ] * b[ d ] - a[ d ] * a[ d ] ) / 2; } );
    auto m2 = pf.with_func( [&]( PI d ) { return ( b[ d ] * b[ d ] * b[ d ] - a[ d ] * a[ d ] * a[ d ] ) / 3; } );

    // volume of the cell = ∏ m0[d]
    TF vol = 1;
    for( PI d = 0; d < dim(); ++d )
        vol *= m0[ d ];

    TF res = 0;
    PI c = 0;

    // order 0: constant → ∫ 1 = vol
    res += pol.coeffs[ c++ ] * vol;

    // order 1: x_i → vol / m0[i] * m1[i]   (replace the i-th factor m0 by m1)
    if constexpr ( order >= 1 ) {
        for( PI i = 0; i < dim(); ++i )
            res += pol.coeffs[ c++ ] * ( vol / m0[ i ] * m1[ i ] );
    }

    // order 2: x_i * x_j  (i ≤ j)
    //   diagonal (i==j): replace m0[i] by m2[i]
    //   off-diagonal:    replace m0[i] and m0[j] by m1[i] and m1[j]
    if constexpr ( order >= 2 ) {
        for( PI i = 0; i < dim(); ++i ) {
            for( PI j = i; j < dim(); ++j ) {
                TF integ = ( i == j )
                    ? vol / m0[ i ] * m2[ i ]
                    : vol / m0[ i ] / m0[ j ] * m1[ i ] * m1[ j ];
                res += pol.coeffs[ c++ ] * integ;
            }
        }
    }

    if constexpr ( order >= 3 ) {
        TODO;
    }

    return res;
}

UTP TF DTP::knot( PI d, PI index ) const {
    if ( ! bounds.empty() )
        TODO;
    return TF( index ) / values.size( d );
}

UTP auto DTP::simplex_facet_integral( std::span<Pt> vertices, const Polynomial &value ) {
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

    // coeff index
    PI c = 0;

    // order 0: ∫ 1 dS = A
    TF res = value.coeffs[ c++ ] * A;

    if constexpr ( order >= 1 ) {
        // centroid of the facet in the ambient space R^n
        Pt centroid( n );
        for ( const auto &p : vertices )
            centroid += p;
        centroid /= TF( m + 1 );

        // order 1: ∫ x_i dS = A * centroid[i]  for all i ∈ 0..n-1
        for ( PI i = 0; i < n; ++i )
            res += value.coeffs[ c++ ] * ( A * centroid[ i ] );

        if constexpr ( order >= 2 ) {
            // order 2: ∫ x_i x_j dS = A/(m+1)/(m+2) * [ S_i*S_j + T_ij ]
            //   S_i = (m+1)*centroid[i],  T_ij = Σ_k vertices[k][i]*vertices[k][j]
            const TF fac = A / TF( m + 1 ) / TF( m + 2 );
            for ( PI i = 0; i < n; ++i ) {
                const TF Si = TF( m + 1 ) * centroid[ i ];
                for ( PI j = i; j < n; ++j ) {
                    const TF Sj = TF( m + 1 ) * centroid[ j ];
                    TF Tij = 0;
                    for ( const auto &p : vertices )
                        Tij += p[ i ] * p[ j ];
                    res += value.coeffs[ c++ ] * fac * ( Si * Sj + Tij );
                }
            }
        }
    }

    if constexpr ( order >= 3 )
        TODO;

    return res;
}

UTP auto DTP::simplex_integral( std::span<Pt> vertices, const Polynomial &value ) {
    const PI n = vertices.size() - 1;

    // Jacobian J, row r = points[r+1] - points[0]
    // volume V = det(J) / n!   (positive since points are well-oriented)
    auto J = SimpleSquareMatrix<TF,ct_dim,Arch>::with_func( n, [&]( PI r, PI c ) { return vertices[ r + 1 ][ c ] - vertices[ 0 ][ c ]; } );
    const TF V = J.determinant() / factorial( n );

    // coeff index
    PI c = 0;

    // order 0: ∫ 1 = V
    TF res = value.coeffs[ c++ ] * V;

    // order 1: ∫ x_i = V * centroid[i],   centroid = (1/(n+1)) * Σ_k points[k]
    if constexpr ( order >= 1 ) {
        Pt centroid( n );
        for ( const auto &p : vertices )
            centroid += p;
        centroid /= TF( n + 1 );

        for ( PI i = 0; i < n; ++i )
            res += value.coeffs[ c++ ] * ( V * centroid[ i ] );

        // order 2: ∫ x_i x_j = V/(n+1)/(n+2) * [ S_i*S_j + T_ij ]
        //   S_i = (n+1)*centroid[i]
        //   T_ij = Σ_k points[k][i] * points[k][j]
        if constexpr ( order >= 2 ) {
            const TF fac = V / TF( n + 1 ) / TF( n + 2 );
            for ( PI i = 0; i < n; ++i ) {
                const TF Si = TF( n + 1 ) * centroid[ i ];
                for ( PI j = i; j < n; ++j ) {
                    const TF Sj = TF( n + 1 ) * centroid[ j ];
                    TF Tij = 0;
                    for ( const auto &p : vertices )
                        Tij += p[ i ] * p[ j ];
                    res += value.coeffs[ c++ ] * fac * ( Si * Sj + Tij );
                }
            }
        }
    }

    if constexpr ( order >= 3 )
        TODO;

    return res;
}

UTP auto DTP::facet_integral( auto facet, const Polynomial &value ) {
    TF res = value.coeffs[ 0 ] * facet.measure();
    if ( order == 0 )
        return res;

    TODO;
}

UTP auto DTP::integral( auto cell, const Polynomial &value ) {
    std::vector<Pt> points;
    points.reserve( cell.nb_vertices() );
    for ( const auto &v : cell.vertices )
        points.push_back( v.pos );
    return simplex_integral( points, value );
}















UTP void DTP::_for_each_piece( const auto &cell, const auto &func, auto beg, auto end, auto &cur, PI d ) const {
    if ( d == dim() ) {
        func( cell, coeff_values * values[ cur ] );
        return;
    }

    for( PI v = beg[ d ]; v < end[ d ]; ++v ) {
        auto new_cell = cell;

        new_cell.cut( pf.value_at( d, - 1 ), - knot( d, v + 0 ), {} );
        new_cell.cut( pf.value_at( d, + 1 ), + knot( d, v + 1 ), {} );

        cur[ d ] = v;
        _for_each_piece( new_cell, func, beg, end, cur, d + 1 );
    }
}


UTP void DTP::for_each_piece( const auto &cell, auto &&func ) const {
    using namespace std;
    if ( ! bounds.empty() )
        TODO;

    // FP boundaries
    const Pt mi = cell.min_pos( pf.ones() );
    const Pt ma = cell.max_pos( pf.zeros() );
    for( PI d = 0; d < dim(); ++d )
        if ( ma[ d ] <= mi[ d ] )
            return;

    // I boundaries
    sdot::PointFactory<PI,ct_dim,Arch> pif( dim() );
    auto b = pif.with_func( [&]( PI d ) { return floor( mi[ d ] * values.size( d ) ); } );
    auto e = pif.with_func( [&]( PI d ) { return ceil( ma[ d ] * values.size( d ) ); } );

    // traversal
    auto c = b;
    _for_each_piece( cell, func, b, e, c, 0 );
}

UTP T_U auto DTP::englobing_cell( PI /* dim */, typename Cell<U,ct_dim,Arch>::CellInfo cell_info, typename Cell<U,ct_dim,Arch>::CutInfo cut_info ) const -> Cell<U,ct_dim,Arch>{
    if ( ! bounds.empty() )
        TODO;
    return Cell<U,ct_dim,Arch>::axis_aligned_hypercube( pf.zeros(), pf.ones(), cell_info, cut_info );
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
