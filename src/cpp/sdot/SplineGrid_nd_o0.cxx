#pragma once

#include "SplineGrid_nd_o0.h"

namespace sdot {

#define UTP template<class TF,int ct_dim,class Arch>
#define DTP SplineGrid<TF,ct_dim,0,Arch>

UTP DTP::SplineGrid( Values values, Bounds bounds, const std::vector<Knots> & ) : values( values ), bounds( bounds ) {
    // if ( values.size() < 2 )
    //     ERROR( "..." );

    // TF s = 0;
    // TF x0 = this->knot( 0 );
    // TF y0 = values[ 0 ];
    // for( PI i1 = 1; i1 < values.size(); ++i1 ) {
    //     const TF x1 = this->knot( i1 );
    //     const TF y1 = values[ i1 ];
    //     s += ( x1 - x0 ) * ( y0 + y1 ) / 2;
    //     x0 = x1;
    //     y0 = y1;
    // }

    // if ( s == 0 )
    //     ERROR( "null mass for g" );
    // coeff_values = 1 / s;
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

// UTP void DTP::accumulate_gradients_dist( const Piece &p, TF g_dist, TF ref_x, TF potential, TensorView<TF,1,Arch> grad_values ) const {
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

UTP DTP::Cell DTP::base_cell( PI dim ) const {
    PointFactory<TF,ct_dim,Cpu> pf( dim );
    return Cell::axis_aligned_hypercube( pf.zeros(), pf.ones() );
}

#undef UTP
#undef DTP

} // namespace sdot
