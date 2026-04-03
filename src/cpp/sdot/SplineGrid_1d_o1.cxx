#pragma once

#include "support/solve_quadratic.h"
#include "support/P.h"
#include "SplineGrid_1d_o1.h"

namespace sdot {

#define UTP template<class TF,class Knots>
#define DTP SplineGrid<TF,1,Knots>

UTP DTP::SplineGrid( Values values, Knots knots ) : values( values ), knots( knots ) {
    if ( values.size() < 2 )
        ERROR( "..." );

    TF s = 0;
    TF x0 = knots[ 0 ];
    TF y0 = values[ 0 ];
    for( PI i1 = 1; i1 < values.size(); ++i1 ) {
        const TF x1 = knots[ i1 ];
        const TF y1 = values[ i1 ];

        s += ( x1 - x0 ) * ( y0 + y1 ) / 2;

        x0 = x1;
        y0 = y1;
    }

    if ( s == 0 )
        ERROR( "null mass for g" );
    coeff_values = 1 / s;
}

UTP TF DTP::Piece::value_at( TF x ) const {
    return y0 + ( y1 - y0 ) * ( x - x0 ) / ( x1 - x0 );
}

UTP void DTP::Piece::take_some_mass( TF mass_to_take, auto &&func ) {
    // enough mass in the current piece ?
    if ( mass_to_take <= mass ) {
        const TF xn = x0 + solve_quadratic<TF>( ( y0 - y1 ) / ( x0 - x1 ), y0, - mass_to_take, 0 );
        const TF yn = value_at( xn );

        func( Part{
            .i1 = i1,
            .x0 = x0,
            .x1 = xn,
            .y0 = y0,
            .y1 = yn
        } );

        mass -= mass_to_take;
        x0 = xn;
        y0 = yn;

        return;
    }

    // else, use the current piece, get to the next, ...
    mass_to_take -= mass;

    func( Part{
        .i1 = i1,
        .x0 = x0,
        .x1 = x1,
        .y0 = y0,
        .y1 = y1
    } );

    // while we can use entire parts
    while ( true ) {
        ++i1;
        if ( i1 == values.size() ) {
            mass = 0;
            x0 = x1;
            y0 = y1;
            --i1;
            return;
        }

        x0 = x1;
        x1 = knots[ i1 ];
        y0 = y1;
        y1 = coeff_values * values[ i1 ];

        mass = ( x1 - x0 ) * ( y0 + y1 ) / 2;

        if ( mass_to_take <= mass )
           break;

        // full interval
        mass_to_take -= mass;

        func( Part{
            .i1 = i1,
            .x0 = x0,
            .x1 = x1,
            .y0 = y0,
            .y1 = y1
        } );
    }

    // take the beginning
    const TF xn = x0 + solve_quadratic<TF>( ( y0 - y1 ) / ( x0 - x1 ), y0, - mass_to_take, 0 );
    const TF yn = value_at( xn );

    func( Part{
        .i1 = i1,
        .x0 = x0,
        .x1 = xn,
        .y0 = y0,
        .y1 = yn
    } );

    mass -= mass_to_take;
    x0 = xn;
    y0 = yn;
}

UTP typename DTP::Piece DTP::first_piece() const {
    const TF y0 = coeff_values * values[ 0 ];
    const TF y1 = coeff_values * values[ 1 ];
    const TF x0 = knots[ 0 ];
    const TF x1 = knots[ 1 ];
     return Piece{
        .coeff_values = coeff_values,
        .values = values,
        .knots = knots,
        .mass = ( x1 - x0 ) * ( y0 + y1 ) / 2,
        .i1 = 1,
        .x0 = x0,
        .x1 = x1,
        .y0 = y0,
        .y1 = y1
    };
}

UTP typename DTP::Piece DTP::last_piece() const {
    const PI i0 = values.size() - 2;
    const PI i1 = values.size() - 1;
    const TF y0 = coeff_values * values[ i0 ];
    const TF y1 = coeff_values * values[ i1 ];
    const TF x0 = knots[ i0 ];
    const TF x1 = knots[ i1 ];
     return Piece{
        .coeff_values = coeff_values,
        .values = values,
        .knots = knots,
        .mass = ( x1 - x0 ) * ( y0 + y1 ) / 2,
        .i1 = i1,
        .x0 = x0,
        .x1 = x1,
        .y0 = y0,
        .y1 = y1
    };
}

UTP TF DTP::Part::w2_dist( TF dirac_pos ) const {
    return ( x0 - x1 ) * (
        + 4 * dirac_pos * ( x0 * ( 2 * y0 + y1 ) + x1 * ( y0 + 2 * y1 ) )
        - 6 * dirac_pos * dirac_pos * ( y0 + y1 )
        - x0 * x0 * ( 3 * y0 + y1 )
        - x1 * x1 * ( y0 + 3 * y1 )
        - 2 * x0 * x1 * ( y0 + y1 )
    ) / 12;
}

UTP TF DTP::Part::moment() const {
    if ( x0 == x1 )
       return 0.0;
    const TF b = ( y1 - y0 ) / ( x1 - x0 );
    const TF a = ( y0 - b * x0 );
    // TODO: better precision handling
    return ( std::pow( x1, 2 ) - std::pow( x0, 2 ) ) * a / 2
         + ( std::pow( x1, 3 ) - std::pow( x0, 3 ) ) * b / 3;
}

UTP void DTP::accumulate_gradients( const Part &part, TF coeff_x2, TF coeff_x1, TF coeff_x0, TensorView<TF,1,Cpu> grad_values ) const {
    const TF X0 = knots[ part.i1 - 1 ];
    const TF X1 = knots[ part.i1 ];
    const TF inv_H = 1 / ( X1 - X0 );

    auto V = [&]( TF x ) { return coeff_x2 * x * x + coeff_x1 * x + coeff_x0; };

    // Simpson's rule for cubic (V is quadratic, shape is linear)
    auto integrate = [&]( auto &&f ) {
        return ( part.x1 - part.x0 ) / 6 * ( f( part.x0 ) + 4 * f( ( part.x0 + part.x1 ) / 2 ) + f( part.x1 ) );
    };

    grad_values[ part.i1 - 1 ] += integrate( [&]( TF x ) { return V( x ) * ( X1 - x ) * inv_H; } );
    grad_values[ part.i1 ]     += integrate( [&]( TF x ) { return V( x ) * ( x - X0 ) * inv_H; } );
}

#undef UTP
#undef DTP

} // namespace sdot
