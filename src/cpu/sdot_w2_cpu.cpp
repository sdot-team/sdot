// #include <iostream>
// #include <numeric>
// #include <utility>
#include "sdot_w2_cpu.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>
#include <cmath>

#include "../P.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

using namespace std;

using PI = size_t;
using TF = double;
using TS = float;

using TV = vector<TF>;

static TV get_dirac_indices( const TS *dirac_xs, const TS *dirac_ws, PI nb_diracs ) {
    TV dirac_indices( nb_diracs );
    for( PI i = 0; i < nb_diracs; ++i )
        dirac_indices[ i ] = i;

    std::sort( dirac_indices.begin(), dirac_indices.end(), [&]( PI a, PI b ) {
        return dirac_xs[ a ] < dirac_xs[ b ];
    });

    return dirac_indices;
}

static TF get_diracs_mass( const TS *dirac_xs, const TS *dirac_ws, PI nb_diracs ) {
    TF res = 0;
    for( PI i = 0; i < nb_diracs; ++i )
        res += dirac_ws[ i ];
    return res;
}

static TF get_points_mass( const TS *point_xs, const TS *point_ys, PI nb_points ) {
    TF res = 0;
    TF px0 = point_xs[ 0 ];
    TF py0 = point_ys[ 0 ];
    for( PI j = 1; j < nb_points; ++j ) {
        const TF px1 = point_xs[ j ];
        const TF py1 = point_ys[ j ];

        res += ( py0 + py1 ) * ( px1 - px0 );

        px0 = px1;
        py0 = py1;
    }
    res /= 2;

    return res;
}

TF solve_quadratic_2( TF a, TF b, TF c, TF default_value ) {
    const TF scale = std::max( { abs( a ), abs( b ), abs( c ) } );
    if ( scale == 0.0 )
        return 0;

    const TF eps = std::numeric_limits<TF>::epsilon() * scale;
    if ( abs( a ) < eps ) {
        if ( abs( b ) < eps )
            return default_value;
        return - c / b;
    }

    TF delta = b * b - 2 * a * c;
    if ( delta < eps * eps ) // Proche de zéro
        return - b / a;

    // double q = -0.5 * (b + ( (b >= 0) ? sqrt_delta : -sqrt_delta ));

    //     // x1 utilise q pour garantir la précision
    //     double x1 = q / a;
    //     // x2 est déduit par la propriété x1*x2 = c/a -> x2 = c/(a*x1) = c/q
    //     double x2 = c / q;
    return ( sqrt( max( TF( 0 ), delta ) ) - b ) / a;
}

TF get_x_new( TF x0, TF x1, TF y0, TF y1, TF mass_to_take ) {
    if ( x0 == x1 )
        return x0;

    // integral( y0 + ( y1 - y0 ) * ( x - x0 ) / ( x1 - x0 ), x, x0, v ) = m
    //  dx ^ 2 * ( y0 - y1 ) / ( x0 - x1 ) / 2 + dx * y0 - m
    const TF a = ( y0 - y1 ) / ( x0 - x1 );
    const TF c = - mass_to_take;
    const TF b = y0;
    return x0 + solve_quadratic_2( a, b, c, 0 );
}

TF partial_w2( TF d, TF x0, TF x1, TF y0, TF y1 ) {
    // integral( ( x - dirac_x )^2 * ( y0 + ( y1 - y0 ) * ( x - x0 ) / ( x1 - x0 ) ), x, x0, x1 )
    return ( x0 - x1 ) * (
        + 4 * d * ( x0 * ( 2 * y0 + y1 ) + x1 * ( y0 + 2 * y1 ) )
        - x0 * x0 * ( 3 * y0 + y1 )
        - x1 * x1 * ( y0 + 3 * y1 )
        - 2 * x0 * x1 * ( y0 + y1 )
        - 6 * d * d * ( y0 + y1 )
    ) / 12;
}

TF partial_moment( TF x0, TF x1, TF y0, TF y1 ) {
    // integral( x * ( y0 + ( y1 - y0 ) * ( x - x0 ) / ( x1 - x0 ) ), x, x0, x1 )
    // integral(
    //      + x * ( py0 - ( py1 - py0 ) * px0 / ( px1 - px0 ) )
    //      + x * x * ( py1 - py0 ) / ( px1 - px0 )
    //  )
    // integral( x * a + x * x * b  )
    const TF b = ( y1 - y0 ) / ( x1 - x0 );
    const TF a = ( y0 - b * x0 );
    return ( pow( x1, 2 ) - pow( x0, 2 ) ) * a / 2
         + ( pow( x1, 3 ) - pow( x0, 3 ) ) * b / 3;
}

void sdot_w2_cpu_single( const TS *dirac_xs, const TS *dirac_ws, PI nb_diracs, const TS *point_xs, const TS *point_ys, PI nb_points, TS *w2_squared, TS *w2_barycenters ) {
    if ( nb_points < 2 )
        throw std::runtime_error( "For affine function, we need at least 2 points" );

    // sweeps
    const TV dirac_indices = get_dirac_indices( dirac_xs, dirac_ws, nb_diracs );
    const TF diracs_mass = get_diracs_mass( dirac_xs, dirac_ws, nb_diracs );
    const TF points_mass = get_points_mass( point_xs, point_ys, nb_points );

    // mass consistency
    if ( diracs_mass == 0 )
        throw std::runtime_error( "diracs_mass is null" );
    if ( points_mass == 0 )
        throw std::runtime_error( "mass of the points is null" );
    const TF dirac_scale = points_mass / diracs_mass;

    // get values for each dirac
    TF px0 = point_xs[ 0 ];
    TF px1 = point_xs[ 1 ];
    TF py0 = point_ys[ 0 ];
    TF py1 = point_ys[ 1 ];
    PI point_index = 1;

    TF remaining_mass = ( px1 - px0 ) * ( py1 + py0 ) / 2;
    TF w2 = 0;
    for( PI i = 0; i < nb_diracs; ++i ) {
        const PI dirac_index = dirac_indices[ i ];
        const TF dirac_x = dirac_xs[ i ];

        const TF dirac_mass = dirac_scale * dirac_ws[ dirac_index ];
        if ( dirac_mass <= 0 )
            throw std::runtime_error( "dirac_mass is not strictly positive" );

        // stay on the first interval ?
        if ( dirac_mass <= remaining_mass ) {
            const TF pxn = get_x_new( px0, px1, py0, py1, dirac_mass );
            const TF pyn = py0 + ( py1 - py0 ) * ( pxn - px0 ) / ( px1 - px0 );

            w2_barycenters[ dirac_index ] = partial_moment( px0, pxn, py0, pyn ) / dirac_mass;
            w2 += partial_w2( dirac_x, px0, pxn, py0, pyn );

            remaining_mass -= dirac_mass;
            px0 = pxn;
            py0 = pyn;
        } else {
            // take the first interval
            TF moment = partial_moment( px0, px1, py0, py1 );
            w2 += partial_w2( dirac_x, px0, px1, py0, py1 );
            TF mass_to_take = dirac_mass - remaining_mass;

            // full intermediate intervals
            while ( true ) {
                px0 = px1;
                py0 = py1;

                if ( ++point_index < nb_points ) {
                    px1 = point_xs[ point_index ];
                    py1 = point_ys[ point_index ];

                    remaining_mass = ( px1 - px0 ) * ( py1 + py0 ) / 2;
                } else {
                    px1 = 1.1 * point_xs[ nb_points - 1 ] - point_xs[ 0 ] * 0.1;
                    py1 = numeric_limits<TF>::max();

                    remaining_mass = numeric_limits<TF>::max();
                }

                if ( mass_to_take <= remaining_mass )
                    break;

                moment += partial_moment( px0, px1, py0, py1 );
                w2 += partial_w2( dirac_x, px0, px1, py0, py1 );
                mass_to_take -= remaining_mass;
            }

            // cut interval
            const TF pxn = get_x_new( px0, px1, py0, py1, mass_to_take );
            const TF pyn = py0 + ( py1 - py0 ) * ( pxn - px0 ) / ( px1 - px0 );

            moment += partial_moment( px0, pxn, py0, pyn );
            w2 += partial_w2( dirac_x, px0, pxn, py0, pyn );

            remaining_mass -= mass_to_take;
            px0 = pxn;
            py0 = pyn;

            w2_barycenters[ dirac_index ] = moment / dirac_mass;
        }
    }

    if ( w2_squared )
        *w2_squared = TS( w2 );
}

void sdot_w2_cpu( const TS *dirac_xs, const TS *dirac_ws, PI nb_diracs, const TS *point_xs, const TS *point_ys, PI nb_points, PI batch_size, TS *w2_squared, TS *w2_barycenters ) {
    #pragma omp parallel for
    for( PI b = 0; b < batch_size; ++b ) {
        sdot_w2_cpu_single(
            dirac_xs + b * nb_diracs, dirac_ws + b * nb_diracs, nb_diracs,
            point_xs + b * nb_points, point_ys + b * nb_points, nb_points,
            w2_squared ? w2_squared + b : nullptr,
            w2_barycenters ? (w2_barycenters + b * nb_diracs) : nullptr
        );
    }
}


void sdot_w2_backward_cpu(
    const TS *grad_distance, const TS *grad_barycenters,
    const TS *w2_barycenters,
    const TS *dirac_xs, const TS *dirac_ws, PI nb_diracs,
    const TS *points_xs, const TS *points_ys, PI nb_points,
    PI batch_size,
    TS *grad_dirac_xs,
    TS *grad_dirac_ws,
    TS *grad_point_xs,
    TS *grad_point_ys
) {
    for( PI i = 0; i < nb_points * batch_size; ++i ) {
        grad_dirac_xs[ i ] = 2 * ( dirac_xs[ i ] - w2_barycenters[ i ] );
        grad_dirac_ws[ i ] = 0;
        grad_point_xs[ i ] = 0;
        grad_point_ys[ i ] = 0;
    }
}
