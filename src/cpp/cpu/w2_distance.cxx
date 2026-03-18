#pragma once

#include "../support/parallel_for.h"
#include "../support/P.h"
#include "w2_distance.h"

namespace sdot {

T_T void w2_distance( DiracSet<const T,0> diracs, Affine1d<const T,0> points, TensorView<T,0> distance, TensorView<T,1> barycenters, TensorView<T,1> potentials, TensorView<T,1> cuts ) {
    using TF = typename IntermediateScalarType<T>::type;

    const std::vector<PI> dirac_indices = diracs.arg_sort();
    const TF diracs_mass = diracs.mass();
    const TF points_mass = points.mass();

    if ( diracs_mass == 0 ) throw std::runtime_error( "mass of the diracs is null" );
    if ( points_mass == 0 ) throw std::runtime_error( "mass of the points is null" );
    const TF dirac_scale = points_mass / diracs_mass;

    PieceOfAffine1d<TF> current_piece = points.get_first_piece();
    TF prev_cut_position = current_piece.x0;
    TF prev_dirac_x = 0;
    TF potential = 0;
    TF w2 = 0;
    for( PI i = 0; i < diracs.nb_diracs(); ++i ) {
        const PI dirac_index = dirac_indices[ i ];

        const TF dirac_mass = dirac_scale * static_cast<TF>( diracs.ws[ dirac_index ] );
        const TF dirac_x = diracs.xs[ dirac_index ];
        if ( dirac_mass <= 0.0 )
            throw std::runtime_error( "dirac_mass must be strictly positive" );

        // Update potential using jumping relation at last_ti
        if ( ! potentials.empty() ) {
            if ( i > 0 )
                 potential += std::pow( dirac_x - prev_cut_position, 2 ) - std::pow( prev_dirac_x - prev_cut_position, 2 );
            potentials[ dirac_index ] = static_cast<T>( potential );
        }

        TF moment = 0;
        const TF cut_position = points.take_some_mass( current_piece, dirac_mass, [&]( const PieceOfAffine1d<TF> &piece ) {
            w2 += piece.w2_dist( dirac_x );
            moment += piece.moment();
        } );

        if ( ! barycenters.empty() )
            barycenters[ dirac_index ] = static_cast<T>( moment / dirac_mass );

        if ( ! cuts.empty() ) {
            cuts[ 2 * dirac_index + 0 ] = prev_cut_position;
            cuts[ 2 * dirac_index + 1 ] = cut_position;
        }

        prev_cut_position = cut_position;
        prev_dirac_x = dirac_x;
    }

    if ( ! distance.empty() )
       distance() = static_cast<T>( w2 );
}

/// Wasserstein 2 distance
T_T void w2_distance( DiracSet<const T,1> diracs, Affine1d<const T,1> functions, TensorView<T,1> distance, TensorView<T,2> barycenters, TensorView<T,2> potentials, TensorView<T,2> cuts ) {
    parallel_for<PI>( 0, ASSERTED_EQUAL( diracs.nb_rows(), functions.nb_rows() ), [&]( PI r ) {
        w2_distance( diracs.row( r ), functions.row( r ), distance.row( r ), barycenters.row( r ), potentials.row( r ), cuts.row( r ) );
    });
}

T_T void w2_distance_backward( TensorView<const T,0> grad_distance, TensorView<const T,1> grad_barycenters, TensorView<const T,1> barycenters, TensorView<const T,1> potentials, TensorView<const T,1> cuts, DiracSet<const T,0> diracs, Affine1d<const T,0> points, DiracSet<T,0> grad_diracs, Affine1d<T,0> grad_functions ) {
    using TF = IntermediateScalarType<T>::type;
    using namespace std;

    const vector<PI> dirac_indices = diracs.arg_sort();
    const TF diracs_mass = diracs.mass();
    const TF points_mass = points.mass();
    const TF dirac_scale = points_mass / diracs_mass;

    // grad_diracs.xs
    const TF g_dist = static_cast<TF>( grad_distance[ 0 ] ) * dirac_scale;
    for( PI i = 0; i < diracs.nb_diracs(); ++i )
        grad_diracs.xs[ i ] = static_cast<T>( 2 * g_dist * diracs.ws[ i ] * ( diracs.xs[ i ] - barycenters[ i ] ) );

    // grad_diracs.ws
    if ( ! grad_diracs.ws.empty() ) {
        TF dws = 0, dirac_ratio = 1;
        for( PI i = 1; i < diracs.nb_diracs(); ++i ) {
            const PI d0 = dirac_indices[ i - 1 ];
            const PI d1 = dirac_indices[ i - 0 ];
            const TF p0 = diracs.xs[ d0 ];
            const TF p1 = diracs.xs[ d1 ];
            const TF w0 = diracs.ws[ d0 ];

            const TF cp = static_cast<TF>( cuts[ 2 * d1 + 0 ] );
            dirac_ratio -= w0 / diracs_mass;

            dws += dirac_ratio * ( pow( cp - p0, 2 ) - pow( cp - p1, 2 ) );
        }

        grad_diracs.ws[ dirac_indices[ 0 ] ] = static_cast<T>( g_dist * dws );
        for( PI i = 1; i < diracs.nb_diracs(); ++i ) {
            const PI d0 = dirac_indices[ i - 1 ];
            const PI d1 = dirac_indices[ i - 0 ];
            const TF p0 = diracs.xs[ d0 ];
            const TF p1 = diracs.xs[ d1 ];

            const TF cp = static_cast<TF>( cuts[ 2 * d1 + 0 ] );
            dws -= pow( cp - p0, 2 ) - pow( cp - p1, 2 );

            grad_diracs.ws[ d1 ] = static_cast<T>( g_dist * dws );
        }
    }

    // grad_points
    if ( ! grad_functions.xs.empty() || ! grad_functions.ys.empty() ) {
        for( PI j = 0; j < points.nb_points(); ++j ) {
            grad_functions.xs[ j ] = 0;
            grad_functions.ys[ j ] = 0;
        }

        //
    }


    // PieceOfAffine1d<TF> current_piece = points.get_first_piece();
    // vector<PI> dirac_indices = diracs.arg_sort();
    // for( PI i = 0; i < diracs.nb_diracs(); ++i ) {
    //     const PI dirac_index = dirac_indices[ i ];

    //     const TF dirac_mass = dirac_scale * static_cast<TF>( diracs.ws[ dirac_index ] );
    //     const TF potential = static_cast<TF>( potentials[ dirac_index ] );
    //     const TF dirac_x = diracs.xs[ dirac_index ];

    //     points.take_some_mass( current_piece, dirac_mass, [&]( const PieceOfAffine1d<TF> &piece ) {
    //         const PI j0 = piece.index - 1;
    //         const PI j1 = piece.index - 0;

    //         TF g0 = 0;
    //         TF g1 = 0;
    //         // The gradient w.r.t Yj is the integral of psi^c = (x_i-t)^2 - psi_i + mean_psi
    //         piece.integrate_w2_shape_functions( dirac_x, potential - mean_potentials, points.xs[ j0 ], points.xs[ j1 ], g0, g1 );

    //         grad_functions.ys[ j0 ] += static_cast<T>( g_dist * g0 );
    //         grad_functions.ys[ j1 ] += static_cast<T>( g_dist * g1 );
    //     } );
    // }
}

/// Gradients of Wasserstein 2 distance
T_T void w2_distance_backward( TensorView<const T,1> grad_distance, TensorView<const T,2> grad_barycenters, TensorView<const T,2> barycenters, TensorView<const T,2> potentials, TensorView<const T,2> cuts, DiracSet<const T,1> diracs, Affine1d<const T,1> functions, DiracSet<T,1> grad_diracs, Affine1d<T,1> grad_functions ) {
    parallel_for<PI>( 0, ASSERTED_EQUAL( diracs.nb_rows(), functions.nb_rows() ), [&]( PI r ) {
        w2_distance_backward( grad_distance.row( r ), grad_barycenters.row( r ), barycenters.row( r ), potentials.row( r ), cuts.row( r ), diracs.row( r ), functions.row( r ), grad_diracs.row( r ), grad_functions.row( r ) );
    });
}


} // namespace sdot
