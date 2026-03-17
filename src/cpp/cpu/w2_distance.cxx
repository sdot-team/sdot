#pragma once

#include "../support/parallel_for.h"
#include "../support/P.h"
#include "w2_distance.h"
// #include <stdexcept>
// #include <algorithm>
// #include <cstddef>
// #include <limits>
// #include <cmath>

namespace sdot {

T_T void w2_distance( DiracSet<const T,0> diracs, Affine1d<const T,0> points, TensorView<T,0> w2_squared, TensorView<T,1> w2_barycenters, TensorView<T,1> w2_potentials ) {
    using TF = typename IntermediateScalarType<T>::type;

    std::vector<PI> dirac_indices = diracs.arg_sort();
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
        if ( ! w2_potentials.empty() ) {
            if ( i > 0 )
                 potential += std::pow( dirac_x - prev_cut_position, 2 ) - std::pow( prev_dirac_x - prev_cut_position, 2 );
            w2_potentials[ dirac_index ] = static_cast<T>( potential );
        }

        TF moment = 0;
        prev_cut_position = points.take_some_mass( current_piece, dirac_mass, [&]( const PieceOfAffine1d<TF> &piece ) {
            w2 += piece.w2_dist( dirac_x );
            moment += piece.moment();
        } );

        if ( ! w2_barycenters.empty() )
            w2_barycenters[ dirac_index ] = static_cast<T>( moment / dirac_mass );

        prev_dirac_x = dirac_x;
    }

    if ( ! w2_squared.empty() )
       w2_squared() = static_cast<T>( w2 );
}

/// Wasserstein 2 distance
T_T void w2_distance( DiracSet<const T,1> diracs, Affine1d<const T,1> functions, TensorView<T,1> w2_squared, TensorView<T,2> w2_barycenters, TensorView<T,2> w2_potentials ) {
    parallel_for<PI>( 0, ASSERTED_EQUAL( diracs.nb_rows(), functions.nb_rows() ), [&]( PI r ) {
        w2_distance( diracs.row( r ), functions.row( r ), w2_squared.row( r ), w2_barycenters.row( r ), w2_potentials.row( r ) );
    });
}

T_T void w2_distance_backward( TensorView<const T,0> grad_w2_squared, TensorView<const T,1> grad_w2_barycenters, TensorView<const T,1> w2_barycenters, TensorView<const T,1> w2_potentials, DiracSet<const T,0> diracs, Affine1d<const T,0> functions, DiracSet<T,0> grad_diracs, Affine1d<T,0> grad_functions ) {
    using TF = IntermediateScalarType<T>::type;
    const TF g_dist = static_cast<TF>( grad_w2_squared[ 0 ] );

    // 1. Potentiel moyen (pour la correction de masse de g)
    // TF mean_psi = 0;
    // TF total_dirac_ws = 0;
    // for( PI i = 0; i < diracs.nb_diracs(); ++i ) {
    //     mean_psi += static_cast<TF>( diracs.ws[ i ] ) * static_cast<TF>( w2_potentials[ i ] );
    //     total_dirac_ws += static_cast<TF>( diracs.ws[ i ] );
    // }
    // mean_psi /= total_dirac_ws;

    // 2. Gradients for diracs
    for( PI num_dirac = 0; num_dirac < diracs.nb_diracs(); ++num_dirac ) {
        grad_diracs.xs[ num_dirac ] = static_cast<T>( 2 * g_dist * diracs.ws[ num_dirac ] * ( diracs.xs[ num_dirac ] - w2_barycenters[ num_dirac ] ) );
        grad_diracs.ws[ num_dirac ] = 0; // static_cast<T>( g_dist * ( static_cast<TF>( w2_potentials[ num_dirac ] ) - mean_psi ) );
    }

    // // 3. Gradients for target function g (Xj, Yj)
    // std::vector<PI> dirac_indices = diracs.arg_sort();
    // const TF diracs_mass = diracs.mass();
    // const TF points_mass = functions.mass();
    // const TF dirac_scale = points_mass / diracs_mass;

    // PieceOfAffine1d<TF> current_piece = functions.get_first_piece();

    // for( PI j = 0; j < functions.nb_points(); ++j ) {
    //     grad_functions.xs[ j ] = 0;
    //     grad_functions.ys[ j ] = 0;
    // }

    // for( PI i = 0; i < diracs.nb_diracs(); ++i ) {
    //     const PI idx = dirac_indices[ i ];
    //     const TF dirac_mass = dirac_scale * static_cast<TF>( diracs.ws[ idx ] );
    //     const TF dirac_x = diracs.xs[ idx ];
    //     const TF psi_i = static_cast<TF>( w2_potentials[ idx ] );

    //     functions.take_some_mass( current_piece, dirac_mass, [&]( const PieceOfAffine1d<TF> &piece ) {
    //         const PI j_left = piece.index - 1;
    //         const PI j_right = piece.index;

    //         TF g_left = 0;
    //         TF g_right = 0;
    //         // The gradient w.r.t Yj is the integral of psi^c = (x_i-t)^2 - psi_i + mean_psi
    //         piece.integrate_w2_shape_functions( dirac_x, psi_i - mean_psi, functions.xs[ j_left ], functions.xs[ j_right ], g_left, g_right );

    //         grad_functions.ys[ j_left ] += static_cast<T>( g_dist * g_left );
    //         grad_functions.ys[ j_right ] += static_cast<T>( g_dist * g_right );
    //     } );
    // }
}

/// Gradients of Wasserstein 2 distance
T_T void w2_distance_backward( TensorView<const T,1> grad_w2_squared, TensorView<const T,2> grad_w2_barycenters, TensorView<const T,2> w2_barycenters, TensorView<const T,2> w2_potentials, DiracSet<const T,1> diracs, Affine1d<const T,1> functions, DiracSet<T,1> grad_diracs, Affine1d<T,1> grad_functions ) {
    parallel_for<PI>( 0, ASSERTED_EQUAL( diracs.nb_rows(), functions.nb_rows() ), [&]( PI r ) {
        w2_distance_backward( grad_w2_squared.row( r ), grad_w2_barycenters.row( r ), w2_barycenters.row( r ), w2_potentials.row( r ), diracs.row( r ), functions.row( r ), grad_diracs.row( r ), grad_functions.row( r ) );
    });
}


} // namespace sdot
