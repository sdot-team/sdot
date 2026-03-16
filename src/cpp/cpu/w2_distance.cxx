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

T_T void w2_distance( DiracSet<const T,0> diracs, Affine1d<const T,0> points, TensorView<T,0> w2_squared, TensorView<T,1> w2_barycenters ) {
    using TF = typename IntermediateScalarType<T>::type;

    // w2_barycenters[ dirac_index ] = static_cast<T>(partial_moment(px0, pxn, py0, pyn) / dirac_mass);
    // w2 += partial_w2(dirac_x, px0, pxn, py0, pyn);

    std::vector<PI> dirac_indices = diracs.arg_sort();
    const TF diracs_mass = diracs.mass();
    const TF points_mass = points.mass();

    if ( diracs_mass == 0 ) throw std::runtime_error( "mass of the diracs is null" );
    if ( points_mass == 0 ) throw std::runtime_error( "mass of the points is null" );
    const TF dirac_scale = points_mass / diracs_mass;

    PieceOfAffine1d<TF> current_piece = points.get_first_piece();
    TF w2 = 0;
    for( PI i = 0; i < diracs.nb_diracs(); ++i ) {
        const PI dirac_index = dirac_indices[ i ];

        const TF dirac_mass = dirac_scale * static_cast<TF>( diracs.ws[ dirac_index ] );
        const TF dirac_x = diracs.xs[ dirac_index ];
        if ( dirac_mass <= 0.0 )
            throw std::runtime_error( "dirac_mass must be strictly positive" );

        TF moment = 0;
        points.take_some_mass( current_piece, dirac_mass, [&]( const PieceOfAffine1d<TF> &piece ) {
            w2 += piece.w2_dist( dirac_x );
            moment += piece.moment();
        } );

        if ( ! w2_barycenters.empty() )
            w2_barycenters[ dirac_index ] = static_cast<T>( moment / dirac_mass );
    }

    if ( ! w2_squared.empty() )
       w2_squared() = static_cast<T>( w2 );
}

/// Wasserstein 2 distance
T_T void w2_distance( DiracSet<const T,1> diracs, Affine1d<const T,1> functions, TensorView<T,1> w2_squared, TensorView<T,2> w2_barycenters ) {
    parallel_for<PI>( 0, ASSERTED_EQUAL( diracs.nb_rows(), functions.nb_rows() ), [&]( PI r ) {
        w2_distance( diracs.row( r ), functions.row( r ), w2_squared.row( r ), w2_barycenters.row( r ) );
    });
}

T_T void w2_distance_backward( TensorView<const T,0> grad_w2_squared, TensorView<const T,1> grad_w2_barycenters, TensorView<const T,1> w2_barycenters, DiracSet<const T,0> diracs, Affine1d<const T,0> functions, DiracSet<T,0> grad_diracs, Affine1d<T,0> grad_functions ) {
    for( PI num_dirac = 0; num_dirac < diracs.nb_diracs(); ++num_dirac ) {
        grad_diracs.xs[ num_dirac ] = grad_w2_squared[ 0 ] * 2 * diracs.ws[ num_dirac ] * ( diracs.xs[ num_dirac ] - w2_barycenters[ num_dirac ] );
        grad_diracs.ws[ num_dirac ] = 0;
    }
    for( PI num_point = 0; num_point < functions.nb_points(); ++num_point ) {
        grad_functions.xs[ num_point ] = 0;
        grad_functions.ys[ num_point ] = 0;
    }
}

/// Gradients of Wasserstein 2 distance
T_T void w2_distance_backward( TensorView<const T,1> grad_w2_squared, TensorView<const T,2> grad_w2_barycenters, TensorView<const T,2> w2_barycenters, DiracSet<const T,1> diracs, Affine1d<const T,1> functions, DiracSet<T,1> grad_diracs, Affine1d<T,1> grad_functions ) {
    P( grad_w2_squared.size() );
    parallel_for<PI>( 0, ASSERTED_EQUAL( diracs.nb_rows(), functions.nb_rows() ), [&]( PI r ) {
        P( r, grad_w2_squared[ r ] );
        w2_distance_backward( grad_w2_squared.row( r ), grad_w2_barycenters.row( r ), w2_barycenters.row( r ), diracs.row( r ), functions.row( r ), grad_diracs.row( r ), grad_functions.row( r ) );
    });
}


} // namespace sdot
