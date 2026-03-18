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
    const TF dirac_scale = 1 / diracs_mass;
    const TF point_scale = 1 / points_mass;

    PieceOfAffine1d<TF> current_piece = points.get_first_piece( point_scale );
    TF prev_cut_position = current_piece.x0;
    TF prev_dirac_x = 42;
    TF potential = 0;
    TF w2 = 0;
    for( PI i = 0; i < diracs.nb_diracs(); ++i ) {
        const PI dirac_index = dirac_indices[ i ];

        const TF normalized_dirac_mass = dirac_scale * static_cast<TF>( diracs.ws[ dirac_index ] );
        const TF dirac_x = diracs.xs[ dirac_index ];
        if ( normalized_dirac_mass <= 0.0 )
            throw std::runtime_error( "dirac_mass must be strictly positive" );

        // Update potential using jumping relation at last_ti
        if ( ! potentials.empty() ) {
            if ( i > 0 )
                 potential += std::pow( dirac_x - prev_cut_position, 2 ) - std::pow( prev_dirac_x - prev_cut_position, 2 );
            potentials[ dirac_index ] = static_cast<T>( potential );
        }

        TF moment = 0;
        const TF cut_position = points.take_some_mass( current_piece, point_scale, normalized_dirac_mass, [&]( const PieceOfAffine1d<TF> &piece ) {
            w2 += piece.w2_dist( dirac_x );
            moment += piece.moment();
        } );

        if ( ! barycenters.empty() )
            barycenters[ dirac_index ] = static_cast<T>( moment / normalized_dirac_mass );

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
    const TF dirac_scale = 1.0 / diracs_mass;
    const TF point_scale = 1.0 / points_mass;

    const TF g_dist = grad_distance.empty() ? 0 : static_cast<TF>( grad_distance[ 0 ] );

    // 1. Barycenter potentials (theta_i)
    // theta_{i+1} - theta_i = T_i * (H_{i+1} - H_i) where H_i = gb_i / m_i
    vector<TF> theta( diracs.nb_diracs(), 0 );
    TF current_theta = 0;
    for( PI i = 1; i < diracs.nb_diracs(); ++i ) {
        const PI d0 = dirac_indices[ i - 1 ];
        const PI d1 = dirac_indices[ i - 0 ];
        const TF Ti = static_cast<TF>( cuts[ 2 * d1 + 0 ] );
        const TF m0 = static_cast<TF>( diracs.ws[ d0 ] ) * dirac_scale;
        const TF m1 = static_cast<TF>( diracs.ws[ d1 ] ) * dirac_scale;
        const TF gb0 = static_cast<TF>( grad_barycenters[ d0 ] );
        const TF gb1 = static_cast<TF>( grad_barycenters[ d1 ] );
        current_theta += Ti * ( gb1 / m1 - gb0 / m0 );
        theta[ d1 ] = current_theta;
    }

    // 2. Compute averages M_psi = sum(m_i * psi_i) and M_theta = sum(m_i * theta_i)
    TF M_psi = 0;
    TF M_theta = 0;
    for( PI i = 0; i < diracs.nb_diracs(); ++i ) {
        const TF mi = static_cast<TF>( diracs.ws[ i ] ) * dirac_scale;
        M_psi += mi * static_cast<TF>( potentials[ i ] );
        M_theta += mi * theta[ i ];
    }

    // 3. Main loop (forward style)
    PieceOfAffine1d<TF> current_piece = points.get_first_piece( point_scale );
    TF w2_total = 0;   // sum_i integral_piece (xi - t)^2 y(t) dt
    TF bary_total = 0; // sum_i integral_piece (Hi * t) y(t) dt
    for( PI i = 0; i < diracs.nb_diracs(); ++i ) {
        const PI dirac_index = dirac_indices[ i ];
        const TF mi = static_cast<TF>( diracs.ws[ dirac_index ] ) * dirac_scale;
        const TF xi = static_cast<TF>( diracs.xs[ dirac_index ] );
        const TF psi_i = static_cast<TF>( potentials[ dirac_index ] );
        const TF theta_i = theta[ dirac_index ];
        const TF gb_i = static_cast<TF>( grad_barycenters[ dirac_index ] );
        const TF Hi = gb_i / mi;
        const TF Bi = static_cast<TF>( barycenters[ dirac_index ] );

        // Gradient w.r.t dirac position
        grad_diracs.xs[ dirac_index ] = static_cast<T>( 2 * g_dist * mi * ( xi - Bi ) );

        points.take_some_mass( current_piece, point_scale, mi, [&]( const PieceOfAffine1d<TF> &piece ) {
            TF gl = 0, gr = 0;
            // W2 contribution: integral((xi - t)^2 - psi_i) * phi_j(t) dt
            piece.integrate_w2_shape_functions( xi, psi_i, points.xs[ piece.index - 1 ], points.xs[ piece.index ], gl, gr );
            grad_functions.ys[ piece.index - 1 ] += static_cast<T>( g_dist * gl * point_scale );
            grad_functions.ys[ piece.index ]     += static_cast<T>( g_dist * gr * point_scale );
            w2_total += piece.w2_dist( xi );

            // Barycenter contribution: integral(Hi * t - theta_i) * phi_j(t) dt
            TF gl_b = 0, gr_b = 0;
            piece.integrate_linear_shape_functions( Hi, -theta_i, points.xs[ piece.index - 1 ], points.xs[ piece.index ], gl_b, gr_b );
            grad_functions.ys[ piece.index - 1 ] += static_cast<T>( gl_b * point_scale );
            grad_functions.ys[ piece.index ]     += static_cast<T>( gr_b * point_scale );
            bary_total += Hi * piece.moment();
        } );
    }

    // 4. Normalization corrections for Ys
    // ratio = sum_i integral_piece [ g_dist * ((xi - t)^2 - psi_i) + (Hi * t - theta_i) ] y(t) / Y^2 dt
    //       = (g_dist * (w2_total - M_psi * points_mass/point_scale) + (bary_total - M_theta * points_mass/point_scale)) / Y^2 ?
    // Actually, w2_total and bary_total are integrals over y(t) dt.
    // Normalized W2 = w2_total / Y. Normalized Phi = bary_total / Y.
    // So the ratio is ( g_dist * ( W2 - M_psi ) + ( Phi - M_theta ) ) * point_scale.
    const TF W2 = w2_total;
    const TF Phi = bary_total;
    const TF ratio = ( g_dist * ( W2 - M_psi ) + ( Phi - M_theta ) ) * point_scale;

    points.get_grad_ys( ratio, grad_functions.ys );

    // 5. Gradients w.r.t dirac weights
    if ( ! grad_diracs.ws.empty() ) {
        for( PI i = 0; i < diracs.nb_diracs(); ++i ) {
            const TF psi_i = static_cast<TF>( potentials[ i ] );
            const TF theta_i = theta[ i ];
            const TF gb_i = static_cast<TF>( grad_barycenters[ i ] );
            const TF Bi = static_cast<TF>( barycenters[ i ] );
            const TF wi = static_cast<TF>( diracs.ws[ i ] );

            // dL/dw_i = ( g_dist * (psi_i - M_psi) + (theta_i - M_theta + Phi) ) / W - gb_i * Bi / wi
            TF g_wi = ( g_dist * ( psi_i - M_psi ) + ( theta_i - M_theta + Phi ) ) * dirac_scale;
            g_wi -= gb_i * Bi / wi;
            grad_diracs.ws[ i ] = static_cast<T>( g_wi );
        }
    }
}

/// Gradients of Wasserstein 2 distance
T_T void w2_distance_backward( TensorView<const T,1> grad_distance, TensorView<const T,2> grad_barycenters, TensorView<const T,2> barycenters, TensorView<const T,2> potentials, TensorView<const T,2> cuts, DiracSet<const T,1> diracs, Affine1d<const T,1> functions, DiracSet<T,1> grad_diracs, Affine1d<T,1> grad_functions ) {
    parallel_for<PI>( 0, ASSERTED_EQUAL( diracs.nb_rows(), functions.nb_rows() ), [&]( PI r ) {
        w2_distance_backward( grad_distance.row( r ), grad_barycenters.row( r ), barycenters.row( r ), potentials.row( r ), cuts.row( r ), diracs.row( r ), functions.row( r ), grad_diracs.row( r ), grad_functions.row( r ) );
    });
}


} // namespace sdot
