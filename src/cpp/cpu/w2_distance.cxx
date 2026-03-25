#pragma once

#include "../support/parallel_for.h"
#include "../support/P.h"
#include "w2_distance.h"

namespace sdot {

T_T void w2_distance( DiracSet<const T,Cpu> diracs, Affine1d<const T,Cpu> points, TensorView<T,0,Cpu> distance, TensorView<T,1,Cpu> barycenters, TensorView<T,1,Cpu> potentials, TensorView<T,2,Cpu> cuts ) {
    using TF = typename IntermediateScalarType<T,Cpu>::type;

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
            cuts( dirac_index, 0 ) = prev_cut_position;
            cuts( dirac_index, 1 ) = cut_position;
        }

        prev_cut_position = cut_position;
        prev_dirac_x = dirac_x;
    }

    if ( ! distance.empty() )
       distance() = static_cast<T>( w2 );
}

T_T void w2_distance( BatchOfDiracSet<const T,Cpu> diracs, BatchOfAffine1d<const T,Cpu> functions, TensorView<T,1,Cpu> distance, TensorView<T,2,Cpu> barycenters, TensorView<T,2,Cpu> potentials, TensorView<T,3,Cpu> cuts ) {
    parallel_for<PI>( 0, ASSERTED_EQUAL( diracs.nb_rows(), functions.nb_rows() ), [&]( PI r ) {
        w2_distance( diracs.row( r ), functions.row( r ), distance.row( r ), barycenters.row( r ), potentials.row( r ), cuts.row( r ) );
    });
}

T_T void w2_distance_backward( TensorView<const T,0,Cpu> grad_distance, TensorView<const T,1,Cpu> grad_barycenters, TensorView<const T,1,Cpu> barycenters, TensorView<const T,1,Cpu> potentials, TensorView<const T,2,Cpu> cuts, DiracSet<const T,Cpu> diracs, Affine1d<const T,Cpu> points, DiracSet<T,Cpu> grad_diracs, Affine1d<T,Cpu> grad_functions ) {
    using TF = typename IntermediateScalarType<T,Cpu>::type;
    using namespace std;

    const vector<PI> dirac_indices = diracs.arg_sort();
    const TF diracs_mass = diracs.mass();
    const TF points_mass = points.mass();
    const TF dirac_scale = 1.0 / diracs_mass;
    const TF point_scale = 1.0 / points_mass;

    const TF g_dist = grad_distance.empty() ? 0 : static_cast<TF>( grad_distance() );

    // Mirror forward state variables
    PieceOfAffine1d<TF> current_piece = points.get_first_piece( point_scale );
    TF prev_cut_position = current_piece.x0;
    // bary_grad_potential: "barycenter gradient potential", mirrors `potential` in the forward.
    // Satisfies the same jumping relation with grad_bary_per_mass_i = grad_barycenters[d] / m_i:
    //   bary_grad_potential_i = bary_grad_potential_{i-1} + T_{i-1} * (grad_bary_per_mass_i - grad_bary_per_mass_{i-1})
    // (the forward has: potential_i = potential_{i-1} + (dirac_x_i - T)^2 - (dirac_x_{i-1} - T)^2)
    TF bary_grad_potential = 0;
    TF prev_grad_bary_per_mass = 0;  // mirrors prev_dirac_x in forward

    // Normalization accumulators, mirror w2 and moment in forward
    TF avg_potential           = 0;  // sum_i m_i * psi_i
    TF avg_bary_grad_potential = 0;  // sum_i m_i * bary_grad_potential_i
    TF w2_total                = 0;  // mirrors w2 in forward
    TF bary_moment_total       = 0;  // sum_i grad_bary_per_mass_i * moment_i

    vector<TF> bary_grad_potentials( diracs.nb_diracs(), 0 );  // per dirac in original order, for ws grad

    // Per-piece left/right shape-function integrals, needed for grad_xs
    const PI nb_pts = points.nb_points();
    vector<TF> accum_gl( nb_pts ); // grad_functions.xs.empty() ? 0 : nb_pts, 0 );
    vector<TF> accum_gr( nb_pts ); // grad_functions.xs.empty() ? 0 : nb_pts, 0 );

    // Boundary dirac data for Leibniz boundary terms at xs[0] and xs[nb_pts-1]
    TF first_dirac_x = 0, first_grad_bary_per_mass = 0;  // psi=0, bary_grad_potential=0 at i=0
    TF last_dirac_x = 0, last_psi = 0, last_grad_bary_per_mass = 0, last_bary_grad_potential = 0;

    for( PI i = 0; i < diracs.nb_diracs(); ++i ) {
        const PI dirac_index = dirac_indices[ i ];

        const TF normalized_dirac_mass  = dirac_scale * static_cast<TF>( diracs.ws[ dirac_index ] );
        const TF dirac_x                = static_cast<TF>( diracs.xs[ dirac_index ] );
        const TF psi_i                  = static_cast<TF>( potentials[ dirac_index ] );  // saved from forward (mirrors `potentials[d] = potential`)
        const TF barycenter_i           = static_cast<TF>( barycenters[ dirac_index ] );
        const TF grad_bary_per_mass     = static_cast<TF>( grad_barycenters[ dirac_index ] ) / normalized_dirac_mass;

        // Mirror of potential update using jumping relation at prev_cut_position:
        //   bary_grad_potential_i = bary_grad_potential_{i-1} + T_{i-1} * (grad_bary_per_mass_i - grad_bary_per_mass_{i-1})
        if ( i > 0 )
            bary_grad_potential += prev_cut_position * ( grad_bary_per_mass - prev_grad_bary_per_mass );
        bary_grad_potentials[ dirac_index ] = bary_grad_potential;

        // Save first/last dirac data for Leibniz boundary terms
        if ( i == 0 ) {
            first_dirac_x = dirac_x;
            first_grad_bary_per_mass = grad_bary_per_mass;
        }
        if ( i == diracs.nb_diracs() - 1 ) {
            last_dirac_x = dirac_x;
            last_psi = psi_i;
            last_grad_bary_per_mass = grad_bary_per_mass;
            last_bary_grad_potential = bary_grad_potential;
        }

        // Accumulate normalization terms (mirrors w2 and moment accumulation in forward)
        avg_potential           += normalized_dirac_mass * psi_i;
        avg_bary_grad_potential += normalized_dirac_mass * bary_grad_potential;

        // Backward of: w2 += piece.w2_dist(dirac_x)  →  d(w2)/d(dirac_x) = 2 * m_i * (dirac_x - barycenter_i)
        if ( ! grad_diracs.xs.empty() )
            grad_diracs.xs[ dirac_index ] = static_cast<T>( 2 * g_dist * normalized_dirac_mass * ( dirac_x - barycenter_i ) );

        // Mirror of take_some_mass loop: traverse the same pieces as forward, accumulating gradients
        const TF cut_position = points.take_some_mass( current_piece, point_scale, normalized_dirac_mass, [&]( const PieceOfAffine1d<TF> &piece ) {
            // Backward of w2 += piece.w2_dist(dirac_x)
            w2_total += piece.w2_dist( dirac_x );
            points.accumulate_w2_grad_ys( piece, dirac_x, psi_i, g_dist * point_scale, grad_functions.ys );

            // Backward of barycenters[d] = moment / m_i
            bary_moment_total += grad_bary_per_mass * piece.moment();
            points.accumulate_linear_grad_ys( piece, grad_bary_per_mass, -bary_grad_potential, point_scale, grad_functions.ys );

            if ( ! grad_functions.xs.empty() ) {
                TF gl_w2 = 0, gr_w2 = 0;
                piece.integrate_w2_shape_functions( dirac_x, psi_i,
                    static_cast<TF>( points.xs[ piece.index - 1 ] ), static_cast<TF>( points.xs[ piece.index ] ), gl_w2, gr_w2 );
                TF gl_b = 0, gr_b = 0;
                piece.integrate_linear_shape_functions( grad_bary_per_mass, -bary_grad_potential,
                    static_cast<TF>( points.xs[ piece.index - 1 ] ), static_cast<TF>( points.xs[ piece.index ] ), gl_b, gr_b );
                accum_gl[ piece.index - 1 ] += g_dist * gl_w2 + gl_b;
                accum_gr[ piece.index - 1 ] += g_dist * gr_w2 + gr_b;
            }
        } );

        prev_cut_position = cut_position;
        prev_grad_bary_per_mass = grad_bary_per_mass;
    }

    // Normalization correction for ys: each ys[j] also affects the total mass Y = points_mass,
    // which normalizes both w2 and barycenters (mirrors the 1/Y = point_scale factor in forward)
    const TF normalization_ratio = ( g_dist * ( w2_total - avg_potential ) + ( bary_moment_total - avg_bary_grad_potential ) ) * point_scale;
    points.get_grad_ys( normalization_ratio, grad_functions.ys );

    // Gradients w.r.t. function node positions xs
    // dL/d xs[j] = point_scale * ( -(ys[j]-ys[j-1])/h_j * accum_gr[j-1]   // piece j  (right boundary)
    //                              + (ys[j]-ys[j+1])/h_{j+1} * accum_gl[j] // piece j+1 (left boundary) )
    //            - normalization_ratio * dY/d xs[j]
    // where dY/d xs[j] = (ys[j-1]+ys[j])/2 - (ys[j]+ys[j+1])/2
    if ( ! grad_functions.xs.empty() ) {
        const PI n = nb_pts - 1;
        for ( PI j = 0; j <= n; ++j ) {
            const TF ys_j = static_cast<TF>( points.ys[ j ] );
            TF g = 0;

            // Direct: piece j has xs[j] as right boundary
            if ( j > 0 ) {
                const TF h_j    = static_cast<TF>( points.xs[ j ] ) - static_cast<TF>( points.xs[ j - 1 ] );
                const TF ys_jm1 = static_cast<TF>( points.ys[ j - 1 ] );
                g += -( ys_j - ys_jm1 ) / h_j * accum_gr[ j - 1 ] * point_scale;
            }

            // Direct: piece j+1 has xs[j] as left boundary
            if ( j < n ) {
                const TF h_jp1  = static_cast<TF>( points.xs[ j + 1 ] ) - static_cast<TF>( points.xs[ j ] );
                const TF ys_jp1 = static_cast<TF>( points.ys[ j + 1 ] );
                g += ( ys_j - ys_jp1 ) / h_jp1 * accum_gl[ j ] * point_scale;
            }

            // Normalization correction: -normalization_ratio * dY/d xs[j]
            TF dY_dx = 0;
            if ( j > 0 ) dY_dx += ( static_cast<TF>( points.ys[ j - 1 ] ) + ys_j ) / 2;
            if ( j < n ) dY_dx -= ( ys_j + static_cast<TF>( points.ys[ j + 1 ] ) ) / 2;
            g -= normalization_ratio * dY_dx;

            grad_functions.xs[ j ] += static_cast<T>( g );
        }

        // Leibniz boundary terms: moving xs[0] or xs[n] shifts the domain boundary.
        // For interior xs[j], contributions from piece j and j+1 cancel; for endpoints they don't.
        // xs[0]: -point_scale * G_pure(xs[0]) * ys[0]  (psi_0=0, bary_grad_pot_0=0)
        // xs[n]: +point_scale * G_pure(xs[n]) * ys[n]
        if ( diracs.nb_diracs() > 0 ) {
            const TF x0 = static_cast<TF>( points.xs[ 0 ] );
            const TF G0 = g_dist * std::pow( x0 - first_dirac_x, 2 ) + first_grad_bary_per_mass * x0;
            grad_functions.xs[ 0 ] -= static_cast<T>( point_scale * G0 * static_cast<TF>( points.ys[ 0 ] ) );

            const TF xn = static_cast<TF>( points.xs[ nb_pts - 1 ] );
            const TF Gn = g_dist * ( std::pow( xn - last_dirac_x, 2 ) - last_psi ) + last_grad_bary_per_mass * xn - last_bary_grad_potential;
            grad_functions.xs[ nb_pts - 1 ] += static_cast<T>( point_scale * Gn * static_cast<TF>( points.ys[ nb_pts - 1 ] ) );
        }
    }

    // Backward of: normalized_dirac_mass = dirac_scale * ws[d]
    // mirrors the dirac_scale normalization (analog of point_scale for the dirac side)
    if ( ! grad_diracs.ws.empty() ) {
        for( PI i = 0; i < diracs.nb_diracs(); ++i ) {
            const TF psi_i               = static_cast<TF>( potentials[ i ] );
            const TF bary_grad_potential_i = bary_grad_potentials[ i ];
            const TF grad_bary_i         = static_cast<TF>( grad_barycenters[ i ] );
            const TF barycenter_i        = static_cast<TF>( barycenters[ i ] );
            const TF wi                  = static_cast<TF>( diracs.ws[ i ] );

            // dL/dw_i = ( g_dist * (psi_i - avg_potential) + (bary_grad_potential_i - avg_bary_grad_potential + bary_moment_total) ) * dirac_scale - grad_bary_i * barycenter_i / wi
            TF g_wi = ( g_dist * ( psi_i - avg_potential ) + ( bary_grad_potential_i - avg_bary_grad_potential + bary_moment_total ) ) * dirac_scale;
            g_wi -= grad_bary_i * barycenter_i / wi;
            grad_diracs.ws[ i ] = static_cast<T>( g_wi );
        }
    }
}

/// Gradients of Wasserstein 2 distance
T_T void w2_distance_backward( TensorView<const T,1,Cpu> grad_distance, TensorView<const T,2,Cpu> grad_barycenters, TensorView<const T,2,Cpu> barycenters, TensorView<const T,2,Cpu> potentials, TensorView<const T,3,Cpu> cuts, BatchOfDiracSet<const T,Cpu> diracs, BatchOfAffine1d<const T,Cpu> functions, BatchOfDiracSet<T,Cpu> grad_diracs, BatchOfAffine1d<T,Cpu> grad_functions ) {
    parallel_for<PI>( 0, ASSERTED_EQUAL( diracs.nb_rows(), functions.nb_rows() ), [&]( PI r ) {
        w2_distance_backward( grad_distance.row( r ), grad_barycenters.row( r ), barycenters.row( r ), potentials.row( r ), cuts.row( r ), diracs.row( r ), functions.row( r ), grad_diracs.row( r ), grad_functions.row( r ) );
    });
}


// -------------------------------------- CUDA  --------------------------------------
#ifdef __CUDACC__
T_T void w2_distance( BatchOfDiracSet<const T,Cuda> diracs, BatchOfAffine1d<const T,Cuda> functions, TensorView<T,1,Cuda> distance, TensorView<T,2,Cuda> barycenters, TensorView<T,2,Cuda> potentials, TensorView<T,3,Cuda> cuts ) {
    using TF = typename IntermediateScalarType<T,Cuda>::type;

    // const std::vector<PI> dirac_indices = diracs.arg_sort();
    auto diracs_masses = diracs.masses();
    P( diracs_masses );
    // const TF points_mass = points.mass();

    // if ( diracs_mass == 0 ) throw std::runtime_error( "mass of the diracs is null" );
    // if ( points_mass == 0 ) throw std::runtime_error( "mass of the points is null" );
    // const TF dirac_scale = 1 / diracs_mass;
    // const TF point_scale = 1 / points_mass;

    // PieceOfAffine1d<TF> current_piece = points.get_first_piece( point_scale );
    // TF prev_cut_position = current_piece.x0;
    // TF prev_dirac_x = 42;
    // TF potential = 0;
    // TF w2 = 0;
    // for( PI i = 0; i < diracs.nb_diracs(); ++i ) {
    //     const PI dirac_index = dirac_indices[ i ];

    //     const TF normalized_dirac_mass = dirac_scale * static_cast<TF>( diracs.ws[ dirac_index ] );
    //     const TF dirac_x = diracs.xs[ dirac_index ];
    //     if ( normalized_dirac_mass <= 0.0 )
    //         throw std::runtime_error( "dirac_mass must be strictly positive" );

    //     // Update potential using jumping relation at last_ti
    //     if ( ! potentials.empty() ) {
    //         if ( i > 0 )
    //              potential += std::pow( dirac_x - prev_cut_position, 2 ) - std::pow( prev_dirac_x - prev_cut_position, 2 );
    //         potentials[ dirac_index ] = static_cast<T>( potential );
    //     }

    //     TF moment = 0;
    //     const TF cut_position = points.take_some_mass( current_piece, point_scale, normalized_dirac_mass, [&]( const PieceOfAffine1d<TF> &piece ) {
    //         w2 += piece.w2_dist( dirac_x );
    //         moment += piece.moment();
    //     } );

    //     if ( ! barycenters.empty() )
    //         barycenters[ dirac_index ] = static_cast<T>( moment / normalized_dirac_mass );

    //     if ( ! cuts.empty() ) {
    //         cuts( dirac_index, 0 ) = prev_cut_position;
    //         cuts( dirac_index, 1 ) = cut_position;
    //     }

    //     prev_cut_position = cut_position;
    //     prev_dirac_x = dirac_x;
    // }

    // if ( ! distance.empty() )
    //    distance() = static_cast<T>( w2 );
    ASSERT( 0 );
}

T_T void w2_distance_backward( TensorView<const T,1,Cuda> grad_distance, TensorView<const T,2,Cuda> grad_barycenters, TensorView<const T,2,Cuda> barycenters, TensorView<const T,2,Cuda> potentials, TensorView<const T,3,Cuda> cuts, BatchOfDiracSet<const T,Cuda> diracs, BatchOfAffine1d<const T,Cuda> functions, BatchOfDiracSet<T,Cuda> grad_diracs, BatchOfAffine1d<T,Cuda> grad_functions ) {
    ASSERT( 0 );
}
#endif

} // namespace sdot
