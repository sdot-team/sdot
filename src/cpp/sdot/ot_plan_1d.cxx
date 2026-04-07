#pragma once

#include "support/IntermediateScalarType.h"
#include "support/common_macros.h"
#include "support/parallel_for.h"
#include "ot_plan_1d.h"
#include "support/P.h"
#include <numeric>

#ifdef __CUDACC__
#include <thrust/extrema.h>
#endif

namespace sdot {

template<class T>
void ot_plan_1d_forward( TensorView<const T,2,Cpu> dirac_xs_, TensorView<const T,1,Cpu> dirac_ws, auto &&primitive, TensorView<T,0,Cpu> distance, TensorView<T,2,Cpu> barycenters, TensorView<T,1,Cpu> potentials, TensorView<T,2,Cpu> cuts ) {
    using TF = typename IntermediateScalarType<T,Cpu>::type;
    using namespace std;

    const auto dirac_xs = dirac_xs_.squeeze( 1 );
    const PI nb_diracs = dirac_xs.size( 0 );
    if ( nb_diracs == 0 )
        return;

    // ws size. TODO: avoid the accumulate
    if ( dirac_ws.size() == 0 ) {
        T value = 1;
        TensorView<const T,1,Cpu> new_dirac_ws( &value, { nb_diracs }, { 0 } );
        return ot_plan_1d_forward( dirac_xs_, new_dirac_ws, FORWARD( primitive ), distance, barycenters, potentials, cuts );
    }
    ASSERT( dirac_ws.size() == nb_diracs );

    // arg sort for dirac_indices
    vector<PI> dirac_indices( nb_diracs );
    iota( dirac_indices.begin(), dirac_indices.end(), PI( 0 ) );
    sort( dirac_indices.begin(), dirac_indices.end(), [&]( PI index_a, PI index_b ) {
        return dirac_xs[ index_a ] < dirac_xs[ index_b ];
    } );

    // diracs_mass
    const TF diracs_mass = accumulate( dirac_ws.begin(), dirac_ws.end(), TF( 0 ) );
    if ( diracs_mass == 0 ) throw std::runtime_error( "mass of the diracs is null" );
    const TF dirac_scale = 1 / diracs_mass;

    //
    auto piece = primitive.first_cursor();
    TF prev_dirac_x = 42;
    TF potential = 0;
    TF prev_x0 = piece.x0;
    TF w2 = 0;
    for( PI i = 0; i < nb_diracs; ++i ) {
        const PI dirac_index = dirac_indices[ i ];
        const TF dirac_x = dirac_xs[ dirac_index ];

        const TF normalized_dirac_mass = dirac_scale * static_cast<TF>( dirac_ws[ dirac_index ] );
        if ( normalized_dirac_mass <= 0.0 )
            throw std::runtime_error( "dirac_mass must be strictly positive" );

        // Update potential using jumping relation at last_ti
        if ( ! potentials.empty() ) {
            if ( i > 0 )
                 potential += std::pow( dirac_x - prev_x0, 2 ) - std::pow( prev_dirac_x - prev_x0, 2 );
            potentials[ dirac_index ] = static_cast<T>( potential );
        }

        TF moment = 0;
        primitive.take_some_mass( piece, normalized_dirac_mass, [&]( const auto &part ) {
            w2 += part.w2_dist( dirac_x );
            moment += part.moment();
        } );

        if ( ! barycenters.empty() )
            barycenters( dirac_index, 0 ) = static_cast<T>( moment / normalized_dirac_mass );

        if ( ! cuts.empty() ) {
            cuts( dirac_index, 0 ) = prev_x0;
            cuts( dirac_index, 1 ) = piece.x0;
        }

        prev_x0 = piece.x0;
        prev_dirac_x = dirac_x;
    }

    if ( ! distance.empty() )
        distance() = static_cast<T>( w2 );
}

T_T void ot_plan_1d_forward( TensorView<const T,3,Cpu> dirac_xs, TensorView<const T,2,Cpu> dirac_ws, auto &&primitive, TensorView<T,1,Cpu> distance, TensorView<T,3,Cpu> barycenters, TensorView<T,2,Cpu> potentials, TensorView<T,3,Cpu> cuts ) {
    parallel_for<PI>( 0, dirac_xs.size( 0 ), [&]( PI r ) {
        ot_plan_1d_forward( dirac_xs.row( r ), dirac_ws.row( r ), primitive.row( r ), distance.row( r ), barycenters.row( r ), potentials.row( r ), cuts.row( r ) );
    } );
}

// ---------------------------------------------------------------------------------------------- backward ----------------------------------------------------------------------------------------------
T_T void ot_plan_1d_backward( TensorView<const T,2,Cpu> dirac_xs_, TensorView<const T,1,Cpu> dirac_ws, auto &&primitive, TensorView<const T,0,Cpu> distance, TensorView<const T,2,Cpu> barycenters, TensorView<const T,1,Cpu> potentials, TensorView<const T,2,Cpu> cuts, TensorView<const T,0,Cpu> grad_distance, TensorView<const T,2,Cpu> grad_barycenters, TensorView<const T,1,Cpu> grad_potentials, TensorView<const T,2,Cpu> grad_cuts, TensorView<T,2,Cpu> grad_dirac_xs, TensorView<T,1,Cpu> grad_dirac_ws, TensorView<T,1,Cpu> grad_g_values, TensorView<T,2,Cpu> grad_g_bounds, TensorView<T,1,Cpu> grad_g_knots ) {
    using TF = typename IntermediateScalarType<T,Cpu>::type;
    using namespace std;

    const auto dirac_xs = dirac_xs_.squeeze( 1 );
    const PI nb_diracs = dirac_xs.size( 0 );
    if ( nb_diracs == 0 )
        return;

    // ws size
    if ( dirac_ws.size() == 0 ) {
        T value = 1;
        TensorView<const T,1,Cpu> new_dirac_ws( &value, { nb_diracs }, { 0 } );
        return ot_plan_1d_backward( dirac_xs_, new_dirac_ws, FORWARD( primitive ), distance, barycenters, potentials, cuts, grad_distance, grad_barycenters, grad_potentials, grad_cuts, grad_dirac_xs, grad_dirac_ws, grad_g_values, grad_g_bounds, grad_g_knots );
    }
    ASSERT( dirac_ws.size() == nb_diracs );

    // arg sort for dirac_indices
    vector<PI> dirac_indices( nb_diracs );
    iota( dirac_indices.begin(), dirac_indices.end(), PI( 0 ) );
    sort( dirac_indices.begin(), dirac_indices.end(), [&]( PI index_a, PI index_b ) {
        return dirac_xs[ index_a ] < dirac_xs[ index_b ];
    } );

    // diracs_mass
    const TF diracs_mass = accumulate( dirac_ws.begin(), dirac_ws.end(), TF( 0 ) );
    const TF dirac_scale = 1 / diracs_mass;

    // grad_distance
    if ( const TF g_dist = grad_distance.empty() ? 0 : static_cast<TF>( grad_distance() ) ) {
        // grad_dirac_xs
        if ( ! grad_dirac_xs.empty() ) {
            for( PI dirac_i = 0; dirac_i < nb_diracs; ++dirac_i ) {
                const TF dirac_w = dirac_scale * static_cast<TF>( dirac_ws[ dirac_i ] );
                grad_dirac_xs( dirac_i, 0 ) = 2 * g_dist * dirac_w * ( dirac_xs[ dirac_i ] - barycenters( dirac_i, 0 ) );
            }
        }

        // grad_dirac_ws
        if ( ! grad_dirac_ws.empty() ) {
            TF w_total = 0, running_acc = 0;
            for( PI i = 0; i < nb_diracs; ++i ) {
                const PI di = dirac_indices[ i ];
                const TF wi = dirac_ws[ di ];
                const TF Li = pow( cuts( di, 0 ) - dirac_xs[ di ], 2 );
                const TF Ri = pow( cuts( di, 1 ) - dirac_xs[ di ], 2 );
                w_total += ( diracs_mass - running_acc - wi ) * Ri;
                w_total -= ( diracs_mass - running_acc ) * Li;
                running_acc += wi;
            }

            const TF base_coeff = g_dist / pow( diracs_mass, 2 );
            TF cum_L = 0, cum_R = 0;
            for( PI i = 0; i < nb_diracs; ++i ) {
                const PI di = dirac_indices[ i ];
                cum_L += pow( cuts( di, 0 ) - dirac_xs[ di ], 2 );
                grad_dirac_ws( di ) = base_coeff * ( w_total + diracs_mass * ( cum_L - cum_R ) );
                cum_R += pow( cuts( di, 1 ) - dirac_xs[ di ], 2 );
            }
        }

        // grad_g_values
        if ( ! grad_g_values.empty() ) {
            for( PI i = 0; i < grad_g_values.size(); ++i )
                grad_g_values[ i ] = 0;

            // direct term: iterate like the forward pass
            auto piece = primitive.first_cursor();
            for( PI i = 0; i < nb_diracs; ++i ) {
                const PI di = dirac_indices[ i ];
                const TF dirac_x = dirac_xs[ di ];
                const TF potential_i = potentials.empty() ? TF( 0 ) : static_cast<TF>( potentials[ di ] );
                const TF normalized_dirac_mass = dirac_scale * static_cast<TF>( dirac_ws[ di ] );

                primitive.take_some_mass( piece, normalized_dirac_mass, [&]( const auto &piece ) {
                    primitive.accumulate_gradients_dist( piece, g_dist, dirac_x, potential_i, grad_g_values );
                } );
            }

            // weighted_pot = sum_i potential_i * w_i/W
            TF weighted_pot = 0;
            if ( ! potentials.empty() ) {
                for( PI i = 0; i < nb_diracs; ++i ) {
                    const PI di = dirac_indices[ i ];
                    weighted_pot += static_cast<TF>( potentials[ di ] ) * dirac_scale * static_cast<TF>( dirac_ws[ di ] );
                }
            }

            // normalization correction
            primitive.apply_normalization_correction( g_dist, static_cast<TF>( distance() ) - weighted_pot, grad_g_values );
        }

        // // grad_g_values
        // if ( ! grad_g_values.empty() ) {
        //     for( PI i = 0; i < grad_g_values.size(); ++i )
        //         grad_g_values[ i ] = 0;

        //     // direct term: iterate like the forward pass
        //     auto piece = primitive.first_piece();
        //     for( PI i = 0; i < nb_diracs; ++i ) {
        //         const PI di = dirac_indices[ i ];
        //         const TF dirac_x = dirac_xs[ di ];
        //         const TF potential_i = potentials.empty() ? TF( 0 ) : static_cast<TF>( potentials[ di ] );
        //         const TF normalized_dirac_mass = dirac_scale * static_cast<TF>( dirac_ws[ di ] );

        //         piece.take_some_mass( normalized_dirac_mass, [&]( const auto &part ) {
        //             part.accumulate_gradients_dist( g_dist, dirac_x, potential_i, primitive.coeff_values, grad_g_values );
        //         } );
        //     }

        //     // weighted_pot = sum_i potential_i * w_i/W
        //     TF weighted_pot = 0;
        //     if ( ! potentials.empty() ) {
        //         for( PI i = 0; i < nb_diracs; ++i ) {
        //             const PI di = dirac_indices[ i ];
        //             weighted_pot += static_cast<TF>( potentials[ di ] ) * dirac_scale * static_cast<TF>( dirac_ws[ di ] );
        //         }
        //     }

        //     // normalization correction
        //     primitive.apply_normalization_correction( g_dist, static_cast<TF>( distance() ) - weighted_pot, grad_g_values );
        // }
    }

}

T_T void ot_plan_1d_backward( TensorView<const T,3,Cpu> dirac_xs, TensorView<const T,2,Cpu> dirac_ws, auto &&primitive, TensorView<const T,1,Cpu> distance, TensorView<const T,3,Cpu> barycenters, TensorView<const T,2,Cpu> potentials, TensorView<const T,3,Cpu> cuts, TensorView<const T,1,Cpu> grad_distance, TensorView<const T,3,Cpu> grad_barycenters, TensorView<const T,2,Cpu> grad_potentials, TensorView<const T,3,Cpu> grad_cuts, TensorView<T,3,Cpu> grad_dirac_xs, TensorView<T,2,Cpu> grad_dirac_ws, TensorView<T,2,Cpu> grad_g_values, TensorView<T,3,Cpu> grad_g_bounds, TensorView<T,2,Cpu> grad_g_knots ) {
    parallel_for<PI>( 0, dirac_xs.size( 0 ), [&]( PI r ) {
        ot_plan_1d_backward(
            dirac_xs.row( r ), dirac_ws.row( r ), primitive.row( r ), distance.row( r ), barycenters.row( r ), potentials.row( r ), cuts.row( r ),
            grad_distance.row( r ), grad_barycenters.row( r ), grad_potentials.row( r ), grad_cuts.row( r ),
            grad_dirac_xs.row( r ), grad_dirac_ws.row( r ), grad_g_values.row( r ), grad_g_bounds.row( r ), grad_g_knots.row( r )
        );
    } );
}

} // namespace sdot
