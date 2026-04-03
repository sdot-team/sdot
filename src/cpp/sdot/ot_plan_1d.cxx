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
    auto piece = primitive.first_piece();
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
        piece.take_some_mass( normalized_dirac_mass, [&]( const auto &part ) {
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
T_T void ot_plan_1d_backward( TensorView<const T,2,Cpu> dirac_xs_, TensorView<const T,1,Cpu> dirac_ws, auto &&primitive, TensorView<const T,0,Cpu> distance, TensorView<const T,2,Cpu> barycenters, TensorView<const T,1,Cpu> potentials, TensorView<const T,2,Cpu> cuts, TensorView<const T,0,Cpu> grad_distance, TensorView<const T,2,Cpu> grad_barycenters, TensorView<const T,1,Cpu> grad_potentials, TensorView<const T,2,Cpu> grad_cuts, TensorView<T,2,Cpu> grad_dirac_xs, TensorView<T,1,Cpu> grad_dirac_ws, TensorView<T,1,Cpu> grad_g_values ) {
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
        return ot_plan_1d_backward( dirac_xs_, new_dirac_ws, FORWARD( primitive ), distance, barycenters, potentials, cuts, grad_distance, grad_barycenters, grad_potentials, grad_cuts, grad_dirac_xs, grad_dirac_ws, grad_g_values );
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
            if ( nb_diracs >= 2 ) {
                //
                vector<TF> acc( nb_diracs + 1 );
                acc[ 0 ] = 0;
                for( PI i = 0; i < nb_diracs; ++i )
                    acc[ i + 1 ] = acc[ i ] + dirac_ws[ i ];
                const TF S = acc.back();

                for( PI z = 0; z < nb_diracs; ++z ) {
                    const PI dirac_z = dirac_indices[ z ];
                    TF g = 0;

                    for( PI i = 0; i + 1 <= dirac_z; ++i ) {
                        const PI dirac_i = dirac_indices[ i ];
                        g -= acc[ i + 1 ] * pow( cuts( dirac_i, 1 ) - dirac_xs[ dirac_i ], 2 );
                    }
                    for( PI i = dirac_z; i < nb_diracs; ++i ) {
                        const PI dirac_i = dirac_indices[ i ];
                        g += ( S - acc[ i + 1 ] ) * pow( cuts( dirac_i, 1 ) - dirac_xs[ dirac_i ], 2 );
                    }

                    for( PI i = 0; i <= dirac_z; ++i ) {
                        const PI dirac_i = dirac_indices[ i ];
                        g += acc[ i ] * pow( cuts( dirac_i, 0 ) - dirac_xs[ dirac_i ], 2 );
                    }
                    for( PI i = dirac_z + 1; i < nb_diracs; ++i ) {
                        const PI dirac_i = dirac_indices[ i ];
                        g -= ( S - acc[ i ] ) * pow( cuts( dirac_i, 0 ) - dirac_xs[ dirac_i ], 2 );
                    }

                    grad_dirac_ws( dirac_z ) = g_dist / pow( S, 2 ) * g;
                }

                // normalization
                // TF dot = 0;
                // for( PI i = 0; i < nb_diracs; ++i )
                //     dot += grad_dirac_ws[ i ] * dirac_ws[ i ];
                // for( PI i = 0; i < nb_diracs; ++i )
                //     grad_dirac_ws[ i ] = dirac_scale * ( grad_dirac_ws[ i ] - dot );
            }
        }
    }



    // // ws size
    // if ( dirac_ws.size() == 0 ) {
    //     T value = 1;
    //     TensorView<const T,1,Cpu> new_dirac_ws( &value, { nb_diracs }, { 0 } );
    //     TensorView<T,1,Cpu> new_grad_dirac_ws; // empty if we don't need it
    //     return ot_plan_1d_backward( dirac_xs_, new_dirac_ws, FORWARD( primitive ), distance, barycenters, potentials, cuts, grad_distance, grad_barycenters, grad_potentials, grad_cuts, grad_dirac_xs, new_grad_dirac_ws, grad_g_values );
    // }
    // ASSERT( dirac_ws.size() == nb_diracs );

    // // arg sort for dirac_indices (same as forward)
    // vector<PI> dirac_indices( nb_diracs );
    // iota( dirac_indices.begin(), dirac_indices.end(), PI( 0 ) );
    // sort( dirac_indices.begin(), dirac_indices.end(), [&]( PI index_a, PI index_b ) {
    //     return dirac_xs[ index_a ] < dirac_xs[ index_b ];
    // });

    // // diracs_mass
    // TF diracs_mass = accumulate( dirac_ws.begin(), dirac_ws.end(), TF( 0 ) );
    // if ( diracs_mass == 0 ) throw std::runtime_error( "mass of the diracs is null" );
    // const TF dirac_scale = 1 / diracs_mass;

    // // Weights for the different components of the gradient
    // const TF g_dist = grad_distance.empty() ? 0 : static_cast<TF>( grad_distance() );

    // // Pass 1: Forward to collect info needed for cut gradients
    // vector<TF> betas( nb_diracs );
    // vector<TF> rhos_at_cuts( nb_diracs );
    // auto piece = primitive.first_piece();
    // for( PI i = 0; i < nb_diracs; ++i ) {
    //     const PI di = dirac_indices[ i ];
    //     const TF bar_w = dirac_scale * static_cast<TF>( dirac_ws[ di ] );
    //     betas[ i ] = grad_barycenters.empty() ? 0 : static_cast<TF>( grad_barycenters( di, 0 ) ) / bar_w;

    //     piece.take_some_mass( bar_w, [&]( const auto &part ) {} );
    //     if ( i < nb_diracs - 1 )
    //         rhos_at_cuts[ i ] = piece.value_at( piece.x0 );
    // }

    // // Pass 2: G_pots from the end
    // vector<TF> G_pots( nb_diracs + 1, 0 );
    // if ( ! grad_potentials.empty() ) {
    //     for( PI i = nb_diracs; i--; )
    //         G_pots[ i ] = G_pots[ i + 1 ] + static_cast<TF>( grad_potentials[ dirac_indices[ i ] ] );
    // }

    // // Pass 3: V_cuts and dL/dbar_w from the end
    // vector<TF> V_cuts_val( nb_diracs, 0 ); // piecewise constant value in interval i
    // vector<TF> dL_dbar_w( nb_diracs, 0 );
    // TF current_V_cut = 0;
    // for ( PI i = nb_diracs; i--; ) {
    //     V_cuts_val[ i ] = current_V_cut;
    //     dL_dbar_w[ i ] = - betas[ i ] * static_cast<TF>( barycenters( dirac_indices[ i ], 0 ) ) - current_V_cut;

    //     if ( i > 0 ) {
    //         const PI di = dirac_indices[ i - 1 ];
    //         const PI di_next = dirac_indices[ i ];
    //         const TF T_i = static_cast<TF>( cuts( di, 1 ) );
    //         const TF x_i = static_cast<TF>( dirac_xs[ di ] );
    //         const TF x_next = static_cast<TF>( dirac_xs[ di_next ] );
    //         const TF rho_Ti = rhos_at_cuts[ i - 1 ];

    //         TF dL_dTi = 0;
    //         dL_dTi += g_dist * ( std::pow( T_i - x_i, 2 ) - std::pow( T_i - x_next, 2 ) ) * ( rho_Ti * primitive.coeff_values );
    //         dL_dTi += ( betas[ i - 1 ] - betas[ i ] ) * T_i * ( rho_Ti * primitive.coeff_values );
    //         dL_dTi += 2 * G_pots[ i ] * ( x_next - x_i );
    //         if ( ! grad_cuts.empty() )
    //             dL_dTi += static_cast<TF>( grad_cuts( di, 1 ) ) + static_cast<TF>( grad_cuts( di_next, 0 ) );

    //         if ( rho_Ti > 0 )
    //             current_V_cut -= dL_dTi / ( rho_Ti * primitive.coeff_values );
    //     }
    // }

    // // Pass 4: Compute total objective derivative w.r.t. unnormalized rho
    // if ( ! grad_g_values.empty() || ! grad_dirac_xs.empty() || ! grad_dirac_ws.empty() ) {
    //     TF avg_V = 0;
    //     {
    //         auto piece_avg = primitive.first_piece();
    //         for ( PI i = 0; i < nb_diracs; ++i ) {
    //             const PI di = dirac_indices[ i ];
    //             const TF x_i = static_cast<TF>( dirac_xs[ di ] );
    //             const TF bar_w = dirac_scale * static_cast<TF>( dirac_ws[ di ] );
    //             const TF beta_i = betas[ i ];
    //             const TF Ci = V_cuts_val[ i ];
    //             piece_avg.take_some_mass( bar_w, [&]( const auto &part ) {
    //                 // W2_dist in part is already \int (x-xi)^2 rho(x) dx
    //                 avg_V += ( g_dist * part.w2_dist( x_i ) + beta_i * part.moment() + Ci * ( part.x1 - part.x0 ) * ( part.y0 + part.y1 ) / 2 );
    //             } );
    //         }
    //     }
    //     avg_V *= primitive.coeff_values;

    //     auto piece_final = primitive.first_piece();
    //     TF avg_dL_dbar_w = 0;
    //     for ( PI i = 0; i < nb_diracs; ++i ) {
    //         const PI di = dirac_indices[ i ];
    //         const TF bar_w = dirac_scale * static_cast<TF>( dirac_ws[ di ] );
    //         avg_dL_dbar_w += bar_w * dL_dbar_w[ i ];
    //     }

    //     for ( PI i = 0; i < nb_diracs; ++i ) {
    //         const PI di = dirac_indices[ i ];
    //         const TF x_i = static_cast<TF>( dirac_xs[ di ] );
    //         const TF bar_w = dirac_scale * static_cast<TF>( dirac_ws[ di ] );
    //         const TF beta_i = betas[ i ];
    //         const TF Ci = V_cuts_val[ i ];
    //         const TF bary_i = static_cast<TF>( barycenters( di, 0 ) );
    //         const TF T_prev = static_cast<TF>( cuts( di, 0 ) );
    //         const TF T_next = static_cast<TF>( cuts( di, 1 ) );

    //         if ( ! grad_dirac_xs.empty() ) {
    //             TF dL_dxi = 2 * g_dist * bar_w * ( x_i - bary_i );
    //             dL_dxi += static_cast<TF>( grad_potentials.empty() ? 0 : grad_potentials[ di ] ) * ( -2 * ( x_i - T_prev ) );
    //             dL_dxi += 2 * G_pots[ i + 1 ] * ( T_next - T_prev );
    //             grad_dirac_xs( di, 0 ) = static_cast<T>( dL_dxi );
    //         }

    //         if ( ! grad_dirac_ws.empty() ) {
    //             grad_dirac_ws[ di ] = static_cast<T>( dirac_scale * ( dL_dbar_w[ i ] - avg_dL_dbar_w ) );
    //         }

    //         if ( ! grad_g_values.empty() ) {
    //             piece_final.take_some_mass( bar_w, [&]( const auto &part ) {
    //                 // dL/d rho = coeff_values * (V(x) - Avg(V))
    //                 primitive.accumulate_gradients( part, g_dist * primitive.coeff_values, ( beta_i - 2 * g_dist * x_i ) * primitive.coeff_values, ( g_dist * x_i * x_i + Ci - avg_V ) * primitive.coeff_values, grad_g_values );
    //             } );
    //         }
    //     }
    // }
}

T_T void ot_plan_1d_backward( TensorView<const T,3,Cpu> dirac_xs, TensorView<const T,2,Cpu> dirac_ws, auto &&primitive, TensorView<const T,1,Cpu> distance, TensorView<const T,3,Cpu> barycenters, TensorView<const T,2,Cpu> potentials, TensorView<const T,3,Cpu> cuts, TensorView<const T,1,Cpu> grad_distance, TensorView<const T,3,Cpu> grad_barycenters, TensorView<const T,2,Cpu> grad_potentials, TensorView<const T,3,Cpu> grad_cuts, TensorView<T,3,Cpu> grad_dirac_xs, TensorView<T,2,Cpu> grad_dirac_ws, TensorView<T,2,Cpu> grad_g_values ) {
    parallel_for<PI>( 0, dirac_xs.size( 0 ), [&]( PI r ) {
        ot_plan_1d_backward( dirac_xs.row( r ), dirac_ws.row( r ), primitive.row( r ), distance.row( r ), barycenters.row( r ), potentials.row( r ), cuts.row( r ), grad_distance.row( r ), grad_barycenters.row( r ), grad_potentials.row( r ), grad_cuts.row( r ), grad_dirac_xs.row( r ), grad_dirac_ws.row( r ), grad_g_values.row( r ) );
    } );
}

} // namespace sdot
