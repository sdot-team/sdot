// #include <catch2/matchers/catch_matchers_floating_point.hpp>
// #include <catch2/matchers/catch_matchers_vector.hpp>
// #include <catch2/catch_test_macros.hpp>
// #include "../src/cpp/cpu/w2_distance.h"
// #include "../src/cpp/support/Tensor.h"
// #include "../src/cpp/support/P.h"
#include "catch_main.h"

using namespace sdot;
using TF = float;

// struct W2Out {
//     W2Out( PI nb_diracs ) : grad_dirac_xs( Extent(), { nb_diracs } ), grad_dirac_ws( Extent(), { nb_diracs } ), grad_point_xs( Extent(), { nb_diracs } ), grad_point_ys( Extent(), { nb_diracs } ),
//                             barycenters( Extent(), { nb_diracs } ), potentials( Extent(), { nb_diracs } ), distance(), cuts( Extent(), { nb_diracs } ) {}
//     Tensor<TF,1> grad_dirac_xs;
//     Tensor<TF,1> grad_dirac_ws;
//     Tensor<TF,1> grad_point_xs;
//     Tensor<TF,1> grad_point_ys;
//     Tensor<TF,1> barycenters;
//     Tensor<TF,1> potentials;
//     Tensor<TF,0> distance;
//     Tensor<TF,2> cuts;
// };

// static W2Out compute_w2( const Tensor<TF,1> &dirac_xs, const Tensor<TF,1> &dirac_ws, const Tensor<TF,1> &point_xs, const Tensor<TF,1> &point_ys ) {
//     Affine1d<const TF,0> functions{ .xs = point_xs, .ys = point_ys };
//     SumOfWeightedDiracs1d<const TF,0> diracs{ .xs = dirac_xs, .ws = dirac_ws };
//     const size_t nb_diracs = dirac_xs.size( 1 );
//     const size_t nb_points = point_xs.size( 1 );


//     // forward
//     W2Out res( nb_diracs );
//     w2_distance( diracs, functions, res.distance.view(), res.barycenters.view(), res.potentials.view(), res.cuts.view() );

//     // Backward
//     Tensor<TF,1> grad_bary( Extent(), { nb_diracs } );
//     Tensor<TF,0> grad_dist;

//     Affine1d<TF,0> grad_functions{ .xs = res.grad_point_xs, .ys = res.grad_point_ys };
//     SumOfWeightedDiracs1d<TF,0> grad_diracs{ .xs = res.grad_dirac_xs, .ws = res.grad_dirac_ws };

//     w2_distance_backward(
//         grad_dist.view(), grad_bary.view(), res.potentials.view(), res.barycenters.view(), res.cuts.view(),
//         diracs, functions, grad_diracs, grad_functions
//     );

//     return res;
// }

TEST_CASE("SDOT W2 CPU Single Dirac", "[cpu][w2]") {
    // SECTION( "0.0 -> 1,1" ) {
    //     auto wo = compute_w2( { 0 }, { 1 }, { 0, 1 }, { 1, 1 } );
    //     CHECK_THAT( wo.barycenters, Catch::Matchers::Approx<TF>( { 0.5 } ) );
    //     CHECK_THAT( wo.distances, Catch::Matchers::Approx<TF>( { 1.0 / 3.0 } ) );
    // }
    // SECTION( "0.5 -> 1,1" ) {
    //     auto wo = compute_w2( { 0.5 }, { 1 }, { 0, 1 }, { 1, 1 } );
    //     CHECK_THAT( wo.barycenters, Catch::Matchers::Approx<TF>( { 0.5 } ) );
    //     CHECK_THAT( wo.distances, Catch::Matchers::Approx<TF>( { 1.0 / 12.0 } ) );
    // }
    // SECTION( "0.5 -> 0,2" ) {
    //     auto wo = compute_w2( { 0.5 }, { 1 }, { 0, 1 }, { 0, 2 } );
    //     CHECK_THAT( wo.barycenters, Catch::Matchers::Approx<TF>( { 2.0 / 3.0 } ) );
    //     CHECK_THAT( wo.distances, Catch::Matchers::Approx<TF>( { 1.0 / 12.0 } ) );
    // }
}

TEST_CASE("SDOT W2 CPU Multiple Points", "[cpu][w2]") {
    // SECTION( "0.5 -> 1,1,1" ) {
    //     auto wo = compute_w2( { 0.5 }, { 1 }, { 0.0, 0.5, 0.6, 1.0 }, { 1, 1, 1, 1 } );
    //     CHECK_THAT( wo.barycenters, Catch::Matchers::Approx<TF>( { 0.5 } ) );
    //     CHECK_THAT( wo.distances, Catch::Matchers::Approx<TF>( { 1.0 / 12.0 } ) );
    // }
}
