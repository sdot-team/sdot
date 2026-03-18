#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/catch_test_macros.hpp>
#include "../src/cpp/cpu/w2_distance.h"
#include "../src/cpp/support/P.h"

using namespace sdot;
using TF = float;
using TV = std::vector<TF>;

struct W2Out {
    TV barycenters, potentials, distances, cuts, grad_dirac_xs, grad_dirac_ws, grad_point_xs, grad_point_ys;
};

static W2Out compute_w2( const TV &dirac_xs, const TV &dirac_ws, const TV &point_xs, const TV &point_ys, PI batch_size = 1 ) {
    const auto cview2 = [&]( const TV &v ) { return TensorView<const TF,2>{ v.data(), { batch_size, v.size() / batch_size } }; };
    const auto cview1 = [&]( const TV &v ) { return TensorView<const TF,1>{ v.data(), batch_size }; };

    const auto view2 = [&]( TV &v ) { return TensorView<TF,2>{ v.data(), { batch_size, v.size() / batch_size } }; };
    const auto view1 = [&]( TV &v ) { return TensorView<TF,1>{ v.data(), batch_size }; };

    const size_t nb_diracs = dirac_xs.size() / batch_size;
    const size_t nb_points = point_xs.size() / batch_size;

    Affine1d<const TF,1> functions{ .xs = cview2( point_xs ), .ys = cview2( point_ys ) };
    DiracSet<const TF,1> diracs{ .xs = cview2( dirac_xs ), .ws = cview2( dirac_ws ) };

    // forward
    W2Out res;
    res.barycenters.resize( nb_diracs * batch_size );
    res.potentials.resize( nb_diracs * batch_size );
    res.distances.resize( batch_size );
    res.cuts.resize( batch_size * nb_diracs * 2 );
    w2_distance( diracs, functions, view1( res.distances ), view2( res.barycenters ), view2( res.potentials ), view2( res.cuts ) );

    // Backward
    res.grad_dirac_xs.resize( nb_diracs * batch_size );
    res.grad_dirac_ws.resize( nb_diracs * batch_size );
    res.grad_point_xs.resize( nb_points * batch_size );
    res.grad_point_ys.resize( nb_points * batch_size );
    TV grad_bary( nb_diracs * batch_size, 0 );
    TV grad_dist( batch_size, 1 );

    Affine1d<TF,1> grad_functions{ .xs = view2( res.grad_point_xs ), .ys = view2( res.grad_point_ys ) };
    DiracSet<TF,1> grad_diracs{ .xs = view2( res.grad_dirac_xs ), .ws = view2( res.grad_dirac_ws ) };

    w2_distance_backward( cview1( grad_dist ), cview2( grad_bary ), cview2( res.potentials ), cview2( res.barycenters ), cview2( res.cuts ), diracs, functions, grad_diracs, grad_functions );

    return res;
}

TEST_CASE("SDOT W2 CPU Single Dirac", "[cpu][w2]") {
    SECTION( "0.0 -> 1,1" ) {
        auto wo = compute_w2( { 0 }, { 1 }, { 0, 1 }, { 1, 1 } );
        CHECK_THAT(wo.barycenters, Catch::Matchers::Approx<TF>( { 0.5 } ) );
        CHECK_THAT(wo.distances, Catch::Matchers::Approx<TF>( { 1.0 / 3.0 } ) );
    }
    SECTION( "0.5 -> 1,1" ) {
        auto wo = compute_w2( { 0.5 }, { 1 }, { 0, 1 }, { 1, 1 } );
        CHECK_THAT(wo.barycenters, Catch::Matchers::Approx<TF>( { 0.5 } ) );
        CHECK_THAT(wo.distances, Catch::Matchers::Approx<TF>( { 1.0 / 12.0 } ) );
    }
    SECTION( "0.5 -> 0,2" ) {
        auto wo = compute_w2( {0.5 }, { 1 }, { 0, 1 }, { 0, 2 });
        CHECK_THAT(wo.barycenters, Catch::Matchers::Approx<TF>( { 2.0 / 3.0 } ) );
        CHECK_THAT(wo.distances, Catch::Matchers::Approx<TF>( { 1.0 / 12.0 } ) );
    }
}

TEST_CASE("SDOT W2 CPU Multiple Points", "[cpu][w2]") {
    SECTION( "0.5 -> 1,1,1" ) {
        auto wo = compute_w2( {0.5 }, { 1 }, { 0.0, 0.5, 0.6, 1.0 }, { 1, 1, 1, 1 } );
        CHECK_THAT( wo.barycenters, Catch::Matchers::Approx<TF>( { 0.5 } ) );
        CHECK_THAT( wo.distances, Catch::Matchers::Approx<TF>( { 1.0 / 12.0} ) );
    }
}
