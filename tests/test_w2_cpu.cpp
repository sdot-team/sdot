#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/catch_test_macros.hpp>
#include "../src/cpu/w2_distance.h"
#include "../src/support/P.h"

using namespace sdot;
using TF = float;
using TV = std::vector<TF>;

struct W2Out {
    TV barycenters, distances, grad_dirac_xs, grad_dirac_ws, grad_point_xs, grad_point_ys;
};

static W2Out compute_w2( const TV &dirac_xs, const TV &dirac_ws, const TV &point_xs, const TV &point_ys, PI nb_batches = 1 ) {
    const auto cview2 = [&]( const TV &v ) -> TensorView<const TF,2>{ return { v.data(), { nb_batches, v.size() / nb_batches } }; };
    const auto cview1 = [&]( const TV &v ) -> TensorView<const TF,1>{ return { v.data(), nb_batches }; };
    const auto view2 = [&]( TV &v ) -> TensorView<TF,2>{ return { v.data(), { nb_batches, v.size() / nb_batches } }; };
    const auto view1 = [&]( TV &v ) -> TensorView<TF,1>{ return { v.data(), nb_batches }; };
    const size_t nb_diracs = dirac_xs.size() / nb_batches;
    const size_t nb_points = point_xs.size() / nb_batches;

    Affine1d<const TF,1> functions{ .xs = cview2( point_xs ), .ys = cview2( point_ys ) };
    DiracSet<const TF,1> diracs{ .xs = cview2( dirac_xs ), .ws = cview2( dirac_ws ) };

    // forward
    W2Out res;
    res.barycenters.resize( nb_diracs * nb_batches );
    res.distances.resize( nb_batches );
    w2_distance( diracs, functions, view1( res.distances ), view2( res.barycenters ) );

    // Numeric diff for validation
    // const TF eps = 1e-4;
    // TV grad_num(dirac_xs.size());
    // for (size_t i = 0; i < dirac_xs.size(); ++i) {
    //     TV dxs_eps = dirac_xs;
    //     dxs_eps[i] += eps;
    //     TV d_eps(nb_batches), b_eps(nb_diracs * nb_batches);
    //     run_fwd(dxs_eps, d_eps, b_eps);
    //     grad_num[i] = (d_eps[0] - res.distances[0]) / eps;
    // }

    // Backward
    res.grad_dirac_xs.resize( nb_diracs * nb_batches );
    res.grad_dirac_ws.resize( nb_diracs * nb_batches );
    res.grad_point_xs.resize( nb_points * nb_batches );
    res.grad_point_ys.resize( nb_points * nb_batches );
    TV grad_bary( nb_diracs * nb_batches, 0 );
    TV grad_dist( nb_batches, 1 );
    // T_T void w2_distance_backward( TensorView<const T,1> grad_w2_squared, TensorView<const T,2> grad_w2_barycenters, TensorView<const T,2> w2_barycenters, DiracSet<T,1> diracs, Affine1d<T,1> functions, DiracSet<T,1> grad_diracs, Affine1d<T,1> grad_functions );

    Affine1d<TF,1> grad_functions{ .xs = view2( res.grad_point_xs ), .ys = view2( res.grad_point_ys ) };
    DiracSet<TF,1> grad_diracs{ .xs = view2( res.grad_dirac_xs ), .ws = view2( res.grad_dirac_ws ) };

    w2_distance_backward( cview1( grad_dist ), cview2( grad_bary ), cview2( res.barycenters ), diracs, functions, grad_diracs, grad_functions );

    // for (size_t i = 0; i < res.grad_dirac_xs.size(); ++i)
    //     CHECK_THAT(res.grad_dirac_xs[i], Catch::Matchers::WithinAbs(grad_num[i], 1e-3));

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
