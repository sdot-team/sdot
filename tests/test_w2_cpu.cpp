#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/catch_test_macros.hpp>
#include "../src/cpu/sdot_w2_cpu.h"
// #include <cmath>

using TS = float;
using PI = size_t;
using TV = std::vector<TS>;

struct W2Out {
    TV barycenters;
    TV distances;
};

static W2Out compute_w2( TV dirac_xs, TV dirac_ws, TV point_xs, TV point_ys, PI nb_batches = 1 ) {
    const PI nb_diracs = dirac_xs.size() / nb_batches;
    const PI nb_points = point_xs.size() / nb_batches;

    W2Out res;
    res.distances.resize( nb_batches, -1 );
    res.barycenters.resize( nb_diracs * nb_batches, -1 );

    sdot_w2_cpu( dirac_xs.data(), dirac_ws.data(), nb_diracs, point_xs.data(), point_ys.data(), nb_points, nb_batches, res.distances.data(), res.barycenters.data() );

    return res;
}

static TV linspace( TS v0, TS v1, PI n ) {
    TV res( n );
    for( PI i = 0; i < n; ++i )
        res[ i ] = v0 + ( v1 - v0 ) * i / ( n - 1 );
    return res;
}

static TV full( TS v, PI n ) {
    return TV( n, v );
}

TEST_CASE("SDOT W2 CPU Single Dirac", "[cpu][w2]") {
    W2Out wo = compute_w2( { 0 }, { 1 }, { 0, 1 }, { 1, 1 } );
    CHECK_THAT( wo.barycenters, Catch::Matchers::Approx<TS>( { 0.5 } ) );
    CHECK_THAT( wo.distances, Catch::Matchers::Approx<TS>( { 1.0 / 3.0 } ) );

    wo = compute_w2( { 0.5 }, { 1 }, { 0, 1 }, { 1, 1 } );
    CHECK_THAT( wo.barycenters, Catch::Matchers::Approx<TS>( { 0.5 } ) );
    CHECK_THAT( wo.distances, Catch::Matchers::Approx<TS>( { 1.0 / 12.0 } ) );
}

// TEST_CASE("SDOT W2 CPU Single Dirac", "[cpu][w2]") {
//     PI n = 10;
//     W2Out wo = compute_w2( linspace( 0, 1, n ), full( 1, n ), { 0.0f, 1.0f }, { 1.0f, 1.0f } );
//     CHECK_THAT( wo.barycenters, Catch::Matchers::Approx<TS>( linspace( 0.05, 0.95, n ) ) );
//     CHECK_THAT( wo.distances, Catch::Matchers::Approx<TS>( { 0.5f } ) );
// }

// TEST_CASE("SDOT W2 CPU Two Diracs", "[cpu][w2]") {
//     W2Out wo = compute_w2( { 0.25f, 0.75f }, { 0.5f, 0.5f }, { 0.0f, 1.0f }, { 1.0f, 1.0f } );
//     CHECK_THAT( wo.distances  , Catch::Matchers::Approx<TS>( { 0.0f } ) );
//     CHECK_THAT( wo.barycenters, Catch::Matchers::Approx<TS>( { 0.25f, 0.75f } ) );
// }

// TEST_CASE("SDOT W2 CPU Batch", "[cpu][w2]") {
//     W2Out wo = compute_w2( { 0.5f, 0.0f }, {1.0f, 1.0f}, { 0.0f, 1.0f, 0.0f, 1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f }, 2 );
//     CHECK_THAT( wo.distances, Catch::Matchers::Approx<TS>( { 0.0f, 1.0 } ) );
// }
