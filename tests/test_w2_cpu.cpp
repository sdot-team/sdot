#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_vector.hpp>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include "../src/cpu/sdot_w2_cpu.h"
#include "../src/P.h"
// #include <cmath>

using TS = float;
using PI = size_t;
using TV = std::vector<TS>;

struct W2Out {
    TV barycenters;
    TV distances;

    TV grad_barycenters;
    TV grad_distances;

    PI nb_batches;

    TV grad_dirac_xs;
    TV grad_dirac_ws;
    TV grad_point_xs;
    TV grad_point_ys;
};

static W2Out compute_w2( TV dirac_xs, TV dirac_ws, TV point_xs, TV point_ys, PI nb_batches = 1 ) {
    const PI nb_diracs = dirac_xs.size() / nb_batches;
    const PI nb_points = point_xs.size() / nb_batches;

    // forward
    W2Out res;
    res.nb_batches = nb_batches;
    res.distances.resize( nb_batches, -1 );
    res.barycenters.resize( nb_diracs * nb_batches, -1 );

    sdot_w2_cpu( dirac_xs.data(), dirac_ws.data(), nb_diracs, point_xs.data(), point_ys.data(), nb_points, nb_batches, res.distances.data(), res.barycenters.data() );

    // numeric diff
    const TS eps = 1e-4;
    TV grad_dirac_xs_num( nb_batches * nb_diracs );
    for( PI o = 0; o < dirac_xs.size(); ++o ) { // TODO: batches
        TV new_dirac_xs = dirac_xs;
        new_dirac_xs[ o ] += eps;

        TV new_distances( nb_batches, -1 );
        TV new_barycenters( nb_diracs * nb_batches, -1 );
        sdot_w2_cpu( new_dirac_xs.data(), dirac_ws.data(), nb_diracs, point_xs.data(), point_ys.data(), nb_points, nb_batches, new_distances.data(), new_barycenters.data() );

        grad_dirac_xs_num[ o ] = ( new_distances[ 0 ] - res.distances[ 0 ] ) / eps;
    }

    // backward
    res.grad_barycenters.resize( nb_diracs * nb_batches, 0 );
    res.grad_distances.resize( nb_batches, 1 );

    res.grad_dirac_xs.resize( nb_diracs * nb_batches, -1 );
    res.grad_dirac_ws.resize( nb_diracs * nb_batches, -1 );
    res.grad_point_xs.resize( nb_diracs * nb_batches, -1 );
    res.grad_point_ys.resize( nb_diracs * nb_batches, -1 );

    sdot_w2_backward_cpu(
       res.grad_distances.data(),
       res.grad_barycenters.data(),
       res.barycenters.data(),
       dirac_xs.data(), dirac_ws.data(), nb_diracs,
       point_xs.data(), point_ys.data(), nb_points,
       nb_batches,
       res.grad_dirac_xs.data(),
       res.grad_dirac_ws.data(),
       res.grad_point_xs.data(),
       res.grad_point_ys.data()
    );

    for( PI i = 0; i < res.grad_dirac_xs.size(); ++i )
        CHECK_THAT( res.grad_dirac_xs[ i ], Catch::Matchers::WithinAbsMatcher( grad_dirac_xs_num[ i ], 1e-3 ) );

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

TEST_CASE( "SDOT W2 CPU Single Dirac", "[cpu][w2]" ) {
    SECTION( "0.0 -> 1,1" ) {
        W2Out wo = compute_w2( { 0 }, { 1 }, { 0, 1 }, { 1, 1 } );
        CHECK_THAT( wo.barycenters, Catch::Matchers::Approx<TS>( { 0.5 } ) );
        CHECK_THAT( wo.distances, Catch::Matchers::Approx<TS>( { 1.0 / 3.0 } ) );
    }

    SECTION( "0.5 -> 1,1" ) {
        W2Out wo = compute_w2( { 0.5 }, { 1 }, { 0, 1 }, { 1, 1 } );
        CHECK_THAT( wo.barycenters, Catch::Matchers::Approx<TS>( { 0.5 } ) );
        CHECK_THAT( wo.distances, Catch::Matchers::Approx<TS>( { 1.0 / 12.0 } ) );
    }

    SECTION( "0.5 -> 0,2" ) {
        W2Out wo = compute_w2( { 0.5 }, { 1 }, { 0, 1 }, { 0, 2 } );
        CHECK_THAT( wo.barycenters, Catch::Matchers::Approx<TS>( { 2.0 / 3.0 } ) );
        CHECK_THAT( wo.distances, Catch::Matchers::Approx<TS>( { 1.0 / 12.0 } ) );
    }
}

TEST_CASE( "SDOT W2 CPU Multiple Points", "[cpu][w2]" ) {
    SECTION( "0.5 -> 1,1,1" ) {
        W2Out wo = compute_w2( { 0.5 }, { 1 }, { 0.0, 0.5, 0.6, 1.0 }, { 1, 1, 1, 1 } );
        CHECK_THAT( wo.barycenters, Catch::Matchers::Approx<TS>( { 0.5 } ) );
        CHECK_THAT( wo.distances, Catch::Matchers::Approx<TS>( { 1.0 / 12.0 } ) );
    }
}

