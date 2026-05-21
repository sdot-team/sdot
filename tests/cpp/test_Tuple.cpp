#include "../../src/cpp/sdot/support/containers/Tuple.h"
#include "catch_main.h"
#include <array>

using namespace sdot;

auto array( auto...values ) {
    return std::array{ values... };
}

TEST_CASE( "AxisValues", "" ) {
    SECTION( "no ct axis" ) {
        auto t = tuple( 1, 2, 3u );
        CHECK_REPR( t, array( 1, 2, 3 ) );

        CHECK_REPR( t.without_index( Ct<int,0>() ), array( 2, 3 ) );
        CHECK_REPR( t.without_index( Ct<int,1>() ), array( 1, 3 ) );
        CHECK_REPR( t.without_index( Ct<int,2>() ), array( 1, 2 ) );

        CHECK_REPR( t.without_index( 0 ), array( 2, 3 ) );
        CHECK_REPR( t.without_index( 1 ), array( 1, 3 ) );
        CHECK_REPR( t.without_index( 2 ), array( 1, 2 ) );

        auto c = tuple( 1, Ct<int,2>(), 3u );
        CHECK_REPR( c[ (Ct<int,0>()) ], 1 );
        CHECK_REPR( c[ (Ct<int,1>()) ], Ct<int,2>() );
        CHECK_REPR( c[ (Ct<int,2>()) ], 3u );

        // AxisValues<int,3> shape( Values(), 10, 20, 30 );
        // CHECK_REPR( shape, "[ 10, 20, 30 ]" );
        // CHECK_REPR( shape.without_index( Ct<int,1>() ), "[ 10, 30 ]" );
        // CHECK_REPR( (shape[ Ct<int,0>() ]), 10 );
        // CHECK_REPR( (shape[ Ct<int,1>() ]), 20 );
        // CHECK_REPR( (shape[ Ct<int,2>() ]), 30 );
        // CHECK_REPR( shape[ 0 ], 10 );
        // CHECK_REPR( shape[ 1 ], 20 );
        // CHECK_REPR( shape[ 2 ], 30 );
    }
    SECTION( "ct axis" ) {
        // AxisValues<int,3,StaticAxisValue<int,1,20>> shape( Values(), 10, 20, 30 );
        // AxisValues<int,3,StaticAxisValue<int,1,50>> epahs( Values(), 40, 50, 60 );
        // CHECK_REPR( shape, "[ 10, 20, 30 ]" );
        // CHECK_REPR( concat( shape, epahs ), "[ 10, 20, 30, 40, 50, 60 ]" );
        // CHECK_REPR( shape.without_index( Ct<int,1>() ), "[ 10, 30 ]" );
        // CHECK_REPR( (shape[ Ct<int,0>() ]), 10 );
        // CHECK_REPR( (shape[ Ct<int,1>() ]), 20 );
        // CHECK_REPR( (shape[ Ct<int,2>() ]), 30 );
        // CHECK_REPR( shape[ 0 ], 10 );
        // CHECK_REPR( shape[ 1 ], 20 );
        // CHECK_REPR( shape[ 2 ], 30 );
    }
}
