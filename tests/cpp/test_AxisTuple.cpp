#include "../../src/cpp/sdot/support/containers/AxisTuple.h"
#include "catch_main.h"

using namespace sdot;
using namespace std;

TEST_CASE( "AxisTuple", "" ) {
    SECTION( "no ct axis" ) {
        AxisTuple<int,3> shape( Values(), 10, 20, 30 );
        CHECK_REPR( shape, "[ 10, 20, 30 ]" );
        CHECK_REPR( shape.without_index( Ct<int,1>() ), "[ 10, 30 ]" );
        CHECK_REPR( (shape[ Ct<int,0>() ]), 10 );
        CHECK_REPR( (shape[ Ct<int,1>() ]), 20 );
        CHECK_REPR( (shape[ Ct<int,2>() ]), 30 );
        CHECK_REPR( shape[ 0 ], 10 );
        CHECK_REPR( shape[ 1 ], 20 );
        CHECK_REPR( shape[ 2 ], 30 );
    }
    SECTION( "ct axis" ) {
        AxisTuple<int,3,KnownAxisSize<int,1,20>> shape( Values(), 10, 20, 30 );
        AxisTuple<int,3,KnownAxisSize<int,1,50>> epahs( Values(), 40, 50, 60 );
        CHECK_REPR( shape, "[ 10, 20, 30 ]" );
        CHECK_REPR( concat( shape, epahs ), "[ 10, 20, 30, 40, 50, 60 ]" );
        CHECK_REPR( shape.without_index( Ct<int,1>() ), "[ 10, 30 ]" );
        CHECK_REPR( (shape[ Ct<int,0>() ]), 10 );
        CHECK_REPR( (shape[ Ct<int,1>() ]), 20 );
        CHECK_REPR( (shape[ Ct<int,2>() ]), 30 );
        CHECK_REPR( shape[ 0 ], 10 );
        CHECK_REPR( shape[ 1 ], 20 );
        CHECK_REPR( shape[ 2 ], 30 );
    }
}
