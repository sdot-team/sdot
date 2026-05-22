#include "../../src/cpp/sdot/support/containers/CartesianProduct.h"
#include "../../src/cpp/sdot/support/containers/Range.h"
#include "../../src/cpp/sdot/support/containers/Tuple.h"
#include "catch_main.h"
#include <array>

using namespace sdot;

auto array( auto...values ) {
    return std::array{ values... };
}

TEST_CASE( "AxisValues", "" ) {
    auto t = tuple( 1, 2, 3u );
    auto c = tuple( 1, 2_c, 3u );
    auto z = tuple();

    SECTION( "no ct axis" ) {
        CHECK_REPR( t, array( 1, 2, 3 ) );

        CHECK_REPR( t.without_index( Ct<int,0>() ), array( 2, 3 ) );
        CHECK_REPR( t.without_index( Ct<int,1>() ), array( 1, 3 ) );
        CHECK_REPR( t.without_index( Ct<int,2>() ), array( 1, 2 ) );

        CHECK_REPR( t.without_index( 0 ), array( 2, 3 ) );
        CHECK_REPR( t.without_index( 1 ), array( 1, 3 ) );
        CHECK_REPR( t.without_index( 2 ), array( 1, 2 ) );

        CHECK_REPR( c[ 0_c ], 1   );
        CHECK_REPR( c[ 1_c ], 2_c );
        CHECK_REPR( c[ 2_c ], 3u  );

        CHECK_REPR( c[ 0 ], 1 );
        CHECK_REPR( c[ 1 ], 2 );
        CHECK_REPR( c[ 2 ], 3 );

        CHECK_REPR( t.with_appended_value( 17 ), array( 1, 2, 3, 17 ) );
        CHECK_REPR( z.with_appended_value( 17 ), array( 17 ) );
    }

    SECTION( "map" ) {
        CHECK_REPR( map( c, []( auto v ) { return 2_c * v; } ), tuple( 2, 4_c, 6u ) );
        info( cartesian_product( map( c, range<PI> ) ) );
    }
}
