#include "../../src/cpp/sdot/support/containers/AxisTuple.h"
#include "../../src/cpp/sdot/support/containers/IndexRange.h"
#include "catch_main.h"
#include <string>

using namespace sdot;

TEST_CASE( "IndexRange", "" ) {
    // rank 2: for_each_item yields multi-indices in row-major order
    AxisTuple<int,2> shape( Values(), 2, 3 );
    auto idx = shape.all_indices();
    CHECK( int( idx.nb_items() ) == 6 );

    std::string seen;
    idx.for_each_item( [&]( auto it ) {
        seen += std::to_string( int( it[ Ct<int,0>() ] ) ) + std::to_string( int( it[ Ct<int,1>() ] ) ) + " ";
    } );
    CHECK( seen == "00 01 02 10 11 12 " );

    // rank 1 split: thread 0/2 sees 0,2,4 ; thread 1/2 sees 1,3
    AxisTuple<int,1> r1( Values(), 5 );
    std::string t0, t1;
    r1.all_indices().for_each_item_split( 0, 2, [&]( auto it ) { t0 += std::to_string( int( it[ Ct<int,0>() ] ) ) + " "; } );
    r1.all_indices().for_each_item_split( 1, 2, [&]( auto it ) { t1 += std::to_string( int( it[ Ct<int,0>() ] ) ) + " "; } );
    CHECK( t0 == "0 2 4 " );
    CHECK( t1 == "1 3 " );

    // rank 0: a single item, run only by thread 0
    AxisTuple<int,0> r0{ Values() };
    int cnt0 = 0, cnt1 = 0;
    r0.all_indices().for_each_item_split( 0, 2, [&]( auto ) { ++cnt0; } );
    r0.all_indices().for_each_item_split( 1, 2, [&]( auto ) { ++cnt1; } );
    CHECK( cnt0 == 1 );
    CHECK( cnt1 == 0 );
}
