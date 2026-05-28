#include "../../src/cpp/sdot/support/containers/CartesianProduct.h"
#include "../../src/cpp/sdot/support/containers/Range.h"
#include "catch_main.h"
#include <algorithm>
#include <string>
#include <vector>

using namespace sdot;

// Range: the strided split must cover every item exactly once across the `mod` threads, and each
// thread must see a strictly increasing run with step `mod` starting at `rel`.
TEST_CASE( "Range::for_each_item_split", "" ) {
    for ( int n : { 0, 1, 5, 13 } ) {
        Range<int> r{ n };
        for ( PI mod : { PI( 1 ), PI( 2 ), PI( 3 ), PI( 8 ) } ) {
            std::vector<int> all;
            for ( PI rel = 0; rel < mod; ++rel ) {
                std::vector<int> per;
                r.for_each_item_split( rel, mod, [&]( int i ) { per.push_back( i ); } );
                if ( ! per.empty() )
                    CHECK( PI( per.front() ) == rel );
                for ( PI j = 1; j < per.size(); ++j )
                    CHECK( PI( per[ j ] - per[ j - 1 ] ) == mod );
                all.insert( all.end(), per.begin(), per.end() );
            }
            std::sort( all.begin(), all.end() );
            REQUIRE( all.size() == size_t( n ) );
            for ( int i = 0; i < n; ++i )
                CHECK( all[ i ] == i );
        }
    }
}

// CartesianProducts: item_at and the strided split must agree with for_each_item's row-major order,
// and the split must cover every multi-index exactly once across the threads.
TEST_CASE( "CartesianProducts::for_each_item_split", "" ) {
    auto key = []( auto it ) {
        std::string s;
        it.apply_values( [&]( auto... v ) { ( ( s += std::to_string( int( v ) ) + "," ), ... ); } );
        return s;
    };

    auto cp = cartesian_product_args( Range<int>{ 2 }, Range<int>{ 3 }, Range<int>{ 2 } );
    const PI N = PI( cp.nb_items() );
    REQUIRE( N == 12 );

    // reference flattened order
    std::vector<std::string> ref;
    cp.for_each_item( [&]( auto it ) { ref.push_back( key( it ) ); } );
    REQUIRE( ref.size() == N );

    // item_at( k ) must equal the k-th item of for_each_item
    for ( PI k = 0; k < N; ++k )
        CHECK( key( cp.item_at( k ) ) == ref[ k ] );

    // split coverage + per-thread stride in linear order
    for ( PI mod : { PI( 1 ), PI( 3 ), PI( 5 ), PI( 16 ) } ) {
        std::vector<std::string> all;
        for ( PI rel = 0; rel < mod; ++rel ) {
            std::vector<std::string> per;
            cp.for_each_item_split( rel, mod, [&]( auto it ) { per.push_back( key( it ) ); } );
            for ( PI j = 0; j < per.size(); ++j )
                CHECK( per[ j ] == ref[ rel + j * mod ] );
            all.insert( all.end(), per.begin(), per.end() );
        }
        std::sort( all.begin(), all.end() );
        auto sorted_ref = ref;
        std::sort( sorted_ref.begin(), sorted_ref.end() );
        CHECK( all == sorted_ref );
    }
}
