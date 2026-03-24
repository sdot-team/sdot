#include "../src/cpp/geometry/SimpleSquareMatrix.h"
#include "catch_main.h"

using namespace sdot;
using namespace std;
using TF = double;

template<int ct_size=-1>
auto make_mat( vector<TF> values ) {
    PI size = round( sqrt( values.size() ) );
    SimpleSquareMatrix<TF,ct_size> res( size );
    auto iter = values.begin();
    for( auto &v : res )
        v = *( iter++ );
    return res;
}

TEST_CASE("matrix", "[PD]") {
    SECTION( "s" ) {
        auto m = make_mat( { 1, 0, 0,  0, 2, 0,  0, 0, 3 } );
        CHECK( m.determinant() == 6 );
        auto solve = m.solve( { 10, 20, 30 } );
        for( auto v : solve )
            CHECK( v == 10 );
    }
    SECTION( "t" ) {
        auto m = make_mat( { 1, 5, 3,  2, 2, 8,  2, 0, 3 } );
        CHECK( m.determinant() == 44 );
        auto solve = m.solve( { 20, 30, 11 } );
        for( PI i = 0; i < 3; ++i )
            CHECK( solve[ i ] == ( i + 1 ) );
    }
}
