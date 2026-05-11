#include "../../src/cpp/sdot/support/Vector.h"
#include "catch_main.h"

using namespace sdot;
using namespace std;

TEST_CASE( "Vector", "" ) {
    SECTION( "constant" ) {
        Vector<double,Cpu,3> v( Values(), 1, 2, 3 );
        CHECK_REPR( v, "[ 1, 2, 3 ]" );
        CHECK_REPR( v.with_pushed_value( 17 ), "[ 1, 2, 3, 17 ]" );
        CHECK_REPR( v.without_index( 1 ), "[ 1, 3 ]" );
        CHECK_REPR( v.size(), 3 );
    }
}
