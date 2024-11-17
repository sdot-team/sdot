#include <sdot/symbolic/Expr.h>
#include "catch_main.h"

using namespace sdot;

TEST_CASE( "BigRational ctors", "" ) {
    Expr e( "x" );
    Expr f( 10 );

    P( e + f );
}
