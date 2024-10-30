#include "catch_main.h"
#include <sdot/Cell.h>

using namespace sdot;
struct Arch {};

TEST_CASE( "Cell", "" ) {
    using C = Cell<Arch,double,2>;

    C cell;
    P( cell );

    cell.cut( { -1, +0 }, 0 );
    P( cell );

    cell.cut( { +1, +1 }, 1 );
    P( cell );

    cell.cut( { +0, -1 }, 0 );
    P( cell );
}
