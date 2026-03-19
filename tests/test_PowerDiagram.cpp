#include "../src/cpp/geometry/Cell.h"
#include "../src/cpp/support/P.h"
#include "catch_main.h"

using namespace sdot;
using TF = double;

TEST_CASE("2D power diagram", "[PD]") {
    // Cell<TF,2> p2( 2, 10.0 );
    // P( p2 );

    Cell<TF,3> p3( 3, 10.0 );
    P( p3 );

    p3.cut( p3.pf( 1, 0, 0 ), 1, 17 );
    P( p3 );

    VtkOutput vo;
    p3.display_vtk( vo );
    vo.save( "build/out.vtk" );
}
