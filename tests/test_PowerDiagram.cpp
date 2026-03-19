#include "../src/cpp/geometry/Cell.h"
#include "../src/cpp/support/P.h"
#include "catch_main.h"

using namespace sdot;
using TF = double;

TEST_CASE("2D power diagram", "[PD]") {
    Cell<TF,1> p1( 1, 1.0 );
    Cell<TF,2> p2( 2, 1.0 );
    Cell<TF,3> p3( 3, 1.0 );
    P( p1 );
    P( p2 );
    P( p3 );

    VtkOutput vo;
    p2.display_vtk( vo );
    p3.display_vtk( vo );
    vo.save( "build/out.vtk" );
}
