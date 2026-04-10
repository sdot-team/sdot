#include "../../src/cpp/sdot/support/Tensor.h"
#include "../../src/cpp/sdot/PolynomialGrid.h"
#include "catch_main.h"

using namespace sdot;
using namespace std;
using TF = double;

TEST_CASE( "pg", "[Pg]" ) {
    Tensor<TF,3,Cpu> values{ { { 1 }, { 2 } }, { { 3 }, { 4 } }, };
    Tensor<TF,1,Cpu> knots;

    PolynomialGrid<TF,2,0,Cpu> pg( values, {}, { knots, knots }, true );
    ASSERT( pg.piece_integral( { 0, 0 }, pg.polynomial( { 0, 0 } ) ) == 0.25 );
    ASSERT( pg.piece_integral( { 0, 1 }, pg.polynomial( { 0, 1 } ) ) == 0.50 );
}

TEST_CASE( "pg d", "[Pg]" ) {
    Tensor<TF,3,Cpu> values{ { { 0, 1, 0 } }, };
    Tensor<TF,1,Cpu> knots;

    PolynomialGrid<TF,2,1,Cpu> pg( values, {}, { knots, knots }, true );
    ASSERT( pg.piece_integral( { 0, 0 }, pg.polynomial( { 0, 0 } ) ) == 0.5 );
}
