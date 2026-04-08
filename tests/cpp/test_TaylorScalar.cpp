#include "../../src/cpp/sdot/support/TaylorScalar.h"
#include "../../src/cpp/sdot/support/P.h"
#include "catch_main.h"

using namespace sdot;
using namespace std;
using TF = double;

TEST_CASE("TS", "[PD]") {
    using TS = TaylorScalar<TF>;

    TS a = TS::variable( 3, 1 );
    TS b = TS::variable( 3, 2 );
    P( a * b );
    P( b * a );
}
