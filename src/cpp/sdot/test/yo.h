#include "../support/DynamicAxis.h"
#include "../support/P.h"
using namespace sdot;

void yo( auto &&p ) {
    p.ret.fill_with( 0 );
    p.nb_elems = 3;
}

void yo_backward( auto && ) {
}
