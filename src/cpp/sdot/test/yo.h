#include "../support/DynamicAxis.h"
#include "../support/P.h"
using namespace sdot;

void yo( auto &&p ) {
    P( p.ct_dim );
    P( p.inp.ct_dim );
    p.ret.fill_with( 0 );
    p.nb_elems = 3;
}

void yo_backward( auto && ) {
}
