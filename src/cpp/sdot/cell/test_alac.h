#include <sdot/support/P.h>

void test_alac( auto &out, const auto &inp ) {
    out() = inp() * inp();
}

void test_alac_backward( auto &out, const auto &inp, auto &grad_out, const auto &grad_inp ) {
    P( out(), inp(), grad_out(), grad_inp() );
    grad_inp() = 2 * grad_out() * inp();
}

