#include <sdot/support/P.h>

void test_alac( auto &out, const auto &inp ) {
    out() = inp() * inp();
}

void test_alac_backward( const auto &/*ret*/, const auto &inp, const auto &grad_out_ret, auto &grad_inp_inp ) {
    if ( ! grad_inp_inp.is_valid() ) return;
    grad_inp_inp() = 2 * grad_out_ret() * inp();
}

