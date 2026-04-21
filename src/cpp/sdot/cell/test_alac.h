#include <sdot/support/P.h>

void test_alac( auto &out ) {
    std::get<0>( out )() += 17;
}

// void test_alac( auto &out_a, const auto &inp, auto &out_b ) {
//     out_a() = inp() * inp();
//     out_b() = inp() + 5;
// }

void test_alac_backward( auto &out, const auto &inp, auto &grad_out, const auto &grad_inp ) {
    P( out(), inp(), grad_out(), grad_inp() );
    grad_inp() = 2 * grad_out() * inp();
}

