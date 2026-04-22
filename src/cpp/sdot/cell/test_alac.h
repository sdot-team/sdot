#include <sdot/support/P.h>

void test_alac( auto &o0, const auto &i0, auto &o1, const auto &i1 ) {
    o0() = i0() * i0();
    o1() = i1() + 178;
}

void test_alac_backward( const auto &/*o0*/, const auto &i0, const auto &/*o1*/, const auto &/* i1 */, const auto &grad_out_o0, const auto &grad_out_o1, auto &grad_inp_i0, auto &grad_inp_i1 ) {
    P( grad_inp_i0.is_valid(), grad_inp_i1.is_valid() );
    P( grad_out_o0.is_valid(), grad_out_o1.is_valid() );

    if ( grad_inp_i0.is_valid() && grad_out_o0.is_valid() )
        grad_inp_i0() = 2 * grad_out_o0() * i0();
    if ( grad_inp_i1.is_valid() && grad_out_o1.is_valid() )
        grad_inp_i1() = grad_out_o1();

    }

