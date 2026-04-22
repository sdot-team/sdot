#include <sdot/support/P.h>

void test_alac( auto &&p ) {
    p.o0() = p.i0() * p.i0();
    p.o1() = p.i1() + 178;
}

void test_alac_backward( auto &&p ) {
    P( p.grad_inp_i0.is_valid(), p.grad_inp_i1.is_valid() );
    P( p.grad_out_o0.is_valid(), p.grad_out_o1.is_valid() );

    if ( p.grad_inp_i0.is_valid() && p.grad_out_o0.is_valid() )
        p.grad_inp_i0() = 2 * p.grad_out_o0() * p.i0();
    if ( p.grad_inp_i1.is_valid() && p.grad_out_o1.is_valid() )
        p.grad_inp_i1() = p.grad_out_o1();

}

