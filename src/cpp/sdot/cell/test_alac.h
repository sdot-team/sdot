#include <sdot/support/P.h>

template<class T>
struct Pouet { T positions; };

void test_alac( auto &&p ) {
    // P( p.b );
    // p.ret.positions( 0 ) = 23;
    p.o0() = p.i0() * p.i0();
    p.o1() = p.i1( 0 ) + 23;
    P( p.i1( 0 ) );
    // if ( p.o0.is_valid() && p.i0.is_valid() )
    //     p.o0() = p.i0() * p.i0();
    // if ( p.o1.is_valid() && p.i1.is_valid() ) //
    //     p.o1() = p.i1() + 178;
}

void test_alac_backward( auto &&p ) {
    if ( p.grad_of_inp_i0.is_valid() && p.grad_of_out_o0.is_valid() )
        p.grad_of_inp_i0() = 2 * p.grad_of_out_o0() * p.i0();
    // if ( p.grad_of_inp_i1.is_valid() && p.grad_of_out_o1.is_valid() )
    //     p.grad_of_inp_i1() = p.grad_of_out_o1();
    P( p.grad_of_inp_i0.is_valid(), p.grad_of_out_o0.is_valid() );
    // P( p.grad_of_inp_i1.is_valid(), p.grad_of_out_o1.is_valid() );

    // if ( p.grad_i0.is_valid() && p.grad_o0.is_valid() && p.i0.is_valid() )
    //     p.grad_i0() = 2 * p.grad_o0() * p.i0();

    // if ( p.grad_i1.is_valid() && p.grad_o1.is_valid() )
    //     p.grad_i1() = p.grad_o1();
}

