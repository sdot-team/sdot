#include <sdot/support/P.h>

template<class T>
struct Pouet { T pouet; };

void test_alac( auto &&p ) {
    if ( p.o0.is_valid() && p.i0.is_valid() )
        p.o0() = p.i0() * p.i0();
    // if ( p.o1.is_valid() ) //  && p.i1.is_valid()
    //     p.o1() = 178;
    p.o1.pouet[ 0 ] = 43;
}

void test_alac_backward( auto &&p ) {
    // P( p.o0.is_valid() );
    // P( p.o1.is_valid() );
    // P( p.i0.is_valid() );
    // P( p.i1.is_valid() );
    // P( p.grad_inp_i1.is_valid(), p.grad_out_o1.is_valid() );

    if ( p.grad_inp_i0.is_valid() && p.grad_out_o0.is_valid() && p.i0.is_valid() )
        p.grad_inp_i0() = 2 * p.grad_out_o0() * p.i0();
    P( p.grad_inp_i0(), p.grad_out_o0(), p.i0() );

    // if ( p.grad_inp_i1.is_valid() && p.grad_out_o1.is_valid() && p.i1.is_valid() )
    //     p.grad_inp_i1() = p.grad_out_o1();

}

