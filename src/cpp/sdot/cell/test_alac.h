#include <sdot/support/P.h>

// template<class T,class S>
// struct Pouet { T t; S s; };

void test_alac( auto &&p ) {
    if ( p.o0.is_valid() && p.i0.is_valid() )
        p.o0() = p.i0() * p.i0();
    if ( p.o1.is_valid() && p.i1.is_valid() ) //
        p.o1() = p.i1() + 178;
}

void test_alac_backward( auto &&p ) {
    // P( p.o0.is_valid() );
    // P( p.o1.is_valid() );
    // P( p.i0.is_valid() );
    // P( p.i1.is_valid() );
    // P( p.grad_i1.is_valid(), p.grad_o1.is_valid() );

    if ( p.grad_i0.is_valid() && p.grad_o0.is_valid() && p.i0.is_valid() )
        p.grad_i0() = 2 * p.grad_o0() * p.i0();
    P( p.grad_i0(), p.grad_o0(), p.i0() );

    if ( p.grad_i1.is_valid() && p.grad_o1.is_valid() )
        p.grad_i1() = p.grad_o1();
}

