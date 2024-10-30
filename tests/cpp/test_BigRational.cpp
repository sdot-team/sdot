#include <tl/support/containers/Vec.h>
#include <sdot/support/BigRational.h>
#include "catch_main.h"
#include <Eigen/LU>

using namespace sdot;
using BR = BigRational;

#define CHECK_BIGR( VALUE, NUM, DEN, EXP ) \
    CHECK_REPR( ( VALUE ).denominator(), DEN ); \
    CHECK_REPR( ( VALUE ).numerator(), NUM ); \
    CHECK_REPR( ( VALUE ).exponent(), EXP );

TEST_CASE( "BigRational ctors", "" ) {
    // int init
    CHECK_BIGR( BR(    ),  0, 1, 0 );
    CHECK_BIGR( BR(  1 ),  1, 1, 0 );
    CHECK_BIGR( BR( -1 ), -1, 1, 0 );
    CHECK_BIGR( BR(  3 ),  3, 1, 0 );
    CHECK_BIGR( BR( -3 ), -3, 1, 0 );
    CHECK_BIGR( BR(  4 ),  1, 1, 2 );
    CHECK_BIGR( BR( -4 ), -1, 1, 2 );

    // rat init
    CHECK_BIGR( BR( 3, 5 ), 3, 5,  0 );
    CHECK_BIGR( BR( 6, 4 ), 3, 1, -1 );

    // operations
    CHECK_BIGR( BR( 3 )    / BR( 5 )   ,  3, 5,  0 );
    CHECK_BIGR( BR( 3 )    / BR( 5, 2 ),  3, 5,  1 );
    CHECK_BIGR( BR( 2 )    / BR( 8 )   ,  1, 1, -2 );
    CHECK_BIGR( BR( 5, 2 ) + BR( 3, 7 ), 41, 7, -1 );

    // conversions to...
    CHECK_REPR( FP64( BR( 3, 5, 1 ) ), 6.0 / 5 );
    CHECK_REPR( PI64( BR( 7, 2 ) )   , 3       );

    // construction from...
    CHECK_BIGR( BR( +0.0 ),  0, 1,  0 );
    CHECK_BIGR( BR( -0.0 ),  0, 1,  0 );
    CHECK_BIGR( BR( +0.5 ), +1, 1, -1 );
    CHECK_BIGR( BR( -0.5 ), -1, 1, -1 );
    CHECK_BIGR( BR( +2.5 ), +5, 1, -1 );
    CHECK_BIGR( BR( -2.5 ), -5, 1, -1 );
    CHECK_BIGR( BR( +10. ), +5, 1, +1 );
    CHECK_BIGR( BR( +11. ), 11, 1,  0 );

    // P( abs( BR( +0.5 ) ) );
    CHECK_REPR( abs( BR( -3.5 ) ), BR( 3.5 ) );
}

// TEST_CASE( "BigRational matrix", "" ) {
//     constexpr int nb_dims = 2;
//     using TM = Eigen::Matrix<BR,nb_dims,nb_dims>;
//     using TV = Eigen::Matrix<BR,nb_dims,1>;

//     TM m;
//     TV v;
//     m.coeffRef( 0, 0 ) = 5;
//     m.coeffRef( 0, 1 ) = 0;
//     m.coeffRef( 1, 0 ) = 0;
//     m.coeffRef( 1, 1 ) = 7;
//     v[ 0 ] = 2;
//     v[ 1 ] = 3;

//     Eigen::FullPivLU<TM> lu( m );
//     Vec<BR,nb_dims> X = lu.solve( v );

//     P( X );
// }
