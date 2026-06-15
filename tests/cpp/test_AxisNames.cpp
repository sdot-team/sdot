#include "../../src/cpp/sdot/support/containers/TensorView.h"
#include "catch_main.h"

#ifdef __CUDACC__
cudaStream_t sdot::ExecutionContext_Cuda::default_stream = 0;
#endif

using namespace sdot;

// axis-name markers (in real code these are generated, one per named axis)
struct ax_row {};
struct ax_col {};

// A 2x3 row-major tensor: element ( i, j ) == data[ i*3 + j ]; axis 0 named "row", axis 1 "col".
template<class F>
static void with_named_2x3( F &&f ) {
    double data[ 6 ] = { 0, 1, 2, 10, 11, 12 };
    auto shape   = tuple( 2, 3 );
    auto strides = contiguous_strides<double>( shape );
    TensorView raw( data, shape, strides, MemorySpace_CpuRam{} );
    f( raw.with_tags<container_tags::AxisNames<ax_row,ax_col>>() );
}

TEST_CASE( "AxisNames — squeeze by name reaches the same element as by position", "" ) {
    with_named_2x3( [&]( auto t ) {
        // ( row 1, col 2 ) == 12, reached by name in either order
        CHECK( t.squeeze( ax_row{}, 1 ).squeeze( ax_col{}, 2 ).value() == 12.0 );
        CHECK( t.squeeze( ax_col{}, 2 ).squeeze( ax_row{}, 1 ).value() == 12.0 );

        // by name == by position ( ax_row is axis 0, ax_col is axis 1 )
        CHECK( t.squeeze( ax_row{}, 1 ).squeeze( ax_col{}, 0 ).value()
            == t.squeeze( Ct<int,0>(), 1 ).squeeze( Ct<int,0>(), 0 ).value() );  // == 10
        CHECK( t.squeeze( ax_row{}, 1 ).squeeze( ax_col{}, 0 ).value() == 10.0 );
    } );
}

TEST_CASE( "AxisNames — squeeze drops the name; the rest stays addressable by name", "" ) {
    with_named_2x3( [&]( auto t ) {
        auto r = t.squeeze( ax_row{}, 0 );             // shape ( 3 ), names < ax_col >
        using R = DECAYED_TYPE_OF( r );
        static_assert(   R::template has_tag<container_tags::AxisNames<ax_col>> );
        static_assert( ! R::template has_tag<container_tags::AxisNames<ax_row,ax_col>> );

        CHECK( r.squeeze( ax_col{}, 1 ).value() == 1.0 ); // ( row 0, col 1 )
    } );
}
