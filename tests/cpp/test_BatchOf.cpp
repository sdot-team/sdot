#include "../../src/cpp/sdot/support/containers/TensorView.h"
#include "../../src/cpp/sdot/support/containers/BatchOf.h"
#include "catch_main.h"

#ifdef __CUDACC__
cudaStream_t sdot::ExecutionContext_Cuda::default_stream = 0;
#endif

using namespace sdot;

struct ax_n {};   // a batch axis
struct ax_x {};   // a second batch axis
struct ax_d {};   // core axis

TEST_CASE( "BatchOf — resolves a named batch axis, broadcasts when absent", "" ) {
    double data[ 6 ] = { 0, 1, 2, 10, 11, 12 };          // ( n, d ) == data[ n*3 + d ]
    auto shape = tuple( 2, 3 );
    TensorView raw( data, shape, contiguous_strides<double>( shape ), MemorySpace_CpuRam{} );
    auto t = raw.with_tags<container_tags::AxisNames<ax_n,ax_d>>();

    BatchOf<DECAYED_TYPE_OF( t ),ax_n> bo{ t };
    CHECK( bo( tuple( axis_index<ax_n>( 1 ) ) ).squeeze( ax_d{}, 2 ).value() == 12.0 );

    // a BatchOf with no axes ignores the index entirely (broadcast)
    BatchOf<DECAYED_TYPE_OF( t )> bcast{ t };
    CHECK( bcast( tuple( axis_index<ax_n>( 1 ) ) ).squeeze( ax_n{}, 0 ).squeeze( ax_d{}, 1 ).value() == 1.0 );
}

TEST_CASE( "BatchOf — partial resolution leaves the unresolved axis named", "" ) {
    double data[ 12 ];                                   // ( x, n, d ) = x*100 + n*10 + d
    for ( int x = 0; x < 2; ++x )
        for ( int n = 0; n < 2; ++n )
            for ( int d = 0; d < 3; ++d )
                data[ x*6 + n*3 + d ] = x*100 + n*10 + d;
    auto shape = tuple( 2, 2, 3 );
    TensorView raw( data, shape, contiguous_strides<double>( shape ), MemorySpace_CpuRam{} );
    auto t = raw.with_tags<container_tags::AxisNames<ax_x,ax_n,ax_d>>();

    BatchOf<DECAYED_TYPE_OF( t ),ax_n,ax_x> bo{ t };
    auto rest = bo( tuple( axis_index<ax_n>( 1 ) ) );    // resolve ax_n only
    using Rest = DECAYED_TYPE_OF( rest );
    static_assert( Rest::template has_tag<container_tags::AxisNames<ax_x,ax_d>>, "ax_n peeled, ax_x kept" );

    CHECK( rest.squeeze( ax_x{}, 1 ).squeeze( ax_d{}, 2 ).value() == 112.0 ); // x=1, d=2
}

// A multi-field aggregate whose batch axis "n" is a named leading dim shared by every field.
// `squeeze( name, i )` squeezes that name on each field and rebuilds — exactly the shape the
// codegen will emit for a generated struct (Cell, ...), so BatchOf<aggregate> works over it.
template<class A,class B>
struct MiniCell {
    A a; B b;
    template<class Name> HD auto squeeze( Name nm, PI i ) const {
        return MiniCell<DECAYED_TYPE_OF( a.squeeze( nm, i ) ),DECAYED_TYPE_OF( b.squeeze( nm, i ) )>{ a.squeeze( nm, i ), b.squeeze( nm, i ) };
    }
};

TEST_CASE( "BatchOf — over an aggregate, peels the shared named axis across all fields", "" ) {
    // field a: per-cell 3-vector, batched on n  -> shape ( 2, 3 ), names < ax_n, ax_d >
    double da[ 6 ] = { 0, 1, 2, 10, 11, 12 };            // ( n, d ) == n*10 + d
    auto sa = tuple( 2, 3 );
    TensorView ra( da, sa, contiguous_strides<double>( sa ), MemorySpace_CpuRam{} );
    auto a = ra.with_tags<container_tags::AxisNames<ax_n,ax_d>>();

    // field b: per-cell scalar, batched on n   -> shape ( 2 ), names < ax_n >
    double db[ 2 ] = { 0, 100 };
    auto sb = tuple( 2 );
    TensorView rb( db, sb, contiguous_strides<double>( sb ), MemorySpace_CpuRam{} );
    auto b = rb.with_tags<container_tags::AxisNames<ax_n>>();

    MiniCell<DECAYED_TYPE_OF( a ),DECAYED_TYPE_OF( b )> mc{ a, b };
    BatchOf<DECAYED_TYPE_OF( mc ),ax_n> bo{ mc };

    auto cell = bo( tuple( axis_index<ax_n>( 1 ) ) );   // resolve n=1 -> a bare MiniCell
    CHECK( cell.a.squeeze( ax_d{}, 2 ).value() == 12.0 );  // a's row n=1, d=2
    CHECK( cell.b.value() == 100.0 );                      // b's n=1
}
