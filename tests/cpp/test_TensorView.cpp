// #include "../../src/cpp/sdot/support/containers/contiguous_strides.h"
// #include "../../src/cpp/sdot/support/hardware/MemorySpace_CpuRam.h"
#include "../../src/cpp/sdot/support/containers/TensorView.h"
// #include "../../src/cpp/sdot/support/hardware/Run.h"
#include "catch_main.h"

#ifdef __CUDACC__
#include "../../src/cpp/sdot/support/hardware/MemorySpace_GlobalCudaRam.h"
// standalone test: no bindings to set it, so default to the CUDA default stream (0)
cudaStream_t sdot::ExecutionSpace_Cuda::default_stream = 0;
#endif

using namespace sdot;

// // run_parallel functor: doubles every element of the tensor passed as an argument.
// // Namespace-scope (a local class cannot have member templates) and HD (runs on host or device).
// struct Doubler {
//     constexpr HD void operator()( auto item, auto tv ) const { tv[ item ] *= 2; }
// };

// // run_parallel functor exercising `a( ... ) = b( ... )` (element/sub-view assignment via operator=)
// struct CopyElem {
//     constexpr HD void operator()( auto item, auto a, auto b ) const { a( item ) = b( item ); }
// };

// // --- CPU -------------------------------------------------------------------------------

// TEST_CASE( "TensorView::make_accessible — CPU pass-through", "" ) {
//     double data[ 4 ] = { 1, 2, 3, 4 };
//     using Shape   = AxisValues<int,1>;
//     using Strides = AxisValues<int,1>;
//     TensorView<double,Shape,Strides,MemorySpace_CpuRam> t(
//         data, Shape( Values(), 4 ), Strides( Values(), sizeof( double ) ) );

//     // CpuRam is accessible from the CPU execution space -> func gets the very same view,
//     // no allocation/copy (the transfer branch is elided at compile time).
//     int calls = 0;
//     t.make_accessible( ExecutionSpace_Cpu{}, [&]( auto a ) {
//         ++calls;
//         CHECK( a.data() == data ); // same buffer => no transfer happened
//         CHECK( a.nb_items() == 4 );
//     } );
//     CHECK( calls == 1 );
// }

// // Exercises the same machinery as make_accessible's transfer branch, but on the host so it
// // is verifiable here: with_reservation (informed Ptr) + contiguous_strides + view rebuild.
// TEST_CASE( "TensorView::with_same_shape — CPU", "" ) {
//     double data[ 4 ] = { 1, 2, 3, 4 };
//     using Shape   = AxisValues<int,1>;
//     using Strides = AxisValues<int,1>;
//     TensorView<double,Shape,Strides,MemorySpace_CpuRam> t(
//         data, Shape( Values(), 4 ), Strides( Values(), sizeof( double ) ) );

//     int calls = 0;
//     t.with_same_shape( MemorySpace_CpuRam{}, [&]( auto res ) {
//         ++calls;
//         CHECK( res.data() != data );        // a fresh contiguous buffer
//         CHECK( res.nb_items() == 4 );
//         CHECK( res.is_contiguous() );       // contiguous_strides<TF> produced row-major strides
//     } );
//     CHECK( calls == 1 );
// }

// TEST_CASE( "run_parallel — doubles a CPU tensor in place", "" ) {
//     double data[ 4 ] = { 1, 2, 3, 4 };
//     using Shape   = AxisValues<int,1>;
//     using Strides = AxisValues<int,1>;
//     Shape shape( Values(), 4 );
//     TensorView<double,Shape,Strides,MemorySpace_CpuRam> t( data, shape, Strides( Values(), sizeof( double ) ) );

//     // data lives in CpuRam -> the dispatch runs on the CPU execution space
//     run_parallel( shape.all_indices(), Doubler{}, t );

//     for ( int i = 0; i < 4; ++i )
//         CHECK( data[ i ] == ( i + 1 ) * 2 );
// }

// TEST_CASE( "TensorView::operator= — element-wise copy (CPU)", "" ) {
//     double da[ 3 ] = { 1, 2, 3 }, db[ 3 ] = { 10, 20, 30 };
//     using Shape   = AxisValues<int,1>;
//     using Strides = AxisValues<int,1>;
//     Shape   shape  ( Values(), 3 );
//     Strides strides( Values(), sizeof( double ) );
//     TensorView<double,Shape,Strides,MemorySpace_CpuRam> a( da, shape, strides ), b( db, shape, strides );

//     a = b;                              // whole rank-1 copy (same type -> deep, element-wise)
//     for ( int i = 0; i < 3; ++i )
//         CHECK( da[ i ] == db[ i ] );

//     da[ 0 ] = 1;
//     a( 0 ) = b( 1 );                    // a( ... ) = b( ... ): rank-0 sub-view assignment
//     CHECK( da[ 0 ] == 20 );
// }

// // --- GPU -------------------------------------------------------------------------------

// #ifdef __CUDACC__

// // Exercises the orthogonal MemorySpace primitives on the device: allocate device memory
// // (with_reservation), copy host->device then device->host (copy + the stream carried by the
// // execution space), and check the round-trip. Does not depend on contiguous_strides.
// TEST_CASE( "MemorySpace transfer round-trip — GPU", "" ) {
//     double host[ 4 ] = { 1, 2, 3, 4 };
//     double back[ 4 ] = { 0, 0, 0, 0 };

//     ExecutionSpace_Cuda es;
//     MemorySpace_GlobalCudaRam{}.with_reservation<double>( 4, [&]( Ptr<double,MemorySpace_GlobalCudaRam> dev ) {
//         copy( es, dev, Ptr<double,MemorySpace_CpuRam>( host ), 4 );                    // H2D (on es.stream)
//         cudaStreamSynchronize( es.stream );
//         copy( ExecutionSpace_Cpu{}, Ptr<double,MemorySpace_CpuRam>( back ), dev, 4 );  // D2H (syncs)
//     } );

//     for ( int i = 0; i < 4; ++i )
//         CHECK( back[ i ] == host[ i ] );
// }

// // Full orchestrator round-trip: a CPU tensor made accessible to the CUDA execution space is
// // copied to the device; we read it back and compare.
// TEST_CASE( "TensorView::make_accessible — CPU tensor to GPU", "" ) {
//     double data[ 4 ] = { 1, 2, 3, 4 }, back[ 4 ] = { 0, 0, 0, 0 };
//     using Shape = AxisValues<int,1>; using Strides = AxisValues<int,1>;
//     TensorView<double,Shape,Strides,MemorySpace_CpuRam> t( data, Shape( Values(), 4 ), Strides( Values(), sizeof( double ) ) );

//     int calls = 0;
//     t.make_accessible( ExecutionSpace_Cuda{}, [&]( auto g ) {
//         ++calls;
//         CHECK( (void *)g.data() != (void *)data );  // a real device copy was made
//         cudaMemcpy( back, g.data(), 4 * sizeof( double ), cudaMemcpyDeviceToHost );
//     } );
//     info( calls );
//     CHECK( calls == 1 );
//     for ( int i = 0; i < 4; ++i )
//         CHECK( back[ i ] == data[ i ] );
// }

// // run_parallel on device-resident data: the dispatch infers the CUDA execution space from the
// // tensor's memory space and launches the kernel; each thread doubles its element.
// TEST_CASE( "run_parallel — doubles a GPU tensor", "" ) {
//     double host[ 4 ] = { 1, 2, 3, 4 }, back[ 4 ] = { 0, 0, 0, 0 };
//     using Shape = AxisValues<int,1>;
//     Shape shape( Values(), 4 );
//     auto  strides = contiguous_strides<double>( shape );

//     MemorySpace_GlobalCudaRam{}.with_reservation<double>( 4, [&]( auto dev ) {
//         copy( ExecutionSpace_Cuda{}, dev, Ptr<double,MemorySpace_CpuRam>( host ), 4 );
//         cudaStreamSynchronize( ExecutionSpace_Cuda{}.stream );

//         TensorView<double,Shape,DECAYED_TYPE_OF( strides ),MemorySpace_GlobalCudaRam>
//             dt( dev.raw, shape, strides, dev.memory_space );
//         run_parallel( shape.all_indices(), Doubler{}, dt ); // GlobalCudaRam -> dispatch picks CUDA
//         cudaStreamSynchronize( ExecutionSpace_Cuda{}.stream );

//         copy( ExecutionSpace_Cpu{}, Ptr<double,MemorySpace_CpuRam>( back ), dev, 4 );
//     } );

//     for ( int i = 0; i < 4; ++i )
//         CHECK( back[ i ] == host[ i ] * 2 );
// }

// // the target use case: a( ... ) = b( ... ) inside a GPU kernel, between two device tensors.
// TEST_CASE( "TensorView::operator= — a(...) = b(...) on GPU", "" ) {
//     double ha[ 4 ] = { 0, 0, 0, 0 }, hb[ 4 ] = { 5, 6, 7, 8 }, back[ 4 ] = { 0, 0, 0, 0 };
//     using Shape = AxisValues<int,1>;
//     Shape shape( Values(), 4 );
//     auto  strides = contiguous_strides<double>( shape );
//     using DT = TensorView<double,Shape,DECAYED_TYPE_OF( strides ),MemorySpace_GlobalCudaRam>;

//     MemorySpace_GlobalCudaRam{}.with_reservation<double>( 4, [&]( auto da ) {
//     MemorySpace_GlobalCudaRam{}.with_reservation<double>( 4, [&]( auto db ) {
//         copy( ExecutionSpace_Cuda{}, db, Ptr<double,MemorySpace_CpuRam>( hb ), 4 );
//         copy( ExecutionSpace_Cuda{}, da, Ptr<double,MemorySpace_CpuRam>( ha ), 4 );
//         cudaStreamSynchronize( ExecutionSpace_Cuda{}.stream );

//         DT a( da.raw, shape, strides, da.memory_space );
//         DT b( db.raw, shape, strides, db.memory_space );
//         run_parallel( shape.all_indices(), CopyElem{}, a, b ); // device tensors -> CUDA
//         cudaStreamSynchronize( ExecutionSpace_Cuda{}.stream );

//         copy( ExecutionSpace_Cpu{}, Ptr<double,MemorySpace_CpuRam>( back ), da, 4 );
//     } ); } );

//     for ( int i = 0; i < 4; ++i )
//         CHECK( back[ i ] == hb[ i ] );
// }

// #endif

TEST_CASE( "TensorView::make_accessible — CPU pass-through", "" ) {
    double data[ 4 ] = { 1, 2, 3, 4 };
    TensorView t( data, tuple( 4 ), tuple( sizeof( double ) ), MemorySpace_CpuRam{} );
    info( t );

    info( contiguous_strides<double>( tuple( 3, 4 ) ) );
    info( contiguous_strides<double>( tuple( 3, 4_c ) ) );
}

#ifdef __CUDACC__
TEST_CASE( "GPU tensor", "" ) {
    double data[ 4 ] = { 1, 2, 3, 4 };
    MemorySpace_GlobalCudaRam gpu_ram;
    gpu_ram.with_reservation<double>( 4, [&]( auto dev ) {
        copy( dev, Ptr<double,MemorySpace_CpuRam>( data ), 4 );
        cudaStreamSynchronize( ExecutionSpace_Cuda{}.stream );

        TensorView dt( dev.raw, tuple( 4 ), tuple( sizeof( double ) ), dev.memory_space );
        // run_parallel( shape.all_indices(), Doubler{}, dt ); // GlobalCudaRam -> dispatch picks CUDA
        // cudaStreamSynchronize( ExecutionSpace_Cuda{}.stream );

        // copy( ExecutionSpace_Cpu{}, Ptr<double,MemorySpace_CpuRam>( back ), dev, 4 );
        info( dt[ 0 ].value() );
    } );

    // TensorView t( data, tuple( 4 ), tuple( sizeof( double ) ), MemorySpace_CpuRam{} );
    // info( t );
}
#endif // __CUDACC__
