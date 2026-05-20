#include "../../src/cpp/sdot/support/hardware/Run.h"
// #include "../../src/cpp/sdot/support/hardware/Ptr.h"
#include "catch_main.h"

using namespace sdot;

// minimal batch_sizes: 1D range supporting the split iteration the runners use
struct Range {
    void for_each_item_split( int tid, int nb, auto &&cb ) const { for ( PI i = tid; i < n; i += nb ) cb( i ); }
    PI   size               () const { return n; }
    PI   n;
};

// // space-aware data argument (carries a memory kind on its pointer)
// struct IntBuf {
//     using             memory_kind = CpuRam;
//     Ptr<int, CpuRam> p;
// };

// // enriched functor: body + optional generic resource hints (methods)
// struct Sum : DefaultKernelTraits {
//     int *total;
//     auto nb_gpu_register_per_thread( const auto &... ) const { return Ct<int,16>(); } // compile-time, args ignored
//     void operator()( PI i, IntBuf buf, int bias ) const { *total += buf.p.as<int>()[ i ] + bias; }
// };

// // race-free functor for parallel runs: each index writes its own output slot
// struct Scale {
//     void operator()( PI i, IntBuf in, IntBuf out, int factor ) const { out.p.as<int>()[ i ] = in.p.as<int>()[ i ] * factor; }
// };

// // functor with a per-thread variable, wrapper style: per_thread() is called once per
// // activated thread; it sets up its variable and calls the continuation with the incoming
// // args plus the extra it injects.
// struct PerThreadProbe {
//     int *nb_calls; // counts per_thread invocations
//     int *out;      // out[i] = the per-thread value seen by operator()
//     void per_thread( auto ti, auto &&cont, const auto &...args ) const {
//         ( *nb_calls )++;
//         cont( args..., ti.global_id() + 100 ); // inject one extra per-thread argument
//     }
//     void operator()( PI i, int ptv ) const { out[ i ] = ptv; }
// };

// // functors for the kernel-traits checks
// struct KtPlain   { void operator()( PI ) const {} };                                                  // no hints -> defaults
// struct KtInherit : DefaultKernelTraits {                                                              // override one hint
//     auto nb_gpu_register_per_thread( const auto &... ) const { return Ct<int,64>(); }
//     void operator()( PI ) const {}
// };
// struct KtArgDep  {                                                                                    // hint depends on inputs
//     PI   local_cpu_memory_size( const auto &...args ) const { return ( PI( args ) + ... + PI( 0 ) ) * sizeof( int ); }
//     void operator()( PI ) const {}
// };

// TEST_CASE( "memory kinds", "" ) {
//     static_assert( host_accessible_v<CpuRam> );
//     static_assert( host_accessible_v<PinnedCpuRam> );
//     static_assert( !host_accessible_v<CudaGlobalRam> );
//     static_assert( device_accessible_v<CudaGlobalRam> );
//     static_assert( device_accessible_v<PinnedCpuRam> );
//     static_assert( !device_accessible_v<CpuRam> );
// }

// TEST_CASE( "kernel traits", "" ) {
//     // compile-time hints come back as Ct<...>
//     static_assert( decltype( kernel_nb_gpu_register_per_thread( KtPlain{}   ) )::value == 32 ); // default
//     static_assert( decltype( kernel_nb_gpu_register_per_thread( KtInherit{} ) )::value == 64 ); // overridden
//     static_assert( decltype( kernel_local_gpu_memory_size     ( KtInherit{} ) )::value == 0 );  // inherited default
//     // hint that depends on the run arguments (runtime value)
//     CHECK( kernel_local_cpu_memory_size( KtArgDep{}, 3, 5 ) == ( 3 + 5 ) * sizeof( int ) );
// }

// TEST_CASE( "execution space relations", "" ) {
//     static_assert( accessible_from_v<CpuExecutionSpace, CpuRam> );
//     static_assert( accessible_from_v<CpuExecutionSpace, PinnedCpuRam> );
//     static_assert( !accessible_from_v<CpuExecutionSpace, CudaGlobalRam> );
//     CHECK( same_memory_space( CpuExecutionSpace{}, CpuExecutionSpace{} ) );
// }

// TEST_CASE( "space inference from args", "" ) {
//     // IntBuf lives in HostRam -> host-accessible -> no device needed
//     static_assert( !arg_needs_device<IntBuf>() );
//     static_assert( !any_arg_needs_device<IntBuf, int>() ); // scalars are kind-less, ignored
//     // a device-kind data argument would force device execution
//     struct DevBuf { using memory_kind = CudaGlobalRam; Ptr<int, CudaGlobalRam> p; };
//     static_assert( arg_needs_device<DevBuf>() );
//     static_assert( any_arg_needs_device<IntBuf, DevBuf>() );
// }

// TEST_CASE( "host launch config", "" ) {
//     LaunchConfig lc = compute_launch( 1000, CpuExecutionSpace{}, KtPlain{} );
//     CHECK( lc.grid == 1000 );
//     CHECK( lc.block == 1 );
// }

// TEST_CASE( "run on host", "" ) {
//     int data[ 5 ] = { 1, 2, 3, 4, 5 };
//     IntBuf buf{ Ptr<int, CpuRam>( data ) };

//     SECTION( "run_sequential: accumulation + kind-less scalar pass-through" ) {
//         int total = 0;
//         run_sequential( Range{ 5 }, Sum{ {}, &total }, buf, 10 );
//         CHECK( total == ( 1 + 2 + 3 + 4 + 5 ) + 5 * 10 );
//     }

//     SECTION( "run_sequential: explicit host space" ) {
//         int total = 0;
//         run_sequential( CpuExecutionSpace{}, Range{ 5 }, Sum{ {}, &total }, buf, 0 );
//         CHECK( total == 15 );
//     }

//     SECTION( "run_parallel: race-free per-index writes" ) {
//         int out[ 5 ] = { 0, 0, 0, 0, 0 };
//         IntBuf obuf{ Ptr<int, CpuRam>( out ) };
//         run_parallel( Range{ 5 }, Scale{}, buf, obuf, 3 );
//         for ( int i = 0; i < 5; ++i )
//             CHECK( out[ i ] == data[ i ] * 3 );
//     }
// }

// TEST_CASE( "per-thread variable", "" ) {
//     // single thread: per_thread() called exactly once; the injected value reaches operator()
//     int nb_calls = 0;
//     int out[ 4 ] = { -1, -1, -1, -1 };
//     run_sequential( Range{ 4 }, PerThreadProbe{ &nb_calls, out } );
//     CHECK( nb_calls == 1 );           // one activated thread -> one per_thread() call
//     for ( int i = 0; i < 4; ++i )
//         CHECK( out[ i ] == 0 + 100 ); // global_id 0 + 100, injected once, seen by every index
// }

// NB: les classes locales (définies dans une fonction) ne peuvent pas avoir de
// templates membres. Comme per_thread/operator() sont des templates (auto&&), le
// functor doit vivre au niveau du namespace, pas dans le corps du TEST_CASE.
struct PerThreadProbe {
    // int *nb_calls; // counts per_thread invocations
    // int *out;      // out[i] = the per-thread value seen by operator()
    void per_thread( const auto &thread_info, const auto &/* list */, auto &&cont, auto &&...args ) const {
        // ( *nb_calls )++;
        cont( FORWARD( args )..., thread_info.global_id() + 100 ); // inject one extra per-thread argument
    }
    void operator()( int range_index, int a, int b, int thread_comp ) const {
        info( range_index, a, b, thread_comp );
    }
};

TEST_CASE( "per-thread variable", "" ) {
    run_sequential( Range{ 5 }, PerThreadProbe{}, 10, 20 );
    run_parallel( Range{ 5 }, PerThreadProbe{}, 10, 20 );
}

