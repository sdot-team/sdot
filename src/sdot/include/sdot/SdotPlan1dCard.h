#pragma once

// =====================================================================================
// `SdotPlan1d` ON A CARD, by device-wide launches ( what `sdot::sdot_plan_1d_card` runs, forward or adjoint ), instead
// of one block per item: a block sorts a few thousand diracs well, and leaves the card idle on one item of 1e7.
//
// The items ( the views of a sinogram, or a single transport ) by CHUNKS, as many as a memory budget holds
// ( `SDOT_OT1D_BUDGET_MB`, 1 GB by default, the one of `ContinuousOtPlan1d` ); each chunk of items:
//   1. the keys: ( item, order-preserving bits of the float position ), the exact positions on the side;
//   2. ONE device-wide radix sort of all the diracs of the chunk ( `cub::DeviceRadixSort` ), then the runs of equal
//      float keys sorted by the exact positions;
//   3. the sorted positions and weights, and the cumulated weights by a scan per item ( `ExclusiveSumByKey` );
//   4. the walks by chunks of `CHUNK` sorted diracs per thread, each from where it starts in the target
//      ( `SdotPlan1d::sweep_chunk`, the code of the host's work-items ), then the costs by a segmented reduction.
// The adjoint sorts again and runs the chunk walks of the gradients ( `position_grad_chunk`, then the two passes of
// `weight_value_grad_chunk` with a scan per item of the chunks' phi between them ).
// =====================================================================================

#if defined( __CUDACC__ )

#ifndef CCCL_DISABLE_CTK_COMPATIBILITY_CHECK
#define CCCL_DISABLE_CTK_COMPATIBILITY_CHECK    // ( see `ContinuousOtPlan1d.h` )
#endif
#include <cub/cub.cuh>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

namespace sdot {

namespace ot1d_dirac_card {

constexpr int BLOCK = 256;
constexpr int CHUNK = 16;   // sorted diracs per thread in the walks

inline int blocks_for( SI n ) { return int( ( n + BLOCK - 1 ) / BLOCK ); }

/// an order-preserving 32-bit key of a position, as a float
__device__ __forceinline__ std::uint32_t float_key( double x ) {
    const std::uint32_t u = __float_as_uint( float( x ) );
    return ( u & 0x80000000u ) ? ~u : ( u | 0x80000000u );
}

/// two chunks of the same item
struct SameInt {
    __host__ __device__ bool operator()( int a, int b ) const { return a == b; }
};

/// two sorted keys of the same item
struct SameItem {
    __host__ __device__ bool operator()( std::uint64_t a, std::uint64_t b ) const { return ( a >> 32 ) == ( b >> 32 ); }
};

/// the keys of the diracs of items [ v0, v0 + V ), their ids and exact positions; the chunks' offsets
template<class Plan,class Batch>
__global__ void __launch_bounds__( BLOCK ) keys_kernel( Plan plan, Batch batch, SI v0, SI V, SI n, SI nch, std::uint64_t *keys,
                                                     int *ids, double *xs, int *off ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q <= V )
        off[ q ] = int( q * nch );
    if ( q >= V * n )
        return;
    const SI v = q / n, i = q - v * n;
    const double x = double( plan( batch[ v0 + v ] ).src_dist.position( i ) );
    keys[ q ] = ( std::uint64_t( v ) << 32 ) | float_key( x );
    ids[ q ] = int( i );
    xs[ q ] = x;
}

/// thread q starts a run of equal SORTED keys and sorts it by the exact positions -- a few elements at most
__global__ void __launch_bounds__( BLOCK ) runs_kernel( SI V, SI n, const std::uint64_t *skeys, const double *xs, int *ids ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q >= V * n )
        return;
    const SI v = q / n, k = q - v * n;
    if ( k + 1 >= n || skeys[ q + 1 ] != skeys[ q ] || ( k && skeys[ q - 1 ] == skeys[ q ] ) )
        return;     // not the start of a run of two or more
    SI e = k + 2;
    while ( e < n && skeys[ v * n + e ] == skeys[ q ] )
        ++e;
    const double *xv = xs + v * n;
    int *iv = ids + v * n;
    for ( SI a = k + 1; a < e; ++a ) {
        const int id = iv[ a ];
        const double x = xv[ id ];
        SI b = a;
        for ( ; b > k && xv[ iv[ b - 1 ] ] > x; --b )
            iv[ b ] = iv[ b - 1 ];
        iv[ b ] = id;
    }
}

/// the sorted positions and weights
template<class Plan,class Batch>
__global__ void __launch_bounds__( BLOCK ) gather_kernel( Plan plan, Batch batch, SI v0, SI V, SI n, const int *sids, const double *xs,
                                                       double *spos, double *sw ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q >= V * n )
        return;
    const SI v = q / n, i = sids[ q ];
    spos[ q ] = xs[ v * n + i ];
    sw[ q ] = double( plan( batch[ v0 + v ] ).src_dist.weights( ::num_dirac = i ) );
}

/// what a walk reads: the sorted ids and positions of item v
struct SortedItem {
    const int *sids; const double *spos;
    __device__ SI ids( SI k ) const { return sids[ k ]; }
    __device__ double pos( SI k ) const { return spos[ k ]; }
};

/// thread ( v, c ) runs the forward walk of the sorted diracs [ c CHUNK, ( c + 1 ) CHUNK ) of item v
template<class Plan,class Batch>
__global__ void __launch_bounds__( BLOCK ) sweep_kernel( Plan plan, Batch batch, SI v0, SI V, SI n, SI nch, const int *sids, const double *spos,
                                                      const double *wscan, double *part ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q >= V * nch )
        return;
    const SI v = q / nch, c = q - v * nch;
    const SI lo = c * CHUNK, hi = lo + CHUNK < n ? lo + CHUNK : n;
    const SortedItem s{ sids + v * n, spos + v * n };
    auto item = plan( batch[ v0 + v ] );
    using TF = typename std::decay_t<decltype( item )>::TF;
    part[ q ] = double( item.sweep_chunk( [&]( SI k ) { return s.ids( k ); }, [&]( SI k ) { return TF( s.pos( k ) ); }, lo, hi, wscan[ v * n + lo ] ) );
}

/// the costs of the items, from the segmented reduction of the chunks'
template<class Plan,class Batch>
__global__ void __launch_bounds__( BLOCK ) cost_kernel( Plan plan, Batch batch, SI v0, SI V, const double *costs ) {
    const SI v = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( v >= V )
        return;
    auto item = plan( batch[ v0 + v ] );
    using TF = typename std::decay_t<decltype( item )>::TF;
    item.set_cost( TF( costs[ v ] ) );
}

/// the gradient of the target's values is accumulated by atomics: zeroed first
template<class Plan,class Grad,class Batch>
__global__ void __launch_bounds__( BLOCK ) zero_values_grad_kernel( Grad grad_plan, Batch batch, SI v0, SI V, SI nbin ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q >= V * nbin )
        return;
    const SI v = q / nbin;
    Plan::zero_target_grad( grad_plan( batch[ v0 + v ] ), q - v * nbin );
}

/// the walks of the adjoint, thread ( v, c ). PHASE 0: the positions' gradient; 1: the chunk's phi ( `phi` ); 2: the
/// weights' and values' gradients, from the scan of the phi per item ( `phi_pre` )
template<int PHASE,class Plan,class Grad,class Batch>
__global__ void __launch_bounds__( BLOCK ) grad_kernel( Plan plan, Grad grad_plan, Batch batch, SI v0, SI V, SI n, SI nch, const int *sids,
                                                     const double *spos, const double *wscan, double *phi, const double *phi_pre ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q >= V * nch )
        return;
    const SI v = q / nch, c = q - v * nch;
    const SI lo = c * CHUNK, hi = lo + CHUNK < n ? lo + CHUNK : n;
    const SortedItem s{ sids + v * n, spos + v * n };
    const auto item = plan( batch[ v0 + v ] );
    auto gitem = grad_plan( batch[ v0 + v ] );
    using TF = typename std::decay_t<decltype( item )>::TF;
    auto si = [&]( SI k ) { return s.ids( k ); };
    auto sp = [&]( SI k ) { return TF( s.pos( k ) ); };
    const double w_lo = wscan[ v * n + lo ];   // ( in double: the start of a chunk, read in float, misses its place )
    if constexpr ( PHASE == 0 )
        item.position_grad_chunk( gitem, si, sp, lo, hi, w_lo );
    else if constexpr ( PHASE == 1 )
        phi[ q ] = double( item.template weight_value_grad_chunk<false>( gitem, si, sp, lo, hi, w_lo, TF( 0 ), TF( 0 ) ) );
    else {
        const SI last = v * nch + nch - 1;
        item.template weight_value_grad_chunk<true>( gitem, si, sp, lo, hi, w_lo, TF( phi_pre[ q ] ), TF( phi_pre[ last ] + phi[ last ] ) );
    }
}

/// the item of each chunk, the keys of the scan of the phi
__global__ void __launch_bounds__( BLOCK ) chunk_items_kernel( SI V, SI nch, int *items ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q < V * nch )
        items[ q ] = int( q / nch );
}

} // namespace ot1d_dirac_card

/// `SdotPlan1d` on a card ( see the header ): the forward when `grad_plan` is `nullptr`, else the adjoint
template<class Queue,class Grad>
void sdot_plan_1d_card( const Queue &queue, const auto &batch, auto &&plan, Grad &&grad_plan, auto &allocator ) {
    using namespace ot1d_dirac_card;
    constexpr bool bwd = ! std::is_same_v<std::decay_t<Grad>,std::nullptr_t>;
    const SI nb_items = SI( batch.size() );
    if ( nb_items == 0 )
        return;
    const auto item0 = plan( batch[ 0 ] );
    const SI n = SI( item0.src_dist.weights.size() ), nbin = SI( item0.dst_dist.udp_nb_cells() );
    if ( n == 0 )
        return;
    const SI nch = ( n + CHUNK - 1 ) / CHUNK;

    // the items per chunk: a memory budget, and 32-bit offsets
    const SI per_dirac = 2 * sizeof( std::uint64_t ) + 2 * sizeof( int ) + 3 * sizeof( double );
    const SI per_item = n * per_dirac + nch * ( 3 * sizeof( double ) + sizeof( int ) );
    SI budget = SI( 1 ) << 30;
    if ( const char *e = std::getenv( "SDOT_OT1D_BUDGET_MB" ) )
        budget = SI( std::atol( e ) ) << 20;
    SI V = std::max<SI>( 1, std::min<SI>( nb_items, budget / per_item ) );
    V = std::min<SI>( V, std::max<SI>( 1, ( ( SI( 1 ) << 31 ) - 1 ) / n - 1 ) );
    const SI M = V * n, C = V * nch;
    int end_bit = 32;
    while ( ( SI( 1 ) << ( end_bit - 32 ) ) < V )
        ++end_bit;

    cudaStream_t stream = queue.stream;
    std::uint64_t *keys    = allocator.template raw<std::uint64_t>( 2 * M );
    int           *ids     = allocator.template raw<int>( 2 * M );
    double        *xs      = allocator.template raw<double>( M );
    double        *spos    = allocator.template raw<double>( M );
    double        *wscan   = allocator.template raw<double>( M );
    double        *part    = allocator.template raw<double>( C );    // the chunks' costs, or phi
    double        *pre     = allocator.template raw<double>( C );    // the items' costs, or the scan of phi
    int           *items   = allocator.template raw<int>( C );
    int           *off     = allocator.template raw<int>( V + 1 );
    double        *sw      = reinterpret_cast<double *>( keys );     // the sorted weights, where the unsorted keys were
    const std::uint64_t *skeys = keys + M;
    const int     *sids    = ids + M;

    std::size_t b_sort = 0, b_scan = 0, b_red = 0, b_phi = 0;
    cub::DeviceRadixSort::SortPairs( nullptr, b_sort, keys, keys + M, ids, ids + M, int( M ), 0, end_bit, stream );
    cub::DeviceScan::ExclusiveSumByKey( nullptr, b_scan, skeys, sw, wscan, int( M ), SameItem{}, stream );
    cub::DeviceSegmentedReduce::Sum( nullptr, b_red, part, pre, int( V ), off, off + 1, stream );
    cub::DeviceScan::ExclusiveSumByKey( nullptr, b_phi, items, part, pre, int( C ), SameInt{}, stream );
    const std::size_t b_tmp = std::max( std::max( b_sort, b_scan ), std::max( b_red, b_phi ) );
    char *tmp = allocator.template raw<char>( SI( b_tmp ) );
    if ( allocator.failed )
        return;

    const auto kplan  = kernel_form( queue, MutList(), plan );
    const auto kbatch = kernel_form( queue, MutList(), batch );
    using PlanT = std::decay_t<decltype( kplan )>;
    using BatchT = std::decay_t<decltype( kbatch )>;
    for ( SI v0 = 0; v0 < nb_items; v0 += V ) {
        const SI Vc = std::min<SI>( V, nb_items - v0 ), Mc = Vc * n, Cc = Vc * nch;
        launch_kernel( queue, &keys_kernel<PlanT,BatchT>, blocks_for( std::max( Mc, Vc + 1 ) ), BLOCK, 0, kplan, kbatch, v0, Vc, n, nch, keys, ids, xs, off );
        std::size_t bytes = b_tmp;
        cub::DeviceRadixSort::SortPairs( tmp, bytes, keys, keys + M, ids, ids + M, int( Mc ), 0, end_bit, stream );
        launch_kernel( queue, &runs_kernel, blocks_for( Mc ), BLOCK, 0, Vc, n, skeys, ( const double * )xs, ids + M );
        launch_kernel( queue, &gather_kernel<PlanT,BatchT>, blocks_for( Mc ), BLOCK, 0, kplan, kbatch, v0, Vc, n, sids, ( const double * )xs, spos, sw );
        bytes = b_tmp;
        cub::DeviceScan::ExclusiveSumByKey( tmp, bytes, skeys, sw, wscan, int( Mc ), SameItem{}, stream );
        if constexpr ( bwd ) {
            const auto kgrad = kernel_form( queue, MutList(), grad_plan );
            using GradT = std::decay_t<decltype( kgrad )>;
            using ItemT = std::decay_t<decltype( kplan( kbatch[ 0 ] ) )>;
            constexpr bool values_grad = ItemT::template target_grad_wanted<std::decay_t<decltype( kgrad( kbatch[ 0 ] ) )>>;
            constexpr bool weight_value_grad = DECAYED_TYPE_OF( kgrad( kbatch[ 0 ] ).src_dist.weights )::is_valid || values_grad;
            launch_kernel( queue, &grad_kernel<0,PlanT,GradT,BatchT>, blocks_for( Cc ), BLOCK, 0, kplan, kgrad, kbatch, v0, Vc, n, nch, sids,
                           ( const double * )spos, ( const double * )wscan, part, ( const double * )pre );
            if constexpr ( weight_value_grad ) {
                if constexpr ( values_grad )
                    launch_kernel( queue, &zero_values_grad_kernel<ItemT,GradT,BatchT>, blocks_for( Vc * nbin ), BLOCK, 0, kgrad, kbatch, v0, Vc, nbin );
                launch_kernel( queue, &grad_kernel<1,PlanT,GradT,BatchT>, blocks_for( Cc ), BLOCK, 0, kplan, kgrad, kbatch, v0, Vc, n, nch, sids,
                               ( const double * )spos, ( const double * )wscan, part, ( const double * )pre );
                launch_kernel( queue, &chunk_items_kernel, blocks_for( Cc ), BLOCK, 0, Vc, nch, items );
                bytes = b_tmp;
                cub::DeviceScan::ExclusiveSumByKey( tmp, bytes, ( const int * )items, ( const double * )part, pre, int( Cc ), SameInt{}, stream );
                launch_kernel( queue, &grad_kernel<2,PlanT,GradT,BatchT>, blocks_for( Cc ), BLOCK, 0, kplan, kgrad, kbatch, v0, Vc, n, nch, sids,
                               ( const double * )spos, ( const double * )wscan, part, ( const double * )pre );
            }
        } else {
            launch_kernel( queue, &sweep_kernel<PlanT,BatchT>, blocks_for( Cc ), BLOCK, 0, kplan, kbatch, v0, Vc, n, nch, sids,
                           ( const double * )spos, ( const double * )wscan, part );
            bytes = b_tmp;
            cub::DeviceSegmentedReduce::Sum( tmp, bytes, ( const double * )part, pre, int( Vc ), ( const int * )off, ( const int * )off + 1, stream );
            launch_kernel( queue, &cost_kernel<PlanT,BatchT>, blocks_for( Vc ), BLOCK, 0, kplan, kbatch, v0, Vc, ( const double * )pre );
        }
    }
}

} // namespace sdot

#endif
