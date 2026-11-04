#pragma once

// =====================================================================================
// `ContinuousOtPlan1d` between TWO IMAGES ( `sdot::image_ot_1d`, what the call's handler runs ): pixels of degree 0 or 1
// on both sides. No sort: the pixels of each image are already in increasing x, and the generic transport
// ( `ot1d::transport_1d` ) merges the two streams of pieces, O( pixels of both ) per item. The forward only.
//   * host: the items split between the threads;
//   * card: device-wide launches -- the masses of the pixels of both images, their scans per item, then one thread per
//     chunk of `CHUNK` source pixels, which finds where it starts in the target ( a binary search on the target's
//     cumulated masses ) and runs the transport from there; the costs by a segmented reduction.
// =====================================================================================

#include <sdot/ot1d/transport_1d.h>

#include <algorithm>
#include <type_traits>

namespace sdot {

namespace ot1d_image {

/// the pixels of a 1D image ( with its defaults ) as a stream of pieces, in physical coordinates, its density divided by
/// its mass ( `transport_1d` wants mass 1 )
template<class Img>
struct ImagePieces1d {
    static constexpr bool affine = std::decay_t<Img>::is_polynomial;
    using TF = typename std::decay_t<Img>::TF;                // ( float images: the transport in single precision )
    using Piece = ot1d::PolyPiece1d<TF, affine ? 1 : 0>;

    const Img &img;
    double inv_mass;
    SI c = 0;
    SI end = -1;    // ( -1: the last pixel )

    HD bool next( Piece &p ) {
        if ( c >= ( end < 0 ? img.udp_nb_cells() : end ) )
            return false;
        const TF sc = TF( img.frame( 0, 0 ) ), of = TF( img.origin( 0 ) );
        p.x0 = of + sc * TF( img.knots( 0, c ) );
        p.x1 = of + sc * TF( img.knots( 0, c + 1 ) );
        p.c[ 0 ] = TF( TF( img.udp_left( c ) ) * inv_mass );
        if constexpr ( affine )
            p.c[ 1 ] = TF( TF( img.udp_slope( c ) ) * inv_mass );
        ++c;
        return true;
    }
};

template<class Img>
HD double mass_of( const Img &img ) {
    double m = 0;
    for ( SI c = 0; c < img.udp_nb_cells(); ++c )
        m += double( img.udp_cell_mass( c ) );
    return m;
}

/// the transport of one item: W2^2 between its two images. Against an affine target the map is not a polynomial ( its
/// cross term with the source integrates a square root of a quadratic: a closed form with an arcsinh, unstable when a
/// slope vanishes ): 7 points of Gauss-Legendre per merged piece, below the rounding at a pixel's scale
HD double item_cost( auto &&item ) {
    double cost = 0;
    item.src_dist.with_defaults( [&]( auto &&src ) {
        item.dst_dist.with_defaults( [&]( auto &&dst ) {
            ImagePieces1d<DECAYED_TYPE_OF( src )> s{ src, 1 / mass_of( src ) };
            ImagePieces1d<DECAYED_TYPE_OF( dst )> d{ dst, 1 / mass_of( dst ) };
            cost = double( ot1d::transport_1d<7>( s, d, []( int, const auto & ) {} ).cost );
        } );
    } );
    return cost;
}

HD void write_item( auto &&item, double cost ) {
    using TF = typename std::decay_t<decltype( item )>::TF;
    item.nb_diracs.set( item.src_dist.udp_nb_cells() );
    item.cost = TF( cost );
}

} // namespace ot1d_image

/// host: the items split between the threads
template<class Grad>
void image_ot_1d( const CpuQueue &queue, const auto &batch, auto &&plan, Grad &&, auto & ) {
    const SI nb_items = SI( batch.size() );
    if ( nb_items == 0 )
        return;
    const int nt = int( std::min<SI>( std::max( queue.nb_workers(), 1 ), nb_items ) );
    auto body = [&]( int t ) {
        for ( SI v = nb_items * t / nt; v < nb_items * ( t + 1 ) / nt; ++v ) {
            auto item = plan( batch[ v ] );
            ot1d_image::write_item( item, ot1d_image::item_cost( item ) );
        }
    };
    if ( nt <= 1 )
        body( 0 );
    else
        queue.run_threads( nt, body );
}

#if defined( __CUDACC__ )
namespace ot1d_image_card {

constexpr int BLOCK = 256;
constexpr int CHUNK = 64;   // source pixels per thread

inline int blocks_for( SI n ) { return int( ( n + BLOCK - 1 ) / BLOCK ); }

/// the masses of the pixels of the items' source ( `SIDE` 0 ) or target ( 1 ), and the item of each ( the keys of the scans )
template<int SIDE,class Plan,class Batch>
__global__ void __launch_bounds__( BLOCK ) masses_kernel( Plan plan, Batch batch, SI V, SI P, double *mass, int *key ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q >= V * P )
        return;
    const SI v = q / P;
    auto item = plan( batch[ v ] );
    auto at = [&]( auto &&img ) { mass[ q ] = double( img.udp_cell_mass( q - v * P ) ); };
    if constexpr ( SIDE == 0 ) item.src_dist.with_defaults( at );
    else                       item.dst_dist.with_defaults( at );
    key[ q ] = int( v );
}

/// thread ( v, c ) transports the source pixels [ c CHUNK, ( c + 1 ) CHUNK ) of item v, from where their mass starts in the
/// target ( `scum`, `dcum`: the exclusive scans of the pixels' masses )
template<class Plan,class Batch>
__global__ void __launch_bounds__( BLOCK ) chunks_kernel( Plan plan, Batch batch, SI V, SI Ps, SI Pd, SI nch, const double *smass,
                                                       const double *scum, const double *dmass, const double *dcum, double *part, int *off ) {
    const SI q = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( q <= V )
        off[ q ] = int( q * nch );
    if ( q >= V * nch )
        return;
    const SI v = q / nch, c = q - v * nch;
    const SI c0 = c * CHUNK, c1 = c0 + CHUNK < Ps ? c0 + CHUNK : Ps;
    const double *sc = scum + v * Ps, *dc = dcum + v * Pd;
    const double ts = sc[ Ps - 1 ] + smass[ v * Ps + Ps - 1 ], td = dc[ Pd - 1 ] + dmass[ v * Pd + Pd - 1 ];
    const double F_lo = sc[ c0 ] / ts;
    SI a = 0, b = Pd - 1;                       // the first target pixel whose cumulated mass exceeds F_lo
    while ( a < b ) {
        const SI mid = ( a + b ) / 2;
        if ( ( mid + 1 < Pd ? dc[ mid + 1 ] : td ) / td > F_lo ) b = mid; else a = mid + 1;
    }
    const double dd0 = F_lo - dc[ a ] / td;
    auto item = plan( batch[ v ] );
    double cost = 0;
    item.src_dist.with_defaults( [&]( auto &&src ) {
        item.dst_dist.with_defaults( [&]( auto &&dst ) {
            ot1d_image::ImagePieces1d<DECAYED_TYPE_OF( src )> s{ src, 1 / ts, c0, c1 };
            ot1d_image::ImagePieces1d<DECAYED_TYPE_OF( dst )> d{ dst, 1 / td, a, -1 };
            cost = double( ot1d::transport_1d<7>( s, d, []( int, const auto & ) {}, dd0 > 0 ? dd0 : 0.0 ).cost );
        } );
    } );
    part[ q ] = cost;
}

template<class Plan,class Batch>
__global__ void __launch_bounds__( BLOCK ) write_kernel( Plan plan, Batch batch, SI V, const double *costs ) {
    const SI v = SI( blockIdx.x ) * BLOCK + threadIdx.x;
    if ( v < V )
        ot1d_image::write_item( plan( batch[ v ] ), costs[ v ] );
}

struct SameInt {
    __host__ __device__ bool operator()( int a, int b ) const { return a == b; }
};

} // namespace ot1d_image_card

/// card: device-wide launches ( see the header )
template<class Queue,class Grad> requires ( ! std::is_same_v<Queue,CpuQueue> )
void image_ot_1d( const Queue &queue, const auto &batch, auto &&plan, Grad &&, auto &allocator ) {
    using namespace ot1d_image_card;
    const SI V = SI( batch.size() );
    if ( V == 0 )
        return;
    const auto item0 = plan( batch[ 0 ] );
    const SI Ps = item0.src_dist.udp_nb_cells(), Pd = item0.dst_dist.udp_nb_cells();
    const SI nch = ( Ps + CHUNK - 1 ) / CHUNK;
    cudaStream_t stream = queue.stream;
    double *smass = allocator.template raw<double>( V * Ps ), *scum = allocator.template raw<double>( V * Ps );
    double *dmass = allocator.template raw<double>( V * Pd ), *dcum = allocator.template raw<double>( V * Pd );
    int    *skey  = allocator.template raw<int>( V * Ps ), *dkey = allocator.template raw<int>( V * Pd );
    double *part  = allocator.template raw<double>( V * nch ), *costs = allocator.template raw<double>( V );
    int    *off   = allocator.template raw<int>( V + 1 );
    std::size_t b_s = 0, b_d = 0, b_r = 0;
    cub::DeviceScan::ExclusiveSumByKey( nullptr, b_s, skey, smass, scum, int( V * Ps ), SameInt{}, stream );
    cub::DeviceScan::ExclusiveSumByKey( nullptr, b_d, dkey, dmass, dcum, int( V * Pd ), SameInt{}, stream );
    cub::DeviceSegmentedReduce::Sum( nullptr, b_r, part, costs, int( V ), off, off + 1, stream );
    const std::size_t b_tmp = std::max( b_s, std::max( b_d, b_r ) );
    char *tmp = allocator.template raw<char>( SI( b_tmp ) );
    if ( allocator.failed )
        return;

    const auto kplan  = kernel_form( queue, MutList(), plan );
    const auto kbatch = kernel_form( queue, MutList(), batch );
    using PlanT = std::decay_t<decltype( kplan )>;
    using BatchT = std::decay_t<decltype( kbatch )>;
    launch_kernel( queue, &masses_kernel<0,PlanT,BatchT>, blocks_for( V * Ps ), BLOCK, 0, kplan, kbatch, V, Ps, smass, skey );
    launch_kernel( queue, &masses_kernel<1,PlanT,BatchT>, blocks_for( V * Pd ), BLOCK, 0, kplan, kbatch, V, Pd, dmass, dkey );
    std::size_t bytes = b_tmp;
    cub::DeviceScan::ExclusiveSumByKey( tmp, bytes, ( const int * )skey, ( const double * )smass, scum, int( V * Ps ), SameInt{}, stream );
    bytes = b_tmp;
    cub::DeviceScan::ExclusiveSumByKey( tmp, bytes, ( const int * )dkey, ( const double * )dmass, dcum, int( V * Pd ), SameInt{}, stream );
    launch_kernel( queue, &chunks_kernel<PlanT,BatchT>, blocks_for( std::max( V * nch, V + 1 ) ), BLOCK, 0, kplan, kbatch, V, Ps, Pd, nch,
                   ( const double * )smass, ( const double * )scum, ( const double * )dmass, ( const double * )dcum, part, off );
    bytes = b_tmp;
    cub::DeviceSegmentedReduce::Sum( tmp, bytes, ( const double * )part, costs, int( V ), ( const int * )off, ( const int * )off + 1, stream );
    launch_kernel( queue, &write_kernel<PlanT,BatchT>, blocks_for( V ), BLOCK, 0, kplan, kbatch, V, ( const double * )costs );
}
#endif

}
