#pragma once

#include <sdot/generated/aggregates/ContinuousOtPlan1d.h>
#include <sdot/ot1d/tents_1d.h>
#include <loom/support/common_macros.h>

#include <loom/support/kernels/CpuQueue.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#if defined( __CUDACC__ )
// the pip wheels of CUDA may pair an nvcc with the runtime headers of the next minor version ( 13.3 and 13.4 ): CCCL
// refuses that by default, though only the major version matters to what is used here
#ifndef CCCL_DISABLE_CTK_COMPATIBILITY_CHECK
#define CCCL_DISABLE_CTK_COMPATIBILITY_CHECK
#endif
#include <cub/cub.cuh>
#if CCCL_MAJOR_VERSION >= 3       // the iterators moved to libcu++ ( thrust's are deprecated )
#include <cuda/iterator>
#define SDOT_OT1D_ITER cuda
#else
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/tabulate_output_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#define SDOT_OT1D_ITER thrust
#endif
#endif

// =====================================================================================
// THE TRANSPORT OF ONE ITEM ( a view ) BY A WORK-GROUP, the same code on the host ( a group of one ) and on the card:
//
//   1. the tents' total mass ( a reduction );
//   2. an LSD radix sort of the 3 n breakpoints ( 32-bit float keys, 4 passes of 8 bits: per-thread histograms, a scan
//      per digit, a stable scatter ), and the inverse permutation;
//   3. the sorted positions and slope jumps, then three exclusive scans: the slope after each breakpoint, the density at
//      it, the cumulated mass before it;
//   4. the sweep in CHUNKS: each thread takes a contiguous range of pieces, finds where it starts in the target by a
//      binary search on the target's cumulated masses, runs the generic transport ( `ot1d::transport_1d`, from that
//      offset ) and records its LOCAL sums at each breakpoint; a scan of the chunk totals makes them global;
//   5. per tent, its moments from the sums at its three breakpoints ( `ot1d::tent_moments` ), then the outputs or the
//      adjoint's atomics.
// Every scratch row is this group's: never one per view.
//
// THE DRIVERS ( `continuous_ot_1d`, what the call's handler runs, forward or adjoint ):
//   * host: the views split between the threads, each a group of one with its own rows ( the code above );
//   * card: the views by CHUNKS ( as many as a memory budget holds ), each chunk by chained device-wide launches --
//     the breakpoints' keys, a segmented sort ( `cub::DeviceSegmentedSort` ), ONE segmented scan that gives the slopes,
//     the densities and the cumulated masses together ( the composition of the pieces' affine maps, `Flow1d` ), the
//     sweep by chunks of `OT_CHUNK` pieces per thread ( a binary search for where each starts in the target ), a
//     segmented scan of the chunks' totals, then one thread per tent over the views of the chunk ( the moments, the
//     outputs or the gradients accumulated in registers: one atomic per tent and per chunk of views ).
// =====================================================================================

namespace sdot {

namespace ot1d_coop {

/// the contiguous chunk [ lo, hi ) of `n` items of thread `t` among `nb`
HD void chunk( SI n, int t, int nb, SI &lo, SI &hi ) {
    lo = ( n * SI( t ) ) / nb;
    hi = ( n * SI( t + 1 ) ) / nb;
}

/// an order-preserving 32-bit key of a float
HD std::uint32_t float_key( double x ) {
    const float f = float( x );
    std::uint32_t u;
#if defined( __CUDA_ARCH__ )
    u = __float_as_uint( f );
#else
    std::memcpy( &u, &f, 4 );
#endif
    return ( u & 0x80000000u ) ? ~u : ( u | 0x80000000u );
}

/// exclusive scan of K doubles across the group: `part` ( this thread's totals ) becomes its offsets; `total` gets the
/// grand totals. `gscan`: [ >= local_size + 1 ][ >= K ] doubles of scratch.
template<int K>
HD void group_exclusive_scan( double *part, double *total, auto &&gscan, int t, int nb, auto &&group ) {
    for ( int j = 0; j < K; ++j )
        gscan( t, j ) = part[ j ];
    group_barrier( group );
    if ( t == 0 ) {
        for ( int j = 0; j < K; ++j ) {
            double run = 0;
            for ( int u = 0; u < nb; ++u ) {
                const double v = gscan( u, j );
                gscan( u, j ) = run;
                run += v;
            }
            gscan( nb, j ) = run;
        }
    }
    group_barrier( group );
    for ( int j = 0; j < K; ++j ) {
        part[ j ] = gscan( t, j );
        if ( total )
            total[ j ] = gscan( nb, j );
    }
    group_barrier( group );
}

/// stable LSD radix sort of `n` ( key, id ) pairs, in place in ( `ka`, `ia` ), `kb` / `ib` the ping-pong rows, `hist`
/// [ >= local_size + 1 ][ 256 ] ints. 4 passes: the result ends back in ( `ka`, `ia` ).
HD void group_radix_sort( SI n, auto &&ka, auto &&ia, auto &&kb, auto &&ib, auto &&hist, int t, int nb, auto &&group ) {
    SI lo, hi;
    chunk( n, t, nb, lo, hi );
    for ( int pass = 0; pass < 4; ++pass ) {
        const int shift = 8 * pass;
        auto digit = [&]( std::uint32_t k ) { return int( ( k >> shift ) & 0xffu ); };
        const bool even = ( pass % 2 ) == 0;
        auto rd_k = [&]( SI k ) { return std::uint32_t( even ? ka( k ) : kb( k ) ); };
        auto rd_i = [&]( SI k ) { return int( even ? ia( k ) : ib( k ) ); };
        for ( int d = 0; d < 256; ++d )
            hist( t, d ) = 0;
        for ( SI k = lo; k < hi; ++k )
            hist( t, digit( rd_k( k ) ) ) = int( hist( t, digit( rd_k( k ) ) ) ) + 1;
        group_barrier( group );
        for ( int d = t; d < 256; d += nb ) {     // per digit: the offsets of the threads, and the digit's total
            int run = 0;
            for ( int u = 0; u < nb; ++u ) {
                const int c = int( hist( u, d ) );
                hist( u, d ) = run;
                run += c;
            }
            hist( nb, d ) = run;
        }
        group_barrier( group );
        if ( t == 0 ) {
            int run = 0;
            for ( int d = 0; d < 256; ++d ) {
                const int c = int( hist( nb, d ) );
                hist( nb, d ) = run;
                run += c;
            }
        }
        group_barrier( group );
        for ( SI k = lo; k < hi; ++k ) {
            const std::uint32_t key = rd_k( k );
            const int d = digit( key );
            const int dst = int( hist( nb, d ) ) + int( hist( t, d ) );
            hist( t, d ) = int( hist( t, d ) ) + 1;
            if ( even ) { kb( dst ) = std::int32_t( key ); ib( dst ) = rd_i( k ); }
            else        { ka( dst ) = std::int32_t( key ); ia( dst ) = rd_i( k ); }
        }
        group_barrier( group );
    }
}

} // namespace ot1d_coop

/// this group's scratch rows, bundled ( built by the call: `sdot::ContinuousOtRows{ scratch.key_a( group_index ), ... }` )
template<class KA,class IA,class KB,class IB,class INV,class POS,class SL,class DE,class SU,class CD,class HI,class GS>
struct ContinuousOtRows {
    KA  key_a;   IA id_a;   KB key_b;   IB id_b;     // the sort's ping-pong rows ( 3 n )
    INV inv;                                         // breakpoint id -> sorted rank ( 3 n )
    POS pos;     SL slope;  DE dens;                 // per sorted breakpoint ( 3 n )
    SU  sums;                                        // per sorted breakpoint: P0, P1, P2, phi ( 3 n x 4 )
    CD  cdf;                                         // the target's cumulated masses ( bins + 1 )
    HI  hist;                                        // the sort's histograms ( local_size + 1 ) x 256
    GS  gscan;                                       // the scans' row ( local_size + 1 ) x 8
};

SDOT_TEMPLATE_DECL_FOR_ContinuousOtPlan1d
struct ContinuousOtPlan1d {
    SDOT_ATTRIBUTES_OF_ContinuousOtPlan1d

    using TF = DECAYED_TYPE_OF( cost )::TF;

    /// the transport of THIS item by the group ( see the header ); `on_tent( i, mass_i, moments_i )` per tent, called
    /// by the thread that owns tent i. Returns the cost ( on every thread ).
    HD double run( auto &&rows, int t, int nb, auto &&group, auto &&on_tent ) const {
        using namespace ot1d_coop;
        const SI n = src_dist.nb_tents(), nbp = 3 * n;
        SI lo, hi;

        // 1. the total mass
        double tot[ 1 ] = { 0 }, total[ 1 ];
        chunk( n, t, nb, lo, hi );
        for ( SI i = lo; i < hi; ++i )
            tot[ 0 ] += double( src_dist.mass( i ) );
        group_exclusive_scan<1>( tot, total, rows.gscan, t, nb, group );

        // 2. the sort of the breakpoints
        auto bp = [&]( SI id, double &x, double &ds ) {
            const SI i = id / 3, kind = id % 3;
            const double s = double( src_dist.position( i ) ), w = double( src_dist.half_width( i ) );
            const double k = double( src_dist.mass( i ) ) / total[ 0 ] / ( w * w );
            x = s + double( kind - 1 ) * w;
            ds = kind == 1 ? -2 * k : k;
        };
        chunk( nbp, t, nb, lo, hi );
        for ( SI k = lo; k < hi; ++k ) {
            double x, ds;
            bp( k, x, ds );
            rows.key_a( k ) = std::int32_t( float_key( x ) );
            rows.id_a( k ) = std::int32_t( k );
        }
        group_barrier( group );
        group_radix_sort( nbp, rows.key_a, rows.id_a, rows.key_b, rows.id_b, rows.hist, t, nb, group );
        if ( t == 0 ) {     // the keys are floats: the breakpoints of a run of equal keys, by their exact positions
            for ( SI r = 0; r < nbp; ) {
                SI e = r + 1;
                while ( e < nbp && rows.key_a( e ) == rows.key_a( r ) )
                    ++e;
                for ( SI a = r + 1; a < e; ++a ) {
                    const std::int32_t id = rows.id_a( a );
                    double x, ds;
                    bp( id, x, ds );
                    SI b = a;
                    for ( ; b > r; --b ) {
                        double y, dy;
                        bp( rows.id_a( b - 1 ), y, dy );
                        if ( ! ( y > x ) )
                            break;
                        rows.id_a( b ) = rows.id_a( b - 1 );
                    }
                    rows.id_a( b ) = id;
                }
                r = e;
            }
        }
        group_barrier( group );

        // 3. the sorted positions, the slopes, the densities, the cumulated masses
        double part[ 1 ] = { 0 }, dummy[ 1 ];
        for ( SI k = lo; k < hi; ++k ) {
            const SI id = rows.id_a( k );
            rows.inv( id ) = std::int32_t( k );
            double x, ds;
            bp( id, x, ds );
            rows.pos( k ) = x;
            rows.slope( k ) = ds;     // the jump, for now
            part[ 0 ] += ds;
        }
        group_exclusive_scan<1>( part, dummy, rows.gscan, t, nb, group );
        double run = part[ 0 ];
        for ( SI k = lo; k < hi; ++k ) {
            run += double( rows.slope( k ) );
            rows.slope( k ) = run;    // the slope after breakpoint k
        }
        group_barrier( group );
        auto dx = [&]( SI k ) {
            if ( k + 1 >= nbp ) return 0.0;
            const double d = double( rows.pos( k + 1 ) ) - double( rows.pos( k ) );
            return d > 0 ? d : 0.0;
        };
        double fm[ 2 ] = { 0, 0 };    // the density increment and the mass of the chunk... ( density first )
        for ( SI k = lo; k < hi; ++k )
            fm[ 0 ] += double( rows.slope( k ) ) * dx( k );
        group_exclusive_scan<1>( fm, dummy, rows.gscan, t, nb, group );
        run = fm[ 0 ];
        for ( SI k = lo; k < hi; ++k ) {
            rows.dens( k ) = run > 0 ? run : 0.0;   // the density at breakpoint k
            run += double( rows.slope( k ) ) * dx( k );
        }
        group_barrier( group );
        double ms[ 1 ] = { 0 }, mtot[ 1 ];
        auto piece_mass = [&]( SI k ) { const double d = dx( k ); return double( rows.dens( k ) ) * d + 0.5 * double( rows.slope( k ) ) * d * d; };
        for ( SI k = lo; k < hi; ++k )
            ms[ 0 ] += piece_mass( k );
        group_exclusive_scan<1>( ms, mtot, rows.gscan, t, nb, group );
        const double F_lo = ms[ 0 ];   // the mass before this thread's first piece

        // 4. the sweep, by chunks
        double sums[ 5 ] = { 0, 0, 0, 0, 0 }, gsum[ 5 ];
        dst_dist.with_defaults( [&]( auto &&img ) {
            const SI nbin = img.values.size();
            const double sc = double( img.frame( 0, 0 ) ), of = double( img.origin( 0 ) );
            auto bin_mass = [&]( SI j ) { return sc * ( double( img.knots( j + 1 ) ) - double( img.knots( j ) ) ) * double( img.values[ j ] ); };
            if ( t == 0 ) {           // the target's cumulated masses ( few bins: one thread )
                double c = 0;
                for ( SI j = 0; j < nbin; ++j ) { rows.cdf( j ) = c; c += bin_mass( j ); }
                rows.cdf( nbin ) = c;
            }
            group_barrier( group );
            const double dtot = double( rows.cdf( nbin ) );
            // where the chunk starts in the target: the first bin whose cumulated mass ( normalized ) exceeds F_lo
            SI a = 0, b = nbin - 1;
            while ( a < b ) {
                const SI mid = ( a + b ) / 2;
                if ( double( rows.cdf( mid + 1 ) ) / dtot > F_lo ) b = mid; else a = mid + 1;
            }
            const double dd0 = F_lo - double( rows.cdf( a ) ) / dtot;
            auto dst = ot1d::func_stream0( [&]( long j ) {
                ot1d::PolyPiece1d<double,0> p;
                p.x0 = of + sc * double( img.knots( j ) );
                p.x1 = of + sc * double( img.knots( j + 1 ) );
                p.c[ 0 ] = double( img.values[ j ] ) / dtot;
                return p;
            }, long( nbin ), long( a ) );
            auto src = ot1d::func_stream1( [&]( long k ) {
                ot1d::PolyPiece1d<double,1> p;
                p.x0 = double( rows.pos( k ) );
                p.x1 = p.x0 + dx( k );
                p.c[ 0 ] = double( rows.dens( k ) );
                p.c[ 1 ] = k + 1 < nbp ? double( rows.slope( k ) ) : 0.0;
                p.event = int( k );
                return p;
            }, long( lo ), long( hi ) );
            const ot1d::Sums1d S = ot1d::transport_1d<3>( src, dst, [&]( int k, const ot1d::Sums1d &s ) {
                rows.sums( k, 0 ) = s.P0; rows.sums( k, 1 ) = s.P1; rows.sums( k, 2 ) = s.P2; rows.sums( k, 3 ) = s.phi;
            }, dd0 > 0 ? dd0 : 0.0 );
            sums[ 0 ] = S.P0; sums[ 1 ] = S.P1; sums[ 2 ] = S.P2; sums[ 3 ] = S.phi; sums[ 4 ] = S.cost;
        } );
        group_exclusive_scan<5>( sums, gsum, rows.gscan, t, nb, group );
        for ( SI k = lo; k < hi; ++k )
            for ( int j = 0; j < 4; ++j )
                rows.sums( k, j ) = double( rows.sums( k, j ) ) + sums[ j ];
        group_barrier( group );

        // 5. per tent
        auto at = [&]( SI id ) {
            const SI k = rows.inv( id );
            ot1d::Sums1d s;
            s.P0 = double( rows.sums( k, 0 ) ); s.P1 = double( rows.sums( k, 1 ) ); s.P2 = double( rows.sums( k, 2 ) ); s.phi = double( rows.sums( k, 3 ) );
            return s;
        };
        chunk( n, t, nb, lo, hi );
        for ( SI i = lo; i < hi; ++i ) {
            const auto m = ot1d::tent_moments( double( src_dist.position( i ) ), double( src_dist.half_width( i ) ),
                                               at( 3 * i ), at( 3 * i + 1 ), at( 3 * i + 2 ) );
            on_tent( i, double( src_dist.mass( i ) ) / total[ 0 ], m );
        }
        group_barrier( group );   // the rows are reused by the next item of this group
        return gsum[ 4 ];
    }

    /// the forward: the cost, and the barycenters / potentials ( the mass derivative ) when they are outputs
    HD void update_outputs( auto &&rows, int t, int nb, auto &&group ) {
        const double c = run( rows, t, nb, group, [&]( SI i, double mi, const ot1d::TentMoments &m ) {
            if constexpr ( DECAYED_TYPE_OF( barycenters )::is_valid )
                barycenters( ::num_dirac = i ) = TF( m.b );
            if constexpr ( DECAYED_TYPE_OF( potentials )::is_valid )
                potentials( ::num_dirac = i ) = TF( m.phi );
        } );
        if ( t == 0 ) {
            nb_diracs.set( src_dist.nb_diracs );
            cost = TF( c );
        }
    }

    /// the forward and, in the same pass, the gradient of the SUMMED cost over the items ( `sg`: `points` and
    /// `half_widths` views, shared by the items, atomics ) -- what `value_and_grad` of `cost.sum()` needs, without the
    /// adjoint's second transport
    HD void update_outputs_sum_grad( auto &&rows, int t, int nb, auto &&group, auto &&sg ) {
        const double c = run( rows, t, nb, group, [&]( SI i, double mi, const ot1d::TentMoments &m ) {
            write_tent( i, m );
            const double s = double( src_dist.position( i ) ), w = double( src_dist.half_width( i ) );
            src_dist.add_position_grad( sg, i, TF( 2 * mi * ( s - m.b ) ) );
            src_dist.add_half_width_grad( sg, i, TF( 2 * mi / w * ( w * w / 6 - m.Q ) ) );
        } );
        if ( t == 0 )
            write_cost( c );
    }

    /// the outputs of tent i ( the card's per-tent kernel ), and the cost of the item
    HD void write_tent( SI i, const ot1d::TentMoments &m ) {
        if constexpr ( DECAYED_TYPE_OF( barycenters )::is_valid )
            barycenters( ::num_dirac = i ) = TF( m.b );
        if constexpr ( DECAYED_TYPE_OF( potentials )::is_valid )
            potentials( ::num_dirac = i ) = TF( m.phi );
    }
    HD void write_cost( double c ) {
        nb_diracs.set( src_dist.nb_diracs );
        cost = TF( c );
    }

    /// what the adjoint derives ( loom's `computed_grads` ): what the source says
    static void update_outputs_computed_grads( auto &&flags ) {
        DECAYED_TYPE_OF( src_dist )::computed_grads( flags.src_dist );
    }

    /// the adjoint of the cost: the same transport, then per tent d / d s_i = 2 m_i ( s_i - b_i ) through the
    /// projection and d / d w_i = ( 2 m_i / w_i ) ( w_i^2 / 6 - Q_i ), by atomics on the shared points / half-widths
    HD void update_outputs_bwd( auto &&grad_plan, auto &&rows, int t, int nb, auto &&group ) const {
        const double g = double( grad_plan.cost );
        run( rows, t, nb, group, [&]( SI i, double mi, const ot1d::TentMoments &m ) {
            const double s = double( src_dist.position( i ) ), w = double( src_dist.half_width( i ) );
            src_dist.add_position_grad( grad_plan.src_dist, i, TF( g * 2 * mi * ( s - m.b ) ) );
            src_dist.add_half_width_grad( grad_plan.src_dist, i, TF( g * 2 * mi / w * ( w * w / 6 - m.Q ) ) );
        } );
    }
};

// ── the drivers ─────────────────────────────────────────────────────────────────────────────────

namespace ot1d_coop {

/// a work-group of one ( the host: one view per thread )
struct SoloGroup {};
HD inline void group_barrier( const SoloGroup & ) {}

/// rows on raw memory
template<class T> struct Arr  { T *p;       HD T &operator()( SI k ) const { return p[ k ]; } };
template<class T> struct Arr2 { T *p; SI w; HD T &operator()( SI i, SI j ) const { return p[ i * w + j ]; } };

/// where the gradient of the summed cost goes ( `update_outputs_sum_grad` ): the shape of a source's gradient
template<class GP,class GW>
struct SumGrad {
    GP points;
    GW half_widths;
};

} // namespace ot1d_coop

/// host: the views split between the threads, one view at a time per thread. `grad_plan`: `nullptr` for the forward.
/// `sg`: a `SumGrad` for the forward with the gradient of the summed cost, `nullptr` otherwise.
template<class Grad,class SG = std::nullptr_t>
void continuous_ot_1d( const CpuQueue &queue, const auto &batch, auto &&plan, Grad &&grad_plan, auto &allocator, SG &&sg = nullptr ) {
    using namespace ot1d_coop;
    constexpr bool bwd = ! std::is_same_v<std::decay_t<Grad>,std::nullptr_t>;
    constexpr bool with_sg = ! std::is_same_v<std::decay_t<SG>,std::nullptr_t>;
    if constexpr ( with_sg ) {
        sg.points.fill_with( queue, 0 );
        sg.half_widths.fill_with( queue, 0 );
    }
    const SI nb_items = SI( batch.size() );
    if ( nb_items == 0 )
        return;
    const auto item0 = plan( batch[ 0 ] );
    const SI n = item0.src_dist.nb_tents(), nbp = 3 * n, nbin = SI( item0.dst_dist.values.size() );
    const int nt = int( std::min<SI>( std::max( queue.nb_workers(), 1 ), nb_items ) );
    const SI ni = 5 * nbp + 2 * 256, nd = 7 * nbp + nbin + 1 + 2 * 8;
    std::int32_t *I = allocator.template raw<std::int32_t>( SI( nt ) * ni );
    double *D = allocator.template raw<double>( SI( nt ) * nd );
    if ( ! I || ! D )
        return;
    auto body = [&]( int t ) {
        std::int32_t *i0 = I + SI( t ) * ni;
        double *d0 = D + SI( t ) * nd;
        ContinuousOtRows rows{ Arr<std::int32_t>{ i0 }, Arr<std::int32_t>{ i0 + nbp }, Arr<std::int32_t>{ i0 + 2 * nbp }, Arr<std::int32_t>{ i0 + 3 * nbp },
                               Arr<std::int32_t>{ i0 + 4 * nbp },
                               Arr<double>{ d0 }, Arr<double>{ d0 + nbp }, Arr<double>{ d0 + 2 * nbp },
                               Arr2<double>{ d0 + 3 * nbp, 4 }, Arr<double>{ d0 + 7 * nbp },
                               Arr2<std::int32_t>{ i0 + 5 * nbp, 256 }, Arr2<double>{ d0 + 7 * nbp + nbin + 1, 8 } };
        for ( SI v = nb_items * t / nt; v < nb_items * ( t + 1 ) / nt; ++v ) {
            if constexpr ( bwd )
                plan( batch[ v ] ).update_outputs_bwd( grad_plan( batch[ v ] ), rows, 0, 1, SoloGroup{} );
            else if constexpr ( with_sg )
                plan( batch[ v ] ).update_outputs_sum_grad( rows, 0, 1, SoloGroup{}, sg );
            else
                plan( batch[ v ] ).update_outputs( rows, 0, 1, SoloGroup{} );
        }
    };
    if ( nt <= 1 )
        body( 0 );
    else
        queue.run_threads( nt, body );
}

#if defined( __CUDACC__ )

namespace ot1d_card {

constexpr int OT_BLOCK = 256;
constexpr int OT_CHUNK = 8;     // pieces per thread in the sweep ( few: the threads walk their pieces with strided loads )
constexpr SI  OT_WIDE_SORT = SI( 1 ) << 16;   // breakpoints per view from which the sort is device-wide

/// the map of a piece on ( slope s, density d, cumulated mass m ): the slope jumps by J at its breakpoint, then the
/// piece runs over a length h. Their compositions keep the form ( s, d, m ) -> ( s + a, d + b s + c, m + b d + f s + g )
/// ( b, the summed length, is both coefficients ), so that ONE scan gives the three. `x`: the position of the last
/// breakpoint, `head`: the first piece of a view.
struct Flow1d {
    double a, b, c, f, g, x;
    int    head;
};
struct Flow1dOp {
    __host__ __device__ Flow1d operator()( const Flow1d &x, const Flow1d &y ) const {
        if ( y.head )
            return y;
        return { x.a + y.a, x.b + y.b, x.c + y.c + y.b * x.a, x.f + y.b * x.b + y.f, x.g + y.b * x.c + y.f * x.a + y.g, y.x, x.head };
    }
};

/// the scan's input, made on the fly from the sorted breakpoints: q -> the map of piece q ( gathers of the unsorted
/// positions `xs` and of the jumps `kc` )
struct MakeFlow1d {
    const int *vals; const double *xs, *kc; int P;
    __host__ __device__ Flow1d operator()( int q ) const {
        const int v = q / P, k = q - v * P;
        const int id = vals[ q ], i = id / 3;
        const double x = xs[ SI( v ) * P + id ];
        const double jump = ( id - 3 * i == 1 ? -2 : 1 ) * kc[ i ];
        double h = 0;
        if ( k + 1 < P ) {
            const double x1 = xs[ SI( v ) * P + vals[ q + 1 ] ];
            h = x1 > x ? x1 - x : 0.0;
        }
        return { jump, h, h * jump, 0.5 * h * h, 0.5 * h * h * jump, x, k == 0 };
    }
};

/// the scan's output, written as the sweep reads it: per breakpoint its position, the slope after it, the density
/// and the cumulated mass at it
struct WriteFlow1d {
    double *pos, *slope, *dens, *F; int P;
    __host__ __device__ void operator()( long q, const Flow1d &e ) const {
        const int k = int( q % P );
        pos[ q ] = e.x;
        slope[ q ] = e.a;
        if ( k == 0 ) { dens[ q ] = 0; F[ q ] = 0; }
        if ( k + 1 < P ) { dens[ q + 1 ] = e.c; F[ q + 1 ] = e.g; }
    }
};

/// the sums of a chunk of the sweep, scanned per view
struct Sum5 {
    double P0, P1, P2, phi, cost;
    int    head;
};
struct Sum5Op {
    __host__ __device__ Sum5 operator()( const Sum5 &x, const Sum5 &y ) const {
        if ( y.head )
            return y;
        return { x.P0 + y.P0, x.P1 + y.P1, x.P2 + y.P2, x.phi + y.phi, x.cost + y.cost, x.head };
    }
};
struct Sum4 {
    double P0, P1, P2, phi;
};

/// whether the tents' half-widths are the same on every item ( not so for `ProjectedSumOfAnisotropicTents` ): the kernels below
/// read them from the first item
template<class Src>
inline constexpr bool shared_half_width_v = [] { if constexpr ( requires { Src::shared_half_width; } ) return Src::shared_half_width; else return true; }();

/// breakpoint `id` ( 3 i + 0 / 1 / 2 ) of an item: its position and the jump of the summed density's slope
__device__ __forceinline__ void breakpoint( const auto &src, SI id, double tot, double &x, double &jump ) {
    const SI i = id / 3;
    const int kind = int( id - 3 * i );
    const double s = double( src.position( i ) ), w = double( src.half_width( i ) );
    const double k = double( src.mass( i ) ) / tot / ( w * w );
    x = s + double( kind - 1 ) * w;
    jump = kind == 1 ? -2 * k : k;
}

/// the total mass of the tents ( shared by the items: the first one )
template<class Plan,class Batch>
__global__ void __launch_bounds__( OT_BLOCK ) total_mass_kernel( Plan plan, Batch batch, double *tot ) {
    const auto item = plan( batch[ 0 ] );
    const SI n = item.src_dist.nb_tents();
    double s = 0;
    for ( SI i = threadIdx.x; i < n; i += OT_BLOCK )
        s += double( item.src_dist.mass( i ) );
    using BR = cub::BlockReduce<double,OT_BLOCK>;
    __shared__ typename BR::TempStorage tmp;
    s = BR( tmp ).Sum( s );
    if ( threadIdx.x == 0 )
        *tot = s;
}

/// per tent, the slope jump at its ends ( m_i / tot / w_i^2, shared by the items )
template<class Plan,class Batch>
__global__ void __launch_bounds__( OT_BLOCK ) jumps_kernel( Plan plan, Batch batch, const double *tot, double *kc ) {
    const SI i = SI( blockIdx.x ) * OT_BLOCK + threadIdx.x;
    const auto src = plan( batch[ 0 ] ).src_dist;
    static_assert( shared_half_width_v<std::decay_t<decltype( src )>>, "the card's kernels need half-widths shared by the items" );
    if ( i >= src.nb_tents() )
        return;
    const double w = double( src.half_width( i ) );
    kc[ i ] = double( src.mass( i ) ) / *tot / ( w * w );
}

/// the keys of the breakpoints of views [ v0, v0 + V ), their unsorted positions, and the segments' offsets
template<class Plan,class Batch>
__global__ void __launch_bounds__( OT_BLOCK ) keys_kernel( Plan plan, Batch batch, SI v0, SI V, SI P, const double *tot, float *keys, std::uint64_t *keys64,
                                                        int *vals, int *off, double *xs ) {
    const SI q = SI( blockIdx.x ) * OT_BLOCK + threadIdx.x;
    if ( q <= V )
        off[ q ] = int( q * P );
    if ( q >= V * P )
        return;
    const SI v = q / P, k = q - v * P;
    double x, jump;
    breakpoint( plan( batch[ v0 + v ] ).src_dist, k, *tot, x, jump );
    if ( keys64 )   // ( view, order-preserving bits of the float ): one device-wide sort
        keys64[ q ] = ( std::uint64_t( v ) << 32 ) | ot1d_coop::float_key( x );
    else
        keys[ q ] = float( x );
    vals[ q ] = int( k );
    xs[ q ] = x;
}

/// the keys are floats: thread q starts a run of equal SORTED keys and sorts it by the exact positions -- a few
/// elements at most, and no gather outside the runs
template<class K>
__global__ void __launch_bounds__( OT_BLOCK ) runs_kernel( SI V, SI P, const K *skeys, const double *xs, int *vals ) {
    const SI q = SI( blockIdx.x ) * OT_BLOCK + threadIdx.x;
    if ( q >= V * P )
        return;
    const SI v = q / P, k = q - v * P;
    if ( k + 1 >= P || skeys[ q + 1 ] != skeys[ q ] || ( k && skeys[ q - 1 ] == skeys[ q ] ) )
        return;     // not the start of a run of two or more
    SI e = k + 2;
    while ( e < P && skeys[ v * P + e ] == skeys[ q ] )
        ++e;
    const double *xv = xs + v * P;
    int *iv = vals + v * P;
    for ( SI a = k + 1; a < e; ++a ) {
        const int id = iv[ a ];
        const double x = xv[ id ];
        SI b = a;
        for ( ; b > k && xv[ iv[ b - 1 ] ] > x; --b )
            iv[ b ] = iv[ b - 1 ];
        iv[ b ] = id;
    }
}

/// the target's cumulated masses, one thread per view
template<class Plan,class Batch>
__global__ void __launch_bounds__( OT_BLOCK ) cdf_kernel( Plan plan, Batch batch, SI v0, SI V, SI nbin, double *cdf ) {
    const SI v = SI( blockIdx.x ) * OT_BLOCK + threadIdx.x;
    if ( v >= V )
        return;
    plan( batch[ v0 + v ] ).dst_dist.with_defaults( [&]( auto &&img ) {
        const double sc = double( img.frame( 0, 0 ) );
        double c = 0;
        for ( SI j = 0; j < nbin; ++j ) {
            cdf[ v * ( nbin + 1 ) + j ] = c;
            c += sc * ( double( img.knots( j + 1 ) ) - double( img.knots( j ) ) ) * double( img.values[ j ] );
        }
        cdf[ v * ( nbin + 1 ) + nbin ] = c;
    } );
}

/// the sweep: thread ( v, c ) transports pieces [ c OT_CHUNK, ( c + 1 ) OT_CHUNK ) of view v, from where they start in
/// the target; local sums at each breakpoint, the chunk's totals
template<class Plan,class Batch>
__global__ void __launch_bounds__( OT_BLOCK ) sweep_kernel( Plan plan, Batch batch, SI v0, SI V, SI P, SI nch, SI nbin, const double *pos_all,
                                                         const double *slope_all, const double *dens_all, const double *F_all,
                                                         const double *cdf_all, Sum4 *S, Sum5 *CH ) {
    const SI q = SI( blockIdx.x ) * OT_BLOCK + threadIdx.x;
    if ( q >= V * nch )
        return;
    const SI v = q / nch, c = q - v * nch;
    const SI k0 = c * OT_CHUNK, k1 = k0 + OT_CHUNK < P ? k0 + OT_CHUNK : P;
    const double *ps = pos_all + v * P, *sl = slope_all + v * P, *de = dens_all + v * P;
    Sum4 *sv = S + v * P;
    const double *cdf = cdf_all + v * ( nbin + 1 );
    const double F_lo = F_all[ v * P + k0 ];
    ot1d::Sums1d res;
    plan( batch[ v0 + v ] ).dst_dist.with_defaults( [&]( auto &&img ) {
        const double sc = double( img.frame( 0, 0 ) ), of = double( img.origin( 0 ) );
        const double dtot = cdf[ nbin ];
        SI a = 0, b = nbin - 1;
        while ( a < b ) {
            const SI mid = ( a + b ) / 2;
            if ( cdf[ mid + 1 ] / dtot > F_lo ) b = mid; else a = mid + 1;
        }
        const double dd0 = F_lo - cdf[ a ] / dtot;
        auto dst = ot1d::func_stream0( [&]( long j ) {
            ot1d::PolyPiece1d<double,0> p;
            p.x0 = of + sc * double( img.knots( j ) );
            p.x1 = of + sc * double( img.knots( j + 1 ) );
            p.c[ 0 ] = double( img.values[ j ] ) / dtot;
            return p;
        }, long( nbin ), long( a ) );
        auto src = ot1d::func_stream1( [&]( long k ) {
            ot1d::PolyPiece1d<double,1> p;
            p.x0 = ps[ k ];
            p.x1 = k + 1 < P && ps[ k + 1 ] > p.x0 ? ps[ k + 1 ] : p.x0;
            const double d = de[ k ];
            p.c[ 0 ] = d > 0 ? d : 0.0;
            p.c[ 1 ] = k + 1 < P ? sl[ k ] : 0.0;
            p.event = int( k );
            return p;
        }, long( k0 ), long( k1 ) );
        res = ot1d::transport_1d<3>( src, dst, [&]( int k, const ot1d::Sums1d &s ) {
            sv[ k ] = { s.P0, s.P1, s.P2, s.phi };
        }, dd0 > 0 ? dd0 : 0.0 );
    } );
    CH[ q ] = { res.P0, res.P1, res.P2, res.phi, res.cost, c == 0 };
}

/// the chunks' offsets into the sums ( coalesced ): the per-tent kernel then gathers one record per breakpoint; and
/// the inverse permutation
template<int = 0>
__global__ void __launch_bounds__( OT_BLOCK ) offsets_kernel( SI V, SI P, SI nch, const Sum5 *CH, const int *svals, int *inv, Sum4 *S ) {
    const SI q = SI( blockIdx.x ) * OT_BLOCK + threadIdx.x;
    if ( q >= V * P )
        return;
    const SI v = q / P, k = q - v * P, c = k / OT_CHUNK;
    inv[ v * P + svals[ q ] ] = int( k );
    if ( c == 0 )
        return;
    const Sum5 &o = CH[ v * nch + c - 1 ];
    Sum4 &s = S[ q ];
    s.P0 += o.P0; s.P1 += o.P1; s.P2 += o.P2; s.phi += o.phi;
}

/// per tent i, over the views of the chunk: the moments, then the outputs ( forward ) and / or the gradients summed in
/// registers ( one atomic per tent and per chunk of views ); the cost of each view. MODE 0: the forward, 1: the
/// adjoint ( `grad` is the plan's cotangent ), 2: the forward and the gradient of the summed cost ( `grad`: a `SumGrad` )
template<int MODE,class Plan,class Grad,class Batch>
__global__ void __launch_bounds__( OT_BLOCK ) tents_kernel( Plan plan, Grad grad_plan, Batch batch, SI v0, SI V, SI P, SI nch, const double *tot,
                                                         const int *inv, const Sum4 *S, const Sum5 *CH ) {
    const SI i = SI( blockIdx.x ) * OT_BLOCK + threadIdx.x;
    const SI n = P / 3;
    if ( i >= n )
        return;
    using Src = std::decay_t<decltype( plan( batch[ 0 ] ).src_dist )>;
    constexpr SI D = Src::ct_proj;
    double gp[ D ], gw = 0;
    for ( SI d = 0; d < D; ++d )
        gp[ d ] = 0;
    const auto src0 = plan( batch[ v0 ] ).src_dist;
    static_assert( shared_half_width_v<Src>, "the card's kernels need half-widths shared by the items" );
    const double mi = double( src0.mass( i ) ) / *tot, w = double( src0.half_width( i ) );
    for ( SI v = 0; v < V; ++v ) {
        auto item = plan( batch[ v0 + v ] );
        const auto &src = item.src_dist;
        auto at = [&]( SI id ) {
            const Sum4 &l = S[ v * P + inv[ v * P + id ] ];
            ot1d::Sums1d s;
            s.P0 = l.P0; s.P1 = l.P1; s.P2 = l.P2; s.phi = l.phi;
            return s;
        };
        const double s = double( src.position( i ) );
        const auto m = ot1d::tent_moments( s, w, at( 3 * i ), at( 3 * i + 1 ), at( 3 * i + 2 ) );
        if constexpr ( MODE != 0 ) {
            double g = 1;
            if constexpr ( MODE == 1 )
                g = double( grad_plan( batch[ v0 + v ] ).cost );
            const double gs = g * 2 * mi * ( s - m.b );
            for ( SI d = 0; d < D; ++d )
                gp[ d ] += gs * double( src.normal_component( d ) );
            gw += g * 2 * mi / w * ( w * w / 6 - m.Q );
        }
        if constexpr ( MODE != 1 ) {
            item.write_tent( i, m );
            if ( i == 0 )
                item.write_cost( CH[ v * nch + nch - 1 ].cost );
        }
    }
    if constexpr ( MODE != 0 ) {
        auto gsrc = [&] {
            if constexpr ( MODE == 1 )
                return grad_plan( batch[ v0 ] ).src_dist;
            else
                return grad_plan;
        }();
        for ( SI d = 0; d < D; ++d )
            src0.add_point_grad( gsrc, i, d, gp[ d ] );
        src0.add_half_width_grad( gsrc, i, gw );
    }
}

inline int blocks_for( SI n ) { return int( ( n + OT_BLOCK - 1 ) / OT_BLOCK ); }

/// never launched: the address names a slot of loom's kernel timing ( `LOOM_KERNEL_TIMING` ) for the CUB calls
template<int K> __global__ void cub_slot() {}

/// a CUB call on the stream, timed in slot K when the timing is on
template<int K>
void timed_cub( cudaStream_t stream, auto &&call ) {
    detail::CudaQueueTiming::timed_launch( &cub_slot<K>, "cub", 0, 0, 0, stream, call );
}

} // namespace ot1d_card

/// card: the views by chunks, each by device-wide launches ( see the header )
template<class Queue,class Grad,class SG = std::nullptr_t> requires ( ! std::is_same_v<Queue,CpuQueue> )
void continuous_ot_1d( const Queue &queue, const auto &batch, auto &&plan, Grad &&grad_plan, auto &allocator, SG &&sg = nullptr ) {
    using namespace ot1d_card;
    constexpr bool bwd = ! std::is_same_v<std::decay_t<Grad>,std::nullptr_t>;
    constexpr bool with_sg = ! std::is_same_v<std::decay_t<SG>,std::nullptr_t>;
    if constexpr ( with_sg ) {
        sg.points.fill_with( queue, 0 );
        sg.half_widths.fill_with( queue, 0 );
    }
    const SI nb_items = SI( batch.size() );
    if ( nb_items == 0 )
        return;
    const auto item0 = plan( batch[ 0 ] );
    const SI n = item0.src_dist.nb_tents(), P = 3 * n, nbin = SI( item0.dst_dist.values.size() );
    if ( n == 0 )
        return;
    const SI nch = ( P + OT_CHUNK - 1 ) / OT_CHUNK;
    const bool wide = P >= OT_WIDE_SORT;

    // the views per chunk: a memory budget ( `SDOT_OT1D_BUDGET_MB`, 1 GB by default ), and 32-bit offsets
    const SI per_bp = ( wide ? 2 * sizeof( std::uint64_t ) : 2 * sizeof( float ) ) + 2 * sizeof( int ) + sizeof( int ) + 5 * sizeof( double ) + sizeof( Sum4 );
    const SI per_view = P * per_bp + ( nbin + 1 ) * sizeof( double ) + nch * sizeof( Sum5 );
    SI budget = SI( 1 ) << 30;
    if ( const char *e = std::getenv( "SDOT_OT1D_BUDGET_MB" ) )
        budget = SI( std::atol( e ) ) << 20;
    SI V = std::max<SI>( 1, std::min<SI>( nb_items, budget / per_view ) );
    V = std::min<SI>( V, std::max<SI>( 1, ( ( SI( 1 ) << 31 ) - 1 ) / P - 1 ) );
    const SI M = V * P;
    int end_bit = 32;
    while ( ( SI( 1 ) << ( end_bit - 32 ) ) < V )
        ++end_bit;

    cudaStream_t stream = queue.stream;
    double        *tot    = allocator.template raw<double>( 1 );
    double        *kc     = allocator.template raw<double>( n );
    float         *keys   = wide ? nullptr : allocator.template raw<float>( 2 * M );
    std::uint64_t *keys64 = wide ? allocator.template raw<std::uint64_t>( 2 * M ) : nullptr;
    int           *vals   = allocator.template raw<int>( 2 * M );
    int           *off    = allocator.template raw<int>( V + 1 );
    int           *inv    = allocator.template raw<int>( M );
    double        *xs     = allocator.template raw<double>( M );
    double        *pos    = allocator.template raw<double>( M );
    double        *slope  = allocator.template raw<double>( M );
    double        *dens   = allocator.template raw<double>( M );
    double        *F      = allocator.template raw<double>( M );
    Sum4          *S      = allocator.template raw<Sum4>( M );
    double        *cdf    = allocator.template raw<double>( V * ( nbin + 1 ) );
    Sum5          *CH     = allocator.template raw<Sum5>( V * nch );
    const int     *svals  = vals + M;   // the sorted ids

    auto flows = SDOT_OT1D_ITER::make_transform_iterator( SDOT_OT1D_ITER::counting_iterator<int>( 0 ), MakeFlow1d{ svals, xs, kc, int( P ) } );
    auto writer = SDOT_OT1D_ITER::make_tabulate_output_iterator( WriteFlow1d{ pos, slope, dens, F, int( P ) } );
    std::size_t b_sort = 0, b_scan = 0, b_ch = 0;
    if ( wide )
        cub::DeviceRadixSort::SortPairs( nullptr, b_sort, keys64, keys64 + M, vals, vals + M, int( M ), 0, end_bit, stream );
    else
        cub::DeviceSegmentedSort::SortPairs( nullptr, b_sort, keys, keys + M, vals, vals + M, int( M ), int( V ), off, off + 1, stream );
    cub::DeviceScan::InclusiveScan( nullptr, b_scan, flows, writer, Flow1dOp{}, int( M ), stream );
    cub::DeviceScan::InclusiveScan( nullptr, b_ch, CH, CH, Sum5Op{}, int( V * nch ), stream );
    const std::size_t b_tmp = std::max( b_sort, std::max( b_scan, b_ch ) );
    char *tmp = allocator.template raw<char>( SI( b_tmp ) );
    if ( allocator.failed )
        return;

    // the arguments as the kernels take them ( device-accessible views )
    const auto kplan  = kernel_form( queue, MutList(), plan );
    const auto kbatch = kernel_form( queue, MutList(), batch );
    using PlanT = std::decay_t<decltype( kplan )>;
    using BatchT = std::decay_t<decltype( kbatch )>;
    launch_kernel( queue, &total_mass_kernel<PlanT,BatchT>, 1, OT_BLOCK, 0, kplan, kbatch, tot );
    launch_kernel( queue, &jumps_kernel<PlanT,BatchT>, blocks_for( n ), OT_BLOCK, 0, kplan, kbatch, ( const double * )tot, kc );
    for ( SI v0 = 0; v0 < nb_items; v0 += V ) {
        const SI Vc = std::min<SI>( V, nb_items - v0 ), Mc = Vc * P;
        launch_kernel( queue, &keys_kernel<PlanT,BatchT>, blocks_for( Mc ), OT_BLOCK, 0, kplan, kbatch, v0, Vc, P, ( const double * )tot, keys, keys64, vals, off, xs );
        std::size_t bytes = b_tmp;
        timed_cub<0>( stream, [&] {
            if ( wide )
                cub::DeviceRadixSort::SortPairs( tmp, bytes, keys64, keys64 + M, vals, vals + M, int( Mc ), 0, end_bit, stream );
            else
                cub::DeviceSegmentedSort::SortPairs( tmp, bytes, keys, keys + M, vals, vals + M, int( Mc ), int( Vc ), off, off + 1, stream );
        } );
        if ( wide )
            launch_kernel( queue, &runs_kernel<std::uint64_t>, blocks_for( Mc ), OT_BLOCK, 0, Vc, P, ( const std::uint64_t * )( keys64 + M ), ( const double * )xs, vals + M );
        else
            launch_kernel( queue, &runs_kernel<float>, blocks_for( Mc ), OT_BLOCK, 0, Vc, P, ( const float * )( keys + M ), ( const double * )xs, vals + M );
        bytes = b_tmp;
        timed_cub<1>( stream, [&] { cub::DeviceScan::InclusiveScan( tmp, bytes, flows, writer, Flow1dOp{}, int( Mc ), stream ); } );
        launch_kernel( queue, &cdf_kernel<PlanT,BatchT>, blocks_for( Vc ), OT_BLOCK, 0, kplan, kbatch, v0, Vc, nbin, cdf );
        launch_kernel( queue, &sweep_kernel<PlanT,BatchT>, blocks_for( Vc * nch ), OT_BLOCK, 0, kplan, kbatch, v0, Vc, P, nch, nbin,
                       ( const double * )pos, ( const double * )slope, ( const double * )dens, ( const double * )F, ( const double * )cdf, S, CH );
        bytes = b_tmp;
        timed_cub<2>( stream, [&] { cub::DeviceScan::InclusiveScan( tmp, bytes, CH, CH, Sum5Op{}, int( Vc * nch ), stream ); } );
        launch_kernel( queue, &offsets_kernel<>, blocks_for( Mc ), OT_BLOCK, 0, Vc, P, nch, ( const Sum5 * )CH, svals, inv, S );
        if constexpr ( bwd ) {
            const auto kgrad = kernel_form( queue, MutList(), grad_plan );
            using GradT = std::decay_t<decltype( kgrad )>;
            launch_kernel( queue, &tents_kernel<1,PlanT,GradT,BatchT>, blocks_for( n ), OT_BLOCK, 0, kplan, kgrad, kbatch, v0, Vc, P, nch, ( const double * )tot,
                           ( const int * )inv, ( const Sum4 * )S, ( const Sum5 * )CH );
        } else if constexpr ( with_sg ) {
            const auto ksg = ot1d_coop::SumGrad{ kernel_form( queue, MutList(), sg.points ), kernel_form( queue, MutList(), sg.half_widths ) };
            using GradT = std::decay_t<decltype( ksg )>;
            launch_kernel( queue, &tents_kernel<2,PlanT,GradT,BatchT>, blocks_for( n ), OT_BLOCK, 0, kplan, ksg, kbatch, v0, Vc, P, nch, ( const double * )tot,
                           ( const int * )inv, ( const Sum4 * )S, ( const Sum5 * )CH );
        } else
            launch_kernel( queue, &tents_kernel<0,PlanT,PlanT,BatchT>, blocks_for( n ), OT_BLOCK, 0, kplan, kplan, kbatch, v0, Vc, P, nch, ( const double * )tot,
                           ( const int * )inv, ( const Sum4 * )S, ( const Sum5 * )CH );
    }
}

#endif

}

#include "ContinuousOtPlan1dGrid.h"
