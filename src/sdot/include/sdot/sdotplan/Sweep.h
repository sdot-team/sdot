#pragma once

// =====================================================================================
// THE SOLVER'S DIAGRAMS : what Newton asks of a power diagram, and nothing else.
//
//     set_weights( w )              set the weights ( user order ), rebuild the majorants
//     measures( a, &fa )             ONE sweep : the measure of each cell AND its facets
//                                   `c_ij = int_{facet} rho / ( 2 |p_i - p_j| )` ( `Laplacian.h` )
//
// Everything runs here on the CPU queue ( `CpuQueue::run_threads` : blocks of cells dealt in turn,
// in storage order, where two consecutive seeds are neighbours in space: `diagram::for_each_seed` ). The scratch
// of the cells is MANAGED HERE, not by loom : a solver chains a hundred diagrams in a single
// call, and an overflow must only restart the sweep in progress, not the whole call. One
// row of words per thread, doubled as long as a cell does not fit.
//
// The cell of a seed is built as everywhere ( `diagram::make_cell` : the domain, then the
// planes that the storage provider proposes ), its mass integrated as everywhere
// ( `diagram::integrate_into` ), and its facets read as `diagram::hessian_row` reads them -- this
// file adds no geometry, it chains.
//
// A SINK ( partial transport, `PowerDiagram( sink_weight = ... )` ): the cell of a seed is integrated inside its ball only
// ( `diagram::cell_dist` ), and the mass beyond goes to the sink, which Newton sees as ONE MORE SEED, identifier 0 ( the user's seeds
// move to `1 .. n` ): its mass is what the cells leave of the density, its weight is the gauge ( `w_0 = 0`: the sink weight of the
// diagram ), and its facet with seed `i` is `c_is = d m_i / d r2_i` -- the sphere inside the cell. The graph stays a Laplacian ( a
// row sums to zero ), and Newton, its gauge and its linear solvers are those of the balanced transport.
//
// A NON-CONSTANT DENSITY on a facet : the distribution must know how to integrate its density on
// a facet ( `facet_mass( piece, cut )` -- 2D gaussians do it in closed form ) ; failing
// that the laplacian cannot be assembled and the compilation says so.
// =====================================================================================

#include <loom/support/kernels/CpuQueue.h>
#include "../diagram/Ops.h"
#include "../bsp_build_level.h"
#include "Laplacian.h"
#include "Groups.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#ifdef _WIN32
    // not <windows.h>: its macros ( INFINITE, min, max ) leak into everything included after
    extern "C" {
        struct sdot_MemoryStatusEx { unsigned long length, load; unsigned long long total_phys, avail_phys, total_page, avail_page, total_virtual, avail_virtual, avail_ext; };
        __declspec( dllimport ) int __stdcall GlobalMemoryStatusEx( sdot_MemoryStatusEx * );
    }
#else
    #include <unistd.h>
#endif
#include <vector>

namespace sdot {
namespace sdotplan {

inline double now() {
    using namespace std::chrono;
    return duration<double>( steady_clock::now().time_since_epoch() ).count();
}

/// the mass of a facet for the density of a piece : `rho * measure` for a constant density, what
/// the distribution can say about it otherwise
template<class Dens,class Pc>
double density_facet_mass( const Dens &dens, const Pc &pc, int cut, double mes ) {
    if constexpr ( Dens::is_constant )
        return double( dens.value ) * mes;
    else if constexpr ( requires { dens.facet_mass( pc, cut ); } )
        return double( dens.facet_mass( pc, cut ) );
    else
        static_assert( Dens::is_constant, "sdotplan : this distribution does not know how to integrate its density on a facet ( `facet_mass` )" );
    return 0;
}

/// ONE CELL of the seed of rank `k` : its mass in `mass`, and `facet( j, c_kj )` for each neighbour
/// `j` ( in ranks ). Returns `false` if the scratch was not enough.
template<class PD,class Local>
bool measure_and_facets( const PD &pd, SI k, Local &c, Local &piece, const auto &dom, const auto &full_dist,
                         double &mass, auto &&facet, bool with_facets, double *to_sink = nullptr ) {
    using TF = typename PD::TF;
    constexpr int D = PD::ct_dim;
    if ( ! diagram::make_cell( pd, c, k, dom ) )
        return false;
    const auto &dist = diagram::cell_dist( pd, k, full_dist );
    TF m = 0;
    if ( ! diagram::integrate_into<TF>( m, c, piece, dist ) )
        return false;
    mass = double( m );
    if ( ! with_facets )
        return true;

    // the neighbours of the cell are its live cuts : one slot per cut, accumulated piece
    // by piece ( a piece carries the same identifiers, plus those of its box )
    c.tidy();
    const int nc = c.nb_cuts();
    // a cell can have MANY neighbours ( the seed off a line of 1000 collinear seeds has one facet per seed of the line ): the
    // stack covers the usual ones, the heap the others. It must NOT be reported as a scratch overflow: more room would not
    // help, and `measures` would double the scratch for ever
    double stack_vals[ 512 ];
    std::vector<double> heap_vals;
    double *vals = stack_vals;
    if ( nc > 512 ) {
        heap_vals.resize( nc );
        vals = heap_vals.data();
    }
    for ( int q = 0; q < nc; ++q )
        vals[ q ] = 0;
    PieceWorkspace<Local> ws{ piece };
    auto plane_of = [&]( int q, auto *dir, auto &off ) {
        typename PD::TF d[ D ], o;
        diagram::plane_of( pd, c, k, q, d, o );
        for ( int i = 0; i < D; ++i )
            dir[ i ] = d[ i ];
        off = o;
    };
    if constexpr ( requires { DECAYED_TYPE_OF( dist )::cuts_facets; } ) {
        // a density that cuts the FACETS of the cell itself ( `Mesh` ): what the pieces would miss -- an element on the far side of
        // a facet that lies on its side -- is not lost
        dist.for_each_facet_mass( c, plane_of, [&]( int cut, auto part ) { vals[ cut ] += double( part ); } );
    } else {
        double sk = 0;
        dist.for_each_piece( c, ws, [&]( const auto &pc, const auto &dens ) {
            if constexpr ( requires { dens.weight_mass( pc ); } )
                sk += double( dens.weight_mass( pc ) );
            pc.template for_each_facet<TF>( [&]( int cut, TF mes ) {
                const int id = pc.cid[ cut ];
                if ( id < 0 )
                    return;                                  // the domain, or a box edge : immobile
                const double val = density_facet_mass( dens, pc, cut, double( mes ) );
                for ( int q = 0; q < nc; ++q )
                    if ( c.cid[ q ] == id ) {
                        vals[ q ] += val;
                        break;
                    }
            } );
        } );
        if ( to_sink )
            *to_sink = sk;
    }
    if ( ws.overflow )
        return false;

    auto coefficient = [&]( int q ) {                    // `c_kj` ( two members of a GROUP: the distance of their virtual diracs in
        // the frame of their lowest common aggregate, `diagram::group_distance2` -- each facet then speaks of its own weights )
        const double d2 = double( diagram::group_distance2( pd, k, c.cid[ q ] ) );
        return d2 > 0 ? vals[ q ] / ( 2 * std::sqrt( d2 ) ) : -1.0;
    };
    if constexpr ( is_periodic<PD>() ) {
        // PERIODIC: a facet with an IMAGE of the cell itself moves with no weight ( both sides carry `w_k` ), and two images of
        // the same seed are ONE neighbour of the laplacian -- their coefficients are summed into the first, under the seed's rank
        for ( int q = 0; q < nc; ++q ) {
            if ( c.cid[ q ] < 0 || pd.seed_of( c.cid[ q ] ) == k ) {
                vals[ q ] = -1;
                continue;
            }
            vals[ q ] = coefficient( q );
            if ( vals[ q ] < 0 )
                continue;
            const SI sq = pd.seed_of( c.cid[ q ] );
            for ( int r = 0; r < q; ++r )
                if ( vals[ r ] >= 0 && pd.seed_of( c.cid[ r ] ) == sq ) {
                    vals[ r ] += vals[ q ];
                    vals[ q ] = -1;                       // merged: not a facet of its own
                    break;
                }
        }
        for ( int q = 0; q < nc; ++q )
            if ( vals[ q ] >= 0 )
                facet( pd.seed_of( c.cid[ q ] ), vals[ q ] );
    } else {
        for ( int q = 0; q < nc; ++q ) {
            if ( c.cid[ q ] < 0 )
                continue;
            const double cq = coefficient( q );
            if ( cq >= 0 )
                facet( c.cid[ q ], cq );
        }
    }
    return true;
}

/// THE DIAGRAM AS SEEN BY NEWTON. `pd` is the storage ( `PowerDiagram_Bsp` / `_Plain` ) whose
/// weights -- and tree majorants -- are WRITABLE views ( `with_weights` ) : this is
/// where they are written.
template<class PD,class Dom,class Dist,class TK>
struct Sweep {
    using TF = typename PD::TF;
    using Local = typename Dom::template Local<TK>;
    static constexpr int D = PD::ct_dim;
    static constexpr int nbc = diagram::nb_work_cells<Dist>();
    /// a SINK: one more seed for Newton, identifier 0 ( see the head of the file ); `off` shifts the user's identifiers
    static constexpr bool sink = diagram::has_sink<PD>();
    static constexpr SI   off  = sink ? 1 : 0;

    const CpuQueue &queue;
    PD             &pd;                                  ///< the weights are written there ( output views )
    const Dom      &dom;
    const Dist     *dist;                                ///< the current density ( it changes from one step to the next )
    int             nt;                                  ///< virtual threads ( blocks of cells dealt in turn )
    SI              cap;                                 ///< vertices per local cell
    SI              words = 0;
    std::vector<std::vector<std::int32_t>> scratch;      ///< one row per thread
    std::vector<std::vector<Facet>>      fa_th;        ///< the facets, per thread
    std::vector<SI> rank_of;                             ///< the rank of seed `i` ( user order; the sink first, rank 0, when there is one )
    double          total_mass = 0;                      ///< SINK: the mass of the density in the domain ( `domain_mass` )
    /// GROUPS ( `Groups.h` ): Newton's unknowns are those of the tree of aggregates, and `set_weights` turns them into the
    /// diagram's outer weights and virtual weights ( which `pd` then carries as writable views, `with_groups` )
    static constexpr bool groups = has_groups<PD>();
    GroupStructure  grp;

    SI     nb_overflowed = 0;                               ///< scratch doublings, in total
    int    nb_diag    = 0;
    double t_majorant = 0, t_diag = 0;

    /// `pd_in` : the storage as it comes in ( its memories, `memo_*`, are copied into those of `pd` )
    Sweep( const CpuQueue &queue, PD &pd, const auto &pd_in, const Dom &dom, const Dist &dist, SI cap0 )
        : queue( queue ), pd( pd ), dom( dom ), dist( &dist ), nt( std::max( queue.nb_workers(), 1 ) ), cap( std::max<SI>( cap0, 8 ) ) {
        const SI n = pd.nb_seeds();
        scratch.resize( nt );
        fa_th.resize( nt );
        rank_of.resize( n + off );
        if constexpr ( sink )
            rank_of[ 0 ] = 0;
        for ( SI k = 0; k < n; ++k )
            rank_of[ off + pd.user_id( k ) ] = off + k;
        resize_scratch();
        if constexpr ( groups ) {                        // the tree, in Newton's numbering ( the sink alone, first )
            const SI m = SI( pd.group_agg_parent.shape( 0 ) );
            grp.n = n + off;
            grp.m = m;
            grp.parent.assign( n + off, -1 );
            grp.sv_init.assign( n + off, 0.0 );
            for ( SI k = 0; k < n; ++k ) {
                const SI i = off + pd.user_id( k );
                grp.parent[ i ] = pd.seed_parent( k );
                grp.sv_init[ i ] = double( pd_in.group_vweights( k ) );
            }
            grp.agg_parent.resize( m );
            grp.agg_scale.resize( m );
            grp.agg_outer.resize( m );
            grp.av_init.resize( m );
            for ( SI a = 0; a < m; ++a ) {
                grp.agg_parent[ a ] = pd.agg_parent( a );
                grp.agg_scale[ a ] = double( pd.agg_scale( a ) );
                grp.agg_outer[ a ] = double( pd.agg_outer( a ) );
                grp.av_init[ a ] = double( pd_in.group_agg_vweights( a ) );
            }
            grp.build();
        }
        if constexpr ( requires { PD::has_memo; } ) {
            if constexpr ( PD::has_memo ) {              // the memory : the previous memories, or nothing
                const SI K = SI( pd.memo_nbrs.shape( 1 ) );
                for ( SI k = 0; k < n; ++k ) {
                    const int c = int( pd_in.memo_counts( k ) );
                    pd.memo_counts( k ) = c;
                    for ( SI q = 0; q < c && q < K; ++q )
                        pd.memo_nbrs( k, q ) = int( pd_in.memo_nbrs( k, q ) );
                }
            }
        }
    }

    /// the seeds Newton sees: those of the diagram, and the sink
    SI n() const { return pd.nb_seeds() + off; }

    /// SINK: the mass of the density in the domain -- the domain itself integrated, once
    double domain_mass() {
        for ( ;; ) {
            Carver cv{ scratch[ 0 ].data(), words };
            Local c, piece;
            c.attach( cv, cap );
            if constexpr ( nbc > 1 ) piece.attach( cv, cap );
            else                     piece = c;
            TF m = 0;
            if ( c.load( dom ) && diagram::integrate_into<TF>( m, c, piece, *dist ) )
                return double( m );
            cap *= 2;
            grow_scratch();
            ++nb_overflowed;
        }
    }

    void resize_scratch() {
        words = diagram::words_for<Local,TF>( cap, nbc, false );
        for ( auto &s : scratch )
            s.assign( size_t( words ) + 16, 0 );
    }

    /// `resize_scratch` after a doubling of `cap` -- which has to end: a cell that fails for another reason than the room
    /// would otherwise double the scratch until the memory of the machine is gone, with nothing said ( the trace
    /// included, since it would never reach the first stage ). The limit is the machine's: half of its physical memory.
    void grow_scratch() {
        resize_scratch();
        const double bytes = double( words ) * 4.0 * double( scratch.size() );
        #ifdef _WIN32
            sdot_MemoryStatusEx status; status.length = sizeof( status );
            const double ram = GlobalMemoryStatusEx( &status ) ? double( status.total_phys ) : 0.0;
        #else
            const double ram = double( sysconf( _SC_PHYS_PAGES ) ) * double( sysconf( _SC_PAGESIZE ) );
        #endif
        if ( ram > 0 && bytes > 0.5 * ram ) {
            std::fprintf( stderr, "sdotplan: the scratch of the cells reached %.1f GiB ( %lld vertices per cell ), more than half of the "
                                  "memory of the machine, and a cell still does not fit -- giving up\n",
                          bytes / double( 1 << 30 ), ( long long ) cap );
            std::abort();
        }
    }

    /// one contiguous slice of `[ 0, n )` per thread
    static void thread_range( SI n, int t, int nt, SI &b, SI &e ) {
        b = SI( ( long long ) t * n / nt );
        e = SI( ( long long ) ( t + 1 ) * n / nt );
    }

    /// THE WEIGHTS `W` ( user order ) set on the storage, and the tree majorants rebuilt
    void set_weights( const std::vector<double> &W_all ) {
        const double t0 = now();
        const SI n = pd.nb_seeds();
        const double *W = W_all.data() + off;           // ( the sink's weight is the gauge, `0`: the diagram's own )
        std::vector<double> outer;
        if constexpr ( groups ) {                        // Newton's unknowns -> the outer weights and the virtual weights of the tree
            outer.resize( n );
            queue.run_threads( nt, [&]( int t ) {
                SI b, e;
                thread_range( n, t, nt, b, e );
                for ( SI k = b; k < e; ++k ) {
                    const SI i = off + pd.user_id( k );
                    outer[ i - off ] = grp.outer( W_all, i );
                    pd.group_vweights( k ) = TF( grp.parent[ i ] >= 0 ? grp.value( W_all, i ) : 0.0 );
                }
            } );
            for ( SI a = 0; a < grp.m; ++a )
                pd.group_agg_vweights( a ) = TF( grp.agg_parent[ a ] >= 0 ? grp.value( W_all, grp.n + a ) : 0.0 );
            W = outer.data();
        }
        if constexpr ( requires { pd.sorted_weights; } ) {
            queue.run_threads( nt, [&]( int t ) {
                SI b, e;
                thread_range( n, t, nt, b, e );
                for ( SI k = b; k < e; ++k )
                    pd.sorted_weights( k ) = TF( W[ pd.user_id( k ) ] );
            } );
            const SI nb_nodes = SI( pd.tree.node_begin.shape( 0 ) );
            queue.run_threads( nt, [&]( int t ) {
                SI b, e;
                thread_range( nb_nodes, t, nt, b, e );
                for ( SI m = b; m < e; ++m )
                    bsp_refresh_majorant( pd.sorted_cloud(), pd.tree.node_begin( m ), pd.tree.node_end( m ),
                                          pd.tree.node_wa( m ), pd.tree.node_wb( m ) );
            } );
        } else {
            queue.run_threads( nt, [&]( int t ) {
                SI b, e;
                thread_range( n, t, nt, b, e );
                for ( SI k = b; k < e; ++k )
                    pd.weights( k ) = TF( W[ k ] );
            } );
        }
        t_majorant += now() - t0;
    }

    /// THE MEASURES `a` ( user order ) at the current weights, and if `fa` is not null the facets
    /// `c_ij` ( user identifiers ). One sweep, restarted on scratch overflow.
    void measures( std::vector<double> &a_all, std::vector<Facet> *fa = nullptr ) {
        const double t0 = now();
        const SI n = pd.nb_seeds();
        a_all.resize( n + off );
        double *a = a_all.data() + off;
        for ( ;; ) {
            std::atomic<bool> overflowed{ false };
            queue.run_threads( nt, [&]( int t ) {
                Carver cv{ scratch[ t ].data(), words };
                Local c, piece;
                c.attach( cv, cap );
                if constexpr ( nbc > 1 ) piece.attach( cv, cap );
                else                     piece = c;
                auto &fv = fa_th[ t ];
                fv.clear();
                diagram::for_each_seed<PD>( n, t, nt, [&]( SI k ) {
                    const SI i = pd.user_id( k );
                    double cs = 0;
                    if ( ! measure_and_facets( pd, k, c, piece, dom, *dist, a[ i ],
                                               [&]( SI j, double cij ) { fv.push_back( Facet{ off + i, off + pd.user_id( j ), cij } ); },
                                               fa != nullptr, sink ? &cs : nullptr ) ) {
                        overflowed = true;
                        return false;
                    }
                    if ( sink && fa != nullptr && cs > 0 ) {     // the facet with the sink, seen from both sides
                        fv.push_back( Facet{ off + i, 0, cs } );
                        fv.push_back( Facet{ 0, off + i, cs } );
                    }
                    // the memory ( 3D ) : the neighbours of this cell, proposed first to the next sweep --
                    // even from a rejected trial, a memory stays exact ( it only orders the cuts )
                    if constexpr ( requires { PD::has_memo; } ) {
                        if constexpr ( PD::has_memo )
                            diagram::memorize( c, k, pd.memo_nbrs, pd.memo_counts );
                    }
                    return true;
                } );
            } );
            if ( ! overflowed )
                break;
            cap *= 2;
            grow_scratch();
            ++nb_overflowed;
        }
        if ( fa ) {
            fa->clear();
            for ( auto &v : fa_th )
                fa->insert( fa->end(), v.begin(), v.end() );
        }
        if constexpr ( sink ) {                          // the sink takes what the cells leave
            double s = 0;
            for ( SI i = 0; i < n; ++i ) s += a[ i ];
            a_all[ 0 ] = total_mass - s;
        }
        t_diag += now() - t0;
        ++nb_diag;
    }

    /// THE MOMENTS at the current weights : the BARYCENTER of each cell ( its seed if it is empty )
    /// and the transport cost `sum_i int_{cell_i} |x - p_i|^2 rho`, already reduced.
    ///
    /// One more sweep, on the SAME scratch and the same per-thread split as `measures` -- this is
    /// what spares the caller from rebuilding a diagram from Python to obtain a cost
    /// ( `integrate_moments_into` returns `int rho`, `int x rho` and `int |x|^2 rho`, in absolute
    /// coordinates, and the cost of a cell follows in one line ).
    ///
    /// THE MASS DOES NOT COME OUT OF IT, and this is deliberate : for a density that is not constant per
    /// piece, `integrate_moments_into` uses quadrature where `integrate_into` has a closed form ( the
    /// exact reduction of `SumOfGaussians` ) -- measured 7e-4 relative gap on a gaussian. The
    /// mass of a cell is the one Newton measured ( `newton.a` ), not this one.
    void moments( std::vector<double> &bary, double &cost ) {
        const double t0 = now();
        const SI n = pd.nb_seeds();
        std::vector<double> mass( n, 0.0 );
        bary.assign( size_t( n ) * D, 0.0 );
        std::vector<double> cost_th( nt, 0.0 );
        for ( ;; ) {
            std::atomic<bool> overflowed{ false };
            queue.run_threads( nt, [&]( int t ) {
                Carver cv{ scratch[ t ].data(), words };
                Local c, piece;
                c.attach( cv, cap );
                if constexpr ( nbc > 1 ) piece.attach( cv, cap );
                else                     piece = c;
                double acc = 0;
                diagram::for_each_seed<PD>( n, t, nt, [&]( SI k ) {
                    const SI i = pd.user_id( k );
                    if ( ! diagram::make_cell( pd, c, k, dom ) ) {
                        overflowed = true;
                        return false;
                    }
                    TF m = 0, m2 = 0;
                    TF mx[ D ];
                    auto first = [&]( int d ) -> TF & { return mx[ d ]; };
                    if ( ! diagram::integrate_moments_into<TF>( m, first, m2, c, piece, diagram::cell_dist( pd, k, *dist ) ) ) {
                        overflowed = true;
                        return false;
                    }
                    const auto p = pd.point( k );
                    double pp = 0, px = 0;
                    for ( int d = 0; d < D; ++d ) {
                        pp += double( p[ d ] ) * double( p[ d ] );
                        px += double( p[ d ] ) * double( mx[ d ] );
                    }
                    mass[ i ] = double( m );
                    acc += double( m2 ) - 2 * px + double( m ) * pp;
                    // an empty cell keeps its seed as barycenter -- there is nothing else to say,
                    // and dividing by zero downstream would be worse
                    for ( int d = 0; d < D; ++d )
                        bary[ size_t( i ) * D + d ] = double( m ) > 0 ? double( mx[ d ] ) / double( m ) : double( p[ d ] );
                    return true;
                } );
                cost_th[ t ] = acc;
            } );
            if ( ! overflowed )
                break;
            cap *= 2;
            grow_scratch();
            ++nb_overflowed;
        }
        cost = 0;
        for ( double v : cost_th )
            cost += v;
        t_diag += now() - t0;
        ++nb_diag;
    }
};

} // namespace sdotplan
} // namespace sdot
