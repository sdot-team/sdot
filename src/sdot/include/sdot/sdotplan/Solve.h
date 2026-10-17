#pragma once

// =====================================================================================
// THE SOLVER'S ENTRY POINT ( what `SdotPlanNd.py` calls, in ONE `driver.call` ): the starting point, the
// width continuation if one is needed, Newton at each step, and what comes out.
//
// = The starting point
//
// Newton ( KMT ) requires an ADMISSIBLE start: no empty cell. We start from the given weights
// ( `w0`: the weights of a neighboring fit, which a reconstruction lives on ) or from zero -- the
// Voronoi, none of whose cells is empty as long as every seed is in the domain. A warm start
// that EMPTIES a cell ( weights inherited from other positions ) is worse than the Voronoi: the
// theory starts from a strictly positive floor, and an empty cell crawls back to it ( its
// gradient is constant, its hessian row zero ). We then restart from zero if that is better.
// ( two ways of doing better than the Voronoi were tried, and rejected: only "reviving" the empty
// cells, and a multiscale start -- both cascade, see `solvers_des_familles` README § 8. )
//
// And the Voronoi itself can leave empty cells ( seeds OUTSIDE the domain ): then the
// weights of a SIMILITUDE that brings the cloud back into the domain -- the Voronoi of the translated and
// contracted cloud is written as a power diagram of the original cloud, `w_i = |p_i|^2 - |a p_i +
// b|^2 / a`, and its cells are all fed.
//
// = With a sink ( partial transport )
//
// The sink is seed 0 of Newton ( `Sweep.h` ): its target is what the density keeps beyond the seeds' masses, its weight the gauge.
// The start is not the Voronoi -- balls of radius 0 hold nothing -- but BALLS: `w_i = nu_i / ( 2 pi rho )`, `rho` the mean
// density of the domain ( half of what the cell would hold alone, so that the sink keeps some mass ), each ball whose cell is
// still empty grown until it meets the density. No continuation, no rescaling of the targets.
//
// = The width continuation ( `Continuation.h` )
//
// There remains the case where cells have no mass because the DENSITY has none where they are
// ( narrow bumps, deserts ): neither a geometric start nor the damping can do anything about it,
// and direct Newton stagnates ( README § 9.1 ). We then first solve for the density convolved
// by a wide gaussian ( positive everywhere ), then narrower and narrower, each step starting from
// the weights of the previous one. `AUTO` triggers it when the best start still leaves a
// cell under `continuation_threshold` times the smallest target mass. It then starts at the narrowest width
// where the start leaves no cell under that threshold ( a start almost admissible needs little smoothing ), and
// goes down adaptively: to the density itself as soon as the solved weights are admissible for it, by narrower
// widths otherwise, a width where Newton fails being abandoned for a wider one. `ALWAYS` runs the fixed
// ladder of widths, from half the diameter of the domain.
//
// = The target mass
//
// The cells PARTITION the domain: their masses sum to the mass of the density in the
// domain, whatever the weights. A density that the domain truncates ( gaussians in
// a box ) does not weigh 1 there: the target masses `nu` are rescaled to it, at each step,
// without which the residual can never vanish. What comes out is the transport towards the density
// RESTRICTED to the domain, normalized -- which is what is meant when a domain is given.
// =====================================================================================

#include "Continuation.h"
#include "Bounds.h"
#include "Newton.h"
#include "Report.h"

namespace sdot {
namespace sdotplan {

struct SolverOptions {
    NewtonOptions newton;
    Lin    lin = Lin::AUTO;
    LinearOptions lin_options;       ///< the tolerance, the AMG variant ( defaults: those of the solver )
    SI     cap0 = 64;                ///< vertices per local cell, at the start
    enum Continuation : int { NEVER = 0, AUTO = 1, ALWAYS = 2 };
    int    continuation = AUTO;
    double continuation_threshold = 1e-2; ///< AUTO: a cell under this factor of the smallest target triggers it
    double conv_s0 = 0;              ///< the first width ( 0: half the diameter of the domain )
    double conv_ratio = 1.4142135623730951;
    double conv_min = 0;             ///< the last width before 0 ( 0: the scale of the distribution )
    double agg_margin = 0;           ///< the aggregation of near-coincident seeds ( `Aggregation.h` ): `kappa`, 0 to switch it off
    double agg_gap = 0;              ///< ... `w_dup = w_rep - gap` for the exact duplicates
    SI     agg_nb_dups = 0;          ///< ... how many rows of `dups_in` are exact duplicates
    bool   keep_start = false;       ///< the given weights are kept even if they empty a cell ( a stage of a continuation run by the caller )
};

/// the weights of the Voronoi of a SIMILITUDE of the cloud that fits it in the box `[ lo, hi ]`: the box
/// of the cloud is contracted ( never dilated ) and translated into the box reduced by a margin
template<int D>
void similarity( const auto &pd, const double *lo, const double *hi, std::vector<double> &w, double margin = 0.1 ) {
    const SI n = pd.nb_seeds();
    double p_lo[ D ], p_hi[ D ];
    for ( int d = 0; d < D; ++d ) { p_lo[ d ] = 1e300; p_hi[ d ] = -1e300; }
    for ( SI k = 0; k < n; ++k ) {
        const auto p = pd.point( k );
        for ( int d = 0; d < D; ++d ) { p_lo[ d ] = std::min( p_lo[ d ], double( p[ d ] ) ); p_hi[ d ] = std::max( p_hi[ d ], double( p[ d ] ) ); }
    }
    double a = 1;
    for ( int d = 0; d < D; ++d ) {
        const double span_dom = ( hi[ d ] - lo[ d ] ) * ( 1 - 2 * margin );
        const double span_pts = std::max( p_hi[ d ] - p_lo[ d ], 1e-300 );
        a = std::min( a, span_dom / span_pts );
    }
    double b[ D ];
    for ( int d = 0; d < D; ++d )                        // the center of the contracted cloud on the center of the domain
        b[ d ] = ( lo[ d ] + hi[ d ] ) / 2 - a * ( p_lo[ d ] + p_hi[ d ] ) / 2;
    w.assign( n, 0.0 );
    for ( SI k = 0; k < n; ++k ) {
        const auto p = pd.point( k );
        double pp = 0, qq = 0;
        for ( int d = 0; d < D; ++d ) {
            const double q = a * double( p[ d ] ) + b[ d ];
            pp += double( p[ d ] ) * double( p[ d ] );
            qq += q * q;
        }
        w[ pd.user_id( k ) ] = pp - qq / a;
    }
}

inline double minimum( const std::vector<double> &v ) {
    double m = v.empty() ? 0 : v[ 0 ];
    for ( double x : v ) m = std::min( m, x );
    return m;
}

/// THE SOLVER. `pd` carries WRITABLE weights and majorants ( `with_weights` ); `nu` and `w0`
/// are in user order. `start_box_in` ( `2 D + 1` values: a flag, `lo`, `hi` ): a box inside the support of the density, where the
/// similarity packs the cloud ( flag 0: the box of the domain ). `dups_in` ( `2 k` values, the first `o.agg_nb_dups` pairs ): the exact duplicates, `( dup, rep )`.
/// `weights` ( user order ), `hist` ( `nb_steps`, `rows [ step,
/// NB_HIST ]`, `weights [ step, n ]` optional ), `stats`, `masses [ n ]`, `bary [ n, D ]`,
/// `cost` ( cost ) and `clusters [ n ]` ( the representative of each seed's cluster, itself if alone ) are the outputs.
template<class TK>
void solve( const CpuQueue &queue, auto &pd, const auto &pd_in, const auto &dom, const auto &dist, const auto &nu_in, const auto &w0_in,
            const auto &dups_in, const auto &start_box_in, const SolverOptions &o, auto &&weights, auto &&hist, auto &&stats, auto &&masses, auto &&bary, auto &&cost,
            auto &&clusters ) {
    using PD = DECAYED_TYPE_OF( pd );
    using Dist = DECAYED_TYPE_OF( dist );
    constexpr int D = PD::ct_dim;
    const SI n = pd.nb_seeds();
    const double t_begin = now();

    Convolved<Dist> conv( dist );
    Sweep<PD,DECAYED_TYPE_OF( dom ),Dist,TK> bal( queue, pd, pd_in, dom, dist, o.cap0 );
    constexpr bool sink = decltype( bal )::sink;
    constexpr SI off = decltype( bal )::off;            // the user's seeds are `off .. off + n - 1` for Newton ( the sink is 0 )
    const SI nn = n + off;
    if constexpr ( sink )
        bal.total_mass = bal.domain_mass();
    auto lin = linear_solver( o.lin, n, D, o.lin_options );
    lin->order( bal.rank_of );                           // the tree order, for the aggregation of the multigrid
    Newton<decltype( bal )> newton( bal, *lin, o.newton );
    // ---- the aggregation ( `Aggregation.h` ): its settings, the exact duplicates given by the caller
    newton.agg.init( nn );
    newton.agg.margin = o.agg_margin;
    newton.agg.gap = o.agg_gap;
    newton.agg.tol_abs = o.newton.tol_abs;
    newton.agg.tol_rel = o.newton.tol_rel;
    if ( o.agg_nb_dups > 0 ) {
        std::vector<SI> dup( o.agg_nb_dups ), rep( o.agg_nb_dups );
        for ( SI k = 0; k < o.agg_nb_dups; ++k ) { dup[ k ] = off + SI( dups_in( 2 * k ) ); rep[ k ] = off + SI( dups_in( 2 * k + 1 ) ); }
        newton.agg.set_duplicates( dup, rep );
    }

    // ---- the history, one row per accepted step
    SI nb_steps = 0;
    int nb_steps_before = 0;                              // the steps of the previous stages
    double s_current = 0;
    const SI cap_steps = SI( hist.rows.shape( 0 ) );
    newton.o.after_step = [&]( int it, double t, int nb_evals ) {
        if ( nb_steps >= cap_steps ) return;
        const auto &A = newton.a;
        double mn = A.empty() ? 0 : A[ 0 ], mx = 0, l2 = 0;
        for ( SI i = 0; i < nn; ++i ) {
            mn = std::min( mn, A[ i ] );
            mx = std::max( mx, std::fabs( A[ i ] - newton.nu[ i ] ) );
            l2 += ( A[ i ] - newton.nu[ i ] ) * ( A[ i ] - newton.nu[ i ] );
        }
        hist.rows( nb_steps, int( H_STEP ) ) = double( nb_steps_before + it );
        hist.rows( nb_steps, int( H_T ) ) = t;
        hist.rows( nb_steps, int( H_RESIDUAL_L2 ) ) = std::sqrt( l2 );
        hist.rows( nb_steps, int( H_MIN_MASS ) ) = mn;
        hist.rows( nb_steps, int( H_MAX_RESIDUAL ) ) = mx;
        hist.rows( nb_steps, int( H_NB_DIAG ) ) = double( bal.nb_diag );
        hist.rows( nb_steps, int( H_NB_EVALS ) ) = double( nb_evals );
        hist.rows( nb_steps, int( H_S ) ) = s_current;
        if constexpr ( DECAYED_TYPE_OF( hist.weights )::is_valid )
            for ( SI i = 0; i < n; ++i )
                hist.weights( nb_steps, i ) = newton.w[ off + i ];
        ++nb_steps;
    };

    // ---- the target, and the start ( on the widest density if the continuation is forced )
    std::vector<double> nu( nn ), w( nn, 0.0 );
    for ( SI i = 0; i < n; ++i ) nu[ off + i ] = double( nu_in( i ) );
    if constexpr ( sink ) {                              // the sink keeps what the seeds do not take
        double s = 0;
        for ( SI i = 0; i < n; ++i ) s += nu[ off + i ];
        nu[ 0 ] = bal.total_mass - s;
    }
    bool given = false;
    if constexpr ( DECAYED_TYPE_OF( w0_in )::is_valid ) {
        for ( SI i = 0; i < n; ++i ) { w[ off + i ] = double( w0_in( i ) ); given |= w[ off + i ] != 0; }
    }
    int start = given ? START_GIVEN : START_VORONOI;
    // GROUPS ( `Groups.h` ): the weights seed by seed are turned into Newton's unknowns, the diagram's virtual weights kept as they are
    auto as_unknowns = [&]( std::vector<double> &W, bool precise ) {
        if constexpr ( decltype( bal )::groups ) {
            if ( bal.grp.active ) {
                std::vector<double> x;
                bal.grp.unknowns_of( W, x, precise );
                W.swap( x );
            }
        }
    };
    as_unknowns( w, true );
    const double gauge = w[ 0 ];
    for ( SI i = 0; i < n; ++i ) if ( newton.level0( i ) ) w[ i ] -= gauge;

    // the stages of the continuation: the list of widths, `0` last
    double s0 = o.conv_s0;
    if ( s0 <= 0 ) {                                     // half the diameter of the domain, or of the cloud
        double lo[ D ], hi[ D ];
        for ( int d = 0; d < D; ++d ) { lo[ d ] = 1e300; hi[ d ] = -1e300; }
        if constexpr ( DECAYED_TYPE_OF( pd.box_min )::is_valid ) {
            for ( int d = 0; d < D; ++d ) { lo[ d ] = double( pd.box_min( d ) ); hi[ d ] = double( pd.box_max( d ) ); }
        } else {
            for ( SI k = 0; k < n; ++k ) {
                const auto p = pd.point( k );
                for ( int d = 0; d < D; ++d ) { lo[ d ] = std::min( lo[ d ], double( p[ d ] ) ); hi[ d ] = std::max( hi[ d ], double( p[ d ] ) ); }
            }
        }
        double diam2 = 0;
        for ( int d = 0; d < D; ++d ) diam2 += ( hi[ d ] - lo[ d ] ) * ( hi[ d ] - lo[ d ] );
        s0 = 0.5 * std::sqrt( diam2 );
    }
    const double s_min = o.conv_min > 0 ? o.conv_min : conv.min_scale( s0 );
    std::vector<double> scales = ( ! sink && o.continuation == SolverOptions::ALWAYS && Convolved<Dist>::possible ) ? continuation_steps( s0, o.conv_ratio, s_min )
                                                                                                                   : std::vector<double>{ 0.0 };

    std::vector<double> a;
    std::vector<Facet> fa;
    s_current = scales[ 0 ];
    bal.dist = &conv.at( s_current );
    if constexpr ( sink ) {
        if ( ! given ) {                                 // the start by BALLS ( see the head of the file )
            double area = 1;
            if constexpr ( DECAYED_TYPE_OF( pd.box_min )::is_valid )
                for ( int d = 0; d < D; ++d ) area *= std::max( double( pd.box_max( d ) ) - double( pd.box_min( d ) ), 1e-300 );
            const double rho = bal.total_mass / area;
            for ( SI i = 0; i < n; ++i ) w[ off + i ] = nu[ off + i ] / ( 2 * M_PI * rho );
            as_unknowns( w, false );
            start = START_BALLS;
        }
    }
    newton.measures_and_facets( w, a, fa );
    if constexpr ( sink ) {
        if ( ! given ) {                                 // an empty cell: its ball grown until it meets the density
            for ( int round = 0; round < 60; ++round ) {
                bool grown = false;
                for ( SI i = 0; i < n; ++i )
                    if ( ! ( a[ off + i ] > 0 ) ) { w[ off + i ] = 4 * std::max( w[ off + i ], 1e-300 ); grown = true; }
                if ( ! grown ) break;
                newton.measures_and_facets( w, a, fa );
            }
        }
    }
    const double nu_min = minimum( nu );
    // ( the smallest masses below leave the exact duplicates out: their cells are empty by construction )
    auto min_mass = [&]( const std::vector<double> &m ) { return newton.agg.floor_min( m ); };
    if ( ! sink && given && ! o.keep_start && min_mass( a ) < 1e-3 * nu_min ) {      // a warm start that empties a cell: the Voronoi, if it does better
        std::vector<double> w0( n, 0.0 ), a0;
        std::vector<Facet> fa0;
        newton.measures_and_facets( w0, a0, fa0 );
        if ( min_mass( a0 ) > min_mass( a ) ) { w.swap( w0 ); a.swap( a0 ); fa.swap( fa0 ); start = START_VORONOI; }
        else newton.bal.set_weights( w );
    }
    if constexpr ( DECAYED_TYPE_OF( pd.box_min )::is_valid ) {
        if ( ! sink && min_mass( a ) <= 0 && ! ( given && o.keep_start ) ) {   // seeds outside the domain: the similarity
            double lo[ D ], hi[ D ];
            for ( int d = 0; d < D; ++d ) { lo[ d ] = double( pd.box_min( d ) ); hi[ d ] = double( pd.box_max( d ) ); }
            if ( double( start_box_in( 0 ) ) > 0 )       // the density is not positive on all the domain: its inscribed box
                for ( int d = 0; d < D; ++d ) { lo[ d ] = double( start_box_in( 1 + d ) ); hi[ d ] = double( start_box_in( 1 + D + d ) ); }
            std::vector<double> w1, a1;
            std::vector<Facet> fa1;
            similarity<D>( pd, lo, hi, w1 );
            as_unknowns( w1, false );
            newton.measures_and_facets( w1, a1, fa1 );
            if ( min_mass( a1 ) > min_mass( a ) ) { w.swap( w1 ); a.swap( a1 ); fa.swap( fa1 ); start = START_SIMILARITY; }
            else newton.bal.set_weights( w );
        }
    }
    const double min_start_mass = min_mass( a );
    // AUTO: the density is missing where cells are -> the continuation, from the same start. It starts at the NARROWEST width
    // where every cell of the start has mass ( a bisection among `s_min 4^k` ), not at the widest one: a start that is almost
    // admissible ( the barycenters of a Lloyd step, next to a density carried by a surface ) needs little smoothing. And it goes
    // down ADAPTIVELY ( see the stages below )
    bool adaptive = false;
    const double admissible = o.continuation_threshold * nu_min;
    if ( ! sink && o.continuation == SolverOptions::AUTO && Convolved<Dist>::possible && scales.size() == 1
      && ( min_start_mass < admissible ) ) {
        adaptive = true;
        std::vector<double> ladder = continuation_steps( s0, 4.0, s_min );     // ( decreasing, 0 last )
        ladder.pop_back();
        if ( ladder.empty() ) ladder.push_back( s0 );
        std::reverse( ladder.begin(), ladder.end() );                         // ( increasing: s_min ... s0 )
        auto mass_at = [&]( double s, std::vector<double> &W, std::vector<double> &A, std::vector<Facet> &F ) {
            bal.dist = &conv.at( s );
            newton.measures_and_facets( W, A, F );
            return min_mass( A );
        };
        // ( a warm start can empty a cell GEOMETRICALLY ( a power cell of no volume, without mass at any width ): measured on the
        //   widest width, where the density is positive everywhere, the Voronoi instead if it does better there. Without a warm start,
        //   the widest width is the last resort of the bisection, measured only if it is the one chosen )
        PI lo = 0, hi = ladder.size() - 1;
        std::vector<double> a_hi, a_try;
        std::vector<Facet> fa_hi, fa_try;
        if ( given && ! o.keep_start && mass_at( ladder[ hi ], w, a_hi, fa_hi ) < admissible ) {
            std::vector<double> w0( n, 0.0 ), a0;
            std::vector<Facet> fa0;
            if ( mass_at( ladder[ hi ], w0, a0, fa0 ) > min_mass( a_hi ) ) { w.swap( w0 ); a_hi.swap( a0 ); fa_hi.swap( fa0 ); start = START_VORONOI; }
        }
        while ( lo < hi ) {                              // ( the smallest index whose width is admissible )
            const PI mid = ( lo + hi ) / 2;
            if ( mass_at( ladder[ mid ], w, a_try, fa_try ) >= admissible ) { hi = mid; a_hi.swap( a_try ); fa_hi.swap( fa_try ); }
            else lo = mid + 1;
        }
        s_current = ladder[ hi ];
        if ( a_hi.empty() ) mass_at( s_current, w, a_hi, fa_hi );
        bal.dist = &conv.at( s_current );
        newton.bal.set_weights( w );
        a.swap( a_hi );
        fa.swap( fa_hi );
        scales.assign( { s_current } );
    }

    // ---- the stages
    // a trace is read WHILE the solve runs ( and above all when it hangs ): the C stdio of a pipe or a file is block-buffered, and
    // the lines would only come out at the end, if ever
    if ( o.newton.trace )
        std::setvbuf( stdout, nullptr, _IOLBF, 0 );
    NewtonStats total;
    double domain_mass = 0;
    // one stage: Newton at `s_current`, from `w`, whose measures at `s_current` are `a`, `fa`; what it leaves is in `newton`
    PI nb_stages = 0;
    auto run_stage = [&]( int nb_planned ) {
        // the target mass, at the scale of what the domain contains of THIS density
        std::vector<double> nu_s = nu;
        double nu_mass = 0;
        domain_mass = 0;
        for ( SI i = 0; i < nn; ++i ) { domain_mass += a[ i ]; nu_mass += nu[ i ]; }
        if ( ! sink && domain_mass > 0 && nu_mass > 0 && domain_mass != nu_mass )
            for ( SI i = 0; i < n; ++i ) nu_s[ i ] *= domain_mass / nu_mass;
        newton.nu = nu_s;
        newton.a = a;
        newton.fa = fa;
        if ( o.newton.trace && nb_planned > 0 )
            std::printf( "  stage %d / %d : s = %.4e, domain mass %.6f, smallest mass %.3e\n",
                         int( nb_stages + 1 ), nb_planned, s_current, domain_mass, minimum( a ) );
        else if ( o.newton.trace )
            std::printf( "  stage %d ( adaptive ) : s = %.4e, domain mass %.6f, smallest mass %.3e\n",
                         int( nb_stages + 1 ), s_current, domain_mass, minimum( a ) );

        // the step by the limits ( 2D ): the pass is plugged into Newton when it is requested
        if constexpr ( D == 2 && ! sink ) {
            Bounds2D<decltype( bal )> lim( bal );
            LocalBounds ll;
            ll.alpha_min = [&]( const std::vector<double> &W, const std::vector<double> &Dd, const std::vector<SI> &bad_cells,
                                double horizon, double eps, const Laplacian &L, SI &nb_cells ) {
                return lim.alpha_min( W, Dd, bad_cells, horizon, eps, L, nb_cells );
            };
            if ( o.newton.step == NewtonOptions::LIMITS )
                newton.bounds = &ll;
            newton.solves( w, true );
            newton.bounds = nullptr;
        } else
            newton.solves( w, true );

        // its counters ( a stage tried and abandoned costs as much as an accepted one )
        nb_steps_before = nb_steps > 0 ? int( double( hist.rows( nb_steps - 1, int( H_STEP ) ) ) ) + 1 : 0;
        total.nb_iter += newton.st.nb_iter;
        total.nb_backtracks += newton.st.nb_backtracks;
        total.nb_cell_lim += newton.st.nb_cell_lim;
        total.nb_limit_rounds += newton.st.nb_limit_rounds;
        total.t_asm += newton.st.t_asm;
        total.t_lim += newton.st.t_lim;
        total.nb_polish += newton.st.nb_polish;
        if ( nb_stages == 0 ) { total.residual0 = newton.st.residual0; total.eps = newton.st.eps; total.it_switch = newton.st.it_switch; }
        total.status = newton.st.status;
        total.residual = newton.st.residual;
        ++nb_stages;
        const bool ok = newton.st.status == NewtonStats::CONVERGED || newton.st.status == NewtonStats::STAGNATION;
        newton.st = NewtonStats{};
        return ok;
    };
    auto accept = [&]() { w = newton.w; a = newton.a; fa = newton.fa; };

    if ( ! adaptive ) {
        // the given widths, one after the other
        for ( PI step = 0; step < scales.size(); ++step ) {
            s_current = scales[ step ];
            if ( step > 0 ) {                           // the next density: the measures of the start have to be redone
                bal.dist = &conv.at( s_current );
                newton.measures_and_facets( w, a, fa );
            }
            run_stage( int( scales.size() ) );
            accept();
            if ( total.status == NewtonStats::LINEAR_FAILURE )
                break;
        }
    } else if ( run_stage( 0 ) ) {
        // ADAPTIVE: after each solved width `s`, the density itself ( `0` ) if the solved weights are admissible for it, otherwise
        // `s / r`, `r` taken from the last width that worked and squared after two successes in a row ( at most 16 ), its square
        // root after a failure ( at least `conv_ratio` ), and never under `s_min` ( a try costs a sweep of the smoothed density ).
        // At `r = conv_ratio`, the step of the fixed ladder, a width is tried as long as no cell is empty: the threshold only
        // spares the wider steps a Newton that would fail, and the descent is never worse than the ladder. A width is tried only if the solved weights leave no cell
        // under the threshold there, and abandoned ( back to the solved weights ) if Newton does not converge on it
        accept();
        double r = 16;
        int nb_ok = 0;
        auto try_width = [&]( double s, bool forced ) {
            std::vector<double> a1;
            std::vector<Facet> fa1;
            bal.dist = &conv.at( s );
            newton.measures_and_facets( w, a1, fa1 );
            if ( forced ? ! ( min_mass( a1 ) > 0 ) : min_mass( a1 ) < admissible ) {
                if ( o.newton.trace ) std::printf( "  ( s = %.4e: a cell under the threshold, %.3e )\n", s, min_mass( a1 ) );
                return false;
            }
            const double s_prev = s_current;
            s_current = s;
            a.swap( a1 );
            fa.swap( fa1 );
            if ( run_stage( 0 ) ) { accept(); return true; }
            if ( o.newton.trace ) std::printf( "  ( s = %.4e: abandoned )\n", s );
            s_current = s_prev;                         // ( abandoned: the solved weights and their measures again )
            a.swap( a1 );
            fa.swap( fa1 );
            return false;
        };
        while ( s_current > 0 ) {
            // ( from `s_min`, the narrowest width, the density itself is tried even with cells under the threshold, as long as none is
            //   empty: this is the last stage of the fixed ladder, and Newton passes it -- the layers of cells along M that wider
            //   widths leave have all crossed it by then )
            if ( try_width( 0, s_current <= s_min * ( 1 + 1e-12 ) ) )
                break;
            bool moved = false;
            while ( ! moved ) {
                const double s = std::max( s_current / r, s_min );
                if ( s < s_current * ( 1 - 1e-12 ) && try_width( s, r <= o.conv_ratio * ( 1 + 1e-12 ) ) ) {
                    moved = true;
                    if ( ++nb_ok >= 2 ) { r = std::min( r * r, 16.0 ); nb_ok = 0; }
                    break;
                }
                nb_ok = 0;
                if ( r <= o.conv_ratio * ( 1 + 1e-12 ) ) break;
                r = std::max( std::sqrt( r ), o.conv_ratio );
            }
            if ( ! moved ) {                            // ( no narrower width works: the solve stops at `s_current`, not on the density )
                bal.dist = &conv.at( s_current );
                newton.bal.set_weights( w );
                total.status = decltype( total.status )( S_FAILURE );
                break;
            }
        }
    } else
        accept();
    scales.resize( nb_stages );                         // ( for the stats: the stages actually run )
    hist.nb_steps.set( nb_steps );

    // THE FULL PROBLEM'S RESIDUAL, and the status when only the aggregated problem passed the test
    double full = total.residual, full_rel = 0;
    if ( newton.agg.any() ) {
        newton.agg.full_residual( a, newton.nu, full, full_rel );
        if ( total.status == NewtonStats::CONVERGED && ! newton.agg.converged( full, full_rel ) )
            total.status = S_CONVERGED_AGGREGATED;
    }
    std::vector<SI> group_reps;                          // ( with groups: the smallest seed of each one's root aggregate )
    if constexpr ( decltype( bal )::groups )
        if ( bal.grp.active ) group_reps = bal.grp.root_of;
    for ( SI i = 0; i < n; ++i ) {
        const SI j = off + i;
        SI r = newton.agg.any() && newton.agg.cluster_of[ j ] >= 0 ? newton.agg.cl_members[ newton.agg.cl_begin[ newton.agg.cluster_of[ j ] ] ] : j;
        if ( ! group_reps.empty() ) r = group_reps[ j ];
        clusters( i ) = r >= off ? r - off : i;
    }

    // the weights seed by seed ( with groups: `groups.py`'s weights; the precise virtual weights are in the diagram's views )
    std::vector<double> w_out = w;
    if constexpr ( decltype( bal )::groups )
        bal.grp.weights_of( w, w_out );
    for ( SI i = 0; i < n; ++i )
        weights( i ) = w_out[ off + i ];

    // the moments sweep below is not a RESIDUAL EVALUATION: the diagram count
    // the caller reads must remain that of the descent, otherwise two versions of the code
    // can no longer be compared
    const int nb_diag_descent = bal.nb_diag;

    // ---- THE MOMENTS, on the TRUE density and at the fitted weights
    // One more sweep, and that is all that the transport cost, the barycenters and the cell
    // masses require: Python no longer has to rebuild a diagram to get them. On `conv.at(
    // 0 )` explicitly -- a solve that stopped midway through the continuation would otherwise leave
    // moments of a CONVOLVED density, which are not the ones that were asked for.
    // the MASS of a cell is the one Newton measured -- not that of the moments sweep,
    // which uses quadrature where the measure has a closed form ( see `Sweep::moments` ).
    for ( SI i = 0; i < n; ++i )
        masses( i ) = a[ off + i ];
    {
        bal.dist = &conv.at( 0 );
        bal.set_weights( w );
        std::vector<double> bc;
        double c = 0;
        bal.moments( bc, c );
        for ( SI i = 0; i < n; ++i )
            for ( int d = 0; d < D; ++d )
                bary( i, d ) = bc[ size_t( i ) * D + d ];
        cost = c;
    }

    auto put = [&]( int i, double v ) { stats( i ) = v; };
    put( STATUS, double( total.status ) );
    put( RESIDUAL, total.residual );
    put( RESIDUAL0, total.residual0 );
    put( NB_ITER, double( total.nb_iter ) );
    put( NB_DIAG, double( nb_diag_descent ) );
    put( NB_BACKTRACKS, double( total.nb_backtracks ) );
    put( T_MAJORANT, bal.t_majorant );
    put( T_DIAG, bal.t_diag );
    put( T_ASM, total.t_asm );
    put( T_LIN, lin->st.total() );
    put( T_LIM, total.t_lim );
    put( EPS, total.eps );
    put( DOMAIN_MASS, domain_mass );
    put( NB_OVERFLOWED, double( bal.nb_overflowed ) );
    put( NB_CELL_LIM, double( total.nb_cell_lim ) );
    put( NB_LIMIT_ROUNDS, double( total.nb_limit_rounds ) );
    put( LIN_NB_HIERARCHIES, double( lin->st.nb_hierarchies ) );
    put( LIN_NB_ITER, double( lin->st.nb_iter ) );
    put( LIN_WORST, lin->st.worst );
    put( START, double( start ) );
    put( T_TOTAL, now() - t_begin );
    put( NB_CONTINUATION_STEPS, double( scales.size() ) );
    put( MIN_START_MASS, min_start_mass );
    put( IT_SWITCH, double( total.it_switch ) );
    put( IT_DOUBLE, 0 );                                 // ( one kernel: the card's mixed precision only )
    put( SCRATCH_BYTES, 0 );                             // ( the card's pool only )
    put( NB_CLUSTERS, double( newton.agg.nb_clusters() ) );
    put( NB_AGGREGATED, double( newton.agg.nb_aggregated() ) );
    put( NB_DUPLICATES, double( newton.agg.dups.size() ) );
    if constexpr ( decltype( bal )::groups ) {          // ( the tree of aggregates: its roots, their seeds, the duplicates )
        if ( bal.grp.active ) {
            double r, mm, dd;
            bal.grp.counts( r, mm, dd );
            put( NB_CLUSTERS, r ); put( NB_AGGREGATED, mm ); put( NB_DUPLICATES, dd );
        }
    }
    put( RESIDUAL_FULL, full );
    put( NB_POLISH, double( total.nb_polish ) );
}

} // namespace sdotplan
} // namespace sdot
