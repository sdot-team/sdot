#pragma once

// =====================================================================================
// WHAT A POWER DIAGRAM COMPUTES, whatever the way it stores its seeds.
//
// The cell of seed `i` is where its POWER DISTANCE wins: `|x - d_i|^2 - w_i <=
// |x - d_j|^2 - w_j` for every other `j`. Expanded, the inequality loses its `|x|^2` and becomes a
// half-space: a power diagram costs exactly what a Voronoi costs, one plane per
// rival, the same cut ( `cell/Plane.h::bisector` ).
//
// There is NO diagram here: not a vertex, not a facet. Everything is redone, cell by cell,
// in the work-item's scratch ( `Cell.py::CellScratch` ): a local cell, a second one
// for the pieces of a distribution, and the cotangents of the adjoint. THE CELL LEADS
// ( `cell/Engine.h` ): it asks a provider for its planes, and it is the STORAGE of the
// diagram that says which one -- all the seeds in order for `PowerDiagram_Plain`, the flipped BSP tree
// for `PowerDiagram_Bsp`.
//
// = WHAT A STORAGE PROVIDES
//
//     ct_dim, TF, has_weights
//     SI   nb_seeds() const                   the seeds, IN ITS ORDER
//     auto point( SI k ) const                seed `k` ( Vector<TF,ct_dim> ), weight `weight( k )`
//     SI   user_id( SI k ) const              the index of this seed for the USER
//     auto provider<TK>( SI k0 ) const     what yields the planes of the cell of `k0`
//
// Cells are built in STORAGE ORDER ( `k`, what a provider puts in the cut
// identifiers, and the index of the gradients on the seeds ); what goes out to the user
// -- a measure, a kept cell and its `cut_ids` -- is written at index `user_id( k )`.
// =====================================================================================

#include <loom/support/math.h>
#include <loom/support/common_macros.h>
#include <loom/support/containers/Matrix.h>
#include <loom/support/containers/Vector.h>
#include <loom/support/atomic_add.h>
#include <limits>
#include "../cell/Providers.h"
#include "../cell/Engine.h"
#include "../engine/Cells.h"
#include "../cell/Ops.h"
#include "../PieceWorkspace.h"
#include "../UnitDensity.h"
#include "Moments.h"
#include "../Periodic.h"
#include "../integration/InBall.h"
#include "../integration/RadialExp.h"

#include <limits>

namespace sdot {
namespace diagram {

/// one cotangent per vertex, in the scratch
template<class TF>
struct GradVp {
    TF *g;
    SI  cap;
    HD TF &operator()( int i, int d ) { return g[ d * cap + i ]; }
    HD TF  operator()( int i, int d ) const { return g[ d * cap + i ]; }
};

/// a distribution that SPLITS asks for a second cell; `UnitDensity` and smooth
/// densities do not
template<class Dist>
HD constexpr int nb_work_cells() {
    if constexpr ( requires { Dist::cuts_pieces; } ) return Dist::cuts_pieces ? 2 : 1;
    else return 1;
}

/// HOW THE SCRATCH IS CARVED UP for a work-item: `nb_cells` local cells of `cap` vertices, and ( for
/// the adjoint ) one cotangent per vertex -- THE SAME FORMULA as `PowerDiagram._scratch_words`.
template<class Local,class TF>
HD SI words_for( SI cap, int nb_cells, bool with_grad ) {
    return nb_cells * Local::words_for( cap ) + ( with_grad ? words_of<TF>( Local::ct_dim * cap ) : 0 );
}

template<class Local,class TF>
HD SI cap_in( SI nb_words, int nb_cells, bool with_grad ) {
    return cap_for_words( nb_words, [&]( SI c ) { return words_for<Local,TF>( c, nb_cells, with_grad ); } );
}

/// The cell of `k0`, built in `c` from the domain `dom`. Returns `false` if the scratch
/// was not enough -- the cell is then left in its last valid state.
template<class PD,class Local>
HD bool make_cell( const PD &pd, Local &c, SI k0, const auto &dom ) {
    using TK = typename Local::TKernel;
    if ( ! c.load( dom ) )
        return false;
#ifdef SDOT_ENGINE_DEVICE_DEBUG
    static_assert( ! PD::on_cpu, "A: on the cpu" );
    static_assert( requires { pd.engine_seeds(); }, "B: no engine_seeds" );
    static_assert( engine::has_unit( PD::ct_dim ), "C: no unit for this dimension" );
    static_assert( std::is_same_v<Local,engine::LocalFor<PD::ct_dim,TK>>, "D: the cell is not the one of the unit" );
#endif
    // on the CPU the engine is a shared library ( `engine/Cells.h` ), where the storage can describe itself without tensor types
#ifdef SDOT_NO_ENGINE_UNIT      // the engine inlined in every kernel, as it was ( to measure what the unit costs: `bench/engine_unit.py` )
    if constexpr ( false ) {
#else
    if constexpr ( PD::on_cpu && requires { pd.engine_seeds(); } && ! is_periodic<PD>() && ! has_groups<PD>() && engine::has_unit( PD::ct_dim )
                   && std::is_same_v<Local,engine::LocalFor<PD::ct_dim,TK>> ) {
#endif
        return engine::engine_run( c, pd.engine_seeds(), k0 ) != CutStatus::NO_ROOM;
#if defined( __CUDACC__ ) && ( defined( SDOT_ENGINE_DEVICE_NOINLINE ) || defined( SDOT_ENGINE_DEVICE_EXTERN ) )
    } else if constexpr ( ! PD::on_cpu && requires { pd.engine_seeds(); } && ! is_periodic<PD>() && ! has_groups<PD>() && engine::has_unit( PD::ct_dim )
                          && std::is_same_v<Local,engine::LocalFor<PD::ct_dim,TK>> ) {
        return engine::engine_run_device( c, pd.engine_seeds(), k0 ) != CutStatus::NO_ROOM;
#endif
    } else {
        auto f = pd.template provider<TK>( k0 );
        return run<PD::on_cpu>( c, f ) != CutStatus::NO_ROOM;
    }
}

/// the plane of cut `k` of `cell`, in the positions' float: the bisector REBUILT
/// from the seeds when the cut faces a seed, the plane read back from the geometry otherwise
template<class PD>
HD void plane_of( const PD &pd, const auto &cell, SI k0, int k, auto &dir, typename PD::TF &off ) {
    using TF = typename PD::TF;
    constexpr int D = PD::ct_dim;
    const int id = cell.cid[ k ];
    if constexpr ( has_groups<PD>() ) {
        if ( id >= 0 && pd.same_group( k0, id ) ) {      // a plane of the group ( `group_plane` )
            TF dg[ D ];
            pd.group_plane( k0, id, dg, off );
            for ( int d = 0; d < D; ++d )
                dir[ d ] = dg[ d ];
            return;
        }
    }
    if ( id >= 0 ) {
        const auto p0 = pd.point( k0 ), p1 = pd.point( id );
        off = ( pd.weight( k0 ) - pd.weight( id ) ) / 2;
        for ( int d = 0; d < D; ++d ) {
            dir[ d ] = p1[ d ] - p0[ d ];
            off += dir[ d ] * ( p0[ d ] + p1[ d ] ) / 2;
        }
    } else {
        TF dk[ D ];
        cell.plane( k, dk, off );
        for ( int d = 0; d < D; ++d )
            dir[ d ] = dk[ d ];
    }
}

// ---- WHAT we integrate against -------------------------------------------------------------------

/// WHAT IS INTEGRATED over the cell of `k`: the distribution, or -- with a SINK ( partial transport, `sink_weight` ) -- the
/// distribution inside the ball `|x - p_k|^2 <= w_k - w_s` only ( `integration/InBall.h` ). The cell stays the Laguerre
/// cell: it is the integration that changes.
/// With a RADIAL integrand ( `radial_func` ), the distribution times `exp( ( w_k - |x - p_k|^2 ) / ( 2 eps ) )`
/// ( `integration/RadialExp.h` ).
template<class Dens>
HD constexpr bool depends_on_seed() {
    if constexpr ( requires { Dens::depends_on_seed; } ) return bool( Dens::depends_on_seed );
    else return false;
}

template<class PD>
HD constexpr bool has_sink() {
    if constexpr ( requires { PD::has_sink; } ) return PD::has_sink;
    else return false;
}

template<class PD>
HD constexpr bool has_radial() {
    if constexpr ( requires { PD::has_radial; } ) return PD::has_radial;
    else return false;
}

/// what makes `d m_k / d w_k` more than the sum of the facets ( `hessian_row`: one more slot )
template<class PD>
HD constexpr bool has_weight_slot() { return has_sink<PD>() || has_radial<PD>(); }

template<class PD>
HD decltype( auto ) cell_dist( const PD &pd, SI k, const auto &dist ) {
    static_assert( ! ( has_sink<PD>() && has_radial<PD>() ), "a sink and a radial integrand together: not written yet" );
    if constexpr ( has_radial<PD>() ) {
        // as for the ball of a sink: the radial factor goes UNDER the moment
        if constexpr ( requires { DECAYED_TYPE_OF( dist )::moment_order; } )
            return moment_dist<DECAYED_TYPE_OF( dist )::moment_order>( integration::radial_exp( dist.dist, pd.point( k ), pd.weight( k ), pd.radial() ) );
        else
            return integration::radial_exp( dist, pd.point( k ), pd.weight( k ), pd.radial() );
    } else if constexpr ( has_sink<PD>() ) {
        // a moment of the density integrates its polynomial INSIDE the ball: the ball goes under it
        if constexpr ( requires { DECAYED_TYPE_OF( dist )::moment_order; } )
            return moment_dist<DECAYED_TYPE_OF( dist )::moment_order>( integration::in_ball( dist.dist, pd.point( k ), pd.weight( k ) - pd.sink() ) );
        else
            return integration::in_ball( dist, pd.point( k ), pd.weight( k ) - pd.sink() );
    } else
        return ( dist );
}

/// the `D + 1` vertices of simplex `chain`, as points -- what a density receives
template<class TF,int D>
HD auto simplex_points( const auto &cell, const auto &chain ) {
    return Vector<Vector<TF,D>,D+1>( Function(), [&]( PI k ) {
        return Vector<TF,D>::with_func( [&]( PI c ) { return TF( cell.coord( int( chain[ k ] ), int( c ) ) ); } );
    } );
}

/// `res` = the integral of `dist` over `cell`. The distribution SPLITS, we INTEGRATE: on a piece of
/// constant density, `value * measure`; otherwise the piece is split into simplices and it is the density
/// that is integrated over each. Returns `false` if a piece did not fit.
template<class TF,class Local>
HD bool integrate_into( auto &&res, const Local &cell, Local &piece, const auto &dist ) {
    constexpr int D = Local::ct_dim;
    PieceWorkspace<Local> ws{ piece };
    TF sum = 0;
    if constexpr ( requires { dist.integrate_cell( cell, ws, sum ); } ) {   // ( a distribution that integrates a cell itself, when it can )
        if ( dist.integrate_cell( cell, ws, sum ) ) {
            res = sum;
            return ! ws.overflow;
        }
    }
    dist.for_each_piece( cell, ws, [&]( const auto &pc, const auto &dens ) {
        if constexpr ( DECAYED_TYPE_OF( dens )::is_constant ) {
            sum += dens.value * pc.template measure<TF>();
        } else {
            // an unbounded cell has no simplices that mean anything
            if ( ! pc.bounded() ) {
                sum = std::numeric_limits<TF>::max();
                return;
            }
            pc.for_each_simplex( [&]( const auto &chain ) {
                sum += dens.integrate_over_simplex( simplex_points<TF,D>( pc, chain ) );
            } );
        }
    } );
    res = sum;
    return ! ws.overflow;
}

/// the volume of simplex `pts`: `| det( p_i - p_0 ) | / D!`
template<class TF,int D>
HD TF simplex_volume( const auto &pts ) {
    const auto M = Matrix<TF,D>::with_func( [&]( auto r, auto c ) { return pts[ int( c ) + 1 ][ int( r ) ] - pts[ 0 ][ int( r ) ]; } );
    TF det = M.determinant();
    if ( det < 0 ) det = - det;
    for ( int i = 2; i <= D; ++i )
        det /= i;
    return det;
}

/// the MOMENTS of `dist` over `cell`: `mass = int rho`, `first = int x rho`, `second = int |x|^2 rho`
/// -- what a transport cost needs ( `sum_i int_{cell_i} |x - p_i|^2 rho` ) and its
/// barycenters. Each piece is split into simplices: with CONSTANT density ( `Image`, Lebesgue ) their
/// moments are closed forms -- `int_T x = |T| g`, `g` the center, and
/// `int_T |x|^2 = |T| ( sum_i |v_i|^2 + |sum_i v_i|^2 ) / ( ( D + 1 )( D + 2 ) )` -- otherwise it is the
/// quadrature of the density that accumulates them ( `PointwiseDensity::integrate_moments_over_simplex` ).
template<class TF,class Local>
HD bool integrate_moments_into( auto &&mass, auto &&first, auto &&second, const Local &cell, Local &piece, const auto &dist ) {
    constexpr int D = Local::ct_dim;
    PieceWorkspace<Local> ws{ piece };
    TF m = 0, m2 = 0;
    auto mx = Vector<TF,D>::zeros();
    if constexpr ( requires { dist.integrate_moments_cell( cell, ws, m, mx, m2 ); } ) {   // ( as `integrate_into`'s `integrate_cell` )
        if ( dist.integrate_moments_cell( cell, ws, m, mx, m2 ) ) {
            mass = m;
            for ( int d = 0; d < D; ++d )
                first( d ) = mx[ d ];
            second = m2;
            return ! ws.overflow;
        }
    }
    dist.for_each_piece( cell, ws, [&]( const auto &pc, const auto &dens ) {
        if ( ! pc.bounded() ) {
            m = std::numeric_limits<TF>::max();
            return;
        }
        if constexpr ( ! DECAYED_TYPE_OF( dens )::is_constant ) {
            pc.for_each_simplex( [&]( const auto &chain ) {
                dens.integrate_moments_over_simplex( simplex_points<TF,D>( pc, chain ), m, mx, m2 );
            } );
        } else {
            const TF rho = TF( dens.value );
            pc.for_each_simplex( [&]( const auto &chain ) {
                const auto pts = simplex_points<TF,D>( pc, chain );
                const TF w = rho * simplex_volume<TF,D>( pts );
                auto sum = Vector<TF,D>::zeros();
                TF sq = 0;
                for ( int k = 0; k <= D; ++k ) {
                    for ( int d = 0; d < D; ++d )
                        sum[ d ] += pts[ k ][ d ];
                    sq += norm_2_p2( pts[ k ] );
                }
                m += w;
                for ( int d = 0; d < D; ++d )
                    mx[ d ] += w * sum[ d ] / ( D + 1 );
                m2 += w * ( sq + norm_2_p2( sum ) ) / ( ( D + 1 ) * ( D + 2 ) );
            } );
        }
    } );
    mass = m;
    for ( int d = 0; d < D; ++d )
        first( d ) = mx[ d ];
    second = m2;
    return ! ws.overflow;
}

/// THE SEEDS OF A WORK-ITEM, `body( k )` for each, until it returns `false`. On the CPU the storage order is cut
/// into BLOCKS of `seed_block` seeds, dealt to the work-items in turn: inside a block two consecutive seeds
/// are neighbors in space, so the tree, the memories and the neighbors' data stay in the core's cache and
/// two cores never write the same line of `memo_*`; and since a work-item owns many blocks scattered over
/// the whole cloud, a dense region ( where cells cost much more ) is shared by all of them -- a single
/// contiguous slice per work-item left the unlucky one alone with it ( +20 % on a clustered cloud ). On a GPU
/// it is the strided loop, which keeps the accesses of a warp together.
static constexpr SI seed_block = 256;

template<class PD>
HD void for_each_seed( SI n, SI thread_index, SI nb_threads, auto &&body ) {
    if constexpr ( PD::on_cpu ) {
        for ( SI b = thread_index * seed_block; b < n; b += nb_threads * seed_block ) {
            const SI e = b + seed_block < n ? b + seed_block : n;
            for ( SI k = b; k < e; ++k )
                if ( ! body( k ) )
                    return;
        }
    } else {
        for ( SI k = thread_index; k < n; k += nb_threads )
            if ( ! body( k ) )
                return;
    }
}

/// `res( k )` = the measure of cell `k`, for the seeds of this work-item ( `for_each_seed` ): `nb_threads`
/// work-items share the cells, in storage order. `scratch` is the work-item's working tensor --
/// and where we say it was too small.
/// THE MEMORY of a cell ( `ProviderBsp`, `MEMO` ): its neighbors, as sorted ranks, written into
/// `memo_nbrs( k, . )` / `memo_counts( k )` -- or nothing ( `0`: zero count ) if they exceed the
/// capacity. `memo_*` are `0` when the call does not want them.
template<class Local>
HD void memorize( Local &c, SI k, auto &&memo_nbrs, auto &&memo_counts ) {
    if constexpr ( requires { memo_nbrs( k, 0 ); } ) {
        c.tidy();
        const int nc = c.nb_cuts(), cap = int( memo_nbrs.shape( 1 ) );
        int m = 0;
        for ( int q = 0; q < nc && m <= cap; ++q ) {
            const int id = c.cid[ q ];
            if ( id < 0 ) continue;                      // the domain
            if ( m == cap ) { m = cap + 1; break; }      // too many for the capacity: no memory
            int r = m++;                                 // insertion, sorted ascending
            while ( r > 0 && int( memo_nbrs( k, r - 1 ) ) > id ) { memo_nbrs( k, r ) = memo_nbrs( k, r - 1 ); --r; }
            memo_nbrs( k, r ) = id;
        }
        memo_counts( k ) = m > cap ? 0 : m;
    }
}

template<class PD>
HD void measures( const PD &pd, auto &&res, const auto &dom, auto &&scratch, const auto &dist,
               auto &&memo_nbrs, auto &&memo_counts, SI thread_index, SI nb_threads ) {
    using TF    = typename PD::TF;
    using TK    = KernelType<DECAYED_TYPE_OF( scratch )>;
    using Local = typename DECAYED_TYPE_OF( dom )::template Local<TK>;
    constexpr int nbc = nb_work_cells<DECAYED_TYPE_OF( dist )>();
    Carver cv = carver_of( scratch );
    const SI cap = cap_in<Local,TF>( cv.nb_words, nbc, false );
    Local c, piece;
    c.attach( cv, cap );
    if constexpr ( nbc > 1 ) piece.attach( cv, cap );
    else                     piece = c;                  // never touched: the density does not split

    for_each_seed<PD>( pd.nb_seeds(), thread_index, nb_threads, [&]( SI k ) {
        if ( ! make_cell( pd, c, k, dom ) || ! integrate_into<TF>( res( pd.user_id( k ) ), c, piece, cell_dist( pd, k, dist ) ) ) {
            ask_more( scratch, cv );
            return false;
        }
        memorize( c, k, memo_nbrs, memo_counts );
        return true;
    } );
}

/// the moments of each cell ( see `integrate_moments_into` ), same sweep as `measures`
template<class PD>
HD void moments( const PD &pd, auto &&mass, auto &&first, auto &&second, const auto &dom, auto &&scratch, const auto &dist,
              SI thread_index, SI nb_threads ) {
    using TF    = typename PD::TF;
    using TK    = KernelType<DECAYED_TYPE_OF( scratch )>;
    using Local = typename DECAYED_TYPE_OF( dom )::template Local<TK>;
    constexpr int nbc = nb_work_cells<DECAYED_TYPE_OF( dist )>();
    Carver cv = carver_of( scratch );
    const SI cap = cap_in<Local,TF>( cv.nb_words, nbc, false );
    Local c, piece;
    c.attach( cv, cap );
    if constexpr ( nbc > 1 ) piece.attach( cv, cap );
    else                     piece = c;

    for_each_seed<PD>( pd.nb_seeds(), thread_index, nb_threads, [&]( SI k ) {
        const SI u = pd.user_id( k );
        if ( ! make_cell( pd, c, k, dom ) || ! integrate_moments_into<TF>( mass( u ), first( u ), second( u ), c, piece, cell_dist( pd, k, dist ) ) ) {
            ask_more( scratch, cv );
            return false;
        }
        return true;
    } );
}

// ---- the adjoint ---------------------------------------------------------------------------------
// The chain is `m_k <- vertices <- planes <- seeds`, and each arrow is a closed form:
// `measure_bwd` answers the first; a vertex is the CORNER of its `D` cuts, so it solves
// `A x = b` with the cut directions as rows, and a small solve per vertex sends
// its cotangent back onto its planes ( `scatter_cell_grad` ); a plane is the weighted bisector of
// two seeds, which differentiates in two lines. Nothing of the forward is kept: the cell is REBUILT.

/// the storage rank of the seed of identifier `k` -- itself, or for an IMAGE ( periodicity ) the seed it is the image of
template<class PD>
HD SI seed_rank( const PD &pd, SI k ) {
    if constexpr ( is_periodic<PD>() ) return pd.seed_of( k );
    else return k;
}

/// THE DERIVATIVE WITH RESPECT TO THE FRAME OF A DENSITY that is the affine image of a fixed one ( an `Image`: `x = o + A t`,
/// `A = F^T` ), accumulated over the pieces of a cell. In the coordinates `t` of the density nothing moves but the facets of the
/// seeds, so `dJ = tr( A^-1 dA ) J + Int grad p . V rho - sum_{v,r} u_r dir_r . V( x_v )` with `V( x ) = dA A^-1 ( x - o ) + do` --
/// the last sum over the seed rows of the vertex adjoint ( `scatter_cell_grad` ), `p` the polynomial of a moment. So
/// `dJ = tr( X^T dA A^-1 ) + y . do`: `X` and `y` are what is accumulated here, and the density turns them into its own cotangents.
template<class TF,int D>
struct AffineAcc {
    TF o[ D ];                                           ///< the origin of the density
    TF X[ D ][ D ];
    TF y[ D ];
};

/// a piece density that gives its share of `AffineAcc` itself ( `MomentDensity` of a constant ): a marker, which nvcc reads where it
/// does not always read a `requires` on a member
template<class Dens>
HD constexpr bool has_affine_terms() {
    if constexpr ( requires { Dens::has_affine_terms; } ) return bool( Dens::has_affine_terms );
    else return false;
}

/// a distribution that is the affine image of a fixed density says so: `affine_origin( o )` ( and `add_affine_grad` )
template<class Dist>
HD constexpr bool is_affine() {
    if constexpr ( requires { Dist::is_affine; } ) return bool( Dist::is_affine );
    else return false;
}

/// THE COTANGENTS OF THE GROUP MEMBERS ( `diagram/Common.h`, groups ): the virtual weights of the tree's children, seeds and
/// aggregates -- `NoGroupGrads` for a diagram without groups. The planes inside a group send their weights' cotangents there, in
/// the frame of the two seeds' lowest common aggregate, never to the outer weights: summed over a pair `delta` apart, their
/// terms in `1 / delta` would cancel only to the rounding ( `eps / delta` on the outer weights )
struct NoGroupGrads {};

template<class GV,class GA>
struct GroupGrads {
    GV vweights;
    GA agg_vweights;
};

HD auto group_grads( auto &&vweights, auto &&agg_vweights ) {
    return GroupGrads<DECAYED_TYPE_OF( vweights ),DECAYED_TYPE_OF( agg_vweights )>{ vweights, agg_vweights };
}

/// nothing to send to the members ( no groups, or no cotangent asked for them )
HD constexpr bool no_group_grad( const auto &g ) {
    if constexpr ( requires { g.vweights; g.agg_vweights; } )
        return DECAYED_TYPE_OF( g.vweights )::surely_null && DECAYED_TYPE_OF( g.agg_vweights )::surely_null;
    else
        return true;
}

/// the cotangent `g` of the weight of seed `k` in the frame of aggregate `L` ( `diagram/Common.h::frame_weight` ) onto the virtual
/// weights of the nodes between them: `V = v_child + ( o_child / s )^2 V_child`, from the top
template<class PD>
HD void scatter_frame_weight( const PD &pd, SI k, SI L, typename PD::TF g, auto &&grad_groups ) {
    using TF = typename PD::TF;
    if constexpr ( requires { grad_groups.vweights; grad_groups.agg_vweights; } ) {
        SI chain[ 64 ];                                  // the aggregates from the parent of `k` up to `L` ( excluded )
        int nc = 0;
        for ( SI a = pd.seed_parent( k ); a >= 0 && a != L && nc < 64; a = pd.agg_parent( a ) ) chain[ nc++ ] = a;
        TF f = g;
        for ( int c = nc - 1; c >= 0; --c ) {
            const SI a = chain[ c ];
            if constexpr ( ! DECAYED_TYPE_OF( grad_groups.agg_vweights )::surely_null )
                atomic_add( grad_groups.agg_vweights( a ).ref(), f );
            const TF r = pd.agg_outer( a ) / pd.agg_scale( pd.agg_parent( a ) );
            f *= r * r;
        }
        if constexpr ( ! DECAYED_TYPE_OF( grad_groups.vweights )::surely_null )
            atomic_add( grad_groups.vweights( k ).ref(), f );
    }
}

/// `grad_vp` ( one cotangent per vertex of `cell` ) -> the seeds. Every seed other than the cell
/// touches receives an atomic add; the share of `k0`, to which EVERY vertex contributes, is
/// summed in a register and added once.
template<class PD>
HD void scatter_cell_grad( const PD &pd, SI k0, const auto &cell, const auto &grad_vp, auto &&grad_positions, auto &&grad_weights,
                           auto &&grad_groups, AffineAcc<typename PD::TF,PD::ct_dim> *acc = nullptr ) {
    using TF = typename PD::TF;
    constexpr int D = PD::ct_dim;
    if ( DECAYED_TYPE_OF( grad_positions )::surely_null && DECAYED_TYPE_OF( grad_weights )::surely_null && no_group_grad( grad_groups ) && ! acc ) {
        return;
    } else {
        if ( ! cell.bounded() || cell.nb_vertices() == 0 )
            return;

        auto atomic_add_to = []( auto &&dst, TF v ) {
            if constexpr ( ! DECAYED_TYPE_OF( dst )::surely_null )
                atomic_add( dst.ref(), v );
        };

        const auto p0 = pd.point( k0 );
        auto acc_p0 = Vector<TF,D>::zeros();
        TF acc_w0 = 0;

        const int nv = cell.nb_vertices();
        for ( int v = 0; v < nv; ++v ) {
            const auto q = Vector<TF,D>::with_func( [&]( PI d ) { return grad_vp( v, int( d ) ); } );
            const auto x = Vector<TF,D>::with_func( [&]( PI d ) { return TF( cell.coord( v, int( d ) ) ); } );

            // `A x = b`, rows = the directions of the vertex's cuts. A cotangent `q` on `x`
            // reaches the planes through `u` with `A^T u = q`: `d off_r -> u[ r ]`, `d dir_r -> - u[ r ] * x`.
            // Rows that do not face a seed are read back from the geometry: their scale
            // changes nothing for the `u` of the rows that matter.
            Matrix<TF,D> At;
            int cuts[ D ];
            for ( int r = 0; r < D; ++r ) {
                cuts[ r ] = cell.vertex_cut( v, r );
                Vector<TF,D> dir;
                TF off;
                plane_of( pd, cell, k0, cuts[ r ], dir, off );
                for ( int c = 0; c < D; ++c )
                    At( c, r ) = dir[ c ];
            }
            const auto u = Matrix<TF,D>::solve_ge( At, q );

            for ( int r = 0; r < D; ++r ) {
                const int k1 = cell.cid[ cuts[ r ] ];
                if ( k1 < 0 )                            // the domain, a piece, or a fake wall
                    continue;
                // the plane of the pair `( k0, k1 )`: `dir = p1 - p0`,
                // `off = ( |p1|^2 - |p0|^2 ) / 2 + ( w0 - w1 ) / 2`. Both differentiate in place.
                const TF g_off = u[ r ];
                if ( acc )                               // ( the facet moved by `- V`, see `AffineAcc` )
                    for ( int a = 0; a < D; ++a ) {
                        acc->y[ a ] -= g_off * At( a, r );
                        for ( int b = 0; b < D; ++b )
                            acc->X[ a ][ b ] -= g_off * At( a, r ) * ( x[ b ] - acc->o[ b ] );
                    }
                if constexpr ( has_groups<PD>() ) {
                    if ( pd.same_group( k0, k1 ) ) {
                        // a plane of the group ( `group_plane` ): the bisector of two virtual diracs divided by the scale `s` of
                        // their frame -- `u` scales by it. Distinct positions: the TRUE bisector of `( p, W )`, onto the positions and
                        // the outer weights like any plane; duplicates: that of `a = apex + s u_k` with the weights `s^2 v_k`, onto the
                        // positions ( the apex follows the site ) and the virtual weights
                        TF V0, V1;
                        const SI L = pd.common_aggregate( k0, k1, V0, V1 );
                        const TF s = pd.agg_scale( L );
                        const bool dups = pd.agg_of_duplicates( L );
                        const TF g_true = g_off / s;
                        const auto p1 = pd.point( k1 );
                        for ( int d = 0; d < D; ++d ) {
                            const TF sh = dups ? TF( pd.group_agg_center( L, d ) ) : TF( 0 );     // ( the apex's shift from the site )
                            const TF a0 = dups ? TF( p0[ d ] ) + sh + s * TF( pd.group_dirs( k0, d ) ) : TF( p0[ d ] );
                            const TF a1 = dups ? TF( p1[ d ] ) + sh + s * TF( pd.group_dirs( k1, d ) ) : TF( p1[ d ] );
                            const TF g_dir = - g_true * x[ d ];
                            atomic_add_to( grad_positions( k1, d ), g_dir + g_true * a1 );
                            acc_p0[ d ] -= g_dir + g_true * a0;
                        }
                        // the weights: `( W0 - W1 ) / 2 = s^2 ( V0 - V1 ) / 2` in the frame of `L` ( the duplicates' too: `s` their radius )
                        scatter_frame_weight( pd, k0, L, s * s * g_true / 2, grad_groups );
                        scatter_frame_weight( pd, k1, L, - s * s * g_true / 2, grad_groups );
                        continue;
                    }
                }
                const auto p1 = pd.point( k1 );          // an image: translated, and its derivative is that of its seed
                const SI   s1 = seed_rank( pd, k1 );
                TF g1[ D ];                              // the cotangent on the image point
                for ( int d = 0; d < D; ++d ) {
                    const TF g_dir = - g_off * x[ d ];
                    g1[ d ] = g_dir + g_off * p1[ d ];
                    acc_p0[ d ] -= g_dir + g_off * p0[ d ];
                }
                if constexpr ( requires { PD::periodic_linear_part; } ) {
                    if constexpr ( PD::periodic_linear_part ) {
                        // an image `L p + s`: its cotangent goes back to the seed through `L^T`
                        const SI t = SI( k1 ) / pd.nb_seeds();
                        TF g2[ D ];
                        for ( int c = 0; c < D; ++c ) {
                            g2[ c ] = 0;
                            for ( int r = 0; r < D; ++r )
                                g2[ c ] += pd.image_linear( t, r, c ) * g1[ r ];
                        }
                        for ( int d = 0; d < D; ++d )
                            g1[ d ] = g2[ d ];
                    }
                }
                if constexpr ( has_groups<PD>() ) {
                    // a SITE of exact duplicates is seen from outside through planes that coincide: the cell keeps one of them, and
                    // its share goes to all the members alike ( the mean of the one-sided derivatives, what a centred difference sees )
                    const SI nb = pd.for_each_in_site( s1, []( SI ) {} );
                    if ( nb > 1 ) {
                        pd.for_each_in_site( s1, [&]( SI j ) {
                            for ( int d = 0; d < D; ++d )
                                atomic_add_to( grad_positions( j, d ), g1[ d ] / TF( nb ) );
                            atomic_add_to( grad_weights( j ), - g_off / 2 / TF( nb ) );
                        } );
                        acc_w0 += g_off / 2;
                        continue;
                    }
                }
                for ( int d = 0; d < D; ++d )
                    atomic_add_to( grad_positions( s1, d ), g1[ d ] );
                atomic_add_to( grad_weights( s1 ), - g_off / 2 );
                acc_w0 += g_off / 2;
            }
        }

        for ( int d = 0; d < D; ++d )
            atomic_add_to( grad_positions( k0, d ), acc_p0[ d ] );
        atomic_add_to( grad_weights( k0 ), acc_w0 );
    }
}

template<class PD,class Local>
HD bool integrate_bwd_into( const PD &pd, SI k0, auto &&grad_res, const Local &cell, Local &piece,
                         GradVp<typename PD::TF> &grad_vp, auto &&grad_positions, auto &&grad_weights, auto &&grad_groups,
                         auto &&grad_dist, const auto &dist ) {
    using TF = typename PD::TF;
    constexpr int D = PD::ct_dim;
    PieceWorkspace<Local> ws{ piece };
    const TF g = grad_res;

    // a density that is the affine image of a fixed one ( an `Image` ), asked for the derivative of its frame or origin
    using Dist = DECAYED_TYPE_OF( dist );
    AffineAcc<TF,D> acc_store, *acc = nullptr;
    if constexpr ( is_affine<Dist>() ) {
        if ( dist.affine_wanted( grad_dist ) ) {
            acc = &acc_store;
            dist.affine_origin( acc->o );
            for ( int a = 0; a < D; ++a ) {
                acc->y[ a ] = 0;
                for ( int b = 0; b < D; ++b )
                    acc->X[ a ][ b ] = 0;
            }
        }
    }

    dist.for_each_piece( cell, ws, [&]( const auto &pc, const auto &dens ) {
        if constexpr ( DECAYED_TYPE_OF( dens )::is_constant ) {
            // the DENSITY's share: mass is linear in it, so the derivative with respect to
            // the value carried by this piece IS its volume ( an infinite piece has none )
            if ( pc.bounded() ) {
                const TF vol = pc.template measure<TF>();
                dens.add_value_grad( grad_dist, g * vol );
                if ( acc )                               // ( `tr( A^-1 dA ) J`: the density's own jacobian )
                    for ( int a = 0; a < D; ++a )
                        acc->X[ a ][ a ] += g * dens.value * vol;
            }
            // ... and the GEOMETRY's share, through the usual chain
            pc.template measure_bwd<TF>( g * dens.value, grad_vp );
            scatter_cell_grad( pd, k0, pc, grad_vp, grad_positions, grad_weights, grad_groups, acc );
        } else {
            if ( ! pc.bounded() )
                return;
            const int nv = pc.nb_vertices();
            for ( int v = 0; v < nv; ++v )
                for ( int c = 0; c < D; ++c )
                    grad_vp( v, c ) = 0;
            pc.for_each_simplex( [&]( const auto &chain ) {
                auto grad_pts = Vector<Vector<TF,D>,D+1>( Function(), [&]( PI ) { return Vector<TF,D>::zeros(); } );
                const auto pts = simplex_points<TF,D>( pc, chain );
                dens.integrate_over_simplex_bwd( pts, g, grad_pts, grad_dist );
                for ( int k = 0; k <= D; ++k )
                    for ( int c = 0; c < D; ++c )
                        grad_vp( int( chain[ k ] ), c ) += grad_pts[ k ][ c ];
                if constexpr ( has_affine_terms<DECAYED_TYPE_OF( dens )>() )
                    if ( acc )                           // ( `J` and the moment's polynomial, which stays in `x` )
                        dens.affine_terms( pts, g, *acc );
            } );
            scatter_cell_grad( pd, k0, pc, grad_vp, grad_positions, grad_weights, grad_groups, acc );
            // an integrand that depends on the seed itself ( the ball of partial transport: its center and its radius )
            if constexpr ( depends_on_seed<DECAYED_TYPE_OF( dens )>() ) {
                TF gp[ D ], gw = 0;
                for ( int c = 0; c < D; ++c )
                    gp[ c ] = 0;
                dens.seed_bwd( pc, g, gp, gw );
                if constexpr ( ! DECAYED_TYPE_OF( grad_positions )::surely_null )
                    for ( int c = 0; c < D; ++c )
                        atomic_add( grad_positions( k0, c ).ref(), gp[ c ] );
                if constexpr ( ! DECAYED_TYPE_OF( grad_weights )::surely_null )
                    atomic_add( grad_weights( k0 ).ref(), gw );
            }
        }
    } );
    if constexpr ( is_affine<Dist>() )
        if ( acc )
            dist.add_affine_grad( grad_dist, *acc );
    return ! ws.overflow;
}

template<class PD>
HD void measures_bwd( const PD &pd, auto &&res, const auto &dom, auto &&grad_res, auto &&grad_positions, auto &&grad_weights,
                   auto &&grad_groups, auto &&scratch, const auto &dist, auto &&grad_dist, SI thread_index, SI nb_threads ) {
    using TF    = typename PD::TF;
    using TK    = KernelType<DECAYED_TYPE_OF( scratch )>;
    using Local = typename DECAYED_TYPE_OF( dom )::template Local<TK>;
    constexpr int nbc = nb_work_cells<DECAYED_TYPE_OF( dist )>();
    Carver cv = carver_of( scratch );
    const SI cap = cap_in<Local,TF>( cv.nb_words, nbc, true );
    Local c, piece;
    c.attach( cv, cap );
    if constexpr ( nbc > 1 ) piece.attach( cv, cap );
    else                     piece = c;
    GradVp<TF> grad_vp{ cv.take<TF>( PD::ct_dim * cap ), cap };

    for_each_seed<PD>( pd.nb_seeds(), thread_index, nb_threads, [&]( SI k ) {
        if ( ! make_cell( pd, c, k, dom )
          || ! integrate_bwd_into( pd, k, grad_res( pd.user_id( k ) ), c, piece, grad_vp, grad_positions, grad_weights, grad_groups, grad_dist, cell_dist( pd, k, dist ) ) ) {
            ask_more( scratch, cv );
            return false;
        }
        return true;
    } );
}

/// The cell of seed `k`, KEPT: built like any other, then put into `res` -- with its cut
/// identifiers translated for the user. The only query whose memory is a function
/// of the number of seeds: which is what a DISPLAY is.
/// The cell of seed `k`, put into `res` with the user's indices in its `cut_ids`. `keep_images`: with a periodicity, the image `t` of
/// the seed `j` a cut faces stays in its identifier, `j + n t` ( `j` the user's index ) -- what the cell complex glues modulo the group
template<class PD>
HD void build_cell( const PD &pd, SI k, const auto &dom, auto &&res, auto &&scratch, SI thread_index, bool keep_images = false ) {
    using TK    = KernelType<DECAYED_TYPE_OF( scratch )>;
    using Local = typename DECAYED_TYPE_OF( dom )::template Local<TK>;
    Carver cv = carver_of( scratch, thread_index );
    Local c = local_on<Local>( scratch, cv );
    if ( ! make_cell( pd, c, k, dom ) ) {
        ask_more( scratch, cv );
        return;
    }
    c.tidy();
    const SI n = pd.nb_seeds();
    for ( int q = 0; q < c.nb_cuts(); ++q )
        if ( c.cid[ q ] >= 0 )
            c.cid[ q ] = int( pd.user_id( c.cid[ q ] ) + ( keep_images ? n * ( SI( c.cid[ q ] ) / n ) : 0 ) );
    c.store( res );
}

/// The cell of seed `k` cut to the BLOCK `b` of the support of a density, put into `res` like `build_cell`
/// -- the cell itself when `b < 0`. What a DISPLAY of the cells restricted to where the density has
/// mass is made of: one call over the pieces ( `PowerDiagram.support_pieces` ).
///
/// A block is a convex polytope given by the distribution, host side ( `Distribution.display_blocks` ):
/// its walls `dirs( b, f ) . x <= offs( b, f )`, each with the identifier it leaves on the cell -- `SEAM`
/// or `SUPPORT` ( `cell/Ids.h` ). A wall with a zero direction pads the list. A wall that no vertex
/// of the cell crosses ( up to the rounding of the kernel float ) is not cut: a cell that lies in its
/// block is not touched, and a wall that coincides with a side of the domain does not leave a sliver.
template<class PD>
HD void build_support_piece( const PD &pd, SI k, SI b, const auto &dirs, const auto &offs, const auto &ids, const auto &dom,
                             auto &&res, auto &&scratch, SI thread_index ) {
    using TK    = KernelType<DECAYED_TYPE_OF( scratch )>;
    using TF    = typename PD::TF;
    using Local = typename DECAYED_TYPE_OF( dom )::template Local<TK>;
    constexpr int D = PD::ct_dim;
    Carver cv = carver_of( scratch, thread_index );
    Local c = local_on<Local>( scratch, cv );
    if ( ! make_cell( pd, c, k, dom ) ) {
        ask_more( scratch, cv );
        return;
    }
    if ( b >= 0 ) {
        const int nf = int( offs.shape( 1 ) );
        for ( int f = 0; f < nf && c.nb_vertices(); ++f ) {
            typename Local::PlaneT p;
            TF n2 = 0;
            for ( int d = 0; d < D; ++d ) {
                const TF x = TF( dirs( b, f, d ) );
                p.dir[ d ] = TK( x );
                n2 += x * x;
            }
            if ( n2 == 0 )
                continue;                                // padding
            p.off = TK( offs( b, f ) );
            p.id  = int( ids( b, f ) );

            // crossed by a vertex, beyond the rounding of the kernel float ( relative to the size of the coordinates )
            TF out = 0, amp = TF( p.off < 0 ? -p.off : p.off );
            for ( int v = 0; v < c.nb_vertices(); ++v ) {
                TF s = - TF( p.off ), a = 0;
                for ( int d = 0; d < D; ++d ) {
                    const TF t = TF( p.dir[ d ] ) * TF( c.coord( v, d ) );
                    s += t;
                    a += t < 0 ? -t : t;
                }
                out = s > out ? s : out;
                amp = a > amp ? a : amp;
            }
            if ( out <= TF( 64 ) * TF( std::numeric_limits<TK>::epsilon() ) * amp )
                continue;
            if ( c.cut( p ) == CutStatus::NO_ROOM ) {
                ask_more( scratch, cv );
                return;
            }
        }
    }
    c.tidy();
    for ( int q = 0; q < c.nb_cuts(); ++q )
        if ( c.cid[ q ] >= 0 )
            c.cid[ q ] = int( pd.user_id( c.cid[ q ] ) );
    c.store( res );
}

/// the squared distance that sets how fast the plane between `k` and `id` slides with the weight of `k`: between their points, or
/// for two members of a group ( `diagram/Common.h::group_distance2` ) in the frame of their lowest common aggregate
template<class PD>
HD typename PD::TF group_distance2( const PD &pd, SI k, SI id ) {
    using TF = typename PD::TF;
    constexpr int D = PD::ct_dim;
    if constexpr ( has_groups<PD>() )
        if ( pd.same_group( k, id ) )
            return pd.group_distance2( k, id );
    TF d2 = 0;
    const auto p0 = pd.point( k ), pj = pd.point( id );
    for ( int d = 0; d < D; ++d )
        d2 += ( pj[ d ] - p0[ d ] ) * ( pj[ d ] - p0[ d ] );
    return d2;
}

/// ONE ROW OF THE transport HESSIAN, `d m_k / d w_j` for the neighbors `j` of cell `k`:
/// the bisector `( k, j )` slides by `dw / ( 2 | p_k - p_j | )` when `w_j` goes up by `dw`, and what
/// it sweeps is the density integrated over the common FACET. So
/// `d m_k / d w_j = - int_{facet} rho / ( 2 | p_k - p_j | )`, and `d m_k / d w_k` is the summed
/// opposite ( a row sums to zero -- which the caller recomposes ). The domain ( negative ids ) does not
/// move. `res`: `ids( r )` ( user identifiers ) and `vals( r )` ( the values, POSITIVE ),
/// `nb_nbrs` the count -- a capacity that loom doubles if it is missing.
///
/// A piecewise-constant density ( `Image`, Lebesgue ): the facet of a piece is flat and the density is a number
/// there. Otherwise the density integrates itself on the facet ( `facet_mass`: gaussians in 2D, `Mesh` ).
///
/// With a SINK ( partial transport ) the row does not sum to zero: the sphere of the ball also moves with `w_k`. One more
/// slot, LAST, with the cell's own identifier, holds that part of `d m_k / d w_k` ( `weight_mass` ). The same with a RADIAL
/// integrand, which grows with `w_k` everywhere in the cell ( `m_k / ( 2 eps )` ).
///
/// With GROUPS ( `diagram/Common.h` ), each facet's value is the derivative with respect to ITS OWN weight gap: the outer weights
/// for a facet between groups, and for two members of a group their weights `V` in the frame of their lowest common aggregate
/// ( `group_distance2`: the distance then that of their virtual diracs in that frame ).
template<class PD>
HD void hessian_row( const PD &pd, SI k, const auto &dom, auto &&res, auto &&scratch, SI thread_index, const auto &dist ) {
    using TF    = typename PD::TF;
    using TK    = KernelType<DECAYED_TYPE_OF( scratch )>;
    using Local = typename DECAYED_TYPE_OF( dom )::template Local<TK>;
    constexpr int D = PD::ct_dim;
    constexpr int nbc = nb_work_cells<DECAYED_TYPE_OF( dist )>();
    Carver cv = carver_of( scratch, thread_index );
    const SI cap = cap_in<Local,TF>( cv.nb_words, nbc, false );
    Local c, piece;
    c.attach( cv, cap );
    if constexpr ( nbc > 1 ) piece.attach( cv, cap );
    else                     piece = c;
    if ( ! make_cell( pd, c, k, dom ) ) {
        ask_more( scratch, cv );
        return;
    }
    c.tidy();

    // the neighbors of the cell are its live cuts: one slot per cut, accumulated piece
    // by piece ( a piece carries the same identifiers, plus those of its tile )
    const int nc = c.nb_cuts();
    if ( ! res.nb_nbrs.set( nc + int( has_weight_slot<PD>() ) ) )
        return;                                          // too many neighbors: loom doubles and calls again
    for ( int q = 0; q < nc; ++q ) {
        res.ids( q ) = c.cid[ q ] >= 0 ? int( pd.user_id( c.cid[ q ] ) ) : int( c.cid[ q ] );
        res.vals( q ) = 0;
    }
    if constexpr ( has_weight_slot<PD>() ) {
        res.ids( nc ) = int( pd.user_id( k ) );
        res.vals( nc ) = 0;
    }
    const auto &cdist = cell_dist( pd, k, dist );

    PieceWorkspace<Local> ws{ piece };
    auto planes = [&]( int q, auto *dir, auto &off ) {
        TF d[ D ], o;
        plane_of( pd, c, k, q, d, o );
        for ( int i = 0; i < D; ++i )
            dir[ i ] = d[ i ];
        off = o;
    };
    if constexpr ( requires { DECAYED_TYPE_OF( cdist )::cuts_facets; } ) {
        // a density that cuts the FACETS of the cell itself ( `Mesh` ), see `Mesh::for_each_facet_mass` -- inside the ball with a sink
        cdist.for_each_facet_mass( c, planes, [&]( int cut, auto part ) { res.vals( cut ) += TF( part ) / ( 2 * sdot::sqrt( group_distance2( pd, k, c.cid[ cut ] ) ) ); } );
        if constexpr ( has_weight_slot<PD>() )
            cdist.for_each_piece( c, ws, [&]( const auto &pc, const auto &dens ) {
                if constexpr ( requires { dens.weight_mass( pc ); } )
                    res.vals( nc ) += dens.weight_mass( pc );
            } );
    } else {
        cdist.for_each_piece( c, ws, [&]( const auto &pc, const auto &dens ) {
            if constexpr ( requires { dens.weight_mass( pc ); } )
                res.vals( nc ) += dens.weight_mass( pc );
            pc.template for_each_facet<TF>( [&]( int cut, TF mes ) {
                const int id = pc.cid[ cut ];
                if ( id < 0 )
                    return;                                  // the domain, or a tile edge: immobile
                const TF d2 = group_distance2( pd, k, id );
                // the density on the facet: a number on a piece of constant density, otherwise what the density says of it
                TF on_facet;
                if constexpr ( DECAYED_TYPE_OF( dens )::is_constant )
                    on_facet = TF( dens.value ) * mes;
                else {
                    static_assert( requires { dens.facet_mass( pc, cut ); }, "hessian: this density does not know how to integrate itself on a facet ( `facet_mass` )" );
                    on_facet = TF( dens.facet_mass( pc, cut ) );
                }
                const TF val = on_facet / ( 2 * sdot::sqrt( d2 ) );
                for ( int q = 0; q < nc; ++q )
                    if ( c.cid[ q ] == id ) {
                        res.vals( q ) += val;
                        break;
                    }
            } );
        } );
    }
    if ( ws.overflow )
        ask_more( scratch, cv );

    if constexpr ( is_periodic<PD>() ) {
        // PERIODIC: a facet with an image of the cell itself does not move with the weights, and two images of the same seed
        // are one neighbour -- summed into the first; the others are left as a negative id ( "to be ignored" ) and a zero
        for ( int q = 0; q < nc; ++q ) {
            const int id = c.cid[ q ];
            if ( id < 0 )
                continue;
            if ( pd.seed_of( id ) == k ) {
                res.ids( q ) = cell_ids::BOUNDARY;
                res.vals( q ) = 0;
                continue;
            }
            for ( int r = 0; r < q; ++r )
                if ( c.cid[ r ] >= 0 && res.ids( r ) == res.ids( q ) ) {
                    res.vals( r ) += res.vals( q );
                    res.ids( q ) = cell_ids::BOUNDARY;
                    res.vals( q ) = 0;
                    break;
                }
        }
    }
}

} // namespace diagram
} // namespace sdot
