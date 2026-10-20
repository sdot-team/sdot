#pragma once

#include <loom/support/common_macros.h> // HD

// WHAT THE TWO STORAGES HAVE IN COMMON, C++ side: the points of `positions`, the weight ( absent
// -> zero, and the term vanishes at compile time ), and the entry points that `PowerDiagram.py`
// calls -- which only hand over to `diagram/Ops.h`, adding themselves.
//
// A storage instantiates it with its own tensors: `DIAGRAM_COMMON( positions, weights )`.

#include <loom/support/containers/Vector.h>
#include "Ops.h"
#include "../Queue.h"

#define SDOT_DIAGRAM_COMMON( POSITIONS, WEIGHTS ) \
    static constexpr int  ct_dim = DECAYED_TYPE_OF( nb_dims )::value; \
    static constexpr bool on_cpu = std::is_same_v<Queue,CpuQueue>; \
    using TF = DECAYED_TYPE_OF( POSITIONS )::TF; \
    static constexpr bool has_weights = DECAYED_TYPE_OF( WEIGHTS )::is_valid; \
    \
    HD SI   nb_seeds() const { return SI( POSITIONS.shape( 0 ) ); } \
    \
    /* PERIODICITY ( `PowerDiagram( periodicity = ... )` ): the IMAGE `t` of seed `j` is the seed `j + n t`, at `positions( j ) +
       periodic_shifts( t )` with the weight of `j` -- an identifier like any other, so the bisectors, the facets and the
       gradients read it through `point` / `weight` / `seed_of` without knowing about images. `periodic_shifts( 0 )` is zero. */ \
    static constexpr bool periodic = DECAYED_TYPE_OF( periodic_shifts )::is_valid; \
    HD SI   image_count() const { if constexpr ( periodic ) return SI( periodic_shifts.shape( 0 ) ); else return 1; } \
    HD SI   seed_of( SI k ) const { if constexpr ( periodic ) return k % nb_seeds(); else return k; } \
    /* ... and with a LINEAR part ( rotations, symmetries ): image `t` of `x` is `periodic_linear( t ) x + periodic_shifts( t )` */ \
    static constexpr bool periodic_linear_part = DECAYED_TYPE_OF( periodic_linear )::is_valid; \
    HD TF   image_shift( SI t, int d ) const { if constexpr ( periodic ) return TF( periodic_shifts( t, d ) ); else return TF( 0 ); } \
    HD TF   image_linear( SI t, int r, int c ) const { if constexpr ( periodic_linear_part ) return TF( periodic_linear( t, r, c ) ); else return TF( r == c ); } \
    HD auto point( SI k ) const { \
        if constexpr ( periodic ) { \
            const SI n = nb_seeds(); \
            if ( k >= n ) { \
                const SI t = k / n, j = k - t * n; \
                return Vector<TF,ct_dim>::with_func( [&]( PI d ) { \
                    TF r = TF( periodic_shifts( t, d ) ); \
                    if constexpr ( periodic_linear_part ) { for ( int c = 0; c < ct_dim; ++c ) r += TF( periodic_linear( t, d, c ) ) * TF( POSITIONS( j, c ) ); } \
                    else r += TF( POSITIONS( j, d ) ); \
                    return r; \
                } ); \
            } \
        } \
        return Vector<TF,ct_dim>::with_func( [&]( PI d ) { return TF( POSITIONS( k, d ) ); } ); \
    } \
    HD TF   weight( SI k ) const { if constexpr ( has_weights ) return TF( WEIGHTS( seed_of( k ) ) ); else return TF( 0 ); } \
    \
    /* A SINK ( partial transport, `PowerDiagram( sink_weight = ... )` ): the cell of `k` is integrated inside the ball \
       `|x - p_k|^2 <= w_k - w_s` only ( `diagram::cell_dist` ) */ \
    static constexpr bool has_sink = DECAYED_TYPE_OF( sink_weight )::is_valid; \
    HD TF   sink() const { if constexpr ( has_sink ) return TF( sink_weight() ); else return TF( 0 ); } \
    \
    /* A RADIAL INTEGRAND ( `PowerDiagram( radial_func = RadialExp( eps ) )` ): the cell of `k` integrates \
       `rho( x ) exp( ( w_k - |x - p_k|^2 ) / ( 2 eps ) )` ( `diagram::cell_dist`, `integration/RadialExp.h` ) */ \
    static constexpr bool has_radial = DECAYED_TYPE_OF( radial_eps )::is_valid; \
    HD TF   radial() const { if constexpr ( has_radial ) return TF( radial_eps() ); else return TF( 0 ); } \
    \
    HD UnitDensity unit_density() const { return {}; } \
    \
    /* GROUPS ( `PowerDiagram( groups = ... )`, `groups.py` ): a TREE OF AGGREGATES -- seeds much closer to one another than to \
       the rest, which share the cell their root would have as one seed, seen from the others through their own positions and \
       `weight` ( the OUTER weight ), and split it among themselves. Each aggregate `a` has a frame, `y = ( x - center ) / scale`, \
       in which its children have virtual weights `v` ( a child's true weight is `W_a + outer_a^2 v` ): the plane between two \
       members is computed at their LOWEST COMMON AGGREGATE, in its frame, where everything is of order one. An aggregate of \
       exact DUPLICATES ( `outer == 0` ) splits its site by the unit directions `group_dirs` of its children, around an apex, at the \
       radius `scale` -- its `group_agg_center` is then the apex MINUS the site ( the fan follows the positions ). `group_rep` / `group_next` are storage ranks ( the smallest index of the root, the next \
       member: a cycle ); `group_parent` and the `group_agg_*` arrays are aggregate numbers, a parent before its children. Absent: \
       no group, and nothing of it at compile time. */ \
    static constexpr bool has_groups = DECAYED_TYPE_OF( group_rep )::is_valid; \
    HD bool same_group( SI k0, SI k1 ) const { \
        if constexpr ( has_groups ) return k0 != k1 && k0 < nb_seeds() && k1 < nb_seeds() && SI( group_rep( k0 ) ) == SI( group_rep( k1 ) ); \
        else return false; \
    } \
    HD SI   seed_parent( SI k ) const { if constexpr ( has_groups ) return SI( group_parent( k ) ); else return -1; } \
    HD SI   agg_parent( SI a ) const { if constexpr ( has_groups ) return SI( group_agg_parent( a ) ); else return -1; } \
    HD TF   agg_scale( SI a ) const { if constexpr ( has_groups ) return TF( group_agg_scale( a ) ); else return TF( 1 ); } \
    HD TF   agg_outer( SI a ) const { if constexpr ( has_groups ) return TF( group_agg_outer( a ) ); else return TF( 0 ); } \
    HD bool agg_of_duplicates( SI a ) const { return agg_outer( a ) == TF( 0 ); } \
    HD bool same_site( SI k0, SI k1 ) const { \
        const SI a = seed_parent( k0 ); \
        return a >= 0 && a == seed_parent( k1 ) && agg_of_duplicates( a ); \
    } \
    HD SI   next_member( SI k ) const { if constexpr ( has_groups ) return SI( group_next( k ) ); else return k; } \
    /* `f( j )` for each member `j` of the site of `k` ( `k` itself included ), and their number */ \
    HD SI   for_each_in_site( SI k, auto &&f ) const { \
        if constexpr ( has_groups ) { \
            if ( k < nb_seeds() && seed_parent( k ) >= 0 && agg_of_duplicates( seed_parent( k ) ) ) { \
                SI nb = 0, j = k; \
                do { if ( j == k || same_site( k, j ) ) { f( j ); ++nb; } j = next_member( j ); } while ( j != k ); \
                return nb; \
            } \
        } \
        f( k ); \
        return 1; \
    } \
    /* THE LOWEST COMMON AGGREGATE of two members of a group, and their weights in its frame ( `V`: the child's `v`, plus the \
       deeper levels' scaled by `( outer_child / scale )^2` ) */ \
    HD SI   common_aggregate( SI k0, SI k1, TF &V0, TF &V1 ) const { \
        if constexpr ( has_groups ) { \
            SI chain[ 64 ]; \
            int nc = 0; \
            for ( SI a = seed_parent( k0 ); a >= 0 && nc < 64; a = agg_parent( a ) ) chain[ nc++ ] = a; \
            SI L = -1; \
            for ( SI a = seed_parent( k1 ); a >= 0 && L < 0; a = agg_parent( a ) ) \
                for ( int c = 0; c < nc; ++c ) if ( chain[ c ] == a ) { L = a; break; } \
            V0 = frame_weight( k0, L ); \
            V1 = frame_weight( k1, L ); \
            return L; \
        } else { \
            V0 = V1 = 0; \
            return -1; \
        } \
    } \
    HD TF   frame_weight( SI k, SI L ) const { \
        if constexpr ( has_groups ) { \
            TF V = TF( group_vweights( k ) ); \
            for ( SI a = seed_parent( k ); a >= 0 && a != L; ) { \
                const SI p = agg_parent( a ); \
                const TF r = agg_outer( a ) / agg_scale( p ); \
                V = TF( group_agg_vweights( a ) ) + r * r * V; \
                a = p; \
            } \
            return V; \
        } else return TF( 0 ); \
    } \
    /* the plane `dir . x <= off` between two members of a group, on the `k0` side: the bisector of their virtual diracs in the \
       frame of their lowest common aggregate -- divided by its scale ( `dir` of order one ), the TRUE bisector for distinct \
       positions ( `diagram::scatter_cell_grad` reads it so ) */ \
    HD void group_plane( SI k0, SI k1, TF *dir, TF &off ) const { \
        if constexpr ( has_groups ) { \
            TF V0, V1; \
            const SI L = common_aggregate( k0, k1, V0, V1 ); \
            const TF s = agg_scale( L ); \
            TF q2 = 0, c = 0; \
            if ( agg_of_duplicates( L ) ) { /* the apex follows the site: `group_agg_center` is its shift, from each one's position */ \
                const auto p0 = point( k0 ), p1 = point( k1 ); \
                for ( int d = 0; d < ct_dim; ++d ) { \
                    const TF b = p0[ d ] + TF( group_agg_center( L, d ) ); \
                    const TF u0 = TF( group_dirs( k0, d ) ), u1 = TF( group_dirs( k1, d ) ) + ( p1[ d ] - p0[ d ] ) / s; \
                    dir[ d ] = u1 - u0; \
                    q2 += u1 * u1 - u0 * u0; \
                    c += dir[ d ] * b; \
                } \
            } else { \
                const auto p0 = point( k0 ), p1 = point( k1 ); \
                for ( int d = 0; d < ct_dim; ++d ) { \
                    const TF b = TF( group_agg_center( L, d ) ); \
                    const TF q0 = ( p0[ d ] - b ) / s, q1 = ( p1[ d ] - b ) / s; \
                    dir[ d ] = q1 - q0; \
                    q2 += q1 * q1 - q0 * q0; \
                    c += dir[ d ] * b; \
                } \
            } \
            off = c + s * ( q2 / 2 + ( V0 - V1 ) / 2 ); \
        } \
    } \
    /* the squared distance that sets how fast that plane slides with the weight of `k0` IN THE FRAME ( `V0` ): `|Delta p|^2 / \
       scale^4` ( true positions ), `|Delta u|^2 / scale^2` ( duplicates ) -- the hessian's coefficients are with respect to it */ \
    HD TF   group_distance2( SI k0, SI k1 ) const { \
        TF V0, V1; \
        const SI L = common_aggregate( k0, k1, V0, V1 ); \
        const TF s = agg_scale( L ); \
        TF d2 = 0; \
        if ( agg_of_duplicates( L ) ) { \
            for ( int d = 0; d < ct_dim; ++d ) { const TF e = TF( group_dirs( k1, d ) ) - TF( group_dirs( k0, d ) ); d2 += e * e; } \
            return d2 / ( s * s ); \
        } \
        const auto p0 = point( k0 ), p1 = point( k1 ); \
        for ( int d = 0; d < ct_dim; ++d ) { const TF e = ( p1[ d ] - p0[ d ] ) / s; d2 += e * e; } \
        return d2 / ( s * s ); \
    } \
    \
    HD void measures( auto &&res, const auto &dom, auto &&scratch, const auto &dist, auto &&memo_nbrs, auto &&memo_counts, SI thread_index, SI nb_threads ) const { \
        diagram::measures( *this, res, dom, scratch, dist, memo_nbrs, memo_counts, thread_index, nb_threads ); \
    } \
    /* what `measures_bwd` derives, for loom's `computed_grads` ( `flags`: this diagram's, under one output ): the seeds, \
       under the names of this storage. Not the box, the boundaries nor the sink weight: their derivatives are not \
       written, so a differentiation that needs them is refused instead of returning a gradient without them */ \
    static void measures_computed_grads( auto &&flags ) { /* a member absent from `flags` has nothing to derive ( no weights ) */ \
        if constexpr ( requires { flags.POSITIONS; } ) flags.POSITIONS = true; \
        if constexpr ( requires { flags.WEIGHTS; } )   flags.WEIGHTS   = true; \
        if constexpr ( has_groups ) { /* the planes inside a group: the virtual weights of the tree ( `diagram::scatter_frame_weight` ) */ \
            if constexpr ( requires { flags.group_vweights; } )     flags.group_vweights     = true; \
            if constexpr ( requires { flags.group_agg_vweights; } ) flags.group_agg_vweights = true; \
        } \
    } \
    /* `grad_groups`: the cotangents of the group members ( `diagram::group_grads` ), `diagram::NoGroupGrads` without groups */ \
    HD void measures_bwd( auto &&res, const auto &dom, auto &&grad_res, auto &&grad_positions, auto &&grad_weights, auto &&grad_groups, \
                       auto &&scratch, const auto &dist, auto &&grad_dist, SI thread_index, SI nb_threads ) const { \
        diagram::measures_bwd( *this, res, dom, grad_res, grad_positions, grad_weights, grad_groups, scratch, dist, grad_dist, thread_index, nb_threads ); \
    } \
    HD void moments( auto &&mass, auto &&first, auto &&second, const auto &dom, auto &&scratch, const auto &dist, SI thread_index, SI nb_threads ) const { \
        diagram::moments( *this, mass, first, second, dom, scratch, dist, thread_index, nb_threads ); \
    } \
    HD void hessian_row( SI k, const auto &dom, auto &&res, auto &&scratch, SI thread_index, const auto &dist ) const { \
        diagram::hessian_row( *this, k, dom, res, scratch, thread_index, dist ); \
    } \
    HD void build_cell( SI k, const auto &dom, auto &&res, auto &&scratch, SI thread_index ) const { \
        diagram::build_cell( *this, k, dom, res, scratch, thread_index ); \
    } \
    HD void build_support_piece( SI k, SI b, const auto &dirs, const auto &offs, const auto &ids, const auto &dom, auto &&res, auto &&scratch, SI thread_index ) const { \
        diagram::build_support_piece( *this, k, b, dirs, offs, ids, dom, res, scratch, thread_index ); \
    }
