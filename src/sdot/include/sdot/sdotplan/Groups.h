#pragma once

// =====================================================================================
// NEWTON ON A DIAGRAM WITH GROUPS ( `PowerDiagram( groups = ... )`, `groups.py`, `diagram/Common.h::group_plane` ).
//
// The groups are a TREE OF AGGREGATES: seeds much closer to one another than to the rest, nested ( a pair `1e-12` apart inside a
// pair `1e-5` apart ). An aggregate `A` has a frame of scale `s_A`, in which each child ( a seed or a nested aggregate ) has a
// virtual weight `v`; its true weight is `W_A + o_A^2 v` ( `o_A` the OUTER scale: `s_A`, or 0 for exact duplicates, whose virtual
// weights only split their site ). The plane between two seeds lives in the frame of their lowest common aggregate, where it
// slides with their weights `V` there ( `diagram/Common.h::frame_weight` ). Built once, from the positions; fixed during Newton.
//
// THE UNKNOWNS, one per seed ( `x`, user order, the sink first when there is one ): the smallest seed `i` of a node is the slot of
// the TOPMOST node it is the smallest seed of -- a root's outer weight `W_R` ( a lone seed is its own root ), or the virtual weight
// `v` of a child that is not the first of its parent ( the first one is the GAUGE of its aggregate, `v = 0` ). As many slots as
// seeds: each aggregate trades the slot of its first child for its own.
//
// THE JACOBIAN is block triangular by LEVELS. Summed over the seeds of a node, the masses do not see the planes inside it: the
// ROOT LEVEL is the laplacian of the facets between roots, the rows and columns of each root summed ( `aggregated`: the members are
// linked to their root, so that the solvers keep the size, the order and the gauge of the seeds ), solved by the solver of the
// transport. Then, per aggregate, top-down, a small dense block on its children ( but the first ): the facets between two of its
// children ( in its frame, of order one -- well conditioned ), and the facets out of it, through `o_A^2`. The coupling between
// levels -- an inner weight moves the outer weights by `o^2 v`, of relative size `( o_child / s_parent )^2 ~ kappa^2` -- is handled
// by BLOCK GAUSS-SEIDEL on the residual: each block solves for a CORRECTION, from the residual of its sums at the current
// direction; a few sweeps, none more when the tree only holds exact duplicates ( whose virtual weights move no outer weight ).
// =====================================================================================

#include "Laplacian.h"
#include "Linear.h"
#include <algorithm>
#include <cmath>
#include <vector>

namespace sdot {
namespace sdotplan {

struct GroupStructure {
    bool                active = false;
    SI                  n = 0;               ///< the unknowns ( the sink included )
    SI                  m = 0;               ///< the aggregates
    std::vector<SI>     parent;              ///< per seed: its aggregate, or -1
    std::vector<SI>     agg_parent;          ///< per aggregate: its parent, or -1 ( a parent before its children )
    std::vector<double> agg_scale, agg_outer;
    std::vector<double> sv_init, av_init;    ///< the virtual weights the diagram came with ( per seed, per aggregate )
    int                 sweeps = 30;         ///< block Gauss-Seidel sweeps at most, when an aggregate holds distinct positions ...
    double              sweep_rtol = 1e-9;   ///< ... until the corrections are below this, relatively to the direction
    int                 nb_sweeps = 0;       ///< the sweeps done, in all

    // derived ( `build` )
    std::vector<SI>     agg_first;           ///< the smallest seed of each aggregate
    std::vector<SI>     root_of;             ///< per seed: the smallest seed of its root ( itself when alone )
    std::vector<SI>     kids, k_begin;       ///< per aggregate, CSR: its children as nodes ( `< n`: a seed, else `n + aggregate` )
    std::vector<SI>     below, b_begin;      ///< per aggregate, CSR: all the seeds below it
    std::vector<char>   root_slot;           ///< per unknown: the outer weight of a root ( the gauge moves it )
    bool                coupled = false;     ///< an aggregate of distinct positions: the levels are coupled ( sweeps )

    SI   node_parent( SI node ) const { return node < n ? parent[ node ] : agg_parent[ node - n ]; }
    SI   node_first( SI node ) const { return node < n ? node : agg_first[ node - n ]; }
    double scale_of( SI a ) const { return a < 0 ? 1.0 : agg_scale[ a ]; }

    bool level0( SI i ) const { return ! active || root_slot[ i ]; }
    SI   nb_groups() const { SI r = 0; for ( SI a = 0; a < m; ++a ) r += agg_parent[ a ] < 0; return r; }

    /// the counters of a solve's stats: the roots, the seeds in them, the exact duplicates ( but the first of each site )
    void counts( double &nb_roots, double &nb_members, double &nb_dups ) const {
        nb_roots = nb_members = nb_dups = 0;
        for ( SI a = 0; a < m; ++a ) nb_roots += agg_parent[ a ] < 0;
        for ( SI i = 0; i < n; ++i ) {
            if ( parent[ i ] < 0 ) continue;
            nb_members += 1;
            nb_dups += agg_outer[ parent[ i ] ] == 0 && agg_first[ parent[ i ] ] != i;
        }
    }

    /// the derived arrays, from `parent`, `agg_parent`, `agg_scale`, `agg_outer` ( set by the caller )
    void build() {
        active = m > 0;
        agg_first.assign( m, n );
        for ( SI i = 0; i < n; ++i ) if ( parent[ i ] >= 0 ) agg_first[ parent[ i ] ] = std::min( agg_first[ parent[ i ] ], i );
        for ( SI a = m - 1; a >= 0; --a ) if ( agg_parent[ a ] >= 0 ) agg_first[ agg_parent[ a ] ] = std::min( agg_first[ agg_parent[ a ] ], agg_first[ a ] );
        std::vector<SI> agg_root( m );
        for ( SI a = 0; a < m; ++a ) agg_root[ a ] = agg_parent[ a ] < 0 ? a : agg_root[ agg_parent[ a ] ];
        root_of.resize( n );
        for ( SI i = 0; i < n; ++i ) root_of[ i ] = parent[ i ] < 0 ? i : agg_first[ agg_root[ parent[ i ] ] ];
        // the children
        k_begin.assign( m + 1, 0 );
        for ( SI i = 0; i < n; ++i ) if ( parent[ i ] >= 0 ) ++k_begin[ parent[ i ] + 1 ];
        for ( SI a = 0; a < m; ++a ) if ( agg_parent[ a ] >= 0 ) ++k_begin[ agg_parent[ a ] + 1 ];
        for ( SI a = 0; a < m; ++a ) k_begin[ a + 1 ] += k_begin[ a ];
        kids.resize( k_begin[ m ] );
        std::vector<SI> at( k_begin.begin(), k_begin.end() - 1 );
        for ( SI i = 0; i < n; ++i ) if ( parent[ i ] >= 0 ) kids[ at[ parent[ i ] ]++ ] = i;
        for ( SI a = 0; a < m; ++a ) if ( agg_parent[ a ] >= 0 ) kids[ at[ agg_parent[ a ] ]++ ] = n + a;
        // the seeds below each aggregate
        b_begin.assign( m + 1, 0 );
        for ( SI i = 0; i < n; ++i ) for ( SI a = parent[ i ]; a >= 0; a = agg_parent[ a ] ) ++b_begin[ a + 1 ];
        for ( SI a = 0; a < m; ++a ) b_begin[ a + 1 ] += b_begin[ a ];
        below.resize( b_begin[ m ] );
        at.assign( b_begin.begin(), b_begin.end() - 1 );
        for ( SI i = 0; i < n; ++i ) for ( SI a = parent[ i ]; a >= 0; a = agg_parent[ a ] ) below[ at[ a ]++ ] = i;
        // the slots of the roots
        root_slot.assign( n, 1 );
        for ( SI a = 0; a < m; ++a )
            for ( SI q = k_begin[ a ]; q < k_begin[ a + 1 ]; ++q )
                if ( node_first( kids[ q ] ) != agg_first[ a ] ) root_slot[ node_first( kids[ q ] ) ] = 0;
        coupled = false;
        for ( SI a = 0; a < m; ++a ) coupled |= agg_outer[ a ] > 0;
    }

    /// the value of node `node` in `x` ( unknowns, or a direction ): a root's outer weight, a child's virtual weight ( 0 for the
    /// first child of its aggregate: the gauge )
    double value( const std::vector<double> &x, SI node ) const {
        const SI p = node_parent( node ), f = node_first( node );
        if ( p < 0 ) return x[ f ];
        return f == agg_first[ p ] ? 0.0 : x[ f ];
    }

    /// the weight of seed `i` in the frame of aggregate `L` ( `-1`: the outer weight ), for the unknowns ( or direction ) `x`
    double frame_weight( const std::vector<double> &x, SI i, SI L ) const {
        double V = value( x, i );
        for ( SI a = parent[ i ]; a != L; ) {
            const SI p = agg_parent[ a ];
            const double r = agg_outer[ a ] / scale_of( p );
            V = value( x, n + a ) + r * r * V;
            a = p;
        }
        return V;
    }
    double outer( const std::vector<double> &x, SI i ) const { return parent[ i ] < 0 ? x[ i ] : frame_weight( x, i, -1 ); }

    /// the lowest common aggregate of two seeds, `-1` when they are in two roots
    SI common( SI i, SI j ) const {
        if ( root_of[ i ] != root_of[ j ] ) return -1;
        for ( SI a = parent[ i ]; a >= 0; a = agg_parent[ a ] )
            for ( SI b = parent[ j ]; b >= 0; b = agg_parent[ b ] )
                if ( a == b ) return a;
        return -1;
    }

    /// the unknowns of weights `w` given seed by seed ( `groups.py`'s weights: the outer ones, and for duplicates their virtual
    /// ones ): the roots' outer weights from `w`, the virtual weights from the diagram's ( `precise` ) or read from `w`
    void unknowns_of( const std::vector<double> &w, std::vector<double> &x, bool precise ) const {
        x = w;
        if ( ! active ) return;
        for ( SI a = 0; a < m; ++a ) {
            for ( SI q = k_begin[ a ]; q < k_begin[ a + 1 ]; ++q ) {
                const SI node = kids[ q ], f = node_first( node );
                if ( f == agg_first[ a ] ) continue;
                if ( precise ) {                         // ( relative to the first child: the gauge of the aggregate )
                    SI first_kid = -1;
                    for ( SI r = k_begin[ a ]; r < k_begin[ a + 1 ]; ++r ) if ( node_first( kids[ r ] ) == agg_first[ a ] ) first_kid = kids[ r ];
                    auto v_in = [&]( SI nd ) { return nd < n ? sv_init[ nd ] : av_init[ nd - n ]; };
                    x[ f ] = v_in( node ) - v_in( first_kid );
                } else
                    x[ f ] = ( w[ f ] - w[ agg_first[ a ] ] ) / ( agg_scale[ a ] * agg_scale[ a ] );
            }
        }
    }
    /// ... and back: the weights seed by seed ( the outer weights, plus `s^2 v` for a duplicate: `groups.py`'s weights )
    void weights_of( const std::vector<double> &x, std::vector<double> &w ) const {
        w = x;
        if ( ! active ) return;
        for ( SI i = 0; i < n; ++i ) {
            if ( parent[ i ] < 0 ) continue;
            w[ i ] = outer( x, i );
            const SI p = parent[ i ];
            if ( agg_outer[ p ] == 0 ) w[ i ] += agg_scale[ p ] * agg_scale[ p ] * value( x, i );
        }
    }

    /// THE ROOT LEVEL: the facets between roots onto their first seeds, the members linked to theirs
    void aggregated( const std::vector<Facet> &fa, std::vector<Facet> &out ) const {
        out.clear();
        out.reserve( fa.size() + 2 * n );
        std::vector<double> sum( n, 0.0 );
        for ( const Facet &e : fa ) {
            const SI ri = root_of[ e.i ], rj = root_of[ e.j ];
            if ( ri == rj ) continue;                    // inside a root: its rows summed, it cancels
            out.push_back( Facet{ ri, rj, e.c } );
            sum[ ri ] += e.c;
        }
        for ( SI i = 0; i < n; ++i ) {
            if ( root_of[ i ] == i ) continue;
            const double c = sum[ root_of[ i ] ] > 0 ? sum[ root_of[ i ] ] : 1.0;
            out.push_back( Facet{ i, root_of[ i ], c } );
            out.push_back( Facet{ root_of[ i ], i, c } );
        }
    }

    /// `A x = b` for a small dense `k x k` ( row major ), partial pivoting; `false` if singular
    static bool dense_solve( std::vector<double> &A, std::vector<double> &b, SI k ) {
        for ( SI c = 0; c < k; ++c ) {
            SI p = c;
            for ( SI r = c + 1; r < k; ++r ) if ( std::fabs( A[ r * k + c ] ) > std::fabs( A[ p * k + c ] ) ) p = r;
            if ( ! ( std::fabs( A[ p * k + c ] ) > 0 ) ) return false;
            if ( p != c ) {
                for ( SI q = 0; q < k; ++q ) std::swap( A[ c * k + q ], A[ p * k + q ] );
                std::swap( b[ c ], b[ p ] );
            }
            for ( SI r = c + 1; r < k; ++r ) {
                const double f = A[ r * k + c ] / A[ c * k + c ];
                if ( f == 0 ) continue;
                for ( SI q = c; q < k; ++q ) A[ r * k + q ] -= f * A[ c * k + q ];
                b[ r ] -= f * b[ c ];
            }
        }
        for ( SI c = k - 1; c >= 0; --c ) {
            double s = b[ c ];
            for ( SI q = c + 1; q < k; ++q ) s -= A[ c * k + q ] * b[ q ];
            b[ c ] = s / A[ c * k + c ];
        }
        return true;
    }

    /// the facets of each seed in an aggregate ( `f_begin` / `f_idx`, CSR over the seeds ) and the frame of each facet ( `-1`:
    /// two roots, or a seed alone )
    struct FacetIndex { std::vector<SI> begin, idx, frame; };
    void index_facets( const std::vector<Facet> &fa, FacetIndex &fx ) const {
        fx.begin.assign( n + 1, 0 );
        for ( const Facet &e : fa ) if ( parent[ e.i ] >= 0 ) ++fx.begin[ e.i + 1 ];
        for ( SI i = 0; i < n; ++i ) fx.begin[ i + 1 ] += fx.begin[ i ];
        std::vector<SI> at( fx.begin.begin(), fx.begin.end() - 1 );
        fx.idx.resize( fx.begin[ n ] );
        fx.frame.assign( fa.size(), -1 );
        for ( SI q = 0; q < SI( fa.size() ); ++q ) {
            if ( parent[ fa[ q ].i ] < 0 ) continue;
            fx.idx[ at[ fa[ q ].i ]++ ] = q;
            fx.frame[ q ] = common( fa[ q ].i, fa[ q ].j );
        }
    }

    /// THE AGGREGATES' BLOCKS, top-down: for each aggregate, a correction of its children's virtual weights ( but the first )
    /// from the residual of their sums at the direction `d` -- the roots' slots of `d` set. `b`: the right-hand side ( read at
    /// the members ), `fa` every facet of the members ( both ways ). Returns the largest correction, in outer weights
    double blocks( const std::vector<Facet> &fa, const FacetIndex &fx, const std::vector<double> &b, std::vector<double> &d ) const {
        auto facet_flux = [&]( const Facet &e, SI frm ) { return e.c * ( frame_weight( d, e.i, frm ) - frame_weight( d, e.j, frm ) ); };
        double change = 0;
        std::vector<double> J, r;
        for ( SI a = 0; a < m; ++a ) {
            const SI kb = k_begin[ a ], ke = k_begin[ a + 1 ];
            std::vector<SI> unk;                         // the children but the first
            for ( SI q = kb; q < ke; ++q ) if ( node_first( kids[ q ] ) != agg_first[ a ] ) unk.push_back( kids[ q ] );
            const SI k = SI( unk.size() );
            if ( k == 0 ) continue;
            auto rank_of_child = [&]( SI node ) -> SI {
                for ( SI u = 0; u < k; ++u ) if ( unk[ u ] == node ) return u;
                return -1;
            };
            auto child_of = [&]( SI i ) -> SI {          // the child of `a` holding seed `i` ( `i` below `a` )
                SI node = i;
                while ( node_parent( node ) != a ) node = n + node_parent( node );
                return node;
            };
            J.assign( size_t( k * k ), 0.0 );
            r.assign( k, 0.0 );
            const double oa = agg_outer[ a ];
            for ( SI q = b_begin[ a ]; q < b_begin[ a + 1 ]; ++q ) {
                const SI i = below[ q ];
                const SI u = rank_of_child( child_of( i ) );
                if ( u < 0 ) continue;                   // ( the first child: the gauge, no equation )
                r[ u ] += b[ i ];
                for ( SI t = fx.begin[ i ]; t < fx.begin[ i + 1 ]; ++t ) {
                    const SI qf = fx.idx[ t ];
                    const Facet &e = fa[ qf ];
                    const SI frm = fx.frame[ qf ];
                    bool inside = false;                 // inside the same child: cancels in its sum
                    for ( SI c = frm; c >= 0; c = agg_parent[ c ] ) if ( c == a ) { inside = frm != a; break; }
                    if ( inside ) continue;
                    r[ u ] -= facet_flux( e, frm );
                    if ( frm == a ) {                    // between two children of `a`, in its frame
                        J[ u * k + u ] += e.c;
                        const SI v = rank_of_child( child_of( e.j ) );
                        if ( v >= 0 ) J[ u * k + v ] -= e.c;
                    } else if ( oa > 0 ) {               // out of `a`: through its outer weight, `( o_a / s_parent )^2` per level up to the frame
                        double f = 1;
                        for ( SI c = a; c != frm; c = agg_parent[ c ] ) f *= agg_outer[ c ] / scale_of( agg_parent[ c ] );
                        J[ u * k + u ] += e.c * f * f;
                    }
                }
            }
            if ( ! dense_solve( J, r, k ) )              // ( a child without facets: its weight is left as it is )
                continue;
            for ( SI u = 0; u < k; ++u ) {
                d[ node_first( unk[ u ] ) ] += r[ u ];
                change = std::max( change, std::fabs( r[ u ] ) * oa * oa );
            }
        }
        return change;
    }

    /// what the inner levels of `d` move of the roots' sums, through the facets between roots ( the members' outer weights
    /// differ from their root's by `o^2 v` ): `out[ root ] += c ( inner_i - inner_j )` for the facets of `fa` across two roots
    void inner_flux( const std::vector<Facet> &fa, const std::vector<double> &d, std::vector<double> &out ) const {
        out.assign( n, 0.0 );
        auto inner = [&]( SI i ) { return parent[ i ] < 0 ? 0.0 : outer( d, i ) - d[ root_of[ i ] ]; };
        for ( const Facet &e : fa )
            if ( root_of[ e.i ] != root_of[ e.j ] )
                out[ root_of[ e.i ] ] += e.c * ( inner( e.i ) - inner( e.j ) );
    }

    /// the outer weight of member `i` as slots: `W_i = sum coef x[ slot ]`
    void outer_expansion( SI i, std::vector<SI> &slots, std::vector<double> &coefs ) const {
        slots.clear(); coefs.clear();
        std::vector<SI> chain{ i };                      // the nodes from the seed up to its root
        for ( SI a = parent[ i ]; a >= 0; a = agg_parent[ a ] ) chain.push_back( n + a );
        double f = 1;
        for ( SI k = SI( chain.size() ) - 1; k >= 0; --k ) {
            const SI node = chain[ k ], p = node_parent( node ), fs = node_first( node );
            if ( ( p < 0 || fs != agg_first[ p ] ) && f != 0 ) { slots.push_back( fs ); coefs.push_back( f ); }
            if ( k > 0 ) {
                const SI a = node - n;
                const double r = agg_outer[ a ] / scale_of( agg_parent[ a ] );
                f *= r * r;
            }
        }
    }

    /// THE NEWTON DIRECTION `d` ( `J d = b` ) from the facets `fa` of the current diagram ( each coefficient with respect to the
    /// weights in the frame of the lowest common aggregate of its two seeds, `diagram::group_distance2` ) and the right-hand side
    /// `b` ( summing to zero ): the root level, then the blocks top-down, block Gauss-Seidel on the residual of their sums
    bool direction( const std::vector<Facet> &fa, const std::vector<double> &b, std::vector<double> &d, LinearSolver &lin,
                    double &t_asm, double &t_lin, double ( *clock )() ) {
        double t0 = clock();
        FacetIndex fx;
        index_facets( fa, fx );
        std::vector<Facet> fa_agg;
        aggregated( fa, fa_agg );
        Laplacian L;
        L.assemble( n, fa_agg );
        t_asm += clock() - t0;

        auto facet_flux = [&]( const Facet &e, SI frm ) { return e.c * ( frame_weight( d, e.i, frm ) - frame_weight( d, e.j, frm ) ); };
        std::vector<double> bR( n ), dX;
        d.assign( n, 0.0 );
        const int max_sweeps = coupled ? sweeps : 1;
        for ( int sweep = 0; sweep < max_sweeps; ++sweep ) {
            double change = 0, size = 0;                 // ( the corrections and the direction, in outer weights: the stopping test )
            // ---- the root level: a correction of the roots' outer weights
            std::fill( bR.begin(), bR.end(), 0.0 );
            for ( SI i = 0; i < n; ++i ) bR[ root_of[ i ] ] += b[ i ];
            if ( sweep > 0 )
                for ( const Facet &e : fa )
                    if ( root_of[ e.i ] != root_of[ e.j ] )
                        bR[ root_of[ e.i ] ] -= facet_flux( e, -1 );
            t0 = clock();
            if ( ! lin.solves( L, bR, dX ) ) return false;
            t_lin += clock() - t0;
            for ( SI i = 0; i < n; ++i ) if ( root_slot[ i ] ) {
                d[ i ] += dX[ i ];
                change = std::max( change, std::fabs( dX[ i ] ) );
                size = std::max( size, std::fabs( d[ i ] ) );
            }
            ++nb_sweeps;
            // ---- the aggregates, top-down: a correction of their children's virtual weights
            change = std::max( change, blocks( fa, fx, b, d ) );
            if ( sweep > 0 && ! ( change > sweep_rtol * size ) ) break;
        }
        return true;
    }
};

} // namespace sdotplan
} // namespace sdot
