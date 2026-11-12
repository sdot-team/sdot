#pragma once

// THE DISTRIBUTED SWEEP, one round on one BOX ( `sdot/distributed.py` ). Each box holds an ordinary diagram of ITS seeds, whose
// cells start from the WHOLE domain. A round takes cells -- the box's own seeds on the first round, cells sent by other boxes
// afterwards --, cuts each one with the seeds of the box, and then either FINISHES it ( no box it has not visited can still cut it:
// its measure goes out with the global id of its seed ) or SENDS it to the nearest box that can ( `can_cut_box`, exact for the
// box's majorant of the weights ). A cell only shrinks: a box that could not cut it never can, hence the `visited` set it carries.
//
// A cell travels as a row of `TF` ( `msg_width` words ):
//
//   0          the global id of its seed ( the owner )
//   1          the set of the visited boxes, as the bits of an integer ( K <= 52, exact in a double )
//   2          the weight of the seed
//   3 .. 3+D   its position
//   3+D        the number of vertices `nb`
//   then       vx[ mcap ], vy[ mcap ], cid[ mcap ]           ( 2D only for now )
//
// The ids of the cuts are GLOBAL: `id_offset` + the rank of the seed in its box ( negative ones are the domain's, as everywhere ).

#include "Ops.h"

namespace sdot {
namespace distributed {

constexpr int header_words( int D ) { return 4 + D; }

/// a provider whose seed cuts carry global ids ( `off` + the rank in the box )
template<class P>
struct OffsetIds {
    using Local = LocalOf<P>;
    P  f;
    SI off;

    template<class State,class Plane>
    HD bool next( const State &e, Local &l, Plane &p ) {
        if ( ! f.next( e, l, p ) )
            return false;
        if ( p.id >= 0 )
            p.id = int( p.id + off );
        return true;
    }
};

/// what a round knows of an item: its cell ( in `c` ), its seed, where it comes from
template<class TF,int D>
struct Item {
    TF            p0[ D ], w0, owner;
    std::uint64_t visited;
};

enum class ItemStatus { OK, EMPTY, NO_ROOM };

/// the cell of item `i`, cut by the seeds of this box: the box's own seed `i` ( `is_first` ), else the row `i` of `inbox`
template<class PD,class Local>
HD ItemStatus cut_item( const PD &pd, const auto &dom, Local &c, SI cap, bool is_first, SI i, const auto &inbox, const auto &gids,
                        int mcap, SI off, Item<typename PD::TF,PD::ct_dim> &it ) {
    using TF = typename PD::TF;
    using TK = typename Local::TKernel;
    constexpr int D = PD::ct_dim;
    if ( is_first ) {
        if ( ! c.load( dom ) )
            return ItemStatus::NO_ROOM;
        const auto p = pd.point( i );
        for ( int d = 0; d < D; ++d ) it.p0[ d ] = p[ d ];
        it.w0      = pd.weight( i );
        it.owner   = TF( gids( pd.user_id( i ) ) );
        it.visited = 0;
    } else {
        it.owner = inbox( i, 0 );
        if ( it.owner < 0 )
            return ItemStatus::EMPTY;                       // an empty row of a fixed-capacity inbox
        it.visited = std::uint64_t( inbox( i, 1 ) );
        it.w0      = inbox( i, 2 );
        for ( int d = 0; d < D; ++d ) it.p0[ d ] = inbox( i, 3 + d );
        const int nb = int( inbox( i, 3 + D ) );
        if ( nb > cap )
            return ItemStatus::NO_ROOM;
        c.nb = nb;
        c.unbounded = false;
        for ( int v = 0; v < nb; ++v ) {
            c.vx [ v ] = TK( inbox( i, header_words( D ) + v ) );
            c.vy [ v ] = TK( inbox( i, header_words( D ) + mcap + v ) );
            c.cid[ v ] = int( inbox( i, header_words( D ) + 2 * mcap + v ) );
            c.unbounded |= c.cid[ v ] == cell_ids::INFINITE;
        }
        c.has_planes = false;
    }

    // the cuts of this box's seeds ( a foreign cell: no rank of the box is its own seed )
    auto inner = pd.template provider<TK>( is_first ? i : 0 );
    if ( ! is_first ) {
        inner.k0 = -1;
        for ( int d = 0; d < D; ++d ) { inner.p0[ d ] = it.p0[ d ]; inner.q0[ d ] = TK( it.p0[ d ] ); }
        inner.w0 = it.w0;
        inner.v0 = TK( it.w0 );
    }
    OffsetIds<decltype( inner )> f{ inner, off };
    if ( run<PD::on_cpu>( c, f ) == CutStatus::NO_ROOM )
        return ItemStatus::NO_ROOM;
    return ItemStatus::OK;
}

/// the nearest box not visited yet ( `visited`, this one included ) that can still cut `c`, or -1
template<class TK,int D>
HD int nearest_box( const auto &c, const TK *q0, TK w0, std::uint64_t visited, const auto &boxes ) {
    const int K = int( boxes.shape( 0 ) );
    const TK *v[ D ] = { c.vx, c.vy };
    int best = -1;
    TK  best_dist = 0;
    for ( int j = 0; j < K; ++j ) {
        if ( visited & ( std::uint64_t( 1 ) << j ) )
            continue;
        Box<TK,D> B;
        TK dist = 0;
        for ( int d = 0; d < D; ++d ) {
            B.lo[ d ] = TK( boxes( j, d ) );
            B.hi[ d ] = TK( boxes( j, D + d ) );
            B.a [ d ] = TK( boxes( j, 2 * D + d ) );
            const TK e = q0[ d ] < B.lo[ d ] ? B.lo[ d ] - q0[ d ] : q0[ d ] > B.hi[ d ] ? q0[ d ] - B.hi[ d ] : TK( 0 );
            dist += e * e;
        }
        B.b = TK( boxes( j, 3 * D ) );
        if ( c.nb && c.unbounded ) { /* an unbounded cell can be cut by anyone */ }
        else if ( ! c.nb || ! can_cut_box<true,TK,D>( c.nb, v, q0, w0, B ) )
            continue;
        if ( best < 0 || dist < best_dist ) { best = j; best_dist = dist; }
    }
    return best;
}

/// one round on box `my_box`. `first`: the items are the box's own seeds, otherwise the rows of `inbox` ( a row whose owner is negative
/// is empty: `out_dest` = -2 ). Out, per item: `out_dest` is `-1` for a finished cell ( `out_val` its measure, `out_gid` the global id of its seed ) or the box it goes to ( `out_msgs` its
/// row ). `status( 0 )` = 1 when a cell has more vertices than a row holds ( the host widens the rows and starts again ).
template<class PD>
HD void dist_round( const PD &pd, const auto &dom, auto &&scratch, SI thread_index, SI nb_threads,
                    const auto &first, const auto &inbox, const auto &boxes, const auto &my_box, const auto &id_offset,
                    const auto &gids, auto &&out_msgs, auto &&out_dest, auto &&out_val, auto &&out_gid, auto &&status ) {
    using TF    = typename PD::TF;
    using TK    = KernelType<DECAYED_TYPE_OF( scratch )>;
    using Local = typename DECAYED_TYPE_OF( dom )::template Local<TK>;
    constexpr int D = PD::ct_dim;
    static_assert( D == 2, "the distributed sweep is 2D only for now" );

    Carver cv = carver_of( scratch );
    const SI cap = diagram::cap_in<Local,TF>( cv.nb_words, 1, true );       // ( as the adjoint carves it: the same cells )
    Local c;
    c.attach( cv, cap );

    const bool is_first = bool( SI( first( 0 ) ) );
    const SI   nb_items = is_first ? SI( pd.nb_seeds() ) : SI( inbox.shape( 0 ) );
    const SI   width    = SI( out_msgs.shape( 1 ) );
    const int  mcap     = int( ( width - header_words( D ) ) / 3 );
    const SI   me       = SI( my_box( 0 ) );
    const SI   off      = SI( id_offset( 0 ) );

    for ( SI i = thread_index; i < nb_items; i += nb_threads ) {
        Item<TF,D> it;
        const ItemStatus st = cut_item( pd, dom, c, cap, is_first, i, inbox, gids, mcap, off, it );
        if ( st == ItemStatus::EMPTY ) { out_dest( i ) = -2; continue; }
        if ( st == ItemStatus::NO_ROOM ) { ask_more( scratch, cv ); return; }
        it.visited |= std::uint64_t( 1 ) << me;

        TK q0[ D ];
        for ( int d = 0; d < D; ++d ) q0[ d ] = TK( it.p0[ d ] );
        const int best = nearest_box<TK,D>( c, q0, TK( it.w0 ), it.visited, boxes );

        if ( best < 0 ) {
            out_dest( i ) = -1;
            out_gid ( i ) = SI( it.owner );
            out_val ( i ) = c.nb ? c.template measure<TF>() : TF( 0 );
            continue;
        }

        const int nb = c.nb;
        if ( nb > mcap ) { status( 0 ) = 1; out_dest( i ) = -1; out_gid( i ) = SI( it.owner ); out_val( i ) = 0; continue; }
        out_dest( i ) = best;
        out_msgs( i, 0 ) = it.owner;
        out_msgs( i, 1 ) = TF( it.visited );
        out_msgs( i, 2 ) = it.w0;
        for ( int d = 0; d < D; ++d ) out_msgs( i, 3 + d ) = it.p0[ d ];
        out_msgs( i, 3 + D ) = TF( nb );
        for ( int q = 0; q < nb; ++q ) {
            out_msgs( i, header_words( D ) + q )            = TF( c.vx[ q ] );
            out_msgs( i, header_words( D ) + mcap + q )     = TF( c.vy[ q ] );
            out_msgs( i, header_words( D ) + 2 * mcap + q ) = TF( c.cid[ q ] );
        }
    }
}

/// THE ADJOINT OF A ROUND. The cell of each item is cut again, and each of its vertices sends its cotangent back onto its two
/// planes ( `A^T u = q`, as `diagram::scatter_cell_grad` ). A plane of a seed of THIS box is the bisector of that seed and of the
/// cell's own ( `( p0, w0 )`, from the row or the box's own seed ): both get their share. A plane the cell came with is known only
/// through the incoming polygon: it is the line of the incoming edge of the same id, so its share goes to the two incoming vertices
/// of that edge -- and through them, to the rounds before. A vertex on two incoming planes is an incoming vertex, unchanged. The
/// domain does not move. `q`: the cotangent of the vertices, that of the row sent or that of the measure of a finished cell.
template<class PD>
HD void dist_round_bwd( const PD &pd, const auto &dom, auto &&scratch, SI thread_index, SI nb_threads,
                        const auto &first, const auto &inbox, const auto &boxes, const auto &my_box, const auto &id_offset,
                        const auto &gids, const auto &out_msgs, const auto &out_dest, const auto &grad_msgs, const auto &grad_val,
                        auto &&grad_positions, auto &&grad_weights, auto &&grad_inbox ) {
    using TF    = typename PD::TF;
    using TK    = KernelType<DECAYED_TYPE_OF( scratch )>;
    using Local = typename DECAYED_TYPE_OF( dom )::template Local<TK>;
    constexpr int D = PD::ct_dim;
    static_assert( D == 2, "the distributed sweep is 2D only for now" );
    constexpr bool g_pos = ! DECAYED_TYPE_OF( grad_positions )::surely_null;
    constexpr bool g_w   = ! DECAYED_TYPE_OF( grad_weights )::surely_null;
    constexpr bool g_in  = DECAYED_TYPE_OF( grad_inbox )::is_valid;
    constexpr bool from_msgs = ! DECAYED_TYPE_OF( grad_msgs )::surely_null;
    constexpr bool from_val  = ! DECAYED_TYPE_OF( grad_val )::surely_null;

    Carver cv = carver_of( scratch );
    const SI cap = diagram::cap_in<Local,TF>( cv.nb_words, 1, true );
    Local c;
    c.attach( cv, cap );
    TF *q = cv.take<TF>( D * cap );                   // the cotangents of the vertices

    const bool is_first = bool( SI( first( 0 ) ) );
    const SI   nb_items = is_first ? SI( pd.nb_seeds() ) : SI( inbox.shape( 0 ) );
    const SI   width    = SI( out_msgs.shape( 1 ) );
    const int  mcap     = int( ( width - header_words( D ) ) / 3 );
    const SI   me       = SI( my_box( 0 ) );
    const SI   off      = SI( id_offset( 0 ) );
    const SI   n        = SI( pd.nb_seeds() );
    const int  H        = header_words( D );

    auto add_in = [&]( SI i, int col, TF v ) {
        if constexpr ( g_in )
            grad_inbox( i, col ).ref() += v;
    };

    for ( SI i = thread_index; i < nb_items; i += nb_threads ) {
        const SI dest = SI( out_dest( i ) );
        if ( dest == -2 )
            continue;
        Item<TF,D> it;
        const ItemStatus st = cut_item( pd, dom, c, cap, is_first, i, inbox, gids, mcap, off, it );
        if ( st == ItemStatus::EMPTY ) continue;
        if ( st == ItemStatus::NO_ROOM ) { ask_more( scratch, cv ); return; }
        const int nb = c.nb;
        if ( ! nb || c.unbounded )
            continue;

        // the cotangents of the vertices, and of the seed the row carries
        TF gp0[ D ] = {}, gw0 = 0;
        bool any = false;
        if ( dest >= 0 ) {
            if constexpr ( ! from_msgs )
                continue;
            else {
                for ( int v = 0; v < nb; ++v ) {
                    q[ D * v + 0 ] = grad_msgs( i, H + v );
                    q[ D * v + 1 ] = grad_msgs( i, H + mcap + v );
                }
                gw0 += grad_msgs( i, 2 );
                for ( int d = 0; d < D; ++d ) gp0[ d ] += grad_msgs( i, 3 + d );
                any = true;
            }
        } else {
            if constexpr ( ! from_val )
                continue;
            else {
                // a cell sent with too many vertices for a row was given 0 ( and the run started again ): not a measure
                TK q0[ D ];
                for ( int d = 0; d < D; ++d ) q0[ d ] = TK( it.p0[ d ] );
                if ( nearest_box<TK,D>( c, q0, TK( it.w0 ), it.visited | ( std::uint64_t( 1 ) << me ), boxes ) >= 0 )
                    continue;
                c.template measure_bwd<TF>( TF( grad_val( i ) ), [&]( int v, int d ) -> TF & { return q[ D * v + d ]; } );
                any = true;
            }
        }

        // the incoming polygon ( a row ): its vertices and the ids of its edges
        const int nin = is_first ? 0 : int( inbox( i, 3 + D ) );
        auto in_x   = [&]( int u, int d ) { return TF( inbox( i, H + d * mcap + u ) ); };
        auto in_cid = [&]( int u ) { return int( inbox( i, H + 2 * mcap + u ) ); };
        auto in_edge = [&]( int id ) {                  // the incoming edge with this id, -1 if none
            for ( int u = 0; u < nin; ++u )
                if ( in_cid( u ) == id )
                    return u;
            return -1;
        };
        auto is_new = [&]( int id ) { return id >= 0 && SI( id ) >= off && SI( id ) < off + n; };

        for ( int v = 0; any && v < nb; ++v ) {
            const TF qx = q[ D * v + 0 ], qy = q[ D * v + 1 ];
            if ( qx == 0 && qy == 0 )
                continue;
            const TF x[ D ] = { TF( c.vx[ v ] ), TF( c.vy[ v ] ) };
            const int e[ 2 ] = { c.vertex_cut( v, 0 ), c.vertex_cut( v, 1 ) };
            const int f[ 2 ] = { c.cid[ e[ 0 ] ], c.cid[ e[ 1 ] ] };

            // two planes it came with: an incoming vertex, unchanged
            if ( ! is_new( f[ 0 ] ) && ! is_new( f[ 1 ] ) ) {
                for ( int u = 0; u < nin; ++u )
                    if ( in_cid( u ? u - 1 : nin - 1 ) == f[ 0 ] && in_cid( u ) == f[ 1 ] ) {
                        add_in( i, H + u, qx );
                        add_in( i, H + mcap + u, qy );
                        break;
                    }
                continue;
            }

            // the directions of its planes, as they are differentiated: `p1 - p0` for a seed of the box, the normal of the incoming
            // edge for one the cell came with, the final edge's for the domain ( which gets nothing: its scale does not matter )
            TF dir[ 2 ][ D ];
            int ea[ 2 ] = { -1, -1 };
            for ( int r = 0; r < 2; ++r ) {
                if ( is_new( f[ r ] ) ) {
                    const auto p1 = pd.point( SI( f[ r ] ) - off );
                    for ( int d = 0; d < D; ++d ) dir[ r ][ d ] = TF( p1[ d ] ) - it.p0[ d ];
                    continue;
                }
                TF a[ D ], b[ D ];
                const int u = f[ r ] >= 0 ? in_edge( f[ r ] ) : -1;
                if ( u >= 0 ) {
                    ea[ r ] = u;
                    const int w = u + 1 < nin ? u + 1 : 0;
                    for ( int d = 0; d < D; ++d ) { a[ d ] = in_x( u, d ); b[ d ] = in_x( w, d ); }
                } else {
                    const int k = e[ r ], w = k + 1 < nb ? k + 1 : 0;
                    a[ 0 ] = TF( c.vx[ k ] ); a[ 1 ] = TF( c.vy[ k ] );
                    b[ 0 ] = TF( c.vx[ w ] ); b[ 1 ] = TF( c.vy[ w ] );
                }
                dir[ r ][ 0 ] = b[ 1 ] - a[ 1 ];
                dir[ r ][ 1 ] = a[ 0 ] - b[ 0 ];
            }

            // `A^T u = q`, the rows of `A` the directions
            const TF det = dir[ 0 ][ 0 ] * dir[ 1 ][ 1 ] - dir[ 1 ][ 0 ] * dir[ 0 ][ 1 ];
            if ( det == 0 )
                continue;                                // ( two parallel planes: no vertex to move )
            const TF u[ 2 ] = { ( qx * dir[ 1 ][ 1 ] - qy * dir[ 1 ][ 0 ] ) / det, ( dir[ 0 ][ 0 ] * qy - dir[ 0 ][ 1 ] * qx ) / det };

            for ( int r = 0; r < 2; ++r ) {
                if ( is_new( f[ r ] ) ) {
                    // the bisector of `( p0, w0 )` and `( p1, w1 )`: `dir = p1 - p0`, `off = ( |p1|^2 - |p0|^2 ) / 2 + ( w0 - w1 ) / 2`
                    const SI k1 = SI( f[ r ] ) - off;
                    const auto p1 = pd.point( k1 );
                    const SI s1 = diagram::seed_rank( pd, k1 );
                    for ( int d = 0; d < D; ++d ) {
                        if constexpr ( g_pos )
                            atomic_add( grad_positions( s1, d ).ref(), u[ r ] * ( TF( p1[ d ] ) - x[ d ] ) );
                        gp0[ d ] += u[ r ] * ( x[ d ] - it.p0[ d ] );
                    }
                    if constexpr ( g_w )
                        atomic_add( grad_weights( s1 ).ref(), - u[ r ] / 2 );
                    gw0 += u[ r ] / 2;
                } else if ( ea[ r ] >= 0 ) {
                    // the line through the incoming `a`, `b`: `n = ( b_y - a_y, a_x - b_x )`, `off = n . a`
                    const int ua = ea[ r ], ub = ua + 1 < nin ? ua + 1 : 0;
                    const TF a[ D ] = { in_x( ua, 0 ), in_x( ua, 1 ) };
                    const TF gn[ D ] = { u[ r ] * ( a[ 0 ] - x[ 0 ] ), u[ r ] * ( a[ 1 ] - x[ 1 ] ) };
                    // n_x = b_y - a_y, n_y = a_x - b_x
                    add_in( i, H + ua,        u[ r ] * dir[ r ][ 0 ] + gn[ 1 ] );
                    add_in( i, H + mcap + ua, u[ r ] * dir[ r ][ 1 ] - gn[ 0 ] );
                    add_in( i, H + ub,        - gn[ 1 ] );
                    add_in( i, H + mcap + ub, gn[ 0 ] );
                }
            }
        }

        // the seed of the cell
        if ( is_first ) {
            const SI s0 = diagram::seed_rank( pd, i );
            for ( int d = 0; d < D; ++d )
                if constexpr ( g_pos )
                    atomic_add( grad_positions( s0, d ).ref(), gp0[ d ] );
            if constexpr ( g_w )
                atomic_add( grad_weights( s0 ).ref(), gw0 );
        } else {
            add_in( i, 2, gw0 );
            for ( int d = 0; d < D; ++d )
                add_in( i, 3 + d, gp0[ d ] );
        }
    }
}

} // namespace distributed
} // namespace sdot
