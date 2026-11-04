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
    const SI cap = diagram::cap_in<Local,TF>( cv.nb_words, 1, false );
    Local c;
    c.attach( cv, cap );

    const bool is_first = bool( SI( first( 0 ) ) );
    const SI   nb_items = is_first ? SI( pd.nb_seeds() ) : SI( inbox.shape( 0 ) );
    const SI   width    = SI( out_msgs.shape( 1 ) );
    const int  mcap     = int( ( width - header_words( D ) ) / 3 );
    const int  K        = int( boxes.shape( 0 ) );
    const SI   me       = SI( my_box( 0 ) );
    const SI   off      = SI( id_offset( 0 ) );

    for ( SI i = thread_index; i < nb_items; i += nb_threads ) {
        // the cell to work on, and its seed
        TF p0[ D ], w0, owner;
        std::uint64_t visited;
        if ( is_first ) {
            if ( ! c.load( dom ) ) { ask_more( scratch, cv ); return; }
            const auto p = pd.point( i );
            for ( int d = 0; d < D; ++d ) p0[ d ] = p[ d ];
            w0      = pd.weight( i );
            owner   = TF( gids( pd.user_id( i ) ) );
            visited = 0;
        } else {
            owner   = inbox( i, 0 );
            if ( owner < 0 ) { out_dest( i ) = -2; continue; }  // an empty row of a fixed-capacity inbox
            visited = std::uint64_t( inbox( i, 1 ) );
            w0      = inbox( i, 2 );
            for ( int d = 0; d < D; ++d ) p0[ d ] = inbox( i, 3 + d );
            const int nb = int( inbox( i, 3 + D ) );
            if ( nb > cap ) { ask_more( scratch, cv ); return; }
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
            for ( int d = 0; d < D; ++d ) { inner.p0[ d ] = p0[ d ]; inner.q0[ d ] = TK( p0[ d ] ); }
            inner.w0 = w0;
            inner.v0 = TK( w0 );
        }
        OffsetIds<decltype( inner )> f{ inner, off };
        if ( run<PD::on_cpu>( c, f ) == CutStatus::NO_ROOM ) { ask_more( scratch, cv ); return; }
        visited |= std::uint64_t( 1 ) << me;

        // the nearest box not visited yet that can still cut it
        TK q0[ D ];
        for ( int d = 0; d < D; ++d ) q0[ d ] = TK( p0[ d ] );
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
            else if ( ! c.nb || ! can_cut_box<true,TK,D>( c.nb, v, q0, TK( w0 ), B ) )
                continue;
            if ( best < 0 || dist < best_dist ) { best = j; best_dist = dist; }
        }

        if ( best < 0 ) {
            out_dest( i ) = -1;
            out_gid ( i ) = SI( owner );
            out_val ( i ) = c.nb ? c.template measure<TF>() : TF( 0 );
            continue;
        }

        const int nb = c.nb;
        if ( nb > mcap ) { status( 0 ) = 1; out_dest( i ) = -1; out_gid( i ) = SI( owner ); out_val( i ) = 0; continue; }
        out_dest( i ) = best;
        out_msgs( i, 0 ) = owner;
        out_msgs( i, 1 ) = TF( visited );
        out_msgs( i, 2 ) = w0;
        for ( int d = 0; d < D; ++d ) out_msgs( i, 3 + d ) = p0[ d ];
        out_msgs( i, 3 + D ) = TF( nb );
        for ( int q = 0; q < nb; ++q ) {
            out_msgs( i, header_words( D ) + q )            = TF( c.vx[ q ] );
            out_msgs( i, header_words( D ) + mcap + q )     = TF( c.vy[ q ] );
            out_msgs( i, header_words( D ) + 2 * mcap + q ) = TF( c.cid[ q ] );
        }
    }
}

} // namespace distributed
} // namespace sdot
