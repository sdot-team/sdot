#pragma once

#include <loom/support/common_macros.h> // HD

// =====================================================================================
// THE CELL IN DIMENSION `D >= 3`: a SIMPLE polytope that cuts itself.
//
// Nothing of 2D survives here: there is no global cyclic order any more, and a cut carries a
// FACE. Connectivity is therefore carried by the VERTICES, each one naming its `D` cuts and its `D`
// neighbors:
//
//     vk[ r ][ i ]   the `r`-th cut of vertex `i`, as an INDEX into `cid` -- sorted ascending
//     vn[ r ][ i ]   the neighbor of `i` ON THE OTHER SIDE of the edge carried by the `D - 1` cuts
//                    other than `vk[ r ][ i ]`: neighbor `r` is "opposite" cut `r`
//
// A vertex of a simple polytope has `D` cuts, hence `D` bundles of `D - 1` cuts, hence `D`
// edges, and this matching is a bijection. It makes FREE what an edge list made us
// search for: the faces that carry edge `r` are the cuts of the vertex minus the
// `r`-th one, without traversing anything.
//
// THE CUT LIST IS LOCAL: `cid[ 0 .. nc )` carries the global identifiers, everything else
// handles only indices into it. It only ever ADDS during the life of the cell -- a
// cut that loses all its vertices leaves a dead entry there -- and `compact()` removes them only
// when it is full.
//
// THE ASSUMPTION: each vertex is on EXACTLY `D` planes ( general position ). This is what makes
// the cut purely combinatorial: two new vertices of the created face are neighbors exactly
// when they share `D - 2` OLD cuts. A plane that goes through a vertex -- exactly, or within a
// band of a few ulps ( `on_the_plane` ) -- does not remove it, which is the only concession made
// to degenerate configurations.
//
// THE CUT FILLS THE HOLES: vertices that are outside leave free slots, the new vertices
// settle into them, and the survivors keep their index -- so their adjacency stays valid
// as is, and there is no renumbering table. The commit is `O( nm )`, not
// `O( nv )`; the only case where a kept vertex moves is that of a cut that removes more
// vertices than it creates, and then only the difference moves.
//
// This is the bench's `Cellule3D.h`, with the dimension as a parameter.
//
// `CANON`: THE CANONICAL PREDICATES ( `Canonical.h` ), any dimension `D >= 2`, in doubles. Each cut and the seed of the cell carry
// their GENERATOR ( set by `canon_start` and `canon_pending`, from `diagram::make_cell` ), each vertex its position relative to the
// smallest seed among its generators; a vertex is created at the solution of its own equations, and whether a plane removes it is a
// function of its generators and of the plane alone -- the same decision in every cell that has it. Without a generator ( a cell
// cut from Python, a made-up wall ), the usual cut.
// =====================================================================================

#include <loom/support/math.h>
#include <loom/support/common_types.h>
#include <loom/support/containers/Matrix.h>
#include <loom/support/containers/Vector.h>
#include <asimd/asimd.h>
#include "Scratch.h"
#include "Plane.h"
#include "State.h"
#include "Ids.h"
#include "Canonical.h"

#include <type_traits>
#include <limits>
#include <cmath>

namespace sdot {

template<class TK,int D,bool CANON = false>
struct __attribute__(( visibility( "default" ) )) LocalN {
    static_assert( D >= 3 || ( CANON && D >= 2 ), "below 3D, it is `Local2` ( or the canonical cell )" );
    static_assert( ! CANON || std::is_same_v<TK,double>, "the canonical predicates are in doubles" );
    static constexpr int  ct_dim = D;
    static constexpr bool canonical = CANON;
    using TKernel = TK;
    using PlaneT  = Plane<TK,D>;

    int  nv = 0, nc = 0;
    int  cap = 0;                                        ///< the capacity: vertices AND cuts
    bool unbounded  = false;
    bool has_planes = false;

    TK  *v [ D ];
    int *vk[ D ];
    int *vn[ D ];
    int *cid;
    TK  *pd[ D ], *po;                                   ///< the plane of cut `k` ( if `has_planes` )

    // the temporaries, in the scratch too
    TK  *s, *nx[ D ], *rate[ D ], *s3[ 3 ];
    int *hole, *nk[ D - 1 ], *rec_v, *rec_f, *dest, *nn_[ D ], *fresh, *src, *dst, *m, *apex[ D ], *fvl[ D ], *fcl[ D ], *v0, *on_face;

    // CANON: per cut its generator ( `gk` the kind, `gr` the rank or wall number, `gp` / `gs` / `gw` ), per vertex its position `ys`
    // relative to its frame and that frame `vs0` ( the cut whose generator it is, `-1` the seed of the cell, `-2` not canonical ), and
    // the same for the new vertices of a cut. `own`: the seed of the cell; `pend`: the generator of the next plane.
    double *gp[ D ], *gs[ D ], *gw, *ys[ D ], *yl[ D ], *nys[ D ], *nyl[ D ];   ///< ( `ys + yl`: a double-double )
    int    *gk, *gr, *vs0, *ns0;
    bool    own_set = false, pend_set = false, canon_active = false;
    canon::Gen<D> own, pend;

    // ---- the scratch --------------------------------------------------------------------------

    /// how many words are needed for `cap` vertices ( and as many cuts ) -- THE SAME FORMULA as
    /// `Cell_N.scratch_words`
    HD static constexpr SI words_for( SI cap ) {
        return ( 4 * D + 5 ) * words_of<TK>( cap ) + ( 7 * D + 10 ) * words_of<int>( cap )
             + ( CANON ? ( 6 * D + 1 ) * words_of<double>( cap ) + 4 * words_of<int>( cap ) : 0 );
    }

    HD bool attach( Carver &c, SI capacity ) {
        cap = int( capacity );
        for ( int d = 0; d < D; ++d ) { v[ d ] = c.take<TK>( cap ); vk[ d ] = c.take<int>( cap ); vn[ d ] = c.take<int>( cap ); }
        cid = c.take<int>( cap );
        for ( int d = 0; d < D; ++d ) pd[ d ] = c.take<TK>( cap );
        po = c.take<TK>( cap );
        s = c.take<TK>( cap );
        for ( int d = 0; d < D; ++d ) { nx[ d ] = c.take<TK>( cap ); rate[ d ] = c.take<TK>( cap ); }
        for ( int d = 0; d < 3; ++d ) s3[ d ] = c.take<TK>( cap );
        hole = c.take<int>( cap );
        for ( int d = 0; d + 1 < D; ++d ) nk[ d ] = c.take<int>( cap );
        rec_v = c.take<int>( cap ); rec_f = c.take<int>( cap ); dest = c.take<int>( cap );
        for ( int d = 0; d < D; ++d ) nn_[ d ] = c.take<int>( cap );
        fresh = c.take<int>( cap ); src = c.take<int>( cap ); dst = c.take<int>( cap ); m = c.take<int>( cap );
        for ( int d = 0; d < D; ++d ) apex[ d ] = c.take<int>( cap );
        for ( int d = 0; d < D; ++d ) fvl[ d ] = c.take<int>( cap );
        for ( int d = 0; d < D; ++d ) fcl[ d ] = c.take<int>( cap );
        v0 = c.take<int>( cap ); on_face = c.take<int>( cap );
        if constexpr ( CANON ) {
            for ( int d = 0; d < D; ++d ) {
                gp[ d ] = c.take<double>( cap ); gs[ d ] = c.take<double>( cap );
                ys[ d ] = c.take<double>( cap ); yl[ d ] = c.take<double>( cap ); nys[ d ] = c.take<double>( cap ); nyl[ d ] = c.take<double>( cap );
            }
            gw = c.take<double>( cap );
            gk = c.take<int>( cap ); gr = c.take<int>( cap ); vs0 = c.take<int>( cap ); ns0 = c.take<int>( cap );
        }
        nv = 0; nc = 0; unbounded = false; has_planes = false; own_set = false; pend_set = false;
        return ! c.overflow;
    }

    HD bool copy_from( const LocalN &o ) {
        if ( o.nv > cap || o.nc > cap )
            return false;
        nv = o.nv; nc = o.nc; unbounded = o.unbounded; has_planes = o.has_planes;
        for ( int i = 0; i < nv; ++i )
            for ( int d = 0; d < D; ++d ) { v[ d ][ i ] = o.v[ d ][ i ]; vk[ d ][ i ] = o.vk[ d ][ i ]; vn[ d ][ i ] = o.vn[ d ][ i ]; }
        for ( int k = 0; k < nc; ++k ) {
            cid[ k ] = o.cid[ k ];
            if ( has_planes ) { for ( int d = 0; d < D; ++d ) pd[ d ][ k ] = o.pd[ d ][ k ]; po[ k ] = o.po[ k ]; }
        }
        if constexpr ( CANON ) {
            own_set = o.own_set; own = o.own; pend_set = false;
            if ( own_set ) {
                for ( int k = 0; k < nc; ++k ) set_gen( k, o.gen_of( k ) );
                for ( int i = 0; i < nv; ++i ) { vs0[ i ] = o.vs0[ i ]; for ( int d = 0; d < D; ++d ) { ys[ d ][ i ] = o.ys[ d ][ i ]; yl[ d ][ i ] = o.yl[ d ][ i ]; } }
            }
        }
        return true;
    }

    // ---- what everybody reads ------------------------------------------------------------
    HD int  nb_vertices() const { return nv; }
    HD int  nb_cuts () const { return nc; }
    HD bool bounded () const { return ! unbounded; }
    HD TK   coord   ( int i, int d ) const { return v[ d ][ i ]; }
    HD int  vertex_cut ( int i, int r ) const { return vk[ r ][ i ]; }
    HD int  vertex_neighbor( int i, int r ) const { return vn[ r ][ i ]; }   ///< across from cut `r`: the edge carried by the other cuts of `i`

    HD StateMemN<TK,D> state() const {
        StateMemN<TK,D> e;
        e.nb = nv;
        for ( int d = 0; d < D; ++d )
            e.v[ d ] = v[ d ];
        e.bounded = ! unbounded;
        return e;
    }

    // ---- the starting states -----------------------------------------------------------------

    /// the parallelotope `origin + sum_j t_j axes( j )`, `t` in `[ 0, 1 ]^D`. Vertex `b` has the bits
    /// of `b` as coordinates; cut `2 j + bit_j( b )` carries it, and its neighbor opposite
    /// that cut is `b ^ ( 1 << j )`. The cuts are in axis order, hence sorted.
    HD bool init_hypercube( const auto &origin, const auto &axes, int cut_id ) {
        if ( ( 1 << D ) > cap )
            return false;
        nv = 1 << D;
        nc = 2 * D;
        for ( int b = 0; b < nv; ++b ) {
            for ( int d = 0; d < D; ++d ) {
                TK x = TK( origin[ d ] );
                for ( int j = 0; j < D; ++j )
                    if ( b & ( 1 << j ) )
                        x += TK( axes( j, d ) );
                v[ d ][ b ] = x;
            }
            for ( int j = 0; j < D; ++j ) {
                vk[ j ][ b ] = 2 * j + ( ( b >> j ) & 1 );
                vn[ j ][ b ] = b ^ ( 1 << j );
            }
        }
        for ( int k = 0; k < nc; ++k )
            cid[ k ] = cut_id;
        unbounded  = cut_id == cell_ids::INFINITE;
        has_planes = false;
        own_set = false; pend_set = false;
        if ( unbounded )
            planes_from_vertices();
        return true;
    }

    /// "THE WHOLE SPACE": the unit simplex, whose `D + 1` walls are marked `INFINITE`. Vertex
    /// 0 is the origin, on cuts `0 .. D-1` ( `x_c >= 0` ); vertex `n >= 1` is
    /// `e_{n-1}`, on the same ones minus `n-1`, plus cut `D` ( `sum x <= 1` ).
    HD bool init_unbounded() {
        if ( D + 1 > cap )
            return false;
        nv = D + 1;
        nc = D + 1;
        for ( int n = 0; n <= D; ++n )
            for ( int d = 0; d < D; ++d )
                v[ d ][ n ] = TK( d + 1 == n );
        for ( int r = 0; r < D; ++r ) {
            vk[ r ][ 0 ] = r;
            vn[ r ][ 0 ] = r + 1;                        // opposite `x_r >= 0`: along `e_r`
        }
        for ( int n = 1; n <= D; ++n ) {
            for ( int r = 0; r + 1 < D; ++r ) {
                const int c = r + ( r >= n - 1 );
                vk[ r ][ n ] = c;
                vn[ r ][ n ] = c + 1;                    // the vertex that also misses cut `c`
            }
            vk[ D - 1 ][ n ] = D;
            vn[ D - 1 ][ n ] = 0;                        // opposite the closing cut: the origin
        }
        for ( int k = 0; k <= D; ++k )
            cid[ k ] = cell_ids::INFINITE;
        for ( int c = 0; c < D; ++c ) {
            for ( int d = 0; d < D; ++d )
                pd[ d ][ c ] = - TK( d == c );
            po[ c ] = 0;
        }
        for ( int d = 0; d < D; ++d )
            pd[ d ][ D ] = 1;
        po[ D ] = 1;
        unbounded  = true;
        has_planes = true;
        own_set = false; pend_set = false;
        return true;
    }

    HD void make_empty() { nv = 0; nc = 0; unbounded = false; has_planes = false; }

    // ---- CANON: the generators ----------------------------------------------------------------

    HD canon::Gen<D> gen_of( int k ) const {
        canon::Gen<D> g;
        if constexpr ( CANON ) {
            g.kind = gk[ k ]; g.rank = gr[ k ]; g.w = gw[ k ];
            for ( int d = 0; d < D; ++d ) { g.p[ d ] = gp[ d ][ k ]; g.s[ d ] = gs[ d ][ k ]; }
        }
        return g;
    }
    HD void set_gen( int k, const canon::Gen<D> &g ) {
        if constexpr ( CANON ) {
            gk[ k ] = g.kind; gr[ k ] = g.rank; gw[ k ] = g.w;
            for ( int d = 0; d < D; ++d ) { gp[ d ][ k ] = g.p[ d ]; gs[ d ][ k ] = g.s[ d ]; }
        }
    }
    /// the generator of the frame of vertex `i` ( `vs0[ i ] >= -1` )
    HD canon::Gen<D> frame_of( int i ) const { return vs0[ i ] < 0 ? own : gen_of( vs0[ i ] ); }

    /// the `D + 1` generators of the vertex of cuts `cuts`: the seed of the cell first ( index `-1` ), then the cuts
    HD void gens_of( const int *cuts, canon::Gen<D> *g ) const {
        g[ 0 ] = own;
        for ( int r = 0; r < D; ++r ) g[ r + 1 ] = gen_of( cuts[ r ] );
    }

    /// solve the vertex of cuts `cuts` canonically: its position relative to its frame `y`, the frame `s0` ( a cut index, `-1` the
    /// seed, `-2` not canonical ), and its position `x` in the frame of the cell
    HD bool canon_vertex( const int *cuts, canon::DD *y, int &s0, TK *x ) const {
        canon::Gen<D> g[ D + 1 ];
        gens_of( cuts, g );
        int f = -1;
        if ( ! canon::solve<D>( g, y, f ) ) { s0 = -2; return false; }
        s0 = f == 0 ? -1 : cuts[ f - 1 ];
        for ( int d = 0; d < D; ++d )
            x[ d ] = TK( ( ( canon::two_sum( g[ f ].p[ d ], g[ f ].s[ d ] ) + y[ d ] ) ).hi );
        return true;
    }
    HD void y_of( int i, canon::DD *y ) const { for ( int d = 0; d < D; ++d ) y[ d ] = { ys[ d ][ i ], yl[ d ][ i ] }; }

    /// THE START OF A CANONICAL CELL ( after `load` of the domain ): the seed of the cell, then each cut of the domain is a wall
    /// numbered by its rank there ( the same in every cell ), with the plane of its vertices; each vertex gets its frame
    HD void canon_start( const canon::Gen<D> &seed ) {
        if constexpr ( CANON ) {
            own = seed;
            own_set = true;
            pend_set = false;
            for ( int k = 0; k < nc; ++k ) {
                canon::Gen<D> g;
                TK dir[ D ], off;
                if ( cid[ k ] != cell_ids::INFINITE && plane_of_cut( k, dir, off ) ) {
                    g.kind = canon::WALL;
                    g.rank = k;
                    for ( int d = 0; d < D; ++d ) g.p[ d ] = double( dir[ d ] );
                    g.w = double( off );
                }
                set_gen( k, g );
            }
            for ( int i = 0; i < nv; ++i ) {
                int cuts[ D ];
                for ( int r = 0; r < D; ++r ) cuts[ r ] = vk[ r ][ i ];
                canon::DD y[ D ];
                TK x[ D ];
                if ( canon_vertex( cuts, y, vs0[ i ], x ) )
                    for ( int d = 0; d < D; ++d ) { ys[ d ][ i ] = y[ d ].hi; yl[ d ][ i ] = y[ d ].lo; }
            }
        }
    }

    /// the generator of the plane that the next `cut` brings ( `diagram::make_cell` sets it from the provider )
    HD void canon_pending( const canon::Gen<D> &g ) {
        if constexpr ( CANON ) { pend = g; pend_set = own_set; }
    }

    /// whether vertex `i` stays on the inside of the plane of generator `j`: the sign of `side`, the perturbation for a tie
    HD bool canon_inside( int i, const canon::Gen<D> &j ) const {
        canon::DD y[ D ];
        y_of( i, y );
        const int sg = canon::sign_of<D>( frame_of( i ), y, j );
        if ( sg )
            return sg > 0;
        int cuts[ D ];
        for ( int r = 0; r < D; ++r ) cuts[ r ] = vk[ r ][ i ];
        canon::Gen<D> g[ D + 1 ];
        gens_of( cuts, g );
        return canon::sos_inside<D>( g, j );
    }

    /// whether the edge between vertex `a` and its neighbor `r` has zero length: each end on the plane of the other's extra
    /// generator, within `zero_band` ( a function of the two vertices alone: the same in every cell )
    HD bool canon_zero_edge( int a, int r ) const {
        const int b = vn[ r ][ a ];
        if ( vs0[ a ] == -2 || vs0[ b ] == -2 )
            return false;
        const int ca = vk[ r ][ a ];
        int cb = -1;
        for ( int q = 0; q < D; ++q ) {
            bool in = false;
            for ( int t = 0; t < D; ++t ) in |= vk[ q ][ b ] == vk[ t ][ a ];
            if ( ! in ) cb = vk[ q ][ b ];
        }
        if ( cb < 0 )
            return true;                                 // ( two vertices of the same cuts )
        canon::DD ya[ D ], yb[ D ];
        y_of( a, ya );
        y_of( b, yb );
        // ( BOTH ends: the extra plane of one end may contain the whole edge -- coincident bisectors, on a grid -- but not that of the
        // other end, which meets the line of the edge at that end only )
        return canon::sign_of<D>( frame_of( a ), ya, gen_of( cb ), canon::zero_band() ) == 0
            && canon::sign_of<D>( frame_of( b ), yb, gen_of( ca ), canon::zero_band() ) == 0;
    }

    // ---- the planes, read back from the geometry ---------------------------------------------------

    /// the OUTWARD normal of face `k` and its offset, read back from its vertices. `false` if the face
    /// does not span a hyperplane ( dead, or degenerate ).
    ///
    /// NOT "the first `D` vertices found": on a facet of dimension `D - 1 >= 3`, `D`
    /// vertices may well lie in the same 2-face, and two vertices coinciding to 1e-15
    /// ( a cut passed through a vertex ) are the common case. So we build an
    /// ORTHONORMAL basis of the space spanned by `p - p0`, by Gram-Schmidt, keeping only a point
    /// whose residual is clear; the normal is the generalized cross product of that basis.
    HD bool plane_of_cut( int k, TK *dir, TK &off ) const {
        int p0 = -1;
        TK  base[ D - 1 ][ D ];
        TK  scale = 0;
        int nb = 0;
        for ( int i = 0; i < nv; ++i ) {
            bool on = false;
            for ( int r = 0; r < D; ++r )
                on |= vk[ r ][ i ] == k;
            on_face[ i ] = on;
            if ( ! on )
                continue;
            if ( p0 < 0 ) { p0 = i; continue; }
            if ( nb == D - 1 )
                continue;
            TK u[ D ];
            TK n2 = 0;
            for ( int d = 0; d < D; ++d ) { u[ d ] = v[ d ][ i ] - v[ d ][ p0 ]; n2 += u[ d ] * u[ d ]; }
            if ( n2 > scale ) scale = n2;
            for ( int b = 0; b < nb; ++b ) {
                TK dot = 0;
                for ( int d = 0; d < D; ++d ) dot += u[ d ] * base[ b ][ d ];
                for ( int d = 0; d < D; ++d ) u[ d ] -= dot * base[ b ][ d ];
            }
            TK r2 = 0;
            for ( int d = 0; d < D; ++d ) r2 += u[ d ] * u[ d ];
            if ( r2 <= scale * TK( 1e-12 ) )
                continue;                                // in the space already spanned, or coincident
            const TK inv = 1 / std::sqrt( r2 );
            for ( int d = 0; d < D; ++d ) base[ nb ][ d ] = u[ d ] * inv;
            ++nb;
        }
        if ( nb < D - 1 )
            return false;

        // the generalized cross product of the `D - 1` basis vectors: `n_i` is the minor without
        // column `i`, with sign `( -1 )^i`
        for ( int i = 0; i < D; ++i ) {
            const auto M = Matrix<TK,D-1>::with_func( [&]( auto r, auto c ) {
                const int col = int( c ) + ( int( c ) >= i );
                return base[ int( r ) ][ col ];
            } );
            const TK m = M.determinant();
            dir[ i ] = ( i % 2 ) ? - m : m;
        }
        off = 0;
        for ( int d = 0; d < D; ++d )
            off += dir[ d ] * v[ d ][ p0 ];

        // outward: the vertex FARTHEST from the plane is inside ( not the first one that comes, which may be
        // a vertex coinciding with the face, 1e-16 off on either side )
        TK far = 0;
        for ( int i = 0; i < nv; ++i ) {
            if ( on_face[ i ] )
                continue;
            TK s = - off;
            for ( int d = 0; d < D; ++d )
                s += dir[ d ] * v[ d ][ i ];
            if ( ( s < 0 ? - s : s ) > ( far < 0 ? - far : far ) )
                far = s;
        }
        if ( far > 0 ) {
            for ( int d = 0; d < D; ++d )
                dir[ d ] = - dir[ d ];
            off = - off;
        }
        return true;
    }

    HD void planes_from_vertices() {
        for ( int k = 0; k < nc; ++k ) {
            TK dir[ D ], off;
            if ( ! plane_of_cut( k, dir, off ) ) {
                for ( int d = 0; d < D; ++d ) dir[ d ] = 0;
                off = 0;
            }
            for ( int d = 0; d < D; ++d )
                pd[ d ][ k ] = dir[ d ];
            po[ k ] = off;
        }
        has_planes = true;
    }

    template<class T>
    HD void plane( int k, T *dir, T &off ) const {
        if ( has_planes ) {
            for ( int d = 0; d < D; ++d ) dir[ d ] = T( pd[ d ][ k ] );
            off = T( po[ k ] );
        } else {
            TK dk[ D ], o = 0;
            if ( ! plane_of_cut( k, dk, o ) )
                for ( int d = 0; d < D; ++d ) dk[ d ] = 0;
            for ( int d = 0; d < D; ++d ) dir[ d ] = T( dk[ d ] );
            off = T( o );
        }
    }

    // ---- the cut ----------------------------------------------------------------------------

    HD int cut( const PlaneT &p ) {
        if ( unbounded )
            grow_for( p );
        return cut_impl( p );
    }

    /// THE FIRST PASS: `si[ i ] = dir . v_i - off` for every vertex, written to `out` ( if not null ),
    /// and the number of vertices strictly outside. Eight vertices per step, in registers: the
    /// carved arrays are runtime pointers, which a compiler cannot prove disjoint from `out`, so it
    /// does not vectorize the scalar loop. The tail is loaded PARTIALLY ( no byte beyond `nv`, and
    /// its dead lanes do not count ).
    HD int signed_distances( const PlaneT &p, TK *out ) const {
#ifdef __CUDACC__
        int nb_out = 0;                                  // the GPU has no register kernel: the plain loop
        for ( int i = 0; i < nv; ++i ) {
            TK si = - p.off;
            for ( int d = 0; d < D; ++d )
                si += p.dir[ d ] * v[ d ][ i ];
            if ( out )
                out[ i ] = si;
            nb_out += si > 0;
        }
        return nb_out;
#else
        using V = asimd::SimdVec<TK,8>;
        const V zero( TK( 0 ) ), moff( - p.off );
        V dir[ D ];
        for ( int d = 0; d < D; ++d )
            dir[ d ] = V( p.dir[ d ] );
        auto dist = [&]( auto load ) {
            V sv = moff;
            for ( int d = 0; d < D; ++d )
                sv = asimd::fma( dir[ d ], load( d ), sv );
            return sv;
        };
        int nb_out = 0;
        const int full = nv & ~7;
        int i = 0;
        for ( ; i < full; i += 8 ) {
            const V sv = dist( [&]( int d ) { return V::load_unaligned( v[ d ] + i ); } );
            if ( out )
                sv.store_unaligned( out + i );
            nb_out += __builtin_popcountll( asimd::to_bits( sv > zero ) );
        }
        if ( i < nv ) {
            const auto q = asimd::LaneRange<0>( nv - i );
            const V sv = dist( [&]( int d ) { return V::load_partial( v[ d ] + i, q ); } );
            if ( out )
                sv.store_partial( out + i, q );
            nb_out += __builtin_popcountll( asimd::to_bits( sv > zero, q ) );
        }
        return nb_out;
#endif
    }

    /// the number of vertices `cut` would remove ( same classification, band included )
    HD int nb_outside( const PlaneT &p ) const {
        const int strict = signed_distances( p, s );
        return strict ? on_the_plane( p, first_band() ) : 0;
    }

    // ---- "ON THE PLANE": a band, not a sign ---------------------------------------------------
    //
    // The cut is combinatorial, and it is only right if the outside set is one a plane can carve:
    // on each face, the outside vertices make ONE run. In the kernel's float, a plane that comes
    // within rounding of a face of the cell -- the bisectors of two seeds closer than the precision
    // seen from a third one, or two seeds at the same place whose weights differ by 1e-16 -- sees
    // the signs of that face's vertices as noise: several runs, new vertices that match three
    // others or none, and a cell that ends up with the wrong measure ( or NaN ). So a vertex is
    // outside only BEYOND a band of a few ulps of the terms of its dot product; inside the band it
    // is on the plane, and stays ( what `s == 0` already did ). What the band ignores is a sliver
    // of the size of the rounding.
    //
    // The band is not a proof: a vertex whose distance is near the band's edge can still flip. The
    // matching of the new vertices sees it ( a new vertex must find exactly `D - 1` neighbors on
    // the new face ), and the cut is then redone with a wider band -- nothing is committed before.

    HD static TK first_band() { return TK( 16 ) * std::numeric_limits<TK>::epsilon(); }

    /// the scale of the rounding of `dir . v_i - off`: `|off| + |dir|_1 max_d |v_d|`. Relative to the SIZE of the vertex, not to the
    /// terms of the dot product: a vertex is exact to a few ulps of its largest coordinate, and the coordinate along the normal may
    /// be only that rounding ( a vertex that should lie on `z = 0`, at `z = 1e-17` among coordinates of order 1 -- relative to its
    /// own terms, the band vanished and the vertex counted as outside: a cut no plane carves, dropped, a pyramid left in the cell )
    HD TK magnitude( const PlaneT &p, int i ) const {
        TK nd = 0, mv = 0;
        for ( int d = 0; d < D; ++d ) {
            const TK a = p.dir[ d ] < 0 ? - p.dir[ d ] : p.dir[ d ];
            const TK x = v[ d ][ i ] < 0 ? - v[ d ][ i ] : v[ d ][ i ];
            nd += a;
            mv = x > mv ? x : mv;
        }
        return ( p.off < 0 ? - p.off : p.off ) + nd * mv;
    }

    /// `s[ i ] = 0` for the vertices that are outside but within `band` of the plane ( relative to
    /// the magnitude of the terms of `dir . v_i - off` ). Returns the number still outside. Only
    /// runs on a cut that has something to remove, never on the common `UNCHANGED` path.
    HD int on_the_plane( const PlaneT &p, TK band ) const {
        int nb_out = 0;
        for ( int i = 0; i < nv; ++i ) {
            if ( ! ( s[ i ] > 0 ) )
                continue;
            if ( s[ i ] <= band * magnitude( p, i ) )
                s[ i ] = 0;
            else
                ++nb_out;
        }
        return nb_out;
    }

    HD int cut_impl( const PlaneT &p ) {
        if constexpr ( CANON )
            if ( pend_set )
                return cut_canonical( p );
        if ( signed_distances( p, s ) == 0 )
            return CutStatus::UNCHANGED;

        // the widest band, 1024 times the first one, is 4e-12 relative in double and 2e-3 in float:
        // past it, the cut is dropped rather than made inconsistent
        TK band = first_band();
        for ( int attempt = 0; attempt < 3; ++attempt, band *= 32 ) {
            const int nb_out = on_the_plane( p, band );
            if ( nb_out == 0 )
                return CutStatus::UNCHANGED;
            if ( nb_out == nv ) {
                nv = 0; nc = 0;
                unbounded = false;
                return CutStatus::EMPTY;
            }
            const int r = cut_classified( p );
            if ( r != INCONSISTENT )
                return r;
        }

        // THE OTHER TIE-BREAK: the vertices within the band count as OUTSIDE, and are made again on the plane ( a crossing edge
        // of an inside vertex then ends at them ). What a degenerate cell needs -- on an exact grid, the bisector of a neighbor goes
        // through vertices and edges of the cell that other bisectors made, and keeping the ties inside leaves an outside set that
        // no plane carves; dropping the cut then left a whole pyramid of the neighbor in the cell.
        signed_distances( p, s );
        {
            int nb_out = 0;
            for ( int i = 0; i < nv; ++i ) {
                if ( s[ i ] > - band * magnitude( p, i ) && ! ( s[ i ] > 0 ) )
                    s[ i ] = std::numeric_limits<TK>::min();
                nb_out += s[ i ] > 0;
            }
            if ( nb_out == nv ) {
                nv = 0; nc = 0;
                unbounded = false;
                return CutStatus::EMPTY;
            }
            const int r = cut_classified( p );
            if ( r != INCONSISTENT )
                return r;
        }
        return CutStatus::UNCHANGED;
    }

    /// what `cut_classified` returns when the outside set is not one a plane can carve ( see
    /// `on_the_plane` ); the cell is then untouched
    static constexpr int INCONSISTENT = -1;

    /// CANON: the cuts whose canonical outside set no plane carves ( the predicates of the band disagreeing with the geometry ),
    /// dropped -- what the check of the cell complex then sees
    int nb_inconsistent = 0;

    /// THE CANONICAL CUT: the geometric distances first, and the canonical predicate for the vertices within `1e-9` of the plane
    /// ( beyond, both are far from rounding and say the same ). `s` keeps the geometric distances for the made-up vertices, made
    /// positive or zero where the canonical decision overrides them.
    HD int cut_canonical( const PlaneT &p ) {
        const canon::Gen<D> gj = pend;
        pend_set = false;
        signed_distances( p, s );
        int nb_out = 0;
        for ( int i = 0; i < nv; ++i ) {
            bool out;
            const TK mg = magnitude( p, i );
            if ( gj.kind == canon::OTHER || vs0[ i ] == -2 )
                out = s[ i ] > 0;
            else if ( s[ i ] > TK( 1e-9 ) * mg )
                out = true;
            else if ( s[ i ] < - TK( 1e-9 ) * mg )
                out = false;
            else
                out = ! canon_inside( i, gj );
            if ( out && ! ( s[ i ] > 0 ) )
                s[ i ] = std::numeric_limits<TK>::min();
            if ( ! out && s[ i ] > 0 )
                s[ i ] = 0;
            nb_out += out;
        }
        if ( nb_out == 0 )
            return CutStatus::UNCHANGED;
        if ( nb_out == nv ) {
            nv = 0; nc = 0;
            unbounded = false;
            return CutStatus::EMPTY;
        }
        pend = gj;
        canon_active = true;
        const int r = cut_classified( p );
        canon_active = false;
        if ( r == INCONSISTENT ) {
            ++nb_inconsistent;
            return CutStatus::UNCHANGED;
        }
        return r;
    }

    /// the cut proper, `s` classified ( `> 0`: outside ), at least one vertex on each side
    HD int cut_classified( const PlaneT &p ) {
        if ( nc >= cap ) {
            compact();
            if ( nc >= cap )
                return CutStatus::NO_ROOM;
        }
        const int knew = nc;
        if constexpr ( CANON )
            if ( canon_active )
                set_gen( knew, pend );

        // ---- A SINGLE PASS OVER THE VERTICES: the holes left by the outside vertices and, for
        // each of them, its crossing edges -- from which the new vertices are born. `nk`: the
        // INHERITED cuts, sorted; `rec_v` / `rec_f`: the INSIDE vertex to reattach, and its slot.
        int nt = 0, nm = 0;

        for ( int o = 0; o < nv; ++o ) {
            if ( ! ( s[ o ] > 0 ) )
                continue;
            hole[ nt++ ] = o;
            for ( int j = 0; j < D; ++j ) {
                const int u = vn[ j ][ o ];
                if ( s[ u ] > 0 )
                    continue;                            // edge entirely outside: it dies
                if ( nm >= cap )
                    return CutStatus::NO_ROOM;

                // ANCHORED ON THE INSIDE VERTEX: with `s_u == 0` the symmetric form does not return `v_u`
                // in floating point, and the vertex would end up on the other side of the plane.
                const TK t = s[ u ] / ( s[ u ] - s[ o ] );
                for ( int d = 0; d < D; ++d )
                    nx[ d ][ nm ] = v[ d ][ u ] + ( v[ d ][ o ] - v[ d ][ u ] ) * t;
                for ( int r = 0, q = 0; r < D; ++r )
                    if ( r != j )
                        nk[ q++ ][ nm ] = vk[ r ][ o ];
                if constexpr ( CANON ) {
                    // the new vertex at the solution of its own equations ( the interpolation for a made-up one )
                    ns0[ nm ] = -2;
                    if ( canon_active ) {
                        int cuts[ D ];
                        for ( int q = 0; q + 1 < D; ++q ) cuts[ q ] = nk[ q ][ nm ];
                        cuts[ D - 1 ] = knew;
                        canon::DD y[ D ];
                        TK x[ D ];
                        if ( canon_vertex( cuts, y, ns0[ nm ], x ) )
                            for ( int d = 0; d < D; ++d ) { nys[ d ][ nm ] = y[ d ].hi; nyl[ d ][ nm ] = y[ d ].lo; nx[ d ][ nm ] = x[ d ]; }
                    }
                }

                rec_v[ nm ] = u;
                int f = 0;
                for ( int r = 0; r < D; ++r )
                    if ( vn[ r ][ u ] == o )
                        f = r;
                rec_f[ nm ] = f;
                ++nm;
            }
        }

        const int nn = nv - nt;
        const int new_nv = nn + nm;
        if ( new_nv > cap )
            return CutStatus::NO_ROOM;

        // where each new vertex goes: into a hole while any remain, then appended
        for ( int j = 0; j < nm; ++j )
            dest[ j ] = j < nt ? hole[ j ] : nv + ( j - nt );

        // ---- THE NEIGHBORS OF THE NEW VERTICES. Opposite `knew` ( slot `D - 1` ): the inside end
        // it comes from. The others are its neighbors ON THE NEW FACE: two new vertices are
        // neighbors exactly when they share `D - 2` old cuts, and the edge joining them
        // is then opposite the inherited cut that they do NOT share.
        for ( int i = 0; i < nm; ++i ) {
            for ( int r = 0; r + 1 < D; ++r )
                nn_[ r ][ i ] = -1;
            nn_[ D - 1 ][ i ] = rec_v[ i ];
            m[ i ] = 0;                                  // how many face neighbors it found
        }
        for ( int i = 0; i < nm; ++i ) {
            int ki[ D - 1 ];
            for ( int a = 0; a < D - 1; ++a )
                ki[ a ] = nk[ a ][ i ];
            for ( int j = i + 1; j < nm; ++j ) {
                // both lists are sorted and strictly increasing: the number of common cuts is the number of
                // entries of `i` found in `j`, and the entry of each that is NOT shared ( the edge's
                // opposite cut ) falls out of the same comparisons. Branch-free: a merge with a
                // data-dependent branch per step mispredicts about as often as it decides.
                int kj[ D - 1 ];
                for ( int b = 0; b < D - 1; ++b )
                    kj[ b ] = nk[ b ][ j ];
                int common = 0, ai = 0, bj = 0;
                for ( int a = 0; a < D - 1; ++a ) {
                    bool in = false;
                    for ( int b = 0; b < D - 1; ++b )
                        in |= ki[ a ] == kj[ b ];
                    common += in;
                    ai = in ? ai : a;
                }
                if ( common != D - 2 )
                    continue;
                for ( int b = 0; b < D - 1; ++b ) {
                    bool in = false;
                    for ( int a = 0; a < D - 1; ++a )
                        in |= kj[ b ] == ki[ a ];
                    bj = in ? bj : b;
                }
                nn_[ ai ][ i ] = dest[ j ];
                nn_[ bj ][ j ] = dest[ i ];
                ++m[ i ];
                ++m[ j ];
            }
        }
        // a plane-carved outside gives each new vertex exactly one neighbor per inherited cut
        for ( int i = 0; i < nm; ++i ) {
            bool ok = m[ i ] == D - 1;
            for ( int r = 0; r + 1 < D; ++r )
                ok &= nn_[ r ][ i ] >= 0;
            if ( ! ok )
                return INCONSISTENT;
        }

        // ---- COMMIT. Nothing has moved so far.
        for ( int j = 0; j < nm; ++j ) {
            const int m = dest[ j ];
            for ( int d = 0; d < D; ++d )
                v[ d ][ m ] = nx[ d ][ j ];
            for ( int r = 0; r + 1 < D; ++r )
                vk[ r ][ m ] = nk[ r ][ j ];             // < `knew`, hence sorted
            vk[ D - 1 ][ m ] = knew;
            for ( int r = 0; r < D; ++r )
                vn[ r ][ m ] = nn_[ r ][ j ];
            if constexpr ( CANON ) {
                vs0[ m ] = ns0[ j ];
                for ( int d = 0; d < D; ++d ) { ys[ d ][ m ] = nys[ d ][ j ]; yl[ d ][ m ] = nyl[ d ][ j ]; }
            }
        }
        for ( int i = 0; i < nm; ++i )                   // the reattachment, on the INSIDE vertex side
            vn[ rec_f[ i ] ][ rec_v[ i ] ] = dest[ i ];

        // ---- THE HOLES THAT REMAIN, when the cut removes more vertices than it creates
        if ( nm < nt ) {
            int th = nt;
            while ( th > nm && hole[ th - 1 ] >= new_nv ) --th;

            int nmv = 0;
            int ct = th, cd = nm;
            for ( int i = new_nv; i < nv; ++i ) {
                if ( ct < nt && hole[ ct ] == i ) { ++ct; continue; }   // this slot IS a hole
                src[ nmv ] = i;
                dst[ nmv ] = hole[ cd++ ];
                fresh[ i - new_nv ] = dst[ nmv ];
                ++nmv;
            }
            for ( int t = 0; t < nmv; ++t ) {
                const int a = src[ t ], b = dst[ t ];
                for ( int d = 0; d < D; ++d ) v[ d ][ b ] = v[ d ][ a ];
                for ( int r = 0; r < D; ++r ) { vk[ r ][ b ] = vk[ r ][ a ]; vn[ r ][ b ] = vn[ r ][ a ]; }
                if constexpr ( CANON ) {
                    vs0[ b ] = vs0[ a ];
                    for ( int d = 0; d < D; ++d ) { ys[ d ][ b ] = ys[ d ][ a ]; yl[ d ][ b ] = yl[ d ][ a ]; }
                }
            }
            // and the neighbors that pointed to them; a neighbor may itself have moved
            for ( int t = 0; t < nmv; ++t ) {
                const int a = src[ t ], b = dst[ t ];
                for ( int j = 0; j < D; ++j ) {
                    const int w = vn[ j ][ b ];
                    const int q = w >= new_nv ? fresh[ w - new_nv ] : w;
                    for ( int r = 0; r < D; ++r )
                        if ( vn[ r ][ q ] == a ) { vn[ r ][ q ] = b; break; }
                }
            }
        }

        cid[ knew ] = p.id;
        if ( has_planes ) {
            for ( int d = 0; d < D; ++d )
                pd[ d ][ knew ] = p.dir[ d ];
            po[ knew ] = p.off;
        }
        nc = knew + 1;
        nv = new_nv;

        if ( unbounded ) {
            unbounded = false;
            for ( int i = 0; i < nv; ++i )
                for ( int r = 0; r < D; ++r )
                    unbounded |= cid[ vk[ r ][ i ] ] == cell_ids::INFINITE;
            if ( ! unbounded )
                has_planes = false;
        }
        return CutStatus::CUT;
    }

    /// before putting the cell into memory: dead cuts do not leave from here
    HD void tidy() { compact(); }

    /// REMOVE THE DEAD CUTS. The renumbering is MONOTONE, so the lists stay sorted.
    HD void compact() {
        for ( int k = 0; k < nc; ++k ) m[ k ] = -1;
        for ( int i = 0; i < nv; ++i )
            for ( int r = 0; r < D; ++r )
                m[ vk[ r ][ i ] ] = 0;
        int q = 0;
        for ( int k = 0; k < nc; ++k ) {
            if ( m[ k ] != 0 )
                continue;
            cid[ q ] = cid[ k ];
            if ( has_planes ) {
                for ( int d = 0; d < D; ++d ) pd[ d ][ q ] = pd[ d ][ k ];
                po[ q ] = po[ k ];
            }
            if constexpr ( CANON )
                if ( own_set && q != k )
                    set_gen( q, gen_of( k ) );
            m[ k ] = q++;
        }
        for ( int i = 0; i < nv; ++i )
            for ( int r = 0; r < D; ++r )
                vk[ r ][ i ] = m[ vk[ r ][ i ] ];
        if constexpr ( CANON )
            if ( own_set )
                for ( int i = 0; i < nv; ++i )
                    if ( vs0[ i ] >= 0 )
                        vs0[ i ] = m[ vs0[ i ] ];
        nc = q;
    }

    // ---- the replacement simplex -----------------------------------------------------------

    /// the velocity of vertex `i` when the `INFINITE` walls are pushed back: it solves the `D x D` of
    /// its cuts with the `INFINITE` indicators as right-hand side
    HD void growth_rate( int i, TK *rate ) const {
        bool any = false;
        for ( int r = 0; r < D; ++r )
            any |= cid[ vk[ r ][ i ] ] == cell_ids::INFINITE;
        if ( ! any ) {
            for ( int d = 0; d < D; ++d ) rate[ d ] = 0;
            return;
        }
        const auto A = Matrix<TK,D>::with_func( [&]( auto r, auto c ) { return pd[ int( c ) ][ vk[ int( r ) ][ i ] ]; } );
        const auto b = Vector<TK,D>::with_func( [&]( PI r ) { return TK( cid[ vk[ r ][ i ] ] == cell_ids::INFINITE ); } );
        const auto x = Matrix<TK,D>::solve_ge( A, b );
        for ( int d = 0; d < D; ++d )
            rate[ d ] = x[ d ];
    }

    HD void grow_for( const PlaneT &p ) {
        if ( ! has_planes )
            planes_from_vertices();

        static constexpr int max_rounds = 4;
        // THE MARGIN IS NOT A MACHINE EPSILON: a vertex pushed back by an epsilon falls back to the
        // precision of the dot product, and two vertices coinciding to 1e-15 then get classified
        // each on its own side of the plane -- which breaks the combinatorics of the cut ( seen in 4D ).
        // It is small compared to the geometry, large compared to rounding: where the FAKE
        // vertices land has no geometric meaning anyway.
        const TK margin = std::is_same_v<TK,float> ? TK( 1e-5 ) : TK( 1e-6 );

        for ( int i = 0; i < nv; ++i ) {
            TK r[ D ];
            growth_rate( i, r );
            for ( int d = 0; d < D; ++d )
                rate[ d ][ i ] = r[ d ];
        }

        // "the velocity does not change the distance to the plane" is judged AT A TOLERANCE, not at zero:
        // the planes are READ BACK from the geometry, so an exactly axial normal comes out with
        // components at 1e-17, and `root = - s / rate` would then make a push of 1e17 -- and 1e38 on the
        // next round. A ray parallel to the plane to within 1e-9 counts as parallel.
        TK nd = 0;
        for ( int d = 0; d < D; ++d )
            nd += p.dir[ d ] * p.dir[ d ];
        const TK tol = TK( 1e-9 ) * std::sqrt( nd );

        TK g = 0;
        for ( int round = 0; round < max_rounds; ++round ) {
            bool push = false;
            TK grow = 0;
            for ( int i = 0; i < nv; ++i ) {
                TK dr = 0, s = - p.off, nr = 0;
                for ( int d = 0; d < D; ++d ) {
                    dr += p.dir[ d ] * rate[ d ][ i ];
                    s  += p.dir[ d ] * ( v[ d ][ i ] + g * rate[ d ][ i ] );
                    nr += rate[ d ][ i ] * rate[ d ][ i ];
                }
                if ( nr == 0 || ( dr < 0 ? - dr : dr ) <= tol * std::sqrt( nr ) )
                    continue;
                const TK root = - s / dr;
                if ( root >= 0 ) {
                    push = true;
                    if ( root > grow )
                        grow = root;
                }
            }
            if ( ! push )
                break;
            g += grow + ( grow + 1 ) * margin;
        }
        if ( g == 0 )
            return;

        for ( int k = 0; k < nc; ++k )
            if ( cid[ k ] == cell_ids::INFINITE )
                po[ k ] += g;
        for ( int i = 0; i < nv; ++i )
            for ( int d = 0; d < D; ++d )
                v[ d ][ i ] += g * rate[ d ][ i ];
    }

    // ---- the measure: a fan of simplices over the face lattice ----------------------

    HD bool has_cut( int i, int k ) const {
        for ( int r = 0; r < D; ++r )
            if ( vk[ r ][ i ] == k )
                return true;
        return false;
    }

    /// `func( chain )` for each simplex of the standard triangulation: a vertex of the cell,
    /// coned over each facet that does not contain it, each triangulated the same way one dimension
    /// lower. For each face encountered we need ONE of its vertices ( "the apex" ): `apex` keeps
    /// one row per depth and one slot per cut, filled in a single pass over the vertices
    /// of the face -- `D * nc` words, where indexing the faces by their SET of cuts costs
    /// `nc^( D - 1 )`. The vertices of the face at each depth are LISTED in `fvl[ depth ]` ( the
    /// parent's list filtered by the new cut ): a face is visited in `O( its vertices )`, not `O( nv )`
    /// -- what makes 5D and beyond affordable, where cells have hundreds of vertices and the fan
    /// thousands of faces.
    HD void for_each_simplex( auto &&func ) const {
        if ( nv == 0 )
            return;
        Vector<SI,D+1> chain;
        Vector<SI,D> face_cuts;
        chain[ 0 ] = 0;
        for ( int i = 0; i < nv; ++i )
            fvl[ 0 ][ i ] = i;
        for ( int d = 0; d < D; ++d )
            for ( int c = 0; c < nc; ++c )
                apex[ d ][ c ] = -1;
        for_each_simplex_rec( chain, face_cuts, nv, func, Ct<int,D>() );
    }

    /// the fan of the face cut by `face_cuts[ 0 .. depth )`, whose `len` vertices are `fvl[ depth ][ 0 .. len )`
    template<int K>
    HD void for_each_simplex_rec( Vector<SI,D+1> &chain, Vector<SI,D> &face_cuts, int len, auto &&func, Ct<int,K> ) const {
        constexpr int depth = D - K;                     // the number of cuts that define the face
        if constexpr ( K == 0 ) {
            func( chain );
        } else {
            const SI p = chain[ depth ];                 // the apex of the face, fixed for this subtree
            const int *fv = fvl[ depth ];
            int *ap = apex[ depth ], *fc = fcl[ depth ], nf = 0;
            for ( int j = 0; j < len; ++j ) {
                const int i = fv[ j ];
                for ( int r = 0; r < D; ++r ) {
                    const int c = vk[ r ][ i ];
                    bool in = false;
                    for ( int m = 0; m < depth; ++m )
                        in |= int( face_cuts[ m ] ) == c;
                    if ( ! in && ap[ c ] < 0 ) {
                        ap[ c ] = i;
                        fc[ nf++ ] = c;
                    }
                }
            }
            // `apex` is -1 everywhere between two visits: reset what this face set, the facets kept in `fcl`
            for ( int f = 0; f < nf; ++f )
                ap[ fc[ f ] ] = -1;
            // ... then cone `p` over the facets that do NOT contain it ( the others would give
            // flat simplices )
            for ( int f = 0; f < nf; ++f ) {
                const int c = fc[ f ];
                if ( has_cut( int( p ), c ) )
                    continue;
                // the vertices of the sub-face, in the same order: its apex is the first one
                int sub = 0, a = -1;
                if constexpr ( K > 1 ) {
                    int *child = fvl[ depth + 1 ];
                    for ( int j = 0; j < len; ++j )
                        if ( has_cut( fv[ j ], c ) )
                            child[ sub++ ] = fv[ j ];
                    a = child[ 0 ];
                } else {
                    for ( int j = 0; j < len && a < 0; ++j )
                        if ( has_cut( fv[ j ], c ) )
                            a = fv[ j ];
                }
                face_cuts[ depth ] = c;
                chain[ depth + 1 ] = a;
                for_each_simplex_rec( chain, face_cuts, sub, func, Ct<int,K-1>() );
            }
        }
    }

    /// IN 3D: ACCUMULATE THE FACES RATHER THAN ORDER THEM. Neither the volume nor the area of a face
    /// needs the ORDER of the vertices: it is enough, per face, to have ONE of its vertices `v_f` and the sum
    /// `S_f` of the cross products of its edges seen from it -- the face is planar and convex,
    /// so the triangles `( v_f, edge )` tile it and their cross products are parallel.
    /// The volume is `sum_f | ( v_f - g ) . S_f | / 6` for any interior `g`. Two
    /// passes without ever searching: `O( V + E )` instead of `O( F ( V + E ) )` for the cycles --
    /// measured on the bench: -15 % on the whole 3D diagram.
    /// IN 3D: per cut `f`, ONE vertex `v0[ f ]` and the sum `s3[ . ][ f ]` of the cross products
    /// of its edges seen from it -- twice the area vector of the face. What `measure_3d` and
    /// `for_each_facet` both read ( see `measure_3d` for why to accumulate rather than order ).
    HD void accumulate_faces_3d() const {
        static_assert( D == 3 );
        TK *const *s = s3;
        for ( int k = 0; k < nc; ++k ) { v0[ k ] = -1; s[ 0 ][ k ] = s[ 1 ][ k ] = s[ 2 ][ k ] = 0; }
        for ( int i = nv - 1; i >= 0; --i )
            for ( int r = 0; r < 3; ++r )
                v0[ vk[ r ][ i ] ] = i;

        for ( int a = 0; a < nv; ++a ) {
            for ( int j = 0; j < 3; ++j ) {
                const int b = vn[ j ][ a ];
                if ( b <= a )
                    continue;                            // each edge seen only once
                // the two faces that carry edge `j`: the cuts of the vertex minus the `j`-th
                for ( int r = 0; r < 3; ++r ) {
                    if ( r == j )
                        continue;
                    const int f = vk[ r ][ a ];
                    const int o = v0[ f ];
                    const TK ax = v[0][a] - v[0][o], ay = v[1][a] - v[1][o], az = v[2][a] - v[2][o];
                    const TK bx = v[0][b] - v[0][o], by = v[1][b] - v[1][o], bz = v[2][b] - v[2][o];
                    TK cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
                    if ( s[0][f] * cx + s[1][f] * cy + s[2][f] * cz < 0 ) { cx = -cx; cy = -cy; cz = -cz; }
                    s[0][f] += cx; s[1][f] += cy; s[2][f] += cz;
                }
            }
        }
    }

    /// IN 3D: `func( o, a, b )`, the vertex indices of a FAN of triangles that tiles the face carried by cut `f` -- `o` one of its
    /// vertices, `( a, b )` each of its edges once ( those through `o` give flat triangles, harmless ). The face is planar and convex, so
    /// the triangles all have the same orientation. Reads the TOPOLOGY only and writes nothing: it may be called from inside
    /// `for_each_facet` ( `SumOfGaussians::facet_mass` does ).
    HD void for_each_facet_triangle( int f, auto &&func ) const {
        static_assert( D == 3 );
        int o = -1;
        for ( int i = 0; i < nv && o < 0; ++i )
            for ( int r = 0; r < 3; ++r )
                if ( vk[ r ][ i ] == f )
                    o = i;
        if ( o < 0 )
            return;
        for ( int a = 0; a < nv; ++a ) {
            for ( int j = 0; j < 3; ++j ) {
                const int b = vn[ j ][ a ];
                if ( b <= a )
                    continue;
                bool on_f = false;                       // the edge is on the face if `f` is one of the two cuts that carry it
                for ( int r = 0; r < 3; ++r )
                    on_f = on_f || ( r != j && vk[ r ][ a ] == f );
                if ( on_f )
                    func( o, a, b );
            }
        }
    }

    /// IN 3D: the AREA and the CENTROID of the face carried by cut `f` ( `false` if it has none ) -- what the integral of an affine
    /// function on the face needs: `area * f( centroid )`. The triangles of `for_each_facet_triangle`, each weighing its own area,
    /// at its own center.
    template<class TF>
    HD bool facet_centroid( int f, TF ( &g )[ 3 ], TF &area ) const {
        static_assert( D == 3 );
        if ( nv < 4 )
            return false;
        TF w = 0, m[ 3 ] = { 0, 0, 0 };
        for_each_facet_triangle( f, [&]( int o, int a, int b ) {
            const TF ax = v[0][a] - v[0][o], ay = v[1][a] - v[1][o], az = v[2][a] - v[2][o];
            const TF bx = v[0][b] - v[0][o], by = v[1][b] - v[1][o], bz = v[2][b] - v[2][o];
            const TF cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
            const TF t = sdot::sqrt( cx * cx + cy * cy + cz * cz ) / 2;
            w += t;
            m[ 0 ] += t * ( v[0][o] + v[0][a] + v[0][b] ) / 3;
            m[ 1 ] += t * ( v[1][o] + v[1][a] + v[1][b] ) / 3;
            m[ 2 ] += t * ( v[2][o] + v[2][a] + v[2][b] ) / 3;
        } );
        if ( ! ( w > 0 ) )
            return false;
        area = w;
        for ( int c = 0; c < 3; ++c )
            g[ c ] = m[ c ] / w;
        return true;
    }

    /// `func( c, measure )` for each cut `c` that carries a face: its ( D - 1 )-measure. In 3D the accumulated cross products;
    /// beyond, the fan of `for_each_simplex` started ON the face ( depth 1, apex one of its vertices ), each simplex measured
    /// by its Gram determinant -- `nc` fans per cell, enough for the dimensions where cells are few.
    template<class TF>
    HD void for_each_facet( auto &&func ) const {
        if ( nv < D + 1 )
            return;
        if constexpr ( D != 3 ) {
            TF fact = 1;
            for ( int i = 2; i < D; ++i )
                fact *= i;
            Vector<SI,D+1> chain;
            Vector<SI,D> face_cuts;
            for ( int d = 0; d < D; ++d )
                for ( int c = 0; c < nc; ++c )
                    apex[ d ][ c ] = -1;
            for ( int c = 0; c < nc; ++c ) {
                int len = 0;
                for ( int i = 0; i < nv; ++i )
                    if ( has_cut( i, c ) )
                        fvl[ 1 ][ len++ ] = i;
                if ( len == 0 )
                    continue;
                const int a = fvl[ 1 ][ 0 ];
                chain[ 0 ] = a;
                chain[ 1 ] = a;
                face_cuts[ 0 ] = c;
                TF sum = 0;
                for_each_simplex_rec( chain, face_cuts, len, [&]( const auto &ch ) {
                    TF e[ D - 1 ][ D ];
                    for ( int k = 0; k < D - 1; ++k )
                        for ( int d = 0; d < D; ++d )
                            e[ k ][ d ] = TF( v[ d ][ ch[ k + 2 ] ] ) - TF( v[ d ][ ch[ 1 ] ] );
                    const auto G = Matrix<TF,D-1>::with_func( [&]( auto r, auto s ) {
                        TF g = 0;
                        for ( int d = 0; d < D; ++d )
                            g += e[ int( r ) ][ d ] * e[ int( s ) ][ d ];
                        return g;
                    } );
                    const TF det = G.determinant();
                    if ( det > 0 )
                        sum += sdot::sqrt( det );
                }, Ct<int,D-1>() );
                if ( sum > 0 )
                    func( c, sum / fact );
            }
        } else {
            accumulate_faces_3d();
            TK *const *s = s3;
            for ( int k = 0; k < nc; ++k ) {
                if ( v0[ k ] < 0 )
                    continue;
                const TF sx = TF( s[0][k] ), sy = TF( s[1][k] ), sz = TF( s[2][k] );
                func( k, sdot::sqrt( sx * sx + sy * sy + sz * sz ) / 2 );
            }
        }
    }

    template<class TF>
    HD TF measure_3d() const {
        static_assert( D == 3 );
        if ( nv < 4 )
            return 0;
        // the accumulation is done in the KERNEL's float ( the `s3` arrays are `TK` );
        // the final sum is in `TF`
        accumulate_faces_3d();
        TK *const *s = s3;

        TF g[ 3 ] = { 0, 0, 0 };
        for ( int i = 0; i < nv; ++i )
            for ( int d = 0; d < 3; ++d )
                g[ d ] += TF( v[ d ][ i ] );
        for ( int d = 0; d < 3; ++d )
            g[ d ] /= nv;

        TF vol = 0;
        for ( int k = 0; k < nc; ++k ) {
            const int o = v0[ k ];
            if ( o < 0 )
                continue;                                // dead cut, not yet compacted
            const TF t = ( TF( v[0][o] ) - g[0] ) * TF( s[0][k] ) + ( TF( v[1][o] ) - g[1] ) * TF( s[1][k] ) + ( TF( v[2][o] ) - g[2] ) * TF( s[2][k] );
            vol += t < 0 ? -t : t;
        }
        return vol / 6;
    }

    template<class TF>
    HD TF measure() const {
        if ( unbounded )
            return std::numeric_limits<TF>::max();
        if constexpr ( D == 3 )
            return measure_3d<TF>();
        TF sum = 0;
        for_each_simplex( [&]( const auto &chain ) {
            const auto M = Matrix<TF,D>::with_func( [&]( auto r, auto c ) {
                return TF( v[ int( r ) ][ chain[ int( c ) + 1 ] ] ) - TF( v[ int( r ) ][ chain[ 0 ] ] );
            } );
            const TF det = M.determinant();
            sum += det < 0 ? - det : det;
        } );
        TF fact = 1;
        for ( int i = 2; i <= D; ++i )
            fact *= i;
        return sum / fact;
    }

    /// the adjoint: `grad_vp( i, d )` ACCUMULATES ( a vertex is in several simplices ), so it
    /// is zeroed first, over the `nv` vertices.
    template<class TF>
    HD void measure_bwd( TF grad_res, auto &&grad_vp ) const {
        for ( int i = 0; i < nv; ++i )
            for ( int d = 0; d < D; ++d )
                grad_vp( i, d ) = 0;
        if ( unbounded )
            return;

        TF fact = 1;
        for ( int i = 2; i <= D; ++i )
            fact *= i;
        const TF g = grad_res / fact;

        // `d|det|/dM = sign( det ) * cofactor( M )`, and each column of `M` is an apex minus the
        // first, which therefore receives MINUS the sum of the columns
        for_each_simplex( [&]( const auto &chain ) {
            const auto M = Matrix<TF,D>::with_func( [&]( auto r, auto c ) {
                return TF( v[ int( r ) ][ chain[ int( c ) + 1 ] ] ) - TF( v[ int( r ) ][ chain[ 0 ] ] );
            } );
            const TF det = M.determinant();
            const TF sg = det < 0 ? - g : g;
            for ( int r = 0; r < D; ++r ) {
                TF row_sum = 0;
                for ( int c = 0; c < D; ++c ) {
                    const TF minor = M.without_row_and_col( r, c ).determinant();
                    const TF cof = ( ( r + c ) % 2 ? - minor : minor ) * sg;
                    grad_vp( chain[ c + 1 ], r ) += cof;
                    row_sum += cof;
                }
                grad_vp( chain[ 0 ], r ) -= row_sum;
            }
        } );
    }

    HD void bbox( TK *lo, TK *hi ) const {
        for ( int d = 0; d < D; ++d )
            lo[ d ] = hi[ d ] = nv ? v[ d ][ 0 ] : TK( 0 );
        for ( int i = 1; i < nv; ++i )
            for ( int d = 0; d < D; ++d ) {
                lo[ d ] = v[ d ][ i ] < lo[ d ] ? v[ d ][ i ] : lo[ d ];
                hi[ d ] = v[ d ][ i ] > hi[ d ] ? v[ d ][ i ] : hi[ d ];
            }
    }

    // ---- the tensors ------------------------------------------------------------------------

    /// from a `Cell_N` view: `vertex_positions [ nv, D ]`, `vertex_cuts` / `vertex_nbrs`
    /// `[ nv, D ]`, `cut_ids [ nc ]`. `false` if it does not fit in `cap`.
    HD bool load( const auto &c ) {
        const int n = int( SI( c.nb_vertices ) ), k = int( SI( c.nb_cuts ) );
        if ( n > cap || k > cap )
            return false;
        nv = n; nc = k;
        for ( int i = 0; i < nv; ++i )
            for ( int d = 0; d < D; ++d ) {
                v [ d ][ i ] = TK( c.vertex_positions( i, d ) );
                vk[ d ][ i ] = int( SI( c.vertex_cuts( i, d ) ) );
                vn[ d ][ i ] = int( SI( c.vertex_nbrs( i, d ) ) );
            }
        for ( int q = 0; q < nc; ++q )
            cid[ q ] = int( SI( c.cut_ids( q ) ) );
        // over the vertices and not over the list: a DEAD cut ( with no vertex ) may linger there
        // until the next compaction, and a dead wall bounds nothing
        unbounded = false;
        for ( int i = 0; i < nv; ++i )
            for ( int r = 0; r < D; ++r )
                unbounded |= cid[ vk[ r ][ i ] ] == cell_ids::INFINITE;
        has_planes = false;
        own_set = false; pend_set = false;
        return true;
    }

    HD bool store( auto &&c ) const {
        if ( ! c.nb_vertices.set( nv ) )
            return false;
        if ( ! c.nb_cuts.set( nc ) )
            return false;
        for ( int i = 0; i < nv; ++i )
            for ( int d = 0; d < D; ++d ) {
                c.vertex_positions( i, d ) = v [ d ][ i ];
                c.vertex_cuts     ( i, d ) = vk[ d ][ i ];
                c.vertex_nbrs     ( i, d ) = vn[ d ][ i ];
            }
        for ( int k = 0; k < nc; ++k )
            c.cut_ids( k ) = cid[ k ];
        // CANON: the edges of zero length that the ties make, a bit per neighbor ( `canon_zero_edge` )
        if constexpr ( requires { c.vertex_ties; } ) {
            for ( int i = 0; i < nv; ++i ) {
                int mask = 0;
                if constexpr ( CANON )
                    if ( own_set )
                        for ( int r = 0; r < D; ++r )
                            if ( canon_zero_edge( i, r ) )
                                mask |= 1 << r;
                c.vertex_ties( i ) = mask;
            }
        }
        return true;
    }
};

} // namespace sdot
