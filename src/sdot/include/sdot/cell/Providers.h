#pragma once

#include <loom/support/common_macros.h> // HD

// =====================================================================================
// THE PROVIDERS OF A POWER DIAGRAM -- "which half-space to cut now?"
//
// The engine ( `Engine.h` ) knows neither seed nor tree: it draws planes. A provider is an
// object with one method, `next( state, local, plan )`, that fills the plane and returns `true`, or
// returns `false` when it has nothing left. Two here:
//
//   `ProviderAll`   all the other seeds, in order. No acceleration -- this is what
//                       `make_cell` does when it is not given an accelerator, and it is the
//                       FLOOR against which the others are measured.
//   `ProviderBsp`    the BSP tree TURNED INSIDE OUT: a descent that would PUSH its candidates becomes
//                       a SUSPENDED traversal whose explicit stack lives in the `Local` that the
//                       engine houses per cell. Each `next` resumes where the previous one
//                       stopped: preorder, the child closest to the seed first, and a
//                       pruning that sees the cell AS IT HAS BECOME, cut after cut.
//
// THE BISECTOR is computed in the positions' float and returned in the kernel's
// ( `bisector`, `Plane.h` ): this is the only place where the two precisions meet.
// =====================================================================================

#include <loom/support/common_types.h>
#include "Pruning.h"
#include "Plane.h"
#include "State.h"
#include "Canonical.h"

namespace sdot {

/// PERIODICITY ( `diagram/Common.h` ): a storage that carries images says so with `periodic`; the wrappers that do not
/// ( `sdotplan/Bounds.h` ) are not periodic.
template<class PD>
HD constexpr bool is_periodic() {
    if constexpr ( requires { PD::periodic; } ) return PD::periodic;
    else return false;
}

/// GROUPS ( `diagram/Common.h` ): a storage that carries them says so with `has_groups`; the engine's seeds and the solver's
/// wrappers do not
template<class PD>
HD constexpr bool has_groups() {
    if constexpr ( requires { PD::has_groups; } ) return PD::has_groups;
    else return false;
}

/// THE PLANE BETWEEN `k0` AND `j`: the power bisector ( `p0`, `w0`: the point and the weight of `k0` ), or, for two members of
/// the same group, the plane the group gives them ( `group_plane` ) -- rounded once to the kernel's float, like `bisector`
///
/// INLINED BY FORCE: left out of line, gcc called it once per proposed neighbour from the 2D register kernel, 6 % of the
/// diagram; inlined, the 2D cells went 1 340 -> 1 190 ns per seed on one thread ( lmo, AVX-512, 2026-10-26 ).
template<class TK,int D,class PD>
#ifdef __CUDACC__
__forceinline__
#else
[[gnu::always_inline]] inline
#endif
HD Plane<TK,D> seed_plane( const PD &pd, SI k0, const typename PD::TF *p0, typename PD::TF w0, SI j ) {
    using TF = typename PD::TF;
    if constexpr ( has_groups<PD>() ) {
        if ( pd.same_group( k0, j ) ) {
            TF dir[ D ], off;
            pd.group_plane( k0, j, dir, off );
            Plane<TK,D> res;
            for ( int d = 0; d < D; ++d )
                res.dir[ d ] = TK( dir[ d ] );
            res.off = TK( off );
            res.id  = int( j );
            return res;
        }
    }
    const auto pj = pd.point( j );
    TF p1[ D ];
    for ( int d = 0; d < D; ++d )
        p1[ d ] = pj[ d ];
    return bisector<TK,D>( p0, w0, p1, pd.weight( j ), int( j ) );
}

/// ALL THE OTHER SEEDS, in storage order -- then, for a periodic storage, all their images, image after image.
template<class PD,class TK,int D>
struct ProviderAll {
    using TF = typename PD::TF;

    const PD &pd;
    SI  n, k0, k = 0, t = 0;                             ///< `t`: the image being swept
    TF  p0[ D ], w0;

    HD ProviderAll( const PD &pd, SI k0 ) : pd( pd ), n( pd.nb_seeds() ), k0( k0 ) {
        const auto p = pd.point( k0 );
        for ( int d = 0; d < D; ++d )
            p0[ d ] = p[ d ];
        w0 = pd.weight( k0 );
    }

    template<class State>
    HD bool next( const State &, NothingLocal &, Plane<TK,D> &p ) {
        for ( ;; ) {
            if ( k == k0 && t == 0 ) ++k;                // we do not cut ourselves ( but our images do )
            if constexpr ( has_groups<PD>() )            // ( a site of duplicates: one plane, that of its first member )
                if ( t == 0 && k < n && pd.hidden_in_site( k0, k ) ) { ++k; continue; }
            if ( k < n ) break;
            if constexpr ( is_periodic<PD>() ) {
                if ( ++t >= pd.image_count() ) return false;
                k = 0;
            } else
                return false;
        }
        const SI j = k++ + n * t;
        p = seed_plane<TK,D>( pd, k0, p0, w0, j );
        return true;
    }
};

/// THE BSP TREE, TRAVERSED ON DEMAND. `pd.tree` is the tree ( `AaBsp.py` ), and the storage's seeds
/// are IN ITS ORDER: the leaf `[ begin, end )` is read in one piece through
/// `pd.point( k )`, and the identifier of a cut is that rank `k`. `WEIGHTED`: Laguerre diagram
/// ( the affine majorant of the weights enters the pruning ) -- a compile-time constant, so that
/// the Euclidean case pays for neither the slopes nor the `a . y` terms.
///
/// `MEMO`: THE MEMORY ( `PowerDiagram_Bsp.memo_nbrs / memo_counts` ). The storage remembers,
/// per seed, the RANKS of the neighbors of its cell at the last `measures` ( sorted increasing ); they
/// are proposed FIRST, before any descent, then the ordinary traversal SKIPS them ( no
/// plane twice: `cut` is not idempotent ). What this spares is not the boxes --
/// a box that contains a true neighbor passes the pruning whatever happens -- it is the
/// TRANSIENT cuts, those that a nearby seed makes before a true neighbor supplants it: half
/// of the effective cuts in 3D, each one a polytope update. Measured on the bench
/// ( `solvers_des_familles`, README § 11 ): -25 to -42 % of the 3D diagram, and stale
/// memories -- those of Voronoi on a Laguerre diagram -- still yield -18 %: a false
/// memory only costs one first pass. A seed without memory ( `memo_counts == 0` ) takes
/// the ordinary path, up to the comparison.
///
/// `SLACK`: the boxes widened by `slack` ( the canonical cell: a plane through a vertex is never pruned ). A compile-time
/// choice: a margin read at run time, even zero, kept the compiler from folding the pruning test -- 5 % of a 2D diagram.
template<class PD,class TK,int D,bool WEIGHTED,bool MEMO = false,bool SLACK = false>
struct ProviderBsp {
    using TF = typename PD::TF;

    /// THE STATE OF THE TRAVERSAL, one per cell, housed in the engine's frame. The stack is bounded by the
    /// DEPTH of the tree, not by its size: at each level one node is popped and two are
    /// pushed. 64 levels are worth 2^64 seeds.
    struct Local {
        SI   stack[ 64 ];
        int  top = 0;
        SI   k = 0, stop = 0;                             ///< the slice of the open leaf
        bool started = false;
        SI   img = 0;                                    ///< PERIODIC: the image being traversed ( the tree is traversed once per image )
        int  ipre = 0;                                   ///< MEMO: where the memories pre-pass stands
        int  iskip = 0;                                  ///< MEMO: the skip cursor in the open leaf
        SI   gcur = -1;                                  ///< GROUPS: the last member proposed ( `-1`: none yet )
        bool gdone = false;                              ///< GROUPS: the members' pre-pass is over
    };

    const PD &pd;
    SI  k0;
    SI  n;                                               ///< the number of seeds: image `t` of rank `k` is `k + n t`
    SI  depth;                                           ///< `nb_nodes == 2^depth - 1`
    int npre = 0;                                        ///< MEMO: how many memories for `k0`
    TK  slack = 0;                                       ///< `SLACK`: the boxes widened by this much ( the canonical cut: a plane through a vertex is never pruned )
    TF  p0[ D ], w0;
    TK  q0[ D ], v0;                                     ///< the same, for the pruning

    HD ProviderBsp( const PD &pd, SI k0 ) : pd( pd ), k0( k0 ), n( pd.nb_seeds() ) {
        const auto p = pd.point( k0 );
        for ( int d = 0; d < D; ++d ) {
            p0[ d ] = p[ d ];
            q0[ d ] = TK( p[ d ] );
        }
        w0 = pd.weight( k0 );
        v0 = TK( w0 );

        depth = 0;
        for ( SI m = SI( pd.tree.node_begin.shape( 0 ) ); m; m >>= 1 )
            ++depth;

        if constexpr ( MEMO )
            npre = int( pd.memo_counts( k0 ) );
    }

    /// MEMO: the `q`-th memory of `k0`, a rank
    HD SI remembered( int q ) const {
        if constexpr ( MEMO ) return SI( pd.memo_nbrs( k0, q ) );
        else return 0;
    }

    HD void plane_of_rank( SI k, Plane<TK,D> &p ) const {
        p = seed_plane<TK,D>( pd, k0, p0, w0, k );
    }

    /// GROUPS: a member of the group of `k0`, which the pre-pass proposes ( the tree's pruning reads the outer planes only )
    HD bool member( SI k ) const {
        if constexpr ( has_groups<PD>() ) return pd.same_group( k0, k );
        else return false;
    }

    /// GROUPS: a member of a site of duplicates past its first, whose plane is the first's ( `hidden_in_site` ): not proposed
    HD bool hidden( SI k ) const {
        if constexpr ( has_groups<PD>() ) return k < n && pd.hidden_in_site( k0, k );
        else return false;
    }

    /// PERIODIC: image `t` maps `x` to `L x + s` ( `L` the identity for a translation, and for the seeds themselves )
    HD TF shift( SI t, int d ) const {
        if constexpr ( is_periodic<PD>() ) return pd.image_shift( t, d );
        else return TF( 0 );
    }
    HD TF linear( SI t, int r, int c ) const {
        if constexpr ( is_periodic<PD>() ) return pd.image_linear( t, r, c );
        else return TF( r == c );
    }
    static constexpr bool with_linear_part = [] { if constexpr ( requires { PD::periodic_linear_part; } ) return bool( PD::periodic_linear_part ); else return false; }();

    /// the box of node `n` for image `t`: the box of the image of the node's box ( translated, or rotated then boxed again )
    HD void image_box( SI n, SI t, TF *lo, TF *hi ) const {
        if constexpr ( with_linear_part ) {
            if ( t ) {
                for ( int r = 0; r < D; ++r ) {
                    TF c = shift( t, r ), h = 0;
                    for ( int q = 0; q < D; ++q ) {
                        const TF a = TF( pd.tree.node_box( n, 0, q ) ), b = TF( pd.tree.node_box( n, 1, q ) ), m = linear( t, r, q );
                        c += m * ( a + b ) / 2;
                        h += ( m < 0 ? -m : m ) * ( b - a ) / 2;
                    }
                    lo[ r ] = c - h;
                    hi[ r ] = c + h;
                }
                return;
            }
        }
        for ( int d = 0; d < D; ++d ) {
            const TF s = shift( t, d );
            lo[ d ] = TF( pd.tree.node_box( n, 0, d ) ) + s;
            hi[ d ] = TF( pd.tree.node_box( n, 1, d ) ) + s;
        }
    }

    /// the node's box, and the majorant of its weights, in the kernel's float -- for image `t`, the box of the image, and the
    /// majorant `w( q ) <= a . q + b` rewritten for `q' = L q + s`: `( L a ) . q' + b - ( L a ) . s`
    HD Box<TK,D> box( SI n, SI t ) const {
        Box<TK,D> B;
        TF lo[ D ], hi[ D ];
        image_box( n, t, lo, hi );
        for ( int d = 0; d < D; ++d ) {
            B.lo[ d ] = TK( lo[ d ] );
            B.hi[ d ] = TK( hi[ d ] );
            if constexpr ( SLACK ) {
                B.lo[ d ] -= slack;
                B.hi[ d ] += slack;
            }
            B.a[ d ] = 0;
        }
        B.b = 0;
        if constexpr ( WEIGHTED ) {
            TF a[ D ], b = TF( pd.tree.node_wb( n ) );
            for ( int d = 0; d < D; ++d )
                a[ d ] = TF( pd.tree.node_wa( n, d ) );
            for ( int r = 0; r < D; ++r ) {
                TF ar = 0;
                if constexpr ( with_linear_part ) { for ( int q = 0; q < D; ++q ) ar += linear( t, r, q ) * a[ q ]; }
                else ar = a[ r ];
                B.a[ r ] = TK( ar );
                b -= ar * shift( t, r );
            }
            B.b = TK( b );
        }
        return B;
    }

    /// the squared distance from the seed to the box of node `n` ( of image `t` ) -- an ORDERING KEY only, to
    /// visit the closest child first, hence the cuts that bite most first.
    HD TF proximity( SI n, SI t ) const {
        TF lo[ D ], hi[ D ];
        image_box( n, t, lo, hi );
        TF res = 0;
        for ( int d = 0; d < D; ++d ) {
            // branch-free ( it is the unpredictable one of the descent ): the distance to `[ lo, hi ]`
            TF e = lo[ d ] - p0[ d ];
            const TF f = p0[ d ] - hi[ d ];
            e = f > e ? f : e;
            e = e > 0 ? e : TF( 0 );
            res += e * e;
        }
        return res;
    }

    template<class State>
    HD bool next( const State &e, Local &l, Plane<TK,D> &p ) {
        if ( ! l.started ) {                              // the root: index 0, height `depth`
            l.stack[ l.top++ ] = depth;
            l.started = true;
        }

        if constexpr ( has_groups<PD>() ) {             // the members of the group first, whatever the tree would say of them
            if ( ! l.gdone ) {
                const SI j = pd.next_member( l.gcur < 0 ? k0 : l.gcur );
                if ( j != k0 ) {
                    l.gcur = j;
                    plane_of_rank( j, p );
                    return true;
                }
                l.gdone = true;
            }
        }

        if constexpr ( MEMO ) {                          // the pre-pass: yesterday's neighbors, without descending
            while ( l.ipre < npre ) {
                const SI r = remembered( l.ipre++ );
                if ( member( r ) || hidden( r ) )        // ( already proposed, or seen through the first of its site )
                    continue;
                plane_of_rank( r, p );
                return true;
            }
        }

        for ( ;; ) {
            // ---- a leaf is open: we return the next seed of its slice
            while ( l.k < l.stop ) {
                const SI k = l.k++;
                if ( k == k0 || member( k ) || hidden( k ) )
                    continue;
                if constexpr ( MEMO ) {                  // already proposed? the cursor advances with `k`
                    while ( l.iskip < npre && remembered( l.iskip ) < k ) ++l.iskip;
                    if ( l.iskip < npre && remembered( l.iskip ) == k ) { ++l.iskip; continue; }
                }
                plane_of_rank( k, p );
                return true;
            }

            if ( l.top == 0 ) {                          // the tree is exhausted ...
                if constexpr ( is_periodic<PD>() ) {     // ... for this image: the next one
                    if ( ++l.img < pd.image_count() ) {
                        l.stack[ l.top++ ] = depth;
                        continue;
                    }
                }
                return false;
            }

            // `h` travels on the stack, packed with the index ( six bits suffice )
            const SI en = l.stack[ --l.top ];
            const SI n  = en >> 6;
            const SI h  = en & 63;

            // an EMPTY slot: the right child of a node that had nothing left to share
            const SI beg = SI( pd.tree.node_begin( n ) );
            const SI end = SI( pd.tree.node_end( n ) );
            if ( beg >= end )
                continue;

            // the whole point of the tree: a subtree that can no longer reach the cell is
            // not descended. Tested at POP time, hence against the cell as it is now.
            if ( ! can_cut<WEIGHTED,TK,D>( e, q0, v0, box( n, l.img ) ) )
                continue;

            if ( h <= 1 ) {                              // a leaf: its ranks, as identifiers of the current image
                l.k = beg + this->n * l.img;
                l.stop = end + this->n * l.img;
                if constexpr ( MEMO ) {                  // the skip cursor: the first memory >= the first identifier
                    int lo = 0, hi = npre;
                    while ( lo < hi ) { const int m = ( lo + hi ) / 2; if ( remembered( m ) < l.k ) lo = m + 1; else hi = m; }
                    l.iskip = lo;
                }
                continue;
            }

            // the children are DEDUCED: the tree is perfect binary in preorder, the left one is right
            // next to it and the right one is `2^( h - 1 )` nodes away. The closest is pushed LAST.
            const SI lc = n + 1, rc = n + ( SI( 1 ) << ( h - 1 ) ), hc = h - 1;
            if ( proximity( lc, l.img ) <= proximity( rc, l.img ) ) {
                l.stack[ l.top++ ] = ( rc << 6 ) | hc;
                l.stack[ l.top++ ] = ( lc  << 6 ) | hc;
            } else {
                l.stack[ l.top++ ] = ( lc  << 6 ) | hc;
                l.stack[ l.top++ ] = ( rc << 6 ) | hc;
            }
        }
    }
};

/// THE GENERATOR of the plane of identifier `j` in the cell of `k0` ( `cell/Canonical.h` ): seed `j` itself for `j == k0`. Its rank,
/// its position without the image and the shift of its image ( with a linear part, its position as it is ); a plane inside a group
/// is not a bisector, and has none.
template<int D,class PD>
HD canon::Gen<D> canon_generator( const PD &pd, SI k0, SI j ) {
    canon::Gen<D> g;
    if constexpr ( has_groups<PD>() )
        if ( pd.same_group( k0, j ) )
            return g;
    const SI n = pd.nb_seeds();
    SI seed = j, t = 0;
    if constexpr ( is_periodic<PD>() ) { seed = j % n; t = j / n; }
    g.kind = canon::SEED;
    g.rank = int( seed );
    g.w = double( pd.weight( j ) );
    constexpr bool linear = [] { if constexpr ( requires { PD::periodic_linear_part; } ) return bool( PD::periodic_linear_part ); else return false; }();
    if ( linear && t ) {
        const auto q = pd.point( j );
        for ( int d = 0; d < D; ++d ) g.p[ d ] = double( q[ d ] );
    } else {
        const auto q = pd.point( seed );
        for ( int d = 0; d < D; ++d ) {
            g.p[ d ] = double( q[ d ] );
            if constexpr ( is_periodic<PD>() ) g.s[ d ] = double( pd.image_shift( t, d ) );
        }
    }
    return g;
}

/// THE PROVIDER OF A CANONICAL CELL: the planes of `f`, each with its generator told to the cell before it cuts
template<class PD,class Inner,class Cell>
struct CanonProvider {
    using Local = LocalOf<Inner>;
    static constexpr int D = Cell::ct_dim;

    const PD &pd;
    Inner    &f;
    Cell     &c;
    SI        k0;

    template<class State>
    HD bool next( const State &e, Local &l, typename Cell::PlaneT &p ) {
        if ( ! f.next( e, l, p ) )
            return false;
        c.canon_pending( canon_generator<D>( pd, k0, SI( p.id ) ) );
        return true;
    }
};

} // namespace sdot
