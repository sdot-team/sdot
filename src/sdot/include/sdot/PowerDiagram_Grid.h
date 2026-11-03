#pragma once

// the axes this body names, as declared symbols (autocompletion, standalone compile) instead of
// globals the generated source happens to define around us. Written to the build include tree.
#include <sdot/generated/aggregates/PowerDiagram_Grid.h>
#include <loom/support/common_macros.h>
#include "diagram/Common.h"
#include "bsp_build_level.h"

namespace sdot {

// THE SEEDS AT THE CENTRES OF THE CELLS OF A REGULAR GRID ( `PowerDiagram_Grid.py`, `SumOfDiracsOnAGrid` ): no positions are
// stored. Seed `k` ( C order of `grid_shape`, the first axis the slowest ) is at `grid_origin + grid_frame^T ( i( k ) + 1/2 )`,
// the rows of `grid_frame` being the edges of one cell of the grid. The rank of a seed is its user index.
//
// THE NEIGHBOURS ( `ProviderGrid` ): the ring of the `3^d - 1` adjacent seeds first ( the whole cell, almost always ), then an
// IMPLICIT quadtree ( octree ) over the grid, walked like the BSP ( `ProviderBsp` ), which prunes what the cell rules out with
// the same exact test ( `cell/Pruning.h` ) -- the guarantee that no far seed was forgotten. Its nodes are blocks of
// `grid_leaf << l` cells along each axis ( level `l`, `0` for the leaves ), laid out level after level, each level in C order
// of its blocks; their boxes are computed, their affine majorants of the weights stored ( `node_wa`, `node_wb`: absent without
// weights ), refreshed by `node_majorant` when the weights change.

/// the layout of the implicit tree: number of blocks per axis at each level, offsets of the levels in the node arrays
template<int D>
struct GridLevels {
    static constexpr int leaf = D == 2 ? 4 : 2;        ///< cells per axis in a leaf block
    static constexpr int max_levels = 40;

    SI  shape[ D ];
    SI  count[ max_levels ][ D ];                        ///< blocks per axis at level `l`
    SI  offset[ max_levels + 1 ];                        ///< first node of level `l`
    int nb_levels = 0;                                   ///< the root is the only node of level `nb_levels - 1`

    HD GridLevels( const SI *sh ) {
        for ( int d = 0; d < D; ++d )
            shape[ d ] = sh[ d ];
        offset[ 0 ] = 0;
        for ( int l = 0; l < max_levels; ++l ) {
            SI nb = 1;
            bool root = true;
            for ( int d = 0; d < D; ++d ) {
                const SI side = SI( leaf ) << l;
                count[ l ][ d ] = ( shape[ d ] + side - 1 ) / side;
                nb *= count[ l ][ d ];
                root = root && count[ l ][ d ] == 1;
            }
            offset[ l + 1 ] = offset[ l ] + nb;
            nb_levels = l + 1;
            if ( root )
                break;
        }
    }

    HD SI nb_nodes() const { return offset[ nb_levels ]; }

    HD SI node( int l, const SI *b ) const {
        SI r = 0;
        for ( int d = 0; d < D; ++d )
            r = r * count[ l ][ d ] + b[ d ];
        return offset[ l ] + r;
    }

    /// the cells of block `b` of level `l`: `[ lo, hi )` per axis
    HD void cells( int l, const SI *b, SI *lo, SI *hi ) const {
        const SI side = SI( leaf ) << l;
        for ( int d = 0; d < D; ++d ) {
            lo[ d ] = b[ d ] * side;
            hi[ d ] = lo[ d ] + side < shape[ d ] ? lo[ d ] + side : shape[ d ];
        }
    }

    /// node -> level and block ( the inverse of `node` )
    HD void block_of( SI n, int &l, SI *b ) const {
        l = 0;
        while ( offset[ l + 1 ] <= n )
            ++l;
        SI r = n - offset[ l ];
        for ( int d = D - 1; d >= 0; --d ) {
            b[ d ] = r % count[ l ][ d ];
            r /= count[ l ][ d ];
        }
    }
};

/// THE NEIGHBOURS OF A GRID SEED: the ring, then the implicit tree, pruned with the cell as it has become
template<class PD,class TK,int D,bool WEIGHTED>
struct ProviderGrid {
    using TF = typename PD::TF;
    static constexpr int nb_ring = D == 2 ? 8 : 26;
    static constexpr int nb_child = 1 << D;

    struct Entry { int l; SI b[ D ]; };

    /// THE STATE OF THE WALK, one per cell: the ring's cursor, the stack of blocks, the open leaf
    struct Local {
        int   phase = 0;                                 ///< 0: the ring, 1: the tree
        int   ring = 0;
        int   top = 0;
        Entry stack[ ( GridLevels<D>::max_levels + 1 ) * nb_child ];
        bool  open = false;                              ///< a leaf is open: `cur` walks `[ lo, hi )`
        SI    lo[ D ], hi[ D ], cur[ D ];
    };

    const PD     &pd;
    GridLevels<D> lv;
    SI            k0, i0[ D ];
    TF            p0[ D ], w0;
    TK            q0[ D ], v0;

    HD ProviderGrid( const PD &pd, SI k0 ) : pd( pd ), lv( pd.shape_array().data ), k0( k0 ) {
        SI r = k0;
        for ( int d = D - 1; d >= 0; --d ) {
            i0[ d ] = r % lv.shape[ d ];
            r /= lv.shape[ d ];
        }
        const auto p = pd.point( k0 );
        for ( int d = 0; d < D; ++d ) {
            p0[ d ] = p[ d ];
            q0[ d ] = TK( p[ d ] );
        }
        w0 = pd.weight( k0 );
        v0 = TK( w0 );
    }

    HD SI rank( const SI *i ) const {
        SI r = 0;
        for ( int d = 0; d < D; ++d )
            r = r * lv.shape[ d ] + i[ d ];
        return r;
    }

    /// within the ring ( or the seed itself ): already proposed
    HD bool in_ring( const SI *i ) const {
        for ( int d = 0; d < D; ++d )
            if ( i[ d ] < i0[ d ] - 1 || i[ d ] > i0[ d ] + 1 )
                return false;
        return true;
    }

    HD void plane( const SI *i, Plane<TK,D> &p ) const {
        p = seed_plane<TK,D>( pd, k0, p0, w0, rank( i ) );
    }

    /// the box of the CENTRES of the cells `[ lo, hi )`, physical coordinates ( a parallelogram, boxed )
    HD void centre_box( const SI *lo, const SI *hi, TF *blo, TF *bhi ) const {
        for ( int e = 0; e < D; ++e ) {
            TF base = TF( pd.grid_origin( e ) ), mi = 0, ma = 0;
            for ( int d = 0; d < D; ++d ) {
                const TF f = TF( pd.grid_frame( d, e ) );
                base += ( TF( lo[ d ] ) + TF( 0.5 ) ) * f;
                const TF s = TF( hi[ d ] - 1 - lo[ d ] ) * f;
                mi += s < 0 ? s : TF( 0 );
                ma += s > 0 ? s : TF( 0 );
            }
            blo[ e ] = base + mi;
            bhi[ e ] = base + ma;
        }
    }

    HD Box<TK,D> box( int l, const SI *b ) const {
        SI lo[ D ], hi[ D ];
        lv.cells( l, b, lo, hi );
        TF blo[ D ], bhi[ D ];
        centre_box( lo, hi, blo, bhi );
        Box<TK,D> B;
        for ( int d = 0; d < D; ++d ) {
            B.lo[ d ] = TK( blo[ d ] );
            B.hi[ d ] = TK( bhi[ d ] );
            B.a[ d ] = 0;
        }
        B.b = 0;
        if constexpr ( WEIGHTED ) {
            const SI n = lv.node( l, b );
            for ( int d = 0; d < D; ++d )
                B.a[ d ] = TK( pd.node_wa( n, d ) );
            B.b = TK( pd.node_wb( n ) );
        }
        return B;
    }

    /// the squared distance from the seed to the box of a block -- an ordering key only
    HD TF proximity( int l, const SI *b ) const {
        SI lo[ D ], hi[ D ];
        lv.cells( l, b, lo, hi );
        TF blo[ D ], bhi[ D ];
        centre_box( lo, hi, blo, bhi );
        TF res = 0;
        for ( int d = 0; d < D; ++d ) {
            TF e = blo[ d ] - p0[ d ];
            const TF f = p0[ d ] - bhi[ d ];
            e = f > e ? f : e;
            e = e > 0 ? e : TF( 0 );
            res += e * e;
        }
        return res;
    }

    /// a block whose cells are all in the ring has nothing left to propose
    HD bool within_ring( int l, const SI *b ) const {
        SI lo[ D ], hi[ D ];
        lv.cells( l, b, lo, hi );
        for ( int d = 0; d < D; ++d )
            if ( lo[ d ] < i0[ d ] - 1 || hi[ d ] - 1 > i0[ d ] + 1 )
                return false;
        return true;
    }

    template<class State>
    HD bool next( const State &e, Local &l, Plane<TK,D> &p ) {
        // ---- the ring, axis neighbours first ( the offsets in order of their number of non-zero components )
        if ( l.phase == 0 ) {
            while ( l.ring < nb_ring ) {
                const int q = l.ring++;
                SI i[ D ];
                bool ok = true;
                ring_offset( q, i );
                for ( int d = 0; d < D; ++d ) {
                    i[ d ] += i0[ d ];
                    ok = ok && i[ d ] >= 0 && i[ d ] < lv.shape[ d ];
                }
                if ( ok ) {
                    plane( i, p );
                    return true;
                }
            }
            l.phase = 1;
            Entry &r = l.stack[ l.top++ ];
            r.l = lv.nb_levels - 1;
            for ( int d = 0; d < D; ++d )
                r.b[ d ] = 0;
        }

        // ---- the tree
        for ( ;; ) {
            while ( l.open ) {                           // the next cell of the open leaf ( C order )
                SI i[ D ];
                for ( int d = 0; d < D; ++d )
                    i[ d ] = l.cur[ d ];
                int d = D - 1;
                while ( d >= 0 && ++l.cur[ d ] == l.hi[ d ] ) {
                    l.cur[ d ] = l.lo[ d ];
                    --d;
                }
                if ( d < 0 )
                    l.open = false;
                if ( in_ring( i ) )
                    continue;
                plane( i, p );
                return true;
            }

            if ( l.top == 0 )
                return false;
            const Entry n = l.stack[ --l.top ];
            if ( within_ring( n.l, n.b ) || ! can_cut<WEIGHTED,TK,D>( e, q0, v0, box( n.l, n.b ) ) )
                continue;

            if ( n.l == 0 ) {                            // a leaf: its cells, one after the other
                lv.cells( 0, n.b, l.lo, l.hi );
                for ( int d = 0; d < D; ++d )
                    l.cur[ d ] = l.lo[ d ];
                l.open = true;
                continue;
            }

            // the children, the closest to the seed pushed last ( popped first )
            Entry ch[ nb_child ];
            TF key[ nb_child ];
            int nc = 0;
            for ( int c = 0; c < nb_child; ++c ) {
                Entry x;
                x.l = n.l - 1;
                bool ok = true;
                for ( int d = 0; d < D; ++d ) {
                    x.b[ d ] = 2 * n.b[ d ] + ( ( c >> ( D - 1 - d ) ) & 1 );
                    ok = ok && x.b[ d ] < lv.count[ x.l ][ d ];
                }
                if ( ! ok )
                    continue;
                const TF k = proximity( x.l, x.b );
                int j = nc++;
                for ( ; j > 0 && key[ j - 1 ] < k; --j ) {
                    ch[ j ] = ch[ j - 1 ];
                    key[ j ] = key[ j - 1 ];
                }
                ch[ j ] = x;
                key[ j ] = k;
            }
            for ( int c = 0; c < nc; ++c )
                l.stack[ l.top++ ] = ch[ c ];
        }
    }

    /// the `q`-th offset of the ring: those with one non-zero component, then two, ( then three )
    HD static void ring_offset( int q, SI *o ) {
        int seen = 0;
        for ( int nz = 1; nz <= D; ++nz ) {
            for ( int c = 0; c < ( D == 2 ? 9 : 27 ); ++c ) {
                SI t[ D ];
                int z = 0, r = c;
                for ( int d = D - 1; d >= 0; --d ) {
                    t[ d ] = r % 3 - 1;
                    r /= 3;
                    z += t[ d ] != 0;
                }
                if ( z != nz )
                    continue;
                if ( seen++ == q ) {
                    for ( int d = 0; d < D; ++d )
                        o[ d ] = t[ d ];
                    return;
                }
            }
        }
    }
};

SDOT_TEMPLATE_DECL_FOR_PowerDiagram_Grid
struct PowerDiagram_Grid {
    SDOT_ATTRIBUTES_OF_PowerDiagram_Grid

    using TF = typename DECAYED_TYPE_OF( grid_origin )::TF;

    struct ShapeArray { SI data[ 3 ]; };
    HD ShapeArray shape_array() const {
        ShapeArray r{};
        for ( int d = 0; d < DECAYED_TYPE_OF( nb_dims )::value; ++d )
            r.data[ d ] = SI( grid_shape( d ) );
        return r;
    }

    HD SI nb_seeds() const {
        SI n = 1;
        for ( int d = 0; d < DECAYED_TYPE_OF( nb_dims )::value; ++d )
            n *= SI( grid_shape( d ) );
        return n;
    }

    /// the coordinate `d` of the centre of cell `j` ( C order )
    HD TF stored_point( SI j, int d ) const {
        constexpr int D = DECAYED_TYPE_OF( nb_dims )::value;
        SI i[ D ];
        for ( int a = D - 1; a >= 0; --a ) {
            const SI s = SI( grid_shape( a ) );
            i[ a ] = j % s;
            j /= s;
        }
        TF r = TF( grid_origin( d ) );
        for ( int a = 0; a < D; ++a )
            r += ( TF( i[ a ] ) + TF( 0.5 ) ) * TF( grid_frame( a, d ) );
        return r;
    }

    SDOT_DIAGRAM_COMMON_BODY( no_positions, weights )

    HD SI user_id( SI k ) const { return seed_of( k ); }

    template<class TK>
    HD auto provider( SI k0 ) const { return ProviderGrid<PowerDiagram_Grid,TK,ct_dim,has_weights>( *this, k0 ); }

    /// THE MAJORANT OF NODE `n` at the current weights: `w <= wa . y + wb` over its seeds ( `bsp_weight_majorant`, the BSP's rule )
    HD void node_majorant( SI n, auto &&wa_out, auto &&wb_out ) const {
        const GridLevels<ct_dim> lv( shape_array().data );
        int l;
        SI b[ ct_dim ], lo[ ct_dim ], hi[ ct_dim ];
        lv.block_of( n, l, b );
        lv.cells( l, b, lo, hi );
        SI ext[ ct_dim ], m = 1;
        for ( int d = 0; d < ct_dim; ++d ) {
            ext[ d ] = hi[ d ] - lo[ d ];
            m *= ext[ d ];
        }
        // the seeds of the block, numbered `0 .. m` in C order of the block
        auto rank = [&]( SI k ) {
            SI i[ ct_dim ];
            for ( int d = ct_dim - 1; d >= 0; --d ) {
                i[ d ] = lo[ d ] + k % ext[ d ];
                k /= ext[ d ];
            }
            SI r = 0;
            for ( int d = 0; d < ct_dim; ++d )
                r = r * lv.shape[ d ] + i[ d ];
            return r;
        };
        struct Pos {
            using TF = typename PowerDiagram_Grid::TF;
            const PowerDiagram_Grid &pd;
            const decltype( rank ) &rank_of;
            HD TF operator()( SI k, int d ) const { return pd.stored_point( rank_of( k ), d ); }
        } pos{ *this, rank };
        auto w = [&]( SI k ) { return weight( rank( k ) ); };
        bsp_weight_majorant<ct_dim>( pos, w, 0, m, wa_out, wb_out );
    }

    HD SI nb_nodes() const { return GridLevels<ct_dim>( shape_array().data ).nb_nodes(); }

    /// THE SAME DIAGRAM, the weights and the majorants read ELSEWHERE ( the `SdotPlanNd` solver, which writes them at each trial,
    /// `sdotplan/Sweep.h` ); the order of the members is that of `PowerDiagram.py` + `PowerDiagram_Grid.py`
    HD auto with_weights( auto &&weights_, auto &&node_wa_, auto &&node_wb_ ) const {
        return ::sdot::PowerDiagram_Grid{ box_min, box_max, bnd_directions, bnd_offsets, start_vertices, start_topo, periodic_shifts, periodic_linear, sink_weight, radial_eps, nb_starts, nb_topo, nb_images, nb_points, nb_boundaries, nb_dims, group_rep, group_next, group_parent, group_vweights, group_dirs, group_agg_parent, group_agg_center, group_agg_scale, group_agg_outer, group_agg_vweights, nb_aggs, grid_origin, grid_frame, grid_shape, weights_, node_wa_, node_wb_, nb_gnodes };
    }
};

}
