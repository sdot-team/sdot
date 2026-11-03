#pragma once

#include "bsp_build_level.h"
#include <type_traits>
#include <algorithm>
#include <limits>
#include <vector>

namespace sdot {

// THE WHOLE BSP CONSTRUCTION IN ONE CALL, on the CPU ( host code: it takes the `queue` ). The same tree as the per-level
// calls of `bsp_build_level.h`, bit for bit: each node runs the same body on the same slice, and only the barrier between
// two levels changes -- the return of `queue.run_threads` instead of the end of a launch. What it saves is the cost of a
// call per level ( about ten milliseconds each through the driver, eighteen levels at 1e6 seeds, against 0.13 s of work ).
//
// The cloud is still double-buffered: level 0 reads `src` and writes `c0`, then the levels alternate between `c0` and `c1`
// ( the last one written is `c[ ( depth - 1 ) % 2 ]`: the seeds in the tree's order ). The nodes come out in PREORDER
// ( `AaBsp._preorder_of_heap` says why ), with their children ( the left one next, the right one past the left subtree; -1
// under a leaf ) and the count of the non-empty leaves: nothing is left to arrange on the host.
//
// The nodes of a level are dealt to the workers in contiguous slices: they hold the same number of seeds within one, so
// the slices are balanced. At the TOP, where a level has fewer nodes than workers, each node gets a GROUP of workers
// instead ( `bsp_wide_level` ): one node per worker made the first levels the sequential part of the construction ( 2.7x
// on 128 cores ). Its halves are a STABLE partition around the exact median, where the per-node body's selection leaves
// them in the order of its sweeps: the same nodes, the same boxes, the same leaves as sets -- the order of the seeds inside
// a leaf differs, which no property of the tree depends on ( the card's host levels, `AaBsp._build_in_kernel`, differ too ).

/// `( element ( k, ... ) )` of a node-batched output, seen as the per-node tensor `bsp_build_level` writes
template<class T>
struct BspNodeView {
    T &t;
    SI k;
    static constexpr bool is_valid = std::decay_t<T>::is_valid;
    decltype( auto ) operator()( auto ...i ) const { return t( k, i... ); }
};

/// ONE LEVEL WITH FEWER NODES THAN WORKERS: node `j` gets the workers `[ j G, ( j + 1 ) G )`, each one a contiguous part of
/// its slice. A cut is the exact median by rank along the longest axis ( as `bsp_build_level` ), found without a sequential
/// sweep: pivots bracketing the rank from a sample, a count, the keys between them gathered and selected, then a STABLE
/// partition written at offsets known from per-worker counts -- `mid - b` seeds go left, those under the median and the first
/// ones equal to it. The result does not depend on the number of workers. The weights' majorant, when there are weights, is
/// still one worker's sweep of the node.
void bsp_wide_level( const auto &queue, const auto &in, auto &&out, auto &&heap, const std::vector<SI> &pre, const std::vector<SI> &beg,
                     const std::vector<SI> &end, std::vector<SI> &mid, SI G, SI leaf_size ) {
    constexpr int D = CT_VALUE( in.nb_dims );
    using TF = typename DECAYED_TYPE_OF( in.positions )::TF;
    constexpr bool weighted = DECAYED_TYPE_OF( in.weights )::is_valid;
    const SI nb = SI( beg.size() ), T = nb * G;
    auto part = [&]( SI t, SI &j, SI &k0, SI &k1 ) {
        j = t / G;
        const SI s = t % G, b = beg[ j ], m = std::max<SI>( end[ j ] - b, 0 );
        k0 = b + m * s / G;
        k1 = b + m * ( s + 1 ) / G;
    };
    auto run = [&]( auto &&f ) { queue.run_threads( int( T ), [&]( int t ) { SI j, k0, k1; part( t, j, k0, k1 ); f( SI( t ), j, k0, k1 ); } ); };

    // ---- the boxes ( and the majorant: the first worker of the group, over the whole node )
    std::vector<Vector<TF,D>> lo( T, Vector<TF,D>::zeros() ), hi( T, Vector<TF,D>::zeros() );
    std::vector<char> seen( T, 0 );
    run( [&]( SI t, SI j, SI k0, SI k1 ) {
        if ( k1 > k0 ) {                              // ( local accumulators: the workers' slots share cache lines )
            auto l = Vector<TF,D>::with_func( [&]( PI d ) { return TF( in.positions( k0, d ) ); } ), u = l;
            for ( SI k = k0 + 1; k < k1; ++k )
                for ( int d = 0; d < D; ++d ) {
                    const TF v = TF( in.positions( k, d ) );
                    l[ d ] = v < l[ d ] ? v : l[ d ];
                    u[ d ] = v > u[ d ] ? v : u[ d ];
                }
            lo[ t ] = l;
            hi[ t ] = u;
        }
        seen[ t ] = k1 > k0;
        if constexpr ( weighted ) {
            const SI h = pre[ j ];
            if ( t % G == 0 && end[ j ] > beg[ j ] )
                bsp_weight_majorant<D>( in.positions, in.weights, beg[ j ], end[ j ], BspNodeView<DECAYED_TYPE_OF( heap.wa )>{ heap.wa, h }, heap.wb( h ) );
        }
    } );

    // per node: its box, its axis, whether it is cut; the sample and the bracket of the median
    std::vector<int> ax( nb, 0 );
    std::vector<char> cut( nb, 0 );
    std::vector<TF> plo( nb ), phi( nb );
    for ( SI j = 0; j < nb; ++j ) {
        const SI h = pre[ j ], b = beg[ j ], e = end[ j ];
        heap.begin( h ) = b;
        heap.end( h ) = e;
        mid[ j ] = e;
        if ( e <= b ) {                               // an empty slot ( see `bsp_build_level` )
            for ( int d = 0; d < D; ++d ) { heap.box( h, 0, d ) = 0; heap.box( h, 1, d ) = 0; }
            if constexpr ( weighted ) { for ( int d = 0; d < D; ++d ) heap.wa( h, d ) = 0; heap.wb( h ) = 0; }
            continue;
        }
        auto l = Vector<TF,D>::zeros(), u = l;
        bool first = true;
        for ( SI t = j * G; t < ( j + 1 ) * G; ++t ) {
            if ( ! seen[ t ] ) continue;
            for ( int d = 0; d < D; ++d ) {
                l[ d ] = first || lo[ t ][ d ] < l[ d ] ? lo[ t ][ d ] : l[ d ];
                u[ d ] = first || hi[ t ][ d ] > u[ d ] ? hi[ t ][ d ] : u[ d ];
            }
            first = false;
        }
        for ( int d = 0; d < D; ++d ) { heap.box( h, 0, d ) = l[ d ]; heap.box( h, 1, d ) = u[ d ]; }
        for ( int d = 1; d < D; ++d )
            if ( u[ d ] - l[ d ] > u[ ax[ j ] ] - l[ ax[ j ] ] )
                ax[ j ] = d;
        if ( e - b > leaf_size && u[ ax[ j ] ] > l[ ax[ j ] ] ) {
            cut[ j ] = 1;
            mid[ j ] = b + ( e - b ) / 2;
            // the bracket: a sample of the node's keys, around the rank of the median
            const SI m = e - b, S = std::min<SI>( m, 4096 ), r = ( mid[ j ] - b ) * S / m;
            std::vector<TF> smp( S );
            for ( SI i = 0; i < S; ++i )
                smp[ i ] = TF( in.positions( b + i * m / S, ax[ j ] ) );
            std::sort( smp.begin(), smp.end() );
            // ( the median's rank in the sample spreads by sqrt( S ) / 2 = 32: 6 of those miss it once in 1e9 )
            plo[ j ] = smp[ std::max<SI>( r - 192, 0 ) ];
            phi[ j ] = smp[ std::min<SI>( r + 192, S - 1 ) ];
        }
    }
    auto key = [&]( SI j, SI k ) { return TF( in.positions( k, ax[ j ] ) ); };

    // ---- the keys inside the bracket: counted, then gathered ( the whole node when the bracket missed the rank )
    std::vector<SI> nl( T, 0 ), nm( T, 0 ), mo( T, 0 );
    run( [&]( SI t, SI j, SI k0, SI k1 ) {
        if ( ! cut[ j ] ) return;
        SI a = 0, c = 0;
        for ( SI k = k0; k < k1; ++k ) {
            const TF v = key( j, k );
            a += v < plo[ j ];
            c += v >= plo[ j ] && v <= phi[ j ];
        }
        nl[ t ] = a; nm[ t ] = c;
    } );
    std::vector<std::vector<TF>> keys( nb );
    std::vector<SI> below( nb, 0 );
    for ( SI j = 0; j < nb; ++j ) {
        if ( ! cut[ j ] ) continue;
        SI a = 0, c = 0;
        for ( SI t = j * G; t < ( j + 1 ) * G; ++t ) { mo[ t ] = c; a += nl[ t ]; c += nm[ t ]; }
        const SI r = mid[ j ] - beg[ j ];
        if ( r < a || r >= a + c ) {                  // the sample missed: the bracket is the whole node
            plo[ j ] = -std::numeric_limits<TF>::infinity();
            phi[ j ] = std::numeric_limits<TF>::infinity();
            a = 0;
            for ( SI t = j * G; t < ( j + 1 ) * G; ++t ) { SI j_, k0, k1; part( t, j_, k0, k1 ); mo[ t ] = k0 - beg[ j ]; }
            c = end[ j ] - beg[ j ];
        }
        below[ j ] = a;
        keys[ j ].resize( c );
    }
    run( [&]( SI t, SI j, SI k0, SI k1 ) {
        if ( ! cut[ j ] ) return;
        TF *o = keys[ j ].data() + mo[ t ];
        for ( SI k = k0; k < k1; ++k ) {
            const TF v = key( j, k );
            if ( v >= plo[ j ] && v <= phi[ j ] )
                *( o++ ) = v;
        }
    } );

    // ---- the median, then how many of the keys equal to it go left
    std::vector<TF> med( nb );
    std::vector<SI> eq_left( nb, 0 );
    for ( SI j = 0; j < nb; ++j ) {
        if ( ! cut[ j ] ) continue;
        auto &kk = keys[ j ];
        const SI r = mid[ j ] - beg[ j ] - below[ j ];
        std::nth_element( kk.begin(), kk.begin() + r, kk.end() );
        med[ j ] = kk[ r ];
        SI less = 0;
        for ( SI i = 0; i < r; ++i )                  // ( `nth_element`: what is before `r` is <= the median )
            less += kk[ i ] < med[ j ];
        eq_left[ j ] = r - less;
        std::vector<TF>().swap( kk );
    }

    // ---- the stable partition: per worker, how many go left ( under the median, and its share of the equal ones )
    std::vector<SI> lt( T, 0 ), eq( T, 0 ), lbase( T, 0 ), rbase( T, 0 ), ebase( T, 0 );
    run( [&]( SI t, SI j, SI k0, SI k1 ) {
        if ( ! cut[ j ] ) return;
        SI a = 0, c = 0;
        for ( SI k = k0; k < k1; ++k ) {
            const TF v = key( j, k );
            a += v < med[ j ];
            c += v == med[ j ];
        }
        lt[ t ] = a; eq[ t ] = c;
    } );
    for ( SI j = 0; j < nb; ++j ) {
        if ( ! cut[ j ] ) continue;
        SI l = beg[ j ], r = mid[ j ], e_ = 0;
        for ( SI t = j * G; t < ( j + 1 ) * G; ++t ) {
            SI j_, k0, k1; part( t, j_, k0, k1 );
            const SI el = std::clamp<SI>( eq_left[ j ] - e_, 0, eq[ t ] );
            lbase[ t ] = l; rbase[ t ] = r; ebase[ t ] = e_;
            l += lt[ t ] + el;
            r += ( k1 - k0 ) - lt[ t ] - el;
            e_ += eq[ t ];
        }
    }

    // ---- the permuted cloud ( a node that is not cut copies its slice as it is )
    run( [&]( SI t, SI j, SI k0, SI k1 ) {
        SI l = lbase[ t ], r = rbase[ t ], e_ = ebase[ t ];
        for ( SI k = k0; k < k1; ++k ) {
            SI o = k;
            if ( cut[ j ] ) {
                const TF v = key( j, k );
                const bool left = v < med[ j ] || ( v == med[ j ] && e_++ < eq_left[ j ] );
                o = left ? l++ : r++;
            }
            out.order( o ) = in.order( k );
            for ( int d = 0; d < D; ++d )
                out.positions( o, d ) = in.positions( k, d );
            if constexpr ( weighted )
                out.weights( o ) = in.weights( k );
        }
    } );
}

void bsp_build_tree( const auto &queue, const auto &src, auto &&c0, auto &&c1, auto &&heap, SI n, SI depth, SI leaf_size ) {
    std::vector<SI> perm( n ), beg( 1, 0 ), end( 1, n ), mid( 1 ), pre( 1, 0 );
    auto pm = [&]( SI i ) -> SI & { return perm[ i ]; };
    const int nt = std::max( queue.nb_workers(), 1 );

    for ( SI level = 0; level < depth; ++level ) {
        // ( in preorder, the right child of a node of this level is past its left subtree, of `2^( depth - level - 1 ) - 1` nodes )
        const SI nb = SI( beg.size() ), jump = ( SI( 1 ) << ( depth - level - 1 ) );
        mid.resize( nb );
        for ( SI j = 0; j < nb; ++j ) {
            heap.left( pre[ j ] ) = level + 1 < depth ? pre[ j ] + 1 : -1;
            heap.right( pre[ j ] ) = level + 1 < depth ? pre[ j ] + jump : -1;
        }
        auto run = [&]( const auto &in, auto &&out ) {
            auto node = [&]( SI j ) {
                const SI h = pre[ j ];
                heap.begin( h ) = beg[ j ];
                heap.end( h ) = end[ j ];
                SI m = 0;
                bsp_build_level( in, out, pm, beg[ j ], end[ j ],
                                 BspNodeView<DECAYED_TYPE_OF( heap.box )>{ heap.box, h },
                                 BspNodeView<DECAYED_TYPE_OF( heap.wa )>{ heap.wa, h },
                                 heap.wb( h ), m, leaf_size );
                mid[ j ] = m;
            };
            const int T = int( std::min<SI>( nt, nb ) );
            if ( T <= 1 || n < 4096 ) {
                for ( SI j = 0; j < nb; ++j ) node( j );
                return;
            }
            queue.run_threads( T, [&]( int t ) {
                const SI e = nb * ( t + 1 ) / T;
                for ( SI j = nb * t / T; j < e; ++j ) node( j );
            } );
        };
        // ( the top: fewer nodes than workers, and enough seeds per node to share -- a group of workers per node )
        const SI G = nt / nb;
        auto level_of = [&]( const auto &in, auto &&out ) {
            if ( G >= 2 && n / nb >= 8192 )
                bsp_wide_level( queue, in, out, heap, pre, beg, end, mid, G, leaf_size );
            else
                run( in, out );
        };
        if ( level == 0 )         level_of( src, c0 );
        else if ( level % 2 == 1 ) level_of( c0, c1 );
        else                      level_of( c1, c0 );

        if ( level + 1 < depth ) {
            std::vector<SI> nb_( 2 * nb ), ne_( 2 * nb ), np_( 2 * nb );
            for ( SI j = 0; j < nb; ++j ) {
                nb_[ 2 * j ] = beg[ j ];  nb_[ 2 * j + 1 ] = mid[ j ];
                ne_[ 2 * j ] = mid[ j ];  ne_[ 2 * j + 1 ] = end[ j ];
                np_[ 2 * j ] = pre[ j ] + 1;  np_[ 2 * j + 1 ] = pre[ j ] + jump;
            }
            beg.swap( nb_ );
            end.swap( ne_ );
            pre.swap( np_ );
        }
    }

    // the leaves are the last level ( a node with nothing to cut passes its seeds down ): those that hold seeds
    SI nb_leaves = 0;
    for ( SI j = 0; j < SI( beg.size() ); ++j )
        nb_leaves += end[ j ] > beg[ j ];
    heap.nb_leaves( 0 ) = nb_leaves;
}

} // namespace sdot
