#pragma once

// =====================================================================================
// THE TREE OF AGGREGATES ON THE CARD ( `groups.py`, `diagram/Common.h::group_plane` ): what the card's cells read of it.
//
// The members of a group ( a ROOT aggregate ) split the cell their root would have as one seed: seen from outside, each is an
// ordinary seed ( its true position, its OUTER weight: the cells of the others are unchanged ), and between two members the
// plane is the bisector of their virtual diracs in the frame of their lowest common aggregate, at its scale. On the card:
//
//   * the cells of the MEMBERS are made by a pass of their own ( `Cell2D.cuh::group_pass`, one warp each ): the planes of the
//     other members of their root first ( `plane`, in double, in the seed's frame ), then the tree's walk, which leaves the
//     members of the root out;
//   * a facet between a member and a seed outside its root is written by the MEMBER ( both sides, whatever the ranks ): a cell
//     outside sees exact duplicates through planes that coincide and keeps one of them, the member's cell sees its own facet;
//   * each facet's coefficient is with respect to its own weights: the outer ones, or in the frame of the lowest common
//     aggregate ( `d2` of `plane`, `sdotplan/Groups.h` reads it so ).
//
// Every array is indexed by RANK ( the tree's order ), the aggregates by their number ( a parent before its children ).
// `root == nullptr`: no groups.
// =====================================================================================

namespace sdot::gpu {

template<class TR,int D>
struct CardGroups {
    const TR     *root       = nullptr;              ///< per rank: an identifier of its root aggregate ( a rank ), -1 when alone
    const TR     *next       = nullptr;              ///< per rank: the next member of its root ( a cycle, ranks )
    const int    *parent     = nullptr;              ///< per rank: its aggregate, -1 when alone
    const double *v          = nullptr;              ///< per rank: its virtual weight in its aggregate's frame
    const double *dirs       = nullptr;              ///< per rank: `[ D ]`, its unit direction in an aggregate of duplicates
    const int    *agg_parent = nullptr;              ///< per aggregate: its parent, -1 for a root
    const double *agg_center = nullptr;              ///< `[ D ]`: the centre of its frame ( duplicates: the apex's shift from the site )
    const double *agg_scale  = nullptr;              ///< the scale of its frame
    const double *agg_outer  = nullptr;              ///< what its children's weights move outside ( 0: duplicates )
    const double *agg_v      = nullptr;              ///< its virtual weight in its parent's frame

    __device__ __forceinline__ bool on() const { return root != nullptr; }
    __device__ __forceinline__ bool member( TR k ) const { return root && root[ k ] >= 0; }
    __device__ __forceinline__ bool same( TR k0, TR k1 ) const { return root && k0 != k1 && root[ k0 ] >= 0 && root[ k0 ] == root[ k1 ]; }
    __device__ __forceinline__ double scale_of( int a ) const { return a < 0 ? 1.0 : agg_scale[ a ]; }

    /// the weight of `k` in the frame of `L` ( `diagram/Common.h::frame_weight` )
    __device__ double frame_weight( TR k, int L ) const {
        double V = v[ k ];
        for ( int a = parent[ k ]; a >= 0 && a != L; ) {
            const int p = agg_parent[ a ];
            const double r = agg_outer[ a ] / scale_of( p );
            V = agg_v[ a ] + r * r * V;
            a = p;
        }
        return V;
    }

    /// the lowest common aggregate of two members of a root
    __device__ int common( TR k0, TR k1 ) const {
        for ( int a = parent[ k0 ]; a >= 0; a = agg_parent[ a ] )
            for ( int b = parent[ k1 ]; b >= 0; b = agg_parent[ b ] )
                if ( a == b ) return a;
        return -1;
    }

    /// the scale of the frame of the lowest common aggregate of two members ( a facet's coefficient with respect to the frame
    /// weights, from a plane of order one: `rho |facet| s / ( 2 |dir| )` )
    __device__ __forceinline__ double scale_between( TR k0, TR k1 ) const { return scale_of( common( k0, k1 ) ); }

    /// THE PLANE between the members `k0` and `k1`, IN THE FRAME OF `k0` ( `y = x - p0` ): `dir . y <= off`, `dir` of order one;
    /// `d2` the squared distance its coefficient reads ( `rho |facet| / ( 2 sqrt( d2 ) )` is with respect to the frame weight )
    __device__ void plane( TR k0, TR k1, const double *p0, const double *p1, double *dir, double &off, double &d2 ) const {
        const int L = common( k0, k1 );
        const double s = scale_of( L );
        const double V0 = frame_weight( k0, L ), V1 = frame_weight( k1, L );
        double n2 = 0, c = 0, q2 = 0;
        if ( agg_outer[ L ] == 0 ) {                     // duplicates: `q = u + ( p - p0 ) / s` around the apex `p0 + shift`
            for ( int d = 0; d < D; ++d ) {
                const double u0 = dirs[ SI( k0 ) * D + d ], u1 = dirs[ SI( k1 ) * D + d ] + ( p1[ d ] - p0[ d ] ) / s;
                dir[ d ] = u1 - u0;
                q2 += u1 * u1 - u0 * u0;
                c += dir[ d ] * agg_center[ SI( L ) * D + d ];
                n2 += dir[ d ] * dir[ d ];
            }
            off = c + s * ( q2 / 2 + ( V0 - V1 ) / 2 );
        } else {                                         // distinct positions: `q1 - q0 = ( p1 - p0 ) / s`, in `k0`'s frame
            for ( int d = 0; d < D; ++d ) {
                dir[ d ] = ( p1[ d ] - p0[ d ] ) / s;
                n2 += dir[ d ] * dir[ d ];
            }
            off = s * ( n2 / 2 + ( V0 - V1 ) / 2 );
        }
        d2 = n2 / ( s * s );
    }
};

} // namespace sdot::gpu
