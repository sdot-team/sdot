#pragma once

// =====================================================================================
// BAKED AMBIENT OCCLUSION ( `sdot.viz.occlusion.ambient_occlusion`, sdot ADR 0001 §4 ): for each sample ( a point near a
// corner of a face, and the face's normal ), how open the space is on each side of the face.
//
// Each side casts `R` cosine-weighted directions of its hemisphere ( a Fibonacci spiral on the disc, lifted, turned by the
// sample's `spin` ) up to `radius`; the first face met at distance `t` occludes by `1 - t / radius`. A side's value is one
// minus the mean occlusion of its rays. `side( i )`: +1 or -1, the side along the normal that is seen ( the outside of a
// closed piece ); 0, both, `out( i )` the more open one ( an open sheet, drawn on both sides ).
//
// The triangles go in a BVH ( median split on the longest axis of the centroids' box, 4 triangles per leaf ), walked for the
// CLOSEST hit, the far bound shrinking as hits come. Samples in parallel over the queue's threads.
// =====================================================================================

#include <loom/support/common_macros.h>
#include <loom/support/kernels/CpuQueue.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <vector>

namespace sdot {
namespace ambient_occlusion {

using V3 = std::array<double,3>;

inline V3 sub( V3 a, V3 b ) { return { a[ 0 ] - b[ 0 ], a[ 1 ] - b[ 1 ], a[ 2 ] - b[ 2 ] }; }
inline V3 crs( V3 a, V3 b ) { return { a[ 1 ] * b[ 2 ] - a[ 2 ] * b[ 1 ], a[ 2 ] * b[ 0 ] - a[ 0 ] * b[ 2 ], a[ 0 ] * b[ 1 ] - a[ 1 ] * b[ 0 ] }; }
inline double dot( V3 a, V3 b ) { return a[ 0 ] * b[ 0 ] + a[ 1 ] * b[ 1 ] + a[ 2 ] * b[ 2 ]; }

struct Node {
    V3 lo, hi;
    int first, count;                                           ///< a leaf: `count > 0`, triangles `order[ first .. first + count )`; else children `first`, `first + 1`
};

struct Bvh {
    std::vector<V3>   a, e1, e2;                                ///< the triangles: a corner and two edges
    std::vector<int>  order;
    std::vector<Node> nodes;

    void build( std::vector<V3> &&A, std::vector<V3> &&B, std::vector<V3> &&C ) {
        const int T = int( A.size() );
        a = A; e1.resize( T ); e2.resize( T );
        std::vector<V3> cen( T ), lo( T ), hi( T );
        for ( int t = 0; t < T; ++t ) {
            e1[ t ] = sub( B[ t ], A[ t ] ); e2[ t ] = sub( C[ t ], A[ t ] );
            for ( int d = 0; d < 3; ++d ) {
                lo[ t ][ d ] = std::min( { A[ t ][ d ], B[ t ][ d ], C[ t ][ d ] } );
                hi[ t ][ d ] = std::max( { A[ t ][ d ], B[ t ][ d ], C[ t ][ d ] } );
                cen[ t ][ d ] = ( A[ t ][ d ] + B[ t ][ d ] + C[ t ][ d ] ) / 3;
            }
        }
        order.resize( T );
        std::iota( order.begin(), order.end(), 0 );
        nodes.clear();
        nodes.reserve( 2 * std::max( T / 2, 1 ) + 1 );
        nodes.push_back( {} );
        split( 0, 0, T, cen, lo, hi );
    }

    void split( int n, int b, int e, const std::vector<V3> &cen, const std::vector<V3> &lo, const std::vector<V3> &hi ) {
        V3 l{ 1e300, 1e300, 1e300 }, h{ -1e300, -1e300, -1e300 }, cl = l, ch = h;
        for ( int i = b; i < e; ++i )
            for ( int d = 0; d < 3; ++d ) {
                const int t = order[ i ];
                l[ d ] = std::min( l[ d ], lo[ t ][ d ] ); h[ d ] = std::max( h[ d ], hi[ t ][ d ] );
                cl[ d ] = std::min( cl[ d ], cen[ t ][ d ] ); ch[ d ] = std::max( ch[ d ], cen[ t ][ d ] );
            }
        nodes[ n ].lo = l; nodes[ n ].hi = h;
        if ( e - b <= 4 ) {
            nodes[ n ].first = b; nodes[ n ].count = e - b;
            return;
        }
        int ax = 0;
        for ( int d = 1; d < 3; ++d )
            if ( ch[ d ] - cl[ d ] > ch[ ax ] - cl[ ax ] ) ax = d;
        const int m = ( b + e ) / 2;
        std::nth_element( order.begin() + b, order.begin() + m, order.begin() + e,
                          [&]( int x, int y ) { return cen[ x ][ ax ] < cen[ y ][ ax ]; } );
        const int c = int( nodes.size() );
        nodes[ n ].first = c; nodes[ n ].count = 0;
        nodes.push_back( {} ); nodes.push_back( {} );
        split( c, b, m, cen, lo, hi );
        split( c + 1, m, e, cen, lo, hi );
    }

    /// the distance of the closest triangle met by `o + t d`, `t` in `( tmin, tmax )`; `tmax` if none
    double closest( V3 o, V3 d, double tmin, double tmax ) const {
        if ( nodes.empty() || order.empty() )
            return tmax;
        V3 inv;
        for ( int k = 0; k < 3; ++k ) inv[ k ] = 1.0 / ( d[ k ] != 0 ? d[ k ] : 1e-300 );
        int stack[ 64 ], sp = 0;
        stack[ sp++ ] = 0;
        double best = tmax;
        while ( sp ) {
            const Node &nd = nodes[ stack[ --sp ] ];
            double t0 = tmin, t1 = best;                        // ( the slab test )
            for ( int k = 0; k < 3; ++k ) {
                double ta = ( nd.lo[ k ] - o[ k ] ) * inv[ k ], tb = ( nd.hi[ k ] - o[ k ] ) * inv[ k ];
                if ( ta > tb ) std::swap( ta, tb );
                t0 = std::max( t0, ta ); t1 = std::min( t1, tb );
            }
            if ( t0 > t1 )
                continue;
            if ( nd.count ) {
                for ( int i = nd.first; i < nd.first + nd.count; ++i ) {
                    const int t = order[ i ];                   // ( Möller-Trumbore )
                    const V3 p = crs( d, e2[ t ] );
                    const double det = dot( e1[ t ], p );
                    if ( std::fabs( det ) < 1e-300 ) continue;
                    const double iv = 1 / det;
                    const V3 tv = sub( o, a[ t ] );
                    const double u = dot( tv, p ) * iv;
                    if ( u < 0 || u > 1 ) continue;
                    const V3 q = crs( tv, e1[ t ] );
                    const double v = dot( d, q ) * iv;
                    if ( v < 0 || u + v > 1 ) continue;
                    const double tt = dot( e2[ t ], q ) * iv;
                    if ( tt > tmin && tt < best ) best = tt;
                }
            } else if ( sp + 2 <= 64 ) {
                stack[ sp++ ] = nd.first; stack[ sp++ ] = nd.first + 1;
            }
        }
        return best;
    }
};

/// THE ENTRY POINT. `pos [ n, 3 ]`, `tris [ T, 3 ]`; per sample `pts [ m, 3 ]`, `nrm [ m, 3 ]` ( unit ), `spin [ m ]`;
/// `side [ m ]`; `params = [ radius, eps, rays ]`; `out [ m ]`.
inline void run( const CpuQueue &queue, const auto &pos, const auto &tris, const auto &pts, const auto &nrm, const auto &spin,
                 const auto &side, const auto &params, auto &&out ) {
    const double radius = double( params( 0 ) ), eps = double( params( 1 ) );
    const int R = std::max( 1, int( params( 2 ) ) );
    const SI T = SI( tris.shape( 0 ) ), M = SI( pts.shape( 0 ) );

    Bvh bvh;
    {
        std::vector<V3> A( T ), B( T ), C( T );
        auto at = [&]( SI t, int c ) { const SI i = SI( tris( t, c ) ); return V3{ double( pos( i, 0 ) ), double( pos( i, 1 ) ), double( pos( i, 2 ) ) }; };
        for ( SI t = 0; t < T; ++t ) { A[ t ] = at( t, 0 ); B[ t ] = at( t, 1 ); C[ t ] = at( t, 2 ); }
        bvh.build( std::move( A ), std::move( B ), std::move( C ) );
    }

    std::vector<V3> dirs( R );                                  // ( the hemisphere z > 0, cosine weighted )
    for ( int k = 0; k < R; ++k ) {
        const double s = ( k + 0.5 ) / R, r = std::sqrt( s ), an = ( k + 0.5 ) * M_PI * ( 3 - std::sqrt( 5.0 ) );
        const double x = r * std::cos( an ), y = r * std::sin( an );
        dirs[ k ] = { x, y, std::sqrt( std::max( 0.0, 1 - x * x - y * y ) ) };
    }

    auto one = [&]( SI i ) {
        const V3 p{ double( pts( i, 0 ) ), double( pts( i, 1 ) ), double( pts( i, 2 ) ) };
        const V3 n{ double( nrm( i, 0 ) ), double( nrm( i, 1 ) ), double( nrm( i, 2 ) ) };
        V3 h = std::fabs( n[ 0 ] ) < 0.9 ? V3{ 1, 0, 0 } : V3{ 0, 1, 0 };
        V3 u = crs( n, h );
        const double lu = std::sqrt( dot( u, u ) );
        for ( auto &c : u ) c /= lu;
        V3 v = crs( n, u );
        const double cs = std::cos( double( spin( i ) ) ), sn = std::sin( double( spin( i ) ) );
        const V3 uu{ cs * u[ 0 ] + sn * v[ 0 ], cs * u[ 1 ] + sn * v[ 1 ], cs * u[ 2 ] + sn * v[ 2 ] };
        const V3 vv{ -sn * u[ 0 ] + cs * v[ 0 ], -sn * u[ 1 ] + cs * v[ 1 ], -sn * u[ 2 ] + cs * v[ 2 ] };
        double best = 0;
        const double want = double( side( i ) );
        for ( double sg : { 1.0, -1.0 } ) {
            if ( want * sg < 0 )
                continue;
            const V3 o{ p[ 0 ] + sg * eps * n[ 0 ], p[ 1 ] + sg * eps * n[ 1 ], p[ 2 ] + sg * eps * n[ 2 ] };
            double occ = 0;
            for ( int k = 0; k < R; ++k ) {
                const V3 &q = dirs[ k ];
                V3 d;
                for ( int c = 0; c < 3; ++c ) d[ c ] = sg * ( q[ 0 ] * uu[ c ] + q[ 1 ] * vv[ c ] + q[ 2 ] * n[ c ] );
                const double t = bvh.closest( o, d, eps, radius );
                if ( t < radius ) occ += 1 - t / radius;
            }
            best = std::max( best, 1 - occ / R );
        }
        out( i ) = best;
    };

    const int nt = int( std::min<SI>( std::max( queue.nb_workers(), 1 ), std::max<SI>( M / 256, 1 ) ) );
    if ( nt <= 1 ) {
        for ( SI i = 0; i < M; ++i ) one( i );
        return;
    }
    queue.run_threads( nt, [&]( int th ) {
        const SI b = M * th / nt, e = M * ( th + 1 ) / nt;
        for ( SI i = b; i < e; ++i ) one( i );
    } );
}

} // namespace ambient_occlusion
} // namespace sdot
