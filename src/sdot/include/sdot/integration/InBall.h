#pragma once

// =====================================================================================
// INTEGRATING ONLY INSIDE A BALL: the cell of seed `i` stays its Laguerre polytope, and what is integrated over it is
// `rho( x ) 1_{ |x - p_i|^2 <= r2 }` -- partial transport ( a sink of weight `w_s`: `r2 = w_i - w_s` ), whose cells are
// the Laguerre cells capped by the balls of radius `sqrt( w_i - w_s )`.
//
// It is the INTEGRATION that changes, not the cell: a distribution still cuts the cell into its pieces of constant density,
// and each piece is handed over with a density that knows the ball ( `InBallDensity` ). What the integrators ask of a
// density that is not constant on a piece, it answers EXACTLY, in closed form:
//
//     integrate_over_simplex( pts )                       rho |T inter B|
//     integrate_moments_over_simplex( pts, m, mx, m2 )    the moments of order 0, 1, 2 of rho over T inter B
//     integrate_moment_over_simplex<K>( pts )             one of them ( `diagram/Moments.h` )
//     facet_mass( pc, cut )                               rho |facet inter B| ( the hessian, `diagram::hessian_row` )
//     weight_mass( pc )                                   d / d r2 of the mass of the piece: the arc of the sphere inside it
//                                                         ( the diagonal of the hessian, beyond its facets )
//
// The other integrands -- radial costs, ... -- take the same place ( `diagram::cell_dist` ).
//
// = The closed forms ( 2D )
//
// A convex polygon inter the disc is the signed sum, over its edges `[ a, b ]`, of the triangles `( c, a, b )` inter the disc,
// `c` its center. Such a triangle splits along its edge into what lies inside the disc ( a triangle `( c, u, v )`, exact
// moments ) and what lies outside ( a SECTOR between the directions of `u` and `v` ): `|S| = r2 theta / 2`,
// `int_S y = r^3 / 3 ( v^_y - u^_y, u^_x - v^_x )`, `int_S |y|^2 = r2^2 theta / 4`. The derivative of the area with respect to
// `r2` is the sum of the half angles of the sectors: the arc inside the polygon, over `2 r`.
// =====================================================================================

#include <loom/support/common_macros.h>
#include <loom/support/containers/Vector.h>
#include <loom/support/math.h>
#include <cmath>

namespace sdot {
namespace integration {

/// the moments of a region about the center of the disc: `m = int 1`, `f = int y`, `q = int |y|^2`, and `a = d m / d r2`
template<class TF>
struct DiscMoments {
    TF m = 0, fx = 0, fy = 0, q = 0, a = 0;

    HD void add_triangle( TF ux, TF uy, TF vx, TF vy, TF s ) {        ///< `( 0, u, v )`, signed, times `s`
        const TF A = s * ( ux * vy - uy * vx ) / 2;
        m  += A;
        fx += A * ( ux + vx ) / 3;
        fy += A * ( uy + vy ) / 3;
        q  += A * ( ux * ux + uy * uy + vx * vx + vy * vy + ux * vx + uy * vy ) / 6;
    }

    HD void add_sector( TF ux, TF uy, TF vx, TF vy, TF r2, TF s ) {   ///< between the directions of `u` and `v`, signed, times `s`
        const TF nu = sdot::sqrt( ux * ux + uy * uy ), nv = sdot::sqrt( vx * vx + vy * vy );
        if ( ! ( nu > 0 ) || ! ( nv > 0 ) )
            return;
        const TF th = s * TF( std::atan2( ux * vy - uy * vx, ux * vx + uy * vy ) );
        const TF r = sdot::sqrt( r2 );
        m  += r2 * th / 2;
        q  += r2 * r2 * th / 4;
        a  += th / 2;
        fx += s * r2 * r / 3 * ( vy / nv - uy / nu );
        fy += s * r2 * r / 3 * ( ux / nu - vx / nv );
    }
};

/// the part `t in [ t0, t1 ]` of the segment `a + t ( b - a )`, `t in [ 0, 1 ]`, inside `|y|^2 <= r2` -- `false` if there is none
template<class TF>
HD bool clip_segment( TF ax, TF ay, TF bx, TF by, TF r2, TF &t0, TF &t1 ) {
    const TF dx = bx - ax, dy = by - ay;
    const TF A = dx * dx + dy * dy;
    if ( ! ( A > 0 ) )
        return false;
    const TF B = ax * dx + ay * dy, C = ax * ax + ay * ay - r2;
    const TF disc = B * B - A * C;
    if ( ! ( disc > 0 ) )
        return false;
    const TF s = sdot::sqrt( disc );
    t0 = ( - B - s ) / A;
    t1 = ( - B + s ) / A;
    if ( t0 < 0 ) t0 = 0;
    if ( t1 > 1 ) t1 = 1;
    return t0 < t1;
}

/// the signed triangle `( 0, a, b )` inter the disc `|y|^2 <= r2`, times `s`, accumulated in `res`
template<class TF>
HD void add_edge_in_disc( DiscMoments<TF> &res, TF ax, TF ay, TF bx, TF by, TF r2, TF s ) {
    TF t0, t1;
    if ( ! clip_segment( ax, ay, bx, by, r2, t0, t1 ) ) {
        res.add_sector( ax, ay, bx, by, r2, s );
        return;
    }
    const TF dx = bx - ax, dy = by - ay;
    const TF px = ax + t0 * dx, py = ay + t0 * dy;
    const TF qx = ax + t1 * dx, qy = ay + t1 * dy;
    if ( t0 > 0 ) res.add_sector( ax, ay, px, py, r2, s );
    res.add_triangle( px, py, qx, qy, s );
    if ( t1 < 1 ) res.add_sector( qx, qy, bx, by, r2, s );
}

/// the length of the segment `[ a, b ]` inside the disc of center `c`
template<class TF>
HD TF segment_in_disc( TF ax, TF ay, TF bx, TF by, TF cx, TF cy, TF r2 ) {
    TF t0, t1;
    if ( ! clip_segment( ax - cx, ay - cy, bx - cx, by - cy, r2, t0, t1 ) )
        return 0;
    const TF dx = bx - ax, dy = by - ay;
    return ( t1 - t0 ) * sdot::sqrt( dx * dx + dy * dy );
}

/// the density `rho` of a piece, integrated only inside the ball `|x - center|^2 <= r2` ( see above )
template<class Dens,class TF_,int D>
struct InBallDensity {
    using TF = TF_;
    static constexpr int ct_dim = D;
    static constexpr bool is_constant = false;
    static constexpr bool exact_moments = true;          ///< its moments and their adjoints are closed forms ( `diagram/Moments.h` )
    static_assert( D == 2, "integration in a ball: in 2D only for now" );
    static_assert( Dens::is_constant, "integration in a ball: only for a density constant on each piece ( `Image`, `Box`, `Polygon` )" );

    Dens          base;                                 ///< the density of the piece ( constant )
    Vector<TF,D>  center;
    TF            r2;

    HD TF rho() const { return TF( base.value ); }

    /// the moments, about the center, of the triangle `pts` inter the ball
    HD DiscMoments<TF> simplex_moments( const auto &pts ) const {
        DiscMoments<TF> res;
        const TF x0 = TF( pts[ 0 ][ 0 ] ) - center[ 0 ], y0 = TF( pts[ 0 ][ 1 ] ) - center[ 1 ];
        const TF x1 = TF( pts[ 1 ][ 0 ] ) - center[ 0 ], y1 = TF( pts[ 1 ][ 1 ] ) - center[ 1 ];
        const TF x2 = TF( pts[ 2 ][ 0 ] ) - center[ 0 ], y2 = TF( pts[ 2 ][ 1 ] ) - center[ 1 ];
        const TF s = ( x1 - x0 ) * ( y2 - y0 ) - ( y1 - y0 ) * ( x2 - x0 ) < 0 ? TF( -1 ) : TF( 1 );
        add_edge_in_disc( res, x0, y0, x1, y1, r2, s );
        add_edge_in_disc( res, x1, y1, x2, y2, r2, s );
        add_edge_in_disc( res, x2, y2, x0, y0, r2, s );
        return res;
    }

    HD TF integrate_over_simplex( const auto &pts ) const {
        return rho() * simplex_moments( pts ).m;
    }

    /// in absolute coordinates: `int x = f + m c`, `int |x|^2 = q + 2 c . f + m |c|^2`
    HD void integrate_moments_over_simplex( const auto &pts, TF &m, auto &mx, TF &m2 ) const {
        const auto r = simplex_moments( pts );
        const TF k = rho();
        m += k * r.m;
        mx[ 0 ] += k * ( r.fx + r.m * center[ 0 ] );
        mx[ 1 ] += k * ( r.fy + r.m * center[ 1 ] );
        m2 += k * ( r.q + 2 * ( center[ 0 ] * r.fx + center[ 1 ] * r.fy ) + r.m * ( center[ 0 ] * center[ 0 ] + center[ 1 ] * center[ 1 ] ) );
    }

    /// `K < D`: `int x_K`, `K == D`: `int |x|^2`, `K == D + 1`: `int 1` ( `diagram/Moments.h` )
    template<int K>
    HD TF integrate_moment_over_simplex( const auto &pts ) const {
        TF m = 0, m2 = 0;
        Vector<TF,D> mx = Vector<TF,D>::zeros();
        integrate_moments_over_simplex( pts, m, mx, m2 );
        if constexpr ( K < D )       return mx[ K ];
        else if constexpr ( K == D ) return m2;
        else                         return m;
    }

    /// the mass of the facet `cut` of the piece inside the ball ( 2D: cut `i` carries the edge `[ v_i, v_i+1 ]` )
    HD TF facet_mass( const auto &pc, int cut ) const {
        const int n = pc.nb_vertices(), j = cut + 1 < n ? cut + 1 : 0;
        return rho() * segment_in_disc( TF( pc.coord( cut, 0 ) ), TF( pc.coord( cut, 1 ) ), TF( pc.coord( j, 0 ) ), TF( pc.coord( j, 1 ) ),
                                        center[ 0 ], center[ 1 ], r2 );
    }

    /// THE POLYNOMIAL that a moment integrates ( `diagram/Moments.h`: `x_K`, `|x|^2`, `1` ), written about the center:
    /// `alpha + beta . y + gamma |y|^2`, `y = x - c`
    struct Poly { TF alpha = 1, bx = 0, by = 0, gamma = 0; };

    template<int K>
    HD Poly poly() const {
        Poly p;
        if constexpr ( K < D ) {
            p.alpha = center[ K ];
            ( K == 0 ? p.bx : p.by ) = 1;
        } else if constexpr ( K == D ) {
            p.alpha = center[ 0 ] * center[ 0 ] + center[ 1 ] * center[ 1 ];
            p.bx = 2 * center[ 0 ];
            p.by = 2 * center[ 1 ];
            p.gamma = 1;
        }
        return p;
    }

    HD static TF value( const Poly &p, TF yx, TF yy ) { return p.alpha + p.bx * yx + p.by * yy + p.gamma * ( yx * yx + yy * yy ); }

    /// THE ADJOINT of the integral of `p rho` over the triangle `pts` inter the ball, with respect to its vertices: the region
    /// grows by what its edges sweep INSIDE the ball, `int_{edge inter B} p n . ( ( 1 - t ) da + t db ) ds` ( a polynomial of degree
    /// 3 in `t`: Gauss with 3 nodes is exact ). The ball's own share ( its center and its radius, i.e. the seed ) is `seed_bwd`'s.
    HD void simplex_bwd( const Poly &p, TF integral, const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const {
        base.add_value_grad( grad_dist, g * integral );
        const TF x0 = TF( pts[ 0 ][ 0 ] ), y0 = TF( pts[ 0 ][ 1 ] );
        const TF s = ( TF( pts[ 1 ][ 0 ] ) - x0 ) * ( TF( pts[ 2 ][ 1 ] ) - y0 ) - ( TF( pts[ 1 ][ 1 ] ) - y0 ) * ( TF( pts[ 2 ][ 0 ] ) - x0 ) < 0 ? TF( -1 ) : TF( 1 );
        const TF k = g * rho();
        const TF gn[ 3 ] = { TF( 0.5 ) - TF( 0.3872983346207417 ), TF( 0.5 ), TF( 0.5 ) + TF( 0.3872983346207417 ) };
        const TF gw[ 3 ] = { TF( 5 ) / 18, TF( 8 ) / 18, TF( 5 ) / 18 };
        for ( int i = 0; i < 3; ++i ) {
            const int j = ( i + 1 ) % 3;
            const TF ax = TF( pts[ i ][ 0 ] ) - center[ 0 ], ay = TF( pts[ i ][ 1 ] ) - center[ 1 ];
            const TF bx = TF( pts[ j ][ 0 ] ) - center[ 0 ], by = TF( pts[ j ][ 1 ] ) - center[ 1 ];
            TF t0, t1;
            if ( ! clip_segment( ax, ay, bx, by, r2, t0, t1 ) )
                continue;
            TF wa = 0, wb = 0;                           // `int p ( 1 - t ) dt`, `int p t dt` on `[ t0, t1 ]`
            for ( int q = 0; q < 3; ++q ) {
                const TF t = t0 + ( t1 - t0 ) * gn[ q ];
                const TF v = gw[ q ] * ( t1 - t0 ) * value( p, ax + t * ( bx - ax ), ay + t * ( by - ay ) );
                wa += v * ( 1 - t );
                wb += v * t;
            }
            const TF nx = s * ( by - ay ), ny = - s * ( bx - ax );           // the outward normal times the length
            grad_pts[ i ][ 0 ] += k * nx * wa;
            grad_pts[ i ][ 1 ] += k * ny * wa;
            grad_pts[ j ][ 0 ] += k * nx * wb;
            grad_pts[ j ][ 1 ] += k * ny * wb;
        }
    }

    HD void integrate_over_simplex_bwd( const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const {
        simplex_bwd( Poly{}, simplex_moments( pts ).m, pts, g, grad_pts, grad_dist );
    }

    template<int K>
    HD void integrate_moment_over_simplex_bwd( const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const {
        simplex_bwd( poly<K>(), integrate_moment_over_simplex<K>( pts ) / rho(), pts, g, grad_pts, grad_dist );
    }

    /// the integrand depends on the seed of the cell: `seed_bwd` gives the cotangent of the integral of `p rho` over the piece on
    /// the seed's position ( the ball's center: `int_{arc} p n ds`, the arc of the sphere inside the piece ) and on its weight
    /// ( `r2 = w - w_s`: `int_{arc} p ds / ( 2 r )` )
    static constexpr bool depends_on_seed = true;

    HD void seed_bwd_poly( const Poly &p, const auto &pc, TF g, TF *gp, TF &gw ) const {
        const int n = pc.nb_vertices();
        if ( n < 3 )
            return;
        TF ar = 0;
        for ( int i = 0; i < n; ++i ) {
            const int j = i + 1 < n ? i + 1 : 0;
            ar += TF( pc.coord( i, 0 ) ) * TF( pc.coord( j, 1 ) ) - TF( pc.coord( j, 0 ) ) * TF( pc.coord( i, 1 ) );
        }
        const TF s = ar < 0 ? TF( -1 ) : TF( 1 );
        const TF r = sdot::sqrt( r2 );
        TF nx = 0, ny = 0, ln = 0;                       // `int_{arc} p n ds / r`, `int_{arc} p ds / r`
        auto arc = [&]( TF ux, TF uy, TF vx, TF vy, TF sg ) {
            if ( ! ( ux * ux + uy * uy > 0 ) || ! ( vx * vx + vy * vy > 0 ) )
                return;
            const TF f1 = TF( std::atan2( uy, ux ) );
            const TF th = sg * TF( std::atan2( ux * vy - uy * vx, ux * vx + uy * vy ) ), f2 = f1 + th;
            const TF c1 = TF( std::cos( f1 ) ), s1 = TF( std::sin( f1 ) ), c2 = TF( std::cos( f2 ) ), s2 = TF( std::sin( f2 ) );
            const TF C = s2 - s1, S = c1 - c2;                                       // int cos, int sin
            const TF CC = th / 2 + ( s2 * c2 - s1 * c1 ) / 2, SS = th - CC;          // int cos^2, int sin^2
            const TF CS = ( s2 * s2 - s1 * s1 ) / 2;                                 // int cos sin
            const TF A = p.alpha + p.gamma * r2;                                     // `p` on the sphere: `A + r beta . n`
            nx += A * C + r * ( p.bx * CC + p.by * CS );
            ny += A * S + r * ( p.bx * CS + p.by * SS );
            ln += A * th + r * ( p.bx * C + p.by * S );
        };
        for ( int i = 0; i < n; ++i ) {
            const int j = i + 1 < n ? i + 1 : 0;
            const TF ax = TF( pc.coord( i, 0 ) ) - center[ 0 ], ay = TF( pc.coord( i, 1 ) ) - center[ 1 ];
            const TF bx = TF( pc.coord( j, 0 ) ) - center[ 0 ], by = TF( pc.coord( j, 1 ) ) - center[ 1 ];
            // the arcs of the triangle `( c, a, b )` inter the disc: its sectors ( `add_edge_in_disc` )
            TF t0, t1;
            if ( ! clip_segment( ax, ay, bx, by, r2, t0, t1 ) ) {
                arc( ax, ay, bx, by, s );
                continue;
            }
            if ( t0 > 0 ) arc( ax, ay, ax + t0 * ( bx - ax ), ay + t0 * ( by - ay ), s );
            if ( t1 < 1 ) arc( ax + t1 * ( bx - ax ), ay + t1 * ( by - ay ), bx, by, s );
        }
        const TF k = g * rho();
        gp[ 0 ] += k * r * nx;
        gp[ 1 ] += k * r * ny;
        gw += k * ln / 2;
    }

    HD void seed_bwd( const auto &pc, TF g, TF *gp, TF &gw ) const { seed_bwd_poly( Poly{}, pc, g, gp, gw ); }

    template<int K>
    HD void seed_bwd_moment( const auto &pc, TF g, TF *gp, TF &gw ) const { seed_bwd_poly( poly<K>(), pc, g, gp, gw ); }

    /// `d mass / d r2` of the piece: what the sphere sweeps inside it when the radius grows
    HD TF weight_mass( const auto &pc ) const {
        DiscMoments<TF> res;
        const int n = pc.nb_vertices();
        for ( int i = 0; i < n; ++i ) {
            const int j = i + 1 < n ? i + 1 : 0;
            add_edge_in_disc( res, TF( pc.coord( i, 0 ) ) - center[ 0 ], TF( pc.coord( i, 1 ) ) - center[ 1 ],
                                   TF( pc.coord( j, 0 ) ) - center[ 0 ], TF( pc.coord( j, 1 ) ) - center[ 1 ], r2, TF( 1 ) );
        }
        return rho() * ( res.a < 0 ? - res.a : res.a );
    }
};

/// a distribution whose pieces are integrated inside the ball `|x - center|^2 <= r2` only. A piece entirely inside the
/// ball goes through unchanged ( its density stays constant: the closed forms of the integrators apply ), one entirely
/// outside is not handed over at all.
template<class Dist,class TF,int D>
struct InBallDist {
    static constexpr bool cuts_pieces = [] { if constexpr ( requires { Dist::cuts_pieces; } ) return bool( Dist::cuts_pieces ); else return false; }();
    static constexpr bool in_ball = true;

    Dist          dist;
    Vector<TF,D>  center;
    TF            r2;

    HD void for_each_piece( const auto &cell, auto &&ws, auto &&func ) const {
        if ( ! ( r2 > 0 ) )
            return;
        dist.for_each_piece( cell, ws, [&]( const auto &pc, const auto &dens ) {
            const int n = pc.nb_vertices();
            if ( n == 0 || ! pc.bounded() )
                return func( pc, InBallDensity<DECAYED_TYPE_OF( dens ),TF,D>{ dens, center, r2 } );
            // inside: every vertex in the ball ( convex ); outside: the box of the piece misses the ball
            bool inside = true;
            TF lo[ D ], hi[ D ];
            for ( int i = 0; i < n; ++i ) {
                TF d2 = 0;
                for ( int d = 0; d < D; ++d ) {
                    const TF x = TF( pc.coord( i, d ) );
                    d2 += ( x - center[ d ] ) * ( x - center[ d ] );
                    if ( i == 0 || x < lo[ d ] ) lo[ d ] = x;
                    if ( i == 0 || x > hi[ d ] ) hi[ d ] = x;
                }
                if ( d2 > r2 )
                    inside = false;
            }
            if ( inside )
                return func( pc, dens );
            TF e2 = 0;
            for ( int d = 0; d < D; ++d ) {
                const TF e = center[ d ] < lo[ d ] ? lo[ d ] - center[ d ] : center[ d ] > hi[ d ] ? center[ d ] - hi[ d ] : TF( 0 );
                e2 += e * e;
            }
            if ( e2 >= r2 )
                return;
            func( pc, InBallDensity<DECAYED_TYPE_OF( dens ),TF,D>{ dens, center, r2 } );
        } );
    }
};

template<class TF,int D>
HD auto in_ball( const auto &dist, const Vector<TF,D> &center, TF r2 ) {
    return InBallDist<DECAYED_TYPE_OF( dist ),TF,D>{ dist, center, r2 };
}

} // namespace integration
} // namespace sdot
