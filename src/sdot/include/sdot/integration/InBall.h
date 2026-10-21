#pragma once

// =====================================================================================
// INTEGRATING ONLY INSIDE A BALL: the cell of seed `i` stays its Laguerre polytope, and what is integrated over it is
// `rho( x ) 1_{ |x - p_i|^2 <= r2 }` -- partial transport ( a sink of weight `w_s`: `r2 = w_i - w_s` ), whose cells are
// the Laguerre cells capped by the balls of radius `sqrt( w_i - w_s )`.
//
// It is the INTEGRATION that changes, not the cell: a distribution still cuts the cell into its pieces of constant or AFFINE
// density ( `Image`, `Box`, `ConvexHull`; `Mesh` ), and each piece is handed over with a density that knows the ball
// ( `InBallDensity` ). What the integrators ask of a density that is not constant on a piece, it answers EXACTLY, in closed form:
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
// `c` its center. Such a triangle splits along its edge into what lies inside the disc ( a triangle `( c, u, v )` ) and what
// lies outside ( a SECTOR between the directions of `u` and `v` ). What is integrated -- the density, affine, times the
// polynomial of a moment, of degree 2 -- is a polynomial of degree 3 at most in `y = x - c`: the integrals of the MONOMIALS
// `y_x^a y_y^b`, `a + b <= 3`, over the two shapes are enough ( `DiscMoments` ), exact on a triangle ( its barycentric
// coordinates ) and on a sector ( polar: `r^( a + b + 2 ) / ( a + b + 2 ) int cos^a sin^b` ). The arcs of the sphere inside the
// polygon ( the derivatives with respect to the seed and to `r2` ) are integrals of `cos^a sin^b` too ( `ArcTrig` ).
// =====================================================================================

#include <loom/support/common_macros.h>
#include <loom/support/containers/Vector.h>
#include <loom/support/math.h>
#include <cmath>

namespace sdot {
namespace integration {


/// `int cos^a sin^b` over angular ranges, `a + b <= 4`, ACCUMULATED: the reduction formulas on the antiderivatives
template<class TF>
struct ArcTrig {
    TF J[ 5 ][ 5 ] = {};                                 ///< `J[ a ][ b ]`, `a + b <= 4`

    HD static void antiderivatives( TF f, TF ( &F )[ 5 ][ 5 ] ) {
        const TF c = TF( std::cos( f ) ), s = TF( std::sin( f ) );
        TF cp[ 6 ], sp[ 6 ];
        cp[ 0 ] = sp[ 0 ] = 1;
        for ( int i = 1; i < 6; ++i ) { cp[ i ] = cp[ i - 1 ] * c; sp[ i ] = sp[ i - 1 ] * s; }
        F[ 0 ][ 0 ] = f;
        F[ 1 ][ 0 ] = s;
        F[ 0 ][ 1 ] = - c;
        for ( int n = 2; n <= 4; ++n )
            for ( int a = 0; a <= n; ++a ) {
                const int b = n - a;
                if ( a >= 2 )      F[ a ][ b ] =   cp[ a - 1 ] * sp[ b + 1 ] / n + TF( a - 1 ) / n * F[ a - 2 ][ b ];
                else if ( b >= 2 ) F[ a ][ b ] = - cp[ a + 1 ] * sp[ b - 1 ] / n + TF( b - 1 ) / n * F[ a ][ b - 2 ];
                else               F[ a ][ b ] = s * s / 2;
            }
    }

    /// the range `[ f, f + th ]` ( `th` signed )
    HD void add( TF f, TF th ) {
        TF F1[ 5 ][ 5 ], F2[ 5 ][ 5 ];
        antiderivatives( f, F1 );
        antiderivatives( f + th, F2 );
        for ( int a = 0; a <= 4; ++a )
            for ( int b = 0; a + b <= 4; ++b )
                J[ a ][ b ] += F2[ a ][ b ] - F1[ a ][ b ];
    }

    /// the arc between the directions of `u` and `v`, signed by `s`
    HD void add_between( TF ux, TF uy, TF vx, TF vy, TF s ) {
        if ( ! ( ux * ux + uy * uy > 0 ) || ! ( vx * vx + vy * vy > 0 ) )
            return;
        add( TF( std::atan2( uy, ux ) ), s * TF( std::atan2( ux * vy - uy * vx, ux * vx + uy * vy ) ) );
    }
};

/// the moments of a region about the center of the disc: `M[ a ][ b ] = int y_x^a y_y^b`, `a + b <= 3`
template<class TF>
struct DiscMoments {
    TF M[ 4 ][ 4 ] = {};

    HD static TF factorial( int n ) { TF r = 1; for ( int i = 2; i <= n; ++i ) r *= i; return r; }
    HD static TF binomial( int n, int k ) { return factorial( n ) / ( factorial( k ) * factorial( n - k ) ); }

    HD void add_triangle( TF ux, TF uy, TF vx, TF vy, TF s ) {        ///< `( 0, u, v )`, signed, times `s`
        // `y = sigma u + tau v`: `int sigma^i tau^j = i! j! / ( i + j + 2 )!`, times `det( u, v )`
        const TF det = s * ( ux * vy - uy * vx );
        TF pux[ 4 ], pvx[ 4 ], puy[ 4 ], pvy[ 4 ];
        pux[ 0 ] = pvx[ 0 ] = puy[ 0 ] = pvy[ 0 ] = 1;
        for ( int i = 1; i < 4; ++i ) { pux[ i ] = pux[ i - 1 ] * ux; pvx[ i ] = pvx[ i - 1 ] * vx; puy[ i ] = puy[ i - 1 ] * uy; pvy[ i ] = pvy[ i - 1 ] * vy; }
        for ( int a = 0; a <= 3; ++a )
            for ( int b = 0; a + b <= 3; ++b ) {
                TF r = 0;
                for ( int i = 0; i <= a; ++i )
                    for ( int j = 0; j <= b; ++j )
                        r += binomial( a, i ) * binomial( b, j ) * pux[ i ] * pvx[ a - i ] * puy[ j ] * pvy[ b - j ]
                           * factorial( i + j ) * factorial( a + b - i - j );
                M[ a ][ b ] += det * r / factorial( a + b + 2 );
            }
    }

    HD void add_sector( TF ux, TF uy, TF vx, TF vy, TF r2, TF s ) {   ///< between the directions of `u` and `v`, signed, times `s`
        ArcTrig<TF> at;
        at.add_between( ux, uy, vx, vy, s );
        const TF r = sdot::sqrt( r2 );
        TF rp = r2;                                      // `r^( n + 2 )`
        for ( int n = 0; n <= 3; ++n, rp *= r )
            for ( int a = 0; a <= n; ++a )
                M[ a ][ n - a ] += rp / ( n + 2 ) * at.J[ a ][ n - a ];
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


/// the density `rho` of a piece, integrated only inside the ball `|x - center|^2 <= r2` ( see above ). `rho` is CONSTANT on the
/// piece ( `Dens::is_constant`: `value`, `add_value_grad` ) or AFFINE ( `Dens::is_affine`: `value_at`, `gradient_at`,
/// `add_value_grad_at` -- `MeshElementDensity` ): written about the center, `rho( c + y ) = r0 + g . y`, one code for both.
template<class Dens,class TF_,int D>
struct InBallDensity {
    using TF = TF_;
    static constexpr int ct_dim = D;
    static constexpr bool is_constant = false;
    static constexpr bool exact_moments = true;          ///< its moments and their adjoints are closed forms ( `diagram/Moments.h` )
    static constexpr bool affine = [] { if constexpr ( requires { Dens::is_affine; } ) return bool( Dens::is_affine ); else return false; }();
    static_assert( D == 2, "integration in a ball: in 2D and 3D only ( `InBall3.h` )" );
    static_assert( Dens::is_constant || affine, "integration in a ball: only for a density constant or affine on each piece ( `Image`, `Box`, "
                                                "`ConvexHull`, `Mesh` )" );

    Dens          base;                                 ///< the density of the piece
    Vector<TF,D>  center;
    TF            r2;

    /// `rho( c + y ) = r0 + gx y_x + gy y_y`
    struct Affine { TF r0, gx, gy; };
    HD Affine about_center() const {
        if constexpr ( Dens::is_constant )
            return { TF( base.value ), TF( 0 ), TF( 0 ) };
        else {
            const auto g = base.gradient_at( center );
            return { TF( base.value_at( center ) ), TF( g[ 0 ] ), TF( g[ 1 ] ) };
        }
    }

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

    /// `int p y_x^a y_y^b` over the region of `R`, `a + b <= 1`
    HD static TF p_moment( const Poly &p, const DiscMoments<TF> &R, int a, int b ) {
        return p.alpha * R.M[ a ][ b ] + p.bx * R.M[ a + 1 ][ b ] + p.by * R.M[ a ][ b + 1 ] + p.gamma * ( R.M[ a + 2 ][ b ] + R.M[ a ][ b + 2 ] );
    }

    /// `int p rho` over the region of `R`
    HD TF integral( const Poly &p, const DiscMoments<TF> &R ) const {
        const Affine f = about_center();
        return f.r0 * p_moment( p, R, 0, 0 ) + f.gx * p_moment( p, R, 1, 0 ) + f.gy * p_moment( p, R, 0, 1 );
    }

    HD TF integrate_over_simplex( const auto &pts ) const {
        return integral( Poly{}, simplex_moments( pts ) );
    }

    HD void integrate_moments_over_simplex( const auto &pts, TF &m, auto &mx, TF &m2 ) const {
        const auto R = simplex_moments( pts );
        m += integral( Poly{}, R );
        mx[ 0 ] += integral( poly<0>(), R );
        mx[ 1 ] += integral( poly<1>(), R );
        m2 += integral( poly<D>(), R );
    }

    /// `K < D`: `int x_K`, `K == D`: `int |x|^2`, `K == D + 1`: `int 1` ( `diagram/Moments.h` ), `K == D + 2`: THE SPHERE, the length
    /// of the circle inside the triangle where the density is not zero ( `sphere_length` )
    template<int K>
    HD TF integrate_moment_over_simplex( const auto &pts ) const {
        if constexpr ( K == D + 2 )
            return sphere_length( pts );
        else
            return integral( poly<K>(), simplex_moments( pts ) );
    }

    // ---- THE SPHERE inside a piece: `|S inter T|`, what a surface energy counts ( `PowerDiagram.sphere_areas` ) -------------
    //
    // Its derivatives are geometric. A line of the triangle pushed outward by `delta` lengthens the arc by `r / rho` at each of its
    // endpoints on that line ( `rho` the half chord ); moving the center by `dc` is moving the lines by `- dc . n`; and the
    // radius scales the arc and moves the endpoints: `d A / d r = A / r - sum_ends h / rho` ( `h` the signed distance of the line ).

    /// `0` where the density is zero: the circle there is not a free surface ( a wall )
    HD TF on_support() const {
        if constexpr ( Dens::is_constant ) return TF( base.value ) > 0 ? TF( 1 ) : TF( 0 );
        else return TF( 1 );
    }

    /// `func( i, j, t, rho, h, nx, ny )` for each point where the circle crosses the edge `[ v_i, v_j ]` of the triangle `pts`: its
    /// parameter `t` on the edge, the half chord `rho`, the signed distance `h` of the line and its outward unit normal
    HD void for_each_crossing( const auto &pts, auto &&func ) const {
        TF P[ 3 ][ 2 ];
        for ( int k = 0; k < 3; ++k ) { P[ k ][ 0 ] = TF( pts[ k ][ 0 ] ) - center[ 0 ]; P[ k ][ 1 ] = TF( pts[ k ][ 1 ] ) - center[ 1 ]; }
        const TF s = ( P[ 1 ][ 0 ] - P[ 0 ][ 0 ] ) * ( P[ 2 ][ 1 ] - P[ 0 ][ 1 ] ) - ( P[ 1 ][ 1 ] - P[ 0 ][ 1 ] ) * ( P[ 2 ][ 0 ] - P[ 0 ][ 0 ] ) < 0 ? TF( -1 ) : TF( 1 );
        for ( int i = 0; i < 3; ++i ) {
            const int j = ( i + 1 ) % 3;
            const TF dx = P[ j ][ 0 ] - P[ i ][ 0 ], dy = P[ j ][ 1 ] - P[ i ][ 1 ], l = sdot::sqrt( dx * dx + dy * dy );
            if ( ! ( l > 0 ) )
                continue;
            const TF nx = s * dy / l, ny = - s * dx / l;
            const TF h = nx * P[ i ][ 0 ] + ny * P[ i ][ 1 ], rho2 = r2 - h * h;
            if ( ! ( rho2 > 0 ) )
                continue;
            const TF A = l * l, B = P[ i ][ 0 ] * dx + P[ i ][ 1 ] * dy, sq = sdot::sqrt( B * B - A * ( P[ i ][ 0 ] * P[ i ][ 0 ] + P[ i ][ 1 ] * P[ i ][ 1 ] - r2 ) );
            for ( TF t : { ( - B - sq ) / A, ( - B + sq ) / A } )
                if ( t > 0 && t < 1 )
                    func( i, j, t, sdot::sqrt( rho2 ), h, nx, ny );
        }
    }

    /// the arcs of the circle inside the triangle `pts`, counterclockwise
    HD ArcTrig<TF> triangle_arcs( const auto &pts ) const {
        TF P[ 3 ][ 2 ];
        for ( int k = 0; k < 3; ++k ) { P[ k ][ 0 ] = TF( pts[ k ][ 0 ] ) - center[ 0 ]; P[ k ][ 1 ] = TF( pts[ k ][ 1 ] ) - center[ 1 ]; }
        if ( ( P[ 1 ][ 0 ] - P[ 0 ][ 0 ] ) * ( P[ 2 ][ 1 ] - P[ 0 ][ 1 ] ) - ( P[ 1 ][ 1 ] - P[ 0 ][ 1 ] ) * ( P[ 2 ][ 0 ] - P[ 0 ][ 0 ] ) < 0 )
            for ( int d = 0; d < 2; ++d ) { const TF t = P[ 1 ][ d ]; P[ 1 ][ d ] = P[ 2 ][ d ]; P[ 2 ][ d ] = t; }
        ArcTrig<TF> at;
        for ( int i = 0; i < 3; ++i ) {
            const int j = ( i + 1 ) % 3;
            const TF ax = P[ i ][ 0 ], ay = P[ i ][ 1 ], bx = P[ j ][ 0 ], by = P[ j ][ 1 ];
            TF t0, t1;
            if ( ! clip_segment( ax, ay, bx, by, r2, t0, t1 ) ) {
                at.add_between( ax, ay, bx, by, TF( 1 ) );
                continue;
            }
            if ( t0 > 0 ) at.add_between( ax, ay, ax + t0 * ( bx - ax ), ay + t0 * ( by - ay ), TF( 1 ) );
            if ( t1 < 1 ) at.add_between( ax + t1 * ( bx - ax ), ay + t1 * ( by - ay ), bx, by, TF( 1 ) );
        }
        return at;
    }

    HD TF sphere_length( const auto &pts ) const { return on_support() * sdot::sqrt( r2 ) * triangle_arcs( pts ).J[ 0 ][ 0 ]; }

    /// its adjoint with respect to the vertices: each endpoint of an arc moves with its edge
    HD void sphere_length_bwd( const auto &pts, TF g, auto &&grad_pts ) const {
        const TF r = sdot::sqrt( r2 ), f = g * on_support();
        for_each_crossing( pts, [&]( int i, int j, TF t, TF rho, TF, TF nx, TF ny ) {
            const TF c = f * r / rho;
            grad_pts[ i ][ 0 ] += c * ( 1 - t ) * nx;
            grad_pts[ i ][ 1 ] += c * ( 1 - t ) * ny;
            grad_pts[ j ][ 0 ] += c * t * nx;
            grad_pts[ j ][ 1 ] += c * t * ny;
        } );
    }

    /// ... and with respect to the seed, over the triangles of the piece `pc` ( its fan from vertex 0 )
    HD void sphere_length_seed_bwd( const auto &pc, TF g, TF *gp, TF &gw ) const {
        const int n = pc.nb_vertices();
        if ( n < 3 )
            return;
        const TF r = sdot::sqrt( r2 ), f = g * on_support();
        TF A = 0, ends = 0;
        for ( int k = 1; k + 1 < n; ++k ) {
            const TF pts[ 3 ][ 2 ] = { { TF( pc.coord( 0, 0 ) ), TF( pc.coord( 0, 1 ) ) }, { TF( pc.coord( k, 0 ) ), TF( pc.coord( k, 1 ) ) },
                                       { TF( pc.coord( k + 1, 0 ) ), TF( pc.coord( k + 1, 1 ) ) } };
            A += r * triangle_arcs( pts ).J[ 0 ][ 0 ];
            for_each_crossing( pts, [&]( int, int, TF, TF rho, TF h, TF nx, TF ny ) {
                gp[ 0 ] -= f * r / rho * nx;
                gp[ 1 ] -= f * r / rho * ny;
                ends += h / rho;
            } );
        }
        gw += f * ( A / r - ends ) / ( 2 * r );
    }

    /// the mass of the facet `cut` of the piece inside the ball ( 2D: cut `i` carries the edge `[ v_i, v_i+1 ]` ): affine on it, its
    /// length times its value at the middle
    HD TF facet_mass( const auto &pc, int cut ) const {
        const int n = pc.nb_vertices(), j = cut + 1 < n ? cut + 1 : 0;
        const TF ax = TF( pc.coord( cut, 0 ) ) - center[ 0 ], ay = TF( pc.coord( cut, 1 ) ) - center[ 1 ];
        const TF bx = TF( pc.coord( j, 0 ) ) - center[ 0 ], by = TF( pc.coord( j, 1 ) ) - center[ 1 ];
        return segment_mass( ax, ay, bx, by );
    }

    /// `int rho` on the segment `[ c + a, c + b ]` inside the ball
    HD TF segment_mass( TF ax, TF ay, TF bx, TF by ) const {
        TF t0, t1;
        if ( ! clip_segment( ax, ay, bx, by, r2, t0, t1 ) )
            return 0;
        const TF dx = bx - ax, dy = by - ay, tm = ( t0 + t1 ) / 2;
        const Affine f = about_center();
        return ( t1 - t0 ) * sdot::sqrt( dx * dx + dy * dy ) * ( f.r0 + f.gx * ( ax + tm * dx ) + f.gy * ( ay + tm * dy ) );
    }

    /// THE ADJOINT of the integral of `p rho` over the triangle `pts` inter the ball, with respect to its vertices: the region
    /// grows by what its edges sweep INSIDE the ball, `int_{edge inter B} p rho n . ( ( 1 - t ) da + t db ) ds` ( a polynomial of
    /// degree 4 in `t`: Gauss with 3 nodes is exact ). The ball's own share ( its center and its radius, i.e. the seed ) is
    /// `seed_bwd`'s. With respect to the density: `int p` over the region for a constant; for an affine one, its value at
    /// each corner is the integral of `p lambda_k`, affine too -- `int p lambda_k( c + y ) = P0 lambda_k( c ) + P1 . grad lambda_k`,
    /// i.e. `lambda_k` at `c` and at `c + e_d` ( `add_value_grad_at` is linear in the point's barycentric coordinates ).
    HD void simplex_bwd( const Poly &p, const DiscMoments<TF> &R, const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const {
        const TF P0 = p_moment( p, R, 0, 0 );
        if constexpr ( Dens::is_constant )
            base.add_value_grad( grad_dist, g * P0 );
        else {
            const TF Px = p_moment( p, R, 1, 0 ), Py = p_moment( p, R, 0, 1 );
            Vector<TF,D> x = center;
            base.add_value_grad_at( grad_dist, x, g * ( P0 - Px - Py ) );
            x[ 0 ] += 1;
            base.add_value_grad_at( grad_dist, x, g * Px );
            x[ 0 ] -= 1;
            x[ 1 ] += 1;
            base.add_value_grad_at( grad_dist, x, g * Py );
        }

        const Affine f = about_center();
        const TF x0 = TF( pts[ 0 ][ 0 ] ), y0 = TF( pts[ 0 ][ 1 ] );
        const TF s = ( TF( pts[ 1 ][ 0 ] ) - x0 ) * ( TF( pts[ 2 ][ 1 ] ) - y0 ) - ( TF( pts[ 1 ][ 1 ] ) - y0 ) * ( TF( pts[ 2 ][ 0 ] ) - x0 ) < 0 ? TF( -1 ) : TF( 1 );
        const TF gn[ 3 ] = { TF( 0.5 ) - TF( 0.3872983346207417 ), TF( 0.5 ), TF( 0.5 ) + TF( 0.3872983346207417 ) };
        const TF gw[ 3 ] = { TF( 5 ) / 18, TF( 8 ) / 18, TF( 5 ) / 18 };
        for ( int i = 0; i < 3; ++i ) {
            const int j = ( i + 1 ) % 3;
            const TF ax = TF( pts[ i ][ 0 ] ) - center[ 0 ], ay = TF( pts[ i ][ 1 ] ) - center[ 1 ];
            const TF bx = TF( pts[ j ][ 0 ] ) - center[ 0 ], by = TF( pts[ j ][ 1 ] ) - center[ 1 ];
            TF t0, t1;
            if ( ! clip_segment( ax, ay, bx, by, r2, t0, t1 ) )
                continue;
            TF wa = 0, wb = 0;                           // `int p rho ( 1 - t ) dt`, `int p rho t dt` on `[ t0, t1 ]`
            for ( int q = 0; q < 3; ++q ) {
                const TF t = t0 + ( t1 - t0 ) * gn[ q ];
                const TF yx = ax + t * ( bx - ax ), yy = ay + t * ( by - ay );
                const TF v = gw[ q ] * ( t1 - t0 ) * value( p, yx, yy ) * ( f.r0 + f.gx * yx + f.gy * yy );
                wa += v * ( 1 - t );
                wb += v * t;
            }
            const TF nx = s * ( by - ay ), ny = - s * ( bx - ax );           // the outward normal times the length
            grad_pts[ i ][ 0 ] += g * nx * wa;
            grad_pts[ i ][ 1 ] += g * ny * wa;
            grad_pts[ j ][ 0 ] += g * nx * wb;
            grad_pts[ j ][ 1 ] += g * ny * wb;
        }
    }

    HD void integrate_over_simplex_bwd( const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const {
        simplex_bwd( Poly{}, simplex_moments( pts ), pts, g, grad_pts, grad_dist );
    }

    template<int K>
    HD void integrate_moment_over_simplex_bwd( const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const {
        if constexpr ( K == D + 2 )
            sphere_length_bwd( pts, g, grad_pts );
        else
            simplex_bwd( poly<K>(), simplex_moments( pts ), pts, g, grad_pts, grad_dist );
    }

    /// the arcs of the sphere inside the piece `pc` ( those of the sectors of `add_edge_in_disc` ), oriented: `int cos^a sin^b`
    HD ArcTrig<TF> arcs( const auto &pc ) const {
        ArcTrig<TF> at;
        const int n = pc.nb_vertices();
        if ( n < 3 )
            return at;
        TF ar = 0;
        for ( int i = 0; i < n; ++i ) {
            const int j = i + 1 < n ? i + 1 : 0;
            ar += TF( pc.coord( i, 0 ) ) * TF( pc.coord( j, 1 ) ) - TF( pc.coord( j, 0 ) ) * TF( pc.coord( i, 1 ) );
        }
        const TF s = ar < 0 ? TF( -1 ) : TF( 1 );
        for ( int i = 0; i < n; ++i ) {
            const int j = i + 1 < n ? i + 1 : 0;
            const TF ax = TF( pc.coord( i, 0 ) ) - center[ 0 ], ay = TF( pc.coord( i, 1 ) ) - center[ 1 ];
            const TF bx = TF( pc.coord( j, 0 ) ) - center[ 0 ], by = TF( pc.coord( j, 1 ) ) - center[ 1 ];
            TF t0, t1;
            if ( ! clip_segment( ax, ay, bx, by, r2, t0, t1 ) ) {
                at.add_between( ax, ay, bx, by, s );
                continue;
            }
            if ( t0 > 0 ) at.add_between( ax, ay, ax + t0 * ( bx - ax ), ay + t0 * ( by - ay ), s );
            if ( t1 < 1 ) at.add_between( ax + t1 * ( bx - ax ), ay + t1 * ( by - ay ), bx, by, s );
        }
        return at;
    }

    /// the integrand depends on the seed of the cell: `seed_bwd` gives the cotangent of the integral of `p rho` over the piece on
    /// the seed's position ( the ball's center: `int_{arc} p rho n ds`, the arc of the sphere inside the piece ) and on its weight
    /// ( `r2 = w - w_s`: `int_{arc} p rho ds / ( 2 r )` ). On the sphere, `y = r e`, `p rho` is a polynomial of degree 3 in `e`.
    static constexpr bool depends_on_seed = true;

    HD void seed_bwd_poly( const Poly &p, const auto &pc, TF g, TF *gp, TF &gw ) const {
        if ( pc.nb_vertices() < 3 )
            return;
        const ArcTrig<TF> at = arcs( pc );
        const auto &J = at.J;
        const TF r = sdot::sqrt( r2 );
        const Affine f = about_center();
        // `p rho` on the sphere: `c0 + cx e_x + cy e_y + cxx e_x^2 + cxy e_x e_y + cyy e_y^2`
        const TF A = p.alpha + p.gamma * r2;
        const TF c0 = A * f.r0;
        const TF cx = r * ( A * f.gx + f.r0 * p.bx ), cy = r * ( A * f.gy + f.r0 * p.by );
        const TF cxx = r2 * p.bx * f.gx, cyy = r2 * p.by * f.gy, cxy = r2 * ( p.bx * f.gy + p.by * f.gx );
        const TF nx = c0 * J[ 1 ][ 0 ] + cx * J[ 2 ][ 0 ] + cy * J[ 1 ][ 1 ] + cxx * J[ 3 ][ 0 ] + cxy * J[ 2 ][ 1 ] + cyy * J[ 1 ][ 2 ];
        const TF ny = c0 * J[ 0 ][ 1 ] + cx * J[ 1 ][ 1 ] + cy * J[ 0 ][ 2 ] + cxx * J[ 2 ][ 1 ] + cxy * J[ 1 ][ 2 ] + cyy * J[ 0 ][ 3 ];
        const TF ln = c0 * J[ 0 ][ 0 ] + cx * J[ 1 ][ 0 ] + cy * J[ 0 ][ 1 ] + cxx * J[ 2 ][ 0 ] + cxy * J[ 1 ][ 1 ] + cyy * J[ 0 ][ 2 ];
        gp[ 0 ] += g * r * nx;
        gp[ 1 ] += g * r * ny;
        gw += g * ln / 2;
    }

    HD void seed_bwd( const auto &pc, TF g, TF *gp, TF &gw ) const { seed_bwd_poly( Poly{}, pc, g, gp, gw ); }

    template<int K>
    HD void seed_bwd_moment( const auto &pc, TF g, TF *gp, TF &gw ) const {
        if constexpr ( K == D + 2 ) sphere_length_seed_bwd( pc, g, gp, gw );
        else                        seed_bwd_poly( poly<K>(), pc, g, gp, gw );
    }

    /// `d mass / d r2` of the piece: what the sphere sweeps inside it when the radius grows, `int_{arc} rho ds / ( 2 r )`
    HD TF weight_mass( const auto &pc ) const {
        const ArcTrig<TF> at = arcs( pc );
        const Affine f = about_center();
        return ( f.r0 * at.J[ 0 ][ 0 ] + sdot::sqrt( r2 ) * ( f.gx * at.J[ 1 ][ 0 ] + f.gy * at.J[ 0 ][ 1 ] ) ) / 2;
    }
};

template<class TF> struct FaceInBall;                // ( `InBall3.h` )

/// a distribution whose facets are cut by itself ( `Mesh::for_each_facet_mass`: `cuts_facets` ) keeps the trait inside a ball
template<bool> struct InBallCutsFacets {};
template<> struct InBallCutsFacets<true> { static constexpr bool cuts_facets = true; };

template<class Dist>
HD constexpr bool dist_cuts_facets() {
    if constexpr ( requires { Dist::cuts_facets; } ) return bool( Dist::cuts_facets );
    else return false;
}

/// a distribution whose pieces are integrated inside the ball `|x - center|^2 <= r2` only. A piece entirely inside the
/// ball goes through unchanged ( its density stays as it is: the closed forms of the integrators apply ), one entirely
/// outside is not handed over at all.
template<class Dist,class TF,int D>
struct InBallDist : InBallCutsFacets<dist_cuts_facets<Dist>()> {
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

    /// the facets cut by the distribution itself ( `Mesh` ), each part of a facet integrated inside the ball only: in 2D the segment
    /// clipped by the sphere ( `InBallDensity::segment_mass` ), in 3D the convex polygon inter the disc the ball cuts in its plane
    /// ( `polygon_mass_in_ball`, a density constant on the piece )
    HD void for_each_facet_mass( const auto &cell, auto &&plane_of, auto &&func ) const requires ( dist_cuts_facets<Dist>() ) {
        if ( ! ( r2 > 0 ) )
            return;
        dist.for_each_facet_mass( cell, plane_of, func, [&]( const auto &A, const auto &B, const auto &dens ) -> TF {
            if constexpr ( D == 2 ) {
                const InBallDensity<DECAYED_TYPE_OF( dens ),TF,D> in{ dens, center, r2 };
                return in.segment_mass( TF( A[ 0 ] ) - center[ 0 ], TF( A[ 1 ] ) - center[ 1 ], TF( B[ 0 ] ) - center[ 0 ], TF( B[ 1 ] ) - center[ 1 ] );
            } else {
                static_assert( DECAYED_TYPE_OF( dens )::is_constant, "integration in a ball, in 3D: only for a density constant on each piece" );
                return TF( dens.value ) * polygon_area_in_ball( A, B );
            }
        } );
    }

    /// 3D: the area of the convex polygon `poly[ 0 .. m )` inside the ball, a fan of triangles ( `FaceInBall::area` )
    HD TF polygon_area_in_ball( const auto &poly, int m ) const {
        TF P[ 3 ][ 3 ], a = 0;
        for ( int d = 0; d < 3; ++d )
            P[ 0 ][ d ] = TF( poly[ 0 ][ d ] ) - center[ d ];
        for ( int k = 1; k + 1 < m; ++k ) {
            for ( int d = 0; d < 3; ++d ) {
                P[ 1 ][ d ] = TF( poly[ k ][ d ] ) - center[ d ];
                P[ 2 ][ d ] = TF( poly[ k + 1 ][ d ] ) - center[ d ];
            }
            TF ab[ 3 ], ac[ 3 ], n[ 3 ];
            for ( int d = 0; d < 3; ++d ) { ab[ d ] = P[ 1 ][ d ] - P[ 0 ][ d ]; ac[ d ] = P[ 2 ][ d ] - P[ 0 ][ d ]; }
            n[ 0 ] = ab[ 1 ] * ac[ 2 ] - ab[ 2 ] * ac[ 1 ];      // ( the side does not change the area )
            n[ 1 ] = ab[ 2 ] * ac[ 0 ] - ab[ 0 ] * ac[ 2 ];
            n[ 2 ] = ab[ 0 ] * ac[ 1 ] - ab[ 1 ] * ac[ 0 ];
            a += FaceInBall<TF>( P[ 0 ], P[ 1 ], P[ 2 ], n, r2 ).area();
        }
        return a;
    }
};

template<class TF,int D>
HD auto in_ball( const auto &dist, const Vector<TF,D> &center, TF r2 ) {
    return InBallDist<DECAYED_TYPE_OF( dist ),TF,D>{ {}, dist, center, r2 };
}

} // namespace integration
} // namespace sdot

#include "InBall3.h"                                     // ( the 3D specialization of `InBallDensity` )
