#pragma once

// =====================================================================================
// INTEGRATING ONLY INSIDE A BALL, IN 3D ( see `InBall.h` for what it is for ): a tetrahedron, or a convex polyhedron, inter the
// ball `|x - c|^2 <= r2`, for a density CONSTANT on the piece.
//
// = The closed forms
//
// By the divergence theorem, a homogeneous polynomial `m` of degree `k` in `y = x - c` integrates over a region `R` as
// `int_R m = 1 / ( 3 + k ) int_{dR} m ( y . n )`. The boundary of `T inter B` is made of the faces of `T` inside the ball,
// where `y . n = h_f` ( the signed distance from `c` to the plane of the face ) is constant, and of the part of the sphere
// inside `T`, where `y . n = r`. So
//
//     int 1      = ( sum_f h_f |F_f inter D_f|                    + r^3 Omega ) / 3
//     int y      = ( sum_f h_f int_{F_f inter D_f} y              + r^4 E     ) / 4
//     int |y|^2  = ( sum_f h_f int_{F_f inter D_f} |y|^2          + r^5 Omega ) / 5
//
// with `D_f` the disc cut by the ball in the plane of face `f` ( center the foot `q_f` of `c`, radius `sqrt( r2 - h_f^2 )` ), and
// `Omega = int de`, `E = int e de`, the SOLID moments of the directions `e` for which `c + r e` lies in `T`. A face integral is
// the 2D polygon inter disc of `InBall.h` ( `DiscMoments`, in the plane of the face, about `q_f` ). The directions of the sphere
// inside `T` are, face by face, those under which the face is seen BEYOND the sphere: `Omega = sum_f sign( h_f ) Omega( cone over
// F_f minus D_f )` -- a ray from `c` leaves `T` once ( `c` inside ), or enters and leaves it ( `c` outside: the signs cancel what
// does not count ). The cone over `F minus D` is the cone over `F` ( a polyhedral cone: Van Oosterom - Strackee for `Omega`, the
// lateral faces for `E` ) minus the cone over `F inter D`, which the 2D decomposition about `q` cuts into triangles `( q, u, v )`
// ( polyhedral cones again ) and SECTORS of the disc, whose cones are pieces of circular cones ( closed forms in spherical
// coordinates about the normal ).
//
// The derivatives need one more solid moment, `T = int e e^T de` ( the sphere's share of a first moment, `seed_bwd` ). For a
// polyhedral cone `K` cut at radius 1, `div( x_j |x|^2 e_i ) = delta_ij |x|^2 + 2 x_i x_j` gives `T = ( Omega I - sum_lat
// n_lat int u dphi ) / 3`, `n_lat` the outward normal of a lateral face and `int u dphi` the integral of the unit direction over
// its planar sector.
// =====================================================================================

#include "InBall.h"

namespace sdot {
namespace integration {

/// the solid moments of a set of directions: `O = int de`, `E = int e de`, `T = int e e^T de`
template<class TF>
struct SolidMoments {
    TF O = 0, E[ 3 ] = {}, T[ 3 ][ 3 ] = {};

    HD void add( const SolidMoments &s, TF w ) {
        O += w * s.O;
        for ( int i = 0; i < 3; ++i ) {
            E[ i ] += w * s.E[ i ];
            for ( int j = 0; j < 3; ++j )
                T[ i ][ j ] += w * s.T[ i ][ j ];
        }
    }
};

template<class TF> HD TF dot3( const TF *a, const TF *b ) { return a[ 0 ] * b[ 0 ] + a[ 1 ] * b[ 1 ] + a[ 2 ] * b[ 2 ]; }
template<class TF> HD void cross3( const TF *a, const TF *b, TF *r ) {
    r[ 0 ] = a[ 1 ] * b[ 2 ] - a[ 2 ] * b[ 1 ];
    r[ 1 ] = a[ 2 ] * b[ 0 ] - a[ 0 ] * b[ 2 ];
    r[ 2 ] = a[ 0 ] * b[ 1 ] - a[ 1 ] * b[ 0 ];
}
template<class TF> HD TF norm3( const TF *a ) { return sdot::sqrt( dot3( a, a ) ); }

/// the cone over the triangle `( a, b, c )` ( vectors from the apex ), UNSIGNED: Van Oosterom - Strackee for `Omega`; `E` and `T`
/// from its three lateral faces
template<class TF>
HD SolidMoments<TF> solid_triangle( const TF *a, const TF *b_, const TF *c_ ) {
    SolidMoments<TF> res;
    TF bc[ 3 ];
    cross3( b_, c_, bc );
    const TF det = dot3( a, bc );
    const TF la = norm3( a ), lb = norm3( b_ ), lc = norm3( c_ );
    if ( ! ( la > 0 && lb > 0 && lc > 0 ) || det == 0 )
        return res;
    const TF den = la * lb * lc + dot3( a, b_ ) * lc + dot3( a, c_ ) * lb + dot3( b_, c_ ) * la;
    res.O = 2 * TF( std::atan2( det < 0 ? - det : det, den ) );
    // `( a, b, c )` counterclockwise seen from the apex: the normals `p x q` of the lateral faces point INTO the cone
    const TF *V[ 3 ] = { a, det < 0 ? c_ : b_, det < 0 ? b_ : c_ };
    for ( int i = 0; i < 3; ++i )
        res.T[ i ][ i ] = res.O;
    for ( int i = 0; i < 3; ++i ) {
        const TF *p = V[ i ], *q = V[ ( i + 1 ) % 3 ];
        TF n[ 3 ], pu[ 3 ], qu[ 3 ], perp[ 3 ], pq[ 3 ];
        cross3( p, q, n );
        const TF ln = norm3( n );
        if ( ! ( ln > 0 ) )
            continue;
        const TF lp = norm3( p ), lq = norm3( q );
        for ( int d = 0; d < 3; ++d ) { n[ d ] /= ln; pu[ d ] = p[ d ] / lp; qu[ d ] = q[ d ] / lq; }
        cross3( pu, qu, pq );
        const TF beta = TF( std::atan2( norm3( pq ), dot3( pu, qu ) ) );
        cross3( n, pu, perp );                           // in the plane of the face, from `pu` towards `qu`
        const TF sb = TF( std::sin( beta ) ), cb1 = 1 - TF( std::cos( beta ) );
        for ( int d = 0; d < 3; ++d ) {
            res.E[ d ] += beta / 2 * n[ d ];
            const TF U = sb * pu[ d ] + cb1 * perp[ d ];  // `int u dphi` over the sector
            for ( int e = 0; e < 3; ++e )
                res.T[ e ][ d ] += n[ e ] * U;           // `- n_out ( x ) U`, `n_out = - n`
        }
    }
    for ( int i = 0; i < 3; ++i )
        for ( int j = 0; j < i; ++j )
            res.T[ i ][ j ] = res.T[ j ][ i ] = ( res.T[ i ][ j ] + res.T[ j ][ i ] ) / 2;
    for ( int i = 0; i < 3; ++i )
        for ( int j = 0; j < 3; ++j )
            res.T[ i ][ j ] /= 3;
    return res;
}

/// the circular cone of axis `ax` and half-angle `alpha` ( `cos alpha = ca` ), between the azimuths `th0` and `th0 + dth` measured
/// in the basis `( e1, e2 )` of its base -- SIGNED by the sign of `dth`
template<class TF>
HD SolidMoments<TF> solid_sector( const TF *ax, const TF *e1, const TF *e2, TF ca, TF th0, TF dth ) {
    SolidMoments<TF> res;
    const TF sa2 = 1 - ca * ca, sa = sa2 > 0 ? sdot::sqrt( sa2 ) : TF( 0 ), alpha = TF( std::atan2( sa, ca ) );
    const TF th1 = th0 + dth;
    const TF c0 = TF( std::cos( th0 ) ), s0 = TF( std::sin( th0 ) ), c1 = TF( std::cos( th1 ) ), s1 = TF( std::sin( th1 ) );
    const TF ic = s1 - s0, is = c0 - c1;                 // `int cos`, `int sin`
    const TF icc = ( dth + ( s1 * c1 - s0 * c0 ) ) / 2, iss = ( dth - ( s1 * c1 - s0 * c0 ) ) / 2, ics = ( s1 * s1 - s0 * s0 ) / 2;
    // the polar integrals over `[ 0, alpha ]`
    const TF i_s2 = ( alpha - sa * ca ) / 2, i_c2s = ( 1 - ca * ca * ca ) / 3, i_cs2 = sa2 * sa / 3, i_s3 = TF( 2 ) / 3 - ca + ca * ca * ca / 3;
    res.O = dth * ( 1 - ca );
    TF w[ 3 ];
    for ( int d = 0; d < 3; ++d ) {
        w[ d ] = ic * e1[ d ] + is * e2[ d ];
        res.E[ d ] = dth * sa2 / 2 * ax[ d ] + i_s2 * w[ d ];
    }
    for ( int i = 0; i < 3; ++i )
        for ( int j = 0; j < 3; ++j )
            res.T[ i ][ j ] = i_c2s * dth * ax[ i ] * ax[ j ] + i_cs2 * ( ax[ i ] * w[ j ] + w[ i ] * ax[ j ] )
                            + i_s3 * ( icc * e1[ i ] * e1[ j ] + iss * e2[ i ] * e2[ j ] + ics * ( e1[ i ] * e2[ j ] + e2[ i ] * e1[ j ] ) );
    return res;
}

/// A TRIANGLE OF A FACE seen from the center of the ball: its plane, the disc the ball cuts there, the moments of the triangle
/// inter the disc ( in the plane, about the foot `q` ) and the solid moments of the cone over the triangle MINUS the disc
template<class TF>
struct FaceInBall {
    TF   n[ 3 ];                                         ///< the outward unit normal
    TF   h;                                              ///< `( x - c ) . n` on the plane: the signed distance
    TF   q[ 3 ];                                         ///< the foot of the center, from the center: `h n`
    TF   e1[ 3 ], e2[ 3 ];                               ///< a basis of the plane
    TF   tri[ 3 ][ 2 ];                                  ///< the triangle in that basis, about the foot
    TF   s;                                              ///< the orientation of `tri`: +1, counterclockwise ( `e2` chosen so )
    bool disc = false;                                   ///< the plane cuts the ball
    TF   rho = 0;                                        ///< the radius of the disc
    DiscMoments<TF>  R;                                  ///< `int u^a v^b` over triangle inter disc ( unsigned )
    ArcTrig<TF>      arcs;                               ///< the arcs of the circle inside the triangle ( `int cos^a sin^b`, about the foot )
    SolidMoments<TF> S;                                  ///< the cone over triangle minus disc, times `sign( h )`

    /// `A`, `B`, `C`: the vertices, from the center; `out`: a direction pointing out of the solid the face bounds
    HD FaceInBall( const TF *A, const TF *B, const TF *C, const TF *out, TF r2 ) {
        TF ab[ 3 ], ac[ 3 ];
        for ( int d = 0; d < 3; ++d ) { ab[ d ] = B[ d ] - A[ d ]; ac[ d ] = C[ d ] - A[ d ]; }
        cross3( ab, ac, n );
        const TF ln = norm3( n );
        h = 0;
        if ( ! ( ln > 0 ) ) {
            n[ 0 ] = n[ 1 ] = n[ 2 ] = 0;
            q[ 0 ] = q[ 1 ] = q[ 2 ] = 0;
            return;
        }
        const TF so = dot3( n, out ) < 0 ? TF( -1 ) : TF( 1 );
        for ( int d = 0; d < 3; ++d )
            n[ d ] *= so / ln;
        h = dot3( A, n );
        for ( int d = 0; d < 3; ++d )
            q[ d ] = h * n[ d ];

        // a basis of the plane: `e1` along the largest of the edges from the foot
        const TF *P[ 3 ] = { A, B, C };
        TF best = -1;
        for ( int k = 0; k < 3; ++k ) {
            TF w[ 3 ];
            for ( int d = 0; d < 3; ++d ) w[ d ] = P[ k ][ d ] - q[ d ];
            const TF l = norm3( w );
            if ( l > best ) { best = l; for ( int d = 0; d < 3; ++d ) e1[ d ] = w[ d ] / l; }
        }
        cross3( n, e1, e2 );
        for ( int k = 0; k < 3; ++k ) {
            TF w[ 3 ];
            for ( int d = 0; d < 3; ++d ) w[ d ] = P[ k ][ d ] - q[ d ];
            tri[ k ][ 0 ] = dot3( w, e1 );
            tri[ k ][ 1 ] = dot3( w, e2 );
        }
        // counterclockwise in `( e1, e2 )`, flipping `e2` if needed: the sectors of `DiscMoments` integrate their first moments
        // over the right angular range only then ( `add_between` turns from `u` by `s` times the angle )
        if ( ( tri[ 1 ][ 0 ] - tri[ 0 ][ 0 ] ) * ( tri[ 2 ][ 1 ] - tri[ 0 ][ 1 ] ) - ( tri[ 1 ][ 1 ] - tri[ 0 ][ 1 ] ) * ( tri[ 2 ][ 0 ] - tri[ 0 ][ 0 ] ) < 0 )
            for ( int k = 0; k < 3; ++k ) {
                e2[ k ] = - e2[ k ];
                tri[ k ][ 1 ] = - tri[ k ][ 1 ];
            }
        s = 1;

        // the triangle inter the disc, in the plane ( even through the center: the hessian and the adjoint need its area )
        const TF rho2 = r2 - h * h;
        if ( rho2 > 0 ) {
            disc = true;
            rho = sdot::sqrt( rho2 );
            for ( int i = 0; i < 3; ++i ) {
                const int j = ( i + 1 ) % 3;
                const TF ax_ = tri[ i ][ 0 ], ay_ = tri[ i ][ 1 ], bx_ = tri[ j ][ 0 ], by_ = tri[ j ][ 1 ];
                add_edge_in_disc( R, ax_, ay_, bx_, by_, rho2, s );
                TF t0, t1;                               // ( the arcs: as `InBallDensity::triangle_arcs` )
                if ( ! clip_segment( ax_, ay_, bx_, by_, rho2, t0, t1 ) )
                    arcs.add_between( ax_, ay_, bx_, by_, TF( 1 ) );
                else {
                    if ( t0 > 0 ) arcs.add_between( ax_, ay_, ax_ + t0 * ( bx_ - ax_ ), ay_ + t0 * ( by_ - ay_ ), TF( 1 ) );
                    if ( t1 < 1 ) arcs.add_between( ax_ + t1 * ( bx_ - ax_ ), ay_ + t1 * ( by_ - ay_ ), bx_, by_, TF( 1 ) );
                }
            }
        }
        if ( h == 0 )
            return;                                      // a plane through the center: a cone of no solid angle

        // the cone over the whole triangle, then minus the part over the disc
        const TF sh = h < 0 ? TF( -1 ) : TF( 1 );
        S.add( solid_triangle( A, B, C ), sh );
        if ( ! disc )
            return;
        const TF ca = ( h < 0 ? - h : h ) / sdot::sqrt( r2 );
        TF ax[ 3 ];                                      // the axis of the cone: towards the plane
        for ( int d = 0; d < 3; ++d ) ax[ d ] = sh * n[ d ];
        // the 2D decomposition about `q` ( `add_edge_in_disc` ), mirrored in solid moments ( the handedness of `( e1, e2, ax )`
        // does not matter to `solid_sector`, signed by its azimuth range )
        for ( int i = 0; i < 3; ++i ) {
            const int j = ( i + 1 ) % 3;
            const TF ax_ = tri[ i ][ 0 ], ay_ = tri[ i ][ 1 ], bx_ = tri[ j ][ 0 ], by_ = tri[ j ][ 1 ];
            auto sector = [&]( TF ux, TF uy, TF vx, TF vy ) {
                if ( ! ( ux * ux + uy * uy > 0 ) || ! ( vx * vx + vy * vy > 0 ) )
                    return;
                const TF t0 = TF( std::atan2( uy, ux ) ), dt = TF( std::atan2( ux * vy - uy * vx, ux * vx + uy * vy ) );
                S.add( solid_sector( ax, e1, e2, ca, t0, dt ), - sh * s );
            };
            auto triangle = [&]( TF ux, TF uy, TF vx, TF vy ) {
                const TF cr = ux * vy - uy * vx;
                if ( cr == 0 )
                    return;
                TF U[ 3 ], V[ 3 ];
                for ( int d = 0; d < 3; ++d ) { U[ d ] = q[ d ] + ux * e1[ d ] + uy * e2[ d ]; V[ d ] = q[ d ] + vx * e1[ d ] + vy * e2[ d ]; }
                S.add( solid_triangle( q, U, V ), - sh * s * ( cr < 0 ? TF( -1 ) : TF( 1 ) ) );
            };
            TF t0, t1;
            if ( ! clip_segment( ax_, ay_, bx_, by_, rho2, t0, t1 ) ) {
                sector( ax_, ay_, bx_, by_ );
                continue;
            }
            const TF dx = bx_ - ax_, dy = by_ - ay_;
            const TF px = ax_ + t0 * dx, py = ay_ + t0 * dy, qx = ax_ + t1 * dx, qy = ay_ + t1 * dy;
            if ( t0 > 0 ) sector( ax_, ay_, px, py );
            triangle( px, py, qx, qy );
            if ( t1 < 1 ) sector( qx, qy, bx_, by_ );
        }
    }

    /// the area of the triangle inside the ball
    HD TF area() const { return disc ? R.M[ 0 ][ 0 ] : TF( 0 ); }
};

/// the integrals of `1`, `y` and `|y|^2` ( `y = x - c` ) over a solid inter the ball, from its faces
template<class TF>
struct BallTerms {
    TF a = 0, ay[ 3 ] = {}, a2 = 0;                      ///< `sum_f h_f int_{F inter D}` of `1`, `y`, `|y|^2`
    SolidMoments<TF> S;

    HD void add_face( const FaceInBall<TF> &f ) {
        S.add( f.S, TF( 1 ) );
        if ( ! f.disc )
            return;
        const auto &M = f.R.M;
        a += f.h * M[ 0 ][ 0 ];
        for ( int d = 0; d < 3; ++d )
            ay[ d ] += f.h * ( f.q[ d ] * M[ 0 ][ 0 ] + f.e1[ d ] * M[ 1 ][ 0 ] + f.e2[ d ] * M[ 0 ][ 1 ] );
        a2 += f.h * ( f.h * f.h * M[ 0 ][ 0 ] + M[ 2 ][ 0 ] + M[ 0 ][ 2 ] );
    }

    HD TF vol( TF r ) const { return ( a + r * r * r * S.O ) / 3; }
    HD TF first( int d, TF r ) const { const TF r2 = r * r; return ( ay[ d ] + r2 * r2 * S.E[ d ] ) / 4; }
    HD TF second( TF r ) const { const TF r2 = r * r; return ( a2 + r2 * r2 * r * S.O ) / 5; }
};


/// `InBallDensity` in 3D ( see `InBall.h` ): the density CONSTANT on the piece
template<class Dens,class TF_>
struct InBallDensity<Dens,TF_,3> {
    using TF = TF_;
    static constexpr int D = 3;
    static constexpr int ct_dim = 3;
    static constexpr bool is_constant = false;
    static constexpr bool exact_moments = true;
    static constexpr bool depends_on_seed = true;
    static_assert( Dens::is_constant, "integration in a ball, in 3D: only for a density constant on each piece ( `Image`, `Box`, "
                                      "`ConvexHull` )" );

    Dens          base;
    Vector<TF,D>  center;
    TF            r2;

    /// the polynomial of a moment about the center: `alpha + beta . y + gamma |y|^2`
    struct Poly { TF alpha = 1, beta[ 3 ] = {}, gamma = 0; };

    template<int K>
    HD Poly poly() const {
        Poly p;
        if constexpr ( K < D ) {
            p.alpha = center[ K ];
            p.beta[ K ] = 1;
        } else if constexpr ( K == D ) {
            p.alpha = center[ 0 ] * center[ 0 ] + center[ 1 ] * center[ 1 ] + center[ 2 ] * center[ 2 ];
            for ( int d = 0; d < 3; ++d )
                p.beta[ d ] = 2 * center[ d ];
            p.gamma = 1;
        }
        return p;
    }

    HD TF rho() const { return TF( base.value ); }
    HD TF radius() const { return sdot::sqrt( r2 ); }

    /// the four faces of the tetrahedron `pts`, as seen from the center
    HD void tet_faces( const auto &pts, auto &&func ) const {
        TF P[ 4 ][ 3 ], g[ 3 ] = { 0, 0, 0 };
        for ( int k = 0; k < 4; ++k )
            for ( int d = 0; d < 3; ++d ) {
                P[ k ][ d ] = TF( pts[ k ][ d ] ) - center[ d ];
                g[ d ] += P[ k ][ d ] / 4;
            }
        static constexpr int F[ 4 ][ 3 ] = { { 1, 2, 3 }, { 0, 2, 3 }, { 0, 1, 3 }, { 0, 1, 2 } };
        for ( int f = 0; f < 4; ++f ) {
            TF out[ 3 ];
            for ( int d = 0; d < 3; ++d )
                out[ d ] = P[ F[ f ][ 0 ] ][ d ] - g[ d ];
            func( FaceInBall<TF>( P[ F[ f ][ 0 ] ], P[ F[ f ][ 1 ] ], P[ F[ f ][ 2 ] ], out, r2 ), F[ f ] );
        }
    }

    HD BallTerms<TF> tet_terms( const auto &pts ) const {
        BallTerms<TF> bt;
        tet_faces( pts, [&]( const FaceInBall<TF> &f, const int * ) { bt.add_face( f ); } );
        return bt;
    }

    HD TF integral( const Poly &p, const BallTerms<TF> &bt ) const {
        const TF r = radius();
        TF res = p.alpha * bt.vol( r ) + p.gamma * bt.second( r );
        for ( int d = 0; d < 3; ++d )
            res += p.beta[ d ] * bt.first( d, r );
        return rho() * res;
    }

    HD TF integrate_over_simplex( const auto &pts ) const { return integral( Poly{}, tet_terms( pts ) ); }

    HD void integrate_moments_over_simplex( const auto &pts, TF &m, auto &mx, TF &m2 ) const {
        const auto bt = tet_terms( pts );
        m += integral( Poly{}, bt );
        mx[ 0 ] += integral( poly<0>(), bt );
        mx[ 1 ] += integral( poly<1>(), bt );
        mx[ 2 ] += integral( poly<2>(), bt );
        m2 += integral( poly<D>(), bt );
    }

    /// `K == D + 2`: THE SPHERE, the area of the sphere inside the tetrahedron where the density is not zero ( `sphere_area` )
    template<int K>
    HD TF integrate_moment_over_simplex( const auto &pts ) const {
        if constexpr ( K == D + 2 ) return sphere_area( pts );
        else                        return integral( poly<K>(), tet_terms( pts ) );
    }

    // ---- THE SPHERE inside a piece: `|S inter T| = r^2 Omega`, what a surface energy counts ( `PowerDiagram.sphere_areas` ) ---------
    //
    // Its derivatives are geometric. A face pushed outward by `delta` ( a field on its plane ) widens the cap by `int_arc delta r / rho`,
    // the arc being the circle of the face inside it and `rho` its radius ( a zone of the sphere: `2 pi r` of area per unit of height,
    // `2 pi rho` of length ); moving the center by `dc` is moving the faces by `- dc . n`; the radius scales the cap and moves the
    // arcs: `d A / d r = 2 A / r - sum_f L_f h_f / rho_f`.

    HD TF on_support() const { return TF( base.value ) > 0 ? TF( 1 ) : TF( 0 ); }

    HD TF sphere_area( const auto &pts ) const { return on_support() * r2 * tet_terms( pts ).S.O; }

    /// its adjoint with respect to the vertices of the tetrahedron: `int_arc lambda_k ds` on each face, `lambda_k` affine
    HD void sphere_area_bwd( const auto &pts, TF g, auto &&grad_pts ) const {
        const TF r = radius(), f = g * on_support();
        if ( f == 0 )
            return;
        tet_faces( pts, [&]( const FaceInBall<TF> &fc, const int *vk ) {
            if ( ! fc.disc || ! ( fc.rho > 0 ) )
                return;
            const auto &J = fc.arcs.J;
            const TF ar = ( fc.tri[ 1 ][ 0 ] - fc.tri[ 0 ][ 0 ] ) * ( fc.tri[ 2 ][ 1 ] - fc.tri[ 0 ][ 1 ] ) - ( fc.tri[ 1 ][ 1 ] - fc.tri[ 0 ][ 1 ] ) * ( fc.tri[ 2 ][ 0 ] - fc.tri[ 0 ][ 0 ] );
            if ( ar == 0 )
                return;
            for ( int k = 0; k < 3; ++k ) {
                const int i = ( k + 1 ) % 3, j = ( k + 2 ) % 3;
                const TF lb = ( fc.tri[ i ][ 1 ] - fc.tri[ j ][ 1 ] ) / ar, lc = ( fc.tri[ j ][ 0 ] - fc.tri[ i ][ 0 ] ) / ar;
                const TF la = ( fc.tri[ i ][ 0 ] * fc.tri[ j ][ 1 ] - fc.tri[ j ][ 0 ] * fc.tri[ i ][ 1 ] ) / ar;
                // `int_arc lambda_k ds`, `u = rho cos`, `v = rho sin`, `ds = rho d theta`
                const TF w = fc.rho * ( la * J[ 0 ][ 0 ] + fc.rho * ( lb * J[ 1 ][ 0 ] + lc * J[ 0 ][ 1 ] ) );
                for ( int d = 0; d < 3; ++d )
                    grad_pts[ vk[ k ] ][ d ] += f * r / fc.rho * w * fc.n[ d ];
            }
        } );
    }

    /// ... and with respect to the seed, over the faces of the piece `pc`
    HD void sphere_area_seed_bwd( const auto &pc, TF g, TF *gp, TF &gw ) const {
        const TF r = radius(), f = g * on_support();
        if ( f == 0 || pc.nb_vertices() < 4 )
            return;
        TF O = 0, ends = 0;
        piece_faces( pc, -1, [&]( const FaceInBall<TF> &fc ) {
            O += fc.S.O;
            if ( ! fc.disc || ! ( fc.rho > 0 ) )
                return;
            const TF L = fc.rho * fc.arcs.J[ 0 ][ 0 ];
            for ( int d = 0; d < 3; ++d )
                gp[ d ] -= f * r / fc.rho * L * fc.n[ d ];
            ends += L * fc.h / fc.rho;
        } );
        gw += f * ( O - ends / ( 2 * r ) );            // `A / r2 - sum L h / ( 2 r rho )`, `A = r2 O`
    }

    /// THE ADJOINT with respect to the vertices of the tetrahedron: each face sweeps, inside the ball, `int_{F inter D} p rho n .
    /// sum_k lambda_k dv_k` -- in the plane, about the foot, `p` is of degree 2 and `lambda_k` affine: the cubic moments of the
    /// triangle inter disc. With respect to the density: `int p` over the tetrahedron inter the ball.
    HD void simplex_bwd( const Poly &p, const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const {
        BallTerms<TF> bt;
        const TF ro = rho();
        tet_faces( pts, [&]( const FaceInBall<TF> &f, const int *vk ) {
            bt.add_face( f );
            if ( ! f.disc )
                return;
            // `p` on the plane, in `( u, v )` about the foot: `c00 + c10 u + c01 v + gamma ( u^2 + v^2 )`
            const TF c00 = p.alpha + dot3( p.beta, f.q ) + p.gamma * f.h * f.h;
            const TF c10 = dot3( p.beta, f.e1 ), c01 = dot3( p.beta, f.e2 );
            const auto &M = f.R.M;
            const TF ar = ( f.tri[ 1 ][ 0 ] - f.tri[ 0 ][ 0 ] ) * ( f.tri[ 2 ][ 1 ] - f.tri[ 0 ][ 1 ] ) - ( f.tri[ 1 ][ 1 ] - f.tri[ 0 ][ 1 ] ) * ( f.tri[ 2 ][ 0 ] - f.tri[ 0 ][ 0 ] );
            if ( ar == 0 )
                return;
            for ( int k = 0; k < 3; ++k ) {
                // the barycentric coordinate of vertex `k`: `la + lb u + lc v`
                const int i = ( k + 1 ) % 3, j = ( k + 2 ) % 3;
                const TF lb = ( f.tri[ i ][ 1 ] - f.tri[ j ][ 1 ] ) / ar, lc = ( f.tri[ j ][ 0 ] - f.tri[ i ][ 0 ] ) / ar;
                const TF la = ( f.tri[ i ][ 0 ] * f.tri[ j ][ 1 ] - f.tri[ j ][ 0 ] * f.tri[ i ][ 1 ] ) / ar;
                // `int p lambda_k` over triangle inter disc
                const TF w = la * ( c00 * M[ 0 ][ 0 ] + c10 * M[ 1 ][ 0 ] + c01 * M[ 0 ][ 1 ] + p.gamma * ( M[ 2 ][ 0 ] + M[ 0 ][ 2 ] ) )
                           + lb * ( c00 * M[ 1 ][ 0 ] + c10 * M[ 2 ][ 0 ] + c01 * M[ 1 ][ 1 ] + p.gamma * ( M[ 3 ][ 0 ] + M[ 1 ][ 2 ] ) )
                           + lc * ( c00 * M[ 0 ][ 1 ] + c10 * M[ 1 ][ 1 ] + c01 * M[ 0 ][ 2 ] + p.gamma * ( M[ 2 ][ 1 ] + M[ 0 ][ 3 ] ) );
                for ( int d = 0; d < 3; ++d )
                    grad_pts[ vk[ k ] ][ d ] += g * ro * w * f.n[ d ];
            }
        } );
        const TF r = radius();
        TF P0 = p.alpha * bt.vol( r ) + p.gamma * bt.second( r );
        for ( int d = 0; d < 3; ++d )
            P0 += p.beta[ d ] * bt.first( d, r );
        base.add_value_grad( grad_dist, g * P0 );
    }

    HD void integrate_over_simplex_bwd( const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const {
        simplex_bwd( Poly{}, pts, g, grad_pts, grad_dist );
    }

    template<int K>
    HD void integrate_moment_over_simplex_bwd( const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const {
        if constexpr ( K == D + 2 ) sphere_area_bwd( pts, g, grad_pts );
        else                        simplex_bwd( poly<K>(), pts, g, grad_pts, grad_dist );
    }

    /// `func( FaceInBall )` for each triangle of the faces of the polyhedron `pc` ( their fans ), and of the face of cut `only` alone
    /// when `only >= 0`
    HD void piece_faces( const auto &pc, int only, auto &&func ) const {
        const int nv = pc.nb_vertices();
        if ( nv < 4 )
            return;
        TF g[ 3 ] = { 0, 0, 0 };
        for ( int i = 0; i < nv; ++i )
            for ( int d = 0; d < 3; ++d )
                g[ d ] += ( TF( pc.coord( i, d ) ) - center[ d ] ) / nv;
        auto face = [&]( int f ) {
            pc.for_each_facet_triangle( f, [&]( int o, int a, int b ) {
                TF P[ 3 ][ 3 ], out[ 3 ];
                const int ids[ 3 ] = { o, a, b };
                for ( int k = 0; k < 3; ++k )
                    for ( int d = 0; d < 3; ++d )
                        P[ k ][ d ] = TF( pc.coord( ids[ k ], d ) ) - center[ d ];
                for ( int d = 0; d < 3; ++d )
                    out[ d ] = P[ 0 ][ d ] - g[ d ];
                func( FaceInBall<TF>( P[ 0 ], P[ 1 ], P[ 2 ], out, r2 ) );
            } );
        };
        if ( only >= 0 )
            face( only );
        else
            for ( int f = 0; f < pc.nb_cuts(); ++f )
                face( f );
    }

    /// the solid moments of the sphere inside the piece
    HD SolidMoments<TF> sphere_in( const auto &pc ) const {
        SolidMoments<TF> S;
        piece_faces( pc, -1, [&]( const FaceInBall<TF> &f ) { S.add( f.S, TF( 1 ) ); } );
        return S;
    }

    /// the mass of the face of cut `cut` inside the ball ( the hessian )
    HD TF facet_mass( const auto &pc, int cut ) const {
        TF a = 0;
        piece_faces( pc, cut, [&]( const FaceInBall<TF> &f ) { a += f.area(); } );
        return rho() * a;
    }

    /// `d mass / d r2`: what the sphere sweeps inside the piece, `int_{S inter piece} rho dS / ( 2 r ) = rho r Omega / 2`
    HD TF weight_mass( const auto &pc ) const {
        return rho() * radius() * sphere_in( pc ).O / 2;
    }

    /// the cotangents of the integral of `p rho` over the piece on the seed: its center, `int_{S inter piece} p rho n dS`, and its
    /// weight, `int_{S inter piece} p rho dS / ( 2 r )`. On the sphere `y = r e`: `p = alpha + gamma r^2 + r beta . e`
    HD void seed_bwd_poly( const Poly &p, const auto &pc, TF g, TF *gp, TF &gw ) const {
        if ( pc.nb_vertices() < 4 )
            return;
        const SolidMoments<TF> S = sphere_in( pc );
        const TF r = radius(), ro = rho(), A = p.alpha + p.gamma * r2;
        for ( int d = 0; d < 3; ++d ) {
            TF tb = 0;
            for ( int e = 0; e < 3; ++e )
                tb += S.T[ d ][ e ] * p.beta[ e ];
            gp[ d ] += g * ro * r2 * ( A * S.E[ d ] + r * tb );
        }
        gw += g * ro * r / 2 * ( A * S.O + r * dot3( p.beta, S.E ) );
    }

    HD void seed_bwd( const auto &pc, TF g, TF *gp, TF &gw ) const { seed_bwd_poly( Poly{}, pc, g, gp, gw ); }

    template<int K>
    HD void seed_bwd_moment( const auto &pc, TF g, TF *gp, TF &gw ) const {
        if constexpr ( K == D + 2 ) sphere_area_seed_bwd( pc, g, gp, gw );
        else                        seed_bwd_poly( poly<K>(), pc, g, gp, gw );
    }
};

} // namespace integration
} // namespace sdot
