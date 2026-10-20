#pragma once

// =====================================================================================
// THE TETRAHEDRAL MESH DENSITY OF THE CARD'S 3D CELLS ( `Mesh`, DG1: a value per corner of each tetrahedron, affine on it ):
// what `Cell3D.cuh::finish_warp` integrates on a cell -- the mass, the integral on each facet ( the laplacian's `c_ij` ) and
// the moments. The 2D sibling is `Density2D.cuh::DensMesh`.
//
// = A CELL AGAINST THE MESH: CONES ON THE BOUNDARY
//
// `int_{ C INTERSECT T } f`, for the cell `C` and an element `T` ( `f` the affine function of `T`, over all the space ), is the sum
// over the faces of `C INTERSECT T` of the signed cones from ONE apex, `g` the mean of the cell's vertices -- and these faces are
//
//   * the faces of `T` clipped by the planes of `C` that cut it ( one to three: the element is classified first );
//   * on each of these planes, the SECTION of `T` ( a triangle or a quadrilateral ) clipped by the others -- a piece of the face
//     of `C`, whose integral is also the facet's share ( `mesh3_cell` ).
//
// The elements are those the box of the cell meets, shared between the lanes of the warp, each done ONCE; only convex polygons
// of a few vertices are clipped, never a polyhedron, and nothing is walked per face of the cell. ( Measured on 1e4 seeds and
// 1.3e4 elements: cutting the tetrahedra of the fans one by one, ~ 175 times the constant density's time; the faces of `C`
// clipped by the elements, each lane walking the hierarchy for its faces, ~ 60 times. )
//
// TWO FACES IN THE SAME PLANE ( a face of `C` on a face of `T` ): an element with no node strictly inside a plane that cuts it
// meets the cell on a flat piece, and is skipped -- of the two elements on both sides of the face, the one inside counts the
// facet. A face of `T` on a plane of `C` is dropped when their outward normals agree ( the section is that face already ).
//
// Every integral is in the SEED's frame ( the cell's ), the mesh's nodes brought to it at the element: the moments are about
// the seed, as the cells write them. `int lambda_a lambda_b lambda_c = ( 6, 2, 1 ) V / 120`, `int lambda_a lambda_b =
// ( 1 + [ a = b ] ) V / 20`.
//
// = THE GAUSSIANS ( `DensGauss3`, `gauss3_cell` ): CONES FROM THE CENTRE, CLOSED FORMS ON THE FACES
//
// For a gaussian of centre `c`, width `s` ( unit coordinates `( x - c ) / s` below ), the mass of a convex cell is the sum over its
// faces of the signed cones from `c`: a face at the signed distance `h` ( outward normal ) gives
//
//     h / ( 4 pi ) int_face Phi3( r ) / r^3 dA,        Phi3( r ) = P( |X| <= r ) = erf( r / sqrt 2 ) - sqrt( 2 / pi ) r exp( -r^2 / 2 )
//
// ( a cone from `c` is a solid angle times the radial law of `|X|` ). In polar coordinates about the foot of `c` on the face, the
// radial integral is closed: `Psi( R ) = int_h^R Phi3( r ) / r^2 dr = Phi3( h ) / h - Phi3( R ) / R + sqrt( 2 / pi ) ( exp( -h^2 / 2 )
// - exp( -R^2 / 2 ) )`, and what is left is ONE integral along each edge, `int |p| Psi( R ) / rho^2 dt` ( `p` the distance from the
// foot to the edge's line, `rho^2 = p^2 + t^2`, `R^2 = h^2 + rho^2` ) -- the 3D sibling of the 2D wedge, by Gauss-Legendre on its
// core and in closed form where `R >= 8`. `Psi / rho^2` is computed without cancellation near the foot: by its series in `R^2`
// for `R <= 2`, by a short quadrature of `Phi3 / r^2` on `[ h, R ]` when `R - h` is small.
//
// The rest are closed forms, as in 2D: on a face, `phi` is `phi1( h )` times a 2D gaussian, whose measure on the face is the sum
// of the wedges of its edges ( `Density2D.cuh::gauss_wedge_pt` ) -- the facet; `int ( x - c ) phi = -s^2 sum_f n_f int_f phi`,
// `int |x - c|^2 phi = 3 s^2 m - s^2 sum_f h_f int_f phi` ( the divergence of `( x - c ) phi` ). The mass of each gaussian is `| sum |`
// over the faces, as in 2D: a cell far from a narrow bump must not get a negative mass from the rounding of its cones.
//
// From 16 gaussians on, a cell sees only the terms of the CPU's tree ( `GaussianTree.h` ): the warp walks it ( `gauss3_fill` ), once
// for the strongest term, once to emit the terms above `eps` times it, 32 at a time -- groups merged at the width of the stage.
// =====================================================================================

#include "Density2D.cuh"

namespace sdot::gpu3d {

/// a DG1 density on a tetrahedral mesh ( `Mesh`, 3D ): the nodes, the values at the corners, the tetrahedra ( node indices ),
/// the gradients of their barycentric coordinates, the bounding volume hierarchy ( `distributions/Mesh.py` ). In doubles.
struct DensMesh3 {
    SI            nt = 0, nb = 0;                        ///< tetrahedra, hierarchy nodes
    const double *pn = nullptr;                          ///< `pn[ 3 q + c ]`: the nodes
    const double *val = nullptr;                         ///< `val[ 4 e + i ]`: the value of tetrahedron `e` at its corner `i`
    const int    *tet = nullptr;                         ///< `tet[ 4 e + i ]`: the node of corner `i`
    const double *grad = nullptr;                        ///< `grad[ 12 e + 3 i + c ]`: component `c` of the gradient of `lambda_i`
    const double *blo = nullptr, *bhi = nullptr;         ///< `[ nb, 3 ]`: the boxes of the hierarchy
    const int    *links = nullptr;                       ///< `[ nb, 4 ]`: left, right, begin, end ( a leaf: left < 0 )
};

/// what a cell accumulates: the mass, `int ( x - p ) rho`, `int |x - p|^2 rho`
struct DensSums3 {
    double m = 0, mx = 0, my = 0, mz = 0, m2 = 0;
};

/// a sum of isotropic gaussians ( `SumOfGaussians`, 3D ): `c[ 3 g + a ]` the centres, `s` the widths ( convolved: `sqrt( sigma^2 +
/// s^2 )` ), `w` the masses ( `gauss3_cell` ).
///
/// With `nn > 0`, THE TREE of the CPU ( `GaussianTree.h`, built by the host at `prepare`, the same parameters ): each cell sees only
/// the terms that weigh on it, groups merged at the continuation width `sqrt( s2 )` -- `c`, `s`, `w` are not read then.
struct DensGauss3 {
    static constexpr int ND = 14;                        ///< doubles per node: lo[ 3 ], hi[ 3 ], mu[ 3 ], mass, var, s_min, s_max, radius
    SI            nb = 0;
    const double *c = nullptr, *s = nullptr, *w = nullptr;
    SI            nn = 0;                                ///< the tree's nodes ( 0: no tree )
    const double *nd = nullptr;                          ///< `nd[ ND * k + . ]`: node `k`
    const int    *nl = nullptr;                          ///< `nl[ 4 * k + . ]`: begin, end ( in `it` ), left, right ( `-1`: a leaf )
    const double *it = nullptr;                          ///< `it[ 5 * i + . ]`: the gaussians in tree order: centre, own width, mass
    double        s2 = 0;                                ///< the continuation width, squared
    double        theta = 0.5, rel = 1e-6;               ///< `GaussianTree::theta`; `eps` ( `s2 > 0` ) or `eps_exact`
};


/// an affine function `c + G . ( x - x0 )` ( the seed's frame ): a barycentric coordinate, the density of an element, the
/// signed distance to a plane of the cell
struct Affine3 {
    double c, gx, gy, gz, x0, y0, z0;
    __device__ __forceinline__ double operator()( double x, double y, double z ) const { return c + gx * ( x - x0 ) + gy * ( y - y0 ) + gz * ( z - z0 ); }
};

/// a convex polygon ( 3D, the seed's frame ), at most `CAP` vertices ( past it, the vertices are dropped: a face of more than
/// `CAP - 4` vertices is cut in triangles by the caller )
template<int _CAP>
struct Poly3 {
    static constexpr int CAP = _CAP;
    int    n = 0;
    double x[ CAP ], y[ CAP ], z[ CAP ];
    __device__ __forceinline__ void push( double a, double b, double c ) { if ( n < CAP ) { x[ n ] = a; y[ n ] = b; z[ n ] = c; ++n; } }
};

/// how a polygon left a clip: cut ( or whole, or empty ), or LYING on the plane
enum { CLIP_DONE = 0, CLIP_ON = 1 };

/// `p` clipped by `h >= 0`, in place. A polygon whose vertices are all within `tol` of the plane is left as it is ( `CLIP_ON`:
/// the caller decides ); a vertex within `tol` is on the plane
template<class P>
__device__ __forceinline__ int clip3( P &p, const Affine3 &h, double tol ) {
    constexpr int CAP = P::CAP;
    double s[ CAP ];
    bool on = true, all_in = true;
    for ( int j = 0; j < p.n; ++j ) {
        const double v = h( p.x[ j ], p.y[ j ], p.z[ j ] );
        s[ j ] = fabs( v ) <= tol ? 0.0 : v;
        on = on && s[ j ] == 0;
        all_in = all_in && s[ j ] >= 0;
    }
    if ( on )
        return CLIP_ON;
    if ( all_in )
        return CLIP_DONE;
    // in place: the kept vertices and the crossings, written behind the reads ( a copy of the first vertex closes the loop )
    const double fx = p.x[ 0 ], fy = p.y[ 0 ], fz = p.z[ 0 ], fs = s[ 0 ];
    double ox[ CAP ], oy[ CAP ], oz[ CAP ];
    int m = 0;
    for ( int j = 0; j < p.n; ++j ) {
        const bool last = j + 1 == p.n;
        const double jx = p.x[ j ], jy = p.y[ j ], jz = p.z[ j ], sj = s[ j ];
        const double kx = last ? fx : p.x[ j + 1 ], ky = last ? fy : p.y[ j + 1 ], kz = last ? fz : p.z[ j + 1 ], sk = last ? fs : s[ j + 1 ];
        if ( sj >= 0 && m < CAP ) { ox[ m ] = jx; oy[ m ] = jy; oz[ m ] = jz; ++m; }
        if ( ( ( sj > 0 && sk < 0 ) || ( sj < 0 && sk > 0 ) ) && m < CAP ) {
            const double t = sj / ( sj - sk );
            ox[ m ] = jx + t * ( kx - jx ); oy[ m ] = jy + t * ( ky - jy ); oz[ m ] = jz + t * ( kz - jz );
            ++m;
        }
    }
    for ( int j = 0; j < m; ++j ) { p.x[ j ] = ox[ j ]; p.y[ j ] = oy[ j ]; p.z[ j ] = oz[ j ]; }
    p.n = m;
    return CLIP_DONE;
}

/// THE TETRAHEDRON `( X, Y, Z )[ 0 .. 3 ]` of signed volume `V`, the affine density `F` at its corners: its share of the mass
/// and of the moments into `acc`
template<bool MOM>
__device__ __forceinline__ void tet3( const double ( &X )[ 4 ], const double ( &Y )[ 4 ], const double ( &Z )[ 4 ], const double ( &F )[ 4 ], double V, DensSums3 &acc ) {
    const double sf = F[ 0 ] + F[ 1 ] + F[ 2 ] + F[ 3 ];
    acc.m += V * sf / 4;
    if constexpr ( MOM ) {
        for ( int a = 0; a < 4; ++a ) {
            acc.mx += V / 20 * X[ a ] * ( sf + F[ a ] );
            acc.my += V / 20 * Y[ a ] * ( sf + F[ a ] );
            acc.mz += V / 20 * Z[ a ] * ( sf + F[ a ] );
        }
        double s2 = 0;
        for ( int a = 0; a < 4; ++a )
            for ( int b = 0; b < 4; ++b ) {
                const double ab = X[ a ] * X[ b ] + Y[ a ] * Y[ b ] + Z[ a ] * Z[ b ];
                for ( int c = 0; c < 4; ++c ) {
                    const int same = ( a == b ) + ( b == c ) + ( a == c );
                    s2 += ab * F[ c ] * ( same == 3 ? 6.0 : same == 1 ? 2.0 : 1.0 );
                }
            }
        acc.m2 += V / 120 * s2;
    }
}

/// the signed cones from `( gx, gy, gz )` on the fan of `p` ( its orientation by the order of its vertices ) of the affine `f`,
/// into `acc`
template<bool MOM,class P>
__device__ __forceinline__ void cones3( const P &p, double gx, double gy, double gz, const Affine3 &f, DensSums3 &acc ) {
    if ( p.n < 3 )
        return;
    const double fg = f( gx, gy, gz );
    const double f0 = f( p.x[ 0 ], p.y[ 0 ], p.z[ 0 ] );
    for ( int j = 1; j + 1 < p.n; ++j ) {
        const double X[ 4 ] = { gx, p.x[ 0 ], p.x[ j ], p.x[ j + 1 ] };
        const double Y[ 4 ] = { gy, p.y[ 0 ], p.y[ j ], p.y[ j + 1 ] };
        const double Z[ 4 ] = { gz, p.z[ 0 ], p.z[ j ], p.z[ j + 1 ] };
        const double ax = X[ 1 ] - gx, ay = Y[ 1 ] - gy, az = Z[ 1 ] - gz;
        const double bx = X[ 2 ] - gx, by = Y[ 2 ] - gy, bz = Z[ 2 ] - gz;
        const double cx = X[ 3 ] - gx, cy = Y[ 3 ] - gy, cz = Z[ 3 ] - gz;
        const double V = ( ax * ( by * cz - bz * cy ) + ay * ( bz * cx - bx * cz ) + az * ( bx * cy - by * cx ) ) / 6;
        const double F[ 4 ] = { fg, f0, f( X[ 2 ], Y[ 2 ], Z[ 2 ] ), f( X[ 3 ], Y[ 3 ], Z[ 3 ] ) };
        tet3<MOM>( X, Y, Z, F, V, acc );
    }
}

/// `int f dA` on `p`, its area counted along `( nx, ny, nz )` ( signed: the orientation of `p` against it )
template<class P>
__device__ __forceinline__ double surface3( const P &p, const Affine3 &f, double nx, double ny, double nz ) {
    if ( p.n < 3 )
        return 0;
    const double nn = sqrt( nx * nx + ny * ny + nz * nz );
    if ( ! ( nn > 0 ) )
        return 0;
    const double f0 = f( p.x[ 0 ], p.y[ 0 ], p.z[ 0 ] );
    double res = 0;
    for ( int j = 1; j + 1 < p.n; ++j ) {
        const double ax = p.x[ j ] - p.x[ 0 ], ay = p.y[ j ] - p.y[ 0 ], az = p.z[ j ] - p.z[ 0 ];
        const double bx = p.x[ j + 1 ] - p.x[ 0 ], by = p.y[ j + 1 ] - p.y[ 0 ], bz = p.z[ j + 1 ] - p.z[ 0 ];
        const double A = ( ( ay * bz - az * by ) * nx + ( az * bx - ax * bz ) * ny + ( ax * by - ay * bx ) * nz ) / ( 2 * nn );
        res += A * ( f0 + f( p.x[ j ], p.y[ j ], p.z[ j ] ) + f( p.x[ j + 1 ], p.y[ j + 1 ], p.z[ j + 1 ] ) ) / 3;
    }
    return res;
}

/// element `e` in the seed's frame ( the seed at `O` ): its nodes, its barycentric coordinates, its density
struct Elem3 {
    double  N[ 4 ][ 3 ];
    Affine3 lam[ 4 ], f;
};

__device__ __forceinline__ void load_elem( const DensMesh3 &m, SI e, const double ( &O )[ 3 ], Elem3 &el ) {
    const int *q = m.tet + 4 * e;
    for ( int i = 0; i < 4; ++i )
        for ( int c = 0; c < 3; ++c )
            el.N[ i ][ c ] = m.pn[ 3 * q[ i ] + c ] - O[ c ];
    for ( int i = 0; i < 4; ++i )
        el.lam[ i ] = Affine3{ i == 0 ? 1.0 : 0.0, m.grad[ 12 * e + 3 * i ], m.grad[ 12 * e + 3 * i + 1 ], m.grad[ 12 * e + 3 * i + 2 ], el.N[ 0 ][ 0 ], el.N[ 0 ][ 1 ], el.N[ 0 ][ 2 ] };
    el.f = Affine3{ m.val[ 4 * e ], 0, 0, 0, el.N[ 0 ][ 0 ], el.N[ 0 ][ 1 ], el.N[ 0 ][ 2 ] };
    for ( int i = 0; i < 4; ++i ) {
        el.f.gx += m.val[ 4 * e + i ] * el.lam[ i ].gx;
        el.f.gy += m.val[ 4 * e + i ] * el.lam[ i ].gy;
        el.f.gz += m.val[ 4 * e + i ] * el.lam[ i ].gz;
    }
}

/// the tolerance on the barycentric coordinates ( a point on a plane of an element )
constexpr double MESH3_TOL = 1e-10;

/// the hierarchy walked with the box `[ lo, hi ]` ( global coordinates ): `body( e )` for each element of a leaf it meets
template<class F>
__device__ __forceinline__ void mesh3_walk( const DensMesh3 &m, const double ( &lo )[ 3 ], const double ( &hi )[ 3 ], F &&body ) {
    if ( m.nb == 0 )
        return;
    double l[ 3 ], h[ 3 ];
    for ( int d = 0; d < 3; ++d ) {
        // a little WIDER: an element that only touches the region must not be lost to a rounding ( `Density2D.cuh::mesh_walk` )
        const double mg = 1e-9 * ( hi[ d ] - lo[ d ] ) + 1e-12 * fmax( fabs( lo[ d ] ), fabs( hi[ d ] ) );
        l[ d ] = lo[ d ] - mg;
        h[ d ] = hi[ d ] + mg;
    }
    SI stack[ 64 ];
    int top = 0;
    stack[ top++ ] = 0;
    while ( top ) {
        const SI n = stack[ --top ];
        if ( m.bhi[ 3 * n ] < l[ 0 ] || m.blo[ 3 * n ] > h[ 0 ] || m.bhi[ 3 * n + 1 ] < l[ 1 ] || m.blo[ 3 * n + 1 ] > h[ 1 ] ||
             m.bhi[ 3 * n + 2 ] < l[ 2 ] || m.blo[ 3 * n + 2 ] > h[ 2 ] )
            continue;
        const int left = m.links[ 4 * n ];
        if ( left >= 0 ) {
            if ( top + 2 <= 64 ) {
                stack[ top++ ] = m.links[ 4 * n + 1 ];
                stack[ top++ ] = left;
            }
            continue;
        }
        for ( SI e = m.links[ 4 * n + 2 ]; e < m.links[ 4 * n + 3 ]; ++e )
            body( e );
    }
}

/// THE WALK IN TWO TIMES: the elements the box meets that `keep( e )` takes are gathered ( up to `B` per lane ), then `body( e )`
/// runs on them. A walk calls its body when ITS lane reaches a leaf, the other lanes of the warp waiting: the clipping, which is
/// the cost, would run one lane at a time. Gathered, the lanes clip together ( a full list is emptied during the walk )
template<int B = 48,class K,class F>
__device__ __forceinline__ void mesh3_gather( const DensMesh3 &m, const double ( &lo )[ 3 ], const double ( &hi )[ 3 ], K &&keep, F &&body ) {
    SI ids[ B ];
    int nb = 0;
    mesh3_walk( m, lo, hi, [&]( SI e ) {
        if ( ! keep( e ) )
            return;
        if ( nb == B ) {
            for ( int q = 0; q < B; ++q ) body( ids[ q ] );
            nb = 0;
        }
        ids[ nb++ ] = e;
    } );
    for ( int q = 0; q < nb; ++q )
        body( ids[ q ] );
}

/// THE SECTION of the element by a plane ( `h` its four nodes' signed values, positive inside, `tol` on that scale ): the nodes on
/// it and the crossings of the edges, as a convex polygon turning counterclockwise about `( nx, ny, nz )` ( at most 4 vertices )
template<class P>
__device__ __forceinline__ void section3( const Elem3 &el, const double ( &h )[ 4 ], double tol, double nx, double ny, double nz, P &cap ) {
    cap.n = 0;
    for ( int i = 0; i < 4; ++i )
        if ( fabs( h[ i ] ) <= tol )
            cap.push( el.N[ i ][ 0 ], el.N[ i ][ 1 ], el.N[ i ][ 2 ] );
    for ( int i = 0; i < 4; ++i )
        for ( int k = i + 1; k < 4; ++k )
            if ( ( h[ i ] > tol && h[ k ] < -tol ) || ( h[ i ] < -tol && h[ k ] > tol ) ) {
                const double t = h[ i ] / ( h[ i ] - h[ k ] );
                cap.push( el.N[ i ][ 0 ] + t * ( el.N[ k ][ 0 ] - el.N[ i ][ 0 ] ), el.N[ i ][ 1 ] + t * ( el.N[ k ][ 1 ] - el.N[ i ][ 1 ] ),
                          el.N[ i ][ 2 ] + t * ( el.N[ k ][ 2 ] - el.N[ i ][ 2 ] ) );
            }
    if ( cap.n < 3 ) {
        cap.n = 0;
        return;
    }
    // the orientation of ( 0, a, b ) about `n`
    auto turn = [&]( int a, int b ) {
        const double ax = cap.x[ a ] - cap.x[ 0 ], ay = cap.y[ a ] - cap.y[ 0 ], az = cap.z[ a ] - cap.z[ 0 ];
        const double bx = cap.x[ b ] - cap.x[ 0 ], by = cap.y[ b ] - cap.y[ 0 ], bz = cap.z[ b ] - cap.z[ 0 ];
        return ( ay * bz - az * by ) * nx + ( az * bx - ax * bz ) * ny + ( ax * by - ay * bx ) * nz;
    };
    auto swap = [&]( int a, int b ) {
        const double tx = cap.x[ a ], ty = cap.y[ a ], tz = cap.z[ a ];
        cap.x[ a ] = cap.x[ b ]; cap.y[ a ] = cap.y[ b ]; cap.z[ a ] = cap.z[ b ];
        cap.x[ b ] = tx; cap.y[ b ] = ty; cap.z[ b ] = tz;
    };
    if ( cap.n == 4 ) {
        // a quadrilateral: vertex 2 is the one opposite vertex 0 ( the two others on both sides of the diagonal )
        if ( turn( 1, 2 ) * turn( 3, 2 ) > 0 ) {
            if ( turn( 2, 1 ) * turn( 3, 1 ) < 0 ) swap( 1, 2 );   // 1 is opposite 0
            else                                   swap( 3, 2 );   // 3 is
        }
        if ( turn( 1, 2 ) < 0 ) swap( 1, 3 );
    } else if ( turn( 1, 2 ) < 0 )
        swap( 1, 2 );
}

/// A CELL AGAINST THE MESH: the planes `pl[ 4 j .. 4 j + 3 ]` ( `nc` of them, the seed's frame, OUTWARD normals: `n . x <= off`
/// inside ), the box `[ lo, hi ]` of the cell ( seed's frame ), `L` its size. The elements the box meets are shared between the
/// lanes ( the `lane`-th of every 32 ), each one CLASSIFIED against the planes:
///
///   * on the outer side of a plane, or with no node strictly inside one of those that cut it ( a flat intersection ): nothing;
///   * strictly inside all the planes: its own integral ( closed form );
///   * else, cut by the `act` planes ( those it has a node not strictly inside ): `C INTERSECT T` is bounded by the faces of `T`
///     clipped by them, and on each of them by the SECTION of `T` clipped by the others -- the cones from `g` of both into `acc`,
///     and the sections are the pieces of the cell's facets: `int rho dA` on plane `j` into `fl[ j ]` ( `FAC`, `j < FL` ).
///
/// An element cut by MANY planes ( larger than the cell, or about its size ) would clip each of its sections by all the others,
/// a cost in their number squared: from `SECT_MAX` planes on, the piece on plane `j` is the cell's face `j` ( `face( j, P )`, the
/// caller's: outward, `false` when it can not ) clipped by the four half-spaces of the element instead.
///
/// Only the planes that carry a face of the cell count ( `has_face( j )` ): the others are cuts made redundant by later ones,
/// which an element larger than the cell would see as cutting it, for nothing.
template<bool MOM,bool FAC,int FL,class Face,class HasFace>
__device__ __forceinline__ void mesh3_cell( const DensMesh3 &m, const double ( &O )[ 3 ], const double ( &g )[ 3 ], const double *pl, int nc,
                                            const double ( &lo )[ 3 ], const double ( &hi )[ 3 ], double L, int lane, DensSums3 &acc, double *fl,
                                            const Face &face, const HasFace &has_face ) {
    constexpr int SECT_MAX = 3;
    double glo[ 3 ], ghi[ 3 ];
    for ( int c = 0; c < 3; ++c ) { glo[ c ] = O[ c ] + lo[ c ]; ghi[ c ] = O[ c ] + hi[ c ]; }
    const double tol = 1e-10 * L;
    int count = 0;
    mesh3_gather( m, glo, ghi, [&]( SI ) { return ( count++ & 31 ) == lane; }, [&]( SI e ) {
        Elem3 el;
        load_elem( m, e, O, el );
        // the planes that cut the element ( `act` ), their values at the nodes ( `|n|_1` times the distance inside: no square root )
        // ( an element larger than the cell is cut by most of its planes: up to `ACT` of them )
        // The FLOAT decides the planes the element is far from ( a margin of `1e-4 L`, far above its rounding ), the double the others
        constexpr int ACT = 64;
        int act[ ACT ], na = 0;
        float fN[ 4 ][ 3 ];
        for ( int i = 0; i < 4; ++i ) for ( int c = 0; c < 3; ++c ) fN[ i ][ c ] = float( el.N[ i ][ c ] );
        for ( int j = 0; j < nc; ++j ) {
            if ( ! has_face( j ) )
                continue;
            const double nx = pl[ 4 * j ], ny = pl[ 4 * j + 1 ], nz = pl[ 4 * j + 2 ], off = pl[ 4 * j + 3 ];
            const float fnx = float( nx ), fny = float( ny ), fnz = float( nz ), foff = float( off );
            const float fm = 1e-4f * float( L ) * ( fabsf( fnx ) + fabsf( fny ) + fabsf( fnz ) );
            int fin = 0, fout = 0;
            for ( int i = 0; i < 4; ++i ) {
                const float hf = foff - ( fnx * fN[ i ][ 0 ] + fny * fN[ i ][ 1 ] + fnz * fN[ i ][ 2 ] );
                fin += hf > fm;
                fout += hf < -fm;
            }
            if ( fout == 4 )
                return;                                  // outside
            if ( fin == 4 )
                continue;                                // inside
            const double tj = tol * ( fabs( nx ) + fabs( ny ) + fabs( nz ) );
            double h[ 4 ];
            int nout = 0, nin = 0;
            for ( int i = 0; i < 4; ++i ) {
                h[ i ] = off - ( nx * el.N[ i ][ 0 ] + ny * el.N[ i ][ 1 ] + nz * el.N[ i ][ 2 ] );
                nout += h[ i ] < -tj;
                nin += h[ i ] > tj;
            }
            if ( nout == 4 || ( nin == 0 && tj > 0 ) )
                return;                                  // outside, or flat against the plane
            if ( nin < 4 ) {
                if ( na == ACT )
                    return;                              // ( a cell of more than 64 planes, all cutting one element )
                act[ na++ ] = j;
            }
        }
        if ( na == 0 ) {
            // strictly inside the cell: the element itself
            const double X[ 4 ] = { el.N[ 0 ][ 0 ], el.N[ 1 ][ 0 ], el.N[ 2 ][ 0 ], el.N[ 3 ][ 0 ] };
            const double Y[ 4 ] = { el.N[ 0 ][ 1 ], el.N[ 1 ][ 1 ], el.N[ 2 ][ 1 ], el.N[ 3 ][ 1 ] };
            const double Z[ 4 ] = { el.N[ 0 ][ 2 ], el.N[ 1 ][ 2 ], el.N[ 2 ][ 2 ], el.N[ 3 ][ 2 ] };
            const double ax = X[ 1 ] - X[ 0 ], ay = Y[ 1 ] - Y[ 0 ], az = Z[ 1 ] - Z[ 0 ];
            const double bx = X[ 2 ] - X[ 0 ], by = Y[ 2 ] - Y[ 0 ], bz = Z[ 2 ] - Z[ 0 ];
            const double cx = X[ 3 ] - X[ 0 ], cy = Y[ 3 ] - Y[ 0 ], cz = Z[ 3 ] - Z[ 0 ];
            const double V = fabs( ax * ( by * cz - bz * cy ) + ay * ( bz * cx - bx * cz ) + az * ( bx * cy - by * cx ) ) / 6;
            const double F[ 4 ] = { el.f( X[ 0 ], Y[ 0 ], Z[ 0 ] ), el.f( X[ 1 ], Y[ 1 ], Z[ 1 ] ), el.f( X[ 2 ], Y[ 2 ], Z[ 2 ] ), el.f( X[ 3 ], Y[ 3 ], Z[ 3 ] ) };
            tet3<MOM>( X, Y, Z, F, V, acc );
            return;
        }
        auto plane = [&]( int q ) {
            const int j = act[ q ];
            return Affine3{ pl[ 4 * j + 3 ], -pl[ 4 * j ], -pl[ 4 * j + 1 ], -pl[ 4 * j + 2 ], 0, 0, 0 };
        };
        auto tol_of = [&]( int q ) {
            const int j = act[ q ];
            return tol * ( fabs( pl[ 4 * j ] ) + fabs( pl[ 4 * j + 1 ] ) + fabs( pl[ 4 * j + 2 ] ) );
        };
        // the polygons in local arrays of `CAP` vertices: 8 for an element cut by one or two planes ( a section of 4 vertices clipped
        // once, a triangle twice ), 32 else ( the faces of the cell ) -- a smaller frame, half the time of the clipping
        auto cut = [&]<int CAP>() {
            // the faces of the element clipped by the planes that cut it ( dropped on one with the same outward normal )
            for ( int i = 0; i < 4; ++i ) {
                Poly3<CAP> r;
                for ( int j = 0; j < 4; ++j )
                    if ( j != i ) r.push( el.N[ j ][ 0 ], el.N[ j ][ 1 ], el.N[ j ][ 2 ] );
                // outward: the normal of `( r0, r1, r2 )` against the gradient of `lambda_i` ( which points inward )
                const double ux = r.x[ 1 ] - r.x[ 0 ], uy = r.y[ 1 ] - r.y[ 0 ], uz = r.z[ 1 ] - r.z[ 0 ];
                const double vx = r.x[ 2 ] - r.x[ 0 ], vy = r.y[ 2 ] - r.y[ 0 ], vz = r.z[ 2 ] - r.z[ 0 ];
                const double nr = ( uy * vz - uz * vy ) * el.lam[ i ].gx + ( uz * vx - ux * vz ) * el.lam[ i ].gy + ( ux * vy - uy * vx ) * el.lam[ i ].gz;
                if ( nr > 0 ) {
                    const double tx = r.x[ 1 ], ty = r.y[ 1 ], tz = r.z[ 1 ];
                    r.x[ 1 ] = r.x[ 2 ]; r.y[ 1 ] = r.y[ 2 ]; r.z[ 1 ] = r.z[ 2 ];
                    r.x[ 2 ] = tx; r.y[ 2 ] = ty; r.z[ 2 ] = tz;
                }
                bool keep = true;
                for ( int q = 0; q < na && keep && r.n >= 3; ++q ) {
                    const int j = act[ q ];
                    if ( clip3( r, plane( q ), tol_of( q ) ) == CLIP_ON )
                        keep = el.lam[ i ].gx * pl[ 4 * j ] + el.lam[ i ].gy * pl[ 4 * j + 1 ] + el.lam[ i ].gz * pl[ 4 * j + 2 ] > 0;   // ( normals opposite )
                }
                if ( keep )
                    cones3<MOM>( r, g[ 0 ], g[ 1 ], g[ 2 ], el.f, acc );
            }
            // the sections: the pieces of the cell's faces in the element
            for ( int q = 0; q < na; ++q ) {
                const int j = act[ q ];
                const Affine3 hq = plane( q );
                const double hn[ 4 ] = { hq( el.N[ 0 ][ 0 ], el.N[ 0 ][ 1 ], el.N[ 0 ][ 2 ] ), hq( el.N[ 1 ][ 0 ], el.N[ 1 ][ 1 ], el.N[ 1 ][ 2 ] ),
                                         hq( el.N[ 2 ][ 0 ], el.N[ 2 ][ 1 ], el.N[ 2 ][ 2 ] ), hq( el.N[ 3 ][ 0 ], el.N[ 3 ][ 1 ], el.N[ 3 ][ 2 ] ) };
                Poly3<CAP> cap;
                bool by_face = false;
                if constexpr ( CAP >= 32 )
                    by_face = face( j, cap );
                if ( by_face ) {
                    for ( int i = 0; i < 4 && cap.n >= 3; ++i )
                        clip3( cap, el.lam[ i ], MESH3_TOL );   // ( on a face of the element: kept whole )
                } else {
                    section3( el, hn, tol_of( q ), pl[ 4 * j ], pl[ 4 * j + 1 ], pl[ 4 * j + 2 ], cap );
                    for ( int k = 0; k < na && cap.n >= 3; ++k )
                        if ( k != q )
                            clip3( cap, plane( k ), tol_of( k ) );
                }
                if ( cap.n < 3 )
                    continue;
                cones3<MOM>( cap, g[ 0 ], g[ 1 ], g[ 2 ], el.f, acc );
                if constexpr ( FAC )
                    if ( j < FL )
                        fl[ j ] += surface3( cap, el.f, pl[ 4 * j ], pl[ 4 * j + 1 ], pl[ 4 * j + 2 ] );
            }
        };
        if ( na < SECT_MAX )
            cut.template operator()<8>();
        else
            cut.template operator()<32>();
    } );
}


// ---- the gaussians ( see the header ) -------------------------------------------------------------------------------------

constexpr double GAUSS3_S2P = 0.79788456080286535588;   ///< sqrt( 2 / pi )

/// `Phi3( x ) / x`, `x >= 0` ( by its series below 0.5, where `erf` and the exponential cancel )
__device__ __forceinline__ double gauss3_phi3_over_x( double x ) {
    if ( x < 0.5 ) {
        const double y = -x * x / 2;
        double s = 0, t = 1;
        for ( int k = 0; k < 30; ++k ) {
            const double term = t / ( 2 * k + 3 );
            s += term;
            if ( fabs( term ) < 1e-18 )
                break;
            t *= y / ( k + 1 );
        }
        return GAUSS3_S2P * x * x * s;
    }
    return ( erf( x * 0.70710678118654752440 ) - GAUSS3_S2P * x * exp( -x * x / 2 ) ) / x;
}

/// `Psi( R ) / rho^2`, `R^2 = h^2 + rho^2`, `h >= 0` ( see the header )
/// `inf = Psi( infinity )`, per edge: past the series, `Phi3( R ) / R + sqrt( 2 / pi ) exp( -R^2 / 2 ) = erf( R / sqrt 2 ) / R`, so
/// `Psi( R ) = inf - erf( R / sqrt 2 ) / R` -- one `erf` per point, the exponentials cancel
__device__ __forceinline__ double gauss3_psi_over_rho2( double h, double rho2, double inf ) {
    const double R2 = h * h + rho2;
    if ( R2 <= 4 ) {
        // `Phi3( r ) / r^2 = sqrt( 2 / pi ) sum_k ( -1 / 2 )^k / k! r^( 2 k + 1 ) / ( 2 k + 3 )`, integrated on `[ h, R ]`:
        // `( R^( 2 k + 2 ) - h^( 2 k + 2 ) ) / ( 2 k + 2 ) = rho^2 S_k / ( 2 k + 2 )`, `S_k = sum_i R^( 2 i ) h^( 2 ( k - i ) )`
        const double h2 = h * h;
        double s = 0, S = 1, h2k = 1, c = 1;
        for ( int k = 0; k < 60; ++k ) {
            const double term = c * S / ( ( 2 * k + 3 ) * ( 2 * k + 2 ) );
            s += term;
            if ( k > 2 && fabs( term ) < 1e-18 * fabs( s ) )
                break;
            c *= -0.5 / ( k + 1 );
            h2k *= h2;
            S = R2 * S + h2k;
        }
        return GAUSS3_S2P * s;
    }
    const double R = sqrt( R2 );
    if ( rho2 < 0.05 ) {
        // `R - h = rho^2 / ( R + h )`, small: Gauss-Legendre on `[ h, R ]`, the `rho^2` simplified away
        const double gx[ 2 ] = { 0.3399810435848563, 0.8611363115940526 }, gw[ 2 ] = { 0.6521451548625461, 0.3478548451374538 };
        const double d = rho2 / ( R + h );
        double acc = 0;
        for ( int q = 0; q < 2; ++q )
            for ( int sg = -1; sg <= 1; sg += 2 ) {
                const double r = h + d * ( 1 + sg * gx[ q ] ) / 2;
                acc += gw[ q ] * gauss3_phi3_over_x( r ) / r;
            }
        return acc / ( 2 * ( R + h ) );
    }
    return ( inf - erf( R * 0.70710678118654752440 ) / R ) / rho2;
}

/// the edge's share of the cones of a face: `sign( p ) int_t0^t1 |p| Psi( R ) / rho^2 dt` ( unit coordinates, `h >= 0` the
/// distance from the centre to the face's plane, `p`, `t0`, `t1` as for `gauss_wedge_pt` )
__device__ __forceinline__ double gauss3_cone_edge( double h, double p, double t0, double t1 ) {
    const double ap = fabs( p );
    if ( ! ( ap > 0 ) )
        return 0;
    const double q2 = h * h + p * p;
    // past `R = 8`, `Psi = Psi_inf - 1 / R` to the last bits, and its integral is closed
    const double inf = gauss3_phi3_over_x( h ) + GAUSS3_S2P * exp( -h * h / 2 );
    auto T = [&]( double t ) {
        const double S = sqrt( q2 + t * t );
        return h > 1e-12 ? atan( h * t / ( ap * S ) ) / h : t / ( ap * S );
    };
    auto tail = [&]( double a, double b ) { return inf * ( atan( b / ap ) - atan( a / ap ) ) - T( b ) + T( a ); };
    const double tc = sqrt( fmax( 64 - q2, 0.0 ) );
    double acc = 0;
    if ( t0 < -tc )
        acc += tail( t0, fmin( t1, -tc ) );
    if ( t1 > tc )
        acc += tail( fmax( t0, tc ), t1 );
    const double c0 = fmax( t0, -tc ), c1 = fmin( t1, tc );
    if ( c1 > c0 ) {
        // composite Gauss-Legendre, 8 points per panel of at most two units ( `Psi / rho^2` is smooth at the scale of the unit: no
        // peak at the foot; measured against panels of one unit, 3e-14 relative on the cells -- four units lose 3e-10 )
        const double gx[ 4 ] = { 0.1834346424956498, 0.5255324099163290, 0.7966664774136267, 0.9602898564975363 };
        const double gw[ 4 ] = { 0.3626837833783620, 0.3137066458778873, 0.2223810344533745, 0.1012285362903763 };
        const int nc = int( ceil( ( c1 - c0 ) / 2 ) ), np = nc < 1 ? 1 : ( nc > 8 ? 8 : nc );
        const double hw = ( c1 - c0 ) / ( 2 * np );
        for ( int k = 0; k < np; ++k ) {
            const double m = c0 + ( 2 * k + 1 ) * hw;
            for ( int q = 0; q < 4; ++q )
                for ( int sg = -1; sg <= 1; sg += 2 ) {
                    const double t = m + sg * hw * gx[ q ];
                    acc += hw * gw[ q ] * ap * gauss3_psi_over_rho2( h, p * p + t * t, inf );
                }
        }
    }
    return p < 0 ? -acc : acc;
}

/// where a warp is in the tree of `DensGauss3` ( `gauss3_fill` ): the depth-first stack, ONE SLOT PER LANE ( entry `k` in the
/// register `stk` of lane `k`, read by a shuffle: no local memory, whose every byte the card reserves for each resident thread ),
/// and the leaf being read. The host keeps the tree under 31 levels: the stack never holds more than the depth plus one.
struct Gauss3Walk {
    int stk = 0, sp = 0, li = 0, le = 0;
    __device__ __forceinline__ void start() { stk = 0; sp = 1; li = le = 0; }   // the root ( node 0 ), in lane 0's slot
};

/// THE TERMS OF A CELL, the walk of `GaussianTree::gather` on the card, by the whole warp at once ( the same box `[ lo, hi ]`, global
/// frame, on every lane: every lane does the same walk ). The nearer child first, a node pruned when its bound is under the cut,
/// a node narrow against its width ( `s2 > 0` ) emitted as ONE term.
///
///   * `EMIT = false`: `best` grows to the largest bound of a term ( pruned by `rel * best` as it grows, as the CPU's walk );
///   * `EMIT = true`: the terms whose bound is at least `cut` ( `rel` times the final `best`: the CPU's set, which it filters at the
///     end ), term `k` into `c, s, w` of lane `k`; returns after 32 of them ( or fewer: the walk is over ), to be called again.
template<bool EMIT>
__device__ __forceinline__ int gauss3_fill( const DensGauss3 &g, const double ( &lo )[ 3 ], const double ( &hi )[ 3 ], int lane, Gauss3Walk &st,
                                            double &best, double cut, double ( &c )[ 3 ], double &s, double &w ) {
    constexpr unsigned FULL = 0xffffffffu;
    const double s2 = g.s2;
    auto dist2 = [&]( const double *a, const double *b ) {    // between the boxes `[ a, b ]` and `[ lo, hi ]`
        double r = 0;
        for ( int d = 0; d < 3; ++d ) {
            const double e = fmax( fmax( a[ d ] - hi[ d ], lo[ d ] - b[ d ] ), 0.0 );
            r += e * e;
        }
        return r;
    };
    auto peak = [&]( double v ) {                        // `( 2 pi v ) ^ ( -3/2 )`
        const double r = 1 / sqrt( 6.283185307179586 * v );
        return r * r * r;
    };
    int cnt = 0;
    auto take = [&]( const double *x, double ts, double tw, double ub ) {
        if constexpr ( EMIT ) {
            if ( ub < cut )
                return;
            if ( lane == cnt ) {
                c[ 0 ] = x[ 0 ]; c[ 1 ] = x[ 1 ]; c[ 2 ] = x[ 2 ];
                s = ts;
                w = tw;
            }
            ++cnt;
        } else
            best = fmax( best, ub );
    };
    for ( ;; ) {
        if ( EMIT && cnt == 32 )
            return cnt;
        if ( st.li < st.le ) {                           // the rest of a leaf
            const double *t = g.it + 5 * st.li++;
            const double v = t[ 3 ] * t[ 3 ] + s2;
            take( t, sqrt( v ), t[ 4 ], fmax( t[ 4 ], 0.0 ) * exp( - dist2( t, t ) / ( 2 * v ) ) * peak( v ) );
            continue;
        }
        if ( ! st.sp )
            return cnt;
        const int id = __shfl_sync( FULL, st.stk, --st.sp );
        const double *n = g.nd + DensGauss3::ND * id;
        const double mass = n[ 9 ], var = n[ 10 ], smin = n[ 11 ], smax = n[ 12 ];
        const double vmax = fmax( smax * smax, var ) + s2, vmin = smin * smin + s2;
        const double ub = mass * exp( - dist2( n, n + 3 ) / ( 2 * vmax ) ) * peak( vmin );
        if ( ub < ( EMIT ? cut : g.rel * best ) || ! ( mass > 0 ) )
            continue;
        if ( s2 > 0 && n[ 13 ] <= g.theta * sqrt( vmin ) ) {   // one term
            const double v = var + s2;
            take( n + 6, sqrt( v ), mass, mass * exp( - dist2( n + 6, n + 6 ) / ( 2 * v ) ) * peak( v ) );
            continue;
        }
        const int *l = g.nl + 4 * id;
        if ( l[ 2 ] < 0 ) {
            st.li = l[ 0 ];
            st.le = l[ 1 ];
            continue;
        }
        // the nearer child pushed last, to be visited first ( `best` grows early, and prunes more )
        const double *a = g.nd + DensGauss3::ND * l[ 2 ], *b = g.nd + DensGauss3::ND * l[ 3 ];
        const bool left_nearer = dist2( a, a + 3 ) <= dist2( b, b + 3 );
        const int far = left_nearer ? l[ 3 ] : l[ 2 ], near = left_nearer ? l[ 2 ] : l[ 3 ];
        if ( lane == st.sp ) st.stk = far;
        if ( lane == st.sp + 1 ) st.stk = near;
        st.sp += 2;
    }
}

/// A CELL AGAINST THE GAUSSIANS ( see the header ): the planes `pl[ 4 j .. 4 j + 3 ]` ( seed's frame, outward ), the box `[ lo, hi ]`
/// of the cell ( seed's frame, the same on every lane ), `face_edges( j, f )` visiting the edges of the face on plane `j` as
/// `f( ax, ay, az, bx, by, bz )`, counterclockwise about its outward normal. Without a tree, a gaussian farther than `9 s` from the
/// box is nothing; with one, the terms are those of `gauss3_fill` ( the CPU's: relative truncation, groups merged ).
///
/// THE LANES SHARE THE GAUSSIANS, 32 at a time, each doing all the faces of its own ( every gaussian's `| sum |` needs the whole
/// cell ): sharing the ~15 faces of a cell instead left half of the warp idle, each lane walking its face's edges alone ( measured on
/// the trefoil of `docs/guide/distributions/gaussians.md`: 6 times slower ). The mass and the moments about the seed go to `acc` of
/// each lane, the caller reduces them. `int rho dA` on plane `j` ( `FAC`, `j < FL` ) is reduced over the warp face by face -- the
/// lanes visit the faces in the same order -- and ADDED to `fm[ j / 32 ]` of lane `j % 32`: no buffer of `FL` doubles per lane, which
/// was half of the kernel's stack ( the card reserves the stack of every resident thread when it launches the kernel, outside
/// XLA's pool: ~2 KB a thread was ~130 MB, which a card shared with another process did not have ).
template<bool MOM,bool FAC,int FL,class FaceEdges,class HasFace>
__device__ __forceinline__ void gauss3_cell( const DensGauss3 &g, const double ( &O )[ 3 ], const double *pl, int nc, const double ( &lo )[ 3 ],
                                             const double ( &hi )[ 3 ], int lane, DensSums3 &acc, double *fm, const FaceEdges &face_edges,
                                             const HasFace &has_face ) {
    constexpr unsigned FULL = 0xffffffffu;
    constexpr double inv_4pi = 0.079577471545947667884, inv_sqrt_2pi = 0.39894228040143267794;
    auto wsum = [&]( double v ) {
        for ( int d = 16; d; d >>= 1 )
            v += __shfl_xor_sync( FULL, v, d );
        return v;
    };
    // with a tree: the box in the global frame, then the strongest term of the cell ( the truncation is relative to it )
    const bool tree = g.nn > 0;
    double glo[ 3 ], ghi[ 3 ], cut = 0;
    Gauss3Walk st;
    if ( tree ) {
        for ( int a = 0; a < 3; ++a ) { glo[ a ] = lo[ a ] + O[ a ]; ghi[ a ] = hi[ a ] + O[ a ]; }
        double best = 0, dc[ 3 ], ds, dw;
        st.start();
        gauss3_fill<false>( g, glo, ghi, lane, st, best, 0.0, dc, ds, dw );
        cut = g.rel * best;
        st.start();
    }
    for ( SI q0 = 0; ; q0 += 32 ) {                      // ( the same on every lane )
        double s = 1, w = 0, c[ 3 ] = { 0, 0, 0 };
        bool act;
        if ( tree ) {                                    // the next 32 terms of the cell, one per lane
            double best = 0, tc[ 3 ] = { 0, 0, 0 }, ts = 1, tw = 0;
            const int cnt = gauss3_fill<true>( g, glo, ghi, lane, st, best, cut, tc, ts, tw );
            if ( ! cnt )
                break;
            act = lane < cnt && tw != 0;
            if ( act ) {
                s = ts;
                w = tw;
                for ( int a = 0; a < 3; ++a ) c[ a ] = tc[ a ] - O[ a ];
            }
        } else {
            if ( q0 >= g.nb )
                break;
            const SI q = q0 + lane;
            act = q < g.nb;
            if ( act ) {
                s = g.s[ q ];
                w = g.w[ q ];
                double d2 = 0;
                for ( int a = 0; a < 3; ++a ) {
                    c[ a ] = g.c[ 3 * q + a ] - O[ a ];
                    const double e = fmax( fmax( lo[ a ] - c[ a ], c[ a ] - hi[ a ] ), 0.0 );
                    d2 += e * e;
                }
                act = d2 < 81 * s * s && w != 0;
            }
        }
        if ( ! __any_sync( FULL, act ) )
            continue;
        const double is = 1 / s;
        double W = 0, Fx = 0, Fy = 0, Fz = 0, H = 0;
        for ( int j = 0; j < nc; ++j ) {
            if ( ! has_face( j ) )
                continue;
            const double nn = sqrt( pl[ 4 * j ] * pl[ 4 * j ] + pl[ 4 * j + 1 ] * pl[ 4 * j + 1 ] + pl[ 4 * j + 2 ] * pl[ 4 * j + 2 ] );
            if ( ! ( nn > 0 ) )
                continue;
            const double nx = pl[ 4 * j ] / nn, ny = pl[ 4 * j + 1 ] / nn, nz = pl[ 4 * j + 2 ] / nn;
            double ff = 0;                               // `int_face w phi dA`
            if ( act ) {
                const double h = ( pl[ 4 * j + 3 ] / nn - nx * c[ 0 ] - ny * c[ 1 ] - nz * c[ 2 ] ) * is;
                double cone = 0, wed = 0;
                face_edges( j, [&]( double ax, double ay, double az, double bx, double by, double bz ) {
                    const double Ax = ( ax - c[ 0 ] ) * is, Ay = ( ay - c[ 1 ] ) * is, Az = ( az - c[ 2 ] ) * is;
                    const double Bx = ( bx - c[ 0 ] ) * is, By = ( by - c[ 1 ] ) * is, Bz = ( bz - c[ 2 ] ) * is;
                    const double ex = Bx - Ax, ey = By - Ay, ez = Bz - Az;
                    const double L = sqrt( ex * ex + ey * ey + ez * ez );
                    if ( ! ( L > 0 ) )
                        return;
                    const double ux = ex / L, uy = ey / L, uz = ez / L;
                    const double mx = uy * nz - uz * ny, my = uz * nx - ux * nz, mz = ux * ny - uy * nx;   // `u x n`: outward in the face
                    const double p = mx * Ax + my * Ay + mz * Az;
                    const double t0 = ux * Ax + uy * Ay + uz * Az, t1 = ux * Bx + uy * By + uz * Bz;
                    if ( h != 0 )
                        cone += gauss3_cone_edge( fabs( h ), p, t0, t1 );
                    if ( FAC || MOM )
                        wed += gpu2d::gauss_wedge_pt( p, t0, t1 );
                } );
                W += h * inv_4pi * cone;
                if constexpr ( FAC || MOM )
                    ff = w * exp( -h * h / 2 ) * inv_sqrt_2pi * is * wed;
                if constexpr ( MOM ) {
                    Fx += nx * ff; Fy += ny * ff; Fz += nz * ff;
                    H += h * s * ff;
                }
            }
            if constexpr ( FAC )
                if ( j < FL ) {
                    const double v = wsum( ff );
                    if ( ( j & 31 ) == lane )
                        fm[ j >> 5 ] += v;
                }
        }
        if ( ! act )
            continue;
        const double m = w * fabs( W );
        acc.m += m;
        if constexpr ( MOM ) {
            // about the centre, then about the seed ( `c` in its frame )
            const double s2 = s * s;
            const double ux = -s2 * Fx, uy = -s2 * Fy, uz = -s2 * Fz, u2 = 3 * s2 * m - s2 * H;
            acc.mx += ux + c[ 0 ] * m;
            acc.my += uy + c[ 1 ] * m;
            acc.mz += uz + c[ 2 ] * m;
            acc.m2 += u2 + 2 * ( c[ 0 ] * ux + c[ 1 ] * uy + c[ 2 ] * uz ) + ( c[ 0 ] * c[ 0 ] + c[ 1 ] * c[ 1 ] + c[ 2 ] * c[ 2 ] ) * m;
        }
    }
}

} // namespace sdot::gpu3d
