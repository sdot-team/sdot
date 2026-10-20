#pragma once

#include <loom/support/common_macros.h>
#include <loom/support/containers/Vector.h>
#include "SumOfGaussians.h"
#include <loom/support/math.h>
#include "integration/StdGaussian2.h"

// Mathematics goes through `sdot::` (`loom/support/math.h`), NEVER through `std::`.
//
// It is not a style preference: `std::exp` / `std::atan` / `std::erf` on a `float`
// do not exist in CUDA device code (they lower to intrinsics that the device compiler
// cannot resolve), whereas the toolkit's overloads do exist. `sdot::` designates
// one or the other depending on the target, in a single place.
//
// `sqrt` is included even though it works everywhere (it is a native instruction): a rule that
// tolerates an exception is not a rule that gets followed.

#define UTP SDOT_TEMPLATE_DECL_FOR_SumOfGaussians
#define DTP SumOfGaussians<SDOT_TEMPLATE_ARGS_FOR_SumOfGaussians>

namespace sdot {

UTP HD auto DTP::kernel_at( SI i, const auto &x ) const {
    struct Kernel { TF phi; TF r2; TF s; };

    const TF s = sigma_of( i );
    TF r2 = 0;
    for ( PI c = 0; c < ct_dim; ++c ) {
        const TF e = TF( x[ c ] ) - center_of( i, c );
        r2 += e * e;
    }

    // `( 2 pi s^2 ) ^ ( -d/2 )`, written as an integer power of `1 / ( s sqrt( 2 pi ) )`: d
    // multiplications instead of a `pow`, and `ct_dim` being known at compile time the loop
    // disappears.
    const TF two_pi = TF( 6.283185307179586476925286766559 );
    const TF inv = TF( 1 ) / ( s * sdot::sqrt( two_pi ) );
    TF norm = 1;
    for ( int k = 0; k < ct_dim; ++k )
        norm *= inv;

    return Kernel{ norm * sdot::exp( - r2 / ( 2 * s * s ) ), r2, s };
}

UTP HD typename DTP::TF DTP::value_at( const auto &x ) const {
    const SI n = nb_terms();
    TF res = 0;
    for ( SI i = 0; i < n; ++i )
        res += weight_of( i ) * kernel_at( i, x ).phi;
    return res;
}

UTP HD auto DTP::gradient_at( const auto &x ) const {
    // `d/dx exp( -r^2 / 2s^2 ) = - ( x - c ) / s^2 * ...`: the gradient of a gaussian points towards
    // its center, with the factor `1 / s^2`.
    const SI n = nb_terms();
    auto res = Vector<TF,ct_dim>::zeros();
    for ( SI i = 0; i < n; ++i ) {
        const auto k = kernel_at( i, x );
        const TF f = weight_of( i ) * k.phi / ( k.s * k.s );
        for ( PI c = 0; c < ct_dim; ++c )
            res[ c ] -= f * ( TF( x[ c ] ) - center_of( i, c ) );
    }
    return res;
}

UTP HD void DTP::add_value_grad_at( auto &&grad_dist, const auto &x, TF g ) const {
    auto add_to = []( auto &&dst, TF v ) {
        if constexpr ( ! DECAYED_TYPE_OF( dst )::surely_null )
            atomic_add( dst.ref(), v );
    };

    // nothing requested: not a read, not an exponential. The test is at COMPILE time, so
    // a pure forward does not pay for the existence of this block.
    if constexpr ( DECAYED_TYPE_OF( grad_dist.weights )::surely_null
                && DECAYED_TYPE_OF( grad_dist.positions )::surely_null
                && DECAYED_TYPE_OF( grad_dist.sigmas )::surely_null ) {
        return;
    } else {
        const SI n = nb_gaussians;
        for ( SI i = 0; i < n; ++i ) {
            const auto k = kernel_at( i, x );
            const TF w = TF( weights( i ) );

            // d rho / d w_i = the normalized kernel
            add_to( grad_dist.weights( i ), g * k.phi );

            // d rho / d c_i = w_i * phi * ( x - c_i ) / s^2   ( the gradient AT x, sign flipped )
            if constexpr ( ! DECAYED_TYPE_OF( grad_dist.positions )::surely_null ) {
                const TF f = g * w * k.phi / ( k.s * k.s );
                for ( PI c = 0; c < ct_dim; ++c )
                    add_to( grad_dist.positions( i, c ), f * ( TF( x[ c ] ) - TF( positions( i, c ) ) ) );
            }

            // d rho / d s_i = w_i * phi * ( r^2 / s^3 - d / s ): the first term comes from
            // the exponential, the second from the normalization constant `s^-d`.
            add_to( grad_dist.sigmas( i ),
                    g * w * k.phi * ( k.r2 / ( k.s * k.s * k.s ) - TF( ct_dim ) / k.s ) * ( TF( sigmas( i ) ) / k.s ) );
        }
    }
}

// ---- exact integration in 2D ------------------------------------------------------------------
// See `SumOfGaussians.h` for the reduction. The corner, the triangle and the edge in the standard frame are
// `integration/StdGaussian2.h`'s; here, the methods.

UTP HD typename DTP::TF DTP::facet_mass( const auto &pc, int cut ) const requires ( ct_dim == 2 || ct_dim == 3 ) {
    if constexpr ( ct_dim == 2 )
        return facet_mass_2d( pc, cut );
    else
        return facet_mass_3d( pc, cut );
}

UTP HD typename DTP::TF DTP::facet_mass_2d( const auto &pc, int cut ) const {
    const int nb = pc.nb_vertices();
    const int j = cut + 1 < nb ? cut + 1 : 0;
    const TF ax = TF( pc.coord( cut, 0 ) ), ay = TF( pc.coord( cut, 1 ) );
    const TF ex = TF( pc.coord( j, 0 ) ) - ax, ey = TF( pc.coord( j, 1 ) ) - ay;
    const TF L = sdot::sqrt( ex * ex + ey * ey );
    if ( ! ( L > 0 ) )
        return 0;
    const TF ux = ex / L, uy = ey / L;                    // the tangent, and a normal
    const TF nx = -uy, ny = ux;
    TF res = 0;
    const SI ng = nb_terms();
    for ( SI i = 0; i < ng; ++i ) {
        const TF s = sigma_of( i ), m = weight_of( i );
        const TF px = ax - center_of( i, 0 ), py = ay - center_of( i, 1 );
        const TF d = px * nx + py * ny;                   // the signed distance from the center to the line
        const TF t0 = px * ux + py * uy, t1 = t0 + L;
        const TF q = d * d / ( 2 * s * s );
        if ( q > TF( 700 ) )
            continue;                                    // nothing, up to rounding
        const TF is2 = TF( 0.70710678118654752440 ) / s;
        const TF E = sdot::erf( t1 * is2 ) - sdot::erf( t0 * is2 );
        res += m * sdot::exp( -q ) * E / ( 2 * s * TF( 2.50662827463100050242 ) );   // `sqrt( 2 pi )`
    }
    return res;
}

UTP HD typename DTP::TF DTP::facet_mass_3d( const auto &pc, int cut ) const {
    // the unit normal of the face: the sum of the cross products of its fan, all of the same orientation ( a convex face )
    const auto pt = [&]( int i ) { return Vector<TF,3>( Function(), [&]( PI c ) { return TF( pc.coord( i, int( c ) ) ); } ); };
    TF N[ 3 ] = { 0, 0, 0 };
    int o0 = -1;                                         // a vertex of the face: the distance to its plane is read there
    pc.for_each_facet_triangle( cut, [&]( int o, int a, int b ) {
        o0 = o;
        const auto O = pt( o ), A = pt( a ), B = pt( b );
        const TF ax = A[ 0 ] - O[ 0 ], ay = A[ 1 ] - O[ 1 ], az = A[ 2 ] - O[ 2 ];
        const TF bx = B[ 0 ] - O[ 0 ], by = B[ 1 ] - O[ 1 ], bz = B[ 2 ] - O[ 2 ];
        TF cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
        if ( N[ 0 ] * cx + N[ 1 ] * cy + N[ 2 ] * cz < 0 ) { cx = -cx; cy = -cy; cz = -cz; }
        N[ 0 ] += cx; N[ 1 ] += cy; N[ 2 ] += cz;
    } );
    const TF nn = sdot::sqrt( N[ 0 ] * N[ 0 ] + N[ 1 ] * N[ 1 ] + N[ 2 ] * N[ 2 ] );
    if ( ! ( nn > 0 ) )
        return 0;
    const TF n[ 3 ] = { N[ 0 ] / nn, N[ 1 ] / nn, N[ 2 ] / nn };

    // an orthonormal frame `( u, w )` of the plane: `u` from the axis `n` is the least aligned with
    const TF an[ 3 ] = { sdot::fabs( n[ 0 ] ), sdot::fabs( n[ 1 ] ), sdot::fabs( n[ 2 ] ) };
    const int k = an[ 0 ] < an[ 1 ] ? ( an[ 0 ] < an[ 2 ] ? 0 : 2 ) : ( an[ 1 ] < an[ 2 ] ? 1 : 2 );
    TF u[ 3 ] = { -n[ k ] * n[ 0 ], -n[ k ] * n[ 1 ], -n[ k ] * n[ 2 ] };
    u[ k ] += 1;
    const TF nu = sdot::sqrt( u[ 0 ] * u[ 0 ] + u[ 1 ] * u[ 1 ] + u[ 2 ] * u[ 2 ] );
    for ( int c = 0; c < 3; ++c )
        u[ c ] /= nu;
    const TF w[ 3 ] = { n[ 1 ] * u[ 2 ] - n[ 2 ] * u[ 1 ], n[ 2 ] * u[ 0 ] - n[ 0 ] * u[ 2 ], n[ 0 ] * u[ 1 ] - n[ 1 ] * u[ 0 ] };

    // `rho = w_i * phi_1( d / s ) / s * phi_2( y / s ) / s^2`: the 1D factor across the plane, times the 2D standard measure of
    // the face in the scaled in-plane coordinates -- the sum over the fan, each triangle positive
    const auto P0 = pt( o0 );
    TF res = 0;
    const SI ng = nb_terms();
#if ! defined( __CUDA_ARCH__ )
    // on the host, the measure of the face is the sum of the corners of its EDGES, each once and counterclockwise ( the corners of the fan
    // cancel along its inner edges ): the edges are listed once per facet, in the absolute in-plane coordinates `( u . x, w . x )`,
    // instead of a walk over the cell for each gaussian and three corners per triangle ( `ed[ 4 * e + ... ]`: `A_u, A_w, B_u, B_w` )
    thread_local std::vector<TF> ed;
    ed.clear();
    TF gu = 0, gw = 0, lo[ 3 ], hi[ 3 ];
    for ( int d = 0; d < 3; ++d ) lo[ d ] = hi[ d ] = P0[ d ];
    pc.for_each_facet_triangle( cut, [&]( int, int a, int b ) {
        for ( int v : { a, b } ) {
            const auto P = pt( v );
            const TF pu = P[ 0 ] * u[ 0 ] + P[ 1 ] * u[ 1 ] + P[ 2 ] * u[ 2 ], pw = P[ 0 ] * w[ 0 ] + P[ 1 ] * w[ 1 ] + P[ 2 ] * w[ 2 ];
            ed.push_back( pu );
            ed.push_back( pw );
            gu += pu;
            gw += pw;
            for ( int d = 0; d < 3; ++d ) { lo[ d ] = sdot::fmin( lo[ d ], P[ d ] ); hi[ d ] = sdot::fmax( hi[ d ], P[ d ] ); }
        }
    } );
    const std::size_t ne = ed.size() / 4;
    if ( ne == 0 )
        return 0;
    gu /= TF( 2 * ne );
    gw /= TF( 2 * ne );                                  // ( the mean of the ends: inside the convex face )
    for ( std::size_t e = 0; e < ne; ++e ) {             // the right normal of each edge out of the face
        TF *E = ed.data() + 4 * e;
        if ( ( E[ 3 ] - E[ 1 ] ) * ( E[ 0 ] - gu ) - ( E[ 2 ] - E[ 0 ] ) * ( E[ 1 ] - gw ) < 0 ) {
            std::swap( E[ 0 ], E[ 2 ] );
            std::swap( E[ 1 ], E[ 3 ] );
        }
    }
    for ( SI i = 0; i < ng; ++i ) {
        const TF s = sigma_of( i ), is = 1 / s;
        const TF c[ 3 ] = { center_of( i, 0 ), center_of( i, 1 ), center_of( i, 2 ) };
        TF d2 = 0;                                       // ( farther than `9 s` from the box of the face: nothing, as `far3` )
        for ( int d = 0; d < 3; ++d ) {
            const TF e = sdot::fmax( sdot::fmax( lo[ d ] - c[ d ], c[ d ] - hi[ d ] ), TF( 0 ) );
            d2 += e * e;
        }
        if ( ! ( d2 < 81 * s * s ) )
            continue;
        const TF d = ( P0[ 0 ] - c[ 0 ] ) * n[ 0 ] + ( P0[ 1 ] - c[ 1 ] ) * n[ 1 ] + ( P0[ 2 ] - c[ 2 ] ) * n[ 2 ];
        const TF cu = c[ 0 ] * u[ 0 ] + c[ 1 ] * u[ 1 ] + c[ 2 ] * u[ 2 ], cw = c[ 0 ] * w[ 0 ] + c[ 1 ] * w[ 1 ] + c[ 2 ] * w[ 2 ];
        TF acc = 0;
        for ( std::size_t e = 0; e < ne; ++e ) {
            const TF *E = ed.data() + 4 * e;
            acc += wedge_measure( Vector<TF,2>( Values(), ( E[ 0 ] - cu ) * is, ( E[ 1 ] - cw ) * is ),
                                  Vector<TF,2>( Values(), ( E[ 2 ] - cu ) * is, ( E[ 3 ] - cw ) * is ) );
        }
        res += weight_of( i ) * sdot::exp( - d * d / ( 2 * s * s ) ) * ( acc < 0 ? -acc : acc ) / ( s * TF( 2.50662827463100050242 ) );
    }
#else
    for ( SI i = 0; i < ng; ++i ) {
        const TF s = sigma_of( i );
        const TF c[ 3 ] = { center_of( i, 0 ), center_of( i, 1 ), center_of( i, 2 ) };
        const TF d = ( P0[ 0 ] - c[ 0 ] ) * n[ 0 ] + ( P0[ 1 ] - c[ 1 ] ) * n[ 1 ] + ( P0[ 2 ] - c[ 2 ] ) * n[ 2 ];
        const TF q = d * d / ( 2 * s * s );
        if ( q > TF( 700 ) )
            continue;                                    // nothing, up to rounding
        const auto proj = [&]( int v ) {
            const auto P = pt( v );
            const TF e[ 3 ] = { P[ 0 ] - c[ 0 ], P[ 1 ] - c[ 1 ], P[ 2 ] - c[ 2 ] };
            return Vector<TF,2>( Values(), ( e[ 0 ] * u[ 0 ] + e[ 1 ] * u[ 1 ] + e[ 2 ] * u[ 2 ] ) / s,
                                           ( e[ 0 ] * w[ 0 ] + e[ 1 ] * w[ 1 ] + e[ 2 ] * w[ 2 ] ) / s );
        };
        TF acc = 0;
        pc.for_each_facet_triangle( cut, [&]( int o, int a, int b ) {
            acc += std_triangle_measure( Vector<Vector<TF,2>,3>( Values(), proj( o ), proj( a ), proj( b ) ) );
        } );
        res += weight_of( i ) * sdot::exp( -q ) * acc / ( s * TF( 2.50662827463100050242 ) );   // `sqrt( 2 pi )`
    }
#endif
    return res;
}

UTP HD typename DTP::TF DTP::wedge_measure( const auto &P, const auto &Q ) const {
    return integration::std_wedge_measure<TF>( P, Q, tail_cut, nb_panels );      // ( `integration/StdGaussian2.h` )
}

UTP HD typename DTP::TF DTP::std_triangle_measure( const auto &ys ) const {
    return integration::std_triangle_measure<TF>( ys, tail_cut, nb_panels );
}

UTP HD typename DTP::EdgeInfo DTP::edge_info( const auto &A, const auto &B, const auto &C ) const {
    const auto e = integration::std_edge_info<TF>( A, B, C );
    return EdgeInfo{ e.n, e.p, e.j0, e.j1a };
}

UTP HD typename DTP::TF DTP::integrate_over_simplex( const auto &pts ) const {
    static_assert( ct_dim == 2 || ct_dim == 3, "the exact path is 2D and 3D; beyond that we go through PointwiseDensity" );
    if constexpr ( ct_dim == 3 )
        return moment3<4>( pts );
    else {

    // the diameter of the triangle, for the choice of the rule ( during a continuation only: at `s = 0` the integral is exact )
    TF diam2 = 0;
    if ( terms && conv_s > 0 )
        for ( int a = 0; a < 3; ++a )
            for ( int b = a + 1; b < 3; ++b ) {
                const TF dx = pts[ a ][ 0 ] - pts[ b ][ 0 ], dy = pts[ a ][ 1 ] - pts[ b ][ 1 ];
                diam2 = sdot::fmax( diam2, dx * dx + dy * dy );
            }

    const SI n = nb_terms();
    TF res = 0;
    for ( SI i = 0; i < n; ++i ) {
        const TF s = sigma_of( i );
        if ( diam2 > 0 && s * s > quadrature_ratio * quadrature_ratio * diam2 ) {
            res += quadrature_term( i, pts );
            continue;
        }
        const auto ys = Vector<Vector<TF,2>,3>( Function(), [&]( PI k ) {
            return Vector<TF,2>( Function(), [&]( PI c ) { return ( pts[ k ][ c ] - center_of( i, c ) ) / s; } );
        } );
        res += weight_of( i ) * std_triangle_measure( ys );
    }
    return res;
    }
}

UTP template<int K> HD typename DTP::TF DTP::integrate_moment_over_simplex( const auto &pts ) const requires ( ct_dim == 2 || ct_dim == 3 ) {
    if constexpr ( ct_dim == 3 )
        return moment3<K>( pts );
    else {
    const SI n = nb_terms();
    TF res = 0;
    for ( SI i = 0; i < n; ++i ) {
        const TF s = sigma_of( i );
        const TF c[ 2 ] = { center_of( i, 0 ), center_of( i, 1 ) };
        const auto ys = Vector<Vector<TF,2>,3>( Function(), [&]( PI k ) {
            return Vector<TF,2>( Function(), [&]( PI d ) { return ( pts[ k ][ d ] - c[ d ] ) / s; } );
        } );
        const TF M = std_triangle_measure( ys );
        if constexpr ( K == 3 ) {
            res += weight_of( i ) * M;
            continue;
        }

        TF N[ 2 ] = { 0, 0 }, P = 0;      // sum_e n_e J_e, sum_e p_e J_e
        for ( int k = 0; k < 3; ++k ) {
            const auto ei = edge_info( ys[ k ], ys[ ( k + 1 ) % 3 ], ys[ ( k + 2 ) % 3 ] );
            N[ 0 ] += ei.n[ 0 ] * ei.j0;
            N[ 1 ] += ei.n[ 1 ] * ei.j0;
            P += ei.p * ei.j0;
        }
        if constexpr ( K < 2 )
            res += weight_of( i ) * ( c[ K ] * M - s * N[ K ] );
        else
            res += weight_of( i ) * ( ( c[ 0 ] * c[ 0 ] + c[ 1 ] * c[ 1 ] ) * M - 2 * s * ( c[ 0 ] * N[ 0 ] + c[ 1 ] * N[ 1 ] )
                                      + s * s * ( 2 * M - P ) );
    }
    return res;
    }
}

UTP HD typename DTP::TF DTP::quadrature_term( SI i, const auto &pts ) const {
    // Dunavant's degree 5 rule: the centroid, and two orbits of three points ( barycentric `( a, b, b )` )
    const TF ax = pts[ 1 ][ 0 ] - pts[ 0 ][ 0 ], ay = pts[ 1 ][ 1 ] - pts[ 0 ][ 1 ];
    const TF bx = pts[ 2 ][ 0 ] - pts[ 0 ][ 0 ], by = pts[ 2 ][ 1 ] - pts[ 0 ][ 1 ];
    TF area = ( ax * by - ay * bx ) / 2;
    area = area < 0 ? -area : area;
    const TF s = sigma_of( i );
    const TF cx = center_of( i, 0 ), cy = center_of( i, 1 );
    auto f = [&]( TF l0, TF l1, TF l2 ) {
        const TF x = l0 * pts[ 0 ][ 0 ] + l1 * pts[ 1 ][ 0 ] + l2 * pts[ 2 ][ 0 ] - cx;
        const TF y = l0 * pts[ 0 ][ 1 ] + l1 * pts[ 1 ][ 1 ] + l2 * pts[ 2 ][ 1 ] - cy;
        return sdot::exp( - ( x * x + y * y ) / ( 2 * s * s ) );
    };
    const TF a1 = TF( 0.059715871789770 ), b1 = TF( 0.470142064105115 );
    const TF a2 = TF( 0.797426985353087 ), b2 = TF( 0.101286507323456 );
    const TF acc = TF( 0.225 ) * f( TF( 1 ) / 3, TF( 1 ) / 3, TF( 1 ) / 3 )
                 + TF( 0.132394152788506 ) * ( f( a1, b1, b1 ) + f( b1, a1, b1 ) + f( b1, b1, a1 ) )
                 + TF( 0.125939180544827 ) * ( f( a2, b2, b2 ) + f( b2, a2, b2 ) + f( b2, b2, a2 ) );
    return weight_of( i ) * area * acc / ( TF( 6.283185307179586476925286766559 ) * s * s );
}

UTP HD void DTP::integrate_over_simplex_bwd( const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const {
    static_assert( ct_dim == 2 || ct_dim == 3, "the exact path is 2D and 3D; beyond that we go through PointwiseDensity" );
    if constexpr ( ct_dim == 3 )
        moment3_bwd<4>( pts, g, grad_pts, grad_dist );
    else {

    auto add_to = []( auto &&dst, TF v ) {
        if constexpr ( ! DECAYED_TYPE_OF( dst )::surely_null )
            atomic_add( dst.ref(), v );
    };

    const SI n = nb_gaussians;
    for ( SI i = 0; i < n; ++i ) {
        const TF s = sigma_of( i );
        const TF w = TF( weights( i ) );
        const auto ys = Vector<Vector<TF,2>,3>( Function(), [&]( PI k ) {
            return Vector<TF,2>( Function(), [&]( PI c ) { return ( pts[ k ][ c ] - TF( positions( i, c ) ) ) / s; } );
        } );

        // one pass over the three edges: each one pays into its TWO vertices (with the
        // barycentric weight that is 1 at one and 0 at the other), and into sigma via `y . n`.
        auto dm = Vector<Vector<TF,2>,3>( Function(), []( PI ) { return Vector<TF,2>::zeros(); } );
        TF dsig = 0;
        for ( SI k = 0; k < 3; ++k ) {
            const SI k1 = ( k + 1 ) % 3, k2 = ( k + 2 ) % 3;
            const auto ei = edge_info( ys[ k ], ys[ k1 ], ys[ k2 ] );
            for ( PI c = 0; c < 2; ++c ) {
                dm[ k  ][ c ] += ei.j1a * ei.n[ c ];
                dm[ k1 ][ c ] += ( ei.j0 - ei.j1a ) * ei.n[ c ];
            }
            dsig += ei.p * ei.j0;
        }

        // the vertices, then the center -- which is MINUS their sum (translating the triangle and the
        // gaussian together changes nothing), which avoids a second differentiation.
        const TF f = g * w / s;
        auto dc = Vector<TF,2>::zeros();
        for ( SI k = 0; k < 3; ++k ) {
            for ( PI c = 0; c < 2; ++c ) {
                grad_pts[ k ][ c ] += f * dm[ k ][ c ];
                dc[ c ] -= f * dm[ k ][ c ];
            }
        }

        if constexpr ( ! DECAYED_TYPE_OF( grad_dist.positions )::surely_null )
            for ( PI c = 0; c < 2; ++c )
                add_to( grad_dist.positions( i, c ), dc[ c ] );

        add_to( grad_dist.sigmas( i ), - f * dsig * ( TF( sigmas( i ) ) / s ) );   // `d sigma' / d sigma`
        add_to( grad_dist.weights( i ), g * std_triangle_measure( ys ) );
    }
    }
}

UTP HD typename DTP::EdgeMoments DTP::edge_moments( const auto &A, const auto &B, const auto &C ) const {
    const auto e = integration::std_edge_moments<TF>( A, B, C );      // ( `integration/StdGaussian2.h` )
    EdgeMoments res{ e.n, e.p, {}, {} };
    for ( int q = 0; q < 4; ++q ) { res.j[ q ] = e.j[ q ]; res.ja[ q ] = e.ja[ q ]; }
    return res;
}

UTP template<int K> HD void DTP::integrate_moment_over_simplex_bwd( const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const requires ( ct_dim == 2 || ct_dim == 3 ) {
    if constexpr ( ct_dim == 3 )
        moment3_bwd<K>( pts, g, grad_pts, grad_dist );
    else {
    auto add_to = []( auto &&dst, TF v ) {
        if constexpr ( ! DECAYED_TYPE_OF( dst )::surely_null )
            atomic_add( dst.ref(), v );
    };

    const SI n = nb_gaussians;
    for ( SI i = 0; i < n; ++i ) {
        const TF s = sigma_of( i );
        const TF w = TF( weights( i ) );
        const TF c[ 2 ] = { TF( positions( i, 0 ) ), TF( positions( i, 1 ) ) };
        const auto ys = Vector<Vector<TF,2>,3>( Function(), [&]( PI k ) {
            return Vector<TF,2>( Function(), [&]( PI d ) { return ( pts[ k ][ d ] - c[ d ] ) / s; } );
        } );

        // the moment's polynomial in the standard frame, `p( c + s y )`, on the basis `1, y_0, y_1, |y|^2`
        TF a[ 4 ] = { 0, 0, 0, 0 };
        if constexpr ( K < 2 ) {
            a[ 0 ] = c[ K ];
            a[ 1 + K ] = s;
        } else if constexpr ( K == 2 ) {
            a[ 0 ] = c[ 0 ] * c[ 0 ] + c[ 1 ] * c[ 1 ];
            a[ 1 ] = 2 * s * c[ 0 ];
            a[ 2 ] = 2 * s * c[ 1 ];
            a[ 3 ] = s * s;
        } else {
            a[ 0 ] = 1;
        }

        // the vertices ( standard frame: `d / d y_k`, without `w` ), and `N = sum_e n_e J_e`, `Pp = sum_e p_e J_e`
        auto dm = Vector<Vector<TF,2>,3>( Function(), []( PI ) { return Vector<TF,2>::zeros(); } );
        TF N[ 2 ] = { 0, 0 }, Pp = 0;
        for ( SI k = 0; k < 3; ++k ) {
            const SI k1 = ( k + 1 ) % 3, k2 = ( k + 2 ) % 3;
            const auto em = edge_moments( ys[ k ], ys[ k1 ], ys[ k2 ] );
            TF ja = 0, jb = 0;
            for ( int q = 0; q < 4; ++q ) {
                ja += a[ q ] * em.ja[ q ];
                jb += a[ q ] * ( em.j[ q ] - em.ja[ q ] );
            }
            for ( PI d = 0; d < 2; ++d ) {
                dm[ k  ][ d ] += ja * em.n[ d ];
                dm[ k1 ][ d ] += jb * em.n[ d ];
                N[ d ] += em.n[ d ] * em.j[ 0 ];
            }
            Pp += em.p * em.j[ 0 ];
        }
        const TF M = std_triangle_measure( ys );

        // `I / w`, `Int_T grad p rho / w` ( G ) and `Int_T grad p . ( x - c ) rho / w` ( H ), from the exact moments:
        // `Int y phi = - N`, `Int |y|^2 phi = 2 M - Pp`
        TF Iw = 0, G[ 2 ] = { 0, 0 }, H = 0;
        if constexpr ( K < 2 ) {
            Iw = c[ K ] * M - s * N[ K ];
            G[ K ] = M;
            H = - s * N[ K ];
        } else if constexpr ( K == 2 ) {
            const TF cN = c[ 0 ] * N[ 0 ] + c[ 1 ] * N[ 1 ], y2 = 2 * M - Pp;
            Iw = ( c[ 0 ] * c[ 0 ] + c[ 1 ] * c[ 1 ] ) * M - 2 * s * cN + s * s * y2;
            for ( int d = 0; d < 2; ++d )
                G[ d ] = 2 * ( c[ d ] * M - s * N[ d ] );
            H = 2 * ( - s * cN + s * s * y2 );
        } else {
            Iw = M;
        }

        // `d I / d P_k = w dm_k / s`; the center gets `w G - sum_k d I / d P_k`, sigma `( w H - w sum_k dm_k . y_k ) / s`
        const TF f = g * w / s;
        TF dc[ 2 ] = { g * w * G[ 0 ], g * w * G[ 1 ] }, ds = g * w * H;
        for ( SI k = 0; k < 3; ++k )
            for ( PI d = 0; d < 2; ++d ) {
                grad_pts[ k ][ d ] += f * dm[ k ][ d ];
                dc[ d ] -= f * dm[ k ][ d ];
                ds -= g * w * dm[ k ][ d ] * ys[ k ][ d ];
            }

        if constexpr ( ! DECAYED_TYPE_OF( grad_dist.positions )::surely_null )
            for ( PI d = 0; d < 2; ++d )
                add_to( grad_dist.positions( i, d ), dc[ d ] );
        add_to( grad_dist.sigmas( i ), ds / s * ( TF( sigmas( i ) ) / s ) );   // `d sigma' / d sigma`
        add_to( grad_dist.weights( i ), g * Iw );
    }
    }
}

// ---- exact integration in 3D ------------------------------------------------------------------
// See `SumOfGaussians.h` ( `face3` ): every integral over a tetrahedron goes to its four faces, in the standard frame of
// the gaussian -- the mass by the cones from the center ( the sibling of the 2D corners, `gpu/Density3D.cuh` ), the moments
// and the adjoints by divergence and Reynolds, whose face integrals split into `phi_1( h )` times 2D moments of the face.

namespace detail {
    LOOM_CONSTANT( double sqrt_2_over_pi ) = 0.79788456080286535588;

    /// `Phi3( x ) / x`, `x >= 0`, `Phi3( r ) = P( |X| <= r )` for the standard normal of R^3 ( its series below 0.5 )
    template<class TF> HD TF g3_phi3_over_x( TF x ) {
        if ( x < TF( 0.5 ) ) {
            const TF y = -x * x / 2;
            TF s = 0, t = 1;
            for ( int k = 0; k < 30; ++k ) {
                const TF term = t / TF( 2 * k + 3 );
                s += term;
                if ( sdot::fabs( term ) < TF( 1e-18 ) )
                    break;
                t *= y / TF( k + 1 );
            }
            return TF( sqrt_2_over_pi ) * x * x * s;
        }
        return ( sdot::erf( x * TF( 0.70710678118654752440 ) ) - TF( sqrt_2_over_pi ) * x * sdot::exp( -x * x / 2 ) ) / x;
    }

    /// `Psi( R ) / rho^2`, `Psi( R ) = Int_h^R Phi3( r ) / r^2 dr`, `R^2 = h^2 + rho^2`, `h >= 0`: by its series in `R^2` below
    /// `R = 2`, by a short quadrature when `R - h` is small, in closed form elsewhere -- never a difference of close numbers
    /// `inf = Psi( infinity ) = Phi3( h ) / h + sqrt( 2 / pi ) exp( -h^2 / 2 )`, per edge: past the series, `Phi3( R ) / R + sqrt( 2 / pi )
    /// exp( -R^2 / 2 ) = erf( R / sqrt 2 ) / R`, and `Psi( R ) = inf - erf( R / sqrt 2 ) / R` -- one `erf` per point, the `exp`s cancel
    template<class TF> HD TF g3_psi_over_rho2( TF h, TF rho2, TF inf ) {
        const TF R2 = h * h + rho2;
        if ( R2 <= 4 ) {
            const TF h2 = h * h;
            TF s = 0, S = 1, h2k = 1, c = 1;
            for ( int k = 0; k < 60; ++k ) {
                const TF term = c * S / TF( ( 2 * k + 3 ) * ( 2 * k + 2 ) );
                s += term;
                if ( k > 2 && sdot::fabs( term ) < TF( 1e-18 ) * sdot::fabs( s ) )
                    break;
                c *= TF( -0.5 ) / TF( k + 1 );
                h2k *= h2;
                S = R2 * S + h2k;
            }
            return TF( sqrt_2_over_pi ) * s;
        }
        const TF R = sdot::sqrt( R2 );
        if ( rho2 < TF( 0.05 ) ) {
            const TF gx[ 2 ] = { TF( 0.3399810435848563 ), TF( 0.8611363115940526 ) }, gw[ 2 ] = { TF( 0.6521451548625461 ), TF( 0.3478548451374538 ) };
            const TF d = rho2 / ( R + h );
            TF acc = 0;
            for ( int q = 0; q < 2; ++q )
                for ( int sg = -1; sg <= 1; sg += 2 ) {
                    const TF r = h + d * ( 1 + TF( sg ) * gx[ q ] ) / 2;
                    acc += gw[ q ] * g3_phi3_over_x( r ) / r;
                }
            return acc / ( 2 * ( R + h ) );
        }
        return ( inf - sdot::erf( R * TF( 0.70710678118654752440 ) ) / R ) / rho2;
    }

    /// an edge's share of the cone of a face: `sign( p ) Int_t0^t1 |p| Psi( R ) / rho^2 dt` ( `rho^2 = p^2 + t^2` ), `h >= 0` the
    /// distance from the center to the plane, `p`, `t0`, `t1` the edge's line as in `wedge_measure`. Past `R = 8`, `Psi = Psi_inf
    /// - 1 / R` to the last bits, and its integral is closed
    template<class TF> HD TF g3_cone_edge( TF h, TF p, TF t0, TF t1 ) {
        const TF ap = p < 0 ? -p : p;
        if ( ! ( ap > 0 ) )
            return 0;
        const TF q2 = h * h + p * p;
        const TF inf = g3_phi3_over_x( h ) + TF( sqrt_2_over_pi ) * sdot::exp( -h * h / 2 );
        auto T = [&]( TF t ) {
            const TF S = sdot::sqrt( q2 + t * t );
            return h > TF( 1e-12 ) ? sdot::atan( h * t / ( ap * S ) ) / h : t / ( ap * S );
        };
        auto tail = [&]( TF a, TF b ) { return inf * ( sdot::atan( b / ap ) - sdot::atan( a / ap ) ) - T( b ) + T( a ); };
        const TF tc = sdot::sqrt( sdot::fmax( 64 - q2, TF( 0 ) ) );
        TF acc = 0;
        if ( t0 < -tc )
            acc += tail( t0, sdot::fmin( t1, -tc ) );
        if ( t1 > tc )
            acc += tail( sdot::fmax( t0, tc ), t1 );
        const TF c0 = sdot::fmax( t0, -tc ), c1 = sdot::fmin( t1, tc );
        if ( c1 > c0 ) {
            // ( panels of at most two units: against panels of one, 3e-14 relative on the cells -- four units lose 3e-10 )
            const int nc = int( sdot::ceil( ( c1 - c0 ) / 2 ) ), np = nc < 1 ? 1 : ( nc > 8 ? 8 : nc );
            const TF hw = ( c1 - c0 ) / ( 2 * np );
            for ( int k = 0; k < np; ++k ) {
                const TF m = c0 + TF( 2 * k + 1 ) * hw;
                for ( int j = 0; j < 4; ++j )
                    for ( int sg = -1; sg <= 1; sg += 2 ) {
                        const TF t = m + TF( sg ) * hw * TF( gl8_x[ j ] );
                        acc += hw * TF( gl8_w[ j ] ) * ap * g3_psi_over_rho2( h, p * p + t * t, inf );
                    }
            }
        }
        return p < 0 ? -acc : acc;
    }

    /// `Phi( u1 ) - Phi( u0 )`, `u0 <= u1`, without the cancellation of two values near 1 ( or 0 ) in a tail
    template<class TF> HD TF g3_cdf_diff( TF u0, TF u1 ) {
        const TF r = TF( 0.70710678118654752440 );
        if ( u0 > 0 ) return ( sdot::erfc( u0 * r ) - sdot::erfc( u1 * r ) ) / 2;
        if ( u1 < 0 ) return ( sdot::erfc( -u1 * r ) - sdot::erfc( -u0 * r ) ) / 2;
        return ( sdot::erf( u1 * r ) - sdot::erf( u0 * r ) ) / 2;
    }
}

UTP HD void DTP::face3( const TF ( &Y )[ 4 ][ 3 ], int opp, int deg, Face3 &f ) const {
    const TF inv_4pi = TF( 0.079577471545947667884 ), two_pi = TF( 6.283185307179586476925286766559 );
    const TF sq_2pi = TF( 2.5066282746310005024157652848110 );
    for ( int q = 0; q < 10; ++q ) f.M[ q ] = 0;
    f.h = f.cone = f.phi1 = 0;
    int v[ 3 ], nv = 0;
    for ( int k = 0; k < 4; ++k )
        if ( k != opp )
            v[ nv++ ] = k;
    auto sub = [&]( int a, int b, int c ) { return Y[ a ][ c ] - Y[ b ][ c ]; };
    TF N[ 3 ] = { sub( v[ 1 ], v[ 0 ], 1 ) * sub( v[ 2 ], v[ 0 ], 2 ) - sub( v[ 1 ], v[ 0 ], 2 ) * sub( v[ 2 ], v[ 0 ], 1 ),
                  sub( v[ 1 ], v[ 0 ], 2 ) * sub( v[ 2 ], v[ 0 ], 0 ) - sub( v[ 1 ], v[ 0 ], 0 ) * sub( v[ 2 ], v[ 0 ], 2 ),
                  sub( v[ 1 ], v[ 0 ], 0 ) * sub( v[ 2 ], v[ 0 ], 1 ) - sub( v[ 1 ], v[ 0 ], 1 ) * sub( v[ 2 ], v[ 0 ], 0 ) };
    const TF nn = sdot::sqrt( N[ 0 ] * N[ 0 ] + N[ 1 ] * N[ 1 ] + N[ 2 ] * N[ 2 ] );
    for ( int a = 0; a < 3; ++a ) f.vid[ a ] = v[ a ];
    if ( ! ( nn > 0 ) ) {
        for ( int c = 0; c < 3; ++c ) f.n[ c ] = 0;
        for ( int a = 0; a < 3; ++a ) for ( int c = 0; c < 3; ++c ) f.lam[ a ][ c ] = 0;
        return;                                          // a flat face: nothing on it
    }
    // OUTWARD ( away from the opposite vertex ), the face's vertices counterclockwise about it
    TF n[ 3 ] = { N[ 0 ] / nn, N[ 1 ] / nn, N[ 2 ] / nn };
    if ( n[ 0 ] * sub( opp, v[ 0 ], 0 ) + n[ 1 ] * sub( opp, v[ 0 ], 1 ) + n[ 2 ] * sub( opp, v[ 0 ], 2 ) > 0 ) {
        for ( int c = 0; c < 3; ++c ) n[ c ] = -n[ c ];
        const int t = v[ 1 ]; v[ 1 ] = v[ 2 ]; v[ 2 ] = t;
    }
    for ( int a = 0; a < 3; ++a ) f.vid[ a ] = v[ a ];
    for ( int c = 0; c < 3; ++c ) f.n[ c ] = n[ c ];
    f.h = n[ 0 ] * Y[ v[ 0 ] ][ 0 ] + n[ 1 ] * Y[ v[ 0 ] ][ 1 ] + n[ 2 ] * Y[ v[ 0 ] ][ 2 ];
    f.phi1 = sdot::exp( -f.h * f.h / 2 ) / sq_2pi;

    // the frame of the plane: `y = h n + u e1 + v e2`, `n = e1 x e2`, the foot of the center at `( 0, 0 )`
    TF L01 = 0;
    for ( int c = 0; c < 3; ++c ) L01 += sub( v[ 1 ], v[ 0 ], c ) * sub( v[ 1 ], v[ 0 ], c );
    L01 = sdot::sqrt( L01 );
    for ( int c = 0; c < 3; ++c ) f.e1[ c ] = sub( v[ 1 ], v[ 0 ], c ) / L01;
    f.e2[ 0 ] = n[ 1 ] * f.e1[ 2 ] - n[ 2 ] * f.e1[ 1 ];
    f.e2[ 1 ] = n[ 2 ] * f.e1[ 0 ] - n[ 0 ] * f.e1[ 2 ];
    f.e2[ 2 ] = n[ 0 ] * f.e1[ 1 ] - n[ 1 ] * f.e1[ 0 ];
    TF Q[ 3 ][ 2 ];
    for ( int a = 0; a < 3; ++a ) {
        Q[ a ][ 0 ] = f.e1[ 0 ] * Y[ v[ a ] ][ 0 ] + f.e1[ 1 ] * Y[ v[ a ] ][ 1 ] + f.e1[ 2 ] * Y[ v[ a ] ][ 2 ];
        Q[ a ][ 1 ] = f.e2[ 0 ] * Y[ v[ a ] ][ 0 ] + f.e2[ 1 ] * Y[ v[ a ] ][ 1 ] + f.e2[ 2 ] * Y[ v[ a ] ][ 2 ];
    }

    // the barycentric coordinates of the face's vertices, affine in `( u, v )`
    const TF ar = ( Q[ 1 ][ 0 ] - Q[ 0 ][ 0 ] ) * ( Q[ 2 ][ 1 ] - Q[ 0 ][ 1 ] ) - ( Q[ 1 ][ 1 ] - Q[ 0 ][ 1 ] ) * ( Q[ 2 ][ 0 ] - Q[ 0 ][ 0 ] );
    for ( int a = 0; a < 3; ++a ) {
        const int b = ( a + 1 ) % 3, c = ( a + 2 ) % 3;
        f.lam[ a ][ 0 ] = ( Q[ b ][ 0 ] * Q[ c ][ 1 ] - Q[ b ][ 1 ] * Q[ c ][ 0 ] ) / ar;
        f.lam[ a ][ 1 ] = ( Q[ b ][ 1 ] - Q[ c ][ 1 ] ) / ar;
        f.lam[ a ][ 2 ] = ( Q[ c ][ 0 ] - Q[ b ][ 0 ] ) / ar;
    }

    // the edges, counterclockwise: the cone, the 2D measure ( the corners ), and for the moments the edge integrals
    // `E_ab = Int_e u^a v^b phi_2 ds`, `a + b <= 2`, of the outward normal `( t_v, -t_u )`
    TF cone = 0, M00 = 0, S[ 2 ][ 6 ] = {};             // `S[ c ][ ab ] = sum_e n_e[ c ] E_ab`
    for ( int a = 0; a < 3; ++a ) {
        const int b = ( a + 1 ) % 3;
        const TF dx = Q[ b ][ 0 ] - Q[ a ][ 0 ], dy = Q[ b ][ 1 ] - Q[ a ][ 1 ];
        const TF L = sdot::sqrt( dx * dx + dy * dy );
        if ( ! ( L > 0 ) )
            continue;
        const TF tu = dx / L, tv = dy / L, nu = tv, nv_ = -tu;
        const TF p = nu * Q[ a ][ 0 ] + nv_ * Q[ a ][ 1 ];
        const TF t0 = tu * Q[ a ][ 0 ] + tv * Q[ a ][ 1 ], t1 = tu * Q[ b ][ 0 ] + tv * Q[ b ][ 1 ];
        if ( f.h != 0 )
            cone += detail::g3_cone_edge( f.h < 0 ? -f.h : f.h, p, t0, t1 );
        M00 += wedge_measure( Vector<TF,2>( Values(), Q[ a ][ 0 ], Q[ a ][ 1 ] ), Vector<TF,2>( Values(), Q[ b ][ 0 ], Q[ b ][ 1 ] ) );
        if ( deg >= 1 ) {
            // along the edge `y = F + t tau`, `F = p n_e`, `phi_2 = exp( -p^2 / 2 ) exp( -t^2 / 2 ) / 2pi`: `G_m = Int t^m exp( -t^2/2 )`
            const TF e0 = sdot::exp( -t0 * t0 / 2 ), e1 = sdot::exp( -t1 * t1 / 2 );
            const TF G0 = sq_2pi * detail::g3_cdf_diff( t0, t1 ), G1 = e0 - e1, G2 = G0 + t0 * e0 - t1 * e1;
            const TF E = sdot::exp( -p * p / 2 ) / two_pi;
            const TF Fu = p * nu, Fv = p * nv_;
            const TF Eab[ 6 ] = { E * G0, E * ( Fu * G0 + tu * G1 ), E * ( Fv * G0 + tv * G1 ),
                                  E * ( Fu * Fu * G0 + 2 * Fu * tu * G1 + tu * tu * G2 ),
                                  E * ( Fu * Fv * G0 + ( Fu * tv + Fv * tu ) * G1 + tu * tv * G2 ),
                                  E * ( Fv * Fv * G0 + 2 * Fv * tv * G1 + tv * tv * G2 ) };
            for ( int q = 0; q < 6; ++q ) {
                S[ 0 ][ q ] += nu * Eab[ q ];
                S[ 1 ][ q ] += nv_ * Eab[ q ];
            }
        }
    }
    f.cone = f.h * inv_4pi * cone;
    // `Int u q phi = Int d_u q phi - sum_e n_u Int_e q phi`, the moments of `phi_2` up to degree 3: `M[ 00, 10, 01, 20, 11, 02, 30, 21, 12, 03 ]`
    f.M[ 0 ] = M00;
    if ( deg >= 1 ) {
        f.M[ 1 ] = -S[ 0 ][ 0 ];
        f.M[ 2 ] = -S[ 1 ][ 0 ];
    }
    if ( deg >= 2 ) {
        f.M[ 3 ] = M00 - S[ 0 ][ 1 ];
        f.M[ 4 ] = -S[ 0 ][ 2 ];
        f.M[ 5 ] = M00 - S[ 1 ][ 2 ];
    }
    if ( deg >= 3 ) {
        f.M[ 6 ] = 2 * f.M[ 1 ] - S[ 0 ][ 3 ];
        f.M[ 7 ] = f.M[ 2 ] - S[ 0 ][ 4 ];
        f.M[ 8 ] = f.M[ 1 ] - S[ 1 ][ 4 ];
        f.M[ 9 ] = 2 * f.M[ 2 ] - S[ 1 ][ 5 ];
    }
}

UTP HD void DTP::tet3( SI i, const auto &pts, int deg, TF ( &Y )[ 4 ][ 3 ], Face3 ( &F )[ 4 ], TF &Mt, TF ( &Nv )[ 3 ], TF &Y2 ) const {
    const TF s = sigma_of( i );
    for ( int k = 0; k < 4; ++k )
        for ( int c = 0; c < 3; ++c )
            Y[ k ][ c ] = ( TF( pts[ k ][ c ] ) - center_of( i, c ) ) / s;
    TF W = 0, H = 0;
    for ( int c = 0; c < 3; ++c ) Nv[ c ] = 0;
    for ( int j = 0; j < 4; ++j ) {
        face3( Y, j, deg, F[ j ] );
        W += F[ j ].cone;
        const TF Ff = F[ j ].phi1 * F[ j ].M[ 0 ];        // `Int_face phi`
        for ( int c = 0; c < 3; ++c ) Nv[ c ] -= F[ j ].n[ c ] * Ff;
        H += F[ j ].h * Ff;
    }
    // ( every face outward: the sum is the mass whatever the orientation; `| . |` only against the rounding of a far tetrahedron )
    Mt = W < 0 ? -W : W;
    Y2 = 3 * Mt - H;                                     // `Int y phi = Nv`, `Int |y|^2 phi = 3 M - sum_f h_f Int_f phi`
}

UTP HD bool DTP::far3( SI i, const auto &pts ) const {
    const TF s = sigma_of( i );
    TF d2 = 0;
    for ( int c = 0; c < 3; ++c ) {
        const TF cc = center_of( i, c );
        TF lo = TF( pts[ 0 ][ c ] ), hi = lo;
        for ( int k = 1; k < 4; ++k ) { lo = sdot::fmin( lo, TF( pts[ k ][ c ] ) ); hi = sdot::fmax( hi, TF( pts[ k ][ c ] ) ); }
        const TF e = sdot::fmax( sdot::fmax( lo - cc, cc - hi ), TF( 0 ) );
        d2 += e * e;
    }
    return ! ( d2 < 81 * s * s );
}

UTP template<int K> HD typename DTP::TF DTP::moment3( const auto &pts ) const {
    const SI n = nb_terms();
    TF res = 0;
    for ( SI i = 0; i < n; ++i ) {
        if ( far3( i, pts ) )
            continue;
        const TF s = sigma_of( i );
        const TF c[ 3 ] = { center_of( i, 0 ), center_of( i, 1 ), center_of( i, 2 ) };
        TF Y[ 4 ][ 3 ], Mt, Nv[ 3 ], Y2;
        Face3 F[ 4 ];
        tet3( i, pts, 0, Y, F, Mt, Nv, Y2 );
        if constexpr ( K < 3 )
            res += weight_of( i ) * ( c[ K ] * Mt + s * Nv[ K ] );
        else if constexpr ( K == 3 )
            res += weight_of( i ) * ( ( c[ 0 ] * c[ 0 ] + c[ 1 ] * c[ 1 ] + c[ 2 ] * c[ 2 ] ) * Mt + 2 * s * ( c[ 0 ] * Nv[ 0 ] + c[ 1 ] * Nv[ 1 ] + c[ 2 ] * Nv[ 2 ] )
                                      + s * s * Y2 );
        else
            res += weight_of( i ) * Mt;
    }
    return res;
}

UTP HD void DTP::moments3( const auto &pts, TF &m, auto &mx, TF &m2 ) const {
    const SI n = nb_terms();
    for ( SI i = 0; i < n; ++i ) {
        if ( far3( i, pts ) )
            continue;
        const TF s = sigma_of( i ), w = weight_of( i );
        const TF c[ 3 ] = { center_of( i, 0 ), center_of( i, 1 ), center_of( i, 2 ) };
        TF Y[ 4 ][ 3 ], Mt, Nv[ 3 ], Y2;
        Face3 F[ 4 ];
        tet3( i, pts, 0, Y, F, Mt, Nv, Y2 );
        m += w * Mt;
        for ( int d = 0; d < 3; ++d )
            mx[ d ] += w * ( c[ d ] * Mt + s * Nv[ d ] );
        m2 += w * ( ( c[ 0 ] * c[ 0 ] + c[ 1 ] * c[ 1 ] + c[ 2 ] * c[ 2 ] ) * Mt + 2 * s * ( c[ 0 ] * Nv[ 0 ] + c[ 1 ] * Nv[ 1 ] + c[ 2 ] * Nv[ 2 ] ) + s * s * Y2 );
    }
}

UTP template<class TI> HD bool DTP::integrate_cell( const auto &cell, auto &/*ws*/, TI &sum ) const requires ( ct_dim == 3 ) {
#if defined( __CUDA_ARCH__ )
    return false;
#else
    TF m, mx[ 3 ], m2;
    if ( ! cell_integrals<false>( cell, m, mx, m2 ) )
        return false;
    sum = m;
    return true;
#endif
}

UTP HD bool DTP::integrate_moments_cell( const auto &cell, auto &/*ws*/, TF &m, auto &mx, TF &m2 ) const requires ( ct_dim == 3 ) {
#if defined( __CUDA_ARCH__ )
    return false;
#else
    TF x[ 3 ];
    if ( ! cell_integrals<true>( cell, m, x, m2 ) )
        return false;
    for ( int d = 0; d < 3; ++d )
        mx[ d ] = x[ d ];
    return true;
#endif
}

UTP template<bool with_moments> bool DTP::cell_integrals( const auto &cell, TF &m, TF ( &mx )[ 3 ], TF &m2 ) const {
    m = m2 = 0;
    for ( int d = 0; d < 3; ++d )
        mx[ d ] = 0;
    if ( ! cell.bounded() )
        return false;
    const int nv = cell.nb_vertices(), nc = cell.nb_cuts();
    if ( nv < 4 )
        return true;

    TF lo[ 3 ], hi[ 3 ], g[ 3 ] = { 0, 0, 0 };
    for ( int d = 0; d < 3; ++d ) { lo[ d ] = TF( cell.coord( 0, d ) ); hi[ d ] = lo[ d ]; }
    for ( int v = 0; v < nv; ++v )
        for ( int d = 0; d < 3; ++d ) {
            const TF x = TF( cell.coord( v, d ) );
            lo[ d ] = sdot::fmin( lo[ d ], x );
            hi[ d ] = sdot::fmax( hi[ d ], x );
            g[ d ] += x / TF( nv );
        }

    // the terms of the cell, when a tree carries them ( as `for_each_piece` )
    if ( tree && ! terms ) {
        double dlo[ 3 ], dhi[ 3 ];
        for ( int d = 0; d < 3; ++d ) { dlo[ d ] = double( lo[ d ] ); dhi[ d ] = double( hi[ d ] ); }
        thread_local std::vector<Term> buf;
        tree->gather( dlo, dhi, double( conv_s ), buf );
        SumOfGaussians local( *this );
        local.terms = buf.data();
        local.nb_local = SI( buf.size() );
        return local.template cell_integrals<with_moments>( cell, m, mx, m2 );
    }

    // THE FACES: outward unit normal ( the cut's plane, turned away from the centroid ) and the mean of their vertices, which is
    // inside them ( `face[ 7 * k + ... ]`: `n[ 3 ], G[ 3 ], count` )
    thread_local std::vector<TF> face;
    face.assign( std::size_t( 7 ) * nc, TF( 0 ) );
    for ( int v = 0; v < nv; ++v )
        for ( int r = 0; r < 3; ++r ) {
            TF *f = face.data() + 7 * cell.vertex_cut( v, r );
            for ( int d = 0; d < 3; ++d )
                f[ 3 + d ] += TF( cell.coord( v, d ) );
            f[ 6 ] += 1;
        }
    for ( int k = 0; k < nc; ++k ) {
        TF *f = face.data() + 7 * k;
        if ( ! ( f[ 6 ] >= 3 ) )
            continue;                                    // no face ( `count` stays the flag: `0`, nothing on it )
        for ( int d = 0; d < 3; ++d )
            f[ 3 + d ] /= f[ 6 ];
        TF dir[ 3 ], off;
        cell.plane( k, dir, off );
        TF l = 0, sg = 0;
        for ( int d = 0; d < 3; ++d ) { l += dir[ d ] * dir[ d ]; sg += dir[ d ] * ( f[ 3 + d ] - g[ d ] ); }
        l = sdot::sqrt( l );
        if ( ! ( l > 0 ) ) {
            f[ 6 ] = 0;
            continue;
        }
        for ( int d = 0; d < 3; ++d )
            f[ d ] = ( sg < 0 ? -dir[ d ] : dir[ d ] ) / l;
    }

    // THE EDGES, once per face: the start point `a`, the direction `t` such that the in-plane normal `m = t x n` points out of the
    // face ( the polygon counterclockwise about `n` ), the length ( `edge[ 13 * e + ... ]`: `a[ 3 ], t[ 3 ], m[ 3 ], n[ 3 ], L` )
    thread_local std::vector<TF> edge;
    edge.clear();
    for ( int a = 0; a < nv; ++a )
        for ( int j = 0; j < 3; ++j ) {
            const int b = cell.vertex_neighbor( a, j );
            if ( b <= a )
                continue;                                // each edge once
            TF A[ 3 ], B[ 3 ], L = 0;
            for ( int d = 0; d < 3; ++d ) {
                A[ d ] = TF( cell.coord( a, d ) );
                B[ d ] = TF( cell.coord( b, d ) );
                L += ( B[ d ] - A[ d ] ) * ( B[ d ] - A[ d ] );
            }
            L = sdot::sqrt( L );
            if ( ! ( L > 0 ) )
                continue;
            for ( int r = 0; r < 3; ++r ) {
                if ( r == j )
                    continue;
                const TF *f = face.data() + 7 * cell.vertex_cut( a, r );
                if ( ! ( f[ 6 ] > 0 ) )
                    continue;
                TF t[ 3 ], mm[ 3 ];
                for ( int d = 0; d < 3; ++d ) t[ d ] = ( B[ d ] - A[ d ] ) / L;
                mm[ 0 ] = t[ 1 ] * f[ 2 ] - t[ 2 ] * f[ 1 ];
                mm[ 1 ] = t[ 2 ] * f[ 0 ] - t[ 0 ] * f[ 2 ];
                mm[ 2 ] = t[ 0 ] * f[ 1 ] - t[ 1 ] * f[ 0 ];
                const bool flip = mm[ 0 ] * ( A[ 0 ] - f[ 3 ] ) + mm[ 1 ] * ( A[ 1 ] - f[ 4 ] ) + mm[ 2 ] * ( A[ 2 ] - f[ 5 ] ) < 0;
                for ( int d = 0; d < 3; ++d ) edge.push_back( flip ? B[ d ] : A[ d ] );
                for ( int d = 0; d < 3; ++d ) edge.push_back( flip ? -t[ d ] : t[ d ] );
                for ( int d = 0; d < 3; ++d ) edge.push_back( flip ? -mm[ d ] : mm[ d ] );
                for ( int d = 0; d < 3; ++d ) edge.push_back( f[ d ] );
                edge.push_back( L );
            }
        }

    // THE TERMS, each in its standard frame: per ( edge, face ), `h = n . y`, `p = m . y`, `t0 = t . y` at the start point
    const TF inv_4pi = TF( 0.079577471545947667884 ), inv_sq_2pi = TF( 0.39894228040143267794 );
    const std::size_t ne = edge.size() / 13;
    const SI n = nb_terms();
    for ( SI i = 0; i < n; ++i ) {
        const TF s = sigma_of( i ), w = weight_of( i ), is = 1 / s;
        const TF c[ 3 ] = { center_of( i, 0 ), center_of( i, 1 ), center_of( i, 2 ) };
        TF d2 = 0;                                       // ( farther than `9 s` from the box of the cell: nothing, as `far3` )
        for ( int d = 0; d < 3; ++d ) {
            const TF e = sdot::fmax( sdot::fmax( lo[ d ] - c[ d ], c[ d ] - hi[ d ] ), TF( 0 ) );
            d2 += e * e;
        }
        if ( ! ( d2 < 81 * s * s ) )
            continue;
        TF W = 0, Nv[ 3 ] = { 0, 0, 0 }, H = 0;
        for ( std::size_t e = 0; e < ne; ++e ) {
            const TF *E = edge.data() + 13 * e;
            const TF y[ 3 ] = { ( E[ 0 ] - c[ 0 ] ) * is, ( E[ 1 ] - c[ 1 ] ) * is, ( E[ 2 ] - c[ 2 ] ) * is };
            const TF t0 = E[ 3 ] * y[ 0 ] + E[ 4 ] * y[ 1 ] + E[ 5 ] * y[ 2 ];
            const TF p  = E[ 6 ] * y[ 0 ] + E[ 7 ] * y[ 1 ] + E[ 8 ] * y[ 2 ];
            const TF h  = E[ 9 ] * y[ 0 ] + E[ 10 ] * y[ 1 ] + E[ 11 ] * y[ 2 ];
            const TF t1 = t0 + E[ 12 ] * is;
            if ( h != 0 )
                W += h * detail::g3_cone_edge( h < 0 ? -h : h, p, t0, t1 );
            if constexpr ( with_moments ) {
                const TF Ff = sdot::exp( -h * h / 2 ) * inv_sq_2pi * wedge_measure( Vector<TF,2>( Values(), p, t0 ), Vector<TF,2>( Values(), p, t1 ) );
                for ( int d = 0; d < 3; ++d )
                    Nv[ d ] -= E[ 9 + d ] * Ff;
                H += h * Ff;
            }
        }
        const TF Mt = ( W < 0 ? -W : W ) * inv_4pi;
        m += w * Mt;
        if constexpr ( with_moments ) {
            for ( int d = 0; d < 3; ++d )
                mx[ d ] += w * ( c[ d ] * Mt + s * Nv[ d ] );
            m2 += w * ( ( c[ 0 ] * c[ 0 ] + c[ 1 ] * c[ 1 ] + c[ 2 ] * c[ 2 ] ) * Mt + 2 * s * ( c[ 0 ] * Nv[ 0 ] + c[ 1 ] * Nv[ 1 ] + c[ 2 ] * Nv[ 2 ] )
                        + s * s * ( 3 * Mt - H ) );
        }
    }
    return true;
}

UTP template<int K> HD void DTP::moment3_bwd( const auto &pts, TF g, auto &&grad_pts, auto &&grad_dist ) const {
    auto add_to = []( auto &&dst, TF v ) {
        if constexpr ( ! DECAYED_TYPE_OF( dst )::surely_null )
            atomic_add( dst.ref(), v );
    };
    constexpr int deg = K < 3 ? 2 : K == 3 ? 3 : 1;     // the degree of `p lambda` on a face
    const SI n = nb_gaussians;
    for ( SI i = 0; i < n; ++i ) {
        if ( far3( i, pts ) )
            continue;
        const TF s = sigma_of( i );
        const TF w = TF( weights( i ) );
        const TF c[ 3 ] = { TF( positions( i, 0 ) ), TF( positions( i, 1 ) ), TF( positions( i, 2 ) ) };
        TF Y[ 4 ][ 3 ], Mt, Nv[ 3 ], Y2;
        Face3 F[ 4 ];
        tet3( i, pts, deg, Y, F, Mt, Nv, Y2 );

        // the moment's polynomial in the standard frame, `p( c + s y )`, on the basis `1, y_0, y_1, y_2, |y|^2`
        TF a[ 5 ] = { 0, 0, 0, 0, 0 };
        if constexpr ( K < 3 ) {
            a[ 0 ] = c[ K ];
            a[ 1 + K ] = s;
        } else if constexpr ( K == 3 ) {
            a[ 0 ] = c[ 0 ] * c[ 0 ] + c[ 1 ] * c[ 1 ] + c[ 2 ] * c[ 2 ];
            for ( int d = 0; d < 3; ++d ) a[ 1 + d ] = 2 * s * c[ d ];
            a[ 4 ] = s * s;
        } else {
            a[ 0 ] = 1;
        }

        // REYNOLDS: moving vertex `k` sweeps its three faces, `d / d y_k = sum_{f ∋ k} n_f Int_f p phi lambda_k dA` -- on a face,
        // `p = Q0 + Qu u + Qv v + Q2 ( u^2 + v^2 )`, times `lambda = l0 + l1 u + l2 v`, against the moments of `phi_2`
        TF dm[ 4 ][ 3 ] = {};
        for ( int j = 0; j < 4; ++j ) {
            const Face3 &f = F[ j ];
            if ( f.phi1 == 0 )
                continue;
            TF Q0 = a[ 0 ] + a[ 4 ] * f.h * f.h, Qu = 0, Qv = 0;
            for ( int d = 0; d < 3; ++d ) {
                Q0 += a[ 1 + d ] * f.h * f.n[ d ];
                Qu += a[ 1 + d ] * f.e1[ d ];
                Qv += a[ 1 + d ] * f.e2[ d ];
            }
            const TF Q2 = a[ 4 ];
            const TF *M = f.M;
            for ( int q = 0; q < 3; ++q ) {
                const TF l0 = f.lam[ q ][ 0 ], l1 = f.lam[ q ][ 1 ], l2 = f.lam[ q ][ 2 ];
                TF J = Q0 * l0 * M[ 0 ] + ( Q0 * l1 + Qu * l0 ) * M[ 1 ] + ( Q0 * l2 + Qv * l0 ) * M[ 2 ];
                if constexpr ( deg >= 2 )
                    J += Qu * l1 * M[ 3 ] + ( Qu * l2 + Qv * l1 ) * M[ 4 ] + Qv * l2 * M[ 5 ];
                if constexpr ( deg >= 3 )
                    J += Q2 * ( l0 * ( M[ 3 ] + M[ 5 ] ) + l1 * ( M[ 6 ] + M[ 8 ] ) + l2 * ( M[ 7 ] + M[ 9 ] ) );
                J *= f.phi1;
                for ( int d = 0; d < 3; ++d )
                    dm[ f.vid[ q ] ][ d ] += J * f.n[ d ];
            }
        }

        // `I / w`, `Int_T grad p rho / w` ( G ) and `Int_T grad p . ( x - c ) rho / w` ( H ), from the exact moments
        TF Iw = 0, G[ 3 ] = { 0, 0, 0 }, H = 0;
        if constexpr ( K < 3 ) {
            Iw = c[ K ] * Mt + s * Nv[ K ];
            G[ K ] = Mt;
            H = s * Nv[ K ];
        } else if constexpr ( K == 3 ) {
            const TF cN = c[ 0 ] * Nv[ 0 ] + c[ 1 ] * Nv[ 1 ] + c[ 2 ] * Nv[ 2 ];
            Iw = ( c[ 0 ] * c[ 0 ] + c[ 1 ] * c[ 1 ] + c[ 2 ] * c[ 2 ] ) * Mt + 2 * s * cN + s * s * Y2;
            for ( int d = 0; d < 3; ++d )
                G[ d ] = 2 * ( c[ d ] * Mt + s * Nv[ d ] );
            H = 2 * ( s * cN + s * s * Y2 );
        } else {
            Iw = Mt;
        }

        // `d I / d P_k = w dm_k / s`; the center gets `w G - sum_k d I / d P_k`, sigma `( w H - w sum_k dm_k . y_k ) / s`
        const TF f = g * w / s;
        TF dc[ 3 ] = { g * w * G[ 0 ], g * w * G[ 1 ], g * w * G[ 2 ] }, ds = g * w * H;
        for ( int k = 0; k < 4; ++k )
            for ( int d = 0; d < 3; ++d ) {
                grad_pts[ k ][ d ] += f * dm[ k ][ d ];
                dc[ d ] -= f * dm[ k ][ d ];
                ds -= g * w * dm[ k ][ d ] * Y[ k ][ d ];
            }
        if constexpr ( ! DECAYED_TYPE_OF( grad_dist.positions )::surely_null )
            for ( int d = 0; d < 3; ++d )
                add_to( grad_dist.positions( i, d ), dc[ d ] );
        add_to( grad_dist.sigmas( i ), ds / s * ( TF( sigmas( i ) ) / s ) );   // `d sigma' / d sigma`
        add_to( grad_dist.weights( i ), g * Iw );
    }
}

#undef UTP
#undef DTP

}
