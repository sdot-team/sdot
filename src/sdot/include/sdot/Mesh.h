#pragma once

// the members + the axes this body names, written to the build include tree by `CallArg_Aggregate`.
#include <sdot/generated/aggregates/Mesh.h>
#include <loom/support/common_macros.h>
#include <loom/support/containers/Vector.h>
#include <loom/support/containers/Matrix.h>
#include <loom/support/atomic_add.h>
#include <loom/support/math.h>
#include <limits>

namespace sdot {

// The density of ONE ELEMENT of a mesh, as the piece contract wants it ( `distributions/Distribution.py`, the case
// `is_constant == false` ): the piece is `cell INTERSECT simplex`, the density is AFFINE on it,
//
//     rho( x ) = sum_i lambda_i( x ) v_i,        lambda_i( x ) = [ i == 0 ] + G_i . ( x - x_0 ),
//
// `v_i` the value of the element at its corner `i` ( DG1: `values( e, i )`, discontinuous between the elements ), `G_i` the gradient of its barycentric coordinate. An affine function has
// closed forms on a simplex, all exact: with `f_k` its values at the vertices `p_k`,
//
//     int f = |S| mean_k f_k,                 int x f = |S| / ( ( d + 1 )( d + 2 ) ) sum_{j,k} ( 1 + [ j == k ] ) p_j f_k,
//     int |x|^2 f = sum_{i,j,k} ( p_i . p_j ) f_k int lambda_i lambda_j lambda_k      ( int lambda^a = d! a! |S| / ( d + |a| )! ).
template<class MeshT>
struct MeshElementDensity {
    using TF = typename MeshT::TF;
    static constexpr int ct_dim = MeshT::ct_dim;
    static constexpr bool is_constant = false;
    static constexpr bool is_affine = true;              ///< `value_at` is affine: what the integration in a ball wants ( `InBall.h` )
    static_assert( ct_dim >= 2 );

    const MeshT *mesh;
    SI           e;                                      ///< the element

    HD SI  node( int i ) const { return SI( mesh->simplices( e, i ) ); }
    HD TF  value( int i ) const { return TF( mesh->values( e, i ) ); }

    /// `lambda_i( x )`
    HD TF lambda( int i, const auto &x ) const {
        const SI n0 = node( 0 );
        TF s = i == 0 ? TF( 1 ) : TF( 0 );
        for ( int c = 0; c < ct_dim; ++c )
            s += TF( mesh->grads( e, i, c ) ) * ( TF( x[ c ] ) - TF( mesh->nodes( n0, c ) ) );
        return s;
    }

    HD TF value_at( const auto &x ) const {
        TF s = 0;
        for ( int i = 0; i <= ct_dim; ++i )
            s += lambda( i, x ) * value( i );
        return s;
    }

    HD Vector<TF,ct_dim> gradient_at( const auto &/*x*/ ) const {
        return Vector<TF,ct_dim>::with_func( [&]( PI c ) {
            TF s = 0;
            for ( int i = 0; i <= ct_dim; ++i )
                s += value( i ) * TF( mesh->grads( e, i, c ) );
            return s;
        } );
    }

    HD void add_value_grad_at( auto &&grad_dist, const auto &x, TF g ) const {
        if constexpr ( DECAYED_TYPE_OF( grad_dist.values )::is_valid )
            for ( int i = 0; i <= ct_dim; ++i )
                atomic_add( grad_dist.values( e, i ).ref(), g * lambda( i, x ) );
    }

    // ---- the simplex integrals ---------------------------------------------------------------------

    HD static TF factorial( int n ) { TF r = 1; for ( int i = 2; i <= n; ++i ) r *= i; return r; }

    HD static auto edge_matrix( const auto &P ) {
        return Matrix<TF,ct_dim>::with_func( [&]( auto r, auto c ) { return P[ c + 1 ][ r ] - P[ 0 ][ r ]; } );
    }

    HD TF integrate_over_simplex( const auto &P ) const {
        const TF det = edge_matrix( P ).determinant();
        const TF vol = ( det < 0 ? -det : det ) / factorial( ct_dim );
        TF s = 0;
        for ( int q = 0; q <= ct_dim; ++q )
            s += value_at( P[ q ] );
        return vol * s / ( ct_dim + 1 );
    }

    /// the adjoint: the volume moves with the vertices ( cofactors of the determinant ), `f` is fixed in space, so its
    /// value at a vertex moves with the gradient, and the corners' values receive the barycentric coordinates
    HD void integrate_over_simplex_bwd( const auto &P, TF g, auto &&grad_pts, auto &&grad_dist ) const {
        const auto M = edge_matrix( P );
        const TF det = M.determinant();
        const TF vol = ( det < 0 ? -det : det ) / factorial( ct_dim );
        TF s = 0;
        for ( int q = 0; q <= ct_dim; ++q )
            s += value_at( P[ q ] );
        s /= ( ct_dim + 1 );

        const TF gv = ( det < 0 ? -g : g ) * s / factorial( ct_dim );
        for ( PI r = 0; r < ct_dim; ++r ) {
            TF row_sum = 0;
            for ( PI c = 0; c < ct_dim; ++c ) {
                const TF minor = M.without_row_and_col( r, c ).determinant();
                const TF cof = ( ( r + c ) % 2 ? -minor : minor ) * gv;
                grad_pts[ c + 1 ][ r ] += cof;
                row_sum += cof;
            }
            grad_pts[ 0 ][ r ] -= row_sum;
        }

        const TF gn = g * vol / ( ct_dim + 1 );
        const auto gr = gradient_at( P[ 0 ] );
        for ( int q = 0; q <= ct_dim; ++q ) {
            for ( PI c = 0; c < ct_dim; ++c )
                grad_pts[ q ][ c ] += gn * gr[ c ];
            add_value_grad_at( grad_dist, P[ q ], gn );
        }
    }

    /// `int rho`, `int x rho`, `int |x|^2 rho` over the simplex, ACCUMULATED
    HD void integrate_moments_over_simplex( const auto &P, TF &m, auto &mx, TF &m2 ) const {
        constexpr int D = ct_dim;
        const TF det = edge_matrix( P ).determinant();
        const TF vol = ( det < 0 ? -det : det ) / factorial( D );
        TF f[ D + 1 ], sf = 0;
        for ( int k = 0; k <= D; ++k ) {
            f[ k ] = value_at( P[ k ] );
            sf += f[ k ];
        }
        m += vol * sf / ( D + 1 );

        const TF c1 = vol / ( ( D + 1 ) * ( D + 2 ) );
        for ( int j = 0; j <= D; ++j )
            for ( PI c = 0; c < D; ++c )
                mx[ c ] += c1 * P[ j ][ c ] * ( sf + f[ j ] );

        // int lambda_i lambda_j lambda_k = d! a! |S| / ( d + 3 )!, a! = 6, 2, 1 ( all equal, two equal, distinct )
        const TF base = factorial( D ) * vol / factorial( D + 3 );
        TF s2 = 0;
        for ( int i = 0; i <= D; ++i )
            for ( int j = 0; j <= D; ++j ) {
                TF pij = 0;
                for ( int c = 0; c < D; ++c )
                    pij += P[ i ][ c ] * P[ j ][ c ];
                for ( int k = 0; k <= D; ++k ) {
                    const int same = ( i == j ) + ( j == k ) + ( i == k );          // 3: all equal, 1: two, 0: distinct
                    s2 += pij * f[ k ] * ( same == 3 ? TF( 6 ) : same == 1 ? TF( 2 ) : TF( 1 ) );
                }
            }
        m2 += base * s2;
    }

    // ---- one moment and its adjoint, exact ( `diagram/Moments.h` ) ------------------------------------
    // `int p rho = |S| Phi( P, f )`, `f_k = rho( P_k )`, with `Phi` a polynomial of the vertices and of the values at the
    // vertices ( the closed forms above ). Its adjoint: the volume moves with the vertices, `Phi` with the vertices
    // directly and through `f_k` ( `d f_k / d P_k = grad rho` ), and the corners' values receive `d Phi / d f_k` times
    // the barycentric coordinates of `P_k`. Exact, where the quadrature of `PointwiseDensity` refines `|x|^2 rho`.

    static constexpr bool exact_moments = true;

    /// `Phi`, and with `grad` its derivatives: `dP[ q ][ c ] = d Phi / d P_q[ c ]` ( `f` fixed ), `df[ q ] = d Phi / d f_q`
    template<int K,bool grad>
    HD static TF moment_phi( const auto &P, const TF *f, TF ( *dP )[ ct_dim ], TF *df ) {
        constexpr int D = ct_dim;
        if constexpr ( grad )
            for ( int q = 0; q <= D; ++q ) {
                df[ q ] = 0;
                for ( int c = 0; c < D; ++c )
                    dP[ q ][ c ] = 0;
            }
        TF sf = 0;
        for ( int k = 0; k <= D; ++k )
            sf += f[ k ];
        if constexpr ( K == D + 1 ) {                    // int rho
            if constexpr ( grad )
                for ( int q = 0; q <= D; ++q )
                    df[ q ] = TF( 1 ) / ( D + 1 );
            return sf / ( D + 1 );
        } else if constexpr ( K < D ) {                  // int x_K rho
            const TF c1 = TF( 1 ) / ( ( D + 1 ) * ( D + 2 ) );
            TF r = 0, sx = 0;
            for ( int j = 0; j <= D; ++j ) {
                r += TF( P[ j ][ K ] ) * ( sf + f[ j ] );
                sx += TF( P[ j ][ K ] );
            }
            if constexpr ( grad )
                for ( int q = 0; q <= D; ++q ) {
                    dP[ q ][ K ] = c1 * ( sf + f[ q ] );
                    df[ q ] = c1 * ( sx + TF( P[ q ][ K ] ) );
                }
            return c1 * r;
        } else {                                         // int |x|^2 rho
            const TF base = factorial( D ) / factorial( D + 3 );
            TF s2 = 0;
            for ( int i = 0; i <= D; ++i )
                for ( int j = 0; j <= D; ++j ) {
                    TF pij = 0;
                    for ( int c = 0; c < D; ++c )
                        pij += TF( P[ i ][ c ] ) * TF( P[ j ][ c ] );
                    for ( int k = 0; k <= D; ++k ) {
                        const int same = ( i == j ) + ( j == k ) + ( i == k );
                        const TF wt = base * ( same == 3 ? TF( 6 ) : same == 1 ? TF( 2 ) : TF( 1 ) );
                        s2 += wt * pij * f[ k ];
                        if constexpr ( grad ) {
                            df[ k ] += wt * pij;
                            // `pij` is symmetric: `d / d P_i` sees `P_j` once from `( i, j )` and once from `( j, i )`
                            for ( int c = 0; c < D; ++c )
                                dP[ i ][ c ] += TF( 2 ) * wt * TF( P[ j ][ c ] ) * f[ k ];
                        }
                    }
                }
            return s2;
        }
    }

    template<int K>
    HD TF integrate_moment_over_simplex( const auto &P ) const {
        const TF det = edge_matrix( P ).determinant();
        TF f[ ct_dim + 1 ];
        for ( int k = 0; k <= ct_dim; ++k )
            f[ k ] = value_at( P[ k ] );
        return ( det < 0 ? -det : det ) / factorial( ct_dim ) * moment_phi<K,false>( P, f, nullptr, nullptr );
    }

    template<int K>
    HD void integrate_moment_over_simplex_bwd( const auto &P, TF g, auto &&grad_pts, auto &&grad_dist ) const {
        constexpr int D = ct_dim;
        const auto M = edge_matrix( P );
        const TF det = M.determinant();
        const TF vol = ( det < 0 ? -det : det ) / factorial( D );
        TF f[ D + 1 ], df[ D + 1 ], dP[ D + 1 ][ D ];
        for ( int k = 0; k <= D; ++k )
            f[ k ] = value_at( P[ k ] );
        const TF phi = moment_phi<K,true>( P, f, dP, df );

        // the volume
        const TF gv = ( det < 0 ? -g : g ) * phi / factorial( D );
        for ( PI r = 0; r < PI( D ); ++r ) {
            TF row_sum = 0;
            for ( PI c = 0; c < PI( D ); ++c ) {
                const TF minor = M.without_row_and_col( r, c ).determinant();
                const TF cof = ( ( r + c ) % 2 ? -minor : minor ) * gv;
                grad_pts[ c + 1 ][ r ] += cof;
                row_sum += cof;
            }
            grad_pts[ 0 ][ r ] -= row_sum;
        }

        // `Phi`, directly and through the values at the vertices
        const auto gr = gradient_at( P[ 0 ] );
        const TF gp = g * vol;
        for ( int q = 0; q <= D; ++q ) {
            for ( PI c = 0; c < PI( D ); ++c )
                grad_pts[ q ][ c ] += gp * ( dP[ q ][ c ] + df[ q ] * gr[ c ] );
            add_value_grad_at( grad_dist, P[ q ], gp * df[ q ] );
        }
    }
};

// The density of ONE ELEMENT of a mesh of degree 0 ( `Mesh( degree = 0 )` ): CONSTANT on the element, the mean of its corner
// values -- the contract `is_constant` ( `ConstantDensity.h` ), what the integration in a ball takes in 3D ( `InBall3.h` ). The
// mean keeps the mass of the corner weights, `sum_i values( e, i ) |e| / ( d + 1 )`, and its derivative goes to each corner by
// `1 / ( d + 1 )`.
template<class MeshT>
struct MeshElementConstDensity {
    using TF = typename MeshT::TF;
    static constexpr int ct_dim = MeshT::ct_dim;
    static constexpr bool is_constant = true;

    const MeshT *mesh;
    SI           e;                                      ///< the element
    TF           value;                                  ///< the mean of its corner values

    HD void add_value_grad( auto &&grad_dist, TF g ) const {
        if constexpr ( DECAYED_TYPE_OF( grad_dist.values )::is_valid )
            for ( int i = 0; i <= ct_dim; ++i )
                atomic_add( grad_dist.values( e, i ).ref(), g / ( ct_dim + 1 ) );
    }
};

/// the value of a piece's density at `x`: constant, or affine
template<class Dens>
HD auto density_value_at( const Dens &dens, const auto &x ) {
    if constexpr ( Dens::is_constant ) return dens.value;
    else                               return dens.value_at( x );
}

// A CONTINUOUS PIECEWISE LINEAR density on a mesh of simplices, seen as a density to integrate over a cell.
//
// A cell is cut by each simplex it meets, `cell INTERSECT simplex` being a piece ( the cuts are the simplex's `d + 1` facets,
// those the cell already satisfies are not made ); the simplices it meets are found in the bounding volume hierarchy
// that Python built ( `bvh_*` ), walked with the cell's own box. The facets carry `PIECE`, as for `Polytope`.
SDOT_TEMPLATE_DECL_FOR_Mesh
struct Mesh {
    SDOT_ATTRIBUTES_OF_Mesh

    static constexpr int ct_dim = DECAYED_TYPE_OF( nb_dims )::value;
    using TF = DECAYED_TYPE_OF( nodes )::TF;

    /// 0: a constant per element ( `MeshElementConstDensity` ), 1: affine ( `MeshElementDensity`, DG1 )
    static constexpr int ct_degree = DECAYED_TYPE_OF( degree )::value;
    static_assert( ct_degree == 0 || ct_degree == 1, "Mesh: degree 0 or 1" );

    /// the density of the element `e`
    HD auto density_of( SI e ) const {
        if constexpr ( ct_degree == 0 ) {
            TF s = 0;
            for ( int i = 0; i <= ct_dim; ++i )
                s += TF( values( e, i ) );
            return MeshElementConstDensity<Mesh>{ this, e, s / ( ct_dim + 1 ) };
        } else
            return MeshElementDensity<Mesh>{ this, e };
    }

    /// the cell is cut by the facets of the simplices: the integrator reserves a spare cell for it
    static constexpr bool cuts_pieces = true;

    /// what this density's adjoint derives, for loom's `computed_grads` ( `flags`: the distribution's, under one
    /// output ). The masses are never read by the integration: their derivative is known to be zero.
    static void computed_grads( auto &&flags ) {
        flags.values = true;   // `add_value_grad_at` -- not the nodes
        if constexpr ( requires { flags.current_mass; } ) flags.current_mass = true;
        if constexpr ( requires { flags.target_mass; } )  flags.target_mass  = true;
    }

    /// the integral of the density on the facets of a cell is computed by cutting the FACETS ( `for_each_facet_mass` ), not read off the
    /// pieces: what `Sweep.h` and `hessian_row` look for ( a plain trait: it resolves the same under every compiler )
    static constexpr bool cuts_facets = true;

    /// the box of the cell, a little WIDER: an element that only touches it -- along a side it may own, see `for_each_facet_mass` --
    /// must not be lost to a rounding of the cell's coordinates. `false`: the cell is unbounded, it has none
    HD bool box_of( const auto &cell, TF ( &clo )[ ct_dim ], TF ( &chi )[ ct_dim ] ) const {
        constexpr int D = ct_dim;
        if ( ! cell.bounded() )
            return false;
        const SI nv = SI( cell.nb_vertices() );
        for ( int c = 0; c < D; ++c ) {
            clo[ c ] = chi[ c ] = TF( cell.coord( 0, c ) );
            for ( SI v = 1; v < nv; ++v ) {
                const TF x = TF( cell.coord( int( v ), c ) );
                clo[ c ] = x < clo[ c ] ? x : clo[ c ];
                chi[ c ] = x > chi[ c ] ? x : chi[ c ];
            }
            const TF a = clo[ c ] < 0 ? -clo[ c ] : clo[ c ], b = chi[ c ] < 0 ? -chi[ c ] : chi[ c ];
            const TF m = TF( 1e-9 ) * ( chi[ c ] - clo[ c ] ) + TF( 1e-12 ) * ( a > b ? a : b );
            clo[ c ] -= m;
            chi[ c ] += m;
        }
        return true;
    }

    /// `body( e )` for each element of a leaf of the hierarchy that meets the box ( `bounded`: else, every element ); `body`
    /// answers `false` to stop
    HD void walk( bool bounded, const TF ( &clo )[ ct_dim ], const TF ( &chi )[ ct_dim ], auto &&body ) const {
        constexpr int D = ct_dim;
        SI stack[ 64 ];
        int top = 0;
        stack[ top++ ] = 0;
        while ( top ) {
            const SI n = stack[ --top ];
            if ( bounded ) {
                bool apart = false;
                for ( int c = 0; c < D && ! apart; ++c )
                    apart = TF( bvh_hi( n, c ) ) < clo[ c ] || TF( bvh_lo( n, c ) ) > chi[ c ];
                if ( apart )
                    continue;
            }
            const SI left = SI( bvh_links( n, 0 ) );
            if ( left >= 0 ) {
                stack[ top++ ] = SI( bvh_links( n, 1 ) );
                stack[ top++ ] = left;
                continue;
            }
            for ( SI e = SI( bvh_links( n, 2 ) ); e < SI( bvh_links( n, 3 ) ); ++e )
                if ( ! body( e ) )
                    return;
        }
    }

    HD void for_each_piece( const auto &cell, auto &&ws, auto &&func ) const {
        if ( cell.nb_vertices() == 0 )
            return;
        TF clo[ ct_dim ], chi[ ct_dim ];
        const bool bounded = box_of( cell, clo, chi );          // ( an UNBOUNDED cell has none: every simplex is tried )
        walk( bounded, clo, chi, [&]( SI e ) { return piece_of( e, cell, bounded, clo, chi, ws, func ); } );
    }

    // ---- THE FACETS, cut by the simplices ------------------------------------------------------------------------------------
    //
    // The integral of the density on a FACET of the cell ( the Hessian of the transport ) is NOT read off the pieces `cell
    // INTERSECT element`: an element on the far side of a facet that lies on a side of the mesh touches the cell by that facet only,
    // has no volume in it, and gives no piece. The facet itself is cut by the elements instead, wherever they are.
    //
    // A facet that lies on a side shared by two elements is held by both, and must count once. It is decided on the NODES -- the
    // same points for the two elements -- with one strict inequality: a node is OUTSIDE the facet's plane when `n . x - off > 0`,
    // INSIDE otherwise ( a node ON the plane is inside, for every element ), the normal `n` being in a canonical orientation ( the
    // first significant component positive ) so that every cell that sees the facet means the same side. The element's section by
    // the plane is the polygon through the edges that join an outside node to an inside one -- empty unless the plane separates
    // its nodes; so of the two elements that share the side, only the one whose third node is outside has a section: 1 and 0,
    // never one half. On the outer boundary of the mesh the same rule counts the facet or not: the density jumps there, and so
    // does the derivative -- a one-sided one.

    /// `func( q, part )` for each facet `q` of `cell` ( a cut that faces a seed ) and each element that cuts it: `part` the integral
    /// of the density on what the element holds of the facet, ACCUMULATED by the caller. `plane_of( r, dir, off )` gives the plane
    /// of cut `r` -- the bisector REBUILT from the seeds, not read back from the vertices: where four cells meet, an edge of length
    /// zero has a plane that the rounding alone decides, and the bisectors of two cells are exactly opposite. `on_segment( A, B, dens )`
    /// ( 2D ): the integral of the element's density `dens` on the part `[ A, B ]` of a facet -- by default all of it, inside a ball
    /// what the sphere leaves of it ( `InBall.h` ). In 3D, `on_segment( poly, m, dens )`: the same on the convex polygon `poly[ 0 .. m )`
    /// ( `polygon_mass` by default )
    HD void for_each_facet_mass( const auto &cell, auto &&plane_of, auto &&func ) const {
        for_each_facet_mass( cell, plane_of, func, []( const auto &A, const auto &B, const auto &dens ) -> TF {
            if constexpr ( ct_dim == 2 ) {
                const TF L = sdot::sqrt( ( B[ 0 ] - A[ 0 ] ) * ( B[ 0 ] - A[ 0 ] ) + ( B[ 1 ] - A[ 1 ] ) * ( B[ 1 ] - A[ 1 ] ) );
                return L * ( density_value_at( dens, A ) + density_value_at( dens, B ) ) / 2;
            } else
                return polygon_mass( A, B, dens );
        } );
    }

    HD void for_each_facet_mass( const auto &cell, auto &&plane_of, auto &&func, auto &&on_segment ) const {
        if ( cell.nb_vertices() == 0 )
            return;
        TF clo[ ct_dim ], chi[ ct_dim ];
        if ( ! box_of( cell, clo, chi ) )
            return;                                      // an unbounded cell has no facet to speak of
        const int nc = cell.nb_cuts();
        walk( true, clo, chi, [&]( SI e ) {
            for ( int q = 0; q < nc; ++q )
                if ( cell.cid[ q ] >= 0 ) {
                    const TF part = facet_part( e, cell, plane_of, q, on_segment );
                    if ( part != 0 )
                        func( q, part );
                }
            return true;
        } );
    }

    /// what element `e` holds of the facet `q` of `cell`: the section of the element by the facet's plane, clipped by the other cuts
    /// of the cell, `f` integrated on it ( affine: measure * `f` at the center )
    HD TF facet_part( SI e, const auto &cell, auto &&plane_of, int q, auto &&on_segment ) const {
        constexpr int D = ct_dim;
        TF n[ D ], off;
        plane_of( q, n, off );
        TF nn = 0;
        for ( int c = 0; c < D; ++c )
            nn += n[ c ] * n[ c ];
        nn = sdot::sqrt( nn );
        if ( ! ( nn > 0 ) )
            return 0;
        TF sg = 1;
        for ( int c = 0; c < D; ++c )
            if ( ( n[ c ] < 0 ? -n[ c ] : n[ c ] ) > TF( 1e-9 ) * nn ) {      // ( a noise of the rounding is not a direction )
                sg = n[ c ] < 0 ? TF( -1 ) : TF( 1 );
                break;
            }

        // the nodes against the plane, in the canonical orientation
        TF x[ D + 1 ][ D ], s[ D + 1 ], scale = off < 0 ? -off : off;
        for ( int i = 0; i <= D; ++i ) {
            const SI nd = SI( simplices( e, i ) );
            TF dot = 0;
            for ( int c = 0; c < D; ++c ) {
                x[ i ][ c ] = TF( nodes( nd, c ) );
                dot += n[ c ] * x[ i ][ c ];
            }
            s[ i ] = sg * ( dot - off );
            scale = ( dot < 0 ? -dot : dot ) > scale ? ( dot < 0 ? -dot : dot ) : scale;
        }
        const TF tol = TF( 1e-11 ) * scale;
        int outs[ D + 1 ], ins[ D + 1 ], no = 0, ni = 0;
        for ( int i = 0; i <= D; ++i ) {
            if ( s[ i ] > tol ) outs[ no++ ] = i;
            else                ins[ ni++ ] = i;
        }
        if ( no == 0 || ni == 0 )
            return 0;

        // the section: the point of each edge from an outside node to an inside one ( a node on the plane is its own )
        TF P[ 6 ][ D ];
        int np = 0;
        auto point = [&]( int o, int in ) {
            const TF t = s[ in ] > -tol ? TF( 1 ) : s[ o ] / ( s[ o ] - s[ in ] );     // ( an inside node on the plane is its own point )
            for ( int c = 0; c < D; ++c )
                P[ np ][ c ] = x[ o ][ c ] + t * ( x[ in ][ c ] - x[ o ][ c ] );
            ++np;
        };
        if ( D == 3 && no == 2 && ni == 2 ) {            // a quadrilateral: its vertices in a cycle
            point( outs[ 0 ], ins[ 0 ] ); point( outs[ 0 ], ins[ 1 ] ); point( outs[ 1 ], ins[ 1 ] ); point( outs[ 1 ], ins[ 0 ] );
        } else {
            for ( int a = 0; a < no; ++a )
                for ( int b = 0; b < ni; ++b )
                    point( outs[ a ], ins[ b ] );
        }

        const auto dens = density_of( e );
        const int nc = cell.nb_cuts();
        if constexpr ( D == 2 ) {
            // a segment `[ P0, P1 ]`, clipped by the cell's other edges
            TF t0 = 0, t1 = 1;
            for ( int r = 0; r < nc && t0 < t1; ++r ) {
                if ( r == q )
                    continue;
                TF dr[ 2 ], orr;
                plane_of( r, dr, orr );
                const TF sa = dr[ 0 ] * P[ 0 ][ 0 ] + dr[ 1 ] * P[ 0 ][ 1 ] - orr, sb = dr[ 0 ] * P[ 1 ][ 0 ] + dr[ 1 ] * P[ 1 ][ 1 ] - orr;
                if ( sa > 0 && sb > 0 ) { t1 = t0; break; }
                if ( sa > 0 )      t0 = t0 > sa / ( sa - sb ) ? t0 : sa / ( sa - sb );
                else if ( sb > 0 ) t1 = t1 < sa / ( sa - sb ) ? t1 : sa / ( sa - sb );
            }
            if ( ! ( t1 > t0 ) )
                return 0;
            const TF A[ 2 ] = { P[ 0 ][ 0 ] + t0 * ( P[ 1 ][ 0 ] - P[ 0 ][ 0 ] ), P[ 0 ][ 1 ] + t0 * ( P[ 1 ][ 1 ] - P[ 0 ][ 1 ] ) };
            const TF B[ 2 ] = { P[ 0 ][ 0 ] + t1 * ( P[ 1 ][ 0 ] - P[ 0 ][ 0 ] ), P[ 0 ][ 1 ] + t1 * ( P[ 1 ][ 1 ] - P[ 0 ][ 1 ] ) };
            return TF( on_segment( A, B, dens ) );
        } else {
            // a polygon, clipped by the cell's other faces ( Sutherland-Hodgman ), then its area times `f` at its center
            constexpr int CAP = 64;
            TF poly[ CAP ][ 3 ], next[ CAP ][ 3 ];
            int m = np;
            for ( int k = 0; k < m; ++k )
                for ( int c = 0; c < 3; ++c )
                    poly[ k ][ c ] = P[ k ][ c ];
            for ( int r = 0; r < nc && m >= 3; ++r ) {
                if ( r == q )
                    continue;
                TF dr[ 3 ], orr;
                plane_of( r, dr, orr );
                int mm = 0;
                TF sj = dr[ 0 ] * poly[ 0 ][ 0 ] + dr[ 1 ] * poly[ 0 ][ 1 ] + dr[ 2 ] * poly[ 0 ][ 2 ] - orr;
                for ( int j = 0; j < m && mm + 2 <= CAP; ++j ) {
                    const int k = j + 1 < m ? j + 1 : 0;
                    const TF sk = dr[ 0 ] * poly[ k ][ 0 ] + dr[ 1 ] * poly[ k ][ 1 ] + dr[ 2 ] * poly[ k ][ 2 ] - orr;
                    if ( ! ( sj > 0 ) ) {
                        for ( int c = 0; c < 3; ++c ) next[ mm ][ c ] = poly[ j ][ c ];
                        ++mm;
                    }
                    if ( ( sj > 0 ) != ( sk > 0 ) ) {
                        const TF t = sj / ( sj - sk );
                        for ( int c = 0; c < 3; ++c ) next[ mm ][ c ] = poly[ j ][ c ] + t * ( poly[ k ][ c ] - poly[ j ][ c ] );
                        ++mm;
                    }
                    sj = sk;
                }
                m = mm;
                for ( int k = 0; k < m; ++k )
                    for ( int c = 0; c < 3; ++c )
                        poly[ k ][ c ] = next[ k ][ c ];
            }
            if ( m < 3 )
                return 0;
            return TF( on_segment( poly, m, dens ) );
        }
    }

    /// the integral of the density `dens` ( constant or affine ) on the convex polygon `poly[ 0 .. m )` of 3D: its area times the
    /// value at its center
    HD static TF polygon_mass( const auto &poly, int m, const auto &dens ) {
        TF area = 0, g[ 3 ] = { 0, 0, 0 };
        for ( int k = 1; k + 1 < m; ++k ) {
            const TF ax = poly[ k ][ 0 ] - poly[ 0 ][ 0 ], ay = poly[ k ][ 1 ] - poly[ 0 ][ 1 ], az = poly[ k ][ 2 ] - poly[ 0 ][ 2 ];
            const TF bx = poly[ k + 1 ][ 0 ] - poly[ 0 ][ 0 ], by = poly[ k + 1 ][ 1 ] - poly[ 0 ][ 1 ], bz = poly[ k + 1 ][ 2 ] - poly[ 0 ][ 2 ];
            const TF cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
            const TF t = sdot::sqrt( cx * cx + cy * cy + cz * cz ) / 2;
            area += t;
            for ( int c = 0; c < 3; ++c )
                g[ c ] += t * ( poly[ 0 ][ c ] + poly[ k ][ c ] + poly[ k + 1 ][ c ] ) / 3;
        }
        if ( ! ( area > 0 ) )
            return 0;
        for ( int c = 0; c < 3; ++c )
            g[ c ] /= area;
        return area * density_value_at( dens, g );
    }

    /// the piece `cell INTERSECT element e`, to `func`. `false` when the cutting ran out of room
    HD bool piece_of( SI e, const auto &cell, bool bounded, const TF ( &clo )[ ct_dim ], const TF ( &chi )[ ct_dim ], auto &&ws, auto &&func ) const {
        constexpr int D = ct_dim;
        const SI nv = SI( cell.nb_vertices() );

        // the box of the element
        if ( bounded ) {
            bool apart = false;
            for ( int c = 0; c < D && ! apart; ++c ) {
                TF lo = TF( nodes( SI( simplices( e, 0 ) ), c ) ), hi = lo;
                for ( int i = 1; i <= D; ++i ) {
                    const TF x = TF( nodes( SI( simplices( e, i ) ), c ) );
                    lo = x < lo ? x : lo;
                    hi = x > hi ? x : hi;
                }
                apart = hi < clo[ c ] || lo > chi[ c ];
            }
            if ( apart )
                return true;
        }

        // the facets: `lambda_i >= 0`, i.e. `n . x <= off` with `n = - G_i / |G_i|`
        bool started = false;
        for ( int i = 0; i <= D; ++i ) {
            TF g2 = 0, gx0 = 0;
            for ( int c = 0; c < D; ++c ) {
                const TF g = TF( grads( e, i, c ) );
                g2 += g * g;
                gx0 += g * TF( nodes( SI( simplices( e, 0 ) ), c ) );
            }
            const TF gn = sdot::sqrt( g2 );
            const auto n = Vector<TF,D>::with_func( [&]( PI c ) { return - TF( grads( e, i, c ) ) / gn; } );
            const TF off = ( ( i == 0 ? TF( 1 ) : TF( 0 ) ) - gx0 ) / gn;
            if ( bounded ) {
                // up to the rounding of the kernel float ( see `Polytope.h` ): a cell that starts from the mesh's boundary, its domain,
                // has vertices on the outer facets, and cutting there again would leave twins -- flat simplices, whose orientation
                // the adjoints ( `d|det|` ) cannot read
                using TK = std::decay_t<decltype( cell.coord( 0, 0 ) )>;
                const TF tol = TF( 64 ) * TF( std::numeric_limits<TK>::epsilon() );
                bool crosses = false;
                for ( SI v = 0; v < nv && ! crosses; ++v ) {
                    TF s = - off, scale = off < 0 ? - off : off;
                    for ( int c = 0; c < D; ++c ) {
                        const TF t = n[ c ] * TF( cell.coord( int( v ), c ) );
                        s += t;
                        scale += t < 0 ? - t : t;
                    }
                    crosses = s > tol * scale;
                }
                if ( ! crosses )
                    continue;
            }
            if ( ! ( started ? ws.cut( n, off ) : ws.start( cell, n, off ) ) )
                return false;                            // no room: recorded by `ws`, the host relaunches with more
            started = true;
            if ( ws.nb_vertices() == 0 )
                return true;                             // the element does not meet the cell
        }

        const auto dens = density_of( e );
        if ( started ) ws.with_current( [&]( const auto &piece ) { func( piece, dens ); } );
        else           func( cell, dens );
        return true;
    }
};

} // namespace sdot
