#pragma once

// the members + the axes this body names, written to the build include tree by `CallArg_Aggregate`.
#include <sdot/generated/aggregates/CodimMesh.h>
#include <loom/support/common_macros.h>
#include <loom/support/containers/Vector.h>
#include <loom/support/containers/Matrix.h>
#include <loom/support/atomic_add.h>
#include <loom/support/math.h>
#include <sdot/integration/CodimMeshIntegrals.h>
#include <sdot/PointwiseDensity.h>
#include <limits>

namespace sdot {

// =====================================================================================
// A DENSITY CARRIED BY A MESH OF LOWER DIMENSION ( lines in 2D or 3D, surfaces in 3D ), smoothed at the width `s`
// ( `distributions/CodimMesh.py`, design: `grill/codim-mesh-density/decisions.md` ). Its support is the ENVELOPE of the mesh
// ( `Envelope.h` ): simplices where the distance `d` to the mesh and the density `rho` ( times the weight of the fans ) are
// affine, given by their corners ( `dists`, `values` ). On them,
//
//     rho( x ) exp( -d( x )^2 / s^2 ),
//
// integrated in closed form ( `integration/GaussDividedDifferences.h` ): with `u = d / s` at the vertices of a simplex `P`,
//
//     int_P lambda^a exp( -u^2 ) = |det| a! G( 0, u + the repeated nodes ),
//
// so `int_P rho exp( -u^2 ) = |det| sum_q f_q G( 0, u, u_q )`, `f_q = rho( P_q )`. The moments add the vertices once or twice
// more, the adjoint the derivative of `G` with respect to a node ( `G( 1, ..., u_a )`, times its multiplicity ).
//
// THE DISTRIBUTION IS THE MEASURE `rho` ON M ( `CodimMeasureDensity` below ); the smoothed densities are the stages of the width
// continuation, at `conv_s` ( `Convolved<CodimMesh>::at` ), down to the smallest, `params( 0 )`. `params( 2 ) != 0` gives the smoothed
// density at `params( 0 )` instead of the measure ( a check ); with `params( 1 ) != 0`, its pieces are integrated by
// `PointwiseDensity` ( another one ).
// =====================================================================================


/// the density of ONE element of the envelope on one piece ( the contract `is_constant == false` of
/// `distributions/Distribution.py` )
template<class MeshT>
struct CodimMeshElementDensity {
    using TF = typename MeshT::TF;
    static constexpr int ct_dim = MeshT::ct_dim;
    static constexpr bool is_constant = false;
    static constexpr int D = ct_dim;

    const MeshT *mesh;
    SI           e;                                      ///< the element
    TF           s;                                      ///< the width

    HD TF lambda( int i, const auto &x ) const {
        const SI n0 = SI( mesh->simplices( e, 0 ) );
        TF r = i == 0 ? TF( 1 ) : TF( 0 );
        for ( int c = 0; c < D; ++c )
            r += TF( mesh->grads( e, i, c ) ) * ( TF( x[ c ] ) - TF( mesh->nodes( n0, c ) ) );
        return r;
    }
    HD TF rho_at( const auto &x ) const { TF r = 0; for ( int i = 0; i <= D; ++i ) r += lambda( i, x ) * TF( mesh->values( e, i ) ); return r; }
    HD TF d_at( const auto &x ) const { TF r = 0; for ( int i = 0; i <= D; ++i ) r += lambda( i, x ) * TF( mesh->dists( e, i ) ); return r; }
    HD TF grad_of( const auto &field, int c ) const { TF r = 0; for ( int i = 0; i <= D; ++i ) r += field( i ) * TF( mesh->grads( e, i, c ) ); return r; }
    HD TF grad_rho( int c ) const { return grad_of( [&]( int i ) { return TF( mesh->values( e, i ) ); }, c ); }
    HD TF grad_d( int c ) const { return grad_of( [&]( int i ) { return TF( mesh->dists( e, i ) ); }, c ); }

    // ---- pointwise ( `PointwiseDensity` ) -----------------------------------------------------------------------------------

    HD TF value_at( const auto &x ) const {
        const TF u = d_at( x ) / s;
        return rho_at( x ) * sdot::exp( - u * u );
    }
    HD Vector<TF,ct_dim> gradient_at( const auto &x ) const {
        const TF d = d_at( x ), g = sdot::exp( - d * d / ( s * s ) ), r = rho_at( x );
        return Vector<TF,ct_dim>::with_func( [&]( PI c ) { return g * ( grad_rho( int( c ) ) - 2 * r * d / ( s * s ) * grad_d( int( c ) ) ); } );
    }
    HD void add_value_grad_at( auto &&grad_dist, const auto &x, TF g ) const {
        if constexpr ( DECAYED_TYPE_OF( grad_dist.values )::is_valid ) {
            const TF u = d_at( x ) / s, w = g * sdot::exp( - u * u );
            for ( int i = 0; i <= D; ++i )
                atomic_add( grad_dist.values( e, i ).ref(), w * lambda( i, x ) );
        }
    }

    // ---- the simplex integrals, closed forms ---------------------------------------------------------------------------------

    HD void at_vertices( const auto &P, TF *u, TF *f ) const {
        for ( int q = 0; q <= D; ++q ) {
            u[ q ] = d_at( P[ q ] ) / s;
            f[ q ] = rho_at( P[ q ] );
        }
    }

    HD TF integrate_over_simplex( const auto &P ) const {
        const TF det = codim::det_of<D,TF>( P );
        TF u[ D + 1 ], f[ D + 1 ];
        at_vertices( P, u, f );
        return ( det < 0 ? -det : det ) * codim::weighted_g( D, u, f );
    }

    /// the adjoint: `I = |det| S( u, f )`. The volume moves with the vertices ( cofactors ); `f_q` moves with `P_q` along the
    /// gradient of `rho`, `u_a` along that of `d / s`, `dS / du_a = sum_q f_q mult_a G( 1, u, u_q, u_a )`; the corners' values
    /// receive `|det| sum_q lambda_i( P_q ) G( 0, u, u_q )`
    HD void integrate_over_simplex_bwd( const auto &P, TF g, auto &&grad_pts, auto &&grad_dist ) const {
        const auto M = Matrix<TF,D>::with_func( [&]( auto r, auto c ) { return TF( P[ c + 1 ][ r ] - P[ 0 ][ r ] ); } );
        const TF det = M.determinant(), adet = det < 0 ? -det : det;
        TF u[ D + 1 ], f[ D + 1 ], E[ D + 1 ], dS[ D + 1 ];
        at_vertices( P, u, f );
        TF S = 0;
        for ( int a = 0; a <= D; ++a ) dS[ a ] = 0;
        for ( int q = 0; q <= D; ++q ) {
            TF x[ gauss_dd::MAX_NODES ];
            for ( int i = 0; i <= D; ++i ) x[ i ] = u[ i ];
            x[ D + 1 ] = u[ q ];
            E[ q ] = gauss_dd::G( 0, x, D + 1 );
            S += f[ q ] * E[ q ];
            for ( int a = 0; a <= D; ++a ) {
                TF y[ gauss_dd::MAX_NODES ];
                for ( int i = 0; i <= D; ++i ) y[ i ] = u[ i ];
                y[ D + 1 ] = u[ q ];
                y[ D + 2 ] = u[ a ];
                dS[ a ] += f[ q ] * TF( a == q ? 2 : 1 ) * gauss_dd::G( 1, y, D + 2 );
            }
        }

        const TF gv = ( det < 0 ? -g : g ) * S;
        for ( PI r = 0; r < D; ++r ) {
            TF row_sum = 0;
            for ( PI c = 0; c < D; ++c ) {
                const TF minor = M.without_row_and_col( r, c ).determinant();
                const TF cof = ( ( r + c ) % 2 ? -minor : minor ) * gv;
                grad_pts[ c + 1 ][ r ] += cof;
                row_sum += cof;
            }
            grad_pts[ 0 ][ r ] -= row_sum;
        }
        for ( int a = 0; a <= D; ++a )
            for ( int c = 0; c < D; ++c )
                grad_pts[ a ][ c ] += g * adet * ( E[ a ] * grad_rho( c ) + dS[ a ] * grad_d( c ) / s );

        if constexpr ( DECAYED_TYPE_OF( grad_dist.values )::is_valid )
            for ( int i = 0; i <= D; ++i ) {
                TF w = 0;
                for ( int q = 0; q <= D; ++q ) w += lambda( i, P[ q ] ) * E[ q ];
                atomic_add( grad_dist.values( e, i ).ref(), g * adet * w );
            }
    }

    /// `int rho`, `int x rho`, `int |x|^2 rho` over the simplex, ACCUMULATED ( the vertices once, twice, three times more )
    HD void integrate_moments_over_simplex( const auto &P, TF &m, auto &mx, TF &m2 ) const {
        const TF det = codim::det_of<D,TF>( P ), adet = det < 0 ? -det : det;
        TF u[ D + 1 ], f[ D + 1 ];
        at_vertices( P, u, f );
        auto G = [&]( int k, const int *extra ) {
            TF x[ gauss_dd::MAX_NODES ];
            for ( int i = 0; i <= D; ++i ) x[ i ] = u[ i ];
            for ( int j = 0; j < k; ++j ) x[ D + 1 + j ] = u[ extra[ j ] ];
            return gauss_dd::G( 0, x, D + k );
        };
        for ( int q = 0; q <= D; ++q ) {
            const int e1[ 1 ] = { q };
            m += adet * f[ q ] * G( 1, e1 );
            for ( int r = 0; r <= D; ++r ) {
                const int e2[ 2 ] = { q, r };
                const TF g2 = adet * f[ q ] * TF( q == r ? 2 : 1 ) * G( 2, e2 );
                for ( int c = 0; c < D; ++c )
                    mx[ c ] += g2 * TF( P[ r ][ c ] );
                for ( int t = 0; t <= D; ++t ) {
                    const int e3[ 3 ] = { q, r, t };
                    const int same = ( q == r ) + ( r == t ) + ( q == t );                 // 3: all equal, 1: two, 0: distinct
                    TF prt = 0;
                    for ( int c = 0; c < D; ++c )
                        prt += TF( P[ r ][ c ] ) * TF( P[ t ][ c ] );
                    m2 += adet * f[ q ] * prt * ( same == 3 ? TF( 6 ) : same == 1 ? TF( 2 ) : TF( 1 ) ) * G( 3, e3 );
                }
            }
        }
    }

    /// `int rho exp( -u^2 )` over a simplex of dimension `D - 1` ( `D` points `T`, a part of a facet ), of measure `meas`
    HD TF integrate_over_facet_simplex( const TF ( &T )[ ct_dim ][ ct_dim ], TF meas ) const {
        TF u[ D ], f[ D ], fact = 1;
        for ( int q = 0; q < D; ++q ) {
            u[ q ] = d_at( T[ q ] ) / s;
            f[ q ] = rho_at( T[ q ] );
        }
        for ( int j = 2; j < D; ++j ) fact *= j;
        return meas * fact * codim::weighted_g( D - 1, u, f );
    }
};

// ---- THE MEASURE ON M ITSELF ( the last stage, `s = 0` ) --------------------------------------------------------------------
//
// `rho` on the simplices of M ( segments in 2D or 3D, triangles in 3D ), P1 on their nodes. A cell is not cut: each simplex `P` of
// its fan gets, for each element `sigma` of M near the cell, `int_{ P ∩ sigma } rho` -- `sigma` clipped by the `D + 1` half-spaces
// `lambda_i >= 0` of `P`. The ADJOINT moves the clipped boundary: on the part of `P ∩ sigma`'s boundary lying on the face `i` of
// `P`, moving the vertex `p_a` by `delta` changes `lambda_i` by `- lambda_a grad lambda_i . delta`, hence
//
//     d / dp_a int_{ P ∩ sigma } rho = - grad lambda_i int_{ boundary on face i } rho lambda_a / | grad_sigma lambda_i |,
//
// `grad_sigma` the gradient along `sigma` ( a point in the case of a segment, an edge in that of a triangle ).

namespace codim {

/// the barycentric coordinates of a simplex `P` ( `D + 1` points ): `lambda_i( x ) = [ i == 0 ] + G[ i ] . ( x - P_0 )`
template<int D,class TF>
struct Bary {
    TF p0[ D ], G[ D + 1 ][ D ];
    bool ok = true;

    HD Bary( const auto &P ) {
        TF A[ D ][ 2 * D ];                                  // ( [ E | I ], Gauss-Jordan: E^-1, E's columns the edges )
        for ( int r = 0; r < D; ++r ) {
            p0[ r ] = TF( P[ 0 ][ r ] );
            for ( int c = 0; c < D; ++c ) { A[ r ][ c ] = TF( P[ c + 1 ][ r ] ) - TF( P[ 0 ][ r ] ); A[ r ][ D + c ] = r == c; }
        }
        for ( int c = 0; c < D; ++c ) {
            int b = c;
            for ( int r = c + 1; r < D; ++r ) if ( ( A[ r ][ c ] < 0 ? -A[ r ][ c ] : A[ r ][ c ] ) > ( A[ b ][ c ] < 0 ? -A[ b ][ c ] : A[ b ][ c ] ) ) b = r;
            if ( A[ b ][ c ] == 0 ) { ok = false; return; }
            for ( int k = 0; k < 2 * D; ++k ) { const TF t = A[ b ][ k ]; A[ b ][ k ] = A[ c ][ k ]; A[ c ][ k ] = t; }
            const TF piv = A[ c ][ c ];
            for ( int k = 0; k < 2 * D; ++k ) A[ c ][ k ] /= piv;
            for ( int r = 0; r < D; ++r )
                if ( r != c ) { const TF f = A[ r ][ c ]; for ( int k = 0; k < 2 * D; ++k ) A[ r ][ k ] -= f * A[ c ][ k ]; }
        }
        for ( int i = 0; i < D; ++i )                        // ( row i of E^-1: the gradient of lambda_{ i + 1 } )
            for ( int c = 0; c < D; ++c ) G[ i + 1 ][ c ] = A[ i ][ D + c ];
        for ( int c = 0; c < D; ++c ) { G[ 0 ][ c ] = 0; for ( int i = 1; i <= D; ++i ) G[ 0 ][ c ] -= G[ i ][ c ]; }
    }
    HD TF lambda( int i, const TF *x ) const {
        TF r = i == 0 ? TF( 1 ) : TF( 0 );
        for ( int c = 0; c < D; ++c ) r += G[ i ][ c ] * ( x[ c ] - p0[ c ] );
        return r;
    }
};

} // namespace codim

/// `rho` on ONE element of M, integrated over the simplices of a cell ( the contract `is_constant == false` )
template<class MeshT>
struct CodimMeasureDensity {
    using TF = typename MeshT::TF;
    static constexpr int ct_dim = MeshT::ct_dim;
    static constexpr bool is_constant = false;
    static constexpr int D = ct_dim;

    const MeshT *mesh;
    SI           e;                                      ///< the element of M

    HD int nb_verts() const { return int( mesh->m_simplices.shape( 1 ) ); }
    HD SI node( int i ) const { return SI( mesh->m_simplices( e, i ) ); }
    HD TF X( int i, int c ) const { return TF( mesh->m_nodes( node( i ), c ) ); }
    HD TF R( int i ) const { return TF( mesh->m_rho( node( i ) ) ); }

    /// the part of the element in `P`, given to `f`:
    ///   * a segment: `f( t0, t1, lo, hi )`, the parameters on `[ A, B ]` and the faces of `P` that bound them ( -1: none );
    ///   * a triangle: `f( np, a[], b[], lab[] )`, the polygon in the parameters `( a, b )` of `A + a ( B - A ) + b ( C - A )`,
    ///     `lab[ k ]` the face of `P` that holds its edge from vertex `k` to `k + 1` ( -1: an edge of the element ).
    template<class F>
    HD void clip( const codim::Bary<D,TF> &bc, F &&f ) const {
        if ( nb_verts() == 2 ) {
            TF A[ D ], B[ D ];
            for ( int c = 0; c < D; ++c ) { A[ c ] = X( 0, c ); B[ c ] = X( 1, c ); }
            TF t0 = 0, t1 = 1;
            int lo = -1, hi = -1;
            for ( int i = 0; i <= D; ++i ) {
                const TF l0 = bc.lambda( i, A ), l1 = bc.lambda( i, B ), dl = l1 - l0;
                if ( dl == 0 ) { if ( l0 < 0 ) return; continue; }
                const TF t = - l0 / dl;                          // lambda_i( t ) = l0 + t dl >= 0
                if ( dl > 0 ) { if ( t > t0 ) { t0 = t; lo = i; } }
                else          { if ( t < t1 ) { t1 = t; hi = i; } }
            }
            if constexpr ( requires { f( t0, t1, lo, hi ); } )
                if ( t1 > t0 ) f( t0, t1, lo, hi );
        } else if constexpr ( D == 3 ) {
            constexpr int CAP = 16;
            TF a[ CAP ] = { 0, 1, 0 }, b[ CAP ] = { 0, 0, 1 }, na[ CAP ], nb[ CAP ];
            int lab[ CAP ] = { -1, -1, -1 }, nl[ CAP ], np = 3;
            TF A[ 3 ], B[ 3 ], C[ 3 ];
            for ( int c = 0; c < 3; ++c ) { A[ c ] = X( 0, c ); B[ c ] = X( 1, c ); C[ c ] = X( 2, c ); }
            for ( int i = 0; i <= 3 && np >= 3; ++i ) {
                const TF l0 = bc.lambda( i, A ), la = bc.lambda( i, B ) - l0, lb = bc.lambda( i, C ) - l0;
                int m = 0;
                for ( int k = 0; k < np && m + 2 <= CAP; ++k ) {
                    const int k1 = k + 1 < np ? k + 1 : 0;
                    const TF s0 = l0 + la * a[ k ] + lb * b[ k ], s1 = l0 + la * a[ k1 ] + lb * b[ k1 ];
                    const bool in0 = s0 >= 0, in1 = s1 >= 0;
                    if ( in0 ) { na[ m ] = a[ k ]; nb[ m ] = b[ k ]; nl[ m ] = lab[ k ]; ++m; }
                    if ( in0 != in1 ) {
                        const TF t = s0 / ( s0 - s1 );
                        na[ m ] = a[ k ] + t * ( a[ k1 ] - a[ k ] ); nb[ m ] = b[ k ] + t * ( b[ k1 ] - b[ k ] );
                        nl[ m ] = in0 ? i : lab[ k ];                // ( leaving: along the face to the next entry )
                        ++m;
                    }
                }
                np = m;
                for ( int k = 0; k < np; ++k ) { a[ k ] = na[ k ]; b[ k ] = nb[ k ]; lab[ k ] = nl[ k ]; }
            }
            if constexpr ( requires { f( np, a, b, lab ); } )
                if ( np >= 3 ) f( np, a, b, lab );
        }
    }

    HD void point_at( TF t, TF *x ) const { for ( int c = 0; c < D; ++c ) x[ c ] = X( 0, c ) + t * ( X( 1, c ) - X( 0, c ) ); }
    HD void point_at( TF a, TF b, TF *x ) const { for ( int c = 0; c < D; ++c ) x[ c ] = X( 0, c ) + a * ( X( 1, c ) - X( 0, c ) ) + b * ( X( 2, c ) - X( 0, c ) ); }
    HD TF length() const { TF l = 0; for ( int c = 0; c < D; ++c ) l += ( X( 1, c ) - X( 0, c ) ) * ( X( 1, c ) - X( 0, c ) ); return sdot::sqrt( l ); }
    /// `|( B - A ) x ( C - A )|` ( a triangle )
    HD TF jacobian() const {
        TF u[ 3 ], v[ 3 ];
        for ( int c = 0; c < 3; ++c ) { u[ c ] = X( 1, c ) - X( 0, c ); v[ c ] = X( 2, c ) - X( 0, c ); }
        const TF n0 = u[ 1 ] * v[ 2 ] - u[ 2 ] * v[ 1 ], n1 = u[ 2 ] * v[ 0 ] - u[ 0 ] * v[ 2 ], n2 = u[ 0 ] * v[ 1 ] - u[ 1 ] * v[ 0 ];
        return sdot::sqrt( n0 * n0 + n1 * n1 + n2 * n2 );
    }

    /// `func( weight, x, phi )` over a rule exact to degree 3 on `P ∩ sigma`, `phi[ i ]` the P1 basis of the element's vertex `i`
    template<class F>
    HD void for_each_point( const auto &P, F &&func ) const {
        const codim::Bary<D,TF> bc( P );
        if ( ! bc.ok ) return;
        if ( nb_verts() == 2 ) {
            clip( bc, [&]( TF t0, TF t1, int, int ) {
                const TF L = length() * ( t1 - t0 );
                const TF gx[ 3 ] = { TF( 0.1127016653792583 ), TF( 0.5 ), TF( 0.8872983346207417 ) }, gw[ 3 ] = { TF( 5 ) / 18, TF( 8 ) / 18, TF( 5 ) / 18 };
                for ( int q = 0; q < 3; ++q ) {
                    const TF t = t0 + gx[ q ] * ( t1 - t0 );
                    TF x[ D ], phi[ 3 ] = { 1 - t, t, 0 };
                    point_at( t, x );
                    func( L * gw[ q ], x, phi );
                }
            } );
        } else {
            clip( bc, [&]( int np, const TF *a, const TF *b, const int * ) {
                const TF J = jacobian();
                // ( a fan; on each triangle the rule of degree 3: the centroid -27/48, ( 3/5, 1/5, 1/5 ) 25/48 )
                for ( int k = 1; k + 1 < np; ++k ) {
                    const TF ta[ 3 ] = { a[ 0 ], a[ k ], a[ k + 1 ] }, tb[ 3 ] = { b[ 0 ], b[ k ], b[ k + 1 ] };
                    const TF area = ( ( ta[ 1 ] - ta[ 0 ] ) * ( tb[ 2 ] - tb[ 0 ] ) - ( ta[ 2 ] - ta[ 0 ] ) * ( tb[ 1 ] - tb[ 0 ] ) ) / 2;
                    const TF aa = ( area < 0 ? -area : area ) * J;
                    for ( int q = 0; q < 4; ++q ) {
                        TF w3[ 3 ];
                        if ( q == 0 ) { w3[ 0 ] = w3[ 1 ] = w3[ 2 ] = TF( 1 ) / 3; }
                        else { for ( int j = 0; j < 3; ++j ) w3[ j ] = j == q - 1 ? TF( 0.6 ) : TF( 0.2 ); }
                        const TF pa = w3[ 0 ] * ta[ 0 ] + w3[ 1 ] * ta[ 1 ] + w3[ 2 ] * ta[ 2 ];
                        const TF pb = w3[ 0 ] * tb[ 0 ] + w3[ 1 ] * tb[ 1 ] + w3[ 2 ] * tb[ 2 ];
                        TF x[ D ], phi[ 3 ] = { 1 - pa - pb, pa, pb };
                        point_at( pa, pb, x );
                        func( aa * ( q == 0 ? TF( -27 ) / 48 : TF( 25 ) / 48 ), x, phi );
                    }
                }
            } );
        }
    }

    HD TF rho_of( const TF *phi ) const { TF r = 0; for ( int i = 0; i < nb_verts(); ++i ) r += phi[ i ] * R( i ); return r; }

    HD TF integrate_over_simplex( const auto &P ) const {
        TF s = 0;
        for_each_point( P, [&]( TF w, const TF *, const TF *phi ) { s += w * rho_of( phi ); } );
        return s;
    }

    HD void integrate_moments_over_simplex( const auto &P, TF &m, auto &mx, TF &m2 ) const {
        for_each_point( P, [&]( TF w, const TF *x, const TF *phi ) {
            const TF r = w * rho_of( phi );
            TF x2 = 0;
            for ( int c = 0; c < D; ++c ) { mx[ c ] += r * x[ c ]; x2 += x[ c ] * x[ c ]; }
            m += r;
            m2 += r * x2;
        } );
    }

    /// see the head of the section: the boundary of `P ∩ sigma` on the faces of `P`
    HD void integrate_over_simplex_bwd( const auto &P, TF g, auto &&grad_pts, auto &&grad_dist ) const {
        const codim::Bary<D,TF> bc( P );
        if ( ! bc.ok ) return;
        // `- g grad lambda_i int rho lambda_a / | grad_sigma lambda_i |` on the boundary part `[ X0, X1 ]` ( a point: `X0` )
        auto add = [&]( int i, const TF *X0, const TF *X1, TF r0, TF r1, TF len, TF gs ) {
            if ( ! ( gs > 0 ) ) return;
            for ( int a = 0; a <= D; ++a ) {
                TF v;
                if ( len < 0 ) v = r0 * bc.lambda( a, X0 );     // ( a point )
                else {
                    TF xm[ D ];
                    for ( int c = 0; c < D; ++c ) xm[ c ] = ( X0[ c ] + X1[ c ] ) / 2;
                    v = len / 6 * ( r0 * bc.lambda( a, X0 ) + 4 * ( r0 + r1 ) / 2 * bc.lambda( a, xm ) + r1 * bc.lambda( a, X1 ) );
                }
                for ( int c = 0; c < D; ++c )
                    grad_pts[ a ][ c ] -= g * v / gs * bc.G[ i ][ c ];
            }
        };
        if ( nb_verts() == 2 ) {
            TF u[ D ], l = length();
            for ( int c = 0; c < D; ++c ) u[ c ] = ( X( 1, c ) - X( 0, c ) ) / l;
            clip( bc, [&]( TF t0, TF t1, int lo, int hi ) {
                for ( int side = 0; side < 2; ++side ) {
                    const int i = side ? hi : lo;
                    if ( i < 0 ) continue;
                    const TF t = side ? t1 : t0;
                    TF x[ D ], gu = 0;
                    point_at( t, x );
                    for ( int c = 0; c < D; ++c ) gu += bc.G[ i ][ c ] * u[ c ];
                    add( i, x, x, R( 0 ) + t * ( R( 1 ) - R( 0 ) ), 0, -1, gu < 0 ? -gu : gu );
                }
            } );
        } else if constexpr ( D == 3 ) {
            TF nu[ 3 ], uu[ 3 ], vv[ 3 ];
            for ( int c = 0; c < 3; ++c ) { uu[ c ] = X( 1, c ) - X( 0, c ); vv[ c ] = X( 2, c ) - X( 0, c ); }
            nu[ 0 ] = uu[ 1 ] * vv[ 2 ] - uu[ 2 ] * vv[ 1 ]; nu[ 1 ] = uu[ 2 ] * vv[ 0 ] - uu[ 0 ] * vv[ 2 ]; nu[ 2 ] = uu[ 0 ] * vv[ 1 ] - uu[ 1 ] * vv[ 0 ];
            const TF J = sdot::sqrt( nu[ 0 ] * nu[ 0 ] + nu[ 1 ] * nu[ 1 ] + nu[ 2 ] * nu[ 2 ] );
            for ( int c = 0; c < 3; ++c ) nu[ c ] /= J;
            clip( bc, [&]( int np, const TF *a, const TF *b, const int *lab ) {
                for ( int k = 0; k < np; ++k ) {
                    const int i = lab[ k ];
                    if ( i < 0 ) continue;
                    const int k1 = k + 1 < np ? k + 1 : 0;
                    TF x0[ 3 ], x1[ 3 ], len = 0, gn = 0, gs = 0;
                    point_at( a[ k ], b[ k ], x0 );
                    point_at( a[ k1 ], b[ k1 ], x1 );
                    for ( int c = 0; c < 3; ++c ) { len += ( x1[ c ] - x0[ c ] ) * ( x1[ c ] - x0[ c ] ); gn += bc.G[ i ][ c ] * nu[ c ]; }
                    for ( int c = 0; c < 3; ++c ) { const TF t = bc.G[ i ][ c ] - gn * nu[ c ]; gs += t * t; }
                    const TF r0 = R( 0 ) + a[ k ] * ( R( 1 ) - R( 0 ) ) + b[ k ] * ( R( 2 ) - R( 0 ) );
                    const TF r1 = R( 0 ) + a[ k1 ] * ( R( 1 ) - R( 0 ) ) + b[ k1 ] * ( R( 2 ) - R( 0 ) );
                    add( i, x0, x1, r0, r1, sdot::sqrt( len ), sdot::sqrt( gs ) );
                }
            } );
        }
        if constexpr ( requires { grad_dist.m_rho; } )
            if constexpr ( DECAYED_TYPE_OF( grad_dist.m_rho )::is_valid )
                for_each_point( P, [&]( TF w, const TF *, const TF *phi ) {
                    for ( int i = 0; i < nb_verts(); ++i )
                        atomic_add( grad_dist.m_rho( node( i ) ).ref(), g * w * phi[ i ] );
                } );
    }

    /// what the element holds of a facet ( plane `n . x = off`, `n` of norm `nn`, the cell's other cuts by `clip_other` ): the
    /// mass that a unit move of the facet along its normal sweeps, `sum rho / | u . n |` at the crossings of a segment, `int rho /
    /// | n x nu |` along the section of a triangle ( `u` its direction, `nu` its normal, `n` unit ). The section is decided on the
    /// nodes, by the rule of `Mesh::facet_part` ( `s > tol` outside, else inside ), so that a crossing at a node counts once
    template<class Clip>
    HD TF facet_part( const TF *n, TF off, TF sg, Clip &&clip_other ) const {
        TF nn = 0;
        for ( int c = 0; c < D; ++c ) nn += n[ c ] * n[ c ];
        nn = sdot::sqrt( nn );
        const int K = nb_verts();
        TF s[ 3 ], scale = off < 0 ? -off : off;
        for ( int i = 0; i < K; ++i ) {
            TF dot = 0;
            for ( int c = 0; c < D; ++c ) dot += n[ c ] * X( i, c );
            s[ i ] = sg * ( dot - off );
            scale = ( dot < 0 ? -dot : dot ) > scale ? ( dot < 0 ? -dot : dot ) : scale;
        }
        const TF tol = TF( 1e-11 ) * scale;
        int outs[ 3 ], ins[ 3 ], no = 0, ni = 0;
        for ( int i = 0; i < K; ++i ) { if ( s[ i ] > tol ) outs[ no++ ] = i; else ins[ ni++ ] = i; }
        if ( no == 0 || ni == 0 )
            return 0;
        auto crossing = [&]( int o, int in, TF *x, TF &r ) {
            const TF t = s[ in ] > -tol ? TF( 1 ) : s[ o ] / ( s[ o ] - s[ in ] );
            for ( int c = 0; c < D; ++c ) x[ c ] = X( o, c ) + t * ( X( in, c ) - X( o, c ) );
            r = R( o ) + t * ( R( in ) - R( o ) );
        };
        if ( K == 2 ) {
            TF x[ D ], r, un = 0, l = length();
            crossing( outs[ 0 ], ins[ 0 ], x, r );
            if ( ! clip_other( x, x ).inside )
                return 0;
            for ( int c = 0; c < D; ++c ) un += ( X( 1, c ) - X( 0, c ) ) / l * n[ c ] / nn;
            un = un < 0 ? -un : un;
            return un > 0 ? r / un : TF( 0 );
        }
        // a triangle: the section is a segment, clipped by the other cuts
        TF x0[ D ], x1[ D ], r0, r1;
        if ( no == 1 ) { crossing( outs[ 0 ], ins[ 0 ], x0, r0 ); crossing( outs[ 0 ], ins[ 1 ], x1, r1 ); }
        else           { crossing( outs[ 0 ], ins[ 0 ], x0, r0 ); crossing( outs[ 1 ], ins[ 0 ], x1, r1 ); }
        const auto cl = clip_other( x0, x1 );
        if ( ! ( cl.t1 > cl.t0 ) )
            return 0;
        TF len = 0, nu[ 3 ], uu[ 3 ], vv[ 3 ];
        for ( int c = 0; c < D; ++c ) len += ( x1[ c ] - x0[ c ] ) * ( x1[ c ] - x0[ c ] );
        len = sdot::sqrt( len ) * ( cl.t1 - cl.t0 );
        for ( int c = 0; c < 3; ++c ) { uu[ c ] = X( 1, c ) - X( 0, c ); vv[ c ] = X( 2, c ) - X( 0, c ); }
        nu[ 0 ] = uu[ 1 ] * vv[ 2 ] - uu[ 2 ] * vv[ 1 ]; nu[ 1 ] = uu[ 2 ] * vv[ 0 ] - uu[ 0 ] * vv[ 2 ]; nu[ 2 ] = uu[ 0 ] * vv[ 1 ] - uu[ 1 ] * vv[ 0 ];
        TF J = 0, nd = 0;
        for ( int c = 0; c < 3; ++c ) { J += nu[ c ] * nu[ c ]; nd += nu[ c ] * n[ c ]; }
        const TF cs = nd / ( sdot::sqrt( J ) * nn ), sn = sdot::sqrt( 1 - cs * cs < 0 ? TF( 0 ) : 1 - cs * cs );
        if ( ! ( sn > 0 ) )
            return 0;
        const TF ra = r0 + cl.t0 * ( r1 - r0 ), rb = r0 + cl.t1 * ( r1 - r0 );
        return len * ( ra + rb ) / 2 / sn;
    }
};

// The density on the envelope: a cell is cut by each element it meets ( as `Mesh`, whose walk and cuts this repeats ), each
// piece integrated by `CodimMeshElementDensity`.
SDOT_TEMPLATE_DECL_FOR_CodimMesh
struct CodimMesh {
    SDOT_ATTRIBUTES_OF_CodimMesh

    static constexpr int ct_dim = DECAYED_TYPE_OF( nb_dims )::value;
    using TF = DECAYED_TYPE_OF( nodes )::TF;

    /// the width of the continuation ( `Convolved<CodimMesh>::at` ): a member APART from the generated attributes, `0` by default,
    /// and then `params( 0 )`
    TF conv_s = 0;

    HD TF width() const { return conv_s > 0 ? conv_s : TF( params( 0 ) ); }
    HD bool by_quadrature() const { return SI( params.shape( 0 ) ) > 1 && TF( params( 1 ) ) != 0; }
    /// the measure on M itself: the last stage ( `conv_s = 0` ), unless `params( 2 )` asks for the smoothed density at `params( 0 )`
    HD bool exact() const { return ! ( conv_s > 0 ) && ! ( SI( params.shape( 0 ) ) > 2 && TF( params( 2 ) ) != 0 ); }

    static constexpr bool cuts_pieces = true;
    static constexpr bool cuts_facets = true;

    static void computed_grads( auto &&flags ) {
        flags.values = true;
        flags.m_rho = true;
        if constexpr ( requires { flags.current_mass; } ) flags.current_mass = true;
        if constexpr ( requires { flags.target_mass; } )  flags.target_mass  = true;
    }

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

    /// as `walk`, over the hierarchy of the elements of M
    HD void walk_m( bool bounded, const TF ( &clo )[ ct_dim ], const TF ( &chi )[ ct_dim ], auto &&body ) const {
        constexpr int D = ct_dim;
        SI stack[ 64 ];
        int top = 0;
        stack[ top++ ] = 0;
        while ( top ) {
            const SI n = stack[ --top ];
            if ( bounded ) {
                bool apart = false;
                for ( int c = 0; c < D && ! apart; ++c )
                    apart = TF( m_bvh_hi( n, c ) ) < clo[ c ] || TF( m_bvh_lo( n, c ) ) > chi[ c ];
                if ( apart )
                    continue;
            }
            const SI left = SI( m_bvh_links( n, 0 ) );
            if ( left >= 0 ) {
                stack[ top++ ] = SI( m_bvh_links( n, 1 ) );
                stack[ top++ ] = left;
                continue;
            }
            for ( SI e = SI( m_bvh_links( n, 2 ) ); e < SI( m_bvh_links( n, 3 ) ); ++e )
                if ( ! body( e ) )
                    return;
        }
    }

    HD void for_each_piece( const auto &cell, auto &&ws, auto &&func ) const {
        if ( cell.nb_vertices() == 0 )
            return;
        TF clo[ ct_dim ], chi[ ct_dim ];
        const bool bounded = box_of( cell, clo, chi );
        if ( exact() ) {                                     // the cell itself, once per element of M near it
            walk_m( bounded, clo, chi, [&]( SI e ) { func( cell, CodimMeasureDensity<CodimMesh>{ this, e } ); return true; } );
            return;
        }
        walk( bounded, clo, chi, [&]( SI e ) { return piece_of( e, cell, bounded, clo, chi, ws, func ); } );
    }

    // ---- the facets ( the Hessian ), cut by the elements: see `Mesh.h`, the same rule for a facet on a side of two elements ----

    HD void for_each_facet_mass( const auto &cell, auto &&plane_of, auto &&func ) const {
        if ( cell.nb_vertices() == 0 )
            return;
        TF clo[ ct_dim ], chi[ ct_dim ];
        if ( ! box_of( cell, clo, chi ) )
            return;
        const int nc = cell.nb_cuts();
        if ( exact() ) {
            walk_m( true, clo, chi, [&]( SI e ) {
                for ( int q = 0; q < nc; ++q )
                    if ( cell.cid[ q ] >= 0 ) {
                        const TF part = m_facet_part( e, cell, plane_of, q );
                        if ( part != 0 )
                            func( q, part );
                    }
                return true;
            } );
            return;
        }
        walk( true, clo, chi, [&]( SI e ) {
            for ( int q = 0; q < nc; ++q )
                if ( cell.cid[ q ] >= 0 ) {
                    const TF part = facet_part( e, cell, plane_of, q );
                    if ( part != 0 )
                        func( q, part );
                }
            return true;
        } );
    }

    /// what the element `e` of M holds of the facet `q` of `cell` ( `CodimMeasureDensity::facet_part` ), the other cuts of the cell
    /// clipping the segment `[ x0, x1 ]` ( `t0 < t1` ), or the point `x0` ( `inside` )
    HD TF m_facet_part( SI e, const auto &cell, auto &&plane_of, int q ) const {
        constexpr int D = ct_dim;
        TF n[ D ], off;
        plane_of( q, n, off );
        TF nn = 0;
        for ( int c = 0; c < D; ++c ) nn += n[ c ] * n[ c ];
        nn = sdot::sqrt( nn );
        if ( ! ( nn > 0 ) )
            return 0;
        TF sg = 1;
        for ( int c = 0; c < D; ++c )
            if ( ( n[ c ] < 0 ? -n[ c ] : n[ c ] ) > TF( 1e-9 ) * nn ) { sg = n[ c ] < 0 ? TF( -1 ) : TF( 1 ); break; }
        struct Clipped { TF t0, t1; bool inside; };
        const int nc = cell.nb_cuts();
        auto clip_other = [&]( const TF *x0, const TF *x1 ) {
            Clipped r{ 0, 1, true };
            for ( int k = 0; k < nc && r.t0 < r.t1; ++k ) {
                if ( k == q ) continue;
                TF dr[ D ], orr;
                plane_of( k, dr, orr );
                TF sa = - orr, sb = - orr, sc = orr < 0 ? -orr : orr;
                for ( int c = 0; c < D; ++c ) { sa += dr[ c ] * x0[ c ]; sb += dr[ c ] * x1[ c ]; sc += ( dr[ c ] * x0[ c ] < 0 ? -dr[ c ] * x0[ c ] : dr[ c ] * x0[ c ] ); }
                const TF tol = TF( 1e-12 ) * sc;
                if ( sa > tol && sb > tol ) { r.t1 = r.t0; r.inside = false; break; }
                if ( sa > tol )      r.t0 = r.t0 > sa / ( sa - sb ) ? r.t0 : sa / ( sa - sb );
                else if ( sb > tol ) r.t1 = r.t1 < sa / ( sa - sb ) ? r.t1 : sa / ( sa - sb );
            }
            return r;
        };
        return CodimMeasureDensity<CodimMesh>{ this, e }.facet_part( n, off, sg, clip_other );
    }

    /// what element `e` holds of the facet `q` of `cell`: its section by the facet's plane, clipped by the other cuts, the density
    /// integrated on it in closed form ( a segment in 2D, a fan of triangles in 3D )
    HD TF facet_part( SI e, const auto &cell, auto &&plane_of, int q ) const {
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
            if ( ( n[ c ] < 0 ? -n[ c ] : n[ c ] ) > TF( 1e-9 ) * nn ) {
                sg = n[ c ] < 0 ? TF( -1 ) : TF( 1 );
                break;
            }

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

        TF P[ 6 ][ D ];
        int np = 0;
        auto point = [&]( int o, int in ) {
            const TF t = s[ in ] > -tol ? TF( 1 ) : s[ o ] / ( s[ o ] - s[ in ] );
            for ( int c = 0; c < D; ++c )
                P[ np ][ c ] = x[ o ][ c ] + t * ( x[ in ][ c ] - x[ o ][ c ] );
            ++np;
        };
        if ( D == 3 && no == 2 && ni == 2 ) {
            point( outs[ 0 ], ins[ 0 ] ); point( outs[ 0 ], ins[ 1 ] ); point( outs[ 1 ], ins[ 1 ] ); point( outs[ 1 ], ins[ 0 ] );
        } else {
            for ( int a = 0; a < no; ++a )
                for ( int b = 0; b < ni; ++b )
                    point( outs[ a ], ins[ b ] );
        }

        CodimMeshElementDensity<CodimMesh> dens{ this, e, width() };
        const int nc = cell.nb_cuts();
        if constexpr ( D == 2 ) {
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
            TF T[ 2 ][ 2 ];
            for ( int c = 0; c < 2; ++c ) {
                T[ 0 ][ c ] = P[ 0 ][ c ] + t0 * ( P[ 1 ][ c ] - P[ 0 ][ c ] );
                T[ 1 ][ c ] = P[ 0 ][ c ] + t1 * ( P[ 1 ][ c ] - P[ 0 ][ c ] );
            }
            const TF L = sdot::sqrt( ( T[ 1 ][ 0 ] - T[ 0 ][ 0 ] ) * ( T[ 1 ][ 0 ] - T[ 0 ][ 0 ] ) + ( T[ 1 ][ 1 ] - T[ 0 ][ 1 ] ) * ( T[ 1 ][ 1 ] - T[ 0 ][ 1 ] ) );
            return dens.integrate_over_facet_simplex( T, L );
        } else {
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
            TF sum = 0;
            for ( int k = 1; k + 1 < m; ++k ) {
                const TF ax = poly[ k ][ 0 ] - poly[ 0 ][ 0 ], ay = poly[ k ][ 1 ] - poly[ 0 ][ 1 ], az = poly[ k ][ 2 ] - poly[ 0 ][ 2 ];
                const TF bx = poly[ k + 1 ][ 0 ] - poly[ 0 ][ 0 ], by = poly[ k + 1 ][ 1 ] - poly[ 0 ][ 1 ], bz = poly[ k + 1 ][ 2 ] - poly[ 0 ][ 2 ];
                const TF cx = ay * bz - az * by, cy = az * bx - ax * bz, cz = ax * by - ay * bx;
                const TF area = sdot::sqrt( cx * cx + cy * cy + cz * cz ) / 2;
                if ( ! ( area > 0 ) )
                    continue;
                TF T[ 3 ][ 3 ];
                for ( int c = 0; c < 3; ++c ) { T[ 0 ][ c ] = poly[ 0 ][ c ]; T[ 1 ][ c ] = poly[ k ][ c ]; T[ 2 ][ c ] = poly[ k + 1 ][ c ]; }
                sum += dens.integrate_over_facet_simplex( T, area );
            }
            return sum;
        }
    }

    /// the piece `cell INTERSECT element e`, to `func` ( see `Mesh::piece_of` )
    HD bool piece_of( SI e, const auto &cell, bool bounded, const TF ( &clo )[ ct_dim ], const TF ( &chi )[ ct_dim ], auto &&ws, auto &&func ) const {
        constexpr int D = ct_dim;
        const SI nv = SI( cell.nb_vertices() );
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
                return false;
            started = true;
            if ( ws.nb_vertices() == 0 )
                return true;
        }

        const CodimMeshElementDensity<CodimMesh> dens{ this, e, width() };
        auto give = [&]( const auto &piece ) {
            if ( by_quadrature() ) func( piece, PointwiseDensity{ dens } );
            else                   func( piece, dens );
        };
        if ( started ) ws.with_current( [&]( const auto &piece ) { give( piece ); } );
        else           give( cell );
        return true;
    }
};
} // namespace sdot

#include "sdotplan/convolved/CodimMesh.h"
