#pragma once

// The closed forms of `CodimMesh.h` on whole simplices, without the distribution: `|det|`, `sum_q f_q G( 0, u, u_q )`, and the
// integrals of the elements of an envelope ( the mass of `sdot.CodimMesh`, the checks ).
#include <loom/support/common_macros.h>
#include <loom/support/containers/Matrix.h>
#include <sdot/integration/GaussDividedDifferences.h>

namespace sdot {
namespace codim {

/// `|det|` of the edges of `P` ( `D + 1` points ), and its sign
template<int D,class TF>
HD TF det_of( const auto &P ) {
    return Matrix<TF,D>::with_func( [&]( auto r, auto c ) { return TF( P[ c + 1 ][ r ] - P[ 0 ][ r ] ) ; } ).determinant();
}

/// `sum_q f_q G( 0, u_0 .. u_n, u_q )`: `int_P rho exp( -u^2 ) / ( |P| n! )`, `n + 1` vertices
template<class TF>
HD TF weighted_g( int n, const TF *u, const TF *f ) {
    TF s = 0;
    for ( int q = 0; q <= n; ++q ) {
        TF x[ gauss_dd::MAX_NODES ];
        for ( int i = 0; i <= n; ++i ) x[ i ] = u[ i ];
        x[ n + 1 ] = u[ q ];
        s += f[ q ] * gauss_dd::G( 0, x, n + 1 );
    }
    return s;
}


/// `out( e ) = int_e rho exp( -d^2 / s^2 )` for the simplices `pts( e, i, c )` with `d( e, i )`, `rho( e, i )` at the corners
template<int D>
void element_integrals( const auto &queue, const auto &pts, const auto &d, const auto &rho, const auto &params, auto &&out ) {
    const SI E = SI( pts.shape( 0 ) );
    const double s = double( params( 0 ) );
    for ( SI e = 0; e < E; ++e ) {
        double P[ D + 1 ][ D ], u[ D + 1 ], f[ D + 1 ];
        for ( int i = 0; i <= D; ++i ) {
            for ( int c = 0; c < D; ++c ) P[ i ][ c ] = double( pts( e, i, c ) );
            u[ i ] = double( d( e, i ) ) / s;
            f[ i ] = double( rho( e, i ) );
        }
        const double det = det_of<D,double>( P );
        out( e ) = ( det < 0 ? -det : det ) * weighted_g( D, u, f );
    }
    ( void )queue;
}

} // namespace codim
} // namespace sdot
