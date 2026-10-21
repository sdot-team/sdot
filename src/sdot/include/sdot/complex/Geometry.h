#pragma once

// =====================================================================================
// THE GEOMETRY OF A CELL COMPLEX, from its seeds ( `CellComplex.py`, `CellGeometry` ): the numbers of the complex as functions of the
// positions and the weights, forward and adjoint, for a topology fixed on the host.
//
//   * `vertex_*`: a vertex is the corner of `d` cuts of the cell of a seed `s`: per cut, the bisector with the image `G_t( p_j )` of
//     a seed -- `2 ( q - p_s ) . x = |q|^2 - |p_s|^2 - w_j + w_s` -- or a wall `n . x = o`. Solved, then moved by the frame of the
//     row ( the cell's copy, with a periodicity ). The adjoint: `A^T l = g`, `dL/db = l`, `dL/dA = - l x^T`, sent to the seeds;
//   * `face_*`: a face is the union of the simplices of its fan ( `CellComplex.simplices` ), whose corners are rows of vertices or
//     APEXES ( the mean of the rows of a sub-face ): its measure, its centroid and, for a facet, its area vector ( the measure times
//     the unit normal, from the cofactors of the edges of each simplex, oriented by `sign` );
//   * `distance_*`: `sign n . ( c - x )`, `n = a / |a|`, from the area vector `a` and the centroid `c` of a facet;
//   * `seed_vector_*`: `U ( G_t( p_j ) - p_a )`, from a seed to the image of another, in the frame `U` of a cell.
//
// Every row of an output is written by its own work-item; the adjoints ADD into the inputs they share ( `atomic_add` ).
// =====================================================================================

#include <loom/support/atomic_add.h>
#include <loom/support/common_macros.h>

#include <cmath>
#include <type_traits>

namespace sdot {
namespace cell_complex {

template<class T> using TF_of = typename std::decay_t<T>::TF;

/// the weight of seed `i`, zero without weights ( `weights` a `NoneTensor` )
template<class W>
HD double weight_of( const W &weights, long i ) {
    if constexpr ( std::decay_t<W>::is_valid ) return double( weights( i ) );
    else return 0.0;
}

/// `target( i... ) += v`, by several work-items
template<class G,class... I>
HD void add_to( G &&target, double v, I... i ) {
    if constexpr ( std::decay_t<G>::is_valid )
        atomic_add( target( i... ).ref(), TF_of<G>( v ) );
}

/// `A x = b` by Gauss with partial pivoting, `D <= 4`; false when singular
template<int D>
HD bool solve( double A[ D ][ D ], double b[ D ], double x[ D ] ) {
    int perm[ D ];
    double M[ D ][ D + 1 ];
    for ( int r = 0; r < D; ++r ) {
        perm[ r ] = r;
        for ( int c = 0; c < D; ++c ) M[ r ][ c ] = A[ r ][ c ];
        M[ r ][ D ] = b[ r ];
    }
    for ( int c = 0; c < D; ++c ) {
        int p = c;
        for ( int r = c + 1; r < D; ++r )
            if ( std::fabs( M[ r ][ c ] ) > std::fabs( M[ p ][ c ] ) ) p = r;
        if ( M[ p ][ c ] == 0 ) return false;
        if ( p != c ) for ( int k = 0; k <= D; ++k ) { double t = M[ p ][ k ]; M[ p ][ k ] = M[ c ][ k ]; M[ c ][ k ] = t; }
        for ( int r = c + 1; r < D; ++r ) {
            const double f = M[ r ][ c ] / M[ c ][ c ];
            for ( int k = c; k <= D; ++k ) M[ r ][ k ] -= f * M[ c ][ k ];
        }
    }
    for ( int r = D - 1; r >= 0; --r ) {
        double s = M[ r ][ D ];
        for ( int k = r + 1; k < D; ++k ) s -= M[ r ][ k ] * x[ k ];
        x[ r ] = s / M[ r ][ r ];
    }
    (void)perm;
    return true;
}

// ---- vertices ---------------------------------------------------------------------------------------

/// the system of a row: `A x = b`. Equation `r`: the bisector of the images `E_ea( p_a )` and `E_eb( p_b )` of two seeds
/// ( `2 ( qb - qa ) . x = |qb|^2 - |qa|^2 - w_b + w_a` ), or for `eq_b < 0` the wall `-1 - eq_b` -- the elements `E` a table
/// ( `maps` ): the cuts of the row's own cell, or at a degenerate vertex the best conditioned of all its generators
template<int D>
struct VertexSystem {
    double A[ D ][ D ], b[ D ], qa[ D ][ D ], qb[ D ][ D ];
    long   a[ D ], ea[ D ], pb[ D ], eb[ D ];
};

template<int D>
HD void image_of( double q[ D ], const auto &maps, long e, const auto &positions, long j ) {
    for ( int c = 0; c < D; ++c ) {
        double v = double( maps( e, c, D ) );
        for ( int k = 0; k < D; ++k ) v += double( maps( e, c, k ) ) * double( positions( j, k ) );
        q[ c ] = v;
    }
}

template<int D>
HD VertexSystem<D> vertex_system( const auto &row, const auto &positions, const auto &weights, const auto &maps, const auto &walls ) {
    VertexSystem<D> S;
    for ( int r = 0; r < D; ++r ) {
        S.a[ r ] = long( row.eq_a( r ) ); S.ea[ r ] = long( row.eq_ea( r ) );
        S.pb[ r ] = long( row.eq_b( r ) ); S.eb[ r ] = long( row.eq_eb( r ) );
        if ( S.pb[ r ] >= 0 ) {
            image_of<D>( S.qa[ r ], maps, S.ea[ r ], positions, S.a[ r ] );
            image_of<D>( S.qb[ r ], maps, S.eb[ r ], positions, S.pb[ r ] );
            double qa2 = 0, qb2 = 0;
            for ( int c = 0; c < D; ++c ) { qa2 += S.qa[ r ][ c ] * S.qa[ r ][ c ]; qb2 += S.qb[ r ][ c ] * S.qb[ r ][ c ]; }
            for ( int c = 0; c < D; ++c ) S.A[ r ][ c ] = 2 * ( S.qb[ r ][ c ] - S.qa[ r ][ c ] );
            S.b[ r ] = qb2 - qa2 - weight_of( weights, S.pb[ r ] ) + weight_of( weights, S.a[ r ] );
        } else {
            const long w = -1 - S.pb[ r ];
            for ( int c = 0; c < D; ++c ) { S.A[ r ][ c ] = double( walls( w, c ) ); S.qa[ r ][ c ] = S.qb[ r ][ c ] = 0; }
            S.b[ r ] = double( walls( w, D ) );
        }
    }
    return S;
}

template<int D>
HD void vertex_fwd( const auto &row, const auto &positions, const auto &weights, const auto &maps, const auto &walls, auto &&out ) {
    VertexSystem<D> S = vertex_system<D>( row, positions, weights, maps, walls );
    double x[ D ];
    if ( ! solve<D>( S.A, S.b, x ) )
        for ( int c = 0; c < D; ++c ) x[ c ] = NAN;
    for ( int c = 0; c < D; ++c ) {
        double v = double( row.frame( c, D ) );
        for ( int k = 0; k < D; ++k ) v += double( row.frame( c, k ) ) * x[ k ];
        out.x( c ) = TF_of<decltype( out.x )>( v );
    }
}

template<int D>
HD void vertex_bwd( const auto &row, const auto &positions, const auto &weights, const auto &maps, const auto &walls, const auto &grad_out,
                    auto &&grad_positions, auto &&grad_weights ) {
    VertexSystem<D> S = vertex_system<D>( row, positions, weights, maps, walls );
    double x[ D ], g[ D ], l[ D ], At[ D ][ D ];
    if ( ! solve<D>( S.A, S.b, x ) )
        return;
    for ( int k = 0; k < D; ++k ) {                                          // through the frame: `U^T g`
        g[ k ] = 0;
        for ( int c = 0; c < D; ++c ) g[ k ] += double( row.frame( c, k ) ) * double( grad_out.x( c ) );
    }
    for ( int r = 0; r < D; ++r ) for ( int c = 0; c < D; ++c ) At[ r ][ c ] = S.A[ c ][ r ];
    if ( ! solve<D>( At, g, l ) )
        return;
    for ( int r = 0; r < D; ++r ) {
        if ( S.pb[ r ] < 0 ) continue;                                         // ( a wall: a constant )
        // dL/dA_r = - l_r x, dL/db_r = l_r, with A_r = 2 ( qb - qa ) and b_r = |qb|^2 - |qa|^2 - w_b + w_a
        double gqa[ D ], gqb[ D ];
        for ( int c = 0; c < D; ++c ) {
            gqb[ c ] = - 2 * l[ r ] * x[ c ] + 2 * S.qb[ r ][ c ] * l[ r ];
            gqa[ c ] = 2 * l[ r ] * x[ c ] - 2 * S.qa[ r ][ c ] * l[ r ];
        }
        for ( int k = 0; k < D; ++k ) {                                          // q = M_e p + V_e
            double va = 0, vb = 0;
            for ( int c = 0; c < D; ++c ) { va += double( maps( S.ea[ r ], c, k ) ) * gqa[ c ]; vb += double( maps( S.eb[ r ], c, k ) ) * gqb[ c ]; }
            add_to( grad_positions, va, S.a[ r ], k );
            add_to( grad_positions, vb, S.pb[ r ], k );
        }
        add_to( grad_weights, - l[ r ], S.pb[ r ] );
        add_to( grad_weights, l[ r ], S.a[ r ] );
    }
}

// ---- faces --------------------------------------------------------------------------------------------

/// corner `i` of simplex `s`: a row of vertices, or the apex `-1 - a` ( the mean of rows `apex_rows[ apex_off[ a ] : apex_off[ a + 1 ] ]` )
template<int D>
HD void corner( double p[ D ], long ref, const auto &X, const auto &apex_off, const auto &apex_rows ) {
    for ( int c = 0; c < D; ++c ) p[ c ] = 0;
    if ( ref >= 0 ) {
        for ( int c = 0; c < D; ++c ) p[ c ] = double( X( ref, c ) );
        return;
    }
    const long a = -1 - ref, b = long( apex_off( a ) ), e = long( apex_off( a + 1 ) );
    for ( long i = b; i < e; ++i )
        for ( int c = 0; c < D; ++c ) p[ c ] += double( X( long( apex_rows( i ) ), c ) );
    for ( int c = 0; c < D; ++c ) p[ c ] /= double( e - b );
}

template<int D>
HD void corner_bwd( const double g[ D ], long ref, auto &&grad_X, const auto &apex_off, const auto &apex_rows ) {
    if ( ref >= 0 ) {
        for ( int c = 0; c < D; ++c ) add_to( grad_X, g[ c ], ref, c );
        return;
    }
    const long a = -1 - ref, b = long( apex_off( a ) ), e = long( apex_off( a + 1 ) );
    for ( long i = b; i < e; ++i )
        for ( int c = 0; c < D; ++c ) add_to( grad_X, g[ c ] / double( e - b ), long( apex_rows( i ) ), c );
}

/// `( vol, Ginv E )` of the simplex of edges `E[ K ][ D ]`: `vol = sqrt( det( E E^T ) ) / K!`, and what its gradient needs
template<int K,int D>
HD double simplex_volume( const double E[ K ][ D ], double GE[ K ][ D ] ) {
    double Gm[ K ][ K ], I[ K ][ K ];
    for ( int a = 0; a < K; ++a ) for ( int b = 0; b < K; ++b ) {
        double s = 0;
        for ( int c = 0; c < D; ++c ) s += E[ a ][ c ] * E[ b ][ c ];
        Gm[ a ][ b ] = s;
    }
    // det and inverse by Gauss-Jordan ( K <= 4 )
    double M[ K ][ 2 * K ];
    for ( int a = 0; a < K; ++a ) for ( int b = 0; b < 2 * K; ++b ) M[ a ][ b ] = b < K ? Gm[ a ][ b ] : double( b - K == a );
    double det = 1;
    for ( int c = 0; c < K; ++c ) {
        int p = c;
        for ( int r = c + 1; r < K; ++r ) if ( std::fabs( M[ r ][ c ] ) > std::fabs( M[ p ][ c ] ) ) p = r;
        if ( M[ p ][ c ] == 0 ) { det = 0; break; }
        if ( p != c ) { for ( int k = 0; k < 2 * K; ++k ) { double t = M[ p ][ k ]; M[ p ][ k ] = M[ c ][ k ]; M[ c ][ k ] = t; } det = -det; }
        det *= M[ c ][ c ];
        const double inv = 1 / M[ c ][ c ];
        for ( int k = 0; k < 2 * K; ++k ) M[ c ][ k ] *= inv;
        for ( int r = 0; r < K; ++r ) if ( r != c ) {
            const double f = M[ r ][ c ];
            for ( int k = 0; k < 2 * K; ++k ) M[ r ][ k ] -= f * M[ c ][ k ];
        }
    }
    double fact = 1;
    for ( int i = 2; i <= K; ++i ) fact *= i;
    if ( ! ( det > 0 ) ) {
        for ( int a = 0; a < K; ++a ) for ( int c = 0; c < D; ++c ) GE[ a ][ c ] = 0;
        return 0;
    }
    for ( int a = 0; a < K; ++a ) for ( int b = 0; b < K; ++b ) I[ a ][ b ] = M[ a ][ K + b ];
    for ( int a = 0; a < K; ++a ) for ( int c = 0; c < D; ++c ) {
        double s = 0;
        for ( int b = 0; b < K; ++b ) s += I[ a ][ b ] * E[ b ][ c ];
        GE[ a ][ c ] = s;
    }
    return std::sqrt( det ) / fact;
}

/// `det( M )`, `M` a `K x K` row-major, by Gauss with partial pivoting
template<int K>
HD double det_of( double M[ K ][ K ] ) {
    double det = 1;
    for ( int c = 0; c < K; ++c ) {
        int p = c;
        for ( int r = c + 1; r < K; ++r ) if ( std::fabs( M[ r ][ c ] ) > std::fabs( M[ p ][ c ] ) ) p = r;
        if ( M[ p ][ c ] == 0 ) return 0;
        if ( p != c ) { for ( int k = 0; k < K; ++k ) { double t = M[ p ][ k ]; M[ p ][ k ] = M[ c ][ k ]; M[ c ][ k ] = t; } det = -det; }
        det *= M[ c ][ c ];
        for ( int r = c + 1; r < K; ++r ) {
            const double f = M[ r ][ c ] / M[ c ][ c ];
            for ( int k = c; k < K; ++k ) M[ r ][ k ] -= f * M[ c ][ k ];
        }
    }
    return det;
}

/// the AREA VECTOR of a simplex of `D - 1` edges ( a facet ): `a_c = ( -1 )^c det( E without column c ) / ( D - 1 )!` -- the normal
/// times the measure, in any dimension ( `e0 x e1 / 2` in 3D )
template<int D>
HD void area_vector( const double E[ D - 1 ][ D ], double a[ D ] ) {
    double fact = 1;
    for ( int i = 2; i <= D - 1; ++i ) fact *= i;
    for ( int c = 0; c < D; ++c ) {
        double M[ D - 1 ][ D - 1 ];
        for ( int r = 0; r < D - 1; ++r )
            for ( int q = 0, k = 0; q < D; ++q )
                if ( q != c ) M[ r ][ k++ ] = E[ r ][ q ];
        a[ c ] = ( c % 2 ? -1.0 : 1.0 ) * det_of<D - 1>( M ) / fact;
    }
}

/// `dL/dE` of `area_vector` for `dL/da = g`: `d det( M ) / dM = cofactors( M )`
template<int D>
HD void area_vector_bwd( const double E[ D - 1 ][ D ], const double g[ D ], double gE[ D - 1 ][ D ] ) {
    constexpr int K = D - 1;
    double fact = 1;
    for ( int i = 2; i <= K; ++i ) fact *= i;
    for ( int r = 0; r < K; ++r ) for ( int q = 0; q < D; ++q ) gE[ r ][ q ] = 0;
    for ( int c = 0; c < D; ++c ) {
        const double sc = ( c % 2 ? -1.0 : 1.0 ) * g[ c ] / fact;
        // the minor of `E` without column `c`: `M[ r ][ k ]`, `k` the columns `q != c`
        for ( int r = 0; r < K; ++r ) {
            for ( int q = 0, k = 0; q < D; ++q ) {
                if ( q == c ) continue;
                // the cofactor of `M[ r ][ k ]`: `( -1 )^( r + k ) det( M without row r and column k )`
                double C = 1;
                if constexpr ( K > 1 ) {
                    double S[ K > 1 ? K - 1 : 1 ][ K > 1 ? K - 1 : 1 ];
                    for ( int rr = 0, a = 0; rr < K; ++rr ) {
                        if ( rr == r ) continue;
                        for ( int qq = 0, b = 0; qq < D; ++qq ) {
                            if ( qq == c || qq == q ) continue;
                            S[ a ][ b++ ] = E[ rr ][ qq ];
                        }
                        ++a;
                    }
                    C = ( ( r + k ) % 2 ? -1.0 : 1.0 ) * det_of<K - 1>( S );
                }
                gE[ r ][ q ] += sc * C;
                ++k;
            }
        }
    }
}

/// the simplices of one face: `face.s0 .. face.s1` in `corners [ ns, K + 1 ]` and `signs [ ns ]`
template<int K,int D>
HD void face_fwd( const auto &face, const auto &X, const auto &corners, const auto &signs, const auto &apex_off, const auto &apex_rows, auto &&out ) {
    double m = 0, cen[ D ] = {}, area[ D ] = {};
    for ( long s = long( face.s0 ); s < long( face.s1 ); ++s ) {
        double P[ K + 1 ][ D ], E[ K ][ D ], GE[ K ][ D ];
        for ( int i = 0; i <= K; ++i ) corner<D>( P[ i ], long( corners( s, i ) ), X, apex_off, apex_rows );
        for ( int i = 0; i < K; ++i ) for ( int c = 0; c < D; ++c ) E[ i ][ c ] = P[ i + 1 ][ c ] - P[ 0 ][ c ];
        const double v = simplex_volume<K,D>( E, GE );
        m += v;
        for ( int c = 0; c < D; ++c ) {
            double g = 0;
            for ( int i = 0; i <= K; ++i ) g += P[ i ][ c ];
            cen[ c ] += v * g / ( K + 1 );
        }
        if constexpr ( K == D - 1 ) {
            double a[ D ];
            area_vector<D>( E, a );
            for ( int c = 0; c < D; ++c ) area[ c ] += double( signs( s ) ) * a[ c ];
        }
    }
    out.measure = TF_of<decltype( out.measure )>( m );
    for ( int c = 0; c < D; ++c ) {
        out.centroid( c ) = TF_of<decltype( out.centroid )>( m > 0 ? cen[ c ] / m : NAN );
        out.area( c ) = TF_of<decltype( out.area )>( area[ c ] );
    }
}

template<int K,int D>
HD void face_bwd( const auto &face, const auto &X, const auto &corners, const auto &signs, const auto &apex_off, const auto &apex_rows,
                  const auto &out, const auto &grad_out, auto &&grad_X ) {
    const double m = double( out.measure ), gm = double( grad_out.measure );
    double C[ D ], gC[ D ], gA[ D ];
    for ( int c = 0; c < D; ++c ) { C[ c ] = double( out.centroid( c ) ); gC[ c ] = double( grad_out.centroid( c ) ); gA[ c ] = double( grad_out.area( c ) ); }
    for ( long s = long( face.s0 ); s < long( face.s1 ); ++s ) {
        double P[ K + 1 ][ D ], E[ K ][ D ], GE[ K ][ D ], gP[ K + 1 ][ D ] = {}, gE[ K ][ D ] = {};
        for ( int i = 0; i <= K; ++i ) corner<D>( P[ i ], long( corners( s, i ) ), X, apex_off, apex_rows );
        for ( int i = 0; i < K; ++i ) for ( int c = 0; c < D; ++c ) E[ i ][ c ] = P[ i + 1 ][ c ] - P[ 0 ][ c ];
        const double v = simplex_volume<K,D>( E, GE );
        // m = sum v, C = sum v g / m ( g the mean of the corners ): dL/dv = gm + gC . ( g - C ) / m, dL/dg = v gC / m
        double gbar[ D ], dv = gm;
        for ( int c = 0; c < D; ++c ) {
            double g = 0;
            for ( int i = 0; i <= K; ++i ) g += P[ i ][ c ];
            g /= K + 1;
            if ( m > 0 ) dv += gC[ c ] * ( g - C[ c ] ) / m;
            gbar[ c ] = m > 0 ? v * gC[ c ] / m : 0;
        }
        for ( int i = 0; i <= K; ++i ) for ( int c = 0; c < D; ++c ) gP[ i ][ c ] += gbar[ c ] / ( K + 1 );
        for ( int i = 0; i < K; ++i ) for ( int c = 0; c < D; ++c ) gE[ i ][ c ] += dv * v * GE[ i ][ c ];   // d vol / dE = vol G^-1 E
        if constexpr ( K == D - 1 ) {
            double ga[ D ], gEa[ D - 1 ][ D ];
            for ( int c = 0; c < D; ++c ) ga[ c ] = double( signs( s ) ) * gA[ c ];
            area_vector_bwd<D>( E, ga, gEa );
            for ( int i = 0; i < K; ++i ) for ( int c = 0; c < D; ++c ) gE[ i ][ c ] += gEa[ i ][ c ];
        }
        for ( int i = 0; i < K; ++i ) for ( int c = 0; c < D; ++c ) { gP[ i + 1 ][ c ] += gE[ i ][ c ]; gP[ 0 ][ c ] -= gE[ i ][ c ]; }
        for ( int i = 0; i <= K; ++i ) corner_bwd<D>( gP[ i ], long( corners( s, i ) ), grad_X, apex_off, apex_rows );
    }
}

/// the corners of one simplex, as points
template<int K,int D>
HD void simplex_fwd( long s, const auto &X, const auto &corners, const auto &apex_off, const auto &apex_rows, auto &&out ) {
    for ( int i = 0; i <= K; ++i ) {
        double p[ D ];
        corner<D>( p, long( corners( s, i ) ), X, apex_off, apex_rows );
        for ( int c = 0; c < D; ++c ) out.points( i, c ) = TF_of<decltype( out.points )>( p[ c ] );
    }
}

template<int K,int D>
HD void simplex_bwd( long s, const auto &corners, const auto &apex_off, const auto &apex_rows, const auto &grad_out, auto &&grad_X ) {
    for ( int i = 0; i <= K; ++i ) {
        double g[ D ];
        for ( int c = 0; c < D; ++c ) g[ c ] = double( grad_out.points( i, c ) );
        corner_bwd<D>( g, long( corners( s, i ) ), grad_X, apex_off, apex_rows );
    }
}

// ---- distances, seed vectors -------------------------------------------------------------------------

/// `sign n . ( c - x )` for the facet `f` of a point, `n = a / |a|`
template<int D>
HD void distance_fwd( const auto &pt, const auto &area, const auto &centroid, auto &&out ) {
    const long f = long( pt.facet );
    double n2 = 0, d = 0;
    for ( int c = 0; c < D; ++c ) n2 += double( area( f, c ) ) * double( area( f, c ) );
    const double inv = n2 > 0 ? 1 / std::sqrt( n2 ) : 0;
    for ( int c = 0; c < D; ++c ) d += double( area( f, c ) ) * inv * ( double( centroid( f, c ) ) - double( pt.x( c ) ) );
    out.distance = TF_of<decltype( out.distance )>( double( pt.sign ) * d );
}

template<int D>
HD void distance_bwd( const auto &pt, const auto &area, const auto &centroid, const auto &grad_out, auto &&grad_pt, auto &&grad_area, auto &&grad_centroid ) {
    const long f = long( pt.facet );
    const double g = double( pt.sign ) * double( grad_out.distance );
    double a[ D ], r[ D ], n2 = 0, ar = 0;
    for ( int c = 0; c < D; ++c ) { a[ c ] = double( area( f, c ) ); r[ c ] = double( centroid( f, c ) ) - double( pt.x( c ) ); n2 += a[ c ] * a[ c ]; ar += a[ c ] * r[ c ]; }
    if ( ! ( n2 > 0 ) ) return;
    const double nn = std::sqrt( n2 );
    for ( int c = 0; c < D; ++c ) {
        const double n = a[ c ] / nn;
        // d ( a . r / |a| ) / da = r / |a| - ( a . r ) a / |a|^3, / dr = a / |a|
        add_to( grad_area, g * ( r[ c ] / nn - ar * a[ c ] / ( n2 * nn ) ), f, c );
        add_to( grad_centroid, g * n, f, c );
        if constexpr ( std::decay_t<decltype( grad_pt.x )>::is_valid )
            grad_pt.x( c ) = TF_of<decltype( grad_pt.x )>( - g * n );
    }
}

/// `U ( G_t( p_j ) - p_a )`, the frame `U` the linear part of the cell's
template<int D>
HD void seed_vector_fwd( const auto &fc, const auto &positions, const auto &maps, auto &&out ) {
    const long a = long( fc.a ), j = long( fc.j ), t = long( fc.t );
    double v[ D ];
    for ( int c = 0; c < D; ++c ) {
        double q = double( maps( t, c, D ) );
        for ( int k = 0; k < D; ++k ) q += double( maps( t, c, k ) ) * double( positions( j, k ) );
        v[ c ] = q - double( positions( a, c ) );
    }
    for ( int c = 0; c < D; ++c ) {
        double s = 0;
        for ( int k = 0; k < D; ++k ) s += double( fc.frame( c, k ) ) * v[ k ];
        out.vector( c ) = TF_of<decltype( out.vector )>( double( fc.sign ) * s );
    }
}

template<int D>
HD void seed_vector_bwd( const auto &fc, const auto &maps, const auto &grad_out, auto &&grad_positions ) {
    const long a = long( fc.a ), j = long( fc.j ), t = long( fc.t );
    double gv[ D ];
    for ( int k = 0; k < D; ++k ) {
        gv[ k ] = 0;
        for ( int c = 0; c < D; ++c ) gv[ k ] += double( fc.frame( c, k ) ) * double( fc.sign ) * double( grad_out.vector( c ) );
    }
    for ( int k = 0; k < D; ++k ) {
        double gj = 0;
        for ( int c = 0; c < D; ++c ) gj += double( maps( t, c, k ) ) * gv[ c ];
        add_to( grad_positions, gj, j, k );
        add_to( grad_positions, - gv[ k ], a, k );
    }
}

} // namespace cell_complex
} // namespace sdot
