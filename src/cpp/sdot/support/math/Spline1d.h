#pragma once

#include "TridiagonalSolver.h"
#include <vector>
#include <cmath>
#include <span>

namespace sdot {

template<class TF>
struct Spline1d {
    struct Coeffs { TF c0, c1, c2, c3; };

    static void values_to_coeffs( std::span<Coeffs> res, std::span<const TF> v, std::span<const TF> x ) {
        PI n_intervals = res.size();
        PI n_points = v.size();
        if ( n_intervals == 0 )
            return;

        std::vector<TF> a( n_points ), b( n_points ), c( n_points ), d( n_points ), slopes( n_intervals ), h( n_intervals ), ds( n_points );

        for ( PI i = 0; i < n_intervals; ++i ) {
            h[ i ] = x[ i + 1 ] - x[ i ];
            TF abs_h = std::abs( h[ i ] );
            TF safe_h = abs_h < 1e-18 ? ( h[ i ] >= 0 ? 1e-18 : -1e-18 ) : h[ i ];
            slopes[ i ] = ( v[ i + 1 ] - v[ i ] ) / safe_h;
        }

        // Natural cubic spline equations
        b[ 0 ] = 2;
        c[ 0 ] = 1;
        d[ 0 ] = 3 * slopes[ 0 ];

        for ( PI i = 1; i < n_intervals; ++i ) {
            a[ i ] = h[ i ];
            b[ i ] = 2 * ( h[ i - 1 ] + h[ i ] );
            c[ i ] = h[ i - 1 ];
            d[ i ] = 3 * ( h[ i ] * slopes[ i - 1 ] + h[ i - 1 ] * slopes[ i ] );
        }

        a[ n_intervals ] = 1;
        b[ n_intervals ] = 2;
        d[ n_intervals ] = 3 * slopes[ n_intervals - 1 ];

        TridiagonalSolver<TF>::solve( ds, a, b, c, d );

        for ( PI i = 0; i < n_intervals; ++i ) {
            TF hi = h[ i ];
            TF abs_h = std::abs( hi );
            TF safe_h = abs_h < 1e-18 ? ( hi >= 0 ? 1e-18 : -1e-18 ) : hi;
            TF inv_h = 1 / safe_h;

            TF si = slopes[ i ];
            TF d_i = ds[ i ];
            TF d_ip1 = ds[ i + 1 ];

            TF cl0 = v[ i ];
            TF cl1 = d_i;
            TF cl2 = ( 3 * si - 2 * d_i - d_ip1 ) * inv_h;
            TF cl3 = ( d_i + d_ip1 - 2 * si ) * ( inv_h * inv_h );

            TF xi = x[ i ];
            res[ i ].c0 = cl0 - xi * ( cl1 - xi * ( cl2 - xi * cl3 ) );
            res[ i ].c1 = cl1 - xi * ( 2 * cl2 - 3 * xi * cl3 );
            res[ i ].c2 = cl2 - 3 * xi * cl3;
            res[ i ].c3 = cl3;
        }
    }

    static void values_to_coeffs_backward( std::span<TF> g_v, std::span<TF> g_x, std::span<const Coeffs> g_C, std::span<const TF> v, std::span<const TF> x ) {
        PI n_intervals = g_C.size();
        PI n_points = v.size();
        if ( n_intervals == 0 ) return;

        std::vector<TF> h( n_intervals ), slopes( n_intervals ), ds( n_points );
        for ( PI i = 0; i < n_intervals; ++i ) {
            h[ i ] = x[ i + 1 ] - x[ i ];
            TF abs_h = std::abs( h[ i ] );
            TF safe_h = abs_h < 1e-18 ? ( h[ i ] >= 0 ? 1e-18 : -1e-18 ) : h[ i ];
            slopes[ i ] = ( v[ i + 1 ] - v[ i ] ) / safe_h;
        }

        std::vector<TF> a( n_points ), b( n_points ), c( n_points ), d( n_points );
        b[ 0 ] = 2; c[ 0 ] = 1; d[ 0 ] = 3 * slopes[ 0 ];
        for ( PI i = 1; i < n_intervals; ++i ) {
            a[ i ] = h[ i ];
            b[ i ] = 2 * ( h[ i - 1 ] + h[ i ] );
            c[ i ] = h[ i - 1 ];
            d[ i ] = 3 * ( h[ i ] * slopes[ i - 1 ] + h[ i - 1 ] * slopes[ i ] );
        }
        a[ n_intervals ] = 1; b[ n_intervals ] = 2; d[ n_intervals ] = 3 * slopes[ n_intervals - 1 ];
        TridiagonalSolver<TF>::solve( ds, a, b, c, d );

        std::vector<TF> g_cl0( n_intervals ), g_cl1( n_intervals ), g_cl2( n_intervals ), g_cl3( n_intervals );
        for ( PI i = 0; i < n_intervals; ++i ) {
            TF xi = x[ i ];
            TF hi = h[ i ];
            TF abs_h = std::abs( hi );
            TF safe_h = abs_h < 1e-18 ? ( hi >= 0 ? 1e-18 : -1e-18 ) : hi;
            TF inv_h = 1 / safe_h;

            TF d_i = ds[ i ];
            TF d_ip1 = ds[ i + 1 ];
            TF si = slopes[ i ];
            TF cl1 = d_i;
            TF cl2 = ( 3 * si - 2 * d_i - d_ip1 ) * inv_h;
            TF cl3 = ( d_i + d_ip1 - 2 * si ) * ( inv_h * inv_h );

            g_cl0[ i ] = g_C[ i ].c0;
            g_cl1[ i ] = g_C[ i ].c1 - xi * g_C[ i ].c0;
            g_cl2[ i ] = g_C[ i ].c2 - 2 * xi * g_C[ i ].c1 + xi * xi * g_C[ i ].c0;
            g_cl3[ i ] = g_C[ i ].c3 - 3 * xi * g_C[ i ].c2 + 3 * xi * xi * g_C[ i ].c1 - xi * xi * xi * g_C[ i ].c0;

            g_x[ i ] += g_C[ i ].c2 * ( -3 * cl3 )
                      + g_C[ i ].c1 * ( -2 * cl2 + 6 * xi * cl3 )
                      + g_C[ i ].c0 * ( -cl1 + 2 * xi * cl2 - 3 * xi * xi * cl3 );
        }

        std::vector<TF> g_ds( n_points, 0 );
        std::vector<TF> g_slopes( n_intervals, 0 );
        std::vector<TF> g_h( n_intervals, 0 );

        for ( PI i = 0; i < n_intervals; ++i ) {
            TF hi = h[ i ];
            TF abs_h = std::abs( hi );
            TF safe_h = abs_h < 1e-18 ? ( hi >= 0 ? 1e-18 : -1e-18 ) : hi;
            TF inv_h = 1 / safe_h;
            TF inv_h2 = inv_h * inv_h;
            TF si = slopes[ i ];
            TF d_i = ds[ i ];
            TF d_ip1 = ds[ i + 1 ];

            g_slopes[ i ] += g_cl2[ i ] * 3 * inv_h;
            g_ds[ i ]     += g_cl2[ i ] * ( -2 ) * inv_h;
            g_ds[ i + 1 ] += g_cl2[ i ] * ( -1 ) * inv_h;
            g_h[ i ]      -= g_cl2[ i ] * ( 3 * si - 2 * d_i - d_ip1 ) * inv_h2;

            g_ds[ i ]     += g_cl3[ i ] * inv_h2;
            g_ds[ i + 1 ] += g_cl3[ i ] * inv_h2;
            g_slopes[ i ] -= g_cl3[ i ] * 2 * inv_h2;
            g_h[ i ]      -= g_cl3[ i ] * 2 * ( d_i + d_ip1 - 2 * si ) * inv_h2 * inv_h;

            g_ds[ i ]     += g_cl1[ i ];
            g_v[ i ]      += g_cl0[ i ];
        }

        std::vector<TF> g_rhs( n_points );
        TridiagonalSolver<TF>::solve_transpose( g_rhs, a, b, c, g_ds );

        g_slopes[ 0 ] += g_rhs[ 0 ] * 3;

        for ( PI i = 1; i < n_intervals; ++i ) {
            TF gr = g_rhs[ i ];
            g_h[ i ]      += gr * ( -ds[ i - 1 ] ) + gr * ( -2 * ds[ i ] ) + gr * ( 3 * slopes[ i - 1 ] );
            g_h[ i - 1 ]  += gr * ( -2 * ds[ i ] ) + gr * ( -ds[ i + 1 ] ) + gr * ( 3 * slopes[ i ] );
            g_slopes[ i - 1 ] += gr * 3 * h[ i ];
            g_slopes[ i ]     += gr * 3 * h[ i - 1 ];
        }

        g_slopes[ n_intervals - 1 ] += g_rhs[ n_intervals ] * 3;

        for ( PI i = 0; i < n_intervals; ++i ) {
            TF hi = h[ i ];
            TF abs_h = std::abs( hi );
            TF safe_h = abs_h < 1e-18 ? ( hi >= 0 ? 1e-18 : -1e-18 ) : hi;
            TF inv_h = 1 / safe_h;
            TF inv_h2 = inv_h * inv_h;
            TF dv = v[ i + 1 ] - v[ i ];

            g_v[ i + 1 ] += g_slopes[ i ] * inv_h;
            g_v[ i ]     -= g_slopes[ i ] * inv_h;
            g_h[ i ]     -= g_slopes[ i ] * dv * inv_h2;

            g_x[ i + 1 ] += g_h[ i ];
            g_x[ i ]     -= g_h[ i ];
        }
    }
};

} // namespace sdot
