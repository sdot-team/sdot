#pragma once

// `Convolved` of an image ( see `sdotplan/Convolved.h` ). Included at the end of `Image.h`.
#include <loom/support/algorithms/CartesianIndices.h>
#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>
#include "../Convolved.h"

namespace sdot {
namespace sdotplan {

/// an image: its blurred values, in a buffer of our own ( a polynomial image: not yet, the primary template )
template<class... T> requires ( ! Image<T...>::is_polynomial )
struct Convolved<Image<T...>> {
    using Dist = Image<T...>;
    using TF = typename Dist::TF;
    static constexpr int D = Dist::ct_dim;
    const Dist &dist;
    std::vector<double> src, buf, tmp;                   ///< the values, flattened ( C order )
    std::vector<TF> out;
    std::optional<Dist> current;
    static constexpr bool possible = true;

    Convolved( const Dist &dist ) : dist( dist ) {
        auto shape = dist.values.shape();
        CartesianIndices<DECAYED_TYPE_OF( shape )> cells{ shape };
        src.resize( PI( cells.size() ) );
        for ( PI flat = 0; flat < PI( cells.size() ); ++flat )
            // `cells[ flat ]` is a `Coords` ( NAMED coordinates ), and a tensor accepts
            // one directly -- no need to unpack it into positional integers any more
            src[ flat ] = double( dist.values( cells[ flat ] ) );
    }

    /// the grid step along axis `a` ( the length of `frame( a )` times the mean spacing of the knots ),
    /// and the smallest of them
    double step( int a ) const {
        return dist.with_defaults( [&]( auto &&img ) {       // `frame` / `knots` may be absent: their defaults
            double l2 = 0;
            for ( int c = 0; c < D; ++c ) { const double f = double( img.frame( a, c ) ); l2 += f * f; }
            const SI nb = SI( img.values.shape( a ) );
            const double extent = double( img.knots( a, nb ) ) - double( img.knots( a, 0 ) );
            return std::sqrt( l2 ) * extent / std::max<SI>( nb, 1 );
        } );
    }
    double min_scale( double ) const {
        double r = step( 0 );
        for ( int a = 1; a < D; ++a ) r = std::min( r, step( a ) );
        return r / 4;
    }

    const Dist &at( double s ) {
        if ( s <= 0 ) return dist;
        // the separable filter: along each axis, a Gaussian of standard deviation `s / step` in pixels,
        // truncated at four standard deviations and renormalized cell by cell ( the border )
        buf = src;
        SI shape[ D ], stride[ D ];
        SI total = 1;
        for ( int a = D - 1; a >= 0; --a ) { shape[ a ] = SI( dist.values.shape( a ) ); stride[ a ] = total; total *= shape[ a ]; }
        for ( int a = 0; a < D; ++a ) {
            const double sp = s / step( a );
            if ( ! ( sp > 1e-3 ) ) continue;
            const SI r = std::min<SI>( SI( std::ceil( 4 * sp ) ), shape[ a ] );
            std::vector<double> ker( 2 * r + 1 );
            for ( SI k = -r; k <= r; ++k ) ker[ k + r ] = std::exp( -0.5 * double( k * k ) / ( sp * sp ) );
            tmp.assign( PI( total ), 0.0 );
            for ( SI flat = 0; flat < total; ++flat ) {
                const SI i = ( flat / stride[ a ] ) % shape[ a ];
                double sum = 0, wsum = 0;
                const SI lo = std::max<SI>( -r, -i ), hi = std::min<SI>( r, shape[ a ] - 1 - i );
                for ( SI k = lo; k <= hi; ++k ) { sum += ker[ k + r ] * buf[ flat + k * stride[ a ] ]; wsum += ker[ k + r ]; }
                tmp[ flat ] = sum / wsum;
            }
            buf.swap( tmp );
        }
        out.assign( buf.begin(), buf.end() );
        using V = DECAYED_TYPE_OF( dist.values );
        auto view = tensor_view<typename V::MemorySpace>( out.data(), dist.values.shape(), typename V::AxisNames{} );
        current.emplace( Dist{ dist.target_mass, dist.nb_dims, dist.shape, view, dist.origin, dist.frame, dist.knots,
                               dist.current_mass, dist.nb_cells_cum, dist.cell_cum_mass, dist.nb_coeffs, dist.coeffs } );
        return *current;
    }
};


/// A POLYNOMIAL image ( `degree = P > 0` ) at width `s`: a COARSER grid, of step about `s / 2`, whose Bernstein coefficients are the
/// blurred density at their Greville points ( the Bernstein-Schoenberg operator: positive, exact on the affine functions ), so
///
///     rho_s = sum_{ K, i } ( g_s * rho )( x_{ K, i } ) B_{ K, i },        x_{ K, i } = knot_K + i / P ( knot_{ K + 1 } - knot_K ),
///
/// then rescaled to the mass of the image. The blur is separable on the grid: along each axis, `g_s * rho` at a point is a linear
/// form of the coefficients of the pixels within `6 s` of it ( `axis_rows`, Gauss-Legendre on each pixel, the kernel renormalized
/// by its part inside the box, as for a constant image ), applied axis by axis. Fewer pixels at large `s`: the sweeps of the first
/// stages are cheap. The coarse knots are knots of the image ( one in `f` ), so the box is the same; the stage has the C++ type of
/// the image ( its `coeffs` and `knots` are views on buffers of our own -- hence the knots that a polynomial image always carries,
/// `distributions/Image.py` ).
template<class... T> requires ( Image<T...>::is_polynomial )
struct Convolved<Image<T...>> {
    using Dist = Image<T...>;
    using TF = typename Dist::TF;
    static constexpr int D = Dist::ct_dim;
    static constexpr int P = Dist::degree;
    static constexpr int Q = P + 1;                      ///< coefficients per pixel and per axis
    static constexpr bool possible = true;

    const Dist &dist;
    std::vector<double> src;                             ///< the coefficients, interleaved: `[ n_0 Q, ..., n_{ D - 1 } Q ]`, C order
    std::vector<double> buf, tmp;
    std::vector<TF> out;
    using KF = typename DECAYED_TYPE_OF( dist.knots )::TF;
    std::vector<KF> knots_out;
    std::optional<Dist> current;
    SI n[ D ];                                           ///< the pixels of the image along each axis
    std::vector<double> knots[ D ];                      ///< its knots

    Convolved( const Dist &dist ) : dist( dist ) {
        SI total = 1;
        for ( int a = 0; a < D; ++a ) {
            n[ a ] = dist.nb_pixels( a );
            total *= n[ a ] * Q;
            knots[ a ].resize( n[ a ] + 1 );
            for ( SI k = 0; k <= n[ a ]; ++k )
                knots[ a ][ k ] = double( dist.knots( a, k ) );
        }
        src.resize( PI( total ) );
        auto shape = dist.coeffs.shape();
        CartesianIndices<DECAYED_TYPE_OF( shape )> cells{ shape };
        for ( PI flat = 0; flat < PI( cells.size() ); ++flat )
            cells[ flat ].values.apply_values( [&]( auto ...i ) {
                const SI idx[] = { SI( i )... };          // the pixel, then the flat Bernstein index ( the first axis the slowest )
                SI al[ D ], rem = idx[ D ];
                for ( int a = D - 1; a >= 0; --a ) { al[ a ] = rem % Q; rem /= Q; }
                SI pos = 0;
                for ( int a = 0; a < D; ++a )
                    pos = pos * ( n[ a ] * Q ) + idx[ a ] * Q + al[ a ];
                src[ PI( pos ) ] = double( dist.coeffs( i... ) );
            } );
    }

    /// the length of a grid unit along axis `a` ( the row `a` of the frame )
    double unit( int a ) const {
        double l2 = 0;
        for ( int c = 0; c < D; ++c ) { const double f = double( dist.frame( a, c ) ); l2 += f * f; }
        return std::sqrt( l2 );
    }
    double step( int a ) const { return unit( a ) * ( knots[ a ][ n[ a ] ] - knots[ a ][ 0 ] ) / std::max<SI>( n[ a ], 1 ); }
    double min_scale( double ) const {
        double r = step( 0 );
        for ( int a = 1; a < D; ++a ) r = std::min( r, step( a ) );
        return r / 4;
    }

    static double bern( int i, double t ) {
        double r = 1;
        for ( int k = 1; k <= i; ++k ) r = r * double( P - k + 1 ) / k;
        return r * std::pow( t, i ) * std::pow( 1 - t, P - i );
    }

    /// Gauss-Legendre on `[ 0, 1 ]`, `m` nodes
    static void gauss_legendre( int m, std::vector<double> &x, std::vector<double> &w ) {
        x.resize( m ); w.resize( m );
        for ( int i = 0; i < m; ++i ) {
            double z = std::cos( M_PI * ( i + 0.75 ) / ( m + 0.5 ) ), dp = 0;
            for ( int it = 0; it < 100; ++it ) {
                double p0 = 1, p1 = z;
                for ( int k = 2; k <= m; ++k ) { const double p2 = ( ( 2 * k - 1 ) * z * p1 - ( k - 1 ) * p0 ) / k; p0 = p1; p1 = p2; }
                dp = m * ( z * p1 - p0 ) / ( z * z - 1 );
                const double dz = p1 / dp;
                z -= dz;
                if ( std::abs( dz ) < 1e-15 ) break;
            }
            x[ i ] = ( 1 - z ) / 2;
            w[ i ] = 1 / ( ( 1 - z * z ) * dp * dp );
        }
    }

    /// the rows of axis `a`: for each coarse point `( K, i )`, the weights of the fine `( k, j )` from `col0`, in `vals`
    struct Rows { std::vector<SI> col0, len; std::vector<double> vals; };
    Rows axis_rows( int a, double sig, const std::vector<double> &kc ) const {
        Rows r;
        const SI m = SI( kc.size() ) - 1;
        const auto &kf = knots[ a ];
        std::vector<double> gx, gw;
        gauss_legendre( P + 16, gx, gw );
        const double lo = kf[ 0 ], hi = kf[ n[ a ] ];
        for ( SI K = 0; K < m; ++K )
            for ( int i = 0; i <= P; ++i ) {
                const double x = kc[ K ] + double( i ) / P * ( kc[ K + 1 ] - kc[ K ] );
                // the pixels within `6 sig`
                const SI k0 = SI( std::upper_bound( kf.begin(), kf.end(), x - 6 * sig ) - kf.begin() ) - 1;
                const SI k1 = SI( std::upper_bound( kf.begin(), kf.end(), x + 6 * sig ) - kf.begin() );
                const SI b = std::max<SI>( k0, 0 ), e = std::min<SI>( k1, n[ a ] );
                r.col0.push_back( b * Q );
                r.len.push_back( ( e - b ) * Q );
                const PI start = r.vals.size();
                r.vals.resize( start + PI( ( e - b ) * Q ), 0.0 );
                for ( SI k = b; k < e; ++k ) {
                    const double w = kf[ k + 1 ] - kf[ k ];
                    for ( PI g = 0; g < gx.size(); ++g ) {
                        const double y = kf[ k ] + gx[ g ] * w, u = ( x - y ) / sig;
                        const double kern = gw[ g ] * w * std::exp( -0.5 * u * u );
                        for ( int j = 0; j <= P; ++j )
                            r.vals[ start + PI( ( k - b ) * Q + j ) ] += kern * bern( j, gx[ g ] );
                    }
                }
                // the kernel renormalized by its part inside the box ( the border: a constant stays a constant )
                const double z = std::erf( ( hi - x ) / ( sig * M_SQRT2 ) ) - std::erf( ( lo - x ) / ( sig * M_SQRT2 ) );
                const double norm = z > 0 ? 2 / ( z * sig * std::sqrt( 2 * M_PI ) ) : 0;
                for ( PI v = start; v < r.vals.size(); ++v )
                    r.vals[ v ] *= norm;
            }
        return r;
    }

    const Dist &at( double s ) {
        if ( s <= 0 ) return dist;
        // the coarse grid: one knot in `f`, `f` such that its step is about `s / 2`
        std::vector<double> kc[ D ];
        SI m[ D ], dims[ D ];
        for ( int a = 0; a < D; ++a ) {
            const double h = step( a );
            const SI f = std::max<SI>( 1, std::min<SI>( n[ a ], SI( s / ( 2 * h ) ) ) );
            for ( SI k = 0; k < n[ a ]; k += f )
                kc[ a ].push_back( knots[ a ][ k ] );
            kc[ a ].push_back( knots[ a ][ n[ a ] ] );
            m[ a ] = SI( kc[ a ].size() ) - 1;
            dims[ a ] = n[ a ] * Q;
        }

        // the blur at the Greville points, axis by axis: `[ ..., n_a Q, ... ]` -> `[ ..., m_a Q, ... ]`
        buf = src;
        for ( int a = 0; a < D; ++a ) {
            const Rows rows = axis_rows( a, s / unit( a ), kc[ a ] );
            SI pre = 1, post = 1;
            for ( int b = 0; b < a; ++b ) pre *= dims[ b ];
            for ( int b = a + 1; b < D; ++b ) post *= dims[ b ];
            const SI nin = dims[ a ], nout = m[ a ] * Q;
            tmp.assign( PI( pre * nout * post ), 0.0 );
            PI v = 0;
            for ( SI r = 0; r < nout; ++r ) {
                for ( SI c = 0; c < rows.len[ r ]; ++c, ++v ) {
                    const double wv = rows.vals[ v ];
                    if ( wv == 0 ) continue;
                    const SI col = rows.col0[ r ] + c;
                    for ( SI i = 0; i < pre; ++i ) {
                        const double *in = &buf[ PI( ( i * nin + col ) * post ) ];
                        double *o = &tmp[ PI( ( i * nout + r ) * post ) ];
                        for ( SI j = 0; j < post; ++j )
                            o[ j ] += wv * in[ j ];
                    }
                }
            }
            buf.swap( tmp );
            dims[ a ] = nout;
        }

        // back to `[ pixels..., Q^D ]`, and the mass of the image: `sum_pixels volume mean( coeffs )` ( the frame cancels out )
        SI total = 1;
        for ( int a = 0; a < D; ++a ) total *= m[ a ] * Q;
        out.assign( PI( total ), TF( 0 ) );
        double mass_c = 0;
        for ( SI pos = 0; pos < total; ++pos ) {
            SI rem = pos, k[ D ], al[ D ];
            for ( int a = D - 1; a >= 0; --a ) { const SI ia = rem % ( m[ a ] * Q ); rem /= m[ a ] * Q; k[ a ] = ia / Q; al[ a ] = ia % Q; }
            SI pix = 0, alf = 0;
            double vol = 1;
            for ( int a = 0; a < D; ++a ) { pix = pix * m[ a ] + k[ a ]; alf = alf * Q + al[ a ]; vol *= kc[ a ][ k[ a ] + 1 ] - kc[ a ][ k[ a ] ]; }
            mass_c += vol * buf[ PI( pos ) ];
            out[ PI( pix * dims_q() + alf ) ] = TF( buf[ PI( pos ) ] );
        }
        double mass_f = 0;
        for ( SI pos = 0; pos < SI( src.size() ); ++pos ) {
            SI rem = pos;
            double vol = 1;
            for ( int a = D - 1; a >= 0; --a ) { const SI ia = rem % ( n[ a ] * Q ); rem /= n[ a ] * Q; vol *= knots[ a ][ ia / Q + 1 ] - knots[ a ][ ia / Q ]; }
            mass_f += vol * src[ PI( pos ) ];
        }
        if ( mass_c > 0 )
            for ( auto &o : out )
                o = TF( double( o ) * mass_f / mass_c );

        // the views: the coefficients `[ m..., Q^D ]`, the knots `[ D, max( m ) + 1 ]` ( padded with the last one )
        SI width = 0;
        for ( int a = 0; a < D; ++a ) width = std::max( width, m[ a ] + 1 );
        knots_out.assign( PI( D * width ), KF( 0 ) );
        for ( int a = 0; a < D; ++a )
            for ( SI k = 0; k < width; ++k )
                knots_out[ PI( a * width + k ) ] = KF( kc[ a ][ std::min( k, m[ a ] ) ] );
        auto cshape = dist.coeffs.shape();
        for ( int a = 0; a < D; ++a )
            cshape.set( a, m[ a ] );
        auto kshape = dist.knots.shape();
        kshape.set( 1, width );
        using C = DECAYED_TYPE_OF( dist.coeffs );
        using K = DECAYED_TYPE_OF( dist.knots );
        auto cview = tensor_view<typename C::MemorySpace>( out.data(), cshape, typename C::AxisNames{} );
        auto kview = tensor_view<typename K::MemorySpace>( knots_out.data(), kshape, typename K::AxisNames{} );
        current.emplace( Dist{ dist.target_mass, dist.nb_dims, dist.shape, dist.values, dist.origin, dist.frame, kview,
                               dist.current_mass, dist.nb_cells_cum, dist.cell_cum_mass, dist.nb_coeffs, cview } );
        return *current;
    }

    static constexpr SI dims_q() { SI r = 1; for ( int a = 0; a < D; ++a ) r *= Q; return r; }
};


} // namespace sdotplan
} // namespace sdot
