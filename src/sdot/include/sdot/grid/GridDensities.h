#pragma once

// THE DENSITIES OF THE GRID SOLVER ( `GridNewton2D.h` ), integrated by themselves on the cells of the stencil: the mass of a
// cell, the integral of the density on each of its faces, and the moments of the outputs. A port of
// `bench/experiments/grid_stencil/density.h`.
//
// Everything is in INDEX UNITS of the level: seed ( a, b ) at ( a, b ), the density `rho_idx( u ) = n^2 h_n^2 rho( x )` at
// `x = o + h_n ( u + 1/2 )` ( `n^2 h_n^2 = N^2 h^2` whatever the level ). The scale does not matter: Newton scales its targets
// to what the domain holds.
//
// The interface ( `simd = true` ):
//   cells<D>( vx, vy, sa, sb, mass, face, bad )       a block of B cells given by their D vertices relative to the seeds
//                                                     ( sa, sb ), edge k from vertex k-1 to k; `bad[ x ]` set: the lane needs
//                                                     the next tier ( `has_simd = false`: every lane )
//   cells_fine<D>( ... )                              the second tier
//   polygon( n, px, py, sa, sb, mass, face )          any polygon, edge e from vertex e to e+1 ( scalar, the last tier )
//   moments( n, px, py, sa, sb, m, mx, my, m2 )       the mass, the first moments and the second moment about the seed
//   exact                                             the faces seen from both sides agree ( no symmetrization needed )
//
//   UnitCells    the uniform density: area ( shoelace ) and lengths, exact
//   ImageCells   constant pixels along the axes: the polygon clipped by the columns then the rows of pixels, exact
//   GaussCells   a sum of isotropic gaussians: fan of triangles from the vertex mean, 7-point degree-5 rule per triangle,
//                3-point Gauss-Legendre per face, one exp per cell and a Taylor polynomial at the nodes; a cell whose
//                variation of log rho is too large goes to a finer rule, then to the scalar path ( adaptive, divergence form
//                for the mass, exact on the faces )

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace sdot {
namespace grid {

namespace quad {
// 1 / j!
constexpr double ifact[ 18 ] = { 1., 1., 1. / 2, 1. / 6, 1. / 24, 1. / 120, 1. / 720, 1. / 5040, 1. / 40320, 1. / 362880, 1. / 3628800, 1. / 39916800,
                                 1. / 479001600, 1. / 6227020800., 1. / 87178291200., 1. / 1307674368000., 1. / 20922789888000., 1. / 355687428096000. };
// 7-point degree-5 rule on the triangle ( barycentric )
constexpr double sq15 = 3.872983346207417;
constexpr double b1 = ( 6 + sq15 ) / 21, a1 = 1 - 2 * b1, w1 = ( 155 + sq15 ) / 1200;
constexpr double b2 = ( 6 - sq15 ) / 21, a2 = 1 - 2 * b2, w2 = ( 155 - sq15 ) / 1200;
constexpr double tri_l0[ 7 ] = { 1.0 / 3, a1, b1, b1, a2, b2, b2 };
constexpr double tri_l1[ 7 ] = { 1.0 / 3, b1, a1, b1, b2, a2, b2 };
constexpr double tri_w [ 7 ] = { 0.225, w1, w1, w1, w2, w2, w2 };
// 3-point Gauss-Legendre on [ 0, 1 ]
constexpr double gl_t[ 3 ] = { 0.5 - 0.3872983346207417, 0.5, 0.5 + 0.3872983346207417 };
constexpr double gl_w[ 3 ] = { 5.0 / 18, 8.0 / 18, 5.0 / 18 };
}

/// the uniform density on the box of the grid ( exact )
struct Lebesgue {
    template<int D, int B>
    static void cells( const double ( &vx )[ D ][ B ], const double ( &vy )[ D ][ B ], double *mass, double ( &face )[ D ][ B ] ) {
        #pragma omp simd
        for( int x = 0; x < B; ++x ) mass[ x ] = 0;
        for( int k = 0; k < D; ++k ) {
            const int km = ( k + D - 1 ) % D;
            #pragma omp simd
            for( int x = 0; x < B; ++x ) {
                mass[ x ] += 0.5 * ( vx[ km ][ x ] * vy[ k ][ x ] - vy[ km ][ x ] * vx[ k ][ x ] );
                const double ex = vx[ k ][ x ] - vx[ km ][ x ], ey = vy[ k ][ x ] - vy[ km ][ x ];
                face[ k ][ x ] = std::sqrt( ex * ex + ey * ey );
            }
        }
    }
    static void polygon( int n, const double *px, const double *py, double &mass, double *face ) {
        mass = 0;
        for( int e = 0; e < n; ++e ) {
            const int f = e + 1 == n ? 0 : e + 1;
            mass += 0.5 * ( px[ e ] * py[ f ] - py[ e ] * px[ f ] );
            face[ e ] = std::hypot( px[ f ] - px[ e ], py[ f ] - py[ e ] );
        }
    }
    /// area, first moment and second moment ( about the origin of the coordinates )
    static void moments( int n, const double *px, const double *py, double &m, double &mx, double &my, double &m2 ) {
        m = mx = my = m2 = 0;
        for( int e = 0; e < n; ++e ) {
            const int f = e + 1 == n ? 0 : e + 1;
            const double x0 = px[ e ], y0 = py[ e ], x1 = px[ f ], y1 = py[ f ], c = x0 * y1 - x1 * y0;
            m  += c / 2;
            mx += c * ( x0 + x1 ) / 6;
            my += c * ( y0 + y1 ) / 6;
            m2 += c * ( x0 * x0 + x0 * x1 + x1 * x1 + y0 * y0 + y0 * y1 + y1 * y1 ) / 12;
        }
    }
};

// ================================================================================================
// uniform
// ================================================================================================
struct UnitCells {
    static constexpr bool simd = true, has_simd = true, exact = true;
    template<int D, int B>
    void cells( const double ( &vx )[ D ][ B ], const double ( &vy )[ D ][ B ], const double *, const double *, double *mass,
                double ( &face )[ D ][ B ], unsigned char * ) const { Lebesgue::cells<D,B>( vx, vy, mass, face ); }
    template<int D, int B>
    void cells_fine( const double ( &vx )[ D ][ B ], const double ( &vy )[ D ][ B ], const double *, const double *, double *mass,
                     double ( &face )[ D ][ B ], unsigned char * ) const { Lebesgue::cells<D,B>( vx, vy, mass, face ); }
    void polygon( int n, const double *px, const double *py, double, double, double &mass, double *face ) const { Lebesgue::polygon( n, px, py, mass, face ); }
    void moments( int n, const double *px, const double *py, double, double, double &m, double &mx, double &my, double &m2 ) const { Lebesgue::moments( n, px, py, m, mx, my, m2 ); }
};

// ================================================================================================
// constant pixels
// ================================================================================================
/// An `Image` of degree 0 whose pixels are along the axes: `kx` ( nx + 1 ), `ky` ( ny + 1 ) the edges of its pixels and `v`
/// ( nx * ny, C order: x the slowest ) their densities, in the index units of the level. Outside of the image: no mass.
struct ImageCells {
    static constexpr bool simd = true, has_simd = false, exact = false;
    static constexpr int C = 160;                        ///< vertices of a clipped polygon ( the cells have at most 128 )
    int nx = 0, ny = 0;
    std::vector<double> kx, ky, v;

    /// `ox, oy, h`: the corner of the grid and the side of a cell at this level, `S h^2` its `n^2 h_n^2`; `kxp`, `kyp` the
    /// PHYSICAL edges of the pixels, `vp` their densities
    ImageCells( int nx, int ny, const double *kxp, const double *kyp, const double *vp, double ox, double oy, double h, double Sh2 ) : nx( nx ), ny( ny ), kx( nx + 1 ), ky( ny + 1 ), v( size_t( nx ) * ny ) {
        for( int i = 0; i <= nx; ++i ) kx[ i ] = ( kxp[ i ] - ox ) / h - 0.5;
        for( int j = 0; j <= ny; ++j ) ky[ j ] = ( kyp[ j ] - oy ) / h - 0.5;
        for( size_t q = 0; q < v.size(); ++q ) v[ q ] = Sh2 * vp[ q ];
    }

    template<int D, int B>
    void cells( const double ( & )[ D ][ B ], const double ( & )[ D ][ B ], const double *, const double *, double *, double ( & )[ D ][ B ], unsigned char *bad ) const {
        for( int x = 0; x < B; ++x ) bad[ x ] = 1;
    }
    template<int D, int B>
    void cells_fine( const double ( & )[ D ][ B ], const double ( & )[ D ][ B ], const double *, const double *, double *, double ( & )[ D ][ B ], unsigned char *bad ) const {
        for( int x = 0; x < B; ++x ) bad[ x ] = 1;
    }

    // the polygon ( x, y ) on the side `s * ( coord - c ) <= 0` of the line `coord = c` ( coord: x if `ax == 0` )
    static int clip( int n, const double *x, const double *y, int ax, double c, double s, double *ox_, double *oy_ ) {
        int m = 0;
        for( int e = 0; e < n; ++e ) {
            const int f = e + 1 == n ? 0 : e + 1;
            const double de = s * ( ( ax ? y[ e ] : x[ e ] ) - c ), df = s * ( ( ax ? y[ f ] : x[ f ] ) - c );
            if( de <= 0 ) { ox_[ m ] = x[ e ]; oy_[ m ] = y[ e ]; ++m; }
            if( ( de < 0 && df > 0 ) || ( de > 0 && df < 0 ) ) {
                const double t = de / ( de - df );
                ox_[ m ] = x[ e ] + t * ( x[ f ] - x[ e ] );
                oy_[ m ] = y[ e ] + t * ( y[ f ] - y[ e ] );
                if( ax ) oy_[ m ] = c; else ox_[ m ] = c;
                ++m;
            }
        }
        return m;
    }
    // the pixel interval of `[ lo, hi ]` along an axis of knots `k` ( n pixels ): [ i0, i1 ), empty outside
    static void range( const std::vector<double> &k, int n, double lo, double hi, int &i0, int &i1 ) {
        i0 = std::max( 0, int( std::upper_bound( k.begin(), k.end(), lo ) - k.begin() ) - 1 );
        i1 = std::min( n, int( std::lower_bound( k.begin(), k.end(), hi ) - k.begin() ) );
    }
    static int locate( const std::vector<double> &k, int n, double u ) {
        return std::clamp( int( std::upper_bound( k.begin(), k.end(), u ) - k.begin() ) - 1, 0, n - 1 );
    }

    /// `f( value, n, px, py )` on each piece of the polygon ( relative to the seed ) in a pixel
    template<class F>
    void pieces( int n, const double *px, const double *py, double sa, double sb, F &&f ) const {
        double xlo = 1e300, xhi = -1e300, ylo = 1e300, yhi = -1e300;
        for( int e = 0; e < n; ++e ) {
            xlo = std::min( xlo, sa + px[ e ] ); xhi = std::max( xhi, sa + px[ e ] );
            ylo = std::min( ylo, sb + py[ e ] ); yhi = std::max( yhi, sb + py[ e ] );
        }
        int i0, i1, j0, j1;
        range( kx, nx, xlo, xhi, i0, i1 );
        range( ky, ny, ylo, yhi, j0, j1 );
        if( i1 - i0 == 1 && j1 - j0 == 1 && kx[ i0 ] <= xlo && xhi <= kx[ i1 ] && ky[ j0 ] <= ylo && yhi <= ky[ j1 ] ) {
            f( v[ size_t( i0 ) * ny + j0 ], n, px, py );
            return;
        }
        double ax[ C ], ay[ C ], bx[ C ], by[ C ], cx[ C ], cy[ C ], dx[ C ], dy[ C ];
        for( int i = i0; i < i1; ++i ) {
            int na = clip( n, px, py, 0, kx[ i ] - sa, -1, ax, ay );
            na = clip( na, ax, ay, 0, kx[ i + 1 ] - sa, 1, bx, by );
            if( na < 3 ) continue;
            double lo = 1e300, hi = -1e300;
            for( int e = 0; e < na; ++e ) { lo = std::min( lo, sb + by[ e ] ); hi = std::max( hi, sb + by[ e ] ); }
            int jj0, jj1;
            range( ky, ny, lo, hi, jj0, jj1 );
            for( int j = jj0; j < jj1; ++j ) {
                int nc = clip( na, bx, by, 1, ky[ j ] - sb, -1, cx, cy );
                nc = clip( nc, cx, cy, 1, ky[ j + 1 ] - sb, 1, dx, dy );
                if( nc >= 3 ) f( v[ size_t( i ) * ny + j ], nc, dx, dy );
            }
        }
    }

    void polygon( int n, const double *px, const double *py, double sa, double sb, double &mass, double *face ) const {
        mass = 0;
        pieces( n, px, py, sa, sb, [&]( double val, int m, const double *qx, const double *qy ) {
            double a = 0;
            for( int e = 0; e < m; ++e ) { const int f = e + 1 == m ? 0 : e + 1; a += qx[ e ] * qy[ f ] - qy[ e ] * qx[ f ]; }
            mass += val * a / 2;
        } );
        // the faces: each edge cut by the lines of the pixels, a piece taking the value of the pixel on the side of the cell
        for( int e = 0; e < n; ++e ) {
            const int f = e + 1 == n ? 0 : e + 1;
            const double x0 = sa + px[ e ], y0 = sb + py[ e ], ex = px[ f ] - px[ e ], ey = py[ f ] - py[ e ], L = std::hypot( ex, ey );
            face[ e ] = 0;
            if( L == 0 ) continue;
            double ts[ 2 * C ];
            int nt = 0;
            ts[ nt++ ] = 0;
            auto cuts = [&]( const std::vector<double> &k, double u0, double du ) {
                if( du == 0 ) return;
                const double ua = std::min( u0, u0 + du ), ub = std::max( u0, u0 + du );
                auto b = std::upper_bound( k.begin(), k.end(), ua ), en = std::lower_bound( k.begin(), k.end(), ub );
                for( ; b < en && nt < 2 * C - 1; ++b ) ts[ nt++ ] = ( *b - u0 ) / du;
            };
            cuts( kx, x0, ex );
            cuts( ky, y0, ey );
            ts[ nt++ ] = 1;
            std::sort( ts, ts + nt );
            const double nx_ = -ey / L * 1e-9, ny_ = ex / L * 1e-9;   // ( inward: the cells are counterclockwise )
            for( int q = 0; q + 1 < nt; ++q ) {
                const double dt = ts[ q + 1 ] - ts[ q ];
                if( dt <= 0 ) continue;
                const double tm = ( ts[ q ] + ts[ q + 1 ] ) / 2, ux = x0 + tm * ex + nx_, uy = y0 + tm * ey + ny_;
                if( ux < kx[ 0 ] || ux > kx[ nx ] || uy < ky[ 0 ] || uy > ky[ ny ] ) continue;
                face[ e ] += v[ size_t( locate( kx, nx, ux ) ) * ny + locate( ky, ny, uy ) ] * dt * L;
            }
        }
    }

    void moments( int n, const double *px, const double *py, double sa, double sb, double &m, double &mx, double &my, double &m2 ) const {
        m = mx = my = m2 = 0;
        pieces( n, px, py, sa, sb, [&]( double val, int k, const double *qx, const double *qy ) {
            double a, ax, ay, a2;
            Lebesgue::moments( k, qx, qy, a, ax, ay, a2 );
            m += val * a; mx += val * ax; my += val * ay; m2 += val * a2;
        } );
    }
};

// ================================================================================================
// gaussians
// ================================================================================================
/// rho = A exp( -|u - c|^2 / ( 2 s^2 ) ), index units
struct Gauss {
    double cx, cy, s, A, inv2s2, tau, tol;
    bool taylor = true;                                  // SIMD tiers: one exp per cell + Taylor polynomial at the nodes ( else one exp per node )
    double lfloor = -1e300;                              // log of a density bound below which a cell counts as empty ( far tails )

    // tau: largest `spread` handled by the SIMD rule; tol: absolute tolerance of the scalar path
    // second SIMD tier: every fan triangle split into 16 ( 4 x 4 barycentric grid ), every face into 4
    static constexpr int NF = 16 * 7, NE = 4 * 3;
    double fl0[ NF ], fl1[ NF ], fw[ NF ], et[ NE ], ew[ NE ];

    Gauss( double cx = 0, double cy = 0, double s = 1, double A = 1, double tau = 0.08, double tol = 1e-14 ) : cx( cx ), cy( cy ), s( s ), A( A ), inv2s2( 1 / ( 2 * s * s ) ), tau( tau ), tol( tol ) {
        // sub-triangles of the 4 x 4 barycentric grid, as ( 3 vertices ) in barycentric coordinates ( l0, l1 )
        int n = 0;
        for( int i = 0; i < 4; ++i )
            for( int j = 0; i + j < 4; ++j ) {
                // up: ( i, j ), ( i + 1, j ), ( i, j + 1 ); down: ( i + 1, j ), ( i + 1, j + 1 ), ( i, j + 1 )
                for( int up = 0; up < 2; ++up ) {
                    if( ! up && i + j + 2 > 4 ) continue;
                    const double A0 = up ? i : i + 1, B0 = j, A1 = i + 1, B1 = up ? j : j + 1, A2 = i, B2 = j + 1;
                    for( int q = 0; q < 7; ++q ) {
                        const double m0 = quad::tri_l0[ q ], m1 = quad::tri_l1[ q ], m2 = 1 - m0 - m1;
                        fl0[ n ] = ( m0 * A0 + m1 * A1 + m2 * A2 ) / 4;
                        fl1[ n ] = ( m0 * B0 + m1 * B1 + m2 * B2 ) / 4;
                        fw[ n ] = quad::tri_w[ q ] / 16;
                        ++n;
                    }
                }
            }
        for( int p = 0; p < 4; ++p )
            for( int q = 0; q < 3; ++q ) { et[ p * 3 + q ] = ( p + quad::gl_t[ q ] ) / 4; ew[ p * 3 + q ] = quad::gl_w[ q ] / 4; }
    }

    double rho( double x, double y ) const { const double dx = x - cx, dy = y - cy; return A * std::exp( -( dx * dx + dy * dy ) * inv2s2 ); }

    // fan of triangles from the vertex mean, `nt` nodes per triangle ( barycentric weights of the mean
    // and of vertices k-1 ), `ne` nodes per face; flags the lanes whose spread exceeds `lim * kf`
    template<int D, int B>
    void cells_rule( const double ( &vx )[ D ][ B ], const double ( &vy )[ D ][ B ], const double *sa, const double *sb,
                     double *mass, double ( &face )[ D ][ B ], unsigned char *bad, double lim,
                     int nt, const double *tl0, const double *tl1, const double *tw, int ne, const double *gt, const double *gw, const double *kf ) const {
        alignas( 64 ) double mx[ B ], my[ B ], R2[ B ];
        #pragma omp simd
        for( int x = 0; x < B; ++x ) { mx[ x ] = 0; my[ x ] = 0; R2[ x ] = 0; mass[ x ] = 0; }
        for( int k = 0; k < D; ++k ) {
            #pragma omp simd
            for( int x = 0; x < B; ++x ) { mx[ x ] += vx[ k ][ x ] / D; my[ x ] += vy[ k ][ x ] / D; }
        }
        for( int k = 0; k < D; ++k ) {
            #pragma omp simd
            for( int x = 0; x < B; ++x ) {
                const double dx = vx[ k ][ x ] - mx[ x ], dy = vy[ k ][ x ] - my[ x ];
                R2[ x ] = std::fmax( R2[ x ], dx * dx + dy * dy );
            }
        }
        // spread of log rho over the cell: R |grad log rho( m )| + R^2 / ( 2 s^2 ); lanes whose
        // bound rho( m ) exp( spread ) is below the floor are empty ( no flag, zeroed at the end )
        alignas( 64 ) unsigned char empty[ B ];
        const double lA = std::log( A );
        #pragma omp simd
        for( int x = 0; x < B; ++x ) {
            const double gx = sa[ x ] + mx[ x ] - cx, gy = sb[ x ] + my[ x ] - cy, R = std::sqrt( R2[ x ] );
            const double spread = R * std::sqrt( gx * gx + gy * gy ) * 2 * inv2s2 + R2[ x ] * inv2s2;
            empty[ x ] = lA - ( gx * gx + gy * gy ) * inv2s2 + spread < lfloor;
            bad[ x ] |= spread > lim * kf[ x ] && ! empty[ x ];
        }
        auto zero_empty = [ & ] {
            for( int x = 0; x < B; ++x ) if( empty[ x ] ) { mass[ x ] = 0; for( int k = 0; k < D; ++k ) face[ k ][ x ] = 0; }
        };
        if( taylor && lim <= 1.5 ) {
            // rho( p_m + d ) = rho( p_m ) exp( delta ), delta = -( 2 p_m.d + |d|^2 ) / 2 s^2, and
            // |delta| <= spread <= lim on the lanes kept: one exp per cell, a Taylor polynomial of
            // exp( delta ) at the nodes ( degree from lim, remainder < 1e-17 )
            int deg = 2;
            for( double rem = lim * lim * lim / 6; rem > 1e-17 && deg < 17; ++deg ) rem *= lim / ( deg + 2 ); // rem = lim^( deg + 1 ) / ( deg + 1 )!
            alignas( 64 ) double pmx[ B ], pmy[ B ], em[ B ], ux[ D ][ B ], uy[ D ][ B ];
            #pragma omp simd
            for( int x = 0; x < B; ++x ) {
                pmx[ x ] = sa[ x ] + mx[ x ] - cx; pmy[ x ] = sb[ x ] + my[ x ] - cy;
                em[ x ] = A * std::exp( -( pmx[ x ] * pmx[ x ] + pmy[ x ] * pmy[ x ] ) * inv2s2 );
            }
            for( int k = 0; k < D; ++k ) {
                #pragma omp simd
                for( int x = 0; x < B; ++x ) { ux[ k ][ x ] = vx[ k ][ x ] - mx[ x ]; uy[ k ][ x ] = vy[ k ][ x ] - my[ x ]; }
            }
            // Horner with a compile-time degree ( a runtime one keeps the lane loops scalar )
            auto run = [ & ]<int DEG>() {
                auto ex = [ & ]( double d ) {
                    double r = quad::ifact[ DEG ];
                    for( int j = DEG - 1; j >= 0; --j ) r = r * d + quad::ifact[ j ];
                    return r;
                };
                for( int k = 0; k < D; ++k ) {
                    const int km = ( k + D - 1 ) % D;
                    // fan triangle ( m, v_km, v_k ): node d = l1 a + l2 b ( a = v_km - m, b = v_k - m ), so
                    // delta = l1 ( al + l1 ga + 2 l2 ep ) + l2 ( be + l2 ze ); the face nodes are l1 = 1 - t, l2 = t
                    alignas( 64 ) double al[ B ], be[ B ], ga[ B ], ep[ B ], ze[ B ], tm[ B ], fm[ B ];
                    #pragma omp simd
                    for( int x = 0; x < B; ++x ) {
                        const double ax = ux[ km ][ x ], ay = uy[ km ][ x ], bx = ux[ k ][ x ], by = uy[ k ][ x ];
                        al[ x ] = -2 * inv2s2 * ( pmx[ x ] * ax + pmy[ x ] * ay );
                        be[ x ] = -2 * inv2s2 * ( pmx[ x ] * bx + pmy[ x ] * by );
                        ga[ x ] = -inv2s2 * ( ax * ax + ay * ay );
                        ep[ x ] = -inv2s2 * ( ax * bx + ay * by );
                        ze[ x ] = -inv2s2 * ( bx * bx + by * by );
                        tm[ x ] = 0; fm[ x ] = 0;
                    }
                    for( int q = 0; q < nt; ++q ) {
                        const double l1 = tl1[ q ], l2 = 1 - tl0[ q ] - l1, w = tw[ q ];
                        #pragma omp simd
                        for( int x = 0; x < B; ++x ) {
                            const double d = l1 * ( al[ x ] + l1 * ga[ x ] + 2 * l2 * ep[ x ] ) + l2 * ( be[ x ] + l2 * ze[ x ] );
                            tm[ x ] += w * ex( d );
                        }
                    }
                    for( int q = 0; q < ne; ++q ) {
                        const double l2 = gt[ q ], l1 = 1 - l2, w = gw[ q ];
                        #pragma omp simd
                        for( int x = 0; x < B; ++x ) {
                            const double d = l1 * ( al[ x ] + l1 * ga[ x ] + 2 * l2 * ep[ x ] ) + l2 * ( be[ x ] + l2 * ze[ x ] );
                            fm[ x ] += w * ex( d );
                        }
                    }
                    #pragma omp simd
                    for( int x = 0; x < B; ++x ) {
                        const double ax = ux[ km ][ x ], ay = uy[ km ][ x ], bx = ux[ k ][ x ], by = uy[ k ][ x ];
                        mass[ x ] += 0.5 * ( ax * by - ay * bx ) * tm[ x ];
                        face[ k ][ x ] = std::sqrt( ( bx - ax ) * ( bx - ax ) + ( by - ay ) * ( by - ay ) ) * fm[ x ];
                    }
                }
            };
            switch( deg ) {
                case 2: case 3: case 4: case 5: case 6: case 7: run.template operator()<7>(); break;
                case 8: case 9: run.template operator()<9>(); break;
                case 10: case 11: run.template operator()<11>(); break;
                case 12: case 13: run.template operator()<13>(); break;
                case 14: case 15: run.template operator()<15>(); break;
                default: run.template operator()<17>(); break;
            }
            #pragma omp simd
            for( int x = 0; x < B; ++x ) mass[ x ] *= em[ x ];
            for( int k = 0; k < D; ++k ) {
                #pragma omp simd
                for( int x = 0; x < B; ++x ) face[ k ][ x ] *= em[ x ];
            }
            zero_empty();
            return;
        }
        for( int k = 0; k < D; ++k ) {
            const int km = ( k + D - 1 ) % D;
            for( int q = 0; q < nt; ++q ) {
                const double l0 = tl0[ q ], l1 = tl1[ q ], l2 = 1 - l0 - l1, w = tw[ q ];
                #pragma omp simd
                for( int x = 0; x < B; ++x ) {
                    const double ar = 0.5 * ( ( vx[ km ][ x ] - mx[ x ] ) * ( vy[ k ][ x ] - my[ x ] ) - ( vy[ km ][ x ] - my[ x ] ) * ( vx[ k ][ x ] - mx[ x ] ) );
                    const double px = sa[ x ] - cx + l0 * mx[ x ] + l1 * vx[ km ][ x ] + l2 * vx[ k ][ x ];
                    const double py = sb[ x ] - cy + l0 * my[ x ] + l1 * vy[ km ][ x ] + l2 * vy[ k ][ x ];
                    mass[ x ] += w * ar * std::exp( -( px * px + py * py ) * inv2s2 );
                }
            }
            #pragma omp simd
            for( int x = 0; x < B; ++x ) face[ k ][ x ] = 0;
            for( int q = 0; q < ne; ++q ) {
                const double t = gt[ q ], w = gw[ q ];
                #pragma omp simd
                for( int x = 0; x < B; ++x ) {
                    const double ex = vx[ k ][ x ] - vx[ km ][ x ], ey = vy[ k ][ x ] - vy[ km ][ x ];
                    const double px = sa[ x ] - cx + vx[ km ][ x ] + t * ex, py = sb[ x ] - cy + vy[ km ][ x ] + t * ey;
                    face[ k ][ x ] += w * std::sqrt( ex * ex + ey * ey ) * std::exp( -( px * px + py * py ) * inv2s2 );
                }
            }
        }
        #pragma omp simd
        for( int x = 0; x < B; ++x ) mass[ x ] *= A;
        for( int k = 0; k < D; ++k ) {
            #pragma omp simd
            for( int x = 0; x < B; ++x ) face[ k ][ x ] *= A;
        }
        zero_empty();
    }

    template<int D, int B>
    void cells( const double ( &vx )[ D ][ B ], const double ( &vy )[ D ][ B ], const double *sa, const double *sb,
                double *mass, double ( &face )[ D ][ B ], unsigned char *bad, const double *kf ) const {
        cells_rule<D,B>( vx, vy, sa, sb, mass, face, bad, tau, 7, quad::tri_l0, quad::tri_l1, quad::tri_w, 3, quad::gl_t, quad::gl_w, kf );
    }
    // second tier ( 4x finer: spread up to 4 tau )
    template<int D, int B>
    void cells_fine( const double ( &vx )[ D ][ B ], const double ( &vy )[ D ][ B ], const double *sa, const double *sb,
                     double *mass, double ( &face )[ D ][ B ], unsigned char *bad, const double *kf ) const {
        cells_rule<D,B>( vx, vy, sa, sb, mass, face, bad, 4 * tau, NF, fl0, fl1, fw, NE, et, ew, kf );
    }

    // ---- scalar path, adaptive ------------------------------------------------------------------
    // ( the tolerance is also relative: the terms of the divergence form can be large, their rounding above `tol` )
    bool done( double sum, double whole ) const { return std::abs( sum - whole ) <= 63 * std::max( tol, 1e-15 * std::abs( sum ) ); }
    // int_{x0}^{x1} exp( -( x - cx )^2 / 2 s^2 ) dx, accurate in the tails ( erfc on the side of the tail )
    double gx_int( double x0, double x1 ) const {
        const double r = 1 / ( std::sqrt( 2.0 ) * s ), u0 = ( x0 - cx ) * r, u1 = ( x1 - cx ) * r, f = std::sqrt( M_PI / 2 ) * s;
        if( u0 >= 0 && u1 >= 0 ) return f * ( std::erfc( u0 ) - std::erfc( u1 ) );
        if( u0 <= 0 && u1 <= 0 ) return f * ( std::erfc( -u1 ) - std::erfc( -u0 ) );
        return f * ( std::erf( u1 ) - std::erf( u0 ) );
    }
    // divergence form: mass = sum_e int_e F dy, F( x, y ) = int_{xmin}^x rho( x', y ) dx'
    double edge3( double xmin, double x0, double y0, double x1, double y1 ) const {
        double r = 0;
        for( int q = 0; q < 3; ++q ) {
            const double t = quad::gl_t[ q ], x = x0 + t * ( x1 - x0 ), y = y0 + t * ( y1 - y0 ), dy = y - cy;
            r += quad::gl_w[ q ] * gx_int( xmin, x ) * std::exp( -dy * dy * inv2s2 );
        }
        return A * r * ( y1 - y0 );
    }
    double edge( double xmin, double x0, double y0, double x1, double y1, double whole, int depth ) const {
        const double mx = ( x0 + x1 ) / 2, my = ( y0 + y1 ) / 2;
        const double c0 = edge3( xmin, x0, y0, mx, my ), c1 = edge3( xmin, mx, my, x1, y1 );
        if( depth > 40 || done( c0 + c1, whole ) ) return c0 + c1 + ( c0 + c1 - whole ) / 63;
        return edge( xmin, x0, y0, mx, my, c0, depth + 1 ) + edge( xmin, mx, my, x1, y1, c1, depth + 1 );
    }
    // int of rho on the segment, exact: on a line, the gaussian is a 1D gaussian in the abscissa t
    // of the foot of the perpendicular from c ( distance h ): A exp( -h^2 / 2 s^2 ) int exp( -t^2 / 2 s^2 )
    double seg_exact( double x0, double y0, double x1, double y1 ) const {
        const double ex = x1 - x0, ey = y1 - y0, L = std::hypot( ex, ey );
        if( L == 0 ) return 0;
        const double ux = ex / L, uy = ey / L, px = x0 - cx, py = y0 - cy;
        const double t0 = px * ux + py * uy, h = px * uy - py * ux;
        return A * std::exp( -h * h * inv2s2 ) * gx_int( cx + t0, cx + t0 + L );
    }
    void polygon( int n, const double *px, const double *py, double sa, double sb, double &mass, double *face ) const {
        double xmin = 1e300;
        for( int e = 0; e < n; ++e ) {
            if( ! std::isfinite( px[ e ] ) || ! std::isfinite( py[ e ] ) ) {    // ( the adaptive rules would not stop )
                mass = std::numeric_limits<double>::quiet_NaN();
                for( int f = 0; f < n; ++f ) face[ f ] = 0;
                return;
            }
            xmin = std::min( xmin, sa + px[ e ] );
        }
        mass = 0;
        for( int e = 0; e < n; ++e ) {
            const int f = e + 1 == n ? 0 : e + 1;
            const double x1 = sa + px[ e ], y1 = sb + py[ e ], x2 = sa + px[ f ], y2 = sb + py[ f ];
            if( y2 != y1 ) mass += edge( xmin, x1, y1, x2, y2, edge3( xmin, x1, y1, x2, y2 ), 0 );
            face[ e ] = seg_exact( x1, y1, x2, y2 );
        }
    }
};

/// A SUM OF ISOTROPIC GAUSSIANS ( at most `K` ): the integrals are linear in rho, every path is the sum over the gaussians. A
/// gaussian v times below the sum at the seed gets a SIMD limit k = min( 4, v^( 1/8 ) ) times larger: its rule error
/// ( ~ spread^6 ) and Taylor remainder ( ~ spread^( deg + 1 ), deg >= 7 ) grow by at most ~v, relative to a v times
/// smaller value.
struct GaussCells {
    static constexpr bool simd = true, has_simd = true, exact = false;
    static constexpr int K = 8;
    int nk = 0;
    std::array<Gauss,K> g;
    double smallest_sigma = 0;                           ///< physical, before the blur

    /// `prm`: per gaussian ( cx, cy, sigma, mass ), physical; `ox, oy, h, Sh2` as `ImageCells`. `blur`: the width of a gaussian
    /// convolved with the density ( the widths become `sqrt( sigma^2 + blur^2 )`, the masses stay ): the continuation of the
    /// solver, from a density wide enough for Newton to start from flat weights
    GaussCells( int nk, const double *prm, double ox, double oy, double h, double Sh2, double blur = 0 ) : nk( nk ) {
        smallest_sigma = prm[ 2 ];
        for( int i = 0; i < nk; ++i ) {
            smallest_sigma = std::min( smallest_sigma, prm[ 4 * i + 2 ] );
            const double *p = prm + 4 * i, s = std::sqrt( p[ 2 ] * p[ 2 ] + blur * blur );
            g[ i ] = Gauss( ( p[ 0 ] - ox ) / h - 0.5, ( p[ 1 ] - oy ) / h - 0.5, s / h, Sh2 * p[ 3 ] / ( 2 * M_PI * s * s ) );
            g[ i ].lfloor = std::log( 1e-30 );
        }
    }
    double rho( double x, double y ) const { double r = 0; for( int i = 0; i < nk; ++i ) r += g[ i ].rho( x, y ); return r; }

    template<int D, int B, bool FINE>
    void run( const double ( &vx )[ D ][ B ], const double ( &vy )[ D ][ B ], const double *sa, const double *sb,
              double *mass, double ( &face )[ D ][ B ], unsigned char *bad ) const {
        // the log of each gaussian at the seed, and their factors
        alignas( 64 ) double lr[ K ][ B ], kf[ B ], ls[ B ];
        for( int i = 0; i < nk; ++i ) {
            const double la = std::log( g[ i ].A );
            #pragma omp simd
            for( int x = 0; x < B; ++x ) {
                const double ax = sa[ x ] - g[ i ].cx, ay = sb[ x ] - g[ i ].cy;
                lr[ i ][ x ] = la - ( ax * ax + ay * ay ) * g[ i ].inv2s2;
            }
        }
        #pragma omp simd
        for( int x = 0; x < B; ++x ) ls[ x ] = lr[ 0 ][ x ];
        for( int i = 1; i < nk; ++i ) {
            #pragma omp simd
            for( int x = 0; x < B; ++x ) ls[ x ] = std::fmax( ls[ x ], lr[ i ][ x ] );
        }
        for( int i = 0; i < nk; ++i ) {
            #pragma omp simd
            for( int x = 0; x < B; ++x ) kf[ x ] = std::fmin( 4.0, std::exp( ( ls[ x ] - lr[ i ][ x ] ) / 8 ) );
            alignas( 64 ) double m1[ B ], f1[ D ][ B ];
            double *mo = i ? m1 : mass;
            auto &fo = i ? f1 : face;
            if constexpr ( FINE ) g[ i ].template cells_fine<D,B>( vx, vy, sa, sb, mo, fo, bad, kf );
            else                  g[ i ].template cells<D,B>( vx, vy, sa, sb, mo, fo, bad, kf );
            if( ! i ) continue;
            #pragma omp simd
            for( int x = 0; x < B; ++x ) mass[ x ] += m1[ x ];
            for( int k = 0; k < D; ++k ) {
                #pragma omp simd
                for( int x = 0; x < B; ++x ) face[ k ][ x ] += f1[ k ][ x ];
            }
        }
    }
    template<int D, int B>
    void cells( const double ( &vx )[ D ][ B ], const double ( &vy )[ D ][ B ], const double *sa, const double *sb,
                double *mass, double ( &face )[ D ][ B ], unsigned char *bad ) const { run<D,B,false>( vx, vy, sa, sb, mass, face, bad ); }
    template<int D, int B>
    void cells_fine( const double ( &vx )[ D ][ B ], const double ( &vy )[ D ][ B ], const double *sa, const double *sb,
                     double *mass, double ( &face )[ D ][ B ], unsigned char *bad ) const { run<D,B,true>( vx, vy, sa, sb, mass, face, bad ); }

    void polygon( int n, const double *px, const double *py, double sa, double sb, double &mass, double *face ) const {
        double f1[ 256 ], m1;
        g[ 0 ].polygon( n, px, py, sa, sb, mass, face );
        for( int i = 1; i < nk; ++i ) {
            g[ i ].polygon( n, px, py, sa, sb, m1, f1 );
            mass += m1;
            for( int e = 0; e < n; ++e ) face[ e ] += f1[ e ];
        }
    }

    // ---- the moments ( outputs, once ): the fan of triangles from the vertex mean, adaptive on ( 1, x, y, |x|^2 ) rho --------
    using V4 = std::array<double,4>;
    V4 tri7( double sa, double sb, double x0, double y0, double x1, double y1, double x2, double y2 ) const {
        const double ar = 0.5 * std::abs( ( x1 - x0 ) * ( y2 - y0 ) - ( y1 - y0 ) * ( x2 - x0 ) );
        V4 r{ 0, 0, 0, 0 };
        for( int q = 0; q < 7; ++q ) {
            const double l0 = quad::tri_l0[ q ], l1 = quad::tri_l1[ q ], l2 = 1 - l0 - l1;
            const double x = l0 * x0 + l1 * x1 + l2 * x2, y = l0 * y0 + l1 * y1 + l2 * y2, w = quad::tri_w[ q ] * ar * rho( sa + x, sb + y );
            r[ 0 ] += w; r[ 1 ] += w * x; r[ 2 ] += w * y; r[ 3 ] += w * ( x * x + y * y );
        }
        return r;
    }
    V4 tri( double sa, double sb, double x0, double y0, double x1, double y1, double x2, double y2, const V4 &whole, int depth ) const {
        const double ax = ( x0 + x1 ) / 2, ay = ( y0 + y1 ) / 2, bx = ( x1 + x2 ) / 2, by = ( y1 + y2 ) / 2, cx_ = ( x2 + x0 ) / 2, cy_ = ( y2 + y0 ) / 2;
        const V4 c0 = tri7( sa, sb, x0, y0, ax, ay, cx_, cy_ ), c1 = tri7( sa, sb, ax, ay, x1, y1, bx, by ), c2 = tri7( sa, sb, cx_, cy_, bx, by, x2, y2 ), c3 = tri7( sa, sb, ax, ay, bx, by, cx_, cy_ );
        V4 sum;
        double err = 0, mag = 0;
        for( int c = 0; c < 4; ++c ) { sum[ c ] = c0[ c ] + c1[ c ] + c2[ c ] + c3[ c ]; err = std::max( err, std::abs( sum[ c ] - whole[ c ] ) ); mag = std::max( mag, std::abs( sum[ c ] ) ); }
        if( depth > 20 || err <= 63 * std::max( g[ 0 ].tol, 1e-15 * mag ) ) {
            for( int c = 0; c < 4; ++c ) sum[ c ] += ( sum[ c ] - whole[ c ] ) / 63;
            return sum;
        }
        const V4 r0 = tri( sa, sb, x0, y0, ax, ay, cx_, cy_, c0, depth + 1 ), r1 = tri( sa, sb, ax, ay, x1, y1, bx, by, c1, depth + 1 ),
                 r2 = tri( sa, sb, cx_, cy_, bx, by, x2, y2, c2, depth + 1 ), r3 = tri( sa, sb, ax, ay, bx, by, cx_, cy_, c3, depth + 1 );
        for( int c = 0; c < 4; ++c ) sum[ c ] = r0[ c ] + r1[ c ] + r2[ c ] + r3[ c ];
        return sum;
    }
    void moments( int n, const double *px, const double *py, double sa, double sb, double &m, double &mx, double &my, double &m2 ) const {
        double cx = 0, cy = 0;
        for( int e = 0; e < n; ++e ) { cx += px[ e ] / n; cy += py[ e ] / n; }
        V4 t{ 0, 0, 0, 0 };
        for( int e = 0; e < n; ++e ) {
            const int f = e + 1 == n ? 0 : e + 1;
            if( ( px[ e ] - cx ) * ( py[ f ] - cy ) - ( py[ e ] - cy ) * ( px[ f ] - cx ) <= 0 ) continue;
            const V4 r = tri( sa, sb, cx, cy, px[ e ], py[ e ], px[ f ], py[ f ], tri7( sa, sb, cx, cy, px[ e ], py[ e ], px[ f ], py[ f ] ), 0 );
            for( int c = 0; c < 4; ++c ) t[ c ] += r[ c ];
        }
        m = t[ 0 ]; mx = t[ 1 ]; my = t[ 2 ]; m2 = t[ 3 ];
    }
};

} // namespace grid
} // namespace sdot
