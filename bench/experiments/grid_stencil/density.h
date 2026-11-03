// Densities for the grid Newton example: mass of a cell and density integral on each face.
//
// Both kernels work in coordinates relative to the seed ( a, b ). A cell given by fixed planes
// ( one vertex per pair of consecutive directions, zero-length edges allowed ) goes through the
// SIMD path; any polygon ( ring-2 or exact cells ) through the scalar path.
//
// Lebesgue: area ( shoelace ) and lengths, exact.
// Gauss: rho = A exp( -|x - c|^2 / ( 2 s^2 ) ) on the box, A such that the total mass is N^2.
//   SIMD path: fan of triangles from the vertex mean, 7-point degree-5 rule per triangle,
//   3-point Gauss-Legendre per face. A cell whose variation of log rho is too large
//   ( `spread` > tau ) is flagged and handed to the scalar path, which subdivides adaptively.
#pragma once
#include <cmath>
#include <vector>
#include <utility>

// 7-point degree-5 rule on the triangle ( barycentric )
namespace quad {
// 1 / j!
constexpr double ifact[ 18 ] = { 1., 1., 1. / 2, 1. / 6, 1. / 24, 1. / 120, 1. / 720, 1. / 5040, 1. / 40320, 1. / 362880, 1. / 3628800, 1. / 39916800,
                                 1. / 479001600, 1. / 6227020800., 1. / 87178291200., 1. / 1307674368000., 1. / 20922789888000., 1. / 355687428096000. };
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

struct Lebesgue {
    static constexpr bool exact = true;
    int N;

    // SIMD: lanes x of a block; vertices v[ k ][ x ] relative to the seed, edge k from vertex k-1 to k.
    // mass[ x ], face[ k ][ x ] ( integral of rho on edge k ), bad[ x ] |= needs the scalar path
    template<int D, int B>
    void cells( const double ( &vx )[ D ][ B ], const double ( &vy )[ D ][ B ], const double *, const double *,
                double *mass, double ( &face )[ D ][ B ], unsigned char * ) const {
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

    template<int D, int B>
    void cells_fine( const double ( &vx )[ D ][ B ], const double ( &vy )[ D ][ B ], const double *sa, const double *sb,
                     double *mass, double ( &face )[ D ][ B ], unsigned char *bad ) const { cells<D,B>( vx, vy, sa, sb, mass, face, bad ); }

    // scalar: polygon ( px, py ) relative to the seed at ( sa, sb ), edge e from vertex e to e+1
    void polygon( int n, const double *px, const double *py, double, double, double &mass, double *face ) const {
        mass = 0;
        for( int e = 0; e < n; ++e ) {
            const int f = e + 1 == n ? 0 : e + 1;
            mass += 0.5 * ( px[ e ] * py[ f ] - py[ e ] * px[ f ] );
            face[ e ] = std::hypot( px[ f ] - px[ e ], py[ f ] - py[ e ] );
        }
    }
};

struct Gauss {
    static constexpr bool exact = false;
    int N;
    double cx, cy, s, A, inv2s2, tau, tol;
    bool taylor = true; // SIMD tiers: one exp per cell + Taylor polynomial at the nodes ( else one exp per node )
    double lfloor = -1e300; // log of a density bound below which a cell counts as empty ( mixtures: far tails )

    // tau: largest `spread` handled by the SIMD rule; tol: absolute tolerance of the scalar path
    // second SIMD tier: every fan triangle split into 16 ( 4 x 4 barycentric grid ), every face into 4
    static constexpr int NF = 16 * 7, NE = 4 * 3;
    double fl0[ NF ], fl1[ NF ], fw[ NF ], et[ NE ], ew[ NE ];

    Gauss( int N, double cx, double cy, double s, double tau = 0.08, double tol = 1e-14 ) : N( N ), cx( cx ), cy( cy ), s( s ), inv2s2( 1 / ( 2 * s * s ) ), tau( tau ), tol( tol ) {
        // sub-triangles of the 4 x 4 barycentric grid, as ( 3 vertices ) in barycentric coordinates ( l0, l1 )
        int n = 0;
        for( int i = 0; i < 4; ++i )
            for( int j = 0; i + j < 4; ++j ) {
                // up: ( i, j ), ( i + 1, j ), ( i, j + 1 ); down: ( i + 1, j ), ( i + 1, j + 1 ), ( i, j + 1 )
                for( int up = 0; up < 2; ++up ) {
                    if( ! up && i + j + 2 > 4 ) continue;
                    const double A0 = up ? i : i + 1, B0 = up ? j : j, A1 = up ? i + 1 : i + 1, B1 = up ? j : j + 1, A2 = up ? i : i, B2 = up ? j + 1 : j + 1;
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
        // integral of exp( -|x-c|^2 / 2 s^2 ) on [ -1/2, N-1/2 ]^2
        auto g = [ & ]( double c ) { return std::sqrt( M_PI / 2 ) * s * ( std::erf( ( N - 0.5 - c ) / ( std::sqrt( 2.0 ) * s ) ) - std::erf( ( -0.5 - c ) / ( std::sqrt( 2.0 ) * s ) ) ); };
        A = double( N ) * N / ( g( cx ) * g( cy ) );
    }

    double rho( double x, double y ) const { const double dx = x - cx, dy = y - cy; return A * std::exp( -( dx * dx + dy * dy ) * inv2s2 ); }

    // fan of triangles from the vertex mean, `nt` nodes per triangle ( barycentric weights of the mean
    // and of vertices k-1 ), `ne` nodes per face; flags the lanes whose spread exceeds `lim`
    template<int D, int B>
    void cells_rule( const double ( &vx )[ D ][ B ], const double ( &vy )[ D ][ B ], const double *sa, const double *sb,
                     double *mass, double ( &face )[ D ][ B ], unsigned char *bad, double lim,
                     int nt, const double *tl0, const double *tl1, const double *tw, int ne, const double *gt, const double *gw, const double *kf = nullptr ) const {
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
            bad[ x ] |= spread > lim * ( kf ? kf[ x ] : 1.0 ) && ! empty[ x ];
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
        }        zero_empty();
    }

    template<int D, int B>
    void cells( const double ( &vx )[ D ][ B ], const double ( &vy )[ D ][ B ], const double *sa, const double *sb,
                double *mass, double ( &face )[ D ][ B ], unsigned char *bad, const double *kf = nullptr ) const {
        cells_rule<D,B>( vx, vy, sa, sb, mass, face, bad, tau, 7, quad::tri_l0, quad::tri_l1, quad::tri_w, 3, quad::gl_t, quad::gl_w, kf );
    }
    // second tier ( 4x finer: spread up to 4 tau )
    template<int D, int B>
    void cells_fine( const double ( &vx )[ D ][ B ], const double ( &vy )[ D ][ B ], const double *sa, const double *sb,
                     double *mass, double ( &face )[ D ][ B ], unsigned char *bad, const double *kf = nullptr ) const {
        cells_rule<D,B>( vx, vy, sa, sb, mass, face, bad, 4 * tau, NF, fl0, fl1, fw, NE, et, ew, kf );
    }

    // ---- scalar path, adaptive ------------------------------------------------------------------
    double tri7( double x0, double y0, double x1, double y1, double x2, double y2 ) const {
        const double ar = 0.5 * std::abs( ( x1 - x0 ) * ( y2 - y0 ) - ( y1 - y0 ) * ( x2 - x0 ) );
        double r = 0;
        for( int q = 0; q < 7; ++q ) {
            const double l0 = quad::tri_l0[ q ], l1 = quad::tri_l1[ q ], l2 = 1 - l0 - l1;
            r += quad::tri_w[ q ] * rho( l0 * x0 + l1 * x1 + l2 * x2, l0 * y0 + l1 * y1 + l2 * y2 );
        }
        return ar * r;
    }
    double tri( double x0, double y0, double x1, double y1, double x2, double y2, double whole, int depth ) const {
        const double ax = ( x0 + x1 ) / 2, ay = ( y0 + y1 ) / 2, bx = ( x1 + x2 ) / 2, by = ( y1 + y2 ) / 2, cx_ = ( x2 + x0 ) / 2, cy_ = ( y2 + y0 ) / 2;
        const double c0 = tri7( x0, y0, ax, ay, cx_, cy_ ), c1 = tri7( ax, ay, x1, y1, bx, by ), c2 = tri7( cx_, cy_, bx, by, x2, y2 ), c3 = tri7( ax, ay, bx, by, cx_, cy_ );
        const double sum = c0 + c1 + c2 + c3;
        if( depth > 30 || std::abs( sum - whole ) <= 63 * tol ) return sum + ( sum - whole ) / 63;
        return tri( x0, y0, ax, ay, cx_, cy_, c0, depth + 1 ) + tri( ax, ay, x1, y1, bx, by, c1, depth + 1 )
             + tri( cx_, cy_, bx, by, x2, y2, c2, depth + 1 ) + tri( ax, ay, bx, by, cx_, cy_, c3, depth + 1 );
    }
    double seg3( double x0, double y0, double x1, double y1 ) const {
        const double L = std::hypot( x1 - x0, y1 - y0 );
        double r = 0;
        for( int q = 0; q < 3; ++q ) r += quad::gl_w[ q ] * rho( x0 + quad::gl_t[ q ] * ( x1 - x0 ), y0 + quad::gl_t[ q ] * ( y1 - y0 ) );
        return L * r;
    }
    double seg( double x0, double y0, double x1, double y1, double whole, int depth ) const {
        const double mx = ( x0 + x1 ) / 2, my = ( y0 + y1 ) / 2;
        const double c0 = seg3( x0, y0, mx, my ), c1 = seg3( mx, my, x1, y1 );
        if( depth > 40 || std::abs( c0 + c1 - whole ) <= 63 * tol ) return c0 + c1 + ( c0 + c1 - whole ) / 63;
        return seg( x0, y0, mx, my, c0, depth + 1 ) + seg( mx, my, x1, y1, c1, depth + 1 );
    }

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
        if( depth > 40 || std::abs( c0 + c1 - whole ) <= 63 * tol ) return c0 + c1 + ( c0 + c1 - whole ) / 63;
        return edge( xmin, x0, y0, mx, my, c0, depth + 1 ) + edge( xmin, mx, my, x1, y1, c1, depth + 1 );
    }

    void polygon( int n, const double *px, const double *py, double sa, double sb, double &mass, double *face ) const {
        double xmin = 1e300;
        for( int e = 0; e < n; ++e ) xmin = std::min( xmin, sa + px[ e ] );
        mass = 0;
        for( int e = 0; e < n; ++e ) {
            const int f = e + 1 == n ? 0 : e + 1;
            const double x1 = sa + px[ e ], y1 = sb + py[ e ], x2 = sa + px[ f ], y2 = sb + py[ f ];
            if( y2 != y1 ) mass += edge( xmin, x1, y1, x2, y2, edge3( xmin, x1, y1, x2, y2 ), 0 );
            face[ e ] = seg_exact( x1, y1, x2, y2 );
        }
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

    // 2D adaptive version ( fan of triangles ), kept for cross-checks
    void polygon_2d( int n, const double *px, const double *py, double sa, double sb, double &mass, double *face ) const {
        double mx = 0, my = 0;
        for( int e = 0; e < n; ++e ) { mx += px[ e ] / n; my += py[ e ] / n; }
        mass = 0;
        for( int e = 0; e < n; ++e ) {
            const int f = e + 1 == n ? 0 : e + 1;
            const double x0 = sa + mx, y0 = sb + my, x1 = sa + px[ e ], y1 = sb + py[ e ], x2 = sa + px[ f ], y2 = sb + py[ f ];
            if( ( x1 - x0 ) * ( y2 - y0 ) - ( y1 - y0 ) * ( x2 - x0 ) > 0 )
                mass += tri( x0, y0, x1, y1, x2, y2, tri7( x0, y0, x1, y1, x2, y2 ), 0 );
            face[ e ] = std::hypot( x2 - x1, y2 - y1 ) > 0 ? seg( x1, y1, x2, y2, seg3( x1, y1, x2, y2 ), 0 ) : 0;
        }
    }
};

// w0 G0 + w1 G1 ( each Gauss of total mass N^2 on the box, w0 + w1 = 1 ): integrals are linear in rho,
// so every path is the sum of the two. A component v times below the other at the seed gets a SIMD
// limit k = min( 4, v^( 1/8 ) ) times larger: its rule error ( ~ spread^6 ) and Taylor remainder
// ( ~ spread^( deg + 1 ), deg >= 7 ) grow by at most ~v, relative to a v times smaller value
struct GaussMix {
    static constexpr bool exact = false;
    int N;
    Gauss g0, g1;
    double tau;

    GaussMix( int N, double w0, Gauss a, Gauss b ) : N( N ), g0( a ), g1( b ), tau( a.tau ) {
        g0.A *= w0; g1.A *= 1 - w0;
        g0.lfloor = g1.lfloor = std::log( 1e-20 );
    }
    double rho( double x, double y ) const { return g0.rho( x, y ) + g1.rho( x, y ); }

    template<int D, int B>
    void cells( const double ( &vx )[ D ][ B ], const double ( &vy )[ D ][ B ], const double *sa, const double *sb,
                double *mass, double ( &face )[ D ][ B ], unsigned char *bad ) const {
        alignas( 64 ) double m1[ B ], f1[ D ][ B ];
        alignas( 64 ) double k0[ B ], k1[ B ];
        kfac<B>( sa, sb, k0, k1 );
        g0.cells<D,B>( vx, vy, sa, sb, mass, face, bad, k0 );
        g1.cells<D,B>( vx, vy, sa, sb, m1, f1, bad, k1 );
        add<D,B>( mass, face, m1, f1 );
    }
    template<int D, int B>
    void cells_fine( const double ( &vx )[ D ][ B ], const double ( &vy )[ D ][ B ], const double *sa, const double *sb,
                     double *mass, double ( &face )[ D ][ B ], unsigned char *bad ) const {
        alignas( 64 ) double m1[ B ], f1[ D ][ B ];
        alignas( 64 ) double k0[ B ], k1[ B ];
        kfac<B>( sa, sb, k0, k1 );
        g0.cells_fine<D,B>( vx, vy, sa, sb, mass, face, bad, k0 );
        g1.cells_fine<D,B>( vx, vy, sa, sb, m1, f1, bad, k1 );
        add<D,B>( mass, face, m1, f1 );
    }
    template<int B>
    void kfac( const double *sa, const double *sb, double *k0, double *k1 ) const {
        const double l0 = std::log( g0.A ), l1 = std::log( g1.A );
        #pragma omp simd
        for( int x = 0; x < B; ++x ) {
            const double ax = sa[ x ] - g0.cx, ay = sb[ x ] - g0.cy, bx = sa[ x ] - g1.cx, by = sb[ x ] - g1.cy;
            const double d = ( l1 - ( bx * bx + by * by ) * g1.inv2s2 ) - ( l0 - ( ax * ax + ay * ay ) * g0.inv2s2 ); // log( rho1 / rho0 )
            k0[ x ] = std::fmin( 4.0, std::exp( std::fmax( d, 0.0 ) / 8 ) );
            k1[ x ] = std::fmin( 4.0, std::exp( std::fmax( -d, 0.0 ) / 8 ) );
        }
    }
    template<int D, int B>
    static void add( double *mass, double ( &face )[ D ][ B ], const double *m1, const double ( &f1 )[ D ][ B ] ) {
        #pragma omp simd
        for( int x = 0; x < B; ++x ) mass[ x ] += m1[ x ];
        for( int k = 0; k < D; ++k ) {
            #pragma omp simd
            for( int x = 0; x < B; ++x ) face[ k ][ x ] += f1[ k ][ x ];
        }
    }
    void polygon( int n, const double *px, const double *py, double sa, double sb, double &mass, double *face ) const {
        std::vector<double> f1( n );
        double m1;
        g0.polygon( n, px, py, sa, sb, mass, face );
        g1.polygon( n, px, py, sa, sb, m1, f1.data() );
        mass += m1;
        for( int e = 0; e < n; ++e ) face[ e ] += f1[ e ];
    }
};
