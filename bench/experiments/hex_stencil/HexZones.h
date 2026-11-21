// The power cells of a whole hexagonal lattice, PROVED exact by zones ( the certificate of sdot/grid/GridNewton2D.h `run_zones` ).
//
// The cells of the fast path ( HexRingReg: ring 1 + exact ring-2 prediction, ring 2 on the cut cells ) are cells C^r_i limited to
// ring 2, so C^r_i contains the true cell C_i, and the true cells tile the box. For a zone Z ( here a rectangular tile ):
//     sum_i | C^r_i n Z | >= | Z |, with equality iff C^r_i n Z = C_i n Z for every i.
// A cell whose C^r meets only zones where the sums match is exact. The others ( their C^r meets a bad zone ) are rebuilt by
// `ring_cell` in rounds: limited to hexagonal distance K = 4, 8, 16 ( still containing the true cells ), deposited again, the bad zones
// checked again ( only cells that met them change ); then the psi_max radius ( exact ).
//
// The pieces: a cell whose box stays in one tile adds its area there; the others are cut by differences, from the polygon ( vertices
// from the tightened offsets t of the kernel ): piece( tile ) = A( xr, yt ) - A( xl, yt ) - A( xr, yb ) + A( xl, yb ), A( X, Y ) the
// area of the cell in { x <= X, y <= Y } ( an infinite X or Y skips that clip ).
//
// Tolerance of a zone: the sum over its pieces of c eps ( 1 + reach ) ( 1 + w + h ), reach the farthest point of the cell from its seed,
// w, h its box ( the vertices are relative to the seed: edges move by eps x reach; see the grid notes ).
#pragma once

#include "HexRingReg.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <array>
#include <chrono>
#include <functional>
#include <unordered_map>
#include <vector>

// timing only ( wrong certificate ): bit 1 no deposit, bit 2 no pieces ( a straddling cell goes whole to its first tile )
#ifndef ZSKIP
#define ZSKIP 0
#endif

namespace hexgrid {

template<int B>
struct HexZones {
    using T = double;
    using H = HexRingReg<B,T>;
    static constexpr T eps = std::numeric_limits<T>::epsilon();

    // inputs
    int Wc, R, P;
    const T *pp;                                   // psi padded by 2 rows / columns of -inf ( HexRingReg's layout )
    T tile = 16;                                   // the side of a zone
    T tol_c = 32.0;                                // the factor of the tolerance ( x eps = 7e-15: the offsets relative to the seed are known to eps reach only, h = ( |o|^2 + dpsi ) / 2 |o| with dpsi ~ 2 |o| reach; 8 eps still flags OT solves at N = 240; the grid takes 2e-15 ( dm + reach + 1 ) )

    // outputs
    std::vector<T> area;                           // R x Wc
    std::vector<signed char> lvl;                  // 1: ring 1, 2: ring 2, 3: exact
    std::vector<float> cbox;                       // the box of each cell ( absolute ), 4 per cell
    std::vector<T> zsum, zcmp, ztol;               // per tile ( the sum compensated: Neumaier, else its rounding grows as n eps | Z | )
    std::vector<unsigned char> zbad, zforce;
    long nstraddle = 0, nbad = 0, nexact = 0, nring2 = 0, nround = 0;
    int K0 = 4, max_K = 16;                        // rounds at hexagonal distance K0, 2 K0, .. max_K, then the psi_max radius ( K0 = 2, 3: slower on bands100 )

    // the hessian ( `want_hess` ): per cell the weights L / ( 2 |o| ) of its 6 ring-1 neighbours ( W6, direction k = u_k ), the other
    // faces ( 2 u, v seeds, cells of the rounds ) as ( neighbour, weight ) lists; both seen from the cell ( symmetrize when using them )
    bool want_hess = false;
    bool round_stats = false; std::vector<std::array<double,4>> round_log; // ( K, cells, proved exact by the psi_max rule, seconds )
    bool no_rounds = false;
    std::function<bool( const std::vector<size_t> & )> pre_rounds; // ( see `run` )
    bool aborted = false;                          // `pre_rounds` said no: the areas of the cells meeting bad zones are upper bounds                        // ( diagnostics: keep the ring <= 2 cells )
    bool diag_zones = false; std::vector<std::array<T,3>> zdiag;
    std::vector<T> W6;
    std::unordered_map<size_t,std::vector<std::pair<int,T>>> ext;
    void hess_clear( size_t i ) { for( int k = 0; k < 6; ++k ) W6[ 6 * i + k ] = 0; ext.erase( i ); }
    /// the cell at axial offset ( dq, dr ) from ( r, c ), -1 out of the lattice
    int nbr( int r, int c, int dq, int dr ) const {
        const int rr = r + dr, cc = c + dcol( dq, dr, r & 1 );
        return rr >= 0 && rr < R && cc >= 0 && cc < Wc ? rr * Wc + cc : -1;
    }
    /// a face of cell i toward seed j with the weight w ( W6 when j is a ring-1 neighbour )
    void hess_face( size_t i, int j, T w ) {
        const int r = int( i / Wc ), c = int( i % Wc );
        for( int k = 0; k < 6; ++k ) if( nbr( r, c, uq[ k ], ur[ k ] ) == j ) { W6[ 6 * i + k ] += w; return; }
        ext[ i ].push_back( { j, w } );
    }

    // geometry
    T X0, X1, Y0, Y1;
    int TX, TY;
    T xs( int r, int c ) const { return T( c ) + T( 0.5 ) * T( r & 1 ); }
    T ys( int r ) const { return T( r ) * T( SQ3 / 2 ); }
    T psi( int r, int c ) const { return pp[ size_t( r + 2 ) * P + c + 2 ]; }

    HexZones( int Wc, int R, int P, const T *pp, T tile = 16 ) : Wc( Wc ), R( R ), P( P ), pp( pp ), tile( tile ) {
        X0 = T( -0.25 ); X1 = T( Wc ) - T( 0.25 ); Y0 = T( -SQ3 / 4 ); Y1 = T( R ) * T( SQ3 / 2 ) - T( SQ3 / 4 );
        TX = int( std::ceil( ( X1 - X0 ) / tile ) ); TY = int( std::ceil( ( Y1 - Y0 ) / tile ) );
        area.resize( size_t( Wc ) * R ); lvl.resize( area.size() ); cbox.resize( 4 * area.size() );
        zsum.resize( size_t( TX ) * TY ); zcmp.resize( zsum.size() ); ztol.resize( zsum.size() ); zbad.resize( zsum.size() ); zforce.resize( zsum.size() );
    }

    // ---------------------------------------------------------------- polygon tools ( vertices relative to the seed )
    template<int C> struct PolyN { int n = 0; T x[ C ], y[ C ]; int l[ C ]; }; // l[ e ]: the seed of the edge from vertex e to e + 1 ( -1: box )
    using Poly = PolyN<64>;                        // ( the exact cell )
    using SPoly = PolyN<24>;                       // ( a ring-2 cell, 12 vertices, clipped by two lines )

    // cos / sin of 360 k / D, D = 6 or 12
    static constexpr T c12[ 13 ] = { 1, SQ3 / 2, 0.5, 0, -0.5, -SQ3 / 2, -1, -SQ3 / 2, -0.5, 0, 0.5, SQ3 / 2, 1 };
    static constexpr T s12[ 13 ] = { 0, 0.5, SQ3 / 2, 1, SQ3 / 2, 0.5, 0, -0.5, -SQ3 / 2, -1, -SQ3 / 2, -0.5, 0 };

    /// the polygon of the offsets t ( D unit normals at 360 / D k degrees ): vertex k between the lines k and k + 1
    template<class P> static void poly_of_t( const T *t, int D, P &p ) {
        const int st = 12 / D;
        const T isa = 1 / s12[ st ];
        p.n = D;
        for( int k = 0; k < D; ++k ) {
            const int k1 = k + 1 == D ? 0 : k + 1;
            const T c0 = c12[ st * k ], s0 = s12[ st * k ], c1 = c12[ st * ( k + 1 ) ], s1 = s12[ st * ( k + 1 ) ];
            p.x[ k ] = ( t[ k ] * s1 - t[ k1 ] * s0 ) * isa;
            p.y[ k ] = ( t[ k1 ] * c0 - t[ k ] * c1 ) * isa;
        }
    }
    /// clip by nx x + ny y <= c
    template<class P> static void clip( const P &in, T nx, T ny, T c, P &out, int lab = -1 ) {
        out.n = 0;
        for( int e = 0; e < in.n; ++e ) {
            const int f = e + 1 == in.n ? 0 : e + 1;
            const T se = nx * in.x[ e ] + ny * in.y[ e ] - c, sf = nx * in.x[ f ] + ny * in.y[ f ] - c;
            if( se <= 0 ) { out.x[ out.n ] = in.x[ e ]; out.y[ out.n ] = in.y[ e ]; out.l[ out.n ] = se == 0 && sf > 0 ? lab : in.l[ e ]; ++out.n; }
            if( ( se < 0 && sf > 0 ) || ( se > 0 && sf < 0 ) ) {
                const T u = se / ( se - sf );
                out.x[ out.n ] = in.x[ e ] + u * ( in.x[ f ] - in.x[ e ] ); out.y[ out.n ] = in.y[ e ] + u * ( in.y[ f ] - in.y[ e ] );
                out.l[ out.n ] = se < 0 ? lab : in.l[ e ]; ++out.n;
            }
        }
    }
    // the same by x <= X / y <= Y, two passes less of multiplications
    template<class P> static void clip_x( const P &in, T X, P &out ) {
        out.n = 0;
        for( int e = 0; e < in.n; ++e ) {
            const int f = e + 1 == in.n ? 0 : e + 1;
            const T se = in.x[ e ] - X, sf = in.x[ f ] - X;
            if( se <= 0 ) { out.x[ out.n ] = in.x[ e ]; out.y[ out.n ] = in.y[ e ]; ++out.n; }
            if( ( se < 0 && sf > 0 ) || ( se > 0 && sf < 0 ) ) { const T u = se / ( se - sf ); out.x[ out.n ] = X; out.y[ out.n ] = in.y[ e ] + u * ( in.y[ f ] - in.y[ e ] ); ++out.n; }
        }
    }
    template<class P> static T area_y( const P &in, T Y ) { // area of in n { y <= Y }, without building the polygon
        T s = 0, px = 0, py = 0, fx = 0, fy = 0;
        bool first = true;
        auto put = [&]( T x, T y ) { if( first ) { fx = x; fy = y; first = false; } else s += px * y - py * x; px = x; py = y; };
        for( int e = 0; e < in.n; ++e ) {
            const int f = e + 1 == in.n ? 0 : e + 1;
            const T se = in.y[ e ] - Y, sf = in.y[ f ] - Y;
            if( se <= 0 ) put( in.x[ e ], in.y[ e ] );
            if( ( se < 0 && sf > 0 ) || ( se > 0 && sf < 0 ) ) { const T u = se / ( se - sf ); put( in.x[ e ] + u * ( in.x[ f ] - in.x[ e ] ), Y ); }
        }
        if( ! first ) s += px * fy - py * fx;
        return s / 2;
    }
    template<class P> static T poly_area( const P &p ) {
        T s = 0;
        for( int e = 0; e < p.n; ++e ) { const int f = e + 1 == p.n ? 0 : e + 1; s += p.x[ e ] * p.y[ f ] - p.y[ e ] * p.x[ f ]; }
        return s / 2;
    }

    // the box of a cell as floats, widened OUTWARD by ( |x| + 1 ) 2^-20, more than the float rounding ( 2^-24 |x| ): a box rounded to
    // nearest could miss a bad zone by 1e-7 |x|
    void set_box( size_t i, T x0, T x1, T y0, T y1 ) {
        constexpr T w = T( 1 ) / ( 1 << 20 );
        cbox[ 4 * i + 0 ] = float( x0 - ( std::abs( x0 ) + 1 ) * w ); cbox[ 4 * i + 1 ] = float( x1 + ( std::abs( x1 ) + 1 ) * w );
        cbox[ 4 * i + 2 ] = float( y0 - ( std::abs( y0 ) + 1 ) * w ); cbox[ 4 * i + 3 ] = float( y1 + ( std::abs( y1 ) + 1 ) * w );
    }
    void set_empty( size_t i ) { cbox[ 4 * i + 0 ] = cbox[ 4 * i + 2 ] = 1e30f; cbox[ 4 * i + 1 ] = cbox[ 4 * i + 3 ] = -1e30f; }
    bool is_empty( size_t i ) const { return cbox[ 4 * i ] > cbox[ 4 * i + 1 ]; }

    void add( size_t z, T a, T tol ) {
        const T s = zsum[ z ], t = s + a;
        zcmp[ z ] += std::abs( s ) >= std::abs( a ) ? ( s - t ) + a : ( a - t ) + s;
        zsum[ z ] = t; ztol[ z ] += tol;
    }

    // ---------------------------------------------------------------- deposit of one cell in the tiles
    void deposit( int r, int c, T ar, const T *zb, const T *t, int D ) {
        if constexpr ( ZSKIP & 1 ) return;
        const size_t i = size_t( r ) * Wc + c;
        const T sx = xs( r, c ), sy = ys( r );
        const T bx0 = sx + zb[ 0 ], bx1 = sx + zb[ 1 ], by0 = sy + zb[ 2 ], by1 = sy + zb[ 3 ];
        set_box( i, bx0, bx1, by0, by1 );
        const T reach = std::max( std::max( -zb[ 0 ], zb[ 1 ] ), std::max( -zb[ 2 ], zb[ 3 ] ) );
        const T tol = tol_c * eps * ( 1 + reach ) * ( 1 + ( zb[ 1 ] - zb[ 0 ] ) + ( zb[ 3 ] - zb[ 2 ] ) );
        const T it = 1 / tile;
        auto tix = [&]( T x ) { return std::clamp( int( ( x - X0 ) * it ), 0, TX - 1 ); }; // ( x >= X0 - eps: the int of a tiny negative is 0 )
        auto tiy = [&]( T y ) { return std::clamp( int( ( y - Y0 ) * it ), 0, TY - 1 ); };
        const int tx0 = tix( bx0 ), tx1 = tix( bx1 ), ty0 = tiy( by0 ), ty1 = tiy( by1 );
        if( ( tx0 == tx1 && ty0 == ty1 ) || ( ZSKIP & 2 ) ) { add( size_t( ty0 ) * TX + tx0, ar, tol ); return; }
        ++nstraddle;
        // A[ a ][ b ] = area in { x <= x-line a, y <= y-line b }, the last of each = no line ( the cell's area )
        SPoly p, q;
        poly_of_t( t, D, p );
        const int nx = tx1 - tx0, ny = ty1 - ty0;
        T A[ 4 ][ 4 ];
        if( nx > 3 || ny > 3 ) { // ( a cell larger than the tiles: its zones are declared bad, it is rebuilt exactly with its neighbours )
            for( int ty = ty0; ty <= ty1; ++ty ) for( int tx = tx0; tx <= tx1; ++tx ) zforce[ size_t( ty ) * TX + tx ] = 1;
            return;
        }
        for( int a = 0; a <= nx; ++a ) {
            const SPoly *pa = &p;
            if( a < nx ) { clip_x( p, X0 + ( tx0 + a + 1 ) * tile - sx, q ); pa = &q; }
            for( int b = 0; b < ny; ++b ) A[ a ][ b ] = area_y( *pa, Y0 + ( ty0 + b + 1 ) * tile - sy );
            A[ a ][ ny ] = a < nx ? poly_area( *pa ) : ar;
        }
        for( int b = 0; b <= ny; ++b )
            for( int a = 0; a <= nx; ++a ) {
                T v = A[ a ][ b ];
                if( a ) v -= A[ a - 1 ][ b ];
                if( b ) v -= A[ a ][ b - 1 ];
                if( a && b ) v += A[ a - 1 ][ b - 1 ];
                add( size_t( ty0 + b ) * TX + tx0 + a, v, tol );
            }
    }

    /// the ring-1 cells of a block ( lanes cut toward ring 2 are skipped: `ring2_list` does them ): boxes, tiles and tolerances in
    /// loops the compiler vectorizes, then the runs of consecutive cells in the same tile summed apart ( independent adds ) and added
    /// to the zone once ( Neumaier ), the error of a plain sum of m terms ( m eps sum |a| ) in the tolerance; straddling cells by pieces
    void deposit1( int r, int c0, int n, const typename H::Out1 &o ) {
        if constexpr ( ZSKIP & 1 ) return;
        const T sy = ys( r ), sx0 = xs( r, c0 ), it = 1 / tile;
        alignas( 64 ) T bx0[ B ], bx1[ B ], by0[ B ], by1[ B ], tl[ B ];
        alignas( 64 ) int zx0[ B ], zx1[ B ], zy0[ B ], zy1[ B ];
        for( int x = 0; x < B; ++x ) {
            const T sx = sx0 + T( x );
            bx0[ x ] = sx + o.zb[ 0 ][ x ]; bx1[ x ] = sx + o.zb[ 1 ][ x ]; by0[ x ] = sy + o.zb[ 2 ][ x ]; by1[ x ] = sy + o.zb[ 3 ][ x ];
            const T a = -o.zb[ 0 ][ x ] > o.zb[ 1 ][ x ] ? -o.zb[ 0 ][ x ] : o.zb[ 1 ][ x ], b = -o.zb[ 2 ][ x ] > o.zb[ 3 ][ x ] ? -o.zb[ 2 ][ x ] : o.zb[ 3 ][ x ];
            tl[ x ] = tol_c * eps * ( 1 + ( a > b ? a : b ) ) * ( 1 + ( o.zb[ 1 ][ x ] - o.zb[ 0 ][ x ] ) + ( o.zb[ 3 ][ x ] - o.zb[ 2 ][ x ] ) );
            zx0[ x ] = int( ( bx0[ x ] - X0 ) * it ); zx1[ x ] = int( ( bx1[ x ] - X0 ) * it );
            zy0[ x ] = int( ( by0[ x ] - Y0 ) * it ); zy1[ x ] = int( ( by1[ x ] - Y0 ) * it );
        }
        const size_t i0 = size_t( r ) * Wc + c0;
        for( int x = 0; x < n; ++x ) set_box( i0 + x, bx0[ x ], bx1[ x ], by0[ x ], by1[ x ] );
        size_t cur = ~size_t( 0 );
        T rs = 0, rt = 0, ra = 0; int m = 0;
        auto flush = [&] { if( m ) add( cur, rs, rt + T( m ) * eps * ra ); rs = rt = ra = 0; m = 0; };
        for( int x = 0; x < n; ++x ) {
            const size_t i = i0 + x;
            if( ( o.inv >> x ) & 1 ) { area[ i ] = 0; lvl[ i ] = 1; set_empty( i ); if( want_hess ) hess_clear( i ); continue; }
            if( ( o.cut >> x ) & 1 ) continue;
            area[ i ] = o.area[ x ]; lvl[ i ] = 1;
            if( want_hess ) for( int k = 0; k < 6; ++k ) W6[ 6 * i + k ] = ( ( o.fin[ k ] >> x ) & 1 ) ? o.wk[ k ][ x ] : T( 0 );
            const int tx0 = std::clamp( zx0[ x ], 0, TX - 1 ), tx1 = std::clamp( zx1[ x ], 0, TX - 1 ), ty0 = std::clamp( zy0[ x ], 0, TY - 1 ), ty1 = std::clamp( zy1[ x ], 0, TY - 1 );
            if( tx0 != tx1 || ty0 != ty1 ) {
                T zb[ 4 ] = { o.zb[ 0 ][ x ], o.zb[ 1 ][ x ], o.zb[ 2 ][ x ], o.zb[ 3 ][ x ] }, t[ 6 ];
                for( int k = 0; k < 6; ++k ) t[ k ] = o.tt[ k ][ x ];
                deposit( r, c0 + x, o.area[ x ], zb, t, 6 );
                continue;
            }
            const size_t z = size_t( ty0 ) * TX + tx0;
            if( z != cur ) { flush(); cur = z; }
            rs += o.area[ x ]; rt += tl[ x ]; ra += std::abs( o.area[ x ] ); ++m;
        }
        flush();
    }

    // ---------------------------------------------------------------- the cells beyond ring 2 ( rounds )
    T psi_max = 0;
    /// the box clipped by the seeds at hexagonal distance <= K, ring after ring; returns true when that is the exact cell: the psi_max
    /// rule says no farther seed can cut ( seeds of ring k are at >= k sqrt( 3 ) / 2 ), or no seed is left
    bool ring_cell( int r, int c, int K, Poly &out ) const {
        const T sx = xs( r, c ), sy = ys( r ), pi = psi( r, c );
        Poly buf;
        Poly *p = &out, *q = &buf;                 // ( ping-pong, no copy of the polygons )
        p->n = 4;
        p->x[ 0 ] = X0 - sx; p->y[ 0 ] = Y0 - sy; p->x[ 1 ] = X1 - sx; p->y[ 1 ] = Y0 - sy;
        p->x[ 2 ] = X1 - sx; p->y[ 2 ] = Y1 - sy; p->x[ 3 ] = X0 - sx; p->y[ 3 ] = Y1 - sy;
        p->l[ 0 ] = p->l[ 1 ] = p->l[ 2 ] = p->l[ 3 ] = -1;
        const int q0 = c - ( r - ( r & 1 ) ) / 2; // axial coordinates of the seed ( odd rows shifted right )
        bool exact = false;
        for( int k = 1; ; ++k ) {
            if( p->n == 0 ) { exact = true; break; }
            T R2 = 0;
            for( int e = 0; e < p->n; ++e ) R2 = std::max( R2, p->x[ e ] * p->x[ e ] + p->y[ e ] * p->y[ e ] );
            const T Rm = std::sqrt( R2 ), d = k * T( SQ3 / 2 );
            if( d >= Rm && d * d - 2 * d * Rm + pi - psi_max > 0 ) { exact = true; break; }
            if( k > Wc + R ) { exact = true; break; }
            if( k > K ) break;
            int aq = uq[ 4 ] * k, ar = ur[ 4 ] * k; // the ring from k u_4, k steps along each u_side
            for( int side = 0; side < 6; ++side )
                for( int st = 0; st < k; ++st ) {
                    const int rr = r + ar, qq = q0 + aq;
                    const int cc = qq + ( rr - ( rr & 1 ) ) / 2;
                    if( rr >= 0 && rr < R && cc >= 0 && cc < Wc && p->n ) {
                        const T dx = xs( rr, cc ) - sx, dy = ys( rr ) - sy;
                        const T cst = T( 0.5 ) * ( dx * dx + dy * dy + ( pi - psi( rr, cc ) ) );
                        // ( clip only when a vertex is outside: most seeds of a far ring do not cut )
                        bool out_ = false;
                        for( int e = 0; e < p->n && ! out_; ++e ) out_ = dx * p->x[ e ] + dy * p->y[ e ] > cst;
                        if( out_ ) { clip( *p, dx, dy, cst, *q, rr * Wc + cc ); std::swap( p, q ); }
                    }
                    aq += uq[ side ]; ar += ur[ side ];
                }
        }
        if( p != &out ) out = *p;
        return exact;
    }

    /// deposit of a polygon cell ( a round ): its box from the vertices, pieces as `deposit`
    void deposit_poly( size_t i, T ar, const Poly &p ) {
        const int r = int( i / Wc ), c = int( i % Wc );
        const T sx = xs( r, c ), sy = ys( r );
        T zb[ 4 ] = { p.x[ 0 ], p.x[ 0 ], p.y[ 0 ], p.y[ 0 ] };
        for( int e = 1; e < p.n; ++e ) { zb[ 0 ] = std::min( zb[ 0 ], p.x[ e ] ); zb[ 1 ] = std::max( zb[ 1 ], p.x[ e ] ); zb[ 2 ] = std::min( zb[ 2 ], p.y[ e ] ); zb[ 3 ] = std::max( zb[ 3 ], p.y[ e ] ); }
        set_box( i, sx + zb[ 0 ], sx + zb[ 1 ], sy + zb[ 2 ], sy + zb[ 3 ] );
        const T reach = std::max( std::max( -zb[ 0 ], zb[ 1 ] ), std::max( -zb[ 2 ], zb[ 3 ] ) );
        const T tol = tol_c * eps * ( 1 + reach ) * ( 1 + ( zb[ 1 ] - zb[ 0 ] ) + ( zb[ 3 ] - zb[ 2 ] ) );
        const T it = 1 / tile;
        const int tx0 = std::clamp( int( ( sx + zb[ 0 ] - X0 ) * it ), 0, TX - 1 ), tx1 = std::clamp( int( ( sx + zb[ 1 ] - X0 ) * it ), 0, TX - 1 );
        const int ty0 = std::clamp( int( ( sy + zb[ 2 ] - Y0 ) * it ), 0, TY - 1 ), ty1 = std::clamp( int( ( sy + zb[ 3 ] - Y0 ) * it ), 0, TY - 1 );
        if( tx0 == tx1 && ty0 == ty1 ) { add( size_t( ty0 ) * TX + tx0, ar, tol ); return; }
        const int nx = tx1 - tx0, ny = ty1 - ty0;
        if( nx > 3 || ny > 3 ) { for( int ty = ty0; ty <= ty1; ++ty ) for( int tx = tx0; tx <= tx1; ++tx ) zforce[ size_t( ty ) * TX + tx ] = 1; return; }
        Poly q;
        T A[ 4 ][ 4 ];
        for( int a = 0; a <= nx; ++a ) {
            const Poly *pa = &p;
            if( a < nx ) { clip_x( p, X0 + ( tx0 + a + 1 ) * tile - sx, q ); pa = &q; }
            for( int b = 0; b < ny; ++b ) A[ a ][ b ] = area_y( *pa, Y0 + ( ty0 + b + 1 ) * tile - sy );
            A[ a ][ ny ] = a < nx ? poly_area( *pa ) : ar;
        }
        for( int b = 0; b <= ny; ++b )
            for( int a = 0; a <= nx; ++a ) {
                T v = A[ a ][ b ];
                if( a ) v -= A[ a - 1 ][ b ];
                if( b ) v -= A[ a ][ b - 1 ];
                if( a && b ) v += A[ a - 1 ][ b - 1 ];
                add( size_t( ty0 + b ) * TX + tx0 + a, v, tol );
            }
    }

    /// does zone z miss the certificate
    bool zone_bad( size_t z ) const {
        const int tx = int( z % TX ), ty = int( z / TX );
        const T zx = std::min( X1, X0 + ( tx + 1 ) * tile ) - ( X0 + tx * tile ), zy = std::min( Y1, Y0 + ( ty + 1 ) * tile ) - ( Y0 + ty * tile );
        // ( + the lines of the zone: each cell places them relative to its rounded seed, so at eps ( |x| + |y| ) apart, which
        // moves the sum by that times the perimeter )
        const T lines = tol_c * eps * ( std::max( std::abs( X0 ), std::abs( X1 ) ) + std::max( std::abs( Y0 ), std::abs( Y1 ) ) ) * 2 * ( zx + zy );
        return zforce[ z ] || std::abs( ( zsum[ z ] - zx * zy ) + zcmp[ z ] ) > ztol[ z ] + tol_c * eps * zx * zy + lines;
    }
    /// does the ( non-empty ) cell i meet a bad zone
    bool meets_bad( size_t i ) const {
        const T it = 1 / tile;
        const int tx0 = std::clamp( int( ( cbox[ 4 * i + 0 ] - X0 ) * it ), 0, TX - 1 ), tx1 = std::clamp( int( ( cbox[ 4 * i + 1 ] - X0 ) * it ), 0, TX - 1 );
        const int ty0 = std::clamp( int( ( cbox[ 4 * i + 2 ] - Y0 ) * it ), 0, TY - 1 ), ty1 = std::clamp( int( ( cbox[ 4 * i + 3 ] - Y0 ) * it ), 0, TY - 1 );
        for( int ty = ty0; ty <= ty1; ++ty ) for( int tx = tx0; tx <= tx1; ++tx ) if( zbad[ size_t( ty ) * TX + tx ] ) return true;
        return false;
    }

    // ---------------------------------------------------------------- the whole lattice
    void run() {
        std::fill( zsum.begin(), zsum.end(), T( 0 ) ); std::fill( zcmp.begin(), zcmp.end(), T( 0 ) ); std::fill( ztol.begin(), ztol.end(), T( 0 ) ); std::fill( zforce.begin(), zforce.end(), 0 );
        nstraddle = nbad = nexact = nring2 = nround = 0; aborted = false;
        if( want_hess ) { W6.resize( 6 * area.size() ); ext.clear(); }
        typename H::Out1 o1;
        typename H::Out2 o2;
        alignas( 64 ) int pend[ B ];
        auto take2 = [&]( int r, const int *cols, int n ) {
            for( int x = 0; x < n; ++x ) {
                const size_t i = size_t( r ) * Wc + cols[ x ];
                const bool inv = ( o2.inv >> x ) & 1;
                area[ i ] = inv ? T( 0 ) : o2.area[ x ]; lvl[ i ] = 2;
                T zb[ 4 ] = { o2.zb[ 0 ][ x ], o2.zb[ 1 ][ x ], o2.zb[ 2 ][ x ], o2.zb[ 3 ][ x ] }, t[ 12 ];
                for( int k = 0; k < 12; ++k ) t[ k ] = o2.tt[ k ][ x ];
                if( ! inv ) deposit( r, cols[ x ], area[ i ], zb, t, 12 );
                else set_empty( i );
                if( want_hess ) {
                    hess_clear( i );
                    if( ! inv )
                        for( int d = 0; d < 12; ++d ) {
                            const T w = o2.wk[ d ][ x ];
                            if( w == 0 ) continue;
                            const int q = d / 2;
                            if( d % 2 ) ext[ i ].push_back( { nbr( r, cols[ x ], vq[ q ], vr[ q ] ), w } );
                            else if( ( o2.sec[ d ] >> x ) & 1 ) ext[ i ].push_back( { nbr( r, cols[ x ], 2 * uq[ q ], 2 * ur[ q ] ), w } );
                            else W6[ 6 * i + q ] = w;
                        }
                }
            }
            nring2 += n;
        };
        for( int r = 0; r < R; ++r ) {
            const bool border = r == 0 || r == R - 1;
            int np = 0;
            auto flush = [&] {
                if( r & 1 ) H::template ring2_list<1>( Wc, R, pp, P, r, pend, np, o2 ); else H::template ring2_list<0>( Wc, R, pp, P, r, pend, np, o2 );
                take2( r, pend, np ); np = 0;
            };
            for( int c0 = 0; c0 < Wc; c0 += B ) {
                const int n = std::min( B, Wc - c0 );
                if( border ) {
                    if( r & 1 ) H::template ring2<1>( Wc, R, pp, P, r, c0, o2 ); else H::template ring2<0>( Wc, R, pp, P, r, c0, o2 );
                    int cols[ B ]; for( int x = 0; x < n; ++x ) cols[ x ] = c0 + x;
                    take2( r, cols, n );
                    continue;
                }
                if( r & 1 ) H::template ring1<1,true>( Wc, R, pp, P, r, c0, o1 ); else H::template ring1<0,true>( Wc, R, pp, P, r, c0, o1 );
                for( int x = 0; x < n; ++x ) if( ! ( ( o1.inv >> x ) & 1 ) && ( ( o1.cut >> x ) & 1 ) ) { pend[ np++ ] = c0 + x; if( np == B ) flush(); }
                deposit1( r, c0, n, o1 );
            }
            if( np ) flush();
        }
        if constexpr ( ZSKIP != 0 ) return;
        // the zones
        bool any = false;
        for( size_t z = 0; z < zsum.size(); ++z ) { zbad[ z ] = zone_bad( z ); nbad += zbad[ z ]; any |= zbad[ z ]; }
        if( diag_zones ) { // ( diagnostics: the excess and the tolerance of the bad zones of the first table )
            zdiag.clear();
            for( size_t z = 0; z < zsum.size(); ++z ) if( zbad[ z ] ) {
                const int tx = int( z % TX ), ty = int( z / TX );
                const T zx = std::min( X1, X0 + ( tx + 1 ) * tile ) - ( X0 + tx * tile ), zy = std::min( Y1, Y0 + ( ty + 1 ) * tile ) - ( Y0 + ty * tile );
                zdiag.push_back( { ( zsum[ z ] - zx * zy ) + zcmp[ z ], ztol[ z ], T( zforce[ z ] ) } );
            }
        }
        if( ! any || no_rounds ) return;
        psi_max = -std::numeric_limits<T>::infinity();
        for( int r = 0; r < R; ++r ) for( int c = 0; c < Wc; ++c ) psi_max = std::max( psi_max, psi( r, c ) );
        // rounds: the cells that meet a bad zone, limited to hexagonal distance K ( still containing the true cells ), deposited again;
        // the sums of the bad zones only ( no certified cell meets them ), the zones that still miss go to the next round
        std::vector<size_t> todo;
        for( size_t i = 0; i < area.size(); ++i ) if( ! is_empty( i ) && meets_bad( i ) ) todo.push_back( i );
        // the caller may stop here: the cells out of `todo` are exact, the areas of those in it are upper bounds ( their cells contain
        // the true ones ). A line search rejects a trial this way without the rounds ( wild trials send cells to the psi_max radius )
        if( pre_rounds && ! pre_rounds( todo ) ) { aborted = true; return; }
        Poly p;
        for( int K = K0; ! todo.empty(); K *= 2 ) {
            const bool last = K > max_K;
            const auto t_round = std::chrono::steady_clock::now();
            const size_t n_round = todo.size();
            size_t n_exact_round = 0;
            std::fill( zsum.begin(), zsum.end(), T( 0 ) ); std::fill( zcmp.begin(), zcmp.end(), T( 0 ) ); std::fill( ztol.begin(), ztol.end(), T( 0 ) ); std::fill( zforce.begin(), zforce.end(), 0 );
            for( size_t i : todo ) {
                const bool exact = ring_cell( int( i / Wc ), int( i % Wc ), last ? std::numeric_limits<int>::max() / 2 : K, p );
                area[ i ] = p.n ? poly_area( p ) : T( 0 );
                lvl[ i ] = exact ? 4 : 3; ++nround; n_exact_round += exact;
                if( p.n ) deposit_poly( i, area[ i ], p ); else set_empty( i );
                if( want_hess ) {
                    hess_clear( i );
                    const int r = int( i / Wc ), c = int( i % Wc );
                    for( int e = 0; e < p.n; ++e ) {
                        const int j = p.l[ e ];
                        if( j < 0 ) continue;
                        const int f = e + 1 == p.n ? 0 : e + 1;
                        const T L = std::hypot( p.x[ f ] - p.x[ e ], p.y[ f ] - p.y[ e ] );
                        if( L == 0 ) continue;
                        const T dx = xs( j / Wc, j % Wc ) - xs( r, c ), dy = ys( j / Wc ) - ys( r );
                        hess_face( i, j, L / ( 2 * std::sqrt( dx * dx + dy * dy ) ) );
                    }
                }
            }
            if( round_stats ) round_log.push_back( { double( K ), double( n_round ), double( n_exact_round ), std::chrono::duration<double>( std::chrono::steady_clock::now() - t_round ).count() } );
            if( last ) { nexact += long( todo.size() ); break; }
            for( size_t z = 0; z < zsum.size(); ++z ) if( zbad[ z ] ) zbad[ z ] = zone_bad( z );
            std::vector<size_t> next;
            for( size_t i : todo ) if( ! is_empty( i ) && meets_bad( i ) ) next.push_back( i );
            todo.swap( next );
        }
    }
};

} // namespace hexgrid
