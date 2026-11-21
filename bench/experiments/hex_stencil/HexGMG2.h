// STRUCTURED geometric multigrid for the hessian of a hexagonal lattice, on the 18-NEIGHBOUR stencil ( rings 1 and 2 ), which the
// Galerkin product keeps: a coupling at distance <= 2 gives coarse couplings at distance <= ( 2 + 1 + 1 ) / 2 = 2 ( parents within
// one fine spacing of each end ). So every level is 18 planes of weights, no list that grows down the levels ( HexGMG.h's ):
// only the faces past ring 2 of the cells of the rounds stay a short list ( level 0, and their images ).
//
// Levels: hexagonal lattices in offset rows again ( level l + 1 = the points of even axial q and r: even rows r = 2 r', columns
// c = 2 c' + ( r' & 1 ) ), arrays padded by 2 rows / columns of zeros ( P = W + 4 ), `valid` 0 on the pads and the missing ends.
// Directions: 0..5 u_k ( ( 1, 0 ), ( 0, 1 ), ( -1, 1 ), ... ), 6..11 2 u_k, 12..17 v_k = u_k + u_{k+1}; the opposite of a direction
// is the same family at k + 3. Interpolation: linear, 1 or 2 parents ( a fine point is a coarse point or the middle of a coarse
// edge ). Galerkin edge by edge: w ( x_i - x_j )^2 with x = P y is - w sum_{a<b} c_a c_b ( y_a - y_b )^2, c = P^T ( e_i - e_j ).
// Smoother: Gauss-Seidel by the 3 colours ( q - r ) mod 3 ( the v neighbours share a colour: sequential within a colour ), rows and
// columns in order before the coarse correction, everything reversed after ( symmetric V-cycle ). Coarsest: dense Cholesky.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace hexgrid {

struct HexGMG2 {
    using T = double;
    static constexpr int D = 18;
    static constexpr int uq_[ 6 ] = { 1, 0, -1, -1, 0, 1 }, ur_[ 6 ] = { 0, 1, 1, 0, -1, -1 };
    static int dq_of( int d ) { const int k = d % 6; return d < 6 ? uq_[ k ] : d < 12 ? 2 * uq_[ k ] : uq_[ k ] + uq_[ ( k + 1 ) % 6 ]; }
    static int dr_of( int d ) { const int k = d % 6; return d < 6 ? ur_[ k ] : d < 12 ? 2 * ur_[ k ] : ur_[ k ] + ur_[ ( k + 1 ) % 6 ]; }
    static int opp( int d ) { return ( d / 6 ) * 6 + ( d % 6 + 3 ) % 6; }
    static int dcol_( int dq, int dr, int SH ) { const int sh2 = ( SH + ( dr & 1 ) ) & 1; return dq + ( dr + SH - sh2 ) / 2; }
    static int axq( int r, int c ) { return c - ( r - ( r & 1 ) ) / 2; }
    static double now_s() { return std::chrono::duration<double>( std::chrono::steady_clock::now().time_since_epoch() ).count(); }
    int dir_of[ 9 ][ 9 ];                          // ( dq + 4, dr + 4 ) -> direction, -1 out of the stencil
    HexGMG2() {
        for( auto &row : dir_of ) for( int &v : row ) v = -1;
        for( int d = 0; d < D; ++d ) dir_of[ dq_of( d ) + 4 ][ dr_of( d ) + 4 ] = d;
    }

    struct Extra { size_t i, j; T w; };
    struct Level {
        int R = 0, W = 0, P = 0; size_t ps = 0;
        std::vector<unsigned char> valid;
        int off[ 2 ][ D ];
        std::vector<signed char> dir_by_off[ 2 ];  // ( padded index offset + 2 P + 2 -> direction, per row parity; -1 out of the stencil )
        std::vector<unsigned char> par;            // ( the parity of the row of each padded index )
        std::vector<T> w, diag, idiag, x, b, res;     // w: D planes of ps
        std::vector<T> cd, ct, cd2;                // ( Chebyshev: the direction, A d, the next direction )
        std::vector<float> wf;                     // ( the planes in float for the kernels of the V-cycle, `wfloat` )
        T lmax = 2;                                // ( the largest eigenvalue of D^-1 A, estimated )
        bool has2 = false;                         // ( some weight in the ring-2 planes )
        std::vector<Extra> ext;                    // ( the couplings past the stencil )
        std::vector<int> xb; std::vector<size_t> xn; std::vector<T> xw;   // ( the same per point, CSR over the padded indices )
        std::vector<int> cst[ 3 ];                 // per colour, the first column of each row
        std::vector<size_t> fi, p0, p1; std::vector<T> w0, w1;   // the valid points and their parents in the next level
        std::vector<T> rw;                         // ( 7 x ps of the NEXT level: restriction weights of the fine point at the coarse
                                                   //   point's place, k = 6, and of its 6 fine neighbours, k = 0..5 )
        size_t pid( int r, int c ) const { return size_t( r + 2 ) * P + c + 2; }
        void rc( size_t q, int &r, int &c ) const { r = int( q / P ) - 2; c = int( q % P ) - 2; }
    };
    std::vector<Level> L;
    int n_coarsest = 200, nu = 1;
    double ring2_planes_ratio = 1.0;                // ( 1: ring 2 always in the lists, the fastest measured )
    int cheb = 0;                                  // > 0: CHEBYSHEV smoothing of this degree instead of Gauss-Seidel ( contiguous, vectorized )
    T cheb_alpha = 4;                              // ( the interval [ lmax / alpha, lmax ] )
    bool wfloat = true;                            // ( the planes read as floats by Ax and the fused Chebyshev step )
    bool fused = true;                             // ( one pass per Chebyshev step: x += d, r -= A d, the next d )              // ( ring 2 as planes when more than this share of the points use it )
    std::vector<T> chol; std::vector<size_t> cidx; int nc = 0;
    double ts[ 3 ] = {}, tv[ 5 ] = {}, tg[ 4 ] = {};   // ( tg: level 0 fill, edges, finish, parents arrays )

    // ---------------------------------------------------------------- levels and interpolation ( once per lattice )
    void geometry( Level &l ) {
        l.P = l.W + 4; l.ps = size_t( l.R + 4 ) * l.P;
        for( int sh = 0; sh < 2; ++sh ) for( int d = 0; d < D; ++d ) l.off[ sh ][ d ] = dr_of( d ) * l.P + dcol_( dq_of( d ), dr_of( d ), sh );
        for( int sh = 0; sh < 2; ++sh ) { l.dir_by_off[ sh ].assign( 4 * l.P + 5, -1 ); for( int d = 0; d < D; ++d ) l.dir_by_off[ sh ][ l.off[ sh ][ d ] + 2 * l.P + 2 ] = ( signed char )d; }
        l.par.assign( size_t( l.R + 4 ) * l.P, 0 );
        for( int r = -2; r < l.R + 2; ++r ) for( int c = 0; c < l.P; ++c ) l.par[ size_t( r + 2 ) * l.P + c ] = ( unsigned char )( r & 1 );
        for( int k = 0; k < 3; ++k ) { l.cst[ k ].assign( l.R, 0 ); for( int r = 0; r < l.R; ++r ) l.cst[ k ][ r ] = ( ( k + r + ( r - ( r & 1 ) ) / 2 ) % 3 + 3 ) % 3; }
        l.w.assign( D * l.ps, 0 ); l.diag.assign( l.ps, 0 ); l.idiag.assign( l.ps, 0 );
        l.x.assign( l.ps, 0 ); l.b.assign( l.ps, 0 ); l.res.assign( l.ps, 0 );
    }
    void setup_levels( int R, int W ) {
        L.clear();
        L.emplace_back();
        Level &l0 = L[ 0 ];
        l0.R = R; l0.W = W; geometry( l0 );
        l0.valid.assign( l0.ps, 0 );
        for( int r = 0; r < R; ++r ) for( int c = 0; c < W; ++c ) l0.valid[ l0.pid( r, c ) ] = 1;
        while( true ) {
            Level &f = L.back();
            int nv = 0; for( auto v : f.valid ) nv += v;
            if( nv <= n_coarsest ) break;
            Level c;
            c.R = ( f.R + 1 ) / 2; c.W = ( f.W + 1 ) / 2; geometry( c );
            c.valid.assign( c.ps, 0 );
            for( int r = 0; r < c.R; ++r ) for( int cc = 0; cc < c.W; ++cc ) {
                const int fr = 2 * r, fc = 2 * cc + ( r & 1 );
                if( fr < f.R && fc < f.W && f.valid[ f.pid( fr, fc ) ] ) c.valid[ c.pid( r, cc ) ] = 1;
            }
            auto coarse_of = [&]( int q, int r ) -> long {
                if( ( q & 1 ) || ( r & 1 ) ) return -1;
                const int rc = r / 2, qc = q / 2, cc = qc + ( rc - ( rc & 1 ) ) / 2;
                if( rc < 0 || rc >= c.R || cc < 0 || cc >= c.W || ! c.valid[ c.pid( rc, cc ) ] ) return -1;
                return long( c.pid( rc, cc ) );
            };
            f.fi.clear(); f.p0.clear(); f.p1.clear(); f.w0.clear(); f.w1.clear();
            for( int r = 0; r < f.R; ++r ) for( int cc = 0; cc < f.W; ++cc ) {
                if( ! f.valid[ f.pid( r, cc ) ] ) continue;
                const int q = axq( r, cc ), oq = q & 1, orr = r & 1;
                long a = -1, b = -1;
                if( ! oq && ! orr ) a = coarse_of( q, r );
                else if( oq && ! orr ) { a = coarse_of( q - 1, r ); b = coarse_of( q + 1, r ); }
                else if( ! oq && orr ) { a = coarse_of( q, r - 1 ); b = coarse_of( q, r + 1 ); }
                else { a = coarse_of( q - 1, r + 1 ); b = coarse_of( q + 1, r - 1 ); }
                if( a < 0 ) std::swap( a, b );
                if( a < 0 ) {
                    int best = 1 << 30;
                    for( int dq = -2; dq <= 2; ++dq ) for( int dr = -2; dr <= 2; ++dr ) {
                        const long k = coarse_of( q + dq, r + dr ); const int d = dq * dq + dr * dr + dq * dr;
                        if( k >= 0 && d < best ) { best = d; a = k; }
                    }
                }
                f.fi.push_back( f.pid( r, cc ) );
                f.p0.push_back( size_t( a ) ); f.w0.push_back( b >= 0 ? T( 0.5 ) : T( 1 ) );
                f.p1.push_back( size_t( b >= 0 ? b : a ) ); f.w1.push_back( b >= 0 ? T( 0.5 ) : T( 0 ) );
            }
            L.push_back( std::move( c ) );
        }
        // the restriction as a gather: coarse ( r', c' ) sits on fine ( 2 r', 2 c' + ( r' & 1 ) ), its sources are that point and its 6
        // fine neighbours ( the middles of its coarse edges )
        for( size_t li = 0; li + 1 < L.size(); ++li ) {
            Level &f = L[ li ]; const Level &c = L[ li + 1 ];
            f.rw.assign( 7 * c.ps, T( 0 ) );
            for( size_t t = 0; t < f.fi.size(); ++t ) {
                const size_t i = f.fi[ t ];
                for( int u = 0; u < 2; ++u ) {
                    const size_t a = u ? f.p1[ t ] : f.p0[ t ]; const T wv = u ? f.w1[ t ] : f.w0[ t ];
                    if( wv == 0 ) continue;
                    int rc_, cc_; c.rc( a, rc_, cc_ );
                    const size_t fa = f.pid( 2 * rc_, 2 * cc_ + ( rc_ & 1 ) );
                    int k = -1;
                    if( fa == i ) k = 6;
                    else for( int d = 0; d < 6 && k < 0; ++d ) if( size_t( long( fa ) + f.off[ 0 ][ d ] ) == i ) k = d;   // ( fine row 2 r' is even )
                    if( k < 0 ) { far_parent = true; continue; }
                    f.rw[ k * c.ps + a ] += wv;
                }
            }
        }
    }
    bool far_parent = false;
    bool lag = false, built = false;                       // ( a corner point whose nearest parent is not a ring-1 neighbour: gather incomplete )

    // ---------------------------------------------------------------- operators ( per Newton step )
    void add_coupling( Level &l, size_t a, size_t b, T v ) {
        if( a == b || v == 0 ) return;
        const long dd = long( b ) - long( a ) + 2 * l.P + 2;
        const int d = dd >= 0 && dd < long( l.dir_by_off[ 0 ].size() ) ? l.dir_by_off[ l.par[ a ] ][ dd ] : -1;
        if( d >= 0 ) { l.w[ d * l.ps + a ] += v; l.w[ opp( d ) * l.ps + b ] += v; return; }
        l.ext.push_back( { a, b, v } );
    }
    void finish( Level &l ) {                      // diagonal, has2, merged ext + CSR
        std::fill( l.diag.begin(), l.diag.end(), T( 0 ) );
        // ring 2: planes when many points use it, else moved into the per-point lists ( 12 planes read for a few cells cost more )
        size_t n2 = 0, nv = 0;
        for( size_t q = 0; q < l.ps; ++q ) { if( ! l.valid[ q ] ) continue; ++nv; bool any = false; for( int d = 6; d < D && ! any; ++d ) any = l.w[ d * l.ps + q ] != 0; n2 += any; }
        l.has2 = n2 > size_t( ring2_planes_ratio * double( nv ) );
        if( ! l.has2 && n2 ) {
            for( size_t q = 0; q < l.ps; ++q ) {
                if( ! l.valid[ q ] ) continue;
                int r, c; l.rc( q, r, c );
                for( int d = 6; d < D; ++d ) {
                    const T v = l.w[ d * l.ps + q ];
                    if( v == 0 ) continue;
                    const size_t j = size_t( long( q ) + l.off[ r & 1 ][ d ] );
                    if( q < j ) l.ext.push_back( { q, j, v } );   // ( each pair once: the planes hold it at both ends )
                    l.w[ d * l.ps + q ] = 0;
                }
            }
        }
        for( int d = 0; d < D; ++d ) for( size_t q = 0; q < l.ps; ++q ) l.diag[ q ] += l.w[ d * l.ps + q ];
        std::vector<Extra> m;
        for( auto e : l.ext ) { if( e.i > e.j ) std::swap( e.i, e.j ); m.push_back( e ); }
        std::sort( m.begin(), m.end(), []( const Extra &u, const Extra &v ) { return u.i != v.i ? u.i < v.i : u.j < v.j; } );
        l.ext.clear();
        for( auto &e : m ) { if( ! l.ext.empty() && l.ext.back().i == e.i && l.ext.back().j == e.j ) l.ext.back().w += e.w; else l.ext.push_back( e ); }
        for( auto &e : l.ext ) { l.diag[ e.i ] += e.w; l.diag[ e.j ] += e.w; }
        for( size_t q = 0; q < l.ps; ++q ) l.idiag[ q ] = l.valid[ q ] && l.diag[ q ] > 0 ? 1 / l.diag[ q ] : 0;
        l.xb.assign( l.ps + 1, 0 );
        for( auto &e : l.ext ) { ++l.xb[ e.i + 1 ]; ++l.xb[ e.j + 1 ]; }
        for( size_t q = 0; q < l.ps; ++q ) l.xb[ q + 1 ] += l.xb[ q ];
        l.xn.resize( l.xb[ l.ps ] ); l.xw.resize( l.xb[ l.ps ] );
        std::vector<int> fill( l.xb.begin(), l.xb.end() - 1 );
        for( auto &e : l.ext ) { l.xn[ fill[ e.i ] ] = e.j; l.xw[ fill[ e.i ]++ ] = e.w; l.xn[ fill[ e.j ] ] = e.i; l.xw[ fill[ e.j ]++ ] = e.w; }
    }
    /// level 0: W6 ( 6 per point, row-major R x W, symmetric pairs ), ext ( unpadded ( i, j, w ), applied to both ends )
    template<class E>
    void setup_ops( const std::vector<T> &W6, const std::vector<E> &ext ) {
        double t0 = now_s();
        Level &l0 = L[ 0 ];
        std::fill( l0.w.begin(), l0.w.end(), T( 0 ) ); l0.ext.clear();
        for( int r = 0; r < l0.R; ++r ) for( int c = 0; c < l0.W; ++c ) {
            const size_t q = l0.pid( r, c ), i = size_t( r ) * l0.W + c;
            for( int k = 0; k < 6; ++k ) l0.w[ k * l0.ps + q ] = W6[ 6 * i + k ];
        }
        for( const auto &e : ext ) add_coupling( l0, l0.pid( e.i / l0.W, e.i % l0.W ), l0.pid( e.j / l0.W, e.j % l0.W ), e.w );
        tg[ 0 ] += now_s() - t0; double tq = now_s();
        finish( l0 );
        tg[ 2 ] += now_s() - tq;
        // ( `lag`: the coarse operators of the previous call kept, only level 0 rebuilt: a looser but SPD preconditioner )
        const bool lagged = lag && built;
        built = true;
        for( size_t li = 0; li + 1 < L.size() && ! lagged; ++li ) {
            Level &f = L[ li ], &c = L[ li + 1 ];
            std::fill( c.w.begin(), c.w.end(), T( 0 ) ); c.ext.clear();
            // the parents of each padded fine index ( through `fi` )
            double tq = now_s();
            std::vector<long> pa( f.ps, -1 ), pb( f.ps, -1 ); std::vector<T> wa( f.ps, 0 ), wb( f.ps, 0 );
            for( size_t t = 0; t < f.fi.size(); ++t ) { pa[ f.fi[ t ] ] = long( f.p0[ t ] ); wa[ f.fi[ t ] ] = f.w0[ t ]; pb[ f.fi[ t ] ] = f.w1[ t ] != 0 ? long( f.p1[ t ] ) : -1; wb[ f.fi[ t ] ] = f.w1[ t ]; }
            tg[ 3 ] += now_s() - tq; tq = now_s();
            auto edge = [&]( size_t i, size_t j, T w ) {
                size_t ids[ 4 ]; T cs[ 4 ]; int m = 0;
                auto put = [&]( long a, T v ) { if( a < 0 || v == 0 ) return; for( int t = 0; t < m; ++t ) if( ids[ t ] == size_t( a ) ) { cs[ t ] += v; return; } ids[ m ] = size_t( a ); cs[ m ] = v; ++m; };
                put( pa[ i ], wa[ i ] ); put( pb[ i ], wb[ i ] ); put( pa[ j ], -wa[ j ] ); put( pb[ j ], -wb[ j ] );
                for( int s = 0; s < m; ++s ) for( int t = s + 1; t < m; ++t ) add_coupling( c, ids[ s ], ids[ t ], -w * cs[ s ] * cs[ t ] );
            };
            static constexpr int half_dirs[ 9 ] = { 0, 1, 2, 6, 7, 8, 12, 13, 14 };   // ( each edge once )
            for( size_t i : f.fi ) {
                int r, cc; f.rc( i, r, cc );
                for( int d : half_dirs ) {
                    if( d >= 6 && ! f.has2 ) break;
                    const T w = f.w[ d * f.ps + i ];
                    if( w == 0 ) continue;
                    edge( i, size_t( long( i ) + f.off[ r & 1 ][ d ] ), w );
                }
            }
            for( const Extra &e : f.ext ) edge( e.i, e.j, e.w );
            tg[ 1 ] += now_s() - tq; tq = now_s();
            finish( c );
            tg[ 2 ] += now_s() - tq;
        }
        ts[ 0 ] += now_s() - t0; t0 = now_s();
        if( lagged ) {
            l0.wf.resize( l0.w.size() ); for( size_t q = 0; q < l0.w.size(); ++q ) l0.wf[ q ] = float( l0.w[ q ] );
            if( cheb ) estimate_lmax( l0 );
            ts[ 1 ] += now_s() - t0;
            return;
        }
        // coarsest: dense Cholesky of A + s 1 1^T / n over its valid points
        Level &c = L.back();
        cidx.clear(); for( size_t q = 0; q < c.ps; ++q ) if( c.valid[ q ] ) cidx.push_back( q );
        nc = int( cidx.size() );
        std::vector<int> pos( c.ps, -1 ); for( int a = 0; a < nc; ++a ) pos[ cidx[ a ] ] = a;
        chol.assign( size_t( nc ) * nc, 0 );
        T sc = 0; for( size_t q : cidx ) sc = std::max( sc, c.diag[ q ] );
        for( int a = 0; a < nc; ++a ) {
            const size_t q = cidx[ a ];
            int r, cc; c.rc( q, r, cc );
            chol[ size_t( a ) * nc + a ] += c.diag[ q ];
            for( int b2 = 0; b2 < nc; ++b2 ) chol[ size_t( a ) * nc + b2 ] += sc / nc;
            for( int d = 0; d < D; ++d ) { const T v = c.w[ d * c.ps + q ]; if( v == 0 ) continue; const int j = pos[ size_t( long( q ) + c.off[ r & 1 ][ d ] ) ]; if( j >= 0 ) chol[ size_t( a ) * nc + j ] -= v; }
        }
        for( auto &e : c.ext ) { const int i = pos[ e.i ], j = pos[ e.j ]; chol[ size_t( i ) * nc + j ] -= e.w; chol[ size_t( j ) * nc + i ] -= e.w; }
        for( int j = 0; j < nc; ++j ) {
            T d = chol[ size_t( j ) * nc + j ];
            for( int k = 0; k < j; ++k ) d -= chol[ size_t( j ) * nc + k ] * chol[ size_t( j ) * nc + k ];
            d = std::sqrt( std::max( d, T( 1e-300 ) ) );
            chol[ size_t( j ) * nc + j ] = d;
            for( int i = j + 1; i < nc; ++i ) {
                T s = chol[ size_t( i ) * nc + j ];
                for( int k = 0; k < j; ++k ) s -= chol[ size_t( i ) * nc + k ] * chol[ size_t( j ) * nc + k ];
                chol[ size_t( i ) * nc + j ] = s / d;
            }
        }
        for( auto &l : L ) { l.wf.resize( l.w.size() ); for( size_t q = 0; q < l.w.size(); ++q ) l.wf[ q ] = float( l.w[ q ] ); }
        if( cheb ) for( size_t li = 0; li + 1 < L.size(); ++li ) estimate_lmax( L[ li ] );
        ts[ 1 ] += now_s() - t0;
    }

    // ---------------------------------------------------------------- V-cycle
    // Gauss-Seidel on one colour, rows / columns forward or all reversed ( ND = 6 or 18 planes, unrolled )
    template<int ND> void smooth_t( Level &l, int k, bool rev ) {
        const size_t ps = l.ps;
        const bool ext = ! l.ext.empty();
        for( int t = 0; t < l.R; ++t ) {
            const int r = rev ? l.R - 1 - t : t;
            const int *o = l.off[ r & 1 ];
            const size_t base = l.pid( r, 0 );
            T *x = l.x.data() + base; const T *b = l.b.data() + base, *id = l.idiag.data() + base, *w = l.w.data() + base;
            const int c0 = l.cst[ k ][ r ];
            if( c0 >= l.W ) continue;
            const int cl = c0 + ( l.W - 1 - c0 ) / 3 * 3;
            for( int u = 0; c0 + 3 * u < l.W; ++u ) {
                const int c = rev ? cl - 3 * u : c0 + 3 * u;
                T s = b[ c ];
                #pragma clang loop unroll( full )
                for( int d = 0; d < ND; ++d ) s += w[ d * ps + c ] * x[ c + o[ d ] ];
                if( ext ) { const size_t q = base + c; for( int e = l.xb[ q ]; e < l.xb[ q + 1 ]; ++e ) s += l.xw[ e ] * l.x[ l.xn[ e ] ]; }
                x[ c ] = id[ c ] * s;
            }
        }
    }
    void smooth( Level &l, int k, bool rev ) { if( l.has2 ) smooth_t<D>( l, k, rev ); else smooth_t<6>( l, k, rev ); }
    template<int ND> void residual_t( Level &l ) {
        const size_t ps = l.ps;
        for( int r = 0; r < l.R; ++r ) {
            const int *o = l.off[ r & 1 ];
            const size_t base = l.pid( r, 0 );
            const T *x = l.x.data() + base, *b = l.b.data() + base, *dg = l.diag.data() + base, *w = l.w.data() + base; T *res = l.res.data() + base;
            for( int c = 0; c < l.W; ++c ) {
                T s = b[ c ] - dg[ c ] * x[ c ];
                #pragma clang loop unroll( full )
                for( int d = 0; d < ND; ++d ) s += w[ d * ps + c ] * x[ c + o[ d ] ];
                res[ c ] = s;
            }
        }
        for( auto &e : l.ext ) { l.res[ e.i ] += e.w * l.x[ e.j ]; l.res[ e.j ] += e.w * l.x[ e.i ]; }
    }
    void residual( Level &l ) { if( l.has2 ) residual_t<D>( l ); else residual_t<6>( l ); }

    // y = A v ( rows contiguous, the planes at fixed offsets: vectorized ), the lists after
    template<int ND,class WT> void Ax_tw( const Level &l, const WT *wplanes, const T *v, T *y ) const {
        const size_t ps = l.ps;
        for( int r = 0; r < l.R; ++r ) {
            const int *o = l.off[ r & 1 ];
            const size_t base = l.pid( r, 0 );
            const T *x = v + base, *dg = l.diag.data() + base; const WT *w = wplanes + base; T *yy = y + base;
            for( int c = 0; c < l.W; ++c ) {
                T s = dg[ c ] * x[ c ];
                #pragma clang loop unroll( full )
                for( int d = 0; d < ND; ++d ) s -= w[ d * ps + c ] * x[ c + o[ d ] ];
                yy[ c ] = s;
            }
        }
        for( auto &e : l.ext ) { y[ e.i ] -= e.w * v[ e.j ]; y[ e.j ] -= e.w * v[ e.i ]; }
    }
    template<int ND> void Ax_t( const Level &l, const T *v, T *y ) const { if( wfloat && ! l.wf.empty() ) Ax_tw<ND>( l, l.wf.data(), v, y ); else Ax_tw<ND>( l, l.w.data(), v, y ); }
    void Ax( const Level &l, const T *v, T *y ) const { if( l.has2 ) Ax_t<D>( l, v, y ); else Ax_t<6>( l, v, y ); }
    // one fused Chebyshev step: x += d, r -= A d, d' = c1 d + c2 D^-1 r ( last: no d' ), rows in one pass ( d' in another buffer:
    // A d reads the neighbouring rows of d ), the lists after ( their part of A d applied to r, then d' fixed up on their points )
    template<int ND,class WT> void cheb_step_t( Level &l, const WT *wpl, T c1, T c2, bool last ) {
        const size_t ps = l.ps;
        T *x = l.x.data(), *r = l.res.data(), *d = l.cd.data(), *d2 = l.cd2.data(); const T *id = l.idiag.data(), *dg = l.diag.data();
        const bool ext = ! l.ext.empty();
        for( int rr = 0; rr < l.R; ++rr ) {
            const int *o = l.off[ rr & 1 ];
            const size_t base = l.pid( rr, 0 );
            const WT *w = wpl + base;
            T *xr = x + base, *rrow = r + base, *d2r = d2 + base; const T *dr = d + base, *idr = id + base, *dgr = dg + base;
            for( int c = 0; c < l.W; ++c ) {
                T ad = dgr[ c ] * dr[ c ];
                #pragma clang loop unroll( full )
                for( int k = 0; k < ND; ++k ) ad -= w[ k * ps + c ] * dr[ c + o[ k ] ];
                xr[ c ] += dr[ c ];
                const T rn = rrow[ c ] - ad;
                rrow[ c ] = rn;
                if( ! last ) d2r[ c ] = c1 * dr[ c ] + c2 * idr[ c ] * rn;
            }
        }
        if( ext ) {
            for( auto &e : l.ext ) { r[ e.i ] += e.w * d[ e.j ]; r[ e.j ] += e.w * d[ e.i ]; }
            if( ! last ) for( auto &e : l.ext ) { d2[ e.i ] = c1 * d[ e.i ] + c2 * id[ e.i ] * r[ e.i ]; d2[ e.j ] = c1 * d[ e.j ] + c2 * id[ e.j ] * r[ e.j ]; }
        }
        if( ! last ) std::swap( l.cd, l.cd2 );
    }
    void cheb_step( Level &l, T c1, T c2, bool last ) {
        const bool fl = wfloat && ! l.wf.empty();
        if( l.has2 ) { if( fl ) cheb_step_t<D>( l, l.wf.data(), c1, c2, last ); else cheb_step_t<D>( l, l.w.data(), c1, c2, last ); }
        else { if( fl ) cheb_step_t<6>( l, l.wf.data(), c1, c2, last ); else cheb_step_t<6>( l, l.w.data(), c1, c2, last ); }
    }
    /// lmax of D^-1 A by power iterations ( x 1.1 )
    void estimate_lmax( Level &l ) {
        std::vector<T> v( l.ps, 0 ), y( l.ps, 0 );
        unsigned sd = 7;
        for( size_t q = 0; q < l.ps; ++q ) if( l.valid[ q ] ) { sd = sd * 1103515245u + 12345u; v[ q ] = T( ( sd >> 8 ) & 0xffff ) / 65536 - T( 0.5 ); }
        T lam = 1;
        for( int it = 0; it < 6; ++it ) {
            Ax( l, v.data(), y.data() );
            T n2 = 0, ny = 0;
            for( size_t q = 0; q < l.ps; ++q ) { y[ q ] *= l.idiag[ q ]; n2 += v[ q ] * v[ q ]; ny += y[ q ] * y[ q ]; }
            lam = std::sqrt( ny / std::max( n2, T( 1e-300 ) ) );
            const T sc = 1 / std::sqrt( std::max( ny, T( 1e-300 ) ) );
            for( size_t q = 0; q < l.ps; ++q ) v[ q ] = y[ q ] * sc;
        }
        l.lmax = T( 1.15 ) * lam;                  // ( 6 power iterations: a lower estimate, the margin covers it )
    }
    /// Chebyshev smoothing ( degree `cheb` ) of A x = b on [ lmax / alpha, lmax ] for D^-1 A; `zero`: x starts at 0. On exit l.res =
    /// b - A x ( the residual the restriction needs, one A d more )
    void chebyshev( Level &l, bool zero ) {
        const T b_ = l.lmax, a_ = l.lmax / cheb_alpha, th = ( b_ + a_ ) / 2, de = ( b_ - a_ ) / 2;
        l.cd.resize( l.ps ); l.ct.resize( l.ps );
        l.cd.resize( l.ps );
        if( zero ) { std::fill( l.x.begin(), l.x.end(), T( 0 ) ); std::copy( l.b.begin(), l.b.end(), l.res.begin() ); }
        else { Ax( l, l.x.data(), l.ct.data() ); for( size_t q = 0; q < l.ps; ++q ) l.res[ q ] = l.b[ q ] - l.ct[ q ]; }
        T *x = l.x.data(), *r = l.res.data(), *d = l.cd.data(), *t = l.ct.data(); const T *id = l.idiag.data();
        const size_t ps = l.ps;
        for( size_t q = 0; q < ps; ++q ) d[ q ] = id[ q ] * r[ q ] / th;
        T rho_old = de / th;
        if( fused ) {
            l.cd2.resize( l.ps );
            for( int k = 0; k < cheb; ++k ) {
                const bool last = k + 1 == cheb;
                const T rho = last ? T( 0 ) : 1 / ( 2 * th / de - rho_old );
                cheb_step( l, rho * rho_old, last ? T( 0 ) : 2 * rho / de, last );
                rho_old = rho;
            }
            return;
        }
        for( int k = 0; k < cheb; ++k ) {
            for( size_t q = 0; q < ps; ++q ) x[ q ] += d[ q ];
            Ax( l, d, t );
            for( size_t q = 0; q < ps; ++q ) r[ q ] -= t[ q ];
            if( k + 1 == cheb ) break;
            const T rho = 1 / ( 2 * th / de - rho_old ), c1 = rho * rho_old, c2 = 2 * rho / de;
            for( size_t q = 0; q < ps; ++q ) d[ q ] = c1 * d[ q ] + c2 * id[ q ] * r[ q ];
            rho_old = rho;
        }
    }
    void vcycle( size_t li ) {
        Level &f = L[ li ];
        double t0 = now_s();
        auto lap = [&]( int k ) { const double t = now_s(); tv[ k ] += t - t0; t0 = t; };
        if( li + 1 == L.size() ) {
            std::vector<T> y( nc );
            T mb = 0; for( int a = 0; a < nc; ++a ) { y[ a ] = f.b[ cidx[ a ] ]; mb += y[ a ]; } mb /= std::max( nc, 1 );
            for( T &v : y ) v -= mb;
            for( int i = 0; i < nc; ++i ) { T s = y[ i ]; for( int k = 0; k < i; ++k ) s -= chol[ size_t( i ) * nc + k ] * y[ k ]; y[ i ] = s / chol[ size_t( i ) * nc + i ]; }
            for( int i = nc - 1; i >= 0; --i ) { T s = y[ i ]; for( int k = i + 1; k < nc; ++k ) s -= chol[ size_t( k ) * nc + i ] * y[ k ]; y[ i ] = s / chol[ size_t( i ) * nc + i ]; }
            std::fill( f.x.begin(), f.x.end(), T( 0 ) );
            for( int a = 0; a < nc; ++a ) f.x[ cidx[ a ] ] = y[ a ];
            lap( 4 );
            return;
        }
        if( cheb ) { chebyshev( f, true ); lap( 0 ); }       // ( leaves the residual in f.res )
        else {
            std::fill( f.x.begin(), f.x.end(), T( 0 ) );
            for( int s = 0; s < nu; ++s ) for( int k = 0; k < 3; ++k ) smooth( f, k, false );
            lap( 0 );
            residual( f );
            lap( 1 );
        }
        Level &c = L[ li + 1 ];
        if( ! far_parent ) {                        // ( the gather: coarse rows, fine rows 2 r' at stride 2 )
            const T *res = f.res.data(), *rw = f.rw.data();
            const int *o = f.off[ 0 ];
            for( int r = 0; r < c.R; ++r ) {
                const size_t cb = c.pid( r, 0 ), fb = f.pid( 2 * r, r & 1 );
                T *cr = c.b.data() + cb;
                for( int cc = 0; cc < c.W; ++cc ) {
                    const size_t q = fb + 2 * size_t( cc ), a = cb + cc;
                    T s = rw[ 6 * c.ps + a ] * res[ q ];
                    for( int d = 0; d < 6; ++d ) s += rw[ d * c.ps + a ] * res[ q + o[ d ] ];
                    cr[ cc ] = s;
                }
            }
        } else {
            std::fill( c.b.begin(), c.b.end(), T( 0 ) );
            for( size_t t = 0; t < f.fi.size(); ++t ) { const T rv = f.res[ f.fi[ t ] ]; c.b[ f.p0[ t ] ] += f.w0[ t ] * rv; c.b[ f.p1[ t ] ] += f.w1[ t ] * rv; }
        }
        lap( 2 );
        vcycle( li + 1 );
        t0 = now_s();
        for( size_t t = 0; t < f.fi.size(); ++t ) f.x[ f.fi[ t ] ] += f.w0[ t ] * c.x[ f.p0[ t ] ] + f.w1[ t ] * c.x[ f.p1[ t ] ];
        lap( 3 );
        if( cheb ) chebyshev( f, false );
        else for( int s = 0; s < nu; ++s ) for( int k = 2; k >= 0; --k ) smooth( f, k, true );
        lap( 0 );
    }
    /// z = M^-1 r, r and z as rows of W at stride rs / zs
    void apply( const T *r, int rs, T *z, int zs ) {
        Level &f = L[ 0 ];
        for( int rr = 0; rr < f.R; ++rr ) std::copy( r + size_t( rr ) * rs, r + size_t( rr ) * rs + f.W, f.b.data() + f.pid( rr, 0 ) );
        vcycle( 0 );
        for( int rr = 0; rr < f.R; ++rr ) std::copy( f.x.data() + f.pid( rr, 0 ), f.x.data() + f.pid( rr, 0 ) + f.W, z + size_t( rr ) * zs );
    }
    void stats() const {
        for( size_t l = 0; l < L.size(); ++l ) std::printf( "      level %zu: %d x %d, ring 2 %s, ext %zu\n", l, L[ l ].R, L[ l ].W, L[ l ].has2 ? "yes" : "no", L[ l ].ext.size() );
    }
};

} // namespace hexgrid
