// STRUCTURED geometric multigrid for the hessian of a hexagonal lattice ( the preconditioner of CG ), no hash map, no CSR.
//
// Each level is again a hexagonal lattice in offset rows ( R_l x W_l, odd rows shifted right by half a spacing, some points
// missing at the ends of the rows: `valid` ): level l + 1 keeps the points of level l with even axial q and r, i.e. the even rows
// r = 2 r' and in them the columns c = 2 c' + ( r' & 1 ) -- the same convention one level up.
//
// Operator of a level: 6 weights per point toward its ring-1 neighbours ( planes, direction k = u_k ), the diagonal their sum
// ( + the faces past ring 1 at level 0, a list ). Interpolation ( linear on the coarse triangles ): a fine point is a coarse
// point or the middle of a coarse edge ( q odd: along u0, r odd: along u1, both: along u2 ), the mean of its 1 or 2 parents.
// Coarse operator by GALERKIN, edge by edge: a fine edge ( i, j, w ) gives w ( sum_a c_a y_a )^2 with c = P^T ( e_i - e_j ), a
// zero-sum vector, = - w sum_{a<b} c_a c_b ( y_a - y_b )^2: a coarse edge weight per pair. With ring-1 fine edges every pair is a
// coarse ring-1 pair; the farther pairs ( from the faces past ring 1 ) are LUMPED on a shortest hexagonal path, weight times its
// length ( ( sum d )^2 <= h sum d^2 ): constants and symmetry kept.
// Smoother: Gauss-Seidel in the 3 colours ( q - r ) mod 3 ( ring-1 neighbours never share a colour ), colours 0, 1, 2 before the
// coarse correction and 2, 1, 0 after ( symmetric V-cycle ). Coarsest level: dense Cholesky of A + 1 1^T s / n ( mean-zero space ).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <vector>

namespace hexgrid {

struct HexGMG {
    using T = double;
    static constexpr int uq_[ 6 ] = { 1, 0, -1, -1, 0, 1 }, ur_[ 6 ] = { 0, 1, 1, 0, -1, -1 };
    static int dcol_( int dq, int dr, int SH ) { const int sh2 = ( SH + ( dr & 1 ) ) & 1; return dq + ( dr + SH - sh2 ) / 2; }
    static int axq( int r, int c ) { return c - ( r - ( r & 1 ) ) / 2; }

    struct Extra { int i, j; T w; };
    static double now_s() { return std::chrono::duration<double>( std::chrono::steady_clock::now().time_since_epoch() ).count(); }
    struct Level {
        int R = 0, W = 0, n = 0;
        std::vector<unsigned char> valid, col;     // ( colour ( q - r ) mod 3 )
        std::vector<int> nb;                       // 6 x n: the ring-1 neighbours ( -1: none )
        std::vector<T> w, diag, idiag;             // 6 x n weights, diagonal
        std::vector<Extra> ext;                    // ( level 0: the faces past ring 1, each applied to both ends )
        std::vector<int> p0, p1; std::vector<T> w0, w1;   // the parents in the next level
        std::vector<T> x, b, res;
        std::vector<int> xbeg, xnb; std::vector<T> xw;
        std::vector<int> by_col[ 3 ];              // the valid points of each colour
        int idx( int r, int c ) const { return r * W + c; }
    };
    std::vector<Level> L;
    int n_coarsest = 200, nu = 1;
    std::vector<T> chol; int nc = 0;
    T min_pivot = 0;
    void debug() const {
        for( size_t l = 0; l < L.size(); ++l ) {
            const Level &f = L[ l ];
            int nz = 0, nv = 0; T dmin = 1e300, dmax = 0, wneg = 0;
            for( int i = 0; i < f.n; ++i ) if( f.valid[ i ] ) { ++nv; nz += !( f.diag[ i ] > 0 ); dmin = std::min( dmin, f.diag[ i ] ); dmax = std::max( dmax, f.diag[ i ] ); for( int k = 0; k < 6; ++k ) wneg = std::min( wneg, f.w[ 6 * size_t( i ) + k ] ); }
            std::printf( "      level %zu: %d points, diag <= 0: %d, diag %.2e .. %.2e, most negative weight %.2e, ext %zu\n", l, nv, nz, dmin, dmax, wneg, f.ext.size() );
        }
        std::printf( "      coarsest %d points, min pivot^2 %.3e\n", nc, min_pivot );
    }

    void geometry( Level &l ) {
        l.n = l.R * l.W;
        l.nb.assign( 6 * size_t( l.n ), -1 ); l.col.assign( l.n, 0 );
        for( int k = 0; k < 3; ++k ) l.by_col[ k ].clear();
        for( int r = 0; r < l.R; ++r ) for( int c = 0; c < l.W; ++c ) {
            const int i = l.idx( r, c );
            l.col[ i ] = ( unsigned char )( ( ( axq( r, c ) - r ) % 3 + 3 ) % 3 );
            if( ! l.valid[ i ] ) continue;
            l.by_col[ l.col[ i ] ].push_back( i );
            for( int k = 0; k < 6; ++k ) {
                const int rr = r + ur_[ k ], cc = c + dcol_( uq_[ k ], ur_[ k ], r & 1 );
                if( rr >= 0 && rr < l.R && cc >= 0 && cc < l.W && l.valid[ l.idx( rr, cc ) ] ) l.nb[ 6 * size_t( i ) + k ] = l.idx( rr, cc );
            }
        }
        l.x.assign( l.n, 0 ); l.b.assign( l.n, 0 ); l.res.assign( l.n, 0 );
    }

    /// the levels and the interpolations ( once per lattice )
    void setup_levels( int R, int W ) {
        L.clear();
        L.emplace_back();
        L[ 0 ].R = R; L[ 0 ].W = W; L[ 0 ].valid.assign( size_t( R ) * W, 1 );
        geometry( L[ 0 ] );
        while( true ) {
            Level &f = L.back();
            int nvalid = 0; for( auto v : f.valid ) nvalid += v;
            if( nvalid <= n_coarsest ) break;
            Level c;
            c.R = ( f.R + 1 ) / 2; c.W = ( f.W + 1 ) / 2;
            c.valid.assign( size_t( c.R ) * c.W, 0 );
            for( int r = 0; r < c.R; ++r ) for( int cc = 0; cc < c.W; ++cc ) {
                const int fr = 2 * r, fc = 2 * cc + ( r & 1 );
                if( fr < f.R && fc < f.W && f.valid[ f.idx( fr, fc ) ] ) c.valid[ c.idx( r, cc ) ] = 1;
            }
            geometry( c );
            // parents of each fine point, in axial coordinates of the fine level
            auto coarse_of = [&]( int q, int r ) -> int { // the coarse point at fine axial ( q, r ) ( both even ), -1 if none
                if( ( q & 1 ) || ( r & 1 ) ) return -1;
                const int rc = r / 2, qc = q / 2, cc = qc + ( rc - ( rc & 1 ) ) / 2;
                if( rc < 0 || rc >= c.R || cc < 0 || cc >= c.W || ! c.valid[ c.idx( rc, cc ) ] ) return -1;
                return c.idx( rc, cc );
            };
            f.p0.assign( f.n, -1 ); f.p1.assign( f.n, -1 ); f.w0.assign( f.n, 0 ); f.w1.assign( f.n, 0 );
            for( int r = 0; r < f.R; ++r ) for( int cc = 0; cc < f.W; ++cc ) {
                const int i = f.idx( r, cc );
                if( ! f.valid[ i ] ) continue;
                const int q = axq( r, cc ), oq = q & 1, orr = r & 1;
                int a = -1, b = -1;
                if( ! oq && ! orr ) a = coarse_of( q, r );
                else if( oq && ! orr ) { a = coarse_of( q - 1, r ); b = coarse_of( q + 1, r ); }
                else if( ! oq && orr ) { a = coarse_of( q, r - 1 ); b = coarse_of( q, r + 1 ); }
                else { a = coarse_of( q - 1, r + 1 ); b = coarse_of( q + 1, r - 1 ); }
                if( a < 0 ) std::swap( a, b );
                if( a < 0 ) { // ( no parent along the axis, a corner: the nearest coarse point )
                    int best = 1 << 30;
                    for( int dq = -2; dq <= 2; ++dq ) for( int dr = -2; dr <= 2; ++dr ) {
                        const int k = coarse_of( q + dq, r + dr );
                        const int d = dq * dq + dr * dr + dq * dr;
                        if( k >= 0 && d < best ) { best = d; a = k; }
                    }
                }
                f.p0[ i ] = a; f.p1[ i ] = b;
                if( b >= 0 ) { f.w0[ i ] = 0.5; f.w1[ i ] = 0.5; } else f.w0[ i ] = 1;
            }
            L.push_back( std::move( c ) );
        }
    }

    // a coarse edge ( a, b, v ): a ring-1 pair into the 6 planes, a farther one into the level's list ( the Galerkin operator kept
    // EXACT: lumping on paths broke its definiteness when a path left the lattice )
    void add_edge( Level &l, int a, int b, T v ) {
        if( a == b || v == 0 ) return;
        const int ra = a / l.W, ca = a % l.W;
        for( int k = 0; k < 6; ++k ) if( l.nb[ 6 * size_t( a ) + k ] == b ) { l.w[ 6 * size_t( a ) + k ] += v; l.w[ 6 * size_t( b ) + ( k + 3 ) % 6 ] += v; return; }
        ( void )ra; ( void )ca;
        l.ext.push_back( { a, b, v } );
    }

    /// the operators from the fine weights ( W6: 6 per point, symmetric pairs; ext: ( i, j, w ) applied to both ends )
    double ts[ 3 ] = {};                          // ( timing: Galerkin, coarsest Cholesky, fast arrays )
    void setup_ops( const std::vector<T> &W6, const std::vector<Extra> &ext ) {
        double t0 = now_s();
        Level &f0 = L[ 0 ];
        f0.w = W6; f0.ext = ext;
        for( size_t l = 0; l < L.size(); ++l ) {
            Level &f = L[ l ];
            ext_adjacency( f );
            f.diag.assign( f.n, 0 );
            for( int i = 0; i < f.n; ++i ) for( int k = 0; k < 6; ++k ) f.diag[ i ] += f.w[ 6 * size_t( i ) + k ];
            for( const Extra &e : f.ext ) { f.diag[ e.i ] += e.w; f.diag[ e.j ] += e.w; }
            f.idiag.resize( f.n );
            for( int i = 0; i < f.n; ++i ) f.idiag[ i ] = f.diag[ i ] > 0 ? 1 / f.diag[ i ] : 0;
            if( l + 1 == L.size() ) break;
            Level &c = L[ l + 1 ];
            c.w.assign( 6 * size_t( c.n ), 0 ); c.ext.clear();
            auto edge = [&]( int i, int j, T w ) {
                int ids[ 4 ]; T cs[ 4 ]; int m = 0;
                auto put = [&]( int a, T v ) { if( a < 0 || v == 0 ) return; for( int t = 0; t < m; ++t ) if( ids[ t ] == a ) { cs[ t ] += v; return; } ids[ m ] = a; cs[ m ] = v; ++m; };
                put( f.p0[ i ], f.w0[ i ] ); put( f.p1[ i ], f.w1[ i ] ); put( f.p0[ j ], -f.w0[ j ] ); put( f.p1[ j ], -f.w1[ j ] );
                for( int s = 0; s < m; ++s ) for( int t = s + 1; t < m; ++t ) add_edge( c, ids[ s ], ids[ t ], -w * cs[ s ] * cs[ t ] );
            };
            for( int i = 0; i < f.n; ++i ) for( int k = 0; k < 3; ++k ) { const int j = f.nb[ 6 * size_t( i ) + k ]; if( j >= 0 ) edge( i, j, f.w[ 6 * size_t( i ) + k ] ); }
            static const bool no_ext_coarse = std::getenv( "GMG_NOEXT" ) != nullptr;   // ( experiment: the coarse levels without the faces past ring 1 )
            if( ! no_ext_coarse ) for( const Extra &e : f.ext ) edge( e.i, e.j, e.w );
        }
        ts[ 0 ] += now_s() - t0; t0 = now_s();
        // coarsest: dense Cholesky of A + s 1 1^T / n ( the mean-zero space )
        const Level &c = L.back();
        nc = c.n;
        min_pivot = 1e300;
        chol.assign( size_t( nc ) * nc, 0 );
        T sc = 0; for( T d : c.diag ) sc = std::max( sc, d );
        for( int i = 0; i < nc; ++i ) {
            if( ! c.valid[ i ] ) { chol[ size_t( i ) * nc + i ] = 1; continue; }
            chol[ size_t( i ) * nc + i ] += c.diag[ i ];
            for( int j = 0; j < nc; ++j ) if( c.valid[ j ] ) chol[ size_t( i ) * nc + j ] += sc / nc;
            for( int k = 0; k < 6; ++k ) { const int j = c.nb[ 6 * size_t( i ) + k ]; if( j >= 0 ) chol[ size_t( i ) * nc + j ] -= c.w[ 6 * size_t( i ) + k ]; }
        }
        for( const Extra &e : c.ext ) { chol[ size_t( e.i ) * nc + e.j ] -= e.w; chol[ size_t( e.j ) * nc + e.i ] -= e.w; }
        for( int j = 0; j < nc; ++j ) {
            T d = chol[ size_t( j ) * nc + j ];
            for( int k = 0; k < j; ++k ) d -= chol[ size_t( j ) * nc + k ] * chol[ size_t( j ) * nc + k ];
            if( d < min_pivot ) min_pivot = d;
            d = std::sqrt( std::max( d, T( 1e-300 ) ) );
            chol[ size_t( j ) * nc + j ] = d;
            for( int i = j + 1; i < nc; ++i ) {
                T s = chol[ size_t( i ) * nc + j ];
                for( int k = 0; k < j; ++k ) s -= chol[ size_t( i ) * nc + k ] * chol[ size_t( j ) * nc + k ];
                chol[ size_t( i ) * nc + j ] = s / d;
            }
        }
        ts[ 1 ] += now_s() - t0; t0 = now_s();
        if( fast ) setup_fast();
        ts[ 2 ] += now_s() - t0;
    }

    // ( A x )_i
    T row( const Level &l, int i, const T *x ) const {
        T s = l.diag[ i ] * x[ i ];
        for( int k = 0; k < 6; ++k ) { const int j = l.nb[ 6 * size_t( i ) + k ]; if( j >= 0 ) s -= l.w[ 6 * size_t( i ) + k ] * x[ j ]; }
        return s;
    }
    void residual( Level &l ) {
        for( int i = 0; i < l.n; ++i ) l.res[ i ] = l.valid[ i ] ? l.b[ i ] - row( l, i, l.x.data() ) : 0;
        for( const Extra &e : l.ext ) { l.res[ e.i ] += e.w * l.x[ e.j ]; l.res[ e.j ] += e.w * l.x[ e.i ]; }   // ( the off-diagonal part only: `diag` already holds w )
    }
    // one colour of Gauss-Seidel: x_i = ( b_i + sum_j w_ij x_j ) / diag_i, the faces past ring 1 with the current values too ( they
    // may join two points of a colour: the order matters, reversed after the coarse correction: symmetric Gauss-Seidel )
    static void ext_adjacency( Level &l ) {           // ( the ext faces of each point, both ends )
        l.xbeg.assign( l.n + 1, 0 );
        for( const Extra &e : l.ext ) { ++l.xbeg[ e.i + 1 ]; ++l.xbeg[ e.j + 1 ]; }
        for( int i = 0; i < l.n; ++i ) l.xbeg[ i + 1 ] += l.xbeg[ i ];
        l.xnb.resize( l.xbeg[ l.n ] ); l.xw.resize( l.xbeg[ l.n ] );
        std::vector<int> fill( l.xbeg.begin(), l.xbeg.end() - 1 );
        for( const Extra &e : l.ext ) { l.xnb[ fill[ e.i ] ] = e.j; l.xw[ fill[ e.i ]++ ] = e.w; l.xnb[ fill[ e.j ] ] = e.i; l.xw[ fill[ e.j ]++ ] = e.w; }
    }
    void smooth_colour( Level &l, int k, bool reverse ) {
        const bool has_ext = ! l.ext.empty();
        const auto &lst = l.by_col[ k ];
        const int m = int( lst.size() );
        for( int t = 0; t < m; ++t ) {
            const int i = lst[ reverse ? m - 1 - t : t ];
            T s = l.b[ i ];
            for( int d = 0; d < 6; ++d ) { const int j = l.nb[ 6 * size_t( i ) + d ]; if( j >= 0 ) s += l.w[ 6 * size_t( i ) + d ] * l.x[ j ]; }
            if( has_ext ) for( int q = l.xbeg[ i ]; q < l.xbeg[ i + 1 ]; ++q ) s += l.xw[ q ] * l.x[ l.xnb[ q ] ];
            l.x[ i ] = l.idiag[ i ] * s;
        }
    }

    void vcycle( size_t li ) {
        Level &f = L[ li ];
        if( li + 1 == L.size() ) {
            std::vector<T> y( f.b );
            T mb = 0; int nv = 0; for( int i = 0; i < nc; ++i ) if( f.valid[ i ] ) { mb += y[ i ]; ++nv; }
            mb /= std::max( nv, 1 );
            for( int i = 0; i < nc; ++i ) y[ i ] = f.valid[ i ] ? y[ i ] - mb : 0;
            for( int i = 0; i < nc; ++i ) { T s = y[ i ]; for( int k = 0; k < i; ++k ) s -= chol[ size_t( i ) * nc + k ] * y[ k ]; y[ i ] = s / chol[ size_t( i ) * nc + i ]; }
            for( int i = nc - 1; i >= 0; --i ) { T s = y[ i ]; for( int k = i + 1; k < nc; ++k ) s -= chol[ size_t( k ) * nc + i ] * y[ k ]; y[ i ] = s / chol[ size_t( i ) * nc + i ]; }
            f.x = y;
            return;
        }
        std::fill( f.x.begin(), f.x.end(), T( 0 ) );
        for( int s = 0; s < nu; ++s ) for( int k = 0; k < 3; ++k ) smooth_colour( f, k, false );
        residual( f );
        Level &c = L[ li + 1 ];
        std::fill( c.b.begin(), c.b.end(), T( 0 ) );
        for( int i = 0; i < f.n; ++i ) if( f.valid[ i ] ) { c.b[ f.p0[ i ] ] += f.w0[ i ] * f.res[ i ]; if( f.p1[ i ] >= 0 ) c.b[ f.p1[ i ] ] += f.w1[ i ] * f.res[ i ]; }
        vcycle( li + 1 );
        for( int i = 0; i < f.n; ++i ) if( f.valid[ i ] ) f.x[ i ] += f.w0[ i ] * c.x[ f.p0[ i ] ] + ( f.p1[ i ] >= 0 ? f.w1[ i ] * c.x[ f.p1[ i ] ] : T( 0 ) );
        for( int s = 0; s < nu; ++s ) for( int k = 2; k >= 0; --k ) smooth_colour( f, k, true );
    }

    // ---------------------------------------------------------------- the fast path ( same V-cycle, same iterates up to rounding
    // in the order of the faces past ring 1 ): per level arrays padded by one row / column of zeros ( P = W + 2 ), the 6 weights as
    // planes read at fixed offsets per row parity, the colours swept row by row with a stride of 3, the faces past ring 1 merged
    // and read with the values at the start of each colour ( a colour step x += Pi_k D^-1 ( b - A x ) is self-adjoint in the A inner
    // product: the V-cycle stays symmetric )
    struct FL {
        int R = 0, W = 0, P = 0; size_t ps = 0;
        std::vector<T> w, idiag, diag, x, b, res, extn;   // w: 6 planes of ps
        int off[ 2 ][ 6 ];
        std::vector<int> cst[ 3 ];                    // per colour, the first column of each row
        std::vector<size_t> ei, ej; std::vector<T> ew;    // the faces past ring 1, merged, padded indices
        std::vector<int> xb; std::vector<size_t> xn; std::vector<T> xw;   // ( the same per point: CSR over the padded indices )
        std::vector<size_t> fi, p0, p1; std::vector<T> w0, w1;   // the valid points and their parents ( padded in the next level )
    };
    std::vector<FL> F;
    bool fast = true;
    size_t pidx( const FL &f, int i ) const { return size_t( i / f.W + 1 ) * f.P + i % f.W + 1; }
    void setup_fast() {
        F.assign( L.size(), {} );
        for( size_t l = 0; l < L.size(); ++l ) {
            const Level &a = L[ l ]; FL &f = F[ l ];
            f.R = a.R; f.W = a.W; f.P = a.W + 2; f.ps = size_t( a.R + 2 ) * f.P;
            f.w.assign( 6 * f.ps, 0 ); f.idiag.assign( f.ps, 0 ); f.diag.assign( f.ps, 0 );
            f.x.assign( f.ps, 0 ); f.b.assign( f.ps, 0 ); f.res.assign( f.ps, 0 ); f.extn.assign( f.ps, 0 );
            for( int sh = 0; sh < 2; ++sh ) for( int k = 0; k < 6; ++k ) f.off[ sh ][ k ] = ur_[ k ] * f.P + dcol_( uq_[ k ], ur_[ k ], sh );
            for( int i = 0; i < a.n; ++i ) {
                if( ! a.valid[ i ] ) continue;
                const size_t q = pidx( f, i );
                for( int k = 0; k < 6; ++k ) f.w[ k * f.ps + q ] = a.nb[ 6 * size_t( i ) + k ] >= 0 ? a.w[ 6 * size_t( i ) + k ] : T( 0 );
                f.idiag[ q ] = a.idiag[ i ]; f.diag[ q ] = a.diag[ i ];
            }
            for( int k = 0; k < 3; ++k ) {
                f.cst[ k ].assign( a.R, 0 );
                for( int r = 0; r < a.R; ++r ) f.cst[ k ][ r ] = ( ( k + r + ( r - ( r & 1 ) ) / 2 ) % 3 + 3 ) % 3;   // ( c - floor( r / 2 ) - r = k mod 3 )
            }
            // the faces past ring 1, merged
            std::vector<std::pair<std::pair<int,int>,T>> e;
            for( const Extra &x : a.ext ) e.push_back( { { std::min( x.i, x.j ), std::max( x.i, x.j ) }, x.w } );
            std::sort( e.begin(), e.end(), []( auto &u, auto &v ) { return u.first < v.first; } );
            for( size_t k = 0; k < e.size(); ++k ) {
                if( k && e[ k ].first == e[ k - 1 ].first ) { f.ew.back() += e[ k ].second; continue; }
                f.ei.push_back( pidx( f, e[ k ].first.first ) ); f.ej.push_back( pidx( f, e[ k ].first.second ) ); f.ew.push_back( e[ k ].second );
            }
            f.xb.assign( f.ps + 1, 0 );
            for( size_t k = 0; k < f.ew.size(); ++k ) { ++f.xb[ f.ei[ k ] + 1 ]; ++f.xb[ f.ej[ k ] + 1 ]; }
            for( size_t q = 0; q < f.ps; ++q ) f.xb[ q + 1 ] += f.xb[ q ];
            f.xn.resize( f.xb[ f.ps ] ); f.xw.resize( f.xb[ f.ps ] );
            { std::vector<int> fill( f.xb.begin(), f.xb.end() - 1 );
              for( size_t k = 0; k < f.ew.size(); ++k ) { f.xn[ fill[ f.ei[ k ] ] ] = f.ej[ k ]; f.xw[ fill[ f.ei[ k ] ]++ ] = f.ew[ k ]; f.xn[ fill[ f.ej[ k ] ] ] = f.ei[ k ]; f.xw[ fill[ f.ej[ k ] ]++ ] = f.ew[ k ]; } }
        }
        for( size_t l = 0; l + 1 < L.size(); ++l ) {
            const Level &a = L[ l ]; FL &f = F[ l ]; const FL &c = F[ l + 1 ];
            for( int i = 0; i < a.n; ++i ) {
                if( ! a.valid[ i ] ) continue;
                f.fi.push_back( pidx( f, i ) );
                f.p0.push_back( pidx( c, a.p0[ i ] ) ); f.w0.push_back( a.w0[ i ] );
                f.p1.push_back( a.p1[ i ] >= 0 ? pidx( c, a.p1[ i ] ) : pidx( c, a.p0[ i ] ) ); f.w1.push_back( a.p1[ i ] >= 0 ? a.w1[ i ] : T( 0 ) );
            }
        }
    }
    void ext_sums( FL &f ) {
        std::fill( f.extn.begin(), f.extn.end(), T( 0 ) );
        for( size_t k = 0; k < f.ew.size(); ++k ) { f.extn[ f.ei[ k ] ] += f.ew[ k ] * f.x[ f.ej[ k ] ]; f.extn[ f.ej[ k ] ] += f.ew[ k ] * f.x[ f.ei[ k ] ]; }
    }
    // ( Gauss-Seidel on the faces past ring 1 too: their per-point lists read with the current values; the colour order reversed
    // after the coarse correction keeps the V-cycle symmetric: `rev` reverses the rows and the columns of the colour )
    void smooth_fast( FL &f, int k, bool rev = false ) {
        const bool ext = ! f.ew.empty();
        const size_t ps = f.ps;
        for( int t = 0; t < f.R; ++t ) {
            const int r = rev ? f.R - 1 - t : t;
            const int *o = f.off[ r & 1 ];
            const size_t base = size_t( r + 1 ) * f.P + 1;
            T *x = f.x.data() + base; const T *b = f.b.data() + base, *id = f.idiag.data() + base;
            const T *w0 = f.w.data() + base, *w1 = w0 + ps, *w2 = w1 + ps, *w3 = w2 + ps, *w4 = w3 + ps, *w5 = w4 + ps;
            const int c0 = f.cst[ k ][ r ];
            const int cl = c0 + ( f.W - 1 - c0 ) / 3 * 3;
            for( int u = 0; c0 + 3 * u < f.W; ++u ) {
                const int c = rev ? cl - 3 * u : c0 + 3 * u;
                T s = b[ c ] + w0[ c ] * x[ c + o[ 0 ] ] + w1[ c ] * x[ c + o[ 1 ] ] + w2[ c ] * x[ c + o[ 2 ] ] + w3[ c ] * x[ c + o[ 3 ] ] + w4[ c ] * x[ c + o[ 4 ] ] + w5[ c ] * x[ c + o[ 5 ] ];
                if( ext ) { const size_t q = base + c; for( int e = f.xb[ q ]; e < f.xb[ q + 1 ]; ++e ) s += f.xw[ e ] * f.x[ f.xn[ e ] ]; }
                x[ c ] = id[ c ] * s;
            }
        }
    }
    void residual_fast( FL &f ) {
        const size_t ps = f.ps;
        for( int r = 0; r < f.R; ++r ) {
            const int *o = f.off[ r & 1 ];
            const size_t base = size_t( r + 1 ) * f.P + 1;
            const T *x = f.x.data() + base, *b = f.b.data() + base, *d = f.diag.data() + base; T *res = f.res.data() + base;
            const T *w0 = f.w.data() + base, *w1 = w0 + ps, *w2 = w1 + ps, *w3 = w2 + ps, *w4 = w3 + ps, *w5 = w4 + ps;
            for( int c = 0; c < f.W; ++c )
                res[ c ] = b[ c ] - d[ c ] * x[ c ] + w0[ c ] * x[ c + o[ 0 ] ] + w1[ c ] * x[ c + o[ 1 ] ] + w2[ c ] * x[ c + o[ 2 ] ] + w3[ c ] * x[ c + o[ 3 ] ] + w4[ c ] * x[ c + o[ 4 ] ] + w5[ c ] * x[ c + o[ 5 ] ];
        }
        for( size_t k = 0; k < f.ew.size(); ++k ) { f.res[ f.ei[ k ] ] += f.ew[ k ] * f.x[ f.ej[ k ] ]; f.res[ f.ej[ k ] ] += f.ew[ k ] * f.x[ f.ei[ k ] ]; }
    }
    double tv[ 5 ] = {};                          // ( timing: smoothing, residual, restriction, prolongation, coarsest )
    void vcycle_fast( size_t li ) {
        FL &f = F[ li ];
        double t0 = now_s();
        auto lap = [&]( int k ) { const double t = now_s(); tv[ k ] += t - t0; t0 = t; };
        if( li + 1 == L.size() ) {                      // ( the coarsest: the dense path of the unpadded level )
            Level &a = L[ li ];
            for( int i = 0; i < a.n; ++i ) a.b[ i ] = a.valid[ i ] ? f.b[ pidx( f, i ) ] : T( 0 );
            vcycle( li );
            std::fill( f.x.begin(), f.x.end(), T( 0 ) );
            for( int i = 0; i < a.n; ++i ) if( a.valid[ i ] ) f.x[ pidx( f, i ) ] = a.x[ i ];
            lap( 4 );
            return;
        }
        std::fill( f.x.begin(), f.x.end(), T( 0 ) );
        for( int s = 0; s < nu; ++s ) for( int k = 0; k < 3; ++k ) smooth_fast( f, k );
        lap( 0 );
        residual_fast( f );
        lap( 1 );
        FL &c = F[ li + 1 ];
        std::fill( c.b.begin(), c.b.end(), T( 0 ) );
        for( size_t t = 0; t < f.fi.size(); ++t ) { const T rv = f.res[ f.fi[ t ] ]; c.b[ f.p0[ t ] ] += f.w0[ t ] * rv; c.b[ f.p1[ t ] ] += f.w1[ t ] * rv; }
        lap( 2 );
        vcycle_fast( li + 1 );
        t0 = now_s();
        for( size_t t = 0; t < f.fi.size(); ++t ) f.x[ f.fi[ t ] ] += f.w0[ t ] * c.x[ f.p0[ t ] ] + f.w1[ t ] * c.x[ f.p1[ t ] ];
        lap( 3 );
        for( int s = 0; s < nu; ++s ) for( int k = 2; k >= 0; --k ) smooth_fast( f, k, true );
        lap( 0 );
    }
    /// z = M^-1 r, r and z as rows of W at stride rs / zs
    void apply_fast( const T *r, int rs, T *z, int zs ) {
        FL &f = F[ 0 ];
        for( int rr = 0; rr < f.R; ++rr ) std::copy( r + size_t( rr ) * rs, r + size_t( rr ) * rs + f.W, f.b.data() + size_t( rr + 1 ) * f.P + 1 );
        vcycle_fast( 0 );
        for( int rr = 0; rr < f.R; ++rr ) std::copy( f.x.data() + size_t( rr + 1 ) * f.P + 1, f.x.data() + size_t( rr + 1 ) * f.P + 1 + f.W, z + size_t( rr ) * zs );
    }

    void apply( const T *r, T *z ) {
        if( fast ) { apply_fast( r, L[ 0 ].W, z, L[ 0 ].W ); return; }
        Level &f = L[ 0 ];
        std::copy( r, r + f.n, f.b.begin() );
        vcycle( 0 );
        std::copy( f.x.begin(), f.x.end(), z );
    }
};

} // namespace hexgrid
