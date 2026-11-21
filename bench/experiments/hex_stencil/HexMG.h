// Geometric multigrid for the hessian of a hexagonal lattice ( a V-cycle, used as the preconditioner of CG ).
//
// The points of a level are lattice points ( axial q, r ); the next level keeps those with q and r even ( the lattice of twice the
// spacing ): every other point is the middle of a coarse edge along u = ( 1, 0 ), ( 0, 1 ) or ( 1, -1 ), so the linear interpolation
// is the mean of two coarse points ( one when the other is out of the box ). Coarse operators by Galerkin ( P^T A P, CSR ), symmetric
// Gauss-Seidel ( forward before, backward after: the V-cycle is symmetric ), the coarsest level by dense Cholesky on the mean-zero space.
#pragma once

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

namespace hexgrid {

struct HexMG {
    using T = double;
    struct CSR {
        int n = 0;
        std::vector<int> beg, col;
        std::vector<T> val, diag;
        void mul( const T *x, T *y ) const { for( int i = 0; i < n; ++i ) { T s = 0; for( int k = beg[ i ]; k < beg[ i + 1 ]; ++k ) s += val[ k ] * x[ col[ k ] ]; y[ i ] = s; } }
    };
    struct Level {
        std::vector<int> q, r;                     // axial coordinates ( in the units of level 0 )
        CSR A;
        // prolongation from the next level: each point of this level = sum of w coarse points ( 1 or 2 )
        std::vector<int> pi0, pi1; std::vector<T> pw0, pw1;
        std::vector<T> x, b, res;
    };
    std::vector<Level> L;
    int n_coarsest = 300;
    std::vector<T> chol; int nc = 0;              // dense factor of the coarsest operator + 1 ( mean-zero )

    static long long key( int q, int r ) { return ( ( long long )( q + ( 1 << 30 ) ) << 32 ) | unsigned( r + ( 1 << 30 ) ); }

    /// the levels from the points of level 0 ( built once per lattice )
    void setup_levels( const std::vector<int> &q0, const std::vector<int> &r0 ) {
        L.clear();
        L.emplace_back();
        L[ 0 ].q = q0; L[ 0 ].r = r0;
        for( int s = 1; ; s *= 2 ) {
            Level &f = L.back();
            if( int( f.q.size() ) <= n_coarsest ) break;
            Level c;
            std::unordered_map<long long,int> idx;
            for( size_t i = 0; i < f.q.size(); ++i )
                if( ( ( f.q[ i ] / s ) % 2 == 0 ) && ( ( f.r[ i ] / s ) % 2 == 0 ) ) { idx[ key( f.q[ i ], f.r[ i ] ) ] = int( c.q.size() ); c.q.push_back( f.q[ i ] ); c.r.push_back( f.r[ i ] ); }
            const size_t nf = f.q.size();
            f.pi0.assign( nf, -1 ); f.pi1.assign( nf, -1 ); f.pw0.assign( nf, 0 ); f.pw1.assign( nf, 0 );
            auto find = [&]( int q, int r ) { auto it = idx.find( key( q, r ) ); return it == idx.end() ? -1 : it->second; };
            for( size_t i = 0; i < nf; ++i ) {
                const int q = f.q[ i ], r = f.r[ i ], oq = ( ( q / s ) % 2 + 2 ) % 2, orr = ( ( r / s ) % 2 + 2 ) % 2;
                int a = -1, b = -1;
                if( ! oq && ! orr ) a = find( q, r );
                else if( oq && ! orr ) { a = find( q - s, r ); b = find( q + s, r ); }
                else if( ! oq && orr ) { a = find( q, r - s ); b = find( q, r + s ); }
                else { a = find( q - s, r + s ); b = find( q + s, r - s ); }
                if( a < 0 ) std::swap( a, b );
                if( a < 0 ) { // ( a corner point with no coarse partner: the nearest coarse point )
                    T best = 1e300;
                    for( int dq = -2 * s; dq <= 2 * s; dq += s ) for( int dr = -2 * s; dr <= 2 * s; dr += s ) {
                        const int k = find( q + dq, r + dr );
                        const T d = T( dq ) * dq + T( dr ) * dr + T( dq ) * dr;
                        if( k >= 0 && d < best ) { best = d; a = k; }
                    }
                }
                f.pi0[ i ] = a; f.pi1[ i ] = b;
                if( b >= 0 ) { f.pw0[ i ] = 0.5; f.pw1[ i ] = 0.5; } else f.pw0[ i ] = 1;
            }
            if( c.q.empty() ) break;
            L.push_back( std::move( c ) );
        }
        for( Level &l : L ) { l.x.resize( l.q.size() ); l.b.resize( l.q.size() ); l.res.resize( l.q.size() ); }
    }

    /// the operators from the fine CSR ( built per Newton step )
    void setup_ops( const CSR &A0 ) {
        L[ 0 ].A = A0;
        for( size_t l = 0; l + 1 < L.size(); ++l ) {
            const Level &f = L[ l ];
            const int nc_ = int( L[ l + 1 ].q.size() );
            std::vector<std::unordered_map<int,T>> rows( nc_ );
            // A_c = P^T A_f P: for each fine row i, its coarse parents a ( weight wa ), each entry ( i, j ) to the parents of j
            for( int i = 0; i < f.A.n; ++i ) {
                const int pa[ 2 ] = { f.pi0[ i ], f.pi1[ i ] }; const T wa[ 2 ] = { f.pw0[ i ], f.pw1[ i ] };
                for( int k = f.A.beg[ i ]; k < f.A.beg[ i + 1 ]; ++k ) {
                    const int j = f.A.col[ k ]; const T v = f.A.val[ k ];
                    const int pb[ 2 ] = { f.pi0[ j ], f.pi1[ j ] }; const T wb[ 2 ] = { f.pw0[ j ], f.pw1[ j ] };
                    for( int u = 0; u < 2; ++u ) if( pa[ u ] >= 0 ) for( int w = 0; w < 2; ++w ) if( pb[ w ] >= 0 ) rows[ pa[ u ] ][ pb[ w ] ] += wa[ u ] * v * wb[ w ];
                }
            }
            CSR &c = L[ l + 1 ].A;
            c.n = nc_; c.beg.assign( nc_ + 1, 0 ); c.col.clear(); c.val.clear(); c.diag.assign( nc_, 0 );
            for( int i = 0; i < nc_; ++i ) {
                std::vector<std::pair<int,T>> e( rows[ i ].begin(), rows[ i ].end() );
                std::sort( e.begin(), e.end() );
                for( auto [ j, v ] : e ) { c.col.push_back( j ); c.val.push_back( v ); if( j == i ) c.diag[ i ] = v; }
                c.beg[ i + 1 ] = int( c.col.size() );
            }
        }
        // coarsest: dense ( A + 1 1^T / n: same solution on the mean-zero space ), Cholesky
        const CSR &c = L.back().A;
        nc = c.n;
        chol.assign( size_t( nc ) * nc, T( 0 ) );
        T sc = 0; for( T d : c.diag ) sc = std::max( sc, d );
        for( int i = 0; i < nc; ++i ) { for( int j = 0; j < nc; ++j ) chol[ size_t( i ) * nc + j ] = sc / nc; for( int k = c.beg[ i ]; k < c.beg[ i + 1 ]; ++k ) chol[ size_t( i ) * nc + c.col[ k ] ] += c.val[ k ]; }
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
    }

    int nu = 1;                                    // Gauss-Seidel sweeps before / after
    void gs( const CSR &A, const T *b, T *x, bool fwd ) const {
        for( int t = 0; t < A.n; ++t ) {
            const int i = fwd ? t : A.n - 1 - t;
            T s = b[ i ], d = 0;
            for( int k = A.beg[ i ]; k < A.beg[ i + 1 ]; ++k ) { if( A.col[ k ] == i ) d = A.val[ k ]; else s -= A.val[ k ] * x[ A.col[ k ] ]; }
            if( d > 0 ) x[ i ] = s / d;
        }
    }
    void vcycle( size_t l ) {
        Level &f = L[ l ];
        const int n = f.A.n;
        if( l + 1 == L.size() ) {
            std::vector<T> y( f.b );
            T mb = 0; for( T v : y ) mb += v; mb /= nc; for( T &v : y ) v -= mb;
            for( int i = 0; i < nc; ++i ) { T s = y[ i ]; for( int k = 0; k < i; ++k ) s -= chol[ size_t( i ) * nc + k ] * y[ k ]; y[ i ] = s / chol[ size_t( i ) * nc + i ]; }
            for( int i = nc - 1; i >= 0; --i ) { T s = y[ i ]; for( int k = i + 1; k < nc; ++k ) s -= chol[ size_t( k ) * nc + i ] * y[ k ]; y[ i ] = s / chol[ size_t( i ) * nc + i ]; }
            f.x = y;
            return;
        }
        std::fill( f.x.begin(), f.x.end(), T( 0 ) );
        for( int s = 0; s < nu; ++s ) gs( f.A, f.b.data(), f.x.data(), true );
        f.A.mul( f.x.data(), f.res.data() );
        for( int i = 0; i < n; ++i ) f.res[ i ] = f.b[ i ] - f.res[ i ];
        Level &c = L[ l + 1 ];
        std::fill( c.b.begin(), c.b.end(), T( 0 ) );
        for( int i = 0; i < n; ++i ) { c.b[ f.pi0[ i ] ] += f.pw0[ i ] * f.res[ i ]; if( f.pi1[ i ] >= 0 ) c.b[ f.pi1[ i ] ] += f.pw1[ i ] * f.res[ i ]; }
        vcycle( l + 1 );
        for( int i = 0; i < n; ++i ) f.x[ i ] += f.pw0[ i ] * c.x[ f.pi0[ i ] ] + ( f.pi1[ i ] >= 0 ? f.pw1[ i ] * c.x[ f.pi1[ i ] ] : T( 0 ) );
        for( int s = 0; s < nu; ++s ) gs( f.A, f.b.data(), f.x.data(), false );
    }
    // ---------------------------------------------------------------- additive tent blurs ( BPX ), no Galerkin
    // B r = D_0^-1 r + sum_l P_( 0 <- l ) D_l^-1 P_( 0 <- l )^T r, P the linear interpolation of the levels ( a free blur on the
    // hexagonal hierarchy ), D_l = c D_agg / 2^l with D_agg the Galerkin diagonal of the aggregation ( each point in its first
    // coarse parent: the sum of the weights of the edges that leave its aggregate ), as grid_stencil/newton.cpp's `tent`
    T c_tent = T( 2 ) / 3;
    std::vector<std::vector<T>> Dinv;              // per level, the inverse of the scaled diagonal
    void setup_bpx( const CSR &A0 ) {
        L[ 0 ].A = A0;
        Dinv.assign( L.size(), {} );
        Dinv[ 0 ].resize( A0.n );
        for( int i = 0; i < A0.n; ++i ) Dinv[ 0 ][ i ] = A0.diag[ i ] > 0 ? 1 / A0.diag[ i ] : 0;
        // the aggregate of each fine point at each level ( the first parent, composed )
        std::vector<int> agg( A0.n );
        for( int i = 0; i < A0.n; ++i ) agg[ i ] = i;
        for( size_t l = 1; l < L.size(); ++l ) {
            const Level &f = L[ l - 1 ];
            for( int i = 0; i < A0.n; ++i ) agg[ i ] = f.pi0[ agg[ i ] ];
            std::vector<T> D( L[ l ].q.size(), 0 );
            for( int i = 0; i < A0.n; ++i )
                for( int k = A0.beg[ i ]; k < A0.beg[ i + 1 ]; ++k ) {
                    const int j = A0.col[ k ];
                    if( j != i && agg[ j ] != agg[ i ] ) D[ agg[ i ] ] -= A0.val[ k ];
                }
            Dinv[ l ].resize( D.size() );
            const T sc = T( 1 << std::min<size_t>( l, 30 ) ) / c_tent;
            for( size_t c = 0; c < D.size(); ++c ) Dinv[ l ][ c ] = D[ c ] > 0 ? sc / D[ c ] : 0;
        }
    }
    void apply_bpx( const T *r, T *z ) {
        const size_t nl = L.size();
        std::copy( r, r + L[ 0 ].A.n, L[ 0 ].b.begin() );
        for( size_t l = 0; l + 1 < nl; ++l ) {    // restriction ( the transpose of the interpolation )
            Level &f = L[ l ], &c = L[ l + 1 ];
            std::fill( c.b.begin(), c.b.end(), T( 0 ) );
            for( size_t i = 0; i < f.q.size(); ++i ) { c.b[ f.pi0[ i ] ] += f.pw0[ i ] * f.b[ i ]; if( f.pi1[ i ] >= 0 ) c.b[ f.pi1[ i ] ] += f.pw1[ i ] * f.b[ i ]; }
        }
        for( size_t l = nl; l-- > 0; ) {           // the diagonals, then the interpolation back, summed
            Level &f = L[ l ];
            for( size_t i = 0; i < f.q.size(); ++i ) f.x[ i ] = Dinv[ l ][ i ] * f.b[ i ];
            if( l + 1 < nl ) {
                const Level &c = L[ l + 1 ];
                for( size_t i = 0; i < f.q.size(); ++i ) f.x[ i ] += f.pw0[ i ] * c.x[ f.pi0[ i ] ] + ( f.pi1[ i ] >= 0 ? f.pw1[ i ] * c.x[ f.pi1[ i ] ] : T( 0 ) );
            }
        }
        std::copy( L[ 0 ].x.begin(), L[ 0 ].x.end(), z );
    }

    /// z = M^-1 r
    void apply( const T *r, T *z ) {
        Level &f = L[ 0 ];
        std::copy( r, r + f.A.n, f.b.begin() );
        vcycle( 0 );
        std::copy( f.x.begin(), f.x.end(), z );
    }
};

} // namespace hexgrid
