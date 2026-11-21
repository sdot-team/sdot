// W2 between the Lebesgue measure of the box and diracs on a hexagonal lattice ( given masses ): damped Newton on psi, cells certified
// by zones ( HexZones.h ), hessian symmetrized, PCG ( Jacobi for now ).
//   hexw2 Wc R kind [ tol ] [ out ]      kind: uniform | sines | bands<c> ( contrast c ); out: psi then areas ( doubles, R x Wc )
// build ( from this folder ):
//   c++ -std=c++20 -O3 -march=native -fno-math-errno -I ../../../src/sdot/include hexw2.cpp -o hexw2
#include "HexZones.h"
#include "HexMG.h"
#include "HexTensor.h"
#include "HexGMG.h"
#include "HexGMG2.h"
#include "lanczos.h"
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

using T = double;
constexpr int B = 16;
using Z = hexgrid::HexZones<B>;
using hexgrid::uq; using hexgrid::ur;
using clk = std::chrono::steady_clock;
static double secs( clk::time_point a ) { return std::chrono::duration<double>( clk::now() - a ).count(); }

struct Solver {
    int Wc, R, P, n;
    std::vector<T> pp, m;                           // padded psi, target masses
    Z z;
    std::vector<int> nb;                            // n x 6: the ring-1 neighbours ( -1 out )
    double t_diag = 0, t_lin = 0; long n_diag = 0, n_cg = 0, n_early = 0;
    T cg_rtol = 1e-4;
    bool use_mg = true, verbose_ = true;
    // PREC=frames: the tensor blurs of the hex FRAMES ( pairs of axes u0 ( 1, 0 ), u1 ( 0, 1 ), u2 ( -1, 1 ), env FRAMES = "01,02,12" ),
    // averaged: z = r / diag + mean over the frames of their blurs
    std::vector<hexgrid::HexTensorT<double>> frames;
    hexgrid::HexGMG gmg;                            // PREC=gmg: the structured geometric multigrid
    hexgrid::HexGMG2 gmg2;                          // PREC=gmg2: the same on the 18-neighbour stencil
    bool gmg_ready = false;
    std::vector<std::array<int,2>> frame_axes;
    std::string prec_kind = "mg";                   // mg ( Galerkin V-cycle ) | bpx ( additive tent blurs ) | jacobi
    hexgrid::HexMG mg;
#ifdef TENS_DOUBLE
    hexgrid::HexTensor tens;
#else
    hexgrid::HexTensorT<float> tens;               // ( float: same CG iterations, -30 % on the blur )
#endif
    double t_setup = 0, t_mv = 0, t_pc = 0, t_vec = 0, t_hess = 0;

    Solver( int Wc, int R ) : Wc( Wc ), R( R ), P( ( Wc + B - 1 ) / B * B + 4 ), n( Wc * R ), pp( size_t( R + 4 ) * P, -std::numeric_limits<T>::infinity() ),
            m( n ), z( Wc, R, P, nullptr, 32 ), nb( 6 * size_t( n ) ) {
        z.pp = pp.data(); z.want_hess = true;
        if( const char *t = std::getenv( "TILE" ) ) { z = Z( Wc, R, P, pp.data(), std::atof( t ) ); z.want_hess = true; }
        if( const char *k = std::getenv( "K0" ) ) { z.K0 = std::atoi( k ); z.max_K = 4 * z.K0; }
        for( int r = 0; r < R; ++r ) for( int c = 0; c < Wc; ++c ) for( int k = 0; k < 6; ++k ) nb[ 6 * ( r * Wc + c ) + k ] = z.nbr( r, c, uq[ k ], ur[ k ] );
        std::vector<int> q0( n ), r0( n );
        for( int r = 0; r < R; ++r ) for( int c = 0; c < Wc; ++c ) { q0[ r * Wc + c ] = c - ( r - ( r & 1 ) ) / 2; r0[ r * Wc + c ] = r; }
        prec_kind = std::getenv( "PREC" ) ? std::string( std::getenv( "PREC" ) ) : "mg";
        if( prec_kind == "bpx" ) { mg.n_coarsest = 1; if( const char *c = std::getenv( "CTENT" ) ) mg.c_tent = std::atof( c ); }
        if( const char *c = std::getenv( "CTENS" ) ) tens.c = std::atof( c );
        tens.fill = std::getenv( "HEX_FILL" ) != nullptr;
        if( const char *m_ = std::getenv( "HEX_EMB" ) ) tens.mode = std::atoi( m_ );
        if( const char *d_ = std::getenv( "HEX_DROP" ) ) tens.drop = std::atoi( d_ );
        tens.cell = std::getenv( "HEX_CELL" ) != nullptr;
        tens.shift = std::getenv( "HEX_SHIFT" ) != nullptr;
        if( const char *w_ = std::getenv( "HEX_FINEW" ) ) tens.fine_w = std::atof( w_ );
        if( const char *c = std::getenv( "PW" ) ) tens.pw = std::atof( c );
        if( const char *c = std::getenv( "BAND" ) ) tens.band = std::atoi( c );
        mg.setup_levels( q0, r0 );
    }
    /// the CSR of the symmetrized hessian, then the multigrid operators
    void setup_frames() {
        if( frames.empty() ) {
            std::string f = std::getenv( "FRAMES" ) ? std::getenv( "FRAMES" ) : "01,02,12";
            for( size_t k = 0; k + 1 < f.size() + 1; k += 3 ) frame_axes.push_back( { f[ k ] - '0', f[ k + 1 ] - '0' } );
            frames.resize( frame_axes.size() );
        }
        // axis k as an axial vector; a frame ( i, j ): a point p = alpha u_i + beta u_j, ( alpha, beta ) = M^-1 ( q, r ), integer ( det +-1 )
        const int ax[ 3 ][ 2 ] = { { 1, 0 }, { 0, 1 }, { -1, 1 } };
        for( size_t f = 0; f < frames.size(); ++f ) {
            auto &H = frames[ f ];
            const int i = frame_axes[ f ][ 0 ], j = frame_axes[ f ][ 1 ];
            const int m00 = ax[ i ][ 0 ], m01 = ax[ j ][ 0 ], m10 = ax[ i ][ 1 ], m11 = ax[ j ][ 1 ], det = m00 * m11 - m01 * m10;
            H.mode = 5; H.cell = ! std::getenv( "VERTEX" ); H.fm[ 0 ] = m11 * det; H.fm[ 1 ] = -m01 * det; H.fm[ 2 ] = -m10 * det; H.fm[ 3 ] = m00 * det; // ( inverse, det = +-1 )
            if( const char *c_ = std::getenv( "CTENS" ) ) H.c = std::atof( c_ );
            if( const char *p_ = std::getenv( "PW" ) ) H.pw = std::atof( p_ );
            if( const char *b_ = std::getenv( "BAND" ) ) H.band = std::atoi( b_ );
            std::vector<T> ka( n, 0 ), kb( n, 0 );
            auto qof = [&]( int ii ) { const int r = ii / Wc; return ii % Wc - ( r - ( r & 1 ) ) / 2; };
            auto add = [&]( int ii, int dq, int dr, T w ) { const T da = H.fm[ 0 ] * dq + H.fm[ 1 ] * dr, db = H.fm[ 2 ] * dq + H.fm[ 3 ] * dr; ka[ ii ] += w * da * da; kb[ ii ] += w * db * db; };
            for( int ii = 0; ii < n; ++ii ) for( int k = 0; k < 6; ++k ) add( ii, uq[ k ], ur[ k ], Ws[ 6 * ii + k ] );
            for( const E &e : ex ) { const int dq = qof( e.j ) - qof( e.i ), dr = e.j / Wc - e.i / Wc; add( e.i, dq, dr, e.w ); add( e.j, -dq, -dr, e.w ); }
            H.build( R, Wc, dg, ka, kb );
        }
    }
    void setup_mg() {
        auto t0 = clk::now();
        if( prec_kind == "frames" ) { setup_frames(); t_setup += secs( t0 ); return; }
        if( prec_kind == "gmg2" ) {
            if( ! gmg_ready ) { if( const char *nu_ = std::getenv( "GMG_NU" ) ) gmg2.nu = std::atoi( nu_ ); if( const char *r2 = std::getenv( "GMG_R2" ) ) gmg2.ring2_planes_ratio = std::atof( r2 ); if( const char *ch = std::getenv( "GMG_CHEB" ) ) gmg2.cheb = std::atoi( ch ); if( const char *al = std::getenv( "GMG_ALPHA" ) ) gmg2.cheb_alpha = std::atof( al ); gmg2.wfloat = ! std::getenv( "GMG_DOUBLE" ); gmg2.fused = ! std::getenv( "GMG_UNFUSED" ); gmg2.lag = std::getenv( "GMG_LAG" ) != nullptr; gmg2.setup_levels( R, Wc ); gmg_ready = true; }
            gmg2.setup_ops( Ws, ex );
            if( std::getenv( "GMG_DEBUG" ) ) gmg2.stats();
            if( std::getenv( "GMG_BENCH" ) ) { // ( timing of the kernels of the fine level )
                auto &l = gmg2.L[ 0 ];
                std::vector<T> v( l.ps, 1.0 ), y( l.ps );
                auto t1 = clk::now(); for( int k = 0; k < 200; ++k ) gmg2.Ax( l, v.data(), y.data() ); const double tA = secs( t1 ) / 200;
                l.cd.assign( l.ps, 0.1 ); l.cd2.assign( l.ps, 0 ); l.res.assign( l.ps, 1 );
                t1 = clk::now(); for( int k = 0; k < 200; ++k ) gmg2.cheb_step( l, 0.5, 0.5, false ); const double tC = secs( t1 ) / 200;
                auto &l1 = gmg2.L[ 1 ];
                std::vector<T> v1( l1.ps, 1.0 ), y1( l1.ps );
                t1 = clk::now(); for( int k = 0; k < 200; ++k ) gmg2.Ax( l1, v1.data(), y1.data() ); const double tA1 = secs( t1 ) / 200;
                std::printf( "      fine Ax %.1f us ( %.2f ns / point, has2 %d, ext %zu ), fused cheb step %.1f us, level-1 Ax %.1f us\n", 1e6 * tA, 1e9 * tA / n, int( l.has2 ), l.ext.size(), 1e6 * tC, 1e6 * tA1 );
            }
            t_setup += secs( t0 );
            return;
        }
        if( prec_kind == "gmg" ) {
            if( ! gmg_ready ) { if( const char *nu_ = std::getenv( "GMG_NU" ) ) gmg.nu = std::atoi( nu_ ); gmg.fast = ! std::getenv( "GMG_SLOW" ); gmg.setup_levels( R, Wc ); gmg_ready = true; }
            std::vector<hexgrid::HexGMG::Extra> ge;
            for( const E &e : ex ) ge.push_back( { e.i, e.j, e.w } );
            gmg.setup_ops( Ws, ge );
            if( std::getenv( "GMG_DEBUG" ) ) {
                gmg.debug();
                // symmetry and positivity of the V-cycle, on random mean-zero vectors
                std::vector<T> u( n ), v( n ), Mu( n ), Mv( n );
                unsigned s_ = 1;
                auto rnd = [&]() { s_ = s_ * 1103515245u + 12345u; return T( ( s_ >> 8 ) & 0xffff ) / 65536 - T( 0.5 ); };
                for( int i = 0; i < n; ++i ) { u[ i ] = rnd(); v[ i ] = rnd(); }
                T mu = 0, mv = 0; for( int i = 0; i < n; ++i ) { mu += u[ i ]; mv += v[ i ]; } for( int i = 0; i < n; ++i ) { u[ i ] -= mu / n; v[ i ] -= mv / n; }
                gmg.apply( u.data(), Mu.data() ); gmg.apply( v.data(), Mv.data() );
                T uMv = 0, vMu = 0, uMu = 0; for( int i = 0; i < n; ++i ) { uMv += u[ i ] * Mv[ i ]; vMu += v[ i ] * Mu[ i ]; uMu += u[ i ] * Mu[ i ]; }
                std::printf( "      V-cycle: u.Mv %.6e v.Mu %.6e ( rel diff %.2e ), u.Mu %.6e\n", uMv, vMu, std::abs( uMv - vMu ) / std::abs( uMv ), uMu );
            }
            t_setup += secs( t0 );
            return;
        }
        if( prec_kind == "tensor" ) { // conductances along the rows ( dr^2 ) and the G columns ( di^2, di = 2 dx )
            std::vector<T> ka( n, 0 ), kb( n, 0 );
            const bool ax = tens.mode == 1;          // ( axial: the column offset is dq; checkerboard: di = 2 dx )
            const bool off = tens.mode == 3;         // ( offset rows: the column offset is the physical dx )
            for( int i = 0; i < n; ++i ) for( int k = 0; k < 6; ++k ) { const T w = Ws[ 6 * i + k ]; const int dr = ur[ k ]; const T di = off ? T( uq[ k ] ) + T( 0.5 ) * ur[ k ] : T( ax ? uq[ k ] : 2 * uq[ k ] + ur[ k ] ); ka[ i ] += w * dr * dr; kb[ i ] += w * di * di; }
            auto qof = [&]( int i ) { const int r = i / Wc; return i % Wc - ( r - ( r & 1 ) ) / 2; };
            for( const E &e : ex ) {
                const int ri = e.i / Wc, rj = e.j / Wc;
                const T dr = rj - ri, dxp = z.xs( rj, e.j % Wc ) - z.xs( ri, e.i % Wc ), di = off ? dxp : ax ? T( qof( e.j ) - qof( e.i ) ) : 2 * dxp;
                ka[ e.i ] += e.w * dr * dr; kb[ e.i ] += e.w * di * di; ka[ e.j ] += e.w * dr * dr; kb[ e.j ] += e.w * di * di;
            }
            tens.build( R, Wc, dg, ka, kb );
            t_setup += secs( t0 );
            return;
        }
        std::vector<std::vector<std::pair<int,T>>> extra( n );
        for( const E &e : ex ) { extra[ e.i ].push_back( { e.j, -e.w } ); extra[ e.j ].push_back( { e.i, -e.w } ); }
        hexgrid::HexMG::CSR A;
        A.n = n; A.beg.assign( n + 1, 0 ); A.diag = dg;
        for( int i = 0; i < n; ++i ) {
            std::vector<std::pair<int,T>> e = extra[ i ];
            e.push_back( { i, dg[ i ] } );
            for( int k = 0; k < 6; ++k ) { const int j = nb[ 6 * i + k ]; if( j >= 0 && Ws[ 6 * i + k ] != 0 ) e.push_back( { j, -Ws[ 6 * i + k ] } ); }
            std::sort( e.begin(), e.end() );
            for( size_t k = 0; k < e.size(); ++k ) {
                if( k && e[ k ].first == e[ k - 1 ].first ) { A.val.back() += e[ k ].second; continue; }
                A.col.push_back( e[ k ].first ); A.val.push_back( e[ k ].second );
            }
            A.beg[ i + 1 ] = int( A.col.size() );
        }
        if( prec_kind == "bpx" ) mg.setup_bpx( A ); else mg.setup_ops( A );
        t_setup += secs( t0 );
    }
    void set_psi( const std::vector<T> &psi ) { for( int r = 0; r < R; ++r ) for( int c = 0; c < Wc; ++c ) pp[ size_t( r + 2 ) * P + c + 2 ] = psi[ r * Wc + c ]; }
    bool dlog = std::getenv( "DIAGLOG" ) != nullptr;
    void diagram( const std::vector<T> &psi ) {
        auto t0 = clk::now(); set_psi( psi );
        if( dlog ) { z.round_stats = true; z.round_log.clear(); }
        z.run(); const double dt = secs( t0 ); t_diag += dt; ++n_diag;
        if( dlog ) {
            std::printf( "    diagram %ld: %.4f s, ring2 %ld, bad zones %ld, ext %zu", n_diag, dt, z.nring2, z.nbad, z.ext.size() );
            for( auto &l : z.round_log ) std::printf( " | K %g: %g cells %.4f s", l[ 0 ], l[ 1 ], l[ 3 ] );
            std::printf( "\n" );
        }
    }

    // the symmetrized hessian: ( w_ij + w_ji ) / 2 on each pair
    std::vector<T> Ws;                              // n x 6
    struct E { int i, j; T w; };
    std::vector<E> ex;                              // each directed ext face, w / 2 to both sides
    std::vector<T> dg;
    void build_hessian() {
        auto th = clk::now();
        struct Lap { clk::time_point t; double &acc; ~Lap() { acc += secs( t ); } } lap{ th, t_hess };
        Ws.assign( 6 * size_t( n ), 0 ); ex.clear(); dg.assign( n, 0 );
        for( int i = 0; i < n; ++i )
            for( int k = 0; k < 6; ++k ) {
                const int j = nb[ 6 * i + k ];
                if( j < 0 ) continue;
                const T w = T( 0.5 ) * ( z.W6[ 6 * i + k ] + z.W6[ 6 * size_t( j ) + ( k + 3 ) % 6 ] );
                Ws[ 6 * i + k ] = w; dg[ i ] += w;
            }
        for( auto &[ i, l ] : z.ext ) for( auto [ j, w ] : l ) if( j >= 0 && w != 0 ) { ex.push_back( { int( i ), j, T( 0.5 ) * w } ); dg[ i ] += T( 0.5 ) * w; dg[ j ] += T( 0.5 ) * w; }
    }
    void matvec( const std::vector<T> &x, std::vector<T> &y ) const {
        for( int i = 0; i < n; ++i ) {
            T s = 0;
            for( int k = 0; k < 6; ++k ) { const int j = nb[ 6 * i + k ]; if( j >= 0 ) s += Ws[ 6 * i + k ] * ( x[ i ] - x[ j ] ); }
            y[ i ] = s;
        }
        for( const E &e : ex ) { const T v = e.w * ( x[ e.i ] - x[ e.j ] ); y[ e.i ] += v; y[ e.j ] -= v; }
    }
    // PCG, Jacobi, on the mean-zero space
    // ---------------------------------------------------------------- the PCG with the tensor blurs, in row passes
    // vectors padded by one row / column of zeros ( P2 = Wc + 2 ), the hessian as 6 planes ( SoA ) at fixed offsets per row parity,
    // the matvec fused with ( p, q ), the updates with ( r, r ) and sum r, the blur reading r and writing z in place
    int P2 = 0;
    std::vector<T> Wp;                              // 6 planes + their sum ( 7 x padded size )
    std::vector<T> W12;                             // when many faces are past ring 1: the 12 ring-2 directions ( 2 u_k, v_k ) as planes
    std::vector<E> exr;                             // the faces left ( farther: the cells of the rounds )
    bool use12 = false;
    void build_planes() {
        P2 = Wc + 4;                                // ( 2 columns / rows of zeros each side: ring 2 reaches them )
        const size_t ps = size_t( R + 4 ) * P2;
        Wp.assign( 7 * ps, 0 );
        for( int r = 0; r < R; ++r ) for( int c = 0; c < Wc; ++c ) {
            const int i = r * Wc + c; const size_t q = size_t( r + 2 ) * P2 + c + 2;
            T s = 0;
            for( int k = 0; k < 6; ++k ) { Wp[ k * ps + q ] = Ws[ 6 * i + k ]; s += Ws[ 6 * i + k ]; }
            Wp[ 6 * ps + q ] = s;
        }
        exr.clear();
        use12 = std::getenv( "USE12" ) && ex.size() > size_t( n ) / 8; // ( off: 12 more passes per row cost more than the list, bands100 0.066 vs 0.043 s )
        if( ! use12 ) { exr = ex; return; }
        W12.assign( 12 * ps, 0 );
        for( const E &e : ex ) {
            const int ri = e.i / Wc, ci = e.i % Wc, rj = e.j / Wc, cj = e.j % Wc;
            int d = -1;
            for( int k = 0; k < 6 && d < 0; ++k ) {
                if( z.nbr( ri, ci, 2 * uq[ k ], 2 * ur[ k ] ) == e.j ) d = k;
                else if( z.nbr( ri, ci, hexgrid::vq[ k ], hexgrid::vr[ k ] ) == e.j ) d = 6 + k;
            }
            if( d < 0 ) { exr.push_back( e ); continue; }
            const int dop = d < 6 ? ( d + 3 ) % 6 : 6 + ( d - 6 + 3 ) % 6;   // the opposite direction, seen from j
            const size_t qi = size_t( ri + 2 ) * P2 + ci + 2, qj = size_t( rj + 2 ) * P2 + cj + 2;
            W12[ d * ps + qi ] += e.w; W12[ dop * ps + qj ] += e.w;
        }
    }
    void solve_fast( const std::vector<T> &b0, std::vector<T> &x0 ) {
        auto t0 = clk::now();
        setup_mg();
        build_planes();
        const size_t ps = size_t( R + 4 ) * P2;
        std::vector<T> x( ps, 0 ), r( ps, 0 ), z( ps, 0 ), p( ps, 0 ), q( ps, 0 );
        T mb = 0; for( T v : b0 ) mb += v; mb /= n;
        for( int rr = 0; rr < R; ++rr ) for( int c = 0; c < Wc; ++c ) r[ size_t( rr + 2 ) * P2 + c + 2 ] = b0[ rr * Wc + c ] - mb;
        auto at = [&]( int i ) { return size_t( i / Wc + 2 ) * P2 + i % Wc + 2; };
        std::vector<size_t> exi( 2 * exr.size() );
        for( size_t k = 0; k < exr.size(); ++k ) { exi[ 2 * k ] = at( exr[ k ].i ); exi[ 2 * k + 1 ] = at( exr[ k ].j ); }
        const T *r0p = r.data() + 2 * P2 + 2; T *z0p = z.data() + 2 * P2 + 2;
        // z = M r, projected to zero mean; returns ( r, z )
        auto prec = [&]() {
            if( prec_kind == "gmg2" ) gmg2.apply( r0p, P2, z0p, P2 ); else if( prec_kind == "gmg" ) gmg.apply_fast( r0p, P2, z0p, P2 ); else tens.apply_fast( Wc, r0p, P2, z0p, P2 );
            T sz = 0, rz = 0, sr = 0;
            for( int rr = 0; rr < R; ++rr ) { const T *zr = z0p + size_t( rr ) * P2, *xr = r0p + size_t( rr ) * P2; for( int c = 0; c < Wc; ++c ) { sz += zr[ c ]; rz += xr[ c ] * zr[ c ]; sr += xr[ c ]; } }
            const T mz = sz / n;
            for( int rr = 0; rr < R; ++rr ) { T *zr = z0p + size_t( rr ) * P2; for( int c = 0; c < Wc; ++c ) zr[ c ] -= mz; }
            return rz - mz * sr;
        };
        auto ta = clk::now();
        T rz = prec();
        t_pc += secs( ta );
        p = z;
        T nb0 = 0; for( T v : r ) nb0 += v * v; nb0 = std::sqrt( nb0 );
        std::vector<double> als, bes;
        for( int it = 0; it < 20 * n; ++it ) {
            ta = clk::now();
            // q = H p, ( p, q )
            T pq = 0;
            for( int rr = 0; rr < R; ++rr ) {
                const int sh = rr & 1;
                const size_t o = size_t( rr + 2 ) * P2 + 2;
                const T *pp_ = p.data() + o; T *qq = q.data() + o;
                const T *w0 = Wp.data() + o, *w1 = w0 + ps, *w2 = w1 + ps, *w3 = w2 + ps, *w4 = w3 + ps, *w5 = w4 + ps, *ws = w5 + ps;
                const int o1 = P2 + hexgrid::dcol( 0, 1, sh ), o2 = P2 + hexgrid::dcol( -1, 1, sh ), o4 = -P2 + hexgrid::dcol( 0, -1, sh ), o5 = -P2 + hexgrid::dcol( 1, -1, sh );
                for( int c = 0; c < Wc; ++c ) {
                    const T v = ws[ c ] * pp_[ c ] - ( w0[ c ] * pp_[ c + 1 ] + w1[ c ] * pp_[ c + o1 ] + w2[ c ] * pp_[ c + o2 ] + w3[ c ] * pp_[ c - 1 ] + w4[ c ] * pp_[ c + o4 ] + w5[ c ] * pp_[ c + o5 ] );
                    qq[ c ] = v;
                }
                if( use12 ) // the ring-2 planes: q += sum_d w_d ( p - p_d ) ( an entry of `ex` already holds half the face, put in both cells' planes )
                    for( int d = 0; d < 12; ++d ) {
                        const int k = d % 6, dq = d < 6 ? 2 * uq[ k ] : hexgrid::vq[ k ], dr = d < 6 ? 2 * ur[ k ] : hexgrid::vr[ k ];
                        const int od = dr * P2 + hexgrid::dcol( dq, dr, sh );
                        const T *wd = W12.data() + d * ps + o;
                        for( int c = 0; c < Wc; ++c ) qq[ c ] += wd[ c ] * ( pp_[ c ] - pp_[ c + od ] );
                    }
                for( int c = 0; c < Wc; ++c ) pq += qq[ c ] * pp_[ c ];
            }
            for( size_t k = 0; k < exr.size(); ++k ) { const size_t i = exi[ 2 * k ], j = exi[ 2 * k + 1 ]; const T w = exr[ k ].w, d = p[ i ] - p[ j ], v = w * d; q[ i ] += v; q[ j ] -= v; pq += v * d; }
            t_mv += secs( ta ); ta = clk::now();
            const T a = rz / pq;
            als.push_back( a );
            T rr2 = 0;
            for( int rr = 0; rr < R; ++rr ) {
                const size_t o = size_t( rr + 2 ) * P2 + 2;
                T *xr = &x[ o ], *rr_ = &r[ o ]; const T *pr = &p[ o ], *qr = &q[ o ];
                for( int c = 0; c < Wc; ++c ) { xr[ c ] += a * pr[ c ]; rr_[ c ] -= a * qr[ c ]; rr2 += rr_[ c ] * rr_[ c ]; }
            }
            ++n_cg;
            t_vec += secs( ta );
            if( std::sqrt( rr2 ) <= cg_rtol * nb0 ) break;
            ta = clk::now();
            const T rz2 = prec();
            t_pc += secs( ta ); ta = clk::now();
            const T be = rz2 / rz;
            bes.push_back( be );
            for( int rr = 0; rr < R; ++rr ) { const size_t o = size_t( rr + 2 ) * P2 + 2; T *pr = &p[ o ]; const T *zr = &z[ o ]; for( int c = 0; c < Wc; ++c ) pr[ c ] = zr[ c ] + be * pr[ c ]; }
            rz = rz2;
            t_vec += secs( ta );
        }
        if( std::getenv( "SPECTRUM" ) ) { double lo, hi; lanczos_extremes( als, bes, lo, hi ); std::printf( "    CG %zu its: lambda %.3e .. %.3e, kappa %.1f\n", als.size(), lo, hi, hi / lo ); }
        x0.resize( n );
        for( int rr = 0; rr < R; ++rr ) for( int c = 0; c < Wc; ++c ) x0[ rr * Wc + c ] = x[ size_t( rr + 2 ) * P2 + c + 2 ];
        t_lin += secs( t0 );
    }

    bool fast_pcg = ! std::getenv( "SLOW_PCG" );
    void solve( const std::vector<T> &b0, std::vector<T> &x ) {
        if( fast_pcg && use_mg && ( ( prec_kind == "tensor" && tens.mode == 0 && ! tens.cell ) || ( prec_kind == "gmg" && gmg.fast ) || prec_kind == "gmg2" ) ) { solve_fast( b0, x ); return; }
        auto t0 = clk::now();
        std::vector<T> b = b0, r( n ), z_( n ), p( n ), q( n );
        T mb = 0; for( T v : b ) mb += v; mb /= n; for( T &v : b ) v -= mb;
        x.assign( n, 0 ); r = b;
        auto dot = []( const std::vector<T> &a, const std::vector<T> &c ) { T s = 0; for( size_t i = 0; i < a.size(); ++i ) s += a[ i ] * c[ i ]; return s; };
        auto prec = [&] {
            if( use_mg ) {
                if( prec_kind == "frames" ) {         // ( the mean of the frames: each has the Jacobi term, the mean keeps it once )
                    std::fill( z_.begin(), z_.end(), T( 0 ) );
                    std::vector<T> zf( n );
                    for( auto &H : frames ) { H.apply( Wc, r.data(), zf.data() ); for( int i = 0; i < n; ++i ) z_[ i ] += zf[ i ] / T( frames.size() ); }
                }
                else if( prec_kind == "gmg" ) gmg.apply( r.data(), z_.data() );
                else if( prec_kind == "bpx" ) mg.apply_bpx( r.data(), z_.data() ); else if( prec_kind == "tensor" ) tens.apply( Wc, r.data(), z_.data() ); else mg.apply( r.data(), z_.data() ); T mz = 0; for( T v : z_ ) mz += v; mz /= n; for( T &v : z_ ) v -= mz; }
            else for( int i = 0; i < n; ++i ) z_[ i ] = dg[ i ] > 0 ? r[ i ] / dg[ i ] : 0;
        };
        if( use_mg ) setup_mg();
        prec();
        p = z_;
        T rz = dot( r, z_ ), nb0 = std::sqrt( dot( b, b ) );
        std::vector<double> als, bes;
        struct Sp { std::vector<double> &a, &b; ~Sp() { if( std::getenv( "SPECTRUM" ) ) { double lo, hi; lanczos_extremes( a, b, lo, hi ); std::printf( "    CG %zu its: lambda %.3e .. %.3e, kappa %.1f\n", a.size(), lo, hi, hi / lo ); } } } sp_{ als, bes };
        const int cg_max = std::getenv( "CG_MAX" ) ? std::atoi( std::getenv( "CG_MAX" ) ) : 20 * n;
        for( int it = 0; it < cg_max; ++it ) {
            auto ta = clk::now();
            matvec( p, q );
            t_mv += secs( ta ); ta = clk::now();
            const T a = rz / dot( p, q );
            als.push_back( a );
            for( int i = 0; i < n; ++i ) { x[ i ] += a * p[ i ]; r[ i ] -= a * q[ i ]; }
            ++n_cg;
            const bool done = std::sqrt( dot( r, r ) ) <= cg_rtol * nb0;
            t_vec += secs( ta );
            if( done ) break;
            ta = clk::now();
            prec();
            t_pc += secs( ta ); ta = clk::now();
            const T rz2 = dot( r, z_ );
            bes.push_back( rz2 / rz );
            for( int i = 0; i < n; ++i ) p[ i ] = z_[ i ] + rz2 / rz * p[ i ];
            rz = rz2;
            t_vec += secs( ta );
        }
        t_lin += secs( t0 );
    }

    /// Newton from psi ( damped: Kitagawa-Merigot-Thibert ); returns the iterations
    int newton( std::vector<T> &psi, T tol, int maxit = 100, bool verbose = true ) {
        diagram( psi );
        T mmax = 0; for( T v : m ) mmax = std::max( mmax, v );
        for( int it = 0; it < maxit; ++it ) {
            std::vector<T> g( n );
            T err = 0, amin = 1e300, mmin = 1e300;
            for( int i = 0; i < n; ++i ) { g[ i ] = z.area[ i ] - m[ i ]; err = std::max( err, std::abs( g[ i ] ) ); amin = std::min( amin, z.area[ i ] ); mmin = std::min( mmin, m[ i ] ); }
            if( verbose ) std::printf( "  it %2d  err %.3e  ( diag %.3f s, lin %.3f s, cg %ld, round cells %ld, bad zones %ld )\n", it, err / mmax, t_diag, t_lin, n_cg, z.nround, z.nbad );
            if( err <= tol * mmax ) return it;
            build_hessian();
            std::vector<T> d, b( n );
            for( int i = 0; i < n; ++i ) b[ i ] = -g[ i ];
            solve( b, d );
            const T eps0 = T( 0.5 ) * std::min( amin, mmin );
            std::vector<T> p2( n );
            for( T tau = 1; ; tau /= 2 ) {
                for( int i = 0; i < n; ++i ) p2[ i ] = psi[ i ] + tau * d[ i ];
                // reject before the rounds when the exact cells already fail, or an upper bound is below eps0
                const T lim = ( 1 - tau / 2 ) * err;
                z.pre_rounds = [&]( const std::vector<size_t> &todo ) {
                    std::vector<unsigned char> in( n, 0 );
                    for( size_t i : todo ) in[ i ] = 1;
                    for( int i = 0; i < n; ++i ) {
                        const T a = z.area[ i ];
                        if( a <= eps0 ) return false;                               // exact, or an upper bound
                        if( ! in[ i ] && std::abs( a - m[ i ] ) > lim ) return false;
                    }
                    return true;
                };
                diagram( p2 );
                z.pre_rounds = nullptr;
                if( z.aborted ) { ++n_early; if( tau < 1e-6 ) { std::printf( "  line search failed\n" ); return -1; } continue; }
                T e2 = 0, a2 = 1e300;
                for( int i = 0; i < n; ++i ) { e2 = std::max( e2, std::abs( z.area[ i ] - m[ i ] ) ); a2 = std::min( a2, z.area[ i ] ); }
                if( a2 > eps0 && e2 <= ( 1 - tau / 2 ) * err ) break;
                if( tau < 1e-6 ) { std::printf( "  line search failed\n" ); return -1; }
            }
            psi = p2;
        }
        return maxit;
    }

    // ---------------------------------------------------------------- the Newton of sdot ( sdotplan/Newton.h, step TRIALS )
    // residual LOG ( g = log( a / nu ), b_i = a_i ( c - g_i ), c the a-weighted mean of g ) then LIN once max |a - nu| / nu <= 2 or
    // when a LOG step is refused; merit | g - mean |_2 ( LOG ) or | a - nu |_2 ( LIN ); t = min( 1, 4 t_last ), halved; a step is
    // taken iff min a >= eps, merit <= ( 1 - t / 2 ) merit0 and merit < merit0; eps = min( nu_min, a_min( start ) ) / 2;
    // b projected on the mean-zero space; stop on max |a - nu| <= tol_abs
    double switch_residual = 2, mult_ok = 4, t_min = 1e-10;
    int max_backtracks = 60;
    int n_backtracks = 0, it_switch = -1;
    int newton_sdot( std::vector<T> &psi, T tol_abs, int maxit = 100, bool trace = true ) {
        diagram( psi );
        std::vector<T> b( n ), d, p2( n ), a2;
        bool lin = false;
        auto g_of = []( T x ) { return std::log( std::max( x, T( 1e-300 ) ) ); };
        auto merit = [&]( const std::vector<T> &A, bool lin_ ) {
            T s = 0;
            if( lin_ ) { for( int i = 0; i < n; ++i ) s += ( m[ i ] - A[ i ] ) * ( m[ i ] - A[ i ] ); return std::sqrt( s ); }
            T mean = 0; for( int i = 0; i < n; ++i ) mean += g_of( A[ i ] / m[ i ] ); mean /= n;
            for( int i = 0; i < n; ++i ) { const T e = g_of( A[ i ] / m[ i ] ) - mean; s += e * e; }
            return std::sqrt( s );
        };
        T eps = 0, t_last = 1;
        for( int it = 0; it < maxit; ++it ) {
            const std::vector<T> &a = z.area;
            T worst = 0, worst_rel = 0;
            for( int i = 0; i < n; ++i ) { worst = std::max( worst, std::abs( m[ i ] - a[ i ] ) ); worst_rel = std::max( worst_rel, std::abs( m[ i ] - a[ i ] ) / m[ i ] ); }
            if( ! lin && worst_rel <= switch_residual ) { lin = true; it_switch = it; if( trace ) std::printf( "      switch: residual -> lin ( max|a-nu|/nu %.3e )\n", worst_rel ); }
            if( lin ) for( int i = 0; i < n; ++i ) b[ i ] = m[ i ] - a[ i ];
            else {
                T su = 0, sug = 0;
                for( int i = 0; i < n; ++i ) { su += a[ i ]; sug += a[ i ] * g_of( a[ i ] / m[ i ] ); }   // u = nu / g'( x ) = a
                const T c = sug / su;
                for( int i = 0; i < n; ++i ) b[ i ] = a[ i ] * ( c - g_of( a[ i ] / m[ i ] ) );
            }
            { T s = 0; for( T v : b ) s += v; s /= n; for( T &v : b ) v -= s; }
            if( it == 0 ) { T am = 1e300, nm = 1e300; for( int i = 0; i < n; ++i ) { am = std::min( am, a[ i ] ); nm = std::min( nm, m[ i ] ); } eps = T( 0.5 ) * std::min( am, nm ); }
            const T nr = merit( a, lin );
            if( worst <= tol_abs ) { if( trace ) std::printf( "    it %2d  |r|_2 %.3e  max|a-nu| %.3e  CONVERGED\n", it, nr, worst ); return it; }
            const long d0 = n_diag;
            build_hessian();
            solve( b, d );
            T t = std::min( T( 1 ), mult_ok * t_last );
            bool taken = false;
            for( int trial = 0; trial < max_backtracks; ++trial ) {
                for( int i = 0; i < n; ++i ) p2[ i ] = psi[ i ] + t * d[ i ];
                const T lim = ( 1 - t / 2 ) * nr;
                // before the rounds: the floor on upper bounds; in LIN the merit of the exact cells alone is a lower bound
                static const bool no_early = std::getenv( "NO_EARLY" ) != nullptr;   // ( the comparison with sqw2, which has no such rejection )
                if( ! no_early ) z.pre_rounds = [&]( const std::vector<size_t> &todo ) {
                    std::vector<unsigned char> in( n, 0 );
                    for( size_t i : todo ) in[ i ] = 1;
                    T s = 0;
                    for( int i = 0; i < n; ++i ) {
                        if( z.area[ i ] < eps ) return false;
                        if( ! in[ i ] ) s += ( m[ i ] - z.area[ i ] ) * ( m[ i ] - z.area[ i ] );
                    }
                    return ! lin || std::sqrt( s ) <= lim;
                };
                diagram( p2 );
                z.pre_rounds = nullptr;
                bool ok = false;
                if( ! z.aborted ) {
                    T m2 = 1e300; for( int i = 0; i < n; ++i ) m2 = std::min( m2, z.area[ i ] );
                    const T n2r = merit( z.area, lin );
                    ok = m2 >= eps && std::isfinite( n2r ) && n2r <= lim && n2r < nr;
                } else ++n_early;
                if( ok ) { taken = true; break; }
                t /= 2; ++n_backtracks;
                if( t < t_min ) break;
            }
            if( trace ) std::printf( "    it %2d  |r|_2 %.3e  max|a-nu| %.3e  step %.2e  %ld diag\n", it, nr, worst, t, n_diag - d0 );
            if( ! taken && ! lin ) { lin = true; it_switch = it; diagram( psi ); --it; if( trace ) std::printf( "      switch: residual -> lin ( the step of the log residual is refused )\n" ); continue; }
            if( ! taken ) { diagram( psi ); if( trace ) std::printf( "    STAGNATION\n" ); return -1; }
            t_last = t;
            psi = p2;
        }
        return maxit;
    }

    /// W2^2 = sum_i int_{C_i} |x - x_i|^2 ( exact cells, polygons rebuilt )
    T w2() {
        typename Z::Poly p;
        for( int r = 0; r < R; ++r ) for( int c = 0; c < Wc; ++c ) z.psi_max = std::max( z.psi_max, z.psi( r, c ) );
        T s = 0;
        for( int r = 0; r < R; ++r )
            for( int c = 0; c < Wc; ++c ) {
                z.ring_cell( r, c, std::numeric_limits<int>::max() / 2, p );
                for( int e = 0; e < p.n; ++e ) {
                    const int f = e + 1 == p.n ? 0 : e + 1;
                    const T cr = p.x[ e ] * p.y[ f ] - p.y[ e ] * p.x[ f ];
                    s += cr * ( p.x[ e ] * p.x[ e ] + p.x[ e ] * p.x[ f ] + p.x[ f ] * p.x[ f ] + p.y[ e ] * p.y[ e ] + p.y[ e ] * p.y[ f ] + p.y[ f ] * p.y[ f ] ) / 12;
                }
            }
        return s;
    }
};

int main( int argc, char **argv ) {
    const int Wc = argc > 1 ? std::atoi( argv[ 1 ] ) : 64, R = argc > 2 ? std::atoi( argv[ 2 ] ) : 74;
    const std::string kind = argc > 3 ? argv[ 3 ] : "sines";
    const T tol = argc > 4 ? std::atof( argv[ 4 ] ) : 1e-9;
    Solver S( Wc, R );
    S.use_mg = ! std::getenv( "JACOBI" ) && S.prec_kind != "jacobi";
    S.verbose_ = ! std::getenv( "QUIET" );
    const T Lx = S.z.X1 - S.z.X0, Ly = S.z.Y1 - S.z.Y0;
    T tot = 0;
    for( int r = 0; r < R; ++r )
        for( int c = 0; c < Wc; ++c ) {
            const T u = ( S.z.xs( r, c ) - S.z.X0 ) / Lx, v = ( S.z.ys( r ) - S.z.Y0 ) / Ly;
            T w = 1;
            if( kind == "sines" ) w = 1 + 0.5 * std::sin( 2 * M_PI * 2 * u ) * std::sin( 2 * M_PI * 3 * v );
            else if( kind.rfind( "bands", 0 ) == 0 ) { const T cc = std::atof( kind.c_str() + 5 ); w = 1 + ( cc - 1 ) * 0.5 * ( 1 + std::tanh( 40 * std::sin( 2 * M_PI * 2.5 * ( u + 0.3 * v ) ) ) ); }
            S.m[ r * Wc + c ] = w; tot += w;
        }
    for( T &v : S.m ) v *= Lx * Ly / tot;
    std::vector<T> psi( S.n, 0 );
    auto t0 = clk::now();
    // NEWTON=kmt: the earlier damping ( max norm, t from 1 ); default: sdot's ( tol = max |a - nu| on masses normalized to 1, as sdot )
    const bool kmt = std::getenv( "NEWTON" ) && std::string( std::getenv( "NEWTON" ) ) == "kmt";
    if( const char *c = std::getenv( "CG_RTOL" ) ) S.cg_rtol = std::atof( c );
    const int its = kmt ? S.newton( psi, tol, 100, S.verbose_ ) : S.newton_sdot( psi, tol * Lx * Ly, 100, S.verbose_ );
    const double tt = secs( t0 );
    if( std::getenv( "ZDIAG" ) ) {
        S.z.diag_zones = true; S.z.round_stats = true; S.z.round_log.clear(); S.diagram( psi );
        for( auto &l : S.z.round_log ) std::printf( "round K %g: %g cells ( %g exact by psi_max ), %.4f s = %.2f us / cell\n", l[ 0 ], l[ 1 ], l[ 2 ], l[ 3 ], 1e6 * l[ 3 ] / l[ 1 ] );
        long lv[ 5 ] = {}; for( auto l : S.z.lvl ) ++lv[ std::clamp( int( l ), 0, 4 ) ];
        std::printf( "final: bad zones %zu, levels ring1 %ld ring2 %ld round %ld exact %ld\n", S.z.zdiag.size(), lv[ 1 ], lv[ 2 ], lv[ 3 ], lv[ 4 ] );
        // the ring <= 2 cells against the exact ones
        S.z.no_rounds = true; S.diagram( psi ); S.z.no_rounds = false;
        typename Z::Poly p;
        for( int r = 0; r < S.R; ++r ) for( int c = 0; c < S.Wc; ++c ) S.z.psi_max = std::max( S.z.psi_max, S.z.psi( r, c ) );
        T emax = 0; long nwrong = 0;
        for( int r = 0; r < S.R; ++r ) for( int c = 0; c < S.Wc; ++c ) {
            S.z.ring_cell( r, c, std::numeric_limits<int>::max() / 2, p );
            const T e = std::abs( ( p.n ? Z::poly_area( p ) : 0 ) - S.z.area[ r * S.Wc + c ] );
            emax = std::max( emax, e ); nwrong += e > 1e-12;
            if( e > 1e-12 && nwrong <= 4 ) { // which seeds cut it beyond ring 2, and how long are their faces
                const int q0 = c - ( r - ( r & 1 ) ) / 2;
                for( int k = 0; k < p.n; ++k ) {
                    const int j = p.l[ k ]; if( j < 0 ) continue;
                    const int rj = j / S.Wc, cj = j % S.Wc, dq = ( cj - ( rj - ( rj & 1 ) ) / 2 ) - q0, dr = rj - r;
                    const int hd = ( std::abs( dq ) + std::abs( dr ) + std::abs( dq + dr ) ) / 2;
                    const int f = k + 1 == p.n ? 0 : k + 1;
                    if( hd > 2 ) std::printf( "    cell ( %d, %d ) err %.2e: face to a seed at hex distance %d, length %.3e ( |psi_i| %.3e )\n", r, c, e, hd, std::hypot( p.x[ f ] - p.x[ k ], p.y[ f ] - p.y[ k ] ), std::abs( S.z.psi( r, c ) ) );
                }
            }
        }
        std::printf( "ring <= 2 cells vs exact: max err %.3e, cells > 1e-12: %ld\n", emax, nwrong );
        { // the cells whose exact polygon has a face past ring 2, by hexagonal distance
            long hist[ 40 ] = {}, need = 0;
            for( int r = 0; r < S.R; ++r ) for( int c = 0; c < S.Wc; ++c ) {
                S.z.ring_cell( r, c, std::numeric_limits<int>::max() / 2, p );
                const int q0 = c - ( r - ( r & 1 ) ) / 2;
                int mx = 0;
                for( int k = 0; k < p.n; ++k ) {
                    const int j = p.l[ k ]; if( j < 0 ) continue;
                    const int f = k + 1 == p.n ? 0 : k + 1;
                    if( std::hypot( p.x[ f ] - p.x[ k ], p.y[ f ] - p.y[ k ] ) < 1e-12 ) continue;
                    const int rj = j / S.Wc, cj = j % S.Wc, dq = ( cj - ( rj - ( rj & 1 ) ) / 2 ) - q0, dr = rj - r;
                    mx = std::max( mx, ( std::abs( dq ) + std::abs( dr ) + std::abs( dq + dr ) ) / 2 );
                }
                ++hist[ std::min( mx, 39 ) ]; need += mx > 2;
            }
            std::printf( "cells by the farthest seed with a face: " );
            for( int k = 0; k < 40; ++k ) if( hist[ k ] ) std::printf( "%d: %ld  ", k, hist[ k ] );
            std::printf( "( past ring 2: %ld )\n", need );
        }
        for( size_t k = 0; k < std::min<size_t>( 8, S.z.zdiag.size() ); ++k ) std::printf( "  excess %.3e tol %.3e forced %g\n", S.z.zdiag[ k ][ 0 ], S.z.zdiag[ k ][ 1 ], S.z.zdiag[ k ][ 2 ] );
    }
    const T w2 = S.w2();
    std::printf( "%d x %d %s %s: %d Newton its, %ld diagrams ( %ld trials rejected before the rounds, %d backtracks, switch at %d ), %ld CG its | total %.3f s ( diagrams %.3f, linear %.3f of which mg setup %.3f ) | W2^2 %.12e\n",
                 Wc, R, kind.c_str(), S.use_mg ? S.prec_kind.c_str() : "jacobi", its, S.n_diag, S.n_early, S.n_backtracks, S.it_switch, S.n_cg, tt, S.t_diag, S.t_lin, S.t_setup, w2 );
    std::printf( "  linear split: matvec %.3f s, preconditioner %.3f s, vector ops %.3f s, setup %.3f s, hessian assembly %.3f s ( %.2f us per CG it )\n",
                 S.t_mv, S.t_pc, S.t_vec, S.t_setup, S.t_hess, 1e6 * ( S.t_mv + S.t_pc + S.t_vec ) / std::max( 1L, S.n_cg ) );
    if( S.prec_kind == "gmg2" ) std::printf( "  gmg2 Galerkin split: level 0 fill %.4f, edges %.4f, finish %.4f, parents %.4f s\n", S.gmg2.tg[ 0 ], S.gmg2.tg[ 1 ], S.gmg2.tg[ 2 ], S.gmg2.tg[ 3 ] );
    if( S.prec_kind == "gmg2" ) std::printf( "  gmg2: Galerkin %.3f, coarsest %.3f s; V-cycle: smoothing %.3f, residual %.3f, restriction %.3f, prolongation %.3f, coarsest %.3f s\n", S.gmg2.ts[ 0 ], S.gmg2.ts[ 1 ], S.gmg2.tv[ 0 ], S.gmg2.tv[ 1 ], S.gmg2.tv[ 2 ], S.gmg2.tv[ 3 ], S.gmg2.tv[ 4 ] );
    if( S.prec_kind == "gmg" ) std::printf( "  setup split: Galerkin %.3f, coarsest %.3f, fast arrays %.3f s; ext per level:", S.gmg.ts[ 0 ], S.gmg.ts[ 1 ], S.gmg.ts[ 2 ] );
    if( S.prec_kind == "gmg" ) { for( auto &l : S.gmg.L ) std::printf( " %zu/%d", l.ext.size(), l.n ); std::printf( "\n" ); }
    if( S.prec_kind == "gmg" ) std::printf( "  V-cycle split: smoothing %.3f, residual %.3f, restriction %.3f, prolongation %.3f, coarsest %.3f s\n", S.gmg.tv[ 0 ], S.gmg.tv[ 1 ], S.gmg.tv[ 2 ], S.gmg.tv[ 3 ], S.gmg.tv[ 4 ] );
    std::printf( "  blur split: lattice -> fine %.3f, restrictions %.3f, scaling %.3f, prolongations %.3f, back %.3f s\n", S.tens.tt[ 0 ], S.tens.tt[ 1 ], S.tens.tt[ 2 ], S.tens.tt[ 3 ], S.tens.tt[ 4 ] );
    if( argc > 5 ) {
        FILE *f = std::fopen( argv[ 5 ], "wb" );
        std::fwrite( psi.data(), sizeof( T ), psi.size(), f ); std::fwrite( S.z.area.data(), sizeof( T ), S.n, f );
        std::fclose( f );
    }
}
