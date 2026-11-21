// The square counterpart of hexw2.cpp: W2 between the Lebesgue measure of the box [ -1/2, N - 1/2 ]^2 and the N x N diracs at ( a, b )
// ( masses uniform | sines | bands<c>, the formulas of hexw2 ), with the SAME Newton ( sdot's, sdotplan/Newton.h step TRIALS, as
// `Solver::newton_sdot` of hexw2 ) on the library's grid machinery, read only: the certified diagrams of `grid::Evaluator<UnitCells>`
// ( ring 1 + zones ), the tensor blurs `TensorPrecond` and `pcg_fused` of sdot/grid/GridNewton2D.h. No coarse to fine, no repair,
// no tau_limit: only what sdot's generic Newton does, so that the iterations can be compared with sdot ( exp_hex_sdot.py lattice=square )
// and with hexw2.
//   sqw2 N kind [ tol ]         ( tol: max |a - nu| on masses normalized to 1, as sdot; env CG_RTOL, QUIET )
// build ( from this folder ):
//   c++ -std=c++23 -O3 -march=native -fno-math-errno -I ../../../src/sdot/include -I ../../../../loom/src/loom/include sqw2.cpp ../../../../loom/src/loom/cpp/runtime/cpu_thread_pool.cpp -o sqw2
#include <sdot/grid/GridNewton2D.h>
#include "lanczos.h"
#include "HexTensor.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

using namespace sdot::grid;
using clk = std::chrono::steady_clock;
static double secs( clk::time_point a ) { return std::chrono::duration<double>( clk::now() - a ).count(); }

int main( int argc, char **argv ) {
    const int N = argc > 1 ? std::atoi( argv[ 1 ] ) : 128;
    const std::string kind = argc > 2 ? argv[ 2 ] : "sines";
    const double tol = argc > 3 ? std::atof( argv[ 3 ] ) : 1e-10;
    const double cg_rtol = std::getenv( "CG_RTOL" ) ? std::atof( std::getenv( "CG_RTOL" ) ) : 1e-4;
    const bool trace = ! std::getenv( "QUIET" );
    const int n = N * N, P = N + 2;
    // masses ( u, v in [ 0, 1 ] over the box ), summing to the area of the box
    std::vector<T> m( n );
    T tot = 0;
    for( int a = 0; a < N; ++a ) for( int b = 0; b < N; ++b ) {
        const T u = ( a + T( 0.5 ) ) / N, v = ( b + T( 0.5 ) ) / N;
        T w = 1;
        if( kind == "sines" ) w = 1 + 0.5 * std::sin( 2 * M_PI * 2 * u ) * std::sin( 2 * M_PI * 3 * v );
        else if( kind.rfind( "bands", 0 ) == 0 ) { const T cc = std::atof( kind.c_str() + 5 ); w = 1 + ( cc - 1 ) * 0.5 * ( 1 + std::tanh( 40 * std::sin( 2 * M_PI * 2.5 * ( u + 0.3 * v ) ) ) ); }
        m[ a * N + b ] = w; tot += w;
    }
    for( T &v : m ) v *= T( n ) / tot;
    const T tol_abs = tol * n;

    Par par;
    Evaluator<UnitCells> ev( N, UnitCells{}, par );
    Operator op;
    TensorPrecond M;
    M.par = par;
    std::vector<T> psi( n, 0 ), p2( n ), b( n ), rhs( size_t( P ) * P ), dx( size_t( P ) * P );
    double t_diag = 0, t_lin = 0;
    long n_diag = 0, n_cg = 0, n_backtracks = 0, n_early = 0;
    const bool early = ! std::getenv( "NO_EARLY" );
    int it_switch = -1;
    auto diagram = [&]( const std::vector<T> &p ) { auto t0 = clk::now(); ev.run( p ); t_diag += secs( t0 ); ++n_diag; };
    auto g_of = []( T x ) { return std::log( std::max( x, T( 1e-300 ) ) ); };
    auto merit = [&]( const std::vector<T> &A, bool lin ) {
        T s = 0;
        if( lin ) { for( int i = 0; i < n; ++i ) s += ( m[ i ] - A[ i ] ) * ( m[ i ] - A[ i ] ); return std::sqrt( s ); }
        T mean = 0; for( int i = 0; i < n; ++i ) mean += g_of( A[ i ] / m[ i ] ); mean /= n;
        for( int i = 0; i < n; ++i ) { const T e = g_of( A[ i ] / m[ i ] ) - mean; s += e * e; }
        return std::sqrt( s );
    };
    const auto tt0 = clk::now();
    diagram( psi );
    bool lin = false;
    T eps = 0, t_last = 1;
    int its = -1;
    for( int it = 0; it < 100; ++it ) {
        const std::vector<T> &a = ev.mass;
        T worst = 0, worst_rel = 0;
        for( int i = 0; i < n; ++i ) { worst = std::max( worst, std::abs( m[ i ] - a[ i ] ) ); worst_rel = std::max( worst_rel, std::abs( m[ i ] - a[ i ] ) / m[ i ] ); }
        if( ! lin && worst_rel <= 2 ) { lin = true; it_switch = it; if( trace ) std::printf( "      switch: residual -> lin ( max|a-nu|/nu %.3e )\n", worst_rel ); }
        if( lin ) for( int i = 0; i < n; ++i ) b[ i ] = m[ i ] - a[ i ];
        else {
            T su = 0, sug = 0;
            for( int i = 0; i < n; ++i ) { su += a[ i ]; sug += a[ i ] * g_of( a[ i ] / m[ i ] ); }
            const T c = sug / su;
            for( int i = 0; i < n; ++i ) b[ i ] = a[ i ] * ( c - g_of( a[ i ] / m[ i ] ) );
        }
        { T s = 0; for( T v : b ) s += v; s /= n; for( T &v : b ) v -= s; }
        if( it == 0 ) { T am = 1e300, nm = 1e300; for( int i = 0; i < n; ++i ) { am = std::min( am, a[ i ] ); nm = std::min( nm, m[ i ] ); } eps = T( 0.5 ) * std::min( am, nm ); }
        const T nr = merit( a, lin );
        if( worst <= tol_abs ) { if( trace ) std::printf( "    it %2d  |r|_2 %.3e  max|a-nu| %.3e  CONVERGED\n", it, nr, worst ); its = it; break; }
        const long d0 = n_diag;
        auto tl = clk::now();
        op.bind( ev );
        M.build( op );
        std::fill( rhs.begin(), rhs.end(), T( 0 ) );
        for( int aa = 0; aa < N; ++aa ) for( int bb = 0; bb < N; ++bb ) rhs[ ( aa + 1 ) * P + bb + 1 ] = b[ aa * N + bb ];
        if( std::getenv( "SPECTRUM" ) ) {
            // ( a plain PCG on the same operator and the same blurs, recording its coefficients: the spectrum of M^-1 H )
            const int PP = P * P;
            std::vector<T> x( PP, 0 ), r = rhs, z( PP, 0 ), pv( PP, 0 ), q( PP, 0 ), zr( N );
            // ( MYTENS=1: hex_stencil's HexTensor in its square mode instead of the library's TensorPrecond )
            static const bool mine = std::getenv( "MYTENS" ) != nullptr;
            hexgrid::HexTensorT<double> ht;
            std::vector<T> ru( n ), zu( n );
            if( mine ) {
                ht.mode = 2; ht.cell = std::getenv( "HEX_CELL" ) != nullptr;
                if( const char *c_ = std::getenv( "CTENS" ) ) ht.c = std::atof( c_ );
                std::vector<double> dg( n, 0 ), ka( n, 0 ), kb( n, 0 );
                const T *W = op.W->data();
                for( int aa = 0; aa < N; ++aa ) for( int bb = 0; bb < N; ++bb ) {
                    const int i = ( aa + 1 ) * P + bb + 1, u = aa * N + bb;
                    for( int k = 0; k < 8; ++k ) { const T w = W[ k * PP + i ]; dg[ u ] += w; ka[ u ] += w * T1::off_x[ k ] * T1::off_x[ k ]; kb[ u ] += w * T1::off_y[ k ] * T1::off_y[ k ]; }
                }
                for( size_t e = 0; e < op.ex_w.size(); ++e ) {
                    const int i = op.ex_ij[ e ].first, j = op.ex_ij[ e ].second, u = ( i / P - 1 ) * N + i % P - 1;
                    const int da = j / P - i / P, db = j % P - i % P;
                    dg[ u ] += op.ex_w[ e ]; ka[ u ] += op.ex_w[ e ] * da * da; kb[ u ] += op.ex_w[ e ] * db * db;
                }
                ht.build( N, N, dg, ka, kb );
            }
            auto apply_M = [&] {
                if( mine ) {
                    for( int aa = 0; aa < N; ++aa ) for( int bb = 0; bb < N; ++bb ) ru[ aa * N + bb ] = r[ ( aa + 1 ) * P + bb + 1 ];
                    ht.apply( N, ru.data(), zu.data() );
                    T sz = 0; for( T v : zu ) sz += v; sz /= n;
                    for( int aa = 0; aa < N; ++aa ) for( int bb = 0; bb < N; ++bb ) z[ ( aa + 1 ) * P + bb + 1 ] = zu[ aa * N + bb ] - sz;
                    return;
                }
                for( int aa = 0; aa < N; ++aa ) M.row_b( aa, &r[ ( aa + 1 ) * P + 1 ] );
                M.coarse();
                T sz = 0;
                for( int aa = 0; aa < N; ++aa ) {
                    for( int bb = 0; bb < N; ++bb ) zr[ bb ] = M.id0[ ( aa + 1 ) * P + bb + 1 ] * r[ ( aa + 1 ) * P + bb + 1 ];
                    M.row_c( aa, zr.data() );
                    for( int bb = 0; bb < N; ++bb ) { z[ ( aa + 1 ) * P + bb + 1 ] = zr[ bb ]; sz += zr[ bb ]; }
                }
                sz /= n;
                for( int aa = 0; aa < N; ++aa ) for( int bb = 0; bb < N; ++bb ) z[ ( aa + 1 ) * P + bb + 1 ] -= sz;
            };
            auto matvec = [&] {
                const T *W = op.W->data();
                for( int aa = 0; aa < N; ++aa ) for( int bb = 0; bb < N; ++bb ) {
                    const int i = ( aa + 1 ) * P + bb + 1;
                    T v = 0;
                    for( int k = 0; k < 8; ++k ) v += W[ k * PP + i ] * ( pv[ i ] - pv[ i + op.off[ k ] ] );
                    q[ i ] = v;
                }
                for( size_t e = 0; e < op.ex_w.size(); ++e ) { const int i = op.ex_ij[ e ].first, j = op.ex_ij[ e ].second; q[ i ] += op.ex_w[ e ] * ( pv[ i ] - pv[ j ] ); }
            };
            auto dot = [&]( const std::vector<T> &u, const std::vector<T> &v ) { T s_ = 0; for( int i = 0; i < PP; ++i ) s_ += u[ i ] * v[ i ]; return s_; };
            apply_M(); pv = z;
            T rz = dot( r, z ), nb0 = std::sqrt( dot( r, r ) );
            std::vector<double> als, bes;
            int k = 0;
            for( ; k < 20000; ++k ) {
                matvec();
                const T al = rz / dot( pv, q ); als.push_back( al );
                for( int i = 0; i < PP; ++i ) { x[ i ] += al * pv[ i ]; r[ i ] -= al * q[ i ]; }
                if( std::sqrt( dot( r, r ) ) <= cg_rtol * nb0 ) { ++k; break; }
                apply_M();
                const T rz2 = dot( r, z ), be = rz2 / rz; bes.push_back( be ); rz = rz2;
                for( int i = 0; i < PP; ++i ) pv[ i ] = z[ i ] + be * pv[ i ];
            }
            double lo, hi; lanczos_extremes( als, bes, lo, hi );
            std::printf( "    CG %d its: lambda %.3e .. %.3e, kappa %.1f\n", k, lo, hi, hi / lo );
            dx = x; n_cg += k;
        } else
        n_cg += pcg_fused( op, M, rhs, dx, cg_rtol, 20000, par );
        t_lin += secs( tl );
        T t = std::min( T( 1 ), 4 * t_last );
        bool taken = false;
        for( int trial = 0; trial < 60; ++trial ) {
            for( int aa = 0; aa < N; ++aa ) for( int bb = 0; bb < N; ++bb ) p2[ aa * N + bb ] = psi[ aa * N + bb ] + t * dx[ ( aa + 1 ) * P + bb + 1 ];
            // the early rejection ( `Evaluator::pre_rounds`, as hexw2 ): the floor on the upper bounds, and in LIN the merit of the
            // exact cells alone as a lower bound
            const T lim = ( 1 - t / 2 ) * nr;
            if( early ) ev.pre_rounds = [&]( const std::vector<T> &A, const std::vector<int> &cells ) {
                std::vector<unsigned char> in( n, 0 );
                for( int i : cells ) in[ i ] = 1;
                T s = 0;
                for( int i = 0; i < n; ++i ) {
                    if( A[ i ] < eps ) return false;
                    if( ! in[ i ] ) s += ( m[ i ] - A[ i ] ) * ( m[ i ] - A[ i ] );
                }
                return ! lin || std::sqrt( s ) <= lim;
            };
            diagram( p2 );
            ev.pre_rounds = nullptr;
            bool ok = false;
            if( ev.aborted ) ++n_early;
            else {
                T m2 = 1e300; for( int i = 0; i < n; ++i ) m2 = std::min( m2, ev.mass[ i ] );
                const T n2r = merit( ev.mass, lin );
                ok = m2 >= eps && std::isfinite( n2r ) && n2r <= lim && n2r < nr;
            }
            if( ok ) { taken = true; break; }
            t /= 2; ++n_backtracks;
            if( t < 1e-10 ) break;
        }
        if( trace ) std::printf( "    it %2d  |r|_2 %.3e  max|a-nu| %.3e  step %.2e  %ld diag\n", it, nr, worst, t, n_diag - d0 );
        if( ! taken && ! lin ) { lin = true; it_switch = it; diagram( psi ); --it; if( trace ) std::printf( "      switch: residual -> lin ( the step of the log residual is refused )\n" ); continue; }
        if( ! taken ) { diagram( psi ); std::printf( "    STAGNATION\n" ); break; }
        t_last = t;
        psi = p2;
    }
    const double tt = secs( tt0 );
    std::vector<T> bx, by, m2v;
    ev.moments( psi, bx, by, m2v );
    T w2 = 0; for( T v : m2v ) w2 += v;
    std::printf( "%d x %d %s square: %d Newton its, %ld diagrams ( %ld trials rejected before the rounds, %ld backtracks, switch at %d ), %ld CG its | total %.3f s ( diagrams %.3f, linear %.3f ) | W2^2 %.12e\n",
                 N, N, kind.c_str(), its, n_diag, n_early, n_backtracks, it_switch, n_cg, tt, t_diag, t_lin, w2 );
}
