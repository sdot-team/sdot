// The whole hexagonal lattice with the zone certificate ( HexZones.h ).
//   hexzones N [ reps ] [ tile ]                 bench: ns/cell of the certified diagram, share of straddling cells, bad zones, exact cells
//   hexzones Wc R psi_file tile                  check mode: writes psi_file.out = areas, then levels ( as doubles ), R x Wc
// build ( from this folder ):
//   c++ -std=c++20 -O3 -march=native -fno-math-errno -I ../../../src/sdot/include hexzones.cpp -o hexzones
#include "HexZones.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>

using T = double;
constexpr int B = 16;
using Z = hexgrid::HexZones<B>;

struct Lat {
    int Wc, R, P;
    std::vector<T> pp;
    Lat( int Wc, int R ) : Wc( Wc ), R( R ), P( ( Wc + B - 1 ) / B * B + 4 ), pp( size_t( R + 4 ) * P, -std::numeric_limits<T>::infinity() ) {}
    T &psi( int r, int c ) { return pp[ size_t( r + 2 ) * P + c + 2 ]; }
};

int main( int argc, char **argv ) {
    if( argc > 4 ) {
        const int Wc = std::atoi( argv[ 1 ] ), R = std::atoi( argv[ 2 ] );
        Lat L( Wc, R );
        FILE *f = std::fopen( argv[ 3 ], "rb" );
        for( int r = 0; r < R; ++r ) for( int c = 0; c < Wc; ++c ) if( std::fread( &L.psi( r, c ), sizeof( T ), 1, f ) != 1 ) return 1;
        std::fclose( f );
        Z z( Wc, R, L.P, L.pp.data(), std::atof( argv[ 4 ] ) );
        z.run();
        f = std::fopen( ( std::string( argv[ 3 ] ) + ".out" ).c_str(), "wb" );
        std::fwrite( z.area.data(), sizeof( T ), z.area.size(), f );
        for( auto l : z.lvl ) { T v = l; std::fwrite( &v, sizeof( T ), 1, f ); }
        std::fclose( f );
        std::printf( "straddle %ld ring2 %ld bad zones %ld / %d round cells %ld psi_max %ld\n", z.nstraddle, z.nring2, z.nbad, z.TX * z.TY, z.nround, z.nexact );
        if( std::getenv( "ZDBG" ) )
            for( int ty = 0, k = 0; ty < z.TY; ++ty ) for( int tx = 0; tx < z.TX; ++tx ) {
                const size_t q = size_t( ty ) * z.TX + tx;
                const double zx = std::min( z.X1, z.X0 + ( tx + 1 ) * z.tile ) - ( z.X0 + tx * z.tile ), zy = std::min( z.Y1, z.Y0 + ( ty + 1 ) * z.tile ) - ( z.Y0 + ty * z.tile );
                if( z.zbad[ q ] && k++ < 6 ) std::printf( "  zone %d %d: sum - area %.3e ( area %g ) tol %.3e\n", tx, ty, ( z.zsum[ q ] - zx * zy ) + z.zcmp[ q ], zx * zy, z.ztol[ q ] );
            }
        return 0;
    }
    const int N = argc > 1 ? std::atoi( argv[ 1 ] ) : 1024;
    const int reps = argc > 2 ? std::atoi( argv[ 2 ] ) : 5;
    const T tile = argc > 3 ? std::atof( argv[ 3 ] ) : 16;
    const int Wc = int( std::lround( N * std::sqrt( std::sqrt( 3.0 ) / 2 ) / B ) ) * B, R = int( std::lround( double( N ) * N / Wc ) );
    std::mt19937 rng( 0 );
    std::normal_distribution<T> nd;
    std::printf( "hexagonal %d x %d, tile %g; ns/cell, single thread\n", Wc, R, tile );
    struct Case { const char *name; double A, noise; };
    for( Case cs : { Case{ "flat", 0, 0 }, Case{ "noise 0.1", 0, 0.1 }, Case{ "noise 0.3", 0, 0.3 }, Case{ "noise 1", 0, 1 },
                     Case{ "smooth 2", 2, 0 }, Case{ "smooth 20", 20, 0 }, Case{ "smooth 20 + 0.1", 20, 0.1 } } ) {
        Lat L( Wc, R );
        for( int r = 0; r < R; ++r ) for( int c = 0; c < Wc; ++c )
            L.psi( r, c ) = cs.A * std::sin( 2 * M_PI * ( c + 0.5 * ( r & 1 ) ) / 64 ) * std::sin( 2 * M_PI * r * std::sqrt( 3.0 ) / 2 / 48 ) + cs.noise * nd( rng );
        Z z( Wc, R, L.P, L.pp.data(), tile );
        double best = 1e30;
        for( int k = 0; k < reps; ++k ) {
            auto t0 = std::chrono::steady_clock::now();
            z.run();
            best = std::min( best, std::chrono::duration<double, std::nano>( std::chrono::steady_clock::now() - t0 ).count() / ( double( Wc ) * R ) );
        }
        const double n = double( Wc ) * R;
        std::printf( "%-16s %7.2f ns/cell | ring 2 %5.2f %%  straddling %5.2f %%  bad zones %ld / %d  round cells %5.3f %%  psi_max %5.3f %%\n", cs.name, best,
                     100 * z.nring2 / n, 100 * z.nstraddle / n, z.nbad, z.TX * z.TY, 100 * z.nround / n, 100 * z.nexact / n );
    }
}
