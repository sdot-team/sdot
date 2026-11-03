// C++ certificate vs true rings from the Python reference ( psi_cases.bin )
#include "stencil_kernel.h"
#include <cstdio>
#include <vector>
#include <string>

template<int R>
void one( const std::string &name, int N, const std::vector<double> &psi, const std::vector<int> &rings ) {
    using SK = StencilKernel<R, 32, double>;
    constexpr int D = SK::D;
    const int P = N + 2 * R;
    std::vector<double> pp( P * P, -std::numeric_limits<double>::infinity() );
    for( int a = 0; a < N; ++a ) for( int b = 0; b < N; ++b ) pp[ ( a + R ) * P + b + R ] = psi[ a * N + b ];
    std::vector<double> area( N * N ), hess( N * N * D ), verts( N * N * D * 2 );
    std::vector<signed char> own( N * N * D );
    std::vector<unsigned char> flag( N * N );
    SK::run( N, pp.data(), area.data(), hess.data(), own.data(), verts.data() );
    int nf = SK::certify( N, own.data(), verts.data(), flag.data(), 1e-9 );
    int bad = 0, miss = 0, miss_nb = 0;
    for( int i = 0; i < N * N; ++i ) {
        bool b = rings[ i ] > R;
        bad += b;
        if( b && ! flag[ i ] ) {
            ++miss;
            // flagged within ring 1 ?
            bool nb = false;
            for( int da = -1; da <= 1; ++da ) for( int db = -1; db <= 1; ++db ) {
                int a = i / N + da, c = i % N + db;
                if( a >= 0 && a < N && c >= 0 && c < N ) nb |= flag[ a * N + c ];
            }
            miss_nb += ! nb;
        }
    }
    std::printf( "%-16s R=%d  bad %4d  flagged %4d  missed %3d  ( missed even with ring-1 dilation %d )\n", name.c_str(), R, bad, nf, miss, miss_nb );
}

int main() {
    FILE *f = std::fopen( "psi_cases.bin", "rb" );
    int hdr[ 2 ];
    std::fread( hdr, 4, 2, f );
    int nc = hdr[ 0 ], N = hdr[ 1 ];
    for( int c = 0; c < nc; ++c ) {
        char name[ 33 ] = {};
        std::fread( name, 1, 32, f );
        std::vector<double> psi( N * N );
        std::vector<int> rings( N * N );
        std::fread( psi.data(), 8, N * N, f );
        std::fread( rings.data(), 4, N * N, f );
        std::string s( name );
        s = s.substr( 0, s.find_last_not_of( ' ' ) + 1 );
        one<1>( s, N, psi, rings );
        one<2>( s, N, psi, rings );
    }
}
