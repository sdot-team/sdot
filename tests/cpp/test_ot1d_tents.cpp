// A standalone check of `ot1d::transport_1d` + `tents_1d.h` against the numpy reference ( printed values compared by
// tests/test_ot1d.py ): reads tents and a piecewise-constant density on stdin, prints the cost and b, Q, phi per tent.
//   c++ -std=c++20 -O2 -I src/sdot/include -I ../loom/src/loom/include tests/cpp/test_ot1d_tents.cpp
#include <sdot/ot1d/tents_1d.h>

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <vector>

using namespace sdot::ot1d;

int main() {
    long n, B;
    double s_min, ds;
    if ( scanf( "%ld %ld %lf %lf", &n, &B, &s_min, &ds ) != 4 ) return 1;
    std::vector<double> s( n ), w( n ), m( n ), v( B );
    for ( long i = 0; i < n; ++i ) scanf( "%lf %lf %lf", &s[ i ], &w[ i ], &m[ i ] );
    double tot = 0;
    for ( long j = 0; j < B; ++j ) { scanf( "%lf", &v[ j ] ); tot += v[ j ] * ds; }

    std::vector<double> bp( 3 * n ), dsl( 3 * n );
    for ( long i = 0; i < n; ++i ) {
        bp[ 3 * i ] = s[ i ] - w[ i ];  dsl[ 3 * i ] = m[ i ] / ( w[ i ] * w[ i ] );
        bp[ 3 * i + 1 ] = s[ i ];       dsl[ 3 * i + 1 ] = -2 * m[ i ] / ( w[ i ] * w[ i ] );
        bp[ 3 * i + 2 ] = s[ i ] + w[ i ]; dsl[ 3 * i + 2 ] = m[ i ] / ( w[ i ] * w[ i ] );
    }
    std::vector<int> order( 3 * n );
    std::iota( order.begin(), order.end(), 0 );
    std::stable_sort( order.begin(), order.end(), [&]( int a, int b ) { return bp[ a ] < bp[ b ]; } );

    auto src = tent_stream( [&]( long k ) { return bp[ order[ k ] ]; }, [&]( long k ) { return dsl[ order[ k ] ]; },
                            [&]( long k ) { return order[ k ]; }, 3 * n );
    auto dst = func_stream0( [&]( long k ) { PolyPiece1d<double,0> p; p.x0 = s_min + k * ds; p.x1 = p.x0 + ds; p.c[ 0 ] = v[ k ] / tot; return p; }, B );
    std::vector<Sums1d> ev( 3 * n );
    Sums1d S = transport_1d<3>( src, dst, [&]( int id, const Sums1d &sums ) { ev[ id ] = sums; } );
    printf( "%.17g\n", S.cost );
    double mean_phi = 0;
    for ( long i = 0; i < n; ++i ) mean_phi += ev[ 3 * i + 1 ].phi / n;
    for ( long i = 0; i < n; ++i ) {
        TentMoments t = tent_moments( s[ i ], w[ i ], ev[ 3 * i ], ev[ 3 * i + 1 ], ev[ 3 * i + 2 ] );
        printf( "%.17g %.17g %.17g\n", t.b, t.Q, t.phi - mean_phi );
    }
    return 0;
}
