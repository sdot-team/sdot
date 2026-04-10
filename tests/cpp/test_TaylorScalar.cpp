#include "../../src/cpp/sdot/support/TaylorScalar.h"
#include "../../src/cpp/sdot/support/P.h"
#include "catch_main.h"

using namespace sdot;
using namespace std;
using TF = double;

template<int order>
using TS = TaylorScalar<TF, order>;

// ── helpers ───────────────────────────────────────────────────────────────────

static constexpr TF eps = 1e-12;

// Evaluate a TaylorScalar<TF,3> at a point p (size n)
static TF eval3( const TS<3> &f, std::vector<TF> p ) {
    const int n = f.n();
    TF r = f.c0;
    for( int i = 0; i < n; ++i ) r += f.c1[i] * p[i];
    for( int i = 0; i < n; ++i )
        for( int j = 0; j < n; ++j )
            r += f.d2(i,j) * p[i] * p[j];
    for( int i = 0; i < n; ++i )
        for( int j = 0; j < n; ++j )
            for( int k = 0; k < n; ++k )
                r += f.d3(i,j,k) * p[i] * p[j] * p[k];
    return r;
}

// Evaluate a TaylorScalar<TF,4> at a point p (size n)
static TF eval4( const TS<4> &f, std::vector<TF> p ) {
    const int n = f.n();
    TF r = f.c0;
    for( int i = 0; i < n; ++i ) r += f.c1[i] * p[i];
    for( int i = 0; i < n; ++i )
        for( int j = 0; j < n; ++j )
            r += f.d2(i,j) * p[i] * p[j];
    for( int i = 0; i < n; ++i )
        for( int j = 0; j < n; ++j )
            for( int k = 0; k < n; ++k )
                r += f.d3(i,j,k) * p[i] * p[j] * p[k];
    for( int i = 0; i < n; ++i )
        for( int j = 0; j < n; ++j )
            for( int k = 0; k < n; ++k )
                for( int l = 0; l < n; ++l )
                    r += f.d4(i,j,k,l) * p[i] * p[j] * p[k] * p[l];
    return r;
}

// ── order-3 tests ─────────────────────────────────────────────────────────────

TEST_CASE( "TaylorScalar3 construction", "[TS3]" ) {
    SECTION( "constant" ) {
        auto c = TS<3>::constant( 3, 7.0 );
        CHECK( c.c0 == 7.0 );
        for( TF v : c.c1 ) CHECK( v == 0 );
        for( TF v : c.c2 ) CHECK( v == 0 );
        for( TF v : c.c3 ) CHECK( v == 0 );
    }

    SECTION( "variable" ) {
        auto x = TS<3>::variable( 3, 1 );
        CHECK( x.c0 == 0 );
        CHECK( x.c1[0] == 0 );
        CHECK( x.c1[1] == 1 );
        CHECK( x.c1[2] == 0 );
        for( TF v : x.c2 ) CHECK( v == 0 );
        for( TF v : x.c3 ) CHECK( v == 0 );
    }
}

TEST_CASE( "TaylorScalar3 (1+x)^3 = 1 + 3x + 3x² + x³", "[TS3]" ) {
    // a = 1+x
    auto a = TS<3>::constant( 1, 1.0 ) + TS<3>::variable( 1, 0 );
    auto a3 = a * a * a;

    CHECK( std::abs( a3.c0           - 1 ) < eps );
    CHECK( std::abs( a3.c1[0]        - 3 ) < eps );
    CHECK( std::abs( a3.d2(0,0)      - 3 ) < eps );
    CHECK( std::abs( a3.d3(0,0,0)    - 1 ) < eps );
}

TEST_CASE( "TaylorScalar3 (1+x+y)^2 — two variables", "[TS3]" ) {
    auto x = TS<3>::variable( 2, 0 );
    auto y = TS<3>::variable( 2, 1 );
    auto a = TS<3>::constant( 2, 1.0 ) + x + y;
    auto a2 = a * a;

    // (1+x+y)² = 1 + 2x + 2y + x² + 2xy + y²
    CHECK( std::abs( a2.c0      - 1 ) < eps );
    CHECK( std::abs( a2.c1[0]   - 2 ) < eps );
    CHECK( std::abs( a2.c1[1]   - 2 ) < eps );
    // x², xy, yx, y²
    CHECK( std::abs( a2.d2(0,0) - 1 ) < eps );
    CHECK( std::abs( a2.d2(0,1) - 1 ) < eps );
    CHECK( std::abs( a2.d2(1,0) - 1 ) < eps );
    CHECK( std::abs( a2.d2(1,1) - 1 ) < eps );
    // order 3: (1+x+y)² has no cubic terms
    for( TF v : a2.c3 ) CHECK( std::abs( v ) < eps );
}

TEST_CASE( "TaylorScalar3 commutativity a*b vs b*a", "[TS3]" ) {
    // a*b and b*a represent the same polynomial — verify at test points
    auto x = TS<3>::variable( 2, 0 );  // x
    auto y = TS<3>::variable( 2, 1 );  // y

    auto ab = x * y;
    auto ba = y * x;

    // evaluate at several points
    for( auto p : std::vector<std::vector<TF>>{ {0.3, 0.7}, {-1.0, 2.0}, {0.0, 1.5} } )
        CHECK( std::abs( eval3(ab, p) - eval3(ba, p) ) < eps );
}

TEST_CASE( "TaylorScalar3 addition and subtraction", "[TS3]" ) {
    auto x = TS<3>::variable( 2, 0 );
    auto y = TS<3>::variable( 2, 1 );
    auto f = x + y;
    auto g = x - y;

    CHECK( std::abs( f.c1[0] - 1 ) < eps );
    CHECK( std::abs( f.c1[1] - 1 ) < eps );
    CHECK( std::abs( g.c1[0] - 1 ) < eps );
    CHECK( std::abs( g.c1[1] + 1 ) < eps );

    // (x+y) + (x-y) = 2x
    auto h = f + g;
    CHECK( std::abs( h.c1[0] - 2 ) < eps );
    CHECK( std::abs( h.c1[1]      ) < eps );
}

TEST_CASE( "TaylorScalar3 scalar ops", "[TS3]" ) {
    auto x = TS<3>::variable( 1, 0 );
    auto f = TF(3) * x + x * TF(2);  // 5x

    CHECK( std::abs( f.c0      ) < eps );
    CHECK( std::abs( f.c1[0] - 5 ) < eps );

    auto g = f / TF(5);  // x
    CHECK( std::abs( g.c1[0] - 1 ) < eps );
}

TEST_CASE( "TaylorScalar3 division (1+x)^3 / (1+x) = (1+x)^2", "[TS3]" ) {
    auto a  = TS<3>::constant( 1, 1.0 ) + TS<3>::variable( 1, 0 );
    auto a2 = a * a;
    auto a3 = a2 * a;

    auto r = a3 / a;

    CHECK( std::abs( r.c0      - 1 ) < eps );
    CHECK( std::abs( r.c1[0]   - 2 ) < eps );
    CHECK( std::abs( r.d2(0,0) - 1 ) < eps );
    for( TF v : r.c3 ) CHECK( std::abs( v ) < eps );
}

TEST_CASE( "TaylorScalar3 TF/TS : 1/(1+x)", "[TS3]" ) {
    // 1/(1+x) = 1 - x + x² - x³ + …
    auto a = TS<3>::constant( 1, 1.0 ) + TS<3>::variable( 1, 0 );
    auto r = TF(1) / a;

    CHECK( std::abs( r.c0        -  1 ) < eps );
    CHECK( std::abs( r.c1[0]     - -1 ) < eps );
    CHECK( std::abs( r.d2(0,0)   -  1 ) < eps );
    CHECK( std::abs( r.d3(0,0,0) - -1 ) < eps );
}

TEST_CASE( "TaylorScalar3 unary minus", "[TS3]" ) {
    auto x = TS<3>::variable( 2, 0 );
    auto y = TS<3>::variable( 2, 1 );
    auto f = x * y;
    auto g = -f;

    CHECK( g.c0 == 0 );
    for( int i = 0; i < 2; ++i ) CHECK( std::abs( g.c1[i] ) < eps );
    // every c2/c3 coefficient is negated
    for( int i = 0; i < 4; ++i ) CHECK( std::abs( g.c2[i] + f.c2[i] ) < eps );
}

TEST_CASE( "TaylorScalar3 sqrt( (1+x)^2 ) = 1+x", "[TS3]" ) {
    auto a  = TS<3>::constant( 1, 1.0 ) + TS<3>::variable( 1, 0 );
    auto a2 = a * a;
    auto r  = sqrt( a2 );

    CHECK( std::abs( r.c0        - 1 ) < eps );
    CHECK( std::abs( r.c1[0]     - 1 ) < eps );
    CHECK( std::abs( r.d2(0,0)       ) < eps );
    CHECK( std::abs( r.d3(0,0,0)     ) < eps );
}

TEST_CASE( "TaylorScalar3 sqrt( 1 + x^2 ) coefficients", "[TS3]" ) {
    // sqrt(1+x²) = 1 + x²/2 - x⁴/8 + …  (order-3 term vanishes)
    auto x  = TS<3>::variable( 1, 0 );
    auto a  = TS<3>::constant( 1, 1.0 ) + x * x;
    auto r  = sqrt( a );

    CHECK( std::abs( r.c0        - 1.0 ) < eps );
    CHECK( std::abs( r.c1[0]           ) < eps );  // odd → 0
    CHECK( std::abs( r.d2(0,0)   - 0.5 ) < eps );
    CHECK( std::abs( r.d3(0,0,0)       ) < eps );  // odd → 0
}

TEST_CASE( "TaylorScalar3 TS/TS general division", "[TS3]" ) {
    auto x = TS<3>::variable( 2, 0 );
    auto y = TS<3>::variable( 2, 1 );
    // f = 1 + x + y,  g = 1 + 2x + y,  h = f/g * g should recover f
    auto f = TS<3>::constant( 2, 1.0 ) + x + y;
    auto g = TS<3>::constant( 2, 1.0 ) + TF(2)*x + y;
    auto h = ( f / g ) * g;

    for( auto p : std::vector<std::vector<TF>>{ {0.1, 0.2}, {-0.1, 0.3} } )
        CHECK( std::abs( eval3(h, p) - eval3(f, p) ) < 1e-10 );
}

// ── order-4 tests ─────────────────────────────────────────────────────────────

TEST_CASE( "TaylorScalar4 construction", "[TS4]" ) {
    SECTION( "constant" ) {
        auto c = TS<4>::constant( 2, 5.0 );
        CHECK( c.c0 == 5.0 );
        for( TF v : c.c1 ) CHECK( v == 0 );
        for( TF v : c.c2 ) CHECK( v == 0 );
        for( TF v : c.c3 ) CHECK( v == 0 );
        for( TF v : c.c4 ) CHECK( v == 0 );
    }

    SECTION( "variable" ) {
        auto y = TS<4>::variable( 3, 2 );
        CHECK( y.c0 == 0 );
        CHECK( y.c1[0] == 0 );
        CHECK( y.c1[1] == 0 );
        CHECK( y.c1[2] == 1 );
        for( TF v : y.c2 ) CHECK( v == 0 );
        for( TF v : y.c3 ) CHECK( v == 0 );
        for( TF v : y.c4 ) CHECK( v == 0 );
    }
}

TEST_CASE( "TaylorScalar4 (1+x)^4 = 1 + 4x + 6x² + 4x³ + x⁴", "[TS4]" ) {
    auto a  = TS<4>::constant( 1, 1.0 ) + TS<4>::variable( 1, 0 );
    auto sq = a * a;
    auto a4 = sq * sq;

    CHECK( std::abs( a4.c0          - 1 ) < eps );
    CHECK( std::abs( a4.c1[0]       - 4 ) < eps );
    CHECK( std::abs( a4.d2(0,0)     - 6 ) < eps );
    CHECK( std::abs( a4.d3(0,0,0)   - 4 ) < eps );
    CHECK( std::abs( a4.d4(0,0,0,0) - 1 ) < eps );
}

TEST_CASE( "TaylorScalar4 commutativity a*b vs b*a", "[TS4]" ) {
    auto x = TS<4>::variable( 2, 0 );
    auto y = TS<4>::variable( 2, 1 );
    auto ab = x * y;
    auto ba = y * x;

    for( auto p : std::vector<std::vector<TF>>{ {0.4, 0.6}, {-0.5, 1.2} } )
        CHECK( std::abs( eval4(ab, p) - eval4(ba, p) ) < eps );
}

TEST_CASE( "TaylorScalar4 TF/TS : 0.5/(1+x)", "[TS4]" ) {
    // 0.5/(1+x) = 0.5 - 0.5x + 0.5x² - 0.5x³ + 0.5x⁴ + …
    auto a = TS<4>::constant( 1, 1.0 ) + TS<4>::variable( 1, 0 );
    auto r = TF(0.5) / a;

    CHECK( std::abs( r.c0          -  0.5 ) < eps );
    CHECK( std::abs( r.c1[0]       - -0.5 ) < eps );
    CHECK( std::abs( r.d2(0,0)     -  0.5 ) < eps );
    CHECK( std::abs( r.d3(0,0,0)   - -0.5 ) < eps );
    CHECK( std::abs( r.d4(0,0,0,0) -  0.5 ) < eps );
}

TEST_CASE( "TaylorScalar4 division (1+x)^4 / (1+x)^2 = (1+x)^2", "[TS4]" ) {
    auto a  = TS<4>::constant( 1, 1.0 ) + TS<4>::variable( 1, 0 );
    auto a2 = a * a;
    auto a4 = a2 * a2;
    auto r  = a4 / a2;

    CHECK( std::abs( r.c0        - 1 ) < eps );
    CHECK( std::abs( r.c1[0]     - 2 ) < eps );
    CHECK( std::abs( r.d2(0,0)   - 1 ) < eps );
    for( TF v : r.c3 ) CHECK( std::abs( v ) < eps );
    for( TF v : r.c4 ) CHECK( std::abs( v ) < eps );
}

TEST_CASE( "TaylorScalar4 sqrt( (1+x)^2 ) = 1+x", "[TS4]" ) {
    auto a  = TS<4>::constant( 1, 1.0 ) + TS<4>::variable( 1, 0 );
    auto a2 = a * a;
    auto r  = sqrt( a2 );

    CHECK( std::abs( r.c0          - 1 ) < eps );
    CHECK( std::abs( r.c1[0]       - 1 ) < eps );
    CHECK( std::abs( r.d2(0,0)         ) < eps );
    CHECK( std::abs( r.d3(0,0,0)       ) < eps );
    CHECK( std::abs( r.d4(0,0,0,0)     ) < eps );
}

TEST_CASE( "TaylorScalar4 sqrt( 1 + x^2 ) = 1 + x²/2 - x⁴/8", "[TS4]" ) {
    // Taylor: 1 + ½x² - ⅛x⁴
    auto x = TS<4>::variable( 1, 0 );
    auto a = TS<4>::constant( 1, 1.0 ) + x * x;
    auto r = sqrt( a );

    CHECK( std::abs( r.c0          - 1.0    ) < eps );
    CHECK( std::abs( r.c1[0]               ) < eps );
    CHECK( std::abs( r.d2(0,0)     - 0.5   ) < eps );
    CHECK( std::abs( r.d3(0,0,0)           ) < eps );
    CHECK( std::abs( r.d4(0,0,0,0) + 0.125 ) < eps );
}

TEST_CASE( "TaylorScalar4 (x+y)^4 — two variables, all c4[i,j,k,l]=1", "[TS4]" ) {
    // (x+y)^4: the coefficient of every monomial x^a y^b (a+b=4) in the
    // tensor c4[i,j,k,l] (i,j,k,l ∈ {0,1}) equals 1.
    auto x  = TS<4>::variable( 2, 0 );
    auto y  = TS<4>::variable( 2, 1 );
    auto a  = x + y;
    auto a4 = a * a * a * a;

    CHECK( std::abs( a4.c0 ) < eps );
    for( TF v : a4.c1 ) CHECK( std::abs( v ) < eps );
    for( TF v : a4.c2 ) CHECK( std::abs( v ) < eps );
    for( TF v : a4.c3 ) CHECK( std::abs( v ) < eps );
    for( TF v : a4.c4 ) CHECK( std::abs( v - 1 ) < eps );
}

TEST_CASE( "TaylorScalar4 += -= *= /= self-assignment", "[TS4]" ) {
    auto a = TS<4>::constant( 1, 3.0 ) + TS<4>::variable( 1, 0 );  // 3+x
    auto b = TS<4>::constant( 1, 1.0 ) + TS<4>::variable( 1, 0 );  // 1+x

    auto r = a;
    r += b;  // 4+2x
    CHECK( std::abs( r.c0 - 4 ) < eps );
    CHECK( std::abs( r.c1[0] - 2 ) < eps );

    r = a;
    r -= b;  // 2
    CHECK( std::abs( r.c0 - 2 ) < eps );
    CHECK( std::abs( r.c1[0] ) < eps );

    r = a;
    r *= b;  // (3+x)(1+x) = 3 + 4x + x²
    CHECK( std::abs( r.c0 - 3 ) < eps );
    CHECK( std::abs( r.c1[0] - 4 ) < eps );
    CHECK( std::abs( r.d2(0,0) - 1 ) < eps );

    r = a;
    r /= b;  // (3+x)/(1+x)
    auto expected = a / b;
    for( int i = 0; i < (int)r.c1.size(); ++i )
        CHECK( std::abs( r.c1[i] - expected.c1[i] ) < eps );
}
