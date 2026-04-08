#pragma once

#include "../support/common_types.h"
#include "../support/ASSERT.h"
#include <vector>
#include <cmath>

namespace sdot {

/**
 * Multivariate Taylor polynomial truncated at order 3, with n variables.
 *
 * Construction: use constant(n, val) or variable(n, k).
 * TF operands are handled via explicit TF overloads — no implicit conversion.
 * Binary TaylorScalar×TaylorScalar operations require equal n (ASSERTed).
 */
template<class TF>
struct TaylorScalar {
    TF              c0;
    std::vector<TF> c1;  // size n
    std::vector<TF> c2;  // size n²
    std::vector<TF> c3;  // size n³

    PI n () const { return c1.size(); }

    PI n1() const { return n(); }
    PI n2() const { return n() * n(); }
    PI n3() const { return n() * n() * n(); }

    // ── constructors ──────────────────────────────────────────────────────────

    TaylorScalar( TF value = 0 ) : c0( value ) {}

    static TaylorScalar constant( int n, TF value ) {
        TaylorScalar r;
        r.c0 = value;
        r.c1.resize( n    , TF( 0 ) );
        r.c2.resize( n*n  , TF( 0 ) );
        r.c3.resize( n*n*n, TF( 0 ) );
        return r;
    }

    static TaylorScalar empty( int n, TF value ) {
        TaylorScalar r;
        r.c0 = value;
        r.c1.resize( n     );
        r.c2.resize( n*n   );
        r.c3.resize( n*n*n );
        return r;
    }

    static TaylorScalar variable( int n, int k ) {
        TaylorScalar r = constant( n, TF( 0 ) );
        r.c1[ k ] = TF( 1 );
        return r;
    }

    // ── indexed access ────────────────────────────────────────────────────────

    TF   d3( int i, int j, int k ) const { return c3[ i * n2() + j * n1() + k ]; }
    TF   d2( int i, int j ) const        { return c2[ i * n1() + j ]; }
    TF   d1( int i ) const               { return c2[ i ]; }

    TF&  d3( int i, int j, int k )       { return c3[ i * n2() + j * n1() + k ]; }
    TF&  d2( int i, int j )              { return c2[ i * n1() + j ]; }
    TF&  d1( int i )                     { return c2[ i ]; }

    // ── unary minus ───────────────────────────────────────────────────────────

    friend TaylorScalar operator-( const TaylorScalar &a ) {
        TaylorScalar r = empty( a.n1(), - a.c0 );
        for( int i = 0; i < a.n1(); ++i ) r.c1[ i ] = - a.c1[ i ];
        for( int i = 0; i < a.n2(); ++i ) r.c2[ i ] = - a.c2[ i ];
        for( int i = 0; i < a.n3(); ++i ) r.c3[ i ] = - a.c3[ i ];
        return r;
    }

    // ── addition ──────────────────────────────────────────────────────────────

    friend TaylorScalar operator+( const TaylorScalar &a, TF b ) {
        TaylorScalar r = a;
        r.c0 += b;
        return r;
    }

    friend TaylorScalar operator+( TF a, const TaylorScalar &b ) {
        TaylorScalar r = b;
        r.c0 += a;
        return r;
    }

    friend TaylorScalar operator+( const TaylorScalar &a, const TaylorScalar &b ) {
        if ( a.n() == 0 ) return operator+( a.c0, b );
        if ( b.n() == 0 ) return operator+( a, b.c0 );

        ASSERT( a.n() == b.n() );
        TaylorScalar r = empty( a.n(), a.c0 + b.c0 );
        for( int i = 0; i < a.n1(); ++i ) r.c1[ i ] = a.c1[ i ] + b.c1[ i ];
        for( int i = 0; i < a.n2(); ++i ) r.c2[ i ] = a.c2[ i ] + b.c2[ i ];
        for( int i = 0; i < a.n3(); ++i ) r.c3[ i ] = a.c3[ i ] + b.c3[ i ];
        return r;
    }

    // ── subtraction ───────────────────────────────────────────────────────────

    friend TaylorScalar operator-( const TaylorScalar &a, TF b ) {
        TaylorScalar r = a;
        r.c0 -= b;
        return r;
    }

    friend TaylorScalar operator-( TF a, const TaylorScalar &b ) {
        TaylorScalar r = - b;
        r.c0 += a;
        return r;
    }

    friend TaylorScalar operator-( const TaylorScalar &a, const TaylorScalar &b ) {
        if ( a.n() == 0 ) return operator-( a.c0, b );
        if ( b.n() == 0 ) return operator-( a, b.c0 );

        ASSERT( a.n() == b.n() );
        TaylorScalar r = empty( a.n(), a.c0 - b.c0 );
        for( int i = 0; i < a.n1(); ++i ) r.c1[ i ] = a.c1[ i ] - b.c1[ i ];
        for( int i = 0; i < a.n2(); ++i ) r.c2[ i ] = a.c2[ i ] - b.c2[ i ];
        for( int i = 0; i < a.n3(); ++i ) r.c3[ i ] = a.c3[ i ] - b.c3[ i ];
        return r;
    }

    // ── multiplication ────────────────────────────────────────────────────────
    friend TaylorScalar operator*( const TaylorScalar &a, TF b ) {
        TaylorScalar r = empty( a.n(), a.c0 * b );
        for( int i = 0; i < a.n1(); ++i ) r.c1[ i ] = a.c1[ i ] * b;
        for( int i = 0; i < a.n2(); ++i ) r.c2[ i ] = a.c2[ i ] * b;
        for( int i = 0; i < a.n3(); ++i ) r.c3[ i ] = a.c3[ i ] * b;
        return r;
    }

    friend TaylorScalar operator*( TF a, const TaylorScalar &b ) {
        TaylorScalar r = empty( b.n(), a * b.c0 );
        for( int i = 0; i < b.n1(); ++i ) r.c1[ i ] = a * b.c1[ i ];
        for( int i = 0; i < b.n2(); ++i ) r.c2[ i ] = a * b.c2[ i ];
        for( int i = 0; i < b.n3(); ++i ) r.c3[ i ] = a * b.c3[ i ];
        return r;
    }

    friend TaylorScalar operator*( const TaylorScalar &a, const TaylorScalar &b ) {
        if ( a.n() == 0 ) return operator*( a.c0, b );
        if ( b.n() == 0 ) return operator*( a, b.c0 );

        ASSERT( a.n() == b.n() );
        TaylorScalar r = empty( a.n(), a.c0 * b.c0 );
        for( int i = 0; i < a.n(); ++i )
            r.c1[ i ] = a.c0 * b.c1[ i ]  +  a.c1[ i ] * b.c0;
        for( int i = 0; i < a.n(); ++i )
            for( int j = 0; j < a.n(); ++j )
                r.d2( i, j ) = a.c0 * b.d2( i, j ) + a.c1[ i ] * b.c1[ j ] + a.d2( i, j ) * b.c0;
        for( int i = 0; i < a.n(); ++i )
            for( int j = 0; j < a.n(); ++j )
                for( int k = 0; k < a.n(); ++k )
                    r.d3( i , j , k ) = a.c0 * b.d3( i, j, k )
                                      + a.c1[ i ] * b.d2( j, k )
                                      + a.d2( i, j ) * b.c1[ k ]
                                      + a.d3( i, j, k ) * b.c0;
        return r;
    }


    // ── division ──────────────────────────────────────────────────────────────

    friend TaylorScalar operator/( const TaylorScalar &a, TF s ) {
        return a * ( TF( 1 ) / s );
    }

    friend TaylorScalar operator/( TF s, const TaylorScalar& a ) {
        const TF inv_a0 = TF( 1 ) / a.c0;
        TaylorScalar r = empty( a.n(), s * inv_a0 );
        for( int i = 0; i < a.n(); ++i )
            r.c1[i] = -r.c0 * a.c1[ i ] * inv_a0;
        for( int i = 0; i < a.n(); ++i )
            for( int j = 0; j < a.n(); ++j )
                r.d2( i, j ) = ( - r.c0 * a.d2( i, j )  -  r.c1[ i ] * a.c1[ j ] ) * inv_a0;
        for( int i = 0; i < a.n(); ++i )
            for( int j = 0; j < a.n(); ++j )
                for( int k = 0; k < a.n(); ++k )
                    r.d3(i,j,k) = ( -r.c0 * a.d3( i, j, k )
                                    -r.c1[ i ]* a.d2( j, k )
                                    -r.d2( i, j ) * a.c1[ k ] ) * inv_a0;
        return r;
    }

    friend TaylorScalar operator/( const TaylorScalar& a, const TaylorScalar& b ) {
        if ( a.n() == 0 ) return operator/( a.c0, b );
        if ( b.n() == 0 ) return operator/( a, b.c0 );

        ASSERT( a.n() == b.n() );
        const TF inv_b0 = TF( 1 ) / b.c0;
        TaylorScalar r = empty( a.n(), a.c0 * inv_b0 );
        for( int i = 0; i < a.n(); ++i )
            r.c1[ i ] = ( a.c1[i]  -  r.c0 * b.c1[ i ] ) * inv_b0;
        for( int i = 0; i < a.n(); ++i )
            for( int j = 0; j < a.n(); ++j )
                r.d2(i,j) = ( a.d2( i, j ) - r.c0 * b.d2( i, j )  -  r.c1[ i ] * b.c1[ j ] ) * inv_b0;
        for( int i = 0; i < a.n(); ++i )
            for( int j = 0; j < a.n(); ++j )
                for( int k = 0; k < a.n(); ++k )
                    r.d3( i, j, k ) = ( a.d3(i,j,k)
                                    - r.c0 * b.d3( i, j, k )
                                    - r.c1[ i ] * b.d2( j, k )
                                    - r.d2( i, j ) * b.c1[ k ] ) * inv_b0;
        return r;
    }


    // ── self op ───────────────────────────────────────────────────────────

    TaylorScalar& operator+=( const TaylorScalar& b ) { return *this = *this + b; }
    TaylorScalar& operator+=( TF s ) { c0 += s; return *this; }

    TaylorScalar& operator-=( const TaylorScalar& b ) { return *this = *this - b; }
    TaylorScalar& operator-=( TF s ) { c0 -= s; return *this; }

    TaylorScalar& operator*=( const TaylorScalar& b ) { return *this = *this * b; }
    TaylorScalar& operator*=( TF s ) { return *this = *this * s; }

    TaylorScalar& operator/=( const TaylorScalar& b ) { return *this = *this / b; }
    TaylorScalar& operator/=( TF s ) { return *this = *this / s; }

    // ── math functions ────────────────────────────────────────────────────────

    friend TaylorScalar sqrt( const TaylorScalar& a ) {
        TaylorScalar r = constant( a.n(), std::sqrt( a.c0 ) );
        const TF inv2r0 = TF(1) / ( TF(2) * r.c0 );
        for( int i = 0; i < a.n(); ++i )
            r.c1[i] = a.c1[i] * inv2r0;
        for( int i = 0; i < a.n(); ++i )
            for( int j = 0; j < a.n(); ++j )
                r.d2(i,j) = ( a.d2(i,j) - r.c1[i] * r.c1[j] ) * inv2r0;
        for( int i = 0; i < a.n(); ++i )
            for( int j = 0; j < a.n(); ++j )
                for( int k = 0; k < a.n(); ++k )
                    r.d3( i, j, k ) = ( a.d3( i, j, k )
                                  - r.c1[i]  * r.d2(j,k)
                                  - r.d2(i,j) * r.c1[k] ) * inv2r0;
        return r;
    }

    // ── comparisons (on c0 only) ──────────────────────────────────────────────

    bool operator> ( TF v ) const  { return c0 >  v; }
    bool operator< ( TF v ) const  { return c0 <  v; }
    bool operator>=( TF v ) const { return c0 >= v; }
    bool operator<=( TF v ) const { return c0 <= v; }
    bool operator==( TF v ) const { return c0 == v; }


    friend std::ostream &operator<<( std::ostream &os, const TaylorScalar &p ) {
        os << p.c0;
        os << ", [" << p.c1 << "]";
        os << ", [" << p.c2 << "]";
        os << ", [" << p.c3 << "]";
        return os;
    }

};

} // namespace sdot
