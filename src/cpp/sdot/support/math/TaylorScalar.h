#pragma once

#include "SimpleSquareMatrix.h"
#include "common_types.h"
#include "ASSERT.h"
#include "TODO.h"
#include <vector>
#include <random>
#include <cmath>

namespace sdot {

template<class TF,int order>
struct TaylorScalar;

/**
 * Multivariate Taylor polynomial truncated at order 3, with n variables.
 *
 * Construction: use constant(n, val) or variable(n, k).
 * TF operands are handled via explicit TF overloads — no implicit conversion.
 * Binary TaylorScalar×TaylorScalar operations require equal n (ASSERTed).
 */
template<class TF>
struct TaylorScalar<TF,3> {
    TF              c0;
    std::vector<TF> c1;  // size n
    std::vector<TF> c2;  // size n²
    std::vector<TF> c3;  // size n³

    PI n () const { return c1.size(); }

    PI n1() const { return c1.size(); }
    PI n2() const { return c2.size(); }
    PI n3() const { return c3.size(); }

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
    std::vector<TF> argmin() const {
        if ( n() == 0 )
            return { TF( 0 ) };
        if ( n() != 1 )
            TODO;
        return { - c1[ 0 ] / ( 2 * c2[ 0 ] ) };
    }

    // ── indexed access ────────────────────────────────────────────────────────

    TF   d3( int i, int j, int k ) const { return c3[ i * n2() + j * n1() + k ]; }
    TF   d2( int i, int j ) const        { return c2[ i * n1() + j ]; }
    TF   d1( int i ) const               { return c2[ i ]; }

    TF&  d3( int i, int j, int k )       { return c3[ i * n2() + j * n1() + k ]; }
    TF&  d2( int i, int j )              { return c2[ i * n1() + j ]; }
    TF&  d1( int i )                     { return c2[ i ]; }


    // ── positive region ───────────────────────────────────────────────────────

    /**
     * Add cuts to `cell` so it is restricted to the region where this
     * polynomial (order-2 part only) is non-negative.
     *
     * The zero-set of the quadratic approximation is an ellipsoid around the
     * minimiser x*. Each tangent plane of that ellipsoid that does not remove
     * the origin is added as a half-space cut (cell.cut keeps dir·x ≤ dot).
     *
     * n_cuts: number of tangent planes (or sphere samples for n > 2).
     */
    void update_bounds_to_stay_positive( auto &cell, PI n_cuts = 24 ) const {
        using Mat = Matrix<TF, -1, Cpu>;
        using Vec = Vector<TF,Arch, -1, Cpu>;
        using Pt  = std::remove_reference_t<decltype(cell)>::Pt;

        const PI nd = n();
        if ( nd == 0 ) return;

        // ── full symmetric Hessian H = C2 + C2^T ────────────────────────────
        Mat H( nd );
        for ( PI i = 0; i < nd; ++i )
            for ( PI j = 0; j < nd; ++j )
                H( i, j ) = d2( i, j ) + d2( j, i );

        // ── quadratic minimiser x* = H^{-1}(-c1) ────────────────────────────
        Vector neg_c1( Size(), nd );
        for ( PI i = 0; i < nd; ++i )
            neg_c1[ i ] = -c1[ i ];
        Vector x_star = H.solve( neg_c1 );

        // ── f(x*) = c0 + ½ c1·x*  (since H x* = -c1) ───────────────────────
        TF f_star = c0;
        for ( PI i = 0; i < nd; ++i )
            f_star += TF(0.5) * c1[ i ] * x_star[ i ];
        if ( f_star >= TF(0) )
            return;   // polynomial is always non-negative

        const TF r = std::sqrt( -TF(2) * f_star );

        // ── Cholesky: H = L L^T  (H is SPD since we are at a strict minimum) ─
        Mat L = H.cholesky();

        // normal of ellipsoid at boundary point parameterised by unit u:
        //   x(u) = x* + r L^{-T} u   =>   ∇f = H(x-x*) = r L u
        auto compute_Lu = [&]( const Vec &u ) {
            Vector Lu( Size(), nd );
            for ( PI i = 0; i < nd; ++i ) {
                TF s = TF(0);
                for ( PI k = 0; k <= i; ++k )
                    s += L( i, k ) * u[ k ];
                Lu[ i ] = s;
            }
            return Lu;
        };

        // ── candidate unit directions on S^{n-1} ────────────────────────────
        std::vector<Vec> candidates;
        if ( nd == 1 ) {
            Vector u1( Size(), 1 ), u2( Size(), 1 );
            u1[ 0 ] = TF( +1 );
            u2[ 0 ] = TF( -1 );
            candidates = { u1, u2 };
        } else if ( nd == 2 ) {
            constexpr TF pi2 = TF( 6.283185307179586 );
            for ( PI k = 0; k < n_cuts; ++k ) {
                TF theta = pi2 * k / n_cuts;
                Vector u( Size(), 2 );
                u[ 0 ] = std::cos( theta );
                u[ 1 ] = std::sin( theta );
                candidates.push_back( u );
            }
        } else {
            std::mt19937_64 rng( 0 );
            std::normal_distribution<double> nd_dist;
            for ( PI k = 0; k < n_cuts; ++k ) {
                Vector u( Size(), nd );
                double norm = 0;
                for ( PI d = 0; d < nd; ++d ) {
                    double v = nd_dist( rng );
                    u[ d ] = TF( v );
                    norm += v * v;
                }
                norm = std::sqrt( norm );
                for ( PI d = 0; d < nd; ++d ) u[ d ] /= TF( norm );
                candidates.push_back( u );
            }
        }

        // ── one cut per candidate direction (if it does not remove the origin)
        for ( auto &u : candidates ) {
            Vec Lu = compute_Lu( u );

            TF Lu_dot_xstar = TF( 0 );
            for ( PI d = 0; d < nd; ++d )
                Lu_dot_xstar += Lu[ d ] * x_star[ d ];
            const TF dot_val = -Lu_dot_xstar - r;   // (-Lu)·x(u) = -(Lu·x*) - r

            if ( dot_val < TF(0) )
                continue; // would remove the origin

            Pt dir( Size(), nd );
            for ( PI d = 0; d < nd; ++d )
                dir[ d ] = -Lu[ d ];
            cell.cut( dir, dot_val, { nd } );
        }
    }

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

    bool operator> ( TF v ) const { return c0 >  v; }
    bool operator< ( TF v ) const { return c0 <  v; }
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


/**
 * Multivariate Taylor polynomial truncated at order 4, with n variables.
 */
template<class TF>
struct TaylorScalar<TF,4> {
    TF              c0;
    std::vector<TF> c1;  // size n
    std::vector<TF> c2;  // size n²
    std::vector<TF> c3;  // size n³
    std::vector<TF> c4;  // size n⁴

    PI n () const { return c1.size(); }

    PI n1() const { return c1.size(); }
    PI n2() const { return c2.size(); }
    PI n3() const { return c3.size(); }
    PI n4() const { return c4.size(); }

    // ── constructors ──────────────────────────────────────────────────────────

    TaylorScalar( TF value = 0 ) : c0( value ) {}

    static TaylorScalar constant( int n, TF value ) {
        TaylorScalar r;
        r.c0 = value;
        r.c1.resize( n        , TF( 0 ) );
        r.c2.resize( n*n      , TF( 0 ) );
        r.c3.resize( n*n*n    , TF( 0 ) );
        r.c4.resize( n*n*n*n  , TF( 0 ) );
        return r;
    }

    static TaylorScalar empty( int n, TF value ) {
        TaylorScalar r;
        r.c0 = value;
        r.c1.resize( n       );
        r.c2.resize( n*n     );
        r.c3.resize( n*n*n   );
        r.c4.resize( n*n*n*n );
        return r;
    }

    static TaylorScalar variable( int n, int k ) {
        TaylorScalar r = constant( n, TF( 0 ) );
        r.c1[ k ] = TF( 1 );
        return r;
    }

    // ── indexed access ────────────────────────────────────────────────────────

    TF   d4( int i, int j, int k, int l ) const { return c4[ i * n3() + j * n2() + k * n1() + l ]; }
    TF   d3( int i, int j, int k ) const        { return c3[ i * n2() + j * n1() + k ]; }
    TF   d2( int i, int j ) const               { return c2[ i * n1() + j ]; }

    TF&  d4( int i, int j, int k, int l )       { return c4[ i * n3() + j * n2() + k * n1() + l ]; }
    TF&  d3( int i, int j, int k )              { return c3[ i * n2() + j * n1() + k ]; }
    TF&  d2( int i, int j )                     { return c2[ i * n1() + j ]; }

    // ── positive region ───────────────────────────────────────────────────────

    /// Same algorithm as TaylorScalar<TF,3>::update_bounds_to_stay_positive —
    /// uses only c0, c1, d2 (order-2 quadratic approximation).
    void update_bounds_to_stay_positive( auto &cell, PI n_cuts = 24 ) const {
        using Mat = Matrix<TF, -1, Cpu>;
        using Vec = Vector<TF,Arch, -1, Cpu>;
        using Pt  = std::remove_reference_t<decltype(cell)>::Pt;

        const PI nd = n();
        if ( nd == 0 ) return;

        Mat H( nd );
        for ( PI i = 0; i < nd; ++i )
            for ( PI j = 0; j < nd; ++j )
                H( i, j ) = d2( i, j ) + d2( j, i );

        Vector neg_c1( nd );
        for ( PI i = 0; i < nd; ++i ) neg_c1[ i ] = -c1[ i ];
        Vector x_star = H.solve( neg_c1 );

        TF f_star = c0;
        for ( PI i = 0; i < nd; ++i ) f_star += TF(0.5) * c1[ i ] * x_star[ i ];
        if ( f_star >= TF(0) ) return;

        const TF r = std::sqrt( -TF(2) * f_star );
        Mat L = H.cholesky();

        auto compute_Lu = [&]( const Vec &u ) {
            Vector Lu( nd );
            for ( PI i = 0; i < nd; ++i ) {
                TF s = TF(0);
                for ( PI k = 0; k <= i; ++k ) s += L( i, k ) * u[ k ];
                Lu[ i ] = s;
            }
            return Lu;
        };

        std::vector<Vec> candidates;
        if ( nd == 1 ) {
            Vector u1( 1 ), u2( 1 );
            u1[ 0 ] = TF(  1 );
            u2[ 0 ] = TF( -1 );
            candidates = { u1, u2 };
        } else if ( nd == 2 ) {
            constexpr TF pi2 = TF( 6.283185307179586 );
            for ( PI k = 0; k < n_cuts; ++k ) {
                TF theta = pi2 * k / n_cuts;
                Vector u( 2 );
                u[ 0 ] = std::cos( theta );
                u[ 1 ] = std::sin( theta );
                candidates.push_back( u );
            }
        } else {
            std::mt19937_64 rng( 0 );
            std::normal_distribution<double> nd_dist;
            for ( PI k = 0; k < n_cuts; ++k ) {
                Vector u( nd );
                double norm = 0;
                for ( PI d = 0; d < nd; ++d ) {
                    double v = nd_dist( rng );
                    u[ d ] = TF( v );
                    norm += v * v;
                }
                norm = std::sqrt( norm );
                for ( PI d = 0; d < nd; ++d ) u[ d ] /= TF( norm );
                candidates.push_back( u );
            }
        }

        for ( auto &u : candidates ) {
            Vec Lu = compute_Lu( u );
            TF Lu_dot_xstar = TF(0);
            for ( PI d = 0; d < nd; ++d ) Lu_dot_xstar += Lu[ d ] * x_star[ d ];
            const TF dot_val = -Lu_dot_xstar - r;
            if ( dot_val < TF(0) ) continue;
            Pt dir( nd );
            for ( PI d = 0; d < nd; ++d ) dir[ d ] = -Lu[ d ];
            cell.cut( dir, dot_val );
        }
    }

    // ── unary minus ───────────────────────────────────────────────────────────

    friend TaylorScalar operator-( const TaylorScalar &a ) {
        TaylorScalar r = empty( a.n(), - a.c0 );
        for( int i = 0; i < a.n1(); ++i ) r.c1[ i ] = - a.c1[ i ];
        for( int i = 0; i < a.n2(); ++i ) r.c2[ i ] = - a.c2[ i ];
        for( int i = 0; i < a.n3(); ++i ) r.c3[ i ] = - a.c3[ i ];
        for( int i = 0; i < a.n4(); ++i ) r.c4[ i ] = - a.c4[ i ];
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
        for( int i = 0; i < a.n4(); ++i ) r.c4[ i ] = a.c4[ i ] + b.c4[ i ];
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
        for( int i = 0; i < a.n4(); ++i ) r.c4[ i ] = a.c4[ i ] - b.c4[ i ];
        return r;
    }

    // ── multiplication ────────────────────────────────────────────────────────

    friend TaylorScalar operator*( const TaylorScalar &a, TF b ) {
        TaylorScalar r = empty( a.n(), a.c0 * b );
        for( int i = 0; i < a.n1(); ++i ) r.c1[ i ] = a.c1[ i ] * b;
        for( int i = 0; i < a.n2(); ++i ) r.c2[ i ] = a.c2[ i ] * b;
        for( int i = 0; i < a.n3(); ++i ) r.c3[ i ] = a.c3[ i ] * b;
        for( int i = 0; i < a.n4(); ++i ) r.c4[ i ] = a.c4[ i ] * b;
        return r;
    }

    friend TaylorScalar operator*( TF a, const TaylorScalar &b ) {
        TaylorScalar r = empty( b.n(), a * b.c0 );
        for( int i = 0; i < b.n1(); ++i ) r.c1[ i ] = a * b.c1[ i ];
        for( int i = 0; i < b.n2(); ++i ) r.c2[ i ] = a * b.c2[ i ];
        for( int i = 0; i < b.n3(); ++i ) r.c3[ i ] = a * b.c3[ i ];
        for( int i = 0; i < b.n4(); ++i ) r.c4[ i ] = a * b.c4[ i ];
        return r;
    }

    friend TaylorScalar operator*( const TaylorScalar &a, const TaylorScalar &b ) {
        if ( a.n() == 0 ) return operator*( a.c0, b );
        if ( b.n() == 0 ) return operator*( a, b.c0 );

        ASSERT( a.n() == b.n() );
        TaylorScalar r = empty( a.n(), a.c0 * b.c0 );
        for( int i = 0; i < a.n(); ++i )
            r.c1[ i ] = a.c0 * b.c1[ i ] + a.c1[ i ] * b.c0;
        for( int i = 0; i < a.n(); ++i )
            for( int j = 0; j < a.n(); ++j )
                r.d2( i, j ) = a.c0 * b.d2( i, j ) + a.c1[ i ] * b.c1[ j ] + a.d2( i, j ) * b.c0;
        for( int i = 0; i < a.n(); ++i )
            for( int j = 0; j < a.n(); ++j )
                for( int k = 0; k < a.n(); ++k )
                    r.d3( i, j, k ) = a.c0 * b.d3( i, j, k )
                                    + a.c1[ i ] * b.d2( j, k )
                                    + a.d2( i, j ) * b.c1[ k ]
                                    + a.d3( i, j, k ) * b.c0;
        for( int i = 0; i < a.n(); ++i )
            for( int j = 0; j < a.n(); ++j )
                for( int k = 0; k < a.n(); ++k )
                    for( int l = 0; l < a.n(); ++l )
                        r.d4( i, j, k, l ) = a.c0 * b.d4( i, j, k, l )
                                           + a.c1[ i ] * b.d3( j, k, l )
                                           + a.d2( i, j ) * b.d2( k, l )
                                           + a.d3( i, j, k ) * b.c1[ l ]
                                           + a.d4( i, j, k, l ) * b.c0;
        return r;
    }

    // ── division ──────────────────────────────────────────────────────────────

    friend TaylorScalar operator/( const TaylorScalar &a, TF s ) {
        return a * ( TF( 1 ) / s );
    }

    friend TaylorScalar operator/( TF s, const TaylorScalar &a ) {
        const TF inv_a0 = TF( 1 ) / a.c0;
        TaylorScalar r = empty( a.n(), s * inv_a0 );
        for( int i = 0; i < a.n(); ++i )
            r.c1[ i ] = -r.c0 * a.c1[ i ] * inv_a0;
        for( int i = 0; i < a.n(); ++i )
            for( int j = 0; j < a.n(); ++j )
                r.d2( i, j ) = ( -r.c0 * a.d2( i, j ) - r.c1[ i ] * a.c1[ j ] ) * inv_a0;
        for( int i = 0; i < a.n(); ++i )
            for( int j = 0; j < a.n(); ++j )
                for( int k = 0; k < a.n(); ++k )
                    r.d3( i, j, k ) = ( -r.c0 * a.d3( i, j, k )
                                        -r.c1[ i ] * a.d2( j, k )
                                        -r.d2( i, j ) * a.c1[ k ] ) * inv_a0;
        for( int i = 0; i < a.n(); ++i )
            for( int j = 0; j < a.n(); ++j )
                for( int k = 0; k < a.n(); ++k )
                    for( int l = 0; l < a.n(); ++l )
                        r.d4( i, j, k, l ) = ( -r.c0 * a.d4( i, j, k, l )
                                               -r.c1[ i ] * a.d3( j, k, l )
                                               -r.d2( i, j ) * a.d2( k, l )
                                               -r.d3( i, j, k ) * a.c1[ l ] ) * inv_a0;
        return r;
    }

    friend TaylorScalar operator/( const TaylorScalar &a, const TaylorScalar &b ) {
        if ( a.n() == 0 ) return operator/( a.c0, b );
        if ( b.n() == 0 ) return operator/( a, b.c0 );

        ASSERT( a.n() == b.n() );
        const TF inv_b0 = TF( 1 ) / b.c0;
        TaylorScalar r = empty( a.n(), a.c0 * inv_b0 );
        for( int i = 0; i < a.n(); ++i )
            r.c1[ i ] = ( a.c1[ i ] - r.c0 * b.c1[ i ] ) * inv_b0;
        for( int i = 0; i < a.n(); ++i )
            for( int j = 0; j < a.n(); ++j )
                r.d2( i, j ) = ( a.d2( i, j ) - r.c0 * b.d2( i, j ) - r.c1[ i ] * b.c1[ j ] ) * inv_b0;
        for( int i = 0; i < a.n(); ++i )
            for( int j = 0; j < a.n(); ++j )
                for( int k = 0; k < a.n(); ++k )
                    r.d3( i, j, k ) = ( a.d3( i, j, k )
                                      - r.c0 * b.d3( i, j, k )
                                      - r.c1[ i ] * b.d2( j, k )
                                      - r.d2( i, j ) * b.c1[ k ] ) * inv_b0;
        for( int i = 0; i < a.n(); ++i )
            for( int j = 0; j < a.n(); ++j )
                for( int k = 0; k < a.n(); ++k )
                    for( int l = 0; l < a.n(); ++l )
                        r.d4( i, j, k, l ) = ( a.d4( i, j, k, l )
                                             - r.c0 * b.d4( i, j, k, l )
                                             - r.c1[ i ] * b.d3( j, k, l )
                                             - r.d2( i, j ) * b.d2( k, l )
                                             - r.d3( i, j, k ) * b.c1[ l ] ) * inv_b0;
        return r;
    }

    // ── self op ───────────────────────────────────────────────────────────────

    TaylorScalar& operator+=( const TaylorScalar& b ) { return *this = *this + b; }
    TaylorScalar& operator+=( TF s ) { c0 += s; return *this; }

    TaylorScalar& operator-=( const TaylorScalar& b ) { return *this = *this - b; }
    TaylorScalar& operator-=( TF s ) { c0 -= s; return *this; }

    TaylorScalar& operator*=( const TaylorScalar& b ) { return *this = *this * b; }
    TaylorScalar& operator*=( TF s ) { return *this = *this * s; }

    TaylorScalar& operator/=( const TaylorScalar& b ) { return *this = *this / b; }
    TaylorScalar& operator/=( TF s ) { return *this = *this / s; }

    // ── math functions ────────────────────────────────────────────────────────

    friend TaylorScalar sqrt( const TaylorScalar &a ) {
        TaylorScalar r = constant( a.n(), std::sqrt( a.c0 ) );
        const TF inv2r0 = TF( 1 ) / ( TF( 2 ) * r.c0 );
        for( int i = 0; i < a.n(); ++i )
            r.c1[ i ] = a.c1[ i ] * inv2r0;
        for( int i = 0; i < a.n(); ++i )
            for( int j = 0; j < a.n(); ++j )
                r.d2( i, j ) = ( a.d2( i, j ) - r.c1[ i ] * r.c1[ j ] ) * inv2r0;
        for( int i = 0; i < a.n(); ++i )
            for( int j = 0; j < a.n(); ++j )
                for( int k = 0; k < a.n(); ++k )
                    r.d3( i, j, k ) = ( a.d3( i, j, k )
                                      - r.c1[ i ] * r.d2( j, k )
                                      - r.d2( i, j ) * r.c1[ k ] ) * inv2r0;
        for( int i = 0; i < a.n(); ++i )
            for( int j = 0; j < a.n(); ++j )
                for( int k = 0; k < a.n(); ++k )
                    for( int l = 0; l < a.n(); ++l )
                        r.d4( i, j, k, l ) = ( a.d4( i, j, k, l )
                                             - r.c1[ i ] * r.d3( j, k, l )
                                             - r.d2( i, j ) * r.d2( k, l )
                                             - r.d3( i, j, k ) * r.c1[ l ] ) * inv2r0;
        return r;
    }

    // ── comparisons (on c0 only) ──────────────────────────────────────────────

    bool operator> ( TF v ) const { return c0 >  v; }
    bool operator< ( TF v ) const { return c0 <  v; }
    bool operator>=( TF v ) const { return c0 >= v; }
    bool operator<=( TF v ) const { return c0 <= v; }
    bool operator==( TF v ) const { return c0 == v; }

    friend std::ostream &operator<<( std::ostream &os, const TaylorScalar &p ) {
        os << p.c0;
        os << ", [" << p.c1 << "]";
        os << ", [" << p.c2 << "]";
        os << ", [" << p.c3 << "]";
        os << ", [" << p.c4 << "]";
        return os;
    }
};


} // namespace sdot
