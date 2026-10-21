#pragma once

#include <loom/support/common_macros.h> // HD

// =====================================================================================
// CANONICAL PREDICATES ( `LocalN< TK, D, true >`, `PowerDiagram( predicates = "canonical" )` ): every cell takes the SAME decision
// about the same vertex.
//
// A vertex of a power diagram is named by its GENERATORS: the seed of its cell and the `D` cuts it is the corner of -- seeds ( with
// their image under a periodicity ) and walls of the domain. The usual cut decides whether a plane removes a vertex from the
// coordinates of that vertex IN THAT CELL, which came from that cell's own history of cuts: two cells that share a vertex can decide
// differently when the plane passes within rounding of it ( a grid of seeds ). Here every decision is a function of the generators
// alone, computed by the same code from the same numbers in every cell:
//
//   * THE FRAME of a vertex is its smallest seed `s0` ( by rank; among several images of that seed, the one whose list of relative
//     shifts is the smallest ). The other generators are sorted by rank and relative shift, the walls by number.
//   * ITS POSITION is solved from its `D` equations in that frame and that order ( `solve` ), relative to `s0`: the bisectors
//     `2 d_g . y = |d_g|^2 - w_g + w_s0`, `d_g` the generator relative to `s0`, and the walls `n . y = o - n . q_s0`.
//   * "IS IT INSIDE THE HALF-SPACE OF `j`" is the sign of `pow_j( x ) - pow_s0( x )` ( a wall: `o - n . x` ), one expression of
//     `( vertex, j )` ( `side` ). The position and the predicate are computed in DOUBLE-DOUBLE ( about 106 bits ): the sign is that of
//     the doubles given, as good as exact. A value within the rounding of the double-double ( `2^-90` of the size of its terms ) is
//     an exact zero of the doubles: a TIE. ( A wider band would mix ties and true signs in one degenerate spot -- a grid nudged by
//     the size of the band --, which no polytope realizes. )
//   * WITHIN THE BAND, A TIE: broken by SIMULATION OF SIMPLICITY ( `sos_inside` ). The weight of each seed is perturbed by an
//     infinitesimal `delta_r`, `delta_0 >> delta_1 >> ...` in the order of the ranks ( the images of a seed share it ): the
//     predicate is LINEAR in the perturbation ( the vertex moves linearly with the weights ), so its sign is that of the first
//     nonzero coefficient. Images of one seed share its weight: what the weights leave tied ( four images on a sphere, the corners
//     of a period ) is broken by a smaller deformation of the lattice of the periodicity. The perturbed diagram is in general position -- each vertex has exactly `D + 1` generators, the cell is
//     a simple polytope -- and it is the same diagram in every cell.
//   * THE ZERO EDGES: what a tie creates ( two vertices at one place ) is an edge of zero length, and so is what doubles meant to be
//     degenerate ( a grid of thirds, rounded: the cells are those of the doubles, with slivers of `1e-17` ) give. An edge is marked as
//     zero ( `LocalN::store`, `vertex_ties` ) when each of its ends is within `2^-48` of the size of the terms -- coordinates included,
//     the rounding of the doubles given -- of the plane of the other's extra generator: a function of the two vertices alone, the same
//     in every cell. The cell complex contracts these edges.
//
// The functions that compute are `noinline`: the same machine code for every cell, hence the same rounding.
// =====================================================================================

#include <cmath>
#include <type_traits>

namespace sdot {
namespace canon {

enum : int { SEED = 0, WALL = 1, OTHER = 2 };

/// A GENERATOR: a seed ( its rank, its position without the image, the shift of the image, its weight ), a wall ( its number, unit
/// normal in `p`, offset in `w` ), or something else ( a made-up wall, a plane of a group ), which makes the vertex non-canonical
template<int D>
struct Gen {
    int    kind = OTHER;
    int    rank = 0;
    double p[ D ] = {};
    double s[ D ] = {};
    double w = 0;
};

/// the band within which a sign is a tie, relative to the size of the terms: the rounding of the double-double
HD inline constexpr double tie_band() { return 0x1p-90; }

/// the band within which an edge has zero length, relative to the size of the terms, coordinates included: the rounding of the doubles
HD inline constexpr double zero_band() { return 0x1p-48; }

// ---- DOUBLE-DOUBLE: `hi + lo`, about 106 bits ( the error-free transformations: no `-ffast-math`, an explicit `fma` ) -------------

struct DD { double hi = 0, lo = 0; };

HD inline DD two_sum( double a, double b ) { const double s = a + b, bb = s - a; return { s, ( a - ( s - bb ) ) + ( b - bb ) }; }
HD inline DD quick_two_sum( double a, double b ) { const double s = a + b; return { s, b - ( s - a ) }; }
HD inline DD two_prod( double a, double b ) { const double p = a * b; return { p, std::fma( a, b, -p ) }; }
HD inline DD dd( double a ) { return { a, 0 }; }
HD inline DD operator-( DD a ) { return { -a.hi, -a.lo }; }
HD inline DD operator+( DD a, DD b ) {
    DD s = two_sum( a.hi, b.hi );
    const DD t = two_sum( a.lo, b.lo );
    s.lo += t.hi;
    s = quick_two_sum( s.hi, s.lo );
    s.lo += t.lo;
    return quick_two_sum( s.hi, s.lo );
}
HD inline DD operator-( DD a, DD b ) { return a + ( - b ); }
HD inline DD operator*( DD a, DD b ) {
    DD p = two_prod( a.hi, b.hi );
    p.lo += a.hi * b.lo + a.lo * b.hi;
    return quick_two_sum( p.hi, p.lo );
}
HD inline DD operator/( DD a, DD b ) {
    const double q1 = a.hi / b.hi;
    DD r = a - b * dd( q1 );
    const double q2 = r.hi / b.hi;
    r = r - b * dd( q2 );
    const double q3 = r.hi / b.hi;
    return quick_two_sum( q1, q2 ) + dd( q3 );
}
HD inline double fabs_dd( DD a ) { return std::fabs( a.hi ); }

/// `a - b` for the generators, relative: `( p_a - p_b ) + ( s_a - s_b )`, exactly ( the images of a lattice of integer periods: the
/// same whatever the frame )
template<int D>
HD inline void rel( const Gen<D> &a, const Gen<D> &b, DD *r ) {
    for ( int d = 0; d < D; ++d )
        r[ d ] = two_sum( a.p[ d ], - b.p[ d ] ) + two_sum( a.s[ d ], - b.s[ d ] );
}

/// the largest coordinate of a generator ( what the rounding of its doubles is relative to )
template<int D>
HD inline double coord_scale( const Gen<D> &g ) {
    double m = 0;
    if ( g.kind == SEED )
        for ( int d = 0; d < D; ++d ) m = std::fmax( m, std::fabs( g.p[ d ] ) + std::fabs( g.s[ d ] ) );
    return m;
}

/// lexicographic comparison of two generators, each relative to its own frame ( seeds: rank, then the shift relative to the frame's;
/// walls after the seeds, by number )
template<int D>
HD inline int compare_rel( const Gen<D> &a, const Gen<D> &oa, const Gen<D> &b, const Gen<D> &ob ) {
    if ( a.kind != b.kind ) return a.kind < b.kind ? -1 : 1;
    if ( a.rank != b.rank ) return a.rank < b.rank ? -1 : 1;
    if ( a.kind == SEED )
        for ( int d = 0; d < D; ++d ) {
            const double x = a.s[ d ] - oa.s[ d ], y = b.s[ d ] - ob.s[ d ];
            if ( x != y ) return x < y ? -1 : 1;
        }
    return 0;
}

/// THE CANONICAL ORDER of the `n` generators `g`: `ord[ 0 ]` the frame `s0`, then the others sorted. `false` if a generator is not
/// canonical, or if there is no seed.
template<int D>
HD inline bool order( const Gen<D> *g, int n, int *ord ) {
    int low = -1;
    for ( int i = 0; i < n; ++i ) {
        if ( g[ i ].kind == OTHER ) return false;
        if ( g[ i ].kind == SEED && ( low < 0 || g[ i ].rank < g[ low ].rank ) ) low = i;
    }
    if ( low < 0 ) return false;
    // the others, sorted relative to a candidate frame `c` ( insertion sort: at most `D + 1` of them )
    auto sorted = [&]( int c, int *o ) {
        int m = 0;
        for ( int i = 0; i < n; ++i ) {
            if ( i == c ) continue;
            int k = m++;
            while ( k > 0 && compare_rel<D>( g[ i ], g[ c ], g[ o[ k - 1 ] ], g[ c ] ) < 0 ) { o[ k ] = o[ k - 1 ]; --k; }
            o[ k ] = i;
        }
    };
    // among the images of the smallest seed ( a cell that meets itself across a period ), the frame whose list is the smallest
    int best[ D + 2 ], cur[ D + 2 ];
    int bc = -1;
    for ( int c = 0; c < n; ++c ) {
        if ( g[ c ].kind != SEED || g[ c ].rank != g[ low ].rank ) continue;
        sorted( c, cur );
        bool better = bc < 0;
        if ( ! better )
            for ( int i = 0; i < n - 1; ++i ) {
                const int q = compare_rel<D>( g[ cur[ i ] ], g[ c ], g[ best[ i ] ], g[ bc ] );
                if ( q != 0 ) { better = q < 0; break; }
            }
        if ( better ) {
            bc = c;
            for ( int i = 0; i < n - 1; ++i ) best[ i ] = cur[ i ];
        }
    }
    ord[ 0 ] = bc;
    for ( int i = 0; i < n - 1; ++i ) ord[ i + 1 ] = best[ i ];
    return true;
}

/// the rows of the system of a vertex, in the canonical order: `A y = b`, `y` the position relative to `q_s0`
template<int D>
HD inline void system( const Gen<D> *g, const int *ord, DD ( *A )[ D ], DD *b ) {
    const Gen<D> &o = g[ ord[ 0 ] ];
    for ( int r = 0; r < D; ++r ) {
        const Gen<D> &e = g[ ord[ r + 1 ] ];
        if ( e.kind == SEED ) {
            DD dv[ D ], n2;
            rel<D>( e, o, dv );
            for ( int d = 0; d < D; ++d ) { A[ r ][ d ] = { 2 * dv[ d ].hi, 2 * dv[ d ].lo }; n2 = n2 + dv[ d ] * dv[ d ]; }
            b[ r ] = ( n2 - dd( e.w ) ) + dd( o.w );
        } else {
            DD np, ns;
            for ( int d = 0; d < D; ++d ) { A[ r ][ d ] = dd( e.p[ d ] ); np = np + two_prod( e.p[ d ], o.p[ d ] ); ns = ns + two_prod( e.p[ d ], o.s[ d ] ); }
            b[ r ] = ( dd( e.w ) - np ) - ns;
        }
    }
}

/// Gaussian elimination with partial pivoting, in place: `false` if singular ( relative to the rows )
template<int D,class T>
HD inline bool gauss( T ( *A )[ D ], T *x ) {
    auto mag = []( const T &v ) { if constexpr ( std::is_same_v<T,DD> ) return std::fabs( v.hi ); else return std::fabs( v ); };
    double scale = 0;
    for ( int r = 0; r < D; ++r ) for ( int c = 0; c < D; ++c ) scale = std::fmax( scale, mag( A[ r ][ c ] ) );
    if ( ! ( scale > 0 ) ) return false;
    for ( int c = 0; c < D; ++c ) {
        int piv = c;
        for ( int r = c + 1; r < D; ++r ) if ( mag( A[ r ][ c ] ) > mag( A[ piv ][ c ] ) ) piv = r;
        if ( ! ( mag( A[ piv ][ c ] ) > 1e-14 * scale ) ) return false;
        if ( piv != c ) {
            for ( int k = 0; k < D; ++k ) { const T t = A[ c ][ k ]; A[ c ][ k ] = A[ piv ][ k ]; A[ piv ][ k ] = t; }
            const T t = x[ c ]; x[ c ] = x[ piv ]; x[ piv ] = t;
        }
        for ( int r = c + 1; r < D; ++r ) {
            const T f = A[ r ][ c ] / A[ c ][ c ];
            for ( int k = c; k < D; ++k ) A[ r ][ k ] = A[ r ][ k ] - f * A[ c ][ k ];
            x[ r ] = x[ r ] - f * x[ c ];
        }
    }
    for ( int r = D - 1; r >= 0; --r ) {
        T s = x[ r ];
        for ( int k = r + 1; k < D; ++k ) s = s - A[ r ][ k ] * x[ k ];
        x[ r ] = s / A[ r ][ r ];
    }
    return true;
}

/// THE POSITION of the vertex of generators `g` ( `D + 1` of them ): `y` relative to `q_s0`, and `s0` the index of the frame in `g`.
/// `false` if it is not canonical ( a made-up generator ) or the planes do not meet at a point.
template<int D>
[[gnu::noinline]] HD bool solve( const Gen<D> *g, DD *y, int &s0 ) {
    int ord[ D + 1 ];
    if ( ! order<D>( g, D + 1, ord ) ) return false;
    DD A[ D ][ D ];
    system<D>( g, ord, A, y );
    if ( ! gauss<D,DD>( A, y ) ) return false;
    s0 = ord[ 0 ];
    return true;
}

/// `> 0`: the point `y` ( relative to `q_s0` of the generator `o` ) is INSIDE the half-space of `j` -- `pow_j - pow_s0` for a seed,
/// `off - n . x` for a wall. `mag`: the size of the terms ( the band is relative to it ).
template<int D>
[[gnu::noinline]] HD double side( const Gen<D> &o, const DD *y, const Gen<D> &j, double &mag ) {
    const double ps = std::fmax( coord_scale<D>( o ), coord_scale<D>( j ) );
    if ( j.kind == SEED ) {
        DD dv[ D ], n2, dy;
        double ny = 0;
        rel<D>( j, o, dv );
        for ( int d = 0; d < D; ++d ) { n2 = n2 + dv[ d ] * dv[ d ]; dy = dy + dv[ d ] * y[ d ]; ny += y[ d ].hi * y[ d ].hi; }
        const double dn = std::sqrt( n2.hi );
        mag = n2.hi + std::fabs( j.w ) + std::fabs( o.w ) + 2 * dn * ( std::sqrt( ny ) + ps );
        const DD f = ( ( n2 - dd( j.w ) ) + dd( o.w ) ) - DD{ 2 * dy.hi, 2 * dy.lo };
        return f.hi;
    }
    DD np, ns, nyv;
    double ny = 0;
    for ( int d = 0; d < D; ++d ) { np = np + two_prod( j.p[ d ], o.p[ d ] ); ns = ns + two_prod( j.p[ d ], o.s[ d ] ); nyv = nyv + dd( j.p[ d ] ) * y[ d ]; ny += y[ d ].hi * y[ d ].hi; }
    mag = std::fabs( j.w ) + std::fabs( np.hi ) + std::fabs( ns.hi ) + std::sqrt( ny ) + ps;
    const DD f = ( ( dd( j.w ) - np ) - ns ) - nyv;
    return f.hi;
}

/// `side` against a band ( `tie_band`, `zero_band` ): `1` inside, `-1` outside, `0` within the band
template<int D>
HD inline int sign_of( const Gen<D> &o, const DD *y, const Gen<D> &j, double band = tie_band() ) {
    double mag = 0;
    const double f = side<D>( o, y, j, mag );
    if ( f > band * mag ) return 1;
    if ( f < - band * mag ) return -1;
    return 0;
}

/// A TIE between the vertex of generators `g` and the half-space of `j`, broken by the perturbation of the weights: `true` if the
/// vertex is inside. The predicate is `f( y ) = gv . y + h`, `y = A^-1 b`: its derivative along `delta_r` is `z . db / ddelta_r +
/// dh / ddelta_r`, `A^T z = gv`. The first rank whose coefficient is not zero decides; none ( a periodic diagram where the images of
/// one seed cancel ): inside.
template<int D>
[[gnu::noinline]] HD bool sos_inside( const Gen<D> *g, const Gen<D> &j ) {
    int ord[ D + 1 ];
    if ( ! order<D>( g, D + 1, ord ) ) return true;
    const Gen<D> &o = g[ ord[ 0 ] ];
    DD A[ D ][ D ], b[ D ];
    system<D>( g, ord, A, b );
    double At[ D ][ D ], z[ D ];
    for ( int r = 0; r < D; ++r ) for ( int c = 0; c < D; ++c ) At[ r ][ c ] = A[ c ][ r ].hi;
    if ( j.kind == SEED ) {
        DD dv[ D ];
        rel<D>( j, o, dv );
        for ( int d = 0; d < D; ++d ) z[ d ] = -2 * dv[ d ].hi;
    } else {
        for ( int d = 0; d < D; ++d ) z[ d ] = - j.p[ d ];
    }
    if ( ! gauss<D,double>( At, z ) ) return true;
    double zs = 1;
    for ( int d = 0; d < D; ++d ) zs += std::fabs( z[ d ] );
    // the ranks in play, ascending
    int ranks[ D + 2 ], nr = 0;
    auto push = [&]( int r ) { for ( int i = 0; i < nr; ++i ) if ( ranks[ i ] == r ) return; ranks[ nr++ ] = r; };
    push( o.rank );
    for ( int r = 0; r < D; ++r ) if ( g[ ord[ r + 1 ] ].kind == SEED ) push( g[ ord[ r + 1 ] ].rank );
    if ( j.kind == SEED ) push( j.rank );
    for ( int a = 1; a < nr; ++a ) for ( int c = a; c > 0 && ranks[ c ] < ranks[ c - 1 ]; --c ) { const int t = ranks[ c ]; ranks[ c ] = ranks[ c - 1 ]; ranks[ c - 1 ] = t; }
    for ( int q = 0; q < nr; ++q ) {
        const int rk = ranks[ q ];
        double df = 0;
        for ( int r = 0; r < D; ++r ) {
            const Gen<D> &e = g[ ord[ r + 1 ] ];
            if ( e.kind != SEED ) continue;
            df += z[ r ] * ( double( o.rank == rk ) - double( e.rank == rk ) );
        }
        if ( j.kind == SEED )
            df += double( o.rank == rk ) - double( j.rank == rk );
        if ( std::fabs( df ) > 1e-9 * zs )
            return df > 0;
    }

    // THE LATTICE, perturbed: the images of one seed share its weight, and four of them can be on one sphere whatever the seeds ( the
    // corners of a period: A, A + e1, A + e2, A + e1 + e2 ). The relative shifts are deformed by `( I + eps M )`, `M` a fixed generic
    // matrix -- a lattice still, the same in every frame. `d_k` moves by `u_k = M ( s_k - s_o )`: the row `2 d_k . y = |d_k|^2 - ...`
    // by `2 u_k . ( d_k - y )`, and the predicate of a seed by `2 u_j . ( d_j - y )`.
    double y[ D ], Ad[ D ][ D ];
    for ( int r = 0; r < D; ++r ) { y[ r ] = b[ r ].hi; for ( int c = 0; c < D; ++c ) Ad[ r ][ c ] = A[ r ][ c ].hi; }
    if ( ! gauss<D,double>( Ad, y ) ) return true;
    auto lattice_move = [&]( const Gen<D> &e, double *u ) {
        for ( int r = 0; r < D; ++r ) {
            u[ r ] = 0;
            for ( int c = 0; c < D; ++c ) {
                const double m = 0.5 - ( ( r * D + c + 1 ) * 0.6180339887498949 - double( int( ( r * D + c + 1 ) * 0.6180339887498949 ) ) );
                u[ r ] += m * ( e.s[ c ] - o.s[ c ] );
            }
        }
    };
    auto move_of = [&]( const Gen<D> &e ) {                // `2 u_e . ( d_e - y )`
        double u[ D ], dv[ D ], res = 0;
        lattice_move( e, u );
        DD dd_[ D ];
        rel<D>( e, o, dd_ );
        for ( int d = 0; d < D; ++d ) { dv[ d ] = dd_[ d ].hi; res += 2 * u[ d ] * ( dv[ d ] - y[ d ] ); }
        return res;
    };
    double df = 0, sc = 0;
    for ( int r = 0; r < D; ++r ) {
        const Gen<D> &e = g[ ord[ r + 1 ] ];
        if ( e.kind != SEED ) continue;
        const double t = z[ r ] * move_of( e );
        df += t;
        sc += std::fabs( t );
    }
    if ( j.kind == SEED ) {
        const double t = move_of( j );
        df += t;
        sc += std::fabs( t );
    }
    if ( std::fabs( df ) > 1e-9 * sc )
        return df > 0;
    return true;
}

} // namespace canon
} // namespace sdot
