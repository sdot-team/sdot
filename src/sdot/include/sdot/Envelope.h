#pragma once

// =====================================================================================
// THE ENVELOPE OF A CODIMENSION >= 1 MESH ( `sdot.envelope` ): the support of the smoothed density of a mesh M of lower dimension
// ( lines in 2D, lines or surfaces in 3D ), as a soup of simplices where the distance `d` to M and the density `rho` are affine.
// Design: `grill/codim-mesh-density/decisions.md` ( "l'enveloppe" ).
//
// It is the generalized Voronoi diagram of the elements `sigma` of M ( the simplices of M and all their faces ), as the minimum
// of affine functions, each one defined on a convex candidate:
//   * the candidate of `sigma` is `sigma x` its polar cone, `{ u in N_sigma : u . e <= 0 }` for the edges `e` leaving `sigma` in
//     the simplices of M containing it ( `N_sigma` the normal space ), cut by the domain box. In it, `d` is the distance to
//     `sigma` and `rho` is `rho` at the projection on `sigma`;
//   * when `N_sigma` has more than one dimension, the cone is discretized into simplicial cones at most `alpha` wide ( a regular
//     polygon of the circle, a subdivided icosahedron of the sphere whose triangles lie in caps of radius `alpha / 2`, clipped by
//     the polar cone ), with `d` P1: exact on the rays,
//     over-estimated by at most `1 / cos( alpha / 2 )` between them. On the maximal simplices of M ( segments of lines in 3D: they
//     carry the mass when `s -> 0` ) a sector of angle `theta` gets the weight `theta / sin( theta )`, which makes the limit exact;
//   * where candidates overlap, the smallest `d` wins: `A \ ( B ∩ { d_B <= d_A } )`, cut into convex pieces by the half-spaces
//     of `B` ( ties: the smallest index wins ). When `A ∩ { d_B <= d_A }` lies in `B`, a single cut by the bisector does it.
// Every point is kept by exactly one candidate ( the one with the smallest `d`, then index ): a tiling, whatever the accuracy of
// the `d` of the fans. No filter yet: each candidate is compared to all the candidates whose box overlaps its own.
//
// The output: the convex pieces fanned into simplices, `pts( n, i, c )`, `d( n, i )`, `rho( n, i )`, `w( n )`, and in
// `stats`: the number of simplices ( the outputs have room for `cap`, the call is redone with more room when it overflows ),
// candidates, pieces.
// =====================================================================================

#include <loom/support/common_macros.h>
#include <loom/support/kernels/CpuQueue.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <numeric>
#include <vector>

namespace sdot {
namespace envelope {

template<int D> using Pt = std::array<double,D>;

template<std::size_t D> double dot( const std::array<double,D> &a, const std::array<double,D> &b ) { double s = 0; for ( std::size_t i = 0; i < D; ++i ) s += a[ i ] * b[ i ]; return s; }
template<std::size_t D> std::array<double,D> sub( const std::array<double,D> &a, const std::array<double,D> &b ) { std::array<double,D> r; for ( std::size_t i = 0; i < D; ++i ) r[ i ] = a[ i ] - b[ i ]; return r; }
template<std::size_t D> std::array<double,D> add( const std::array<double,D> &a, const std::array<double,D> &b ) { std::array<double,D> r; for ( std::size_t i = 0; i < D; ++i ) r[ i ] = a[ i ] + b[ i ]; return r; }
template<std::size_t D> std::array<double,D> mul( const std::array<double,D> &a, double s ) { std::array<double,D> r; for ( std::size_t i = 0; i < D; ++i ) r[ i ] = a[ i ] * s; return r; }
template<std::size_t D> double norm( const std::array<double,D> &a ) { return std::sqrt( dot( a, a ) ); }
inline Pt<3> cross( const Pt<3> &a, const Pt<3> &b ) { return { a[ 1 ] * b[ 2 ] - a[ 2 ] * b[ 1 ], a[ 2 ] * b[ 0 ] - a[ 0 ] * b[ 2 ], a[ 0 ] * b[ 1 ] - a[ 1 ] * b[ 0 ] }; }

/// `g . x + c`
template<int D> struct Affine {
    Pt<D>  g{};
    double c = 0;
    double operator()( const Pt<D> &x ) const { return dot( g, x ) + c; }
};

/// `n . x <= c`, `|n| = 1`
template<int D> struct HalfSpace {
    Pt<D>  n{};
    double c = 0;
    double dist( const Pt<D> &x ) const { return dot( n, x ) - c; }
    HalfSpace flipped() const { return { mul( n, -1.0 ), -c }; }
    static HalfSpace normalized( Pt<D> n, double c ) { const double l = norm( n ); return { mul( n, 1 / l ), c / l }; }
};

/// `p` without its consecutive duplicates ( cyclic )
template<std::size_t D> void dedupe( std::vector<std::array<double,D>> &p, double tol ) {
    std::vector<std::array<double,D>> r;
    for ( const std::array<double,D> &x : p )
        if ( r.empty() || norm( sub( x, r.back() ) ) > tol )
            r.push_back( x );
    while ( r.size() > 1 && norm( sub( r.front(), r.back() ) ) <= tol )
        r.pop_back();
    p = std::move( r );
}

// ---- convex polytopes ----

template<int D> struct Poly;

/// a convex polygon, counter-clockwise
template<> struct Poly<2> {
    std::vector<Pt<2>> v;

    static Poly box( const Pt<2> &lo, const Pt<2> &hi ) { return { { lo, { hi[ 0 ], lo[ 1 ] }, hi, { lo[ 0 ], hi[ 1 ] } } }; }
    bool empty() const { return v.size() < 3; }
    const std::vector<Pt<2>> &vertices() const { return v; }

    Poly clip( const HalfSpace<2> &h, double tol ) const {
        if ( empty() ) return {};
        double mi = 1e300, ma = -1e300;
        std::vector<double> s( v.size() );
        for ( std::size_t i = 0; i < v.size(); ++i ) { s[ i ] = h.dist( v[ i ] ); mi = std::min( mi, s[ i ] ); ma = std::max( ma, s[ i ] ); }
        if ( ma <= tol ) return *this;
        if ( mi >= -tol ) return {};
        Poly r;
        for ( std::size_t i = 0, k = v.size(); i < k; ++i ) {
            const std::size_t j = ( i + 1 ) % k;
            if ( s[ i ] <= tol ) r.v.push_back( v[ i ] );
            if ( ( s[ i ] < -tol && s[ j ] > tol ) || ( s[ i ] > tol && s[ j ] < -tol ) )
                r.v.push_back( add( v[ i ], mul( sub( v[ j ], v[ i ] ), s[ i ] / ( s[ i ] - s[ j ] ) ) ) );
        }
        dedupe( r.v, tol );
        return r;
    }

    double measure() const {
        double a = 0;
        for ( std::size_t i = 0, k = v.size(); i < k; ++i ) {
            const Pt<2> &p = v[ i ], &q = v[ ( i + 1 ) % k ];
            a += p[ 0 ] * q[ 1 ] - p[ 1 ] * q[ 0 ];
        }
        return 0.5 * std::fabs( a );
    }

    /// the supporting half-spaces of the edges
    std::vector<HalfSpace<2>> half_spaces() const {
        std::vector<HalfSpace<2>> r;
        for ( std::size_t i = 0, k = v.size(); i < k; ++i ) {
            const Pt<2> e = sub( v[ ( i + 1 ) % k ], v[ i ] );
            if ( norm( e ) > 0 ) r.push_back( HalfSpace<2>::normalized( { e[ 1 ], - e[ 0 ] }, e[ 1 ] * v[ i ][ 0 ] - e[ 0 ] * v[ i ][ 1 ] ) );
        }
        return r;
    }

    /// `f( std::array<Pt<2>,3> )`, a fan from the vertex `k`
    template<class F> void for_each_simplex( std::size_t k, F &&f ) const {
        for ( std::size_t i = 1; i + 1 < v.size(); ++i )
            f( std::array<Pt<2>,3>{ v[ k ], v[ ( k + i ) % v.size() ], v[ ( k + i + 1 ) % v.size() ] } );
    }
};

/// a convex polyhedron: its faces, each a polygon counter-clockwise seen from outside
template<> struct Poly<3> {
    std::vector<std::vector<Pt<3>>> faces;
    std::vector<Pt<3>>              verts;                      ///< ( the vertices, without duplicates )

    static Poly box( const Pt<3> &lo, const Pt<3> &hi ) {
        auto c = [&]( int i ) { return Pt<3>{ i & 1 ? hi[ 0 ] : lo[ 0 ], i & 2 ? hi[ 1 ] : lo[ 1 ], i & 4 ? hi[ 2 ] : lo[ 2 ] }; };
        Poly r;
        for ( auto f : { std::array<int,4>{ 0, 2, 3, 1 }, { 4, 5, 7, 6 }, { 0, 1, 5, 4 }, { 2, 6, 7, 3 }, { 0, 4, 6, 2 }, { 1, 3, 7, 5 } } )
            r.faces.push_back( { c( f[ 0 ] ), c( f[ 1 ] ), c( f[ 2 ] ), c( f[ 3 ] ) } );
        r.update_verts( 0 );
        return r;
    }
    bool empty() const { return faces.size() < 4; }
    const std::vector<Pt<3>> &vertices() const { return verts; }

    void update_verts( double tol ) {
        verts.clear();
        for ( const auto &f : faces )
            for ( const Pt<3> &x : f ) {
                bool seen = false;
                for ( const Pt<3> &y : verts ) if ( norm( sub( x, y ) ) <= tol ) { seen = true; break; }
                if ( ! seen ) verts.push_back( x );
            }
    }

    static Pt<3> newell( const std::vector<Pt<3>> &f ) {
        Pt<3> n{};
        for ( std::size_t i = 0; i < f.size(); ++i ) {
            const Pt<3> &a = f[ i ], &b = f[ ( i + 1 ) % f.size() ];
            n[ 0 ] += ( a[ 1 ] - b[ 1 ] ) * ( a[ 2 ] + b[ 2 ] );
            n[ 1 ] += ( a[ 2 ] - b[ 2 ] ) * ( a[ 0 ] + b[ 0 ] );
            n[ 2 ] += ( a[ 0 ] - b[ 0 ] ) * ( a[ 1 ] + b[ 1 ] );
        }
        return n;
    }

    Poly clip( const HalfSpace<3> &h, double tol ) const {
        if ( empty() ) return {};
        double mi = 1e300, ma = -1e300;
        for ( const Pt<3> &x : verts ) { const double s = h.dist( x ); mi = std::min( mi, s ); ma = std::max( ma, s ); }
        if ( ma <= tol ) return *this;
        if ( mi >= -tol ) return {};
        Poly r;
        std::vector<Pt<3>> cap;
        for ( const auto &f : faces ) {
            std::vector<double> s( f.size() );
            bool flat = true;
            for ( std::size_t i = 0; i < f.size(); ++i ) { s[ i ] = h.dist( f[ i ] ); flat &= std::fabs( s[ i ] ) <= tol; }
            if ( flat ) {                                       // ( a face in the plane: the cap is rebuilt from its points, which
                cap.insert( cap.end(), f.begin(), f.end() );    //   is robust to several faces almost in the plane )
                continue;
            }
            std::vector<Pt<3>> o;
            for ( std::size_t i = 0, k = f.size(); i < k; ++i ) {
                const std::size_t j = ( i + 1 ) % k;
                if ( s[ i ] <= tol ) {
                    o.push_back( f[ i ] );
                    if ( s[ i ] >= -tol ) cap.push_back( f[ i ] );
                }
                if ( ( s[ i ] < -tol && s[ j ] > tol ) || ( s[ i ] > tol && s[ j ] < -tol ) ) {
                    const Pt<3> p = add( f[ i ], mul( sub( f[ j ], f[ i ] ), s[ i ] / ( s[ i ] - s[ j ] ) ) );
                    o.push_back( p );
                    cap.push_back( p );
                }
            }
            dedupe( o, tol );
            if ( o.size() >= 3 )
                r.faces.push_back( std::move( o ) );
        }
        {                                                       // the cap: the convex hull of the points on the plane
            Pt<3> e1 = std::fabs( h.n[ 0 ] ) < 0.9 ? Pt<3>{ 1, 0, 0 } : Pt<3>{ 0, 1, 0 };
            e1 = sub( e1, mul( h.n, dot( e1, h.n ) ) );
            e1 = mul( e1, 1 / norm( e1 ) );
            const Pt<3> e2 = cross( h.n, e1 );               // ( e1 x e2 = n: counter-clockwise in ( e1, e2 ) is outward )
            std::vector<std::pair<Pt<2>,Pt<3>>> q;
            for ( const Pt<3> &x : cap ) {                     // ( without duplicates: the same point comes from two faces )
                bool seen = false;
                for ( const auto &y : q ) if ( norm( sub( x, y.second ) ) <= tol ) { seen = true; break; }
                if ( ! seen ) q.push_back( { Pt<2>{ dot( x, e1 ), dot( x, e2 ) }, x } );
            }
            std::sort( q.begin(), q.end(), []( const auto &a, const auto &b ) { return a.first < b.first; } );
            auto turn = []( const Pt<2> &o, const Pt<2> &a, const Pt<2> &b ) { return ( a[ 0 ] - o[ 0 ] ) * ( b[ 1 ] - o[ 1 ] ) - ( a[ 1 ] - o[ 1 ] ) * ( b[ 0 ] - o[ 0 ] ); };
            std::vector<std::pair<Pt<2>,Pt<3>>> hull( 2 * q.size() + 1 );
            std::size_t k = 0;
            if ( q.size() >= 3 )
            for ( std::size_t i = 0; i < q.size(); ++i ) {          // ( monotone chain )
                while ( k >= 2 && turn( hull[ k - 2 ].first, hull[ k - 1 ].first, q[ i ].first ) <= tol * norm( sub( hull[ k - 1 ].first, hull[ k - 2 ].first ) ) ) --k;
                hull[ k++ ] = q[ i ];
            }
            if ( q.size() >= 3 )
            for ( std::size_t i = q.size() - 1, t = k + 1; i-- > 0; ) {
                while ( k >= t && turn( hull[ k - 2 ].first, hull[ k - 1 ].first, q[ i ].first ) <= tol * norm( sub( hull[ k - 1 ].first, hull[ k - 2 ].first ) ) ) --k;
                hull[ k++ ] = q[ i ];
            }
            std::vector<Pt<3>> c;
            for ( std::size_t i = 0; i + 1 < k && q.size() >= 3; ++i ) c.push_back( hull[ i ].second );
            dedupe( c, tol );
            if ( c.size() >= 3 )
                r.faces.push_back( std::move( c ) );
        }
        r.update_verts( tol );
        if ( r.faces.size() < 4 ) return {};
        return r;
    }

    double measure() const {
        if ( empty() ) return 0;
        const Pt<3> &o = verts[ 0 ];
        double v = 0;
        for ( const auto &f : faces )
            for ( std::size_t i = 1; i + 1 < f.size(); ++i )
                v += dot( sub( f[ 0 ], o ), cross( sub( f[ i ], o ), sub( f[ i + 1 ], o ) ) );
        return std::fabs( v ) / 6;
    }

    /// the supporting half-spaces of the faces
    std::vector<HalfSpace<3>> half_spaces() const {
        std::vector<HalfSpace<3>> r;
        for ( const auto &f : faces ) {
            const Pt<3> n = newell( f );
            if ( norm( n ) > 0 ) r.push_back( HalfSpace<3>::normalized( n, dot( n, f[ 0 ] ) ) );
        }
        return r;
    }

    /// `f( std::array<Pt<3>,4> )`: tetrahedra from the vertex `k` to the fans of the faces not containing it
    template<class F> void for_each_simplex( std::size_t k, F &&f, double tol ) const {
        const Pt<3> &o = verts[ k ];
        for ( const auto &fa : faces ) {
            const Pt<3> n = newell( fa );
            const double l = norm( n );
            if ( l == 0 || std::fabs( dot( n, sub( o, fa[ 0 ] ) ) ) <= tol * l )
                continue;
            for ( std::size_t i = 1; i + 1 < fa.size(); ++i )
                f( std::array<Pt<3>,4>{ o, fa[ 0 ], fa[ i ], fa[ i + 1 ] } );
        }
    }
};

template<int D> bool boxes_overlap( const Pt<D> &alo, const Pt<D> &ahi, const Pt<D> &blo, const Pt<D> &bhi ) {
    for ( int i = 0; i < D; ++i ) if ( alo[ i ] >= bhi[ i ] || blo[ i ] >= ahi[ i ] ) return false;
    return true;
}

template<int D> void bbox( const Poly<D> &p, Pt<D> &lo, Pt<D> &hi ) {
    lo.fill( 1e300 ); hi.fill( -1e300 );
    for ( const Pt<D> &x : p.vertices() ) for ( int i = 0; i < D; ++i ) { lo[ i ] = std::min( lo[ i ], x[ i ] ); hi[ i ] = std::max( hi[ i ], x[ i ] ); }
}

/// `p ∪ q` in `p` when it is convex: then its faces are faces of `p` or `q` that leave every vertex of both inside, and
/// these half-spaces cut a polytope of the volume of both ( more when the union is not convex )
template<int D> bool merge_into( Poly<D> &p, const Poly<D> &q, double tol, double amin ) {
    const std::array<const Poly<D> *,2> both{ &p, &q };
    Pt<D> lo, hi;
    lo.fill( 1e300 ); hi.fill( -1e300 );
    for ( const Poly<D> *a : both )
        for ( const Pt<D> &x : a->vertices() ) for ( int i = 0; i < D; ++i ) { lo[ i ] = std::min( lo[ i ], x[ i ] ); hi[ i ] = std::max( hi[ i ], x[ i ] ); }
    for ( int i = 0; i < D; ++i ) { lo[ i ] -= 1e3 * tol; hi[ i ] += 1e3 * tol; }
    std::vector<HalfSpace<D>> hs;                               // ( without duplicates: p and q can have faces in the same plane )
    for ( const Poly<D> *a : both )
        for ( const HalfSpace<D> &h : a->half_spaces() ) {
            bool all_in = true;
            for ( const Poly<D> *b : both ) for ( const Pt<D> &x : b->vertices() ) all_in &= h.dist( x ) <= tol;
            for ( const HalfSpace<D> &o : hs ) all_in &= ! ( norm( sub( o.n, h.n ) ) <= 1e-9 && std::fabs( o.c - h.c ) <= tol );
            if ( all_in ) hs.push_back( h );
        }
    Poly<D> H = Poly<D>::box( lo, hi );
    for ( const HalfSpace<D> &h : hs ) {
        H = H.clip( h, tol );
        if ( H.empty() ) return false;
    }
    if constexpr ( D == 3 ) {                                   // ( a closed surface: its oriented areas sum to 0 )
        Pt<3> s{};
        double a = 0;
        for ( const auto &f : H.faces ) { const Pt<3> n = Poly<3>::newell( f ); s = add( s, n ); a += norm( n ); }
        if ( norm( s ) > 1e-9 * a ) return false;
    }
    const double vp = p.measure(), vq = q.measure();
    if ( std::fabs( H.measure() - vp - vq ) > 1e-9 * ( vp + vq ) + amin )
        return false;
    for ( const Pt<D> &x : H.vertices() ) {                     // ( and the vertices of the hull are vertices of p or q )
        bool found = false;
        for ( const Poly<D> *a : both ) for ( const Pt<D> &y : a->vertices() ) found |= norm( sub( x, y ) ) <= 1e2 * tol;
        if ( ! found ) return false;
    }
    p = std::move( H );
    return true;
}

/// merges the fragments whose union is convex, greedily; only the pairs with a `fresh` fragment are tried ( the others were ).
/// Two convex pieces with a convex union share a whole face: one of them has the plane of a face of the other, flipped
template<int D> void merge_all( std::vector<Poly<D>> &frags, std::vector<char> &fresh, double tol, double amin ) {
    std::vector<Pt<D>> lo( frags.size() ), hi( frags.size() );
    std::vector<std::vector<HalfSpace<D>>> hs( frags.size() );
    auto box_of = [&]( std::size_t i ) {
        lo[ i ].fill( 1e300 ); hi[ i ].fill( -1e300 );
        for ( const Pt<D> &x : frags[ i ].vertices() ) for ( int c = 0; c < D; ++c ) { lo[ i ][ c ] = std::min( lo[ i ][ c ], x[ c ] - tol ); hi[ i ][ c ] = std::max( hi[ i ][ c ], x[ c ] + tol ); }
        hs[ i ] = frags[ i ].half_spaces();
    };
    auto facing = [&]( std::size_t i, std::size_t j ) {
        for ( const HalfSpace<D> &a : hs[ i ] ) for ( const HalfSpace<D> &b : hs[ j ] )
            if ( norm( add( a.n, b.n ) ) <= 1e-9 && std::fabs( a.c + b.c ) <= tol ) return true;
        return false;
    };
    for ( std::size_t i = 0; i < frags.size(); ++i ) box_of( i );
    for ( std::size_t i = 0; i < frags.size(); ++i ) {
        for ( std::size_t j = i + 1; j < frags.size(); ) {
            if ( ( fresh[ i ] || fresh[ j ] ) && boxes_overlap<D>( lo[ i ], hi[ i ], lo[ j ], hi[ j ] ) && facing( i, j ) && merge_into<D>( frags[ i ], frags[ j ], tol, amin ) ) {
                frags.erase( frags.begin() + j ); fresh.erase( fresh.begin() + j ); lo.erase( lo.begin() + j ); hi.erase( hi.begin() + j ); hs.erase( hs.begin() + j );
                fresh[ i ] = 1;
                box_of( i );
                j = i + 1;                                      // ( i grew: the pairs before j again )
                continue;
            }
            ++j;
        }
    }
    std::fill( fresh.begin(), fresh.end(), 0 );
}

// ---- the candidates ----

template<int D> struct Candidate {
    Poly<D>                     poly;
    std::vector<HalfSpace<D>>   hs;                             ///< ( poly = box ∩ hs )
    Affine<D>                   d, rho;
    double                      w = 1;
    Pt<D>                       base{};                         ///< ( the centroid of the element )
    double                      rad = 0;                        ///< ( the radius of the element around `base` )
    Pt<D>                       lo{}, hi{};                     ///< ( the box of poly )
};

/// an orthonormal basis of `R^D` whose first `k` vectors span the directions `dirs` ( Gram-Schmidt, then completed ); the number
/// of independent directions found
template<int D> int basis_of( const std::vector<Pt<D>> &dirs, std::array<Pt<D>,D> &b, double tol ) {
    int k = 0;
    auto push = [&]( Pt<D> x ) {
        for ( int j = 0; j < k; ++j ) x = sub( x, mul( b[ j ], dot( x, b[ j ] ) ) );
        const double l = norm( x );
        if ( l > tol && k < D ) b[ k++ ] = mul( x, 1 / l );
    };
    for ( const Pt<D> &x : dirs ) push( x );
    const int nt = k;
    for ( int i = 0; i < D; ++i ) { Pt<D> e{}; e[ i ] = 1; push( e ); }
    return nt;
}

/// the sectors of the unit sphere of `R^m`, at most `alpha` wide: `m` unit vectors each
template<int m> std::vector<std::array<Pt<m>,m>> sphere_sectors( double alpha ) {
    std::vector<std::array<Pt<m>,m>> out;
    if constexpr ( m == 1 ) {
        out.push_back( { Pt<1>{ 1 } } );
        out.push_back( { Pt<1>{ -1 } } );
    } else if constexpr ( m == 2 ) {
        const int k = std::max( 3, int( std::ceil( 2 * M_PI / alpha - 1e-9 ) ) );
        for ( int i = 0; i < k; ++i ) {
            const double a = 2 * M_PI * i / k, b = 2 * M_PI * ( i + 1 ) / k;
            out.push_back( { Pt<2>{ std::cos( a ), std::sin( a ) }, Pt<2>{ std::cos( b ), std::sin( b ) } } );
        }
    } else {                                                    // a subdivided icosahedron, the points pushed on the sphere
        const double t = ( 1 + std::sqrt( 5.0 ) ) / 2;
        std::vector<Pt<3>> v = { { -1, t, 0 }, { 1, t, 0 }, { -1, -t, 0 }, { 1, -t, 0 }, { 0, -1, t }, { 0, 1, t }, { 0, -1, -t }, { 0, 1, -t },
                                 { t, 0, -1 }, { t, 0, 1 }, { -t, 0, -1 }, { -t, 0, 1 } };
        const int f[ 20 ][ 3 ] = { { 0, 11, 5 }, { 0, 5, 1 }, { 0, 1, 7 }, { 0, 7, 10 }, { 0, 10, 11 }, { 1, 5, 9 }, { 5, 11, 4 }, { 11, 10, 2 },
                                   { 10, 7, 6 }, { 7, 1, 8 }, { 3, 9, 4 }, { 3, 4, 2 }, { 3, 2, 6 }, { 3, 6, 8 }, { 3, 8, 9 }, { 4, 9, 5 },
                                   { 2, 4, 11 }, { 6, 2, 10 }, { 8, 6, 7 }, { 9, 8, 1 } };
        auto unit = []( Pt<3> x ) { return mul( x, 1 / norm( x ) ); };
        auto subdivide = [&]( int n ) {
            std::vector<std::array<Pt<3>,3>> r;
            for ( auto &tri : f ) {
                const Pt<3> A = v[ tri[ 0 ] ], B = v[ tri[ 1 ] ], C = v[ tri[ 2 ] ];
                auto at = [&]( int i, int j ) { return unit( add( A, add( mul( sub( B, A ), double( i ) / n ), mul( sub( C, A ), double( j ) / n ) ) ) ); };
                for ( int i = 0; i < n; ++i )
                    for ( int j = 0; i + j < n; ++j ) {
                        r.push_back( { at( i, j ), at( i + 1, j ), at( i, j + 1 ) } );
                        if ( i + j + 1 < n )
                            r.push_back( { at( i + 1, j ), at( i + 1, j + 1 ), at( i, j + 1 ) } );
                    }
            }
            return r;
        };
        // the finest subdivision needed for every sector to lie in a cap of angular radius alpha / 2 ( its circumcircle ): the P1
        // of the distance is then over-estimated by at most 1 / cos( alpha / 2 ), as for the polygons in 2D
        for ( int n = 1; ; ++n ) {
            out = subdivide( n );
            double worst = 0;
            for ( const auto &s : out ) {
                const Pt<3> c = unit( cross( sub( s[ 1 ], s[ 0 ] ), sub( s[ 2 ], s[ 0 ] ) ) );
                worst = std::max( worst, std::acos( std::clamp( std::fabs( dot( c, s[ 0 ] ) ), -1.0, 1.0 ) ) );
            }
            if ( worst <= alpha / 2 )
                break;
        }
    }
    return out;
}

/// the simplicial cones of the sectors clipped by `{ u : e . u <= 0 }` for the `cons`: their rays ( unit )
template<int m> std::vector<std::array<Pt<m>,m>> cone_sectors( const std::vector<Pt<m>> &cons, double alpha, double tol ) {
    std::vector<std::array<Pt<m>,m>> out;
    for ( const auto &sec : sphere_sectors<m>( alpha ) ) {
        if constexpr ( m == 1 ) {
            bool ok = true;
            for ( const Pt<1> &e : cons ) ok &= e[ 0 ] * sec[ 0 ][ 0 ] <= tol;
            if ( ok ) out.push_back( sec );
        } else {
            std::vector<Pt<m>> p( sec.begin(), sec.end() );     // ( the flat sector, clipped by the half-spaces through 0 )
            for ( const Pt<m> &e : cons ) {
                std::vector<Pt<m>> q;
                const std::size_t k = p.size();
                for ( std::size_t i = 0; i < k; ++i ) {
                    if ( m == 2 && i + 1 == k ) {               // ( a segment: no closing edge )
                        if ( dot( e, p[ i ] ) <= tol ) q.push_back( p[ i ] );
                        break;
                    }
                    const std::size_t j = ( i + 1 ) % k;
                    const double si = dot( e, p[ i ] ), sj = dot( e, p[ j ] );
                    if ( si <= tol ) q.push_back( p[ i ] );
                    if ( ( si < -tol && sj > tol ) || ( si > tol && sj < -tol ) )
                        q.push_back( add( p[ i ], mul( sub( p[ j ], p[ i ] ), si / ( si - sj ) ) ) );
                }
                p = std::move( q );
                if ( p.size() < m ) break;
            }
            dedupe( p, tol );
            if ( p.size() < m ) continue;
            auto unit = []( Pt<m> x ) { return mul( x, 1 / norm( x ) ); };
            if constexpr ( m == 2 ) {
                if ( norm( sub( unit( p.front() ), unit( p.back() ) ) ) > tol )
                    out.push_back( { unit( p.front() ), unit( p.back() ) } );
            } else {
                for ( std::size_t i = 1; i + 1 < p.size(); ++i ) {
                    const std::array<Pt<3>,3> r{ unit( p[ 0 ] ), unit( p[ i ] ), unit( p[ i + 1 ] ) };
                    if ( std::fabs( dot( r[ 0 ], cross( r[ 1 ], r[ 2 ] ) ) ) > tol )
                        out.push_back( r );
                }
            }
        }
    }
    return out;
}

/// `x` solving `A x = b` ( `n x n`, partial pivoting ); false when singular
template<int n> bool solve( std::array<std::array<double,n>,n> A, std::array<double,n> &b ) {
    for ( int c = 0; c < n; ++c ) {
        int p = c;
        for ( int r = c + 1; r < n; ++r ) if ( std::fabs( A[ r ][ c ] ) > std::fabs( A[ p ][ c ] ) ) p = r;
        if ( std::fabs( A[ p ][ c ] ) < 1e-300 ) return false;
        std::swap( A[ p ], A[ c ] ); std::swap( b[ p ], b[ c ] );
        for ( int r = 0; r < n; ++r ) {
            if ( r == c ) continue;
            const double f = A[ r ][ c ] / A[ c ][ c ];
            for ( int k = c; k < n; ++k ) A[ r ][ k ] -= f * A[ c ][ k ];
            b[ r ] -= f * b[ c ];
        }
    }
    for ( int c = 0; c < n; ++c ) b[ c ] /= A[ c ][ c ];
    return true;
}

/// the candidates of an element of M: its `k + 1` vertices `x`, `rho` there, the directions `leaving` it in the simplices of M
/// containing it, whether it is a simplex of M
template<int D,int k> void candidates_of( const std::array<Pt<D>,k + 1> &x, const std::array<double,k + 1> &r, const std::vector<Pt<D>> &leaving,
                                          bool maximal, double alpha, const Poly<D> &box, double tol, double amin, std::vector<Candidate<D>> &out ) {
    constexpr int m = D - k;
    std::vector<Pt<D>> dirs;
    for ( int i = 1; i <= k; ++i ) dirs.push_back( sub( x[ i ], x[ 0 ] ) );
    std::array<Pt<D>,D> b;
    if ( basis_of<D>( dirs, b, tol ) != k )
        return;                                                 // ( a flat element )

    // the tangent half-spaces: the prism over the element
    std::vector<HalfSpace<D>> tang;
    for ( int i = 0; i <= k && k >= 1; ++i ) {                  // ( the facet opposite to i, its outward normal in the tangent space )
        std::vector<Pt<D>> fd;
        const int f0 = i == 0 ? 1 : 0;
        for ( int j = 0; j <= k; ++j ) if ( j != i && j != f0 ) fd.push_back( sub( x[ j ], x[ f0 ] ) );
        Pt<D> nu = sub( x[ f0 ], x[ i ] );
        std::array<Pt<D>,D> fb;
        const int nf = basis_of<D>( fd, fb, tol );
        for ( int j = 0; j < nf; ++j ) nu = sub( nu, mul( fb[ j ], dot( nu, fb[ j ] ) ) );
        for ( int j = k; j < D; ++j ) nu = sub( nu, mul( b[ j ], dot( nu, b[ j ] ) ) );
        tang.push_back( HalfSpace<D>::normalized( nu, dot( nu, x[ f0 ] ) ) );
    }

    // rho: affine along the element, constant across
    Affine<D> rho;
    {
        std::array<std::array<double,k>,k> A{};
        std::array<double,k> a{};
        for ( int i = 0; i < k; ++i ) {
            for ( int j = 0; j < k; ++j ) A[ i ][ j ] = dot( sub( x[ i + 1 ], x[ 0 ] ), b[ j ] );
            a[ i ] = r[ i + 1 ] - r[ 0 ];
        }
        if constexpr ( k > 0 ) solve<k>( A, a );
        for ( int j = 0; j < k; ++j ) rho.g = add( rho.g, mul( b[ j ], a[ j ] ) );
        rho.c = r[ 0 ] - dot( rho.g, x[ 0 ] );
    }

    // the polar cone, in the coordinates of the normal space
    std::vector<Pt<m>> cons;
    for ( const Pt<D> &e : leaving ) {
        Pt<m> c;
        for ( int j = 0; j < m; ++j ) c[ j ] = dot( e, b[ k + j ] );
        if ( norm( c ) > tol ) cons.push_back( mul( c, 1 / norm( c ) ) );
    }
    Pt<D> base{};
    for ( int i = 0; i <= k; ++i ) base = add( base, mul( x[ i ], 1.0 / ( k + 1 ) ) );
    double rad = 0;
    for ( int i = 0; i <= k; ++i ) rad = std::max( rad, norm( sub( x[ i ], base ) ) );
    for ( const auto &sec : cone_sectors<m>( cons, alpha, 1e-12 ) ) {
        std::array<Pt<D>,m> ray;
        for ( int i = 0; i < m; ++i ) {
            ray[ i ] = Pt<D>{};
            for ( int j = 0; j < m; ++j ) ray[ i ] = add( ray[ i ], mul( b[ k + j ], sec[ i ][ j ] ) );
        }
        // the dual basis in the normal space: `w_j . ray_i = delta_ij`; the cone is `w_j . u >= 0`, `d = sum_j w_j . u`
        std::array<std::array<double,m>,m> R{};
        for ( int i = 0; i < m; ++i ) for ( int j = 0; j < m; ++j ) R[ i ][ j ] = sec[ i ][ j ];
        Candidate<D> c;
        c.hs = tang;
        Pt<D> g{};
        for ( int j = 0; j < m; ++j ) {
            std::array<double,m> e{};                           // ( R w = e_j in normal coordinates: the column j of R^-1 )
            e[ j ] = 1;
            if ( ! solve<m>( R, e ) ) return;
            Pt<D> wj{};
            for ( int l = 0; l < m; ++l ) wj = add( wj, mul( b[ k + l ], e[ l ] ) );
            g = add( g, wj );
            c.hs.push_back( HalfSpace<D>::normalized( mul( wj, -1.0 ), -dot( wj, x[ 0 ] ) ) );
        }
        c.d = { g, -dot( g, x[ 0 ] ) };
        c.rho = rho;
        if ( maximal && m == 2 ) {                              // ( the weight of a sector of angle theta: theta / sin theta )
            const double ct = std::clamp( dot( sec[ 0 ], sec[ 1 ] ), -1.0, 1.0 ), th = std::acos( ct );
            c.w = th / std::sin( th );
        }
        c.base = base;
        c.rad = rad;
        c.poly = box;
        for ( const HalfSpace<D> &h : c.hs ) {
            c.poly = c.poly.clip( h, tol );
            if ( c.poly.empty() ) break;
        }
        if ( c.poly.empty() || c.poly.measure() <= amin )
            continue;
        out.push_back( std::move( c ) );
    }
}

/// the pieces of the candidate `ia`: what is left of it once every other candidate has taken where its `d` is smaller
template<int D> std::vector<Poly<D>> pieces_of( const std::vector<Candidate<D>> &cand, std::size_t ia, double tol, double amin ) {
    const Candidate<D> &A = cand[ ia ];
    std::vector<std::size_t> near;
    for ( std::size_t ib = 0; ib < cand.size(); ++ib )
        if ( ib != ia && boxes_overlap<D>( A.lo, A.hi, cand[ ib ].lo, cand[ ib ].hi ) )
            near.push_back( ib );
    // the closest first ( A shrinks fast ), by a lower bound of the distance between the elements
    std::vector<double> dist( cand.size() );
    for ( std::size_t ib : near ) dist[ ib ] = norm( sub( cand[ ib ].base, A.base ) ) - cand[ ib ].rad - A.rad;
    std::stable_sort( near.begin(), near.end(), [&]( std::size_t a, std::size_t b ) { return dist[ a ] < dist[ b ]; } );

    // the security radius: `d >= ` the distance to the element ( exact, or over-estimated by the fans ), so on the remaining
    // fragments, where `d_A <= r`, `d_B >= dist( elements ) - r`: B can only win when `dist( elements ) <= 2 r`
    auto r_of = [&]( const Poly<D> &p ) { double r = 0; for ( const Pt<D> &x : p.vertices() ) r = std::max( r, A.d( x ) ); return r; };

    // ( `fresh`: the fragments cut by the last step, the only ones that can merge with others )
    std::vector<Poly<D>> frags{ A.poly };
    std::vector<char> fresh{ 0 };
    for ( std::size_t ib : near ) {
        if ( frags.empty() ) break;
        double r = 0;
        for ( const Poly<D> &f : frags ) r = std::max( r, r_of( f ) );
        if ( dist[ ib ] > 2 * r + tol ) break;
        const Candidate<D> &B = cand[ ib ];
        Pt<D> dg = sub( B.d.g, A.d.g );                         // d_B - d_A <= 0
        double dc = A.d.c - B.d.c;
        bool whole = false;                                     // ( parallel: B wins everywhere, or nowhere; ties to the smallest index )
        if ( norm( dg ) < 1e-12 ) {
            if ( dc < -tol || ( dc <= tol && ib > ia ) ) continue;
            whole = true;
        }
        std::vector<Poly<D>> nxt;
        std::vector<char> nfresh;
        auto keep = [&]( Poly<D> &&p, char f ) { nxt.push_back( std::move( p ) ); nfresh.push_back( f ); };
        for ( Poly<D> &fr : frags ) {
            Pt<D> lo, hi;
            bbox<D>( fr, lo, hi );
            if ( ! boxes_overlap<D>( lo, hi, B.lo, B.hi ) ) { keep( std::move( fr ), 0 ); continue; }
            bool apart = false;                                 // ( a face of B leaving fr out )
            for ( const HalfSpace<D> &hb : B.hs ) {
                double m = 1e300;
                for ( const Pt<D> &x : fr.vertices() ) m = std::min( m, hb.dist( x ) );
                if ( ( apart = m >= -tol ) ) break;
            }
            if ( apart ) { keep( std::move( fr ), 0 ); continue; }
            std::vector<HalfSpace<D>> C = B.hs;
            if ( ! whole ) {
                const HalfSpace<D> h = HalfSpace<D>::normalized( dg, dc );
                double mi = 1e300;                              // ( B loses on all of fr: nothing to clip )
                for ( const Pt<D> &x : fr.vertices() ) mi = std::min( mi, h.dist( x ) );
                if ( mi >= -tol ) { keep( std::move( fr ), 0 ); continue; }
                const Poly<D> X = fr.clip( h, tol );            // fr ∩ { d_B <= d_A }
                if ( X.empty() || X.measure() <= amin ) { keep( std::move( fr ), 0 ); continue; }
                bool inside = true;                             // ( X in B: a single cut by the bisector )
                for ( const Pt<D> &x : X.vertices() ) for ( const HalfSpace<D> &hb : B.hs ) inside &= hb.dist( x ) <= tol;
                if ( inside ) {
                    Poly<D> Y = fr.clip( h.flipped(), tol );
                    if ( ! Y.empty() && Y.measure() > amin ) keep( std::move( Y ), 1 );
                    continue;
                }
                for ( const HalfSpace<D> &hb : B.hs ) {         // ( X ∩ B empty: nothing to take; first, a face of B separating them )
                    double m = 1e300;
                    for ( const Pt<D> &x : X.vertices() ) m = std::min( m, hb.dist( x ) );
                    if ( ( apart = m >= -tol ) ) break;
                }
                if ( apart ) { keep( std::move( fr ), 0 ); continue; }
                Poly<D> Z = X;
                for ( const HalfSpace<D> &hb : B.hs ) {
                    Z = Z.clip( hb, tol );
                    if ( Z.empty() ) break;
                }
                if ( Z.empty() || Z.measure() <= amin ) { keep( std::move( fr ), 0 ); continue; }
                C.push_back( h );
            }
            Poly<D> rest = fr;                                  // fr \ C, half-space by half-space
            for ( const HalfSpace<D> &hc : C ) {
                Poly<D> o = rest.clip( hc.flipped(), tol );
                if ( ! o.empty() && o.measure() > amin ) keep( std::move( o ), 1 );
                rest = rest.clip( hc, tol );
                if ( rest.empty() ) break;
            }
        }
        frags = std::move( nxt );
        fresh = std::move( nfresh );
        if ( frags.size() > 1 && std::find( fresh.begin(), fresh.end(), 1 ) != fresh.end() ) merge_all<D>( frags, fresh, tol, amin );
        else fresh.assign( frags.size(), 0 );
    }
    return frags;
}

/// `f( i )` for `i` in `[ 0, n )`, the threads taking the indices one by one ( uneven costs )
template<class F> void par_for( const CpuQueue &queue, SI n, F &&f ) {
    const int nt = int( std::min<SI>( std::max( queue.nb_workers(), 1 ), std::max<SI>( n, 1 ) ) );
    if ( nt <= 1 ) { for ( SI i = 0; i < n; ++i ) f( i ); return; }
    std::atomic<SI> next{ 0 };
    queue.run_threads( nt, [&]( int ) { for ( SI i; ( i = next++ ) < n; ) f( i ); } );
}

/// THE ENTRY POINT: `nodes( N, D )`, `simplices( E, K )` ( K = 2: lines, K = 3: surfaces in 3D ), `rho( N )`, `params`: `lo[ D ]`,
/// `hi[ D ]`, `alpha` ( radians ). Outputs: `pts( cap, D + 1, D )`, `d( cap, D + 1 )`, `r( cap, D + 1 )`, `w( cap )`, `stats`:
/// number of simplices, candidates, pieces
template<int D,int K>
void run( const CpuQueue &queue, const auto &nodes, const auto &simplices, const auto &rho, const auto &params,
          auto &&pts, auto &&d, auto &&r, auto &&w, auto &&stats ) {
    const SI E = SI( simplices.shape( 0 ) ), cap = SI( w.shape( 0 ) );
    Pt<D> lo, hi;
    for ( int i = 0; i < D; ++i ) { lo[ i ] = double( params( i ) ); hi[ i ] = double( params( D + i ) ); }
    const double alpha = double( params( 2 * D ) );
    const bool raw = params.shape( 0 ) > 2 * D + 1 && params( 2 * D + 1 ) != 0;   // ( the candidates, without the minimum: a check )
    const double scale = norm( sub( hi, lo ) ), tol = 1e-11 * scale, amin = 1e-15 * std::pow( scale, D );
    const Poly<D> box = Poly<D>::box( lo, hi );
    auto X = [&]( SI n ) { Pt<D> p; for ( int i = 0; i < D; ++i ) p[ i ] = double( nodes( n, i ) ); return p; };

    // the elements: the simplices of M and all their faces ( sorted vertex lists ), and for each one the simplices containing it
    std::vector<std::vector<SI>> elems;
    std::vector<std::vector<SI>> owners;                        // ( the simplices of M containing each element )
    {
        std::vector<std::pair<std::vector<SI>,SI>> all;
        for ( SI e = 0; e < E; ++e )
            for ( int mask = 1; mask < ( 1 << K ); ++mask ) {
                std::vector<SI> v;
                for ( int i = 0; i < K; ++i ) if ( mask >> i & 1 ) v.push_back( SI( simplices( e, i ) ) );
                std::sort( v.begin(), v.end() );
                all.push_back( { v, e } );
            }
        std::sort( all.begin(), all.end() );
        for ( std::size_t i = 0; i < all.size(); ) {
            std::size_t j = i;
            elems.push_back( all[ i ].first );
            owners.push_back( {} );
            for ( ; j < all.size() && all[ j ].first == all[ i ].first; ++j ) owners.back().push_back( all[ j ].second );
            i = j;
        }
    }

    // the candidates
    std::vector<std::vector<Candidate<D>>> per( elems.size() );
    par_for( queue, SI( elems.size() ), [&]( SI q ) {
        const std::vector<SI> &v = elems[ q ];
        std::vector<Pt<D>> leaving;
        bool maximal = int( v.size() ) == K;
        for ( SI e : owners[ q ] )
            for ( int i = 0; i < K; ++i ) {
                const SI n = SI( simplices( e, i ) );
                if ( std::find( v.begin(), v.end(), n ) == v.end() )
                    leaving.push_back( sub( X( n ), X( v[ 0 ] ) ) );
            }
        auto make = [&]<int k>() {
            if constexpr ( k < D ) {
                std::array<Pt<D>,k + 1> x;
                std::array<double,k + 1> rr;
                for ( int i = 0; i <= k; ++i ) { x[ i ] = X( v[ i ] ); rr[ i ] = double( rho( v[ i ] ) ); }
                candidates_of<D,k>( x, rr, leaving, maximal, alpha, box, tol, amin, per[ q ] );
            }
        };
        switch ( v.size() ) {
            case 1: make.template operator()<0>(); break;
            case 2: make.template operator()<1>(); break;
            case 3: make.template operator()<2>(); break;
            default: break;
        }
    } );
    std::vector<Candidate<D>> cand;
    for ( auto &p : per ) for ( auto &c : p ) cand.push_back( std::move( c ) );
    for ( Candidate<D> &c : cand ) bbox<D>( c.poly, c.lo, c.hi );

    // the pieces, fanned
    struct Simplex { std::array<Pt<D>,D + 1> x; std::array<double,D + 1> d, r; double w; };
    std::vector<std::vector<Simplex>> res( cand.size() );
    std::vector<SI> nb_pieces( cand.size() );
    par_for( queue, SI( cand.size() ), [&]( SI ia ) {
        const Candidate<D> &A = cand[ ia ];
        const std::vector<Poly<D>> pcs = raw ? std::vector<Poly<D>>{ A.poly } : pieces_of<D>( cand, std::size_t( ia ), tol, amin );
        nb_pieces[ ia ] = SI( pcs.size() );
        for ( const Poly<D> &p : pcs ) {
            const auto &vs = p.vertices();
            std::size_t k = 0;
            for ( std::size_t i = 1; i < vs.size(); ++i ) if ( A.d( vs[ i ] ) > A.d( vs[ k ] ) ) k = i;
            auto push = [&]( const std::array<Pt<D>,D + 1> &s ) {
                Simplex o;
                o.x = s;
                for ( int i = 0; i <= D; ++i ) {
                    o.d[ i ] = std::max( A.d( s[ i ] ), 0.0 );
                    if ( o.d[ i ] < tol ) o.d[ i ] = 0;
                    o.r[ i ] = A.rho( s[ i ] );
                }
                o.w = A.w;
                double vol;                                     // ( the slivers out )
                if constexpr ( D == 2 ) vol = 0.5 * std::fabs( ( s[ 1 ][ 0 ] - s[ 0 ][ 0 ] ) * ( s[ 2 ][ 1 ] - s[ 0 ][ 1 ] ) - ( s[ 1 ][ 1 ] - s[ 0 ][ 1 ] ) * ( s[ 2 ][ 0 ] - s[ 0 ][ 0 ] ) );
                else vol = std::fabs( dot( sub( s[ 1 ], s[ 0 ] ), cross( sub( s[ 2 ], s[ 0 ] ), sub( s[ 3 ], s[ 0 ] ) ) ) ) / 6;
                if ( vol > amin ) res[ ia ].push_back( o );
            };
            if constexpr ( D == 2 ) p.for_each_simplex( k, push );
            else p.for_each_simplex( k, push, tol );
        }
    } );

    SI n = 0, np = 0;
    for ( SI ia = 0; ia < SI( cand.size() ); ++ia ) {
        np += nb_pieces[ ia ];
        for ( const Simplex &s : res[ ia ] ) {
            if ( n < cap ) {
                for ( int i = 0; i <= D; ++i ) {
                    for ( int c = 0; c < D; ++c ) pts( n, i, c ) = s.x[ i ][ c ];
                    d( n, i ) = s.d[ i ];
                    r( n, i ) = s.r[ i ];
                }
                w( n ) = s.w;
            }
            ++n;
        }
    }
    stats( 0 ) = n;
    stats( 1 ) = SI( cand.size() );
    stats( 2 ) = np;
}

} // namespace envelope
} // namespace sdot
