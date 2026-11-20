#pragma once

// =====================================================================================
// THE LOSS OF BALLS ON 1D DETECTORS IN ONE PASS: for each view, the deposit of the balls on the cells of the detector,
// the squared Wasserstein distance between that image and the data ( diracs at the centres of the cells ), and the
// derivative of that distance with respect to the centres ( and the radii, the masses ) of the balls, in ONE call,
// without the image or its potential going through anything else. On the card, three kernels ( the SPLIT path, the
// default: a deposit and a gather as parallel as `Projected1d.h`'s, the rows in a global scratch that stays in the L2
// cache, the transport a thread per row ), or one ( TILE < 0: a block per tile of views, its rows in shared memory --
// slower, its threads wait for their balls ); on the CPU, the rows split between the threads.
//
//   loss( b, a )            W2^2( normalized image of view a, normalized data of view a )
//   grad_points( b, i, : )  d sum_a loss( b, a ) / d point i
//   grad_radii( b, i )      d sum_a loss( b, a ) / d radius i       ( MODE 2 only: at fixed mass )
//   grad_masses( b, i )     d sum_a loss( b, a ) / d mass i         ( MODE 2 only )
//
// THE GEOMETRY is a parameter ( `Geometry` ): what a view does to a point, `( s, scale )` -- where its centre falls on
// the detector, and the factor of its width there -- with their derivatives. `Parallel`: s = p . n_a, scale = 1. Other
// geometries ( a sample that moves during the acquisition, a fan or a cone beam: a magnification ) give the same three
// things; the profile, the transport and the gather do not change.
//
// THE BALLS: MODE = 0, one radius for all ( `radii( b, 0 )` ) and equal masses ( 1 / n ); 1, a radius and a mass each
// ( `radii( b, i )`, `masses( b, i )`: sized balls, the mass following the size ), fixed; 2, the same, with the
// derivatives of the loss with respect to them ( the caller chains them to its parameters ). The profile is that of `Projected1d.h`: a dome ( 1 - t^2 )^K ( K = 0: the half
// power, a disc ), G its cumulative. `params`: [ s_min, h, max_w, background, c, a_0 .. a_K ]; the image is lifted by
// `background` times its mean before the transport ( as `loss.Model.finish`, no blur ), the data are taken as given.
//
// THE TRANSPORT of a row: the image is uniform inside each cell, the data are diracs at the centres y_i of the cells. In
// 1D the plan is monotone; with T the map and phi the potential ( phi' = 2 ( x - T( x ) ) ),
//     W2^2 = sum of int ( x - y_i )^2 f( x ) dx over the pieces where T = y_i,
//     d W2^2 / d ( normalized mass of cell c ) = the mean of phi over the cell ( Phi_c ),
// one sweep over the cells, in double; then the normalization and the background by the chain rule.
// =====================================================================================

#include <loom/support/common_macros.h>
#include <loom/support/kernels/CpuQueue.h>

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>
#include <vector>

#if defined( __CUDACC__ )
#define SDOT_P1F_HD __host__ __device__ __forceinline__
#else
#define SDOT_P1F_HD inline
#endif

namespace sdot {
namespace projected1d_fused {

/// a strided view ( byte strides ), readable on the host and on the card
template<class T,int N>
struct View {
    char *p = nullptr;
    SI    s[ N ] = {};
    SDOT_P1F_HD T &operator()( SI i ) const { return *reinterpret_cast<T *>( p + i * s[ 0 ] ); }
    SDOT_P1F_HD T &operator()( SI i, SI j ) const { return *reinterpret_cast<T *>( p + i * s[ 0 ] + j * s[ 1 ] ); }
    SDOT_P1F_HD T &operator()( SI i, SI j, SI k ) const { return *reinterpret_cast<T *>( p + i * s[ 0 ] + j * s[ 1 ] + k * s[ 2 ] ); }
};

template<class V>
auto view( const V &v ) {
    using T = std::remove_const_t<typename std::decay_t<V>::TF>;
    constexpr int N = std::decay_t<V>::ct_rank;
    View<T,N> res;
    res.p = reinterpret_cast<char *>( const_cast<T *>( v.data().raw ) );
    [&]<int... I>( std::integer_sequence<int,I...> ) {
        ( ( res.s[ I ] = SI( v._strides[ Ct<int,I>() ] ) ), ... );
    }( std::make_integer_sequence<int,N>() );
    return res;
}

constexpr int MAX_K = 8;

/// the profile and the grid
struct Profile {
    double s_min, h, ih, max_w, bg, c, a[ MAX_K + 1 ];
    float  cf, af[ MAX_K + 1 ];                               ///< the same, in float ( `PRECISION` )
    SI     M;
    int    K;

    /// G and f in the float `T` ( `double`, or `float` for the profile at `PRECISION` >= 1 )
    template<class T>
    SDOT_P1F_HD T Gt( T t ) const {
        if constexpr ( std::is_same_v<T,double> ) {
            return G( t );
        } else {
            if ( t <= T( -1 ) ) return T( 0 );
            if ( t >= T( 1 ) ) return T( 1 );
            if ( K == 0 )
                return T( 0.5 ) + cf / 2 * ( t * sqrtf( 1 - t * t ) + asinf( t ) );
            const T t2 = t * t;
            T s = T( 0.5 ), p = t;
            for ( int j = 0; j <= K; ++j ) {
                s += af[ j ] * p;
                p *= t2;
            }
            return s;
        }
    }
    template<class T>
    SDOT_P1F_HD T ft( T t ) const {
        if constexpr ( std::is_same_v<T,double> ) {
            return f( t );
        } else {
            const T v = 1 - t * t;
            if ( ! ( v > 0 ) )
                return T( 0 );
            if ( K == 0 )
                return cf * sqrtf( v );
            T r = cf;
            for ( int j = 0; j < K; ++j )
                r *= v;
            return r;
        }
    }

    /// the cumulative of the profile at t ( exactly 0 / 1 beyond the support )
    SDOT_P1F_HD double G( double t ) const {
        if ( t <= -1 ) return 0;
        if ( t >= 1 ) return 1;
        if ( K == 0 )
            return 0.5 + c / 2 * ( t * sqrt( 1 - t * t ) + asin( t ) );
        const double t2 = t * t;
        double s = 0.5, p = t;
        for ( int j = 0; j <= K; ++j ) {
            s += a[ j ] * p;
            p *= t2;
        }
        return s;
    }

    /// the profile at t
    SDOT_P1F_HD double f( double t ) const {
        const double v = 1 - t * t;
        if ( ! ( v > 0 ) )
            return 0;
        if ( K == 0 )
            return c * sqrt( v );
        double r = c;
        for ( int j = 0; j < K; ++j )
            r *= v;
        return r;
    }

    /// the cells a support [ s - w, s + w ] meets: the first one, their number ( clamped to the grid: what leaves it
    /// stays in the border cells, the two extreme edges being at -inf / +inf )
    SDOT_P1F_HD SI cells( double s, double w, SI &j0 ) const {
        const double u = ( s - s_min ) * ih, wc = w * ih;
        SI j1 = SI( floor( u + wc ) );
        j0 = SI( floor( u - wc ) );
        j0 = j0 < 0 ? 0 : ( j0 > M - 1 ? M - 1 : j0 );
        j1 = j1 < j0 ? j0 : ( j1 > M - 1 ? M - 1 : j1 );
        return j1 - j0 + 1;
    }

    /// the parameter of edge e for a ball at s ( `iw`: 1 / its half-width )
    SDOT_P1F_HD double edge_t( SI e, double s, double iw ) const {
        return e == 0 ? -2.0 : ( e == M ? 2.0 : ( s_min + double( e ) * h - s ) * iw );
    }
};

template<class VP>
SDOT_P1F_HD Profile read_profile( const VP &params, SI M, int K ) {
    Profile pr;
    pr.s_min = params( 0 ); pr.h = params( 1 ); pr.max_w = params( 2 ); pr.bg = params( 3 ); pr.c = params( 4 );
    for ( int j = 0; j <= MAX_K; ++j )
        pr.a[ j ] = j <= K ? params( 5 + j ) : 0.0;
    pr.ih = 1 / pr.h;
    pr.cf = float( pr.c );
    for ( int j = 0; j <= MAX_K; ++j )
        pr.af[ j ] = float( pr.a[ j ] );
    pr.M = M;
    pr.K = K;
    return pr;
}

// ── the geometries ────────────────────────────────────────────────────────────

/// PARALLEL beams, the sample still: view a sees p at s = p . n_a, at its own width
struct Parallel {
    View<double,2> normals;                                   ///< [ A, 2 ]: the detector axis of each view

    /// `( s, scale )` of the point p in view a, and their derivatives with respect to p
    SDOT_P1F_HD void project( SI a, double px, double py, double &s, double &scale, double ds[ 2 ], double dscale[ 2 ] ) const {
        const double nx = normals( a, 0 ), ny = normals( a, 1 );
        s = px * nx + py * ny;
        scale = 1;
        ds[ 0 ] = nx; ds[ 1 ] = ny;
        dscale[ 0 ] = 0; dscale[ 1 ] = 0;
    }
};

// ── one ball, one view ────────────────────────────────────────────────────────

/// PRECISION P ( as `Projected1d.h` ): 0, everything in double; 1, the PROFILE in float ( the parameters t of the
/// edges, from the offset of the first one in double, by steps of h / w, and G, f of them ); 2, the SUMS too ( along a
/// window: the differences of g taken in double, then in float; over the views ). The projection, the window and the
/// cells always in double. ( `Projected1d.h`'s level 3, the projection in float too, is not offered here )
template<int P>
struct Prec {
    using T = std::conditional_t<( P >= 1 ),float,double>;   ///< the profile
    using A = std::conditional_t<( P >= 2 ),float,double>;   ///< the sums
};

/// the masses of a ball at s, of half-width w ( `iw` = 1 / w ), added to the row by `add( cell, mass )`
template<int P,class Add>
SDOT_P1F_HD void deposit_ball( const Profile &pr, Add &&add, double s, double w, double iw, double mass ) {
    using T = typename Prec<P>::T;
    using A = typename Prec<P>::A;
    SI j0;
    const SI n = pr.cells( s, w, j0 );
    const T t0 = T( ( pr.s_min + double( j0 ) * pr.h - s ) * iw ), dt = T( pr.h * iw );
    T G_prev = j0 == 0 ? T( 0 ) : pr.Gt<T>( t0 );
    for ( SI l = 0; l < n; ++l ) {
        const T G_next = j0 + l + 1 == pr.M ? T( 1 ) : pr.Gt<T>( t0 + T( l + 1 ) * dt );
        const double m = double( A( G_next - G_prev ) * A( mass ) );
        G_prev = G_next;
        if ( m != 0 )
            add( j0 + l, m );
    }
}

/// over the edges of the window of a ball at s, with g the derivative of the loss with respect to the cell masses
/// ( `g( c )`, c = 0 .. M - 1, in double ): ds = sum_e f( t_e ) dg_e and dw = sum_e t_e f( t_e ) dg_e ( dg_e = g_e -
/// g_e-1, 0 on the two end edges ), what the derivatives with respect to s and w of sum_l g_l m_l are made of; with
/// MASS, dm = sum_l g_l x the share of cell l, summed BY PARTS ( g_last G_n - g_first G_0 - sum_e G_e dg_e ), in double
/// ( the normalization of the masses takes a mean off it: its errors count 1e4 times more ), G as the deposit took it
template<int P,bool MASS,class Gr,class A>
SDOT_P1F_HD void gather_ball( const Profile &pr, const Gr &g, double s, double w, double iw, A &ds, A &dw, double &dm ) {
    using T = typename Prec<P>::T;
    SI j0;
    const SI n = pr.cells( s, w, j0 );
    const T t0 = T( ( pr.s_min + double( j0 ) * pr.h - s ) * iw ), dt = T( pr.h * iw );
    ds = 0;
    dw = 0;
    double sum_Gdg = 0;
    for ( SI l = 0; l <= n; ++l ) {
        const SI e = j0 + l;
        if ( e == 0 || e == pr.M )
            continue;
        const T t = t0 + T( l ) * dt;
        const double dd = g( e ) - g( e - 1 );
        // dm: EVERY inner edge, with the G the deposit took there ( a t rounded onto the support's end in float must not
        // drop a term: G is 1 or 0 there, not its derivative )
        if ( MASS && l > 0 && l < n )
            sum_Gdg += double( pr.Gt<T>( t ) ) * dd;
        if ( ! ( t > T( -1 ) && t < T( 1 ) ) )
            continue;
        const A d = A( dd );
        const T fv = pr.ft<T>( t );
        ds += d * A( fv );
        dw += d * A( t * fv );
    }
    if ( MASS ) {                                              // ( the G of the two ends as the deposit took them )
        const double G0 = j0 == 0 ? 0.0 : double( pr.Gt<T>( t0 ) );
        const double Gn = j0 + n == pr.M ? 1.0 : double( pr.Gt<T>( t0 + T( n ) * dt ) );
        dm = g( j0 + n - 1 ) * Gn - g( j0 ) * G0 - sum_Gdg;
    } else {
        dm = 0;
    }
}

// ── one row: the transport ────────────────────────────────────────────────────

/// `row [ M ]` holds the deposit of the view; `nu( i )` the data. Returns W2^2 between the two, normalized, the image
/// lifted by `bg` times its mean, and leaves in `row` the derivative of W2^2 with respect to the deposit
template<class Row,class Nu>
SDOT_P1F_HD double transport_row( const Profile &pr, Row &&row, const Nu &nu ) {
    const SI M = pr.M;
    const double h = pr.h;
    double S0 = 0, Sn = 0;
    for ( SI c = 0; c < M; ++c ) {
        S0 += row( c );
        Sn += nu( c );
    }
    const double lift = pr.bg * S0 / double( M ), S = S0 * ( 1 + pr.bg ), iS = 1 / S, iSn = 1 / Sn;

    // the sweep: the potential phi ( 0 at the start of the grid ), its mean over each cell into row( c )
    double loss = 0, phi = 0, F = 0, Cn = nu( 0 ) * iSn, sum_Phi_m = 0;
    SI k = 0;
    for ( SI c = 0; c < M; ++c ) {
        const double m = ( row( c ) + lift ) * iS, f = m / h, x1 = double( c + 1 ) * h;
        double x = double( c ) * h, int_phi = 0;
        if ( ! ( m > 0 ) )
            while ( k < M - 1 && Cn <= F ) { ++k; Cn += nu( k ) * iSn; }
        while ( true ) {
            double xe = x1;
            const bool cut = m > 0 && k < M - 1 && Cn < F + m;
            if ( cut )
                xe = double( c ) * h + ( Cn - F ) / f;
            xe = xe < x ? x : ( xe > x1 ? x1 : xe );
            const double y = ( double( k ) + 0.5 ) * h, a = x - y, b = xe - y;
            const double cube = ( b * b * b - a * a * a ) / 3;
            loss += f * cube;
            int_phi += ( phi - a * a ) * ( xe - x ) + cube;
            phi += b * b - a * a;
            x = xe;
            if ( ! cut )
                break;
            ++k;
            Cn += nu( k ) * iSn;
        }
        F += m;
        const double Phi = int_phi / h;
        row( c ) = Phi;
        sum_Phi_m += Phi * m;
    }

    // d W2^2 / d deposit: the normalization ( by S ), then the background ( lift = bg S0 / M )
    double sum_g = 0;
    for ( SI c = 0; c < M; ++c ) {
        const double g = ( row( c ) - sum_Phi_m ) * iS;
        row( c ) = g;
        sum_g += g;
    }
    const double extra = pr.bg / double( M ) * sum_g;
    for ( SI c = 0; c < M; ++c )
        row( c ) += extra;
    return loss;
}

// ── one ball over the views of a tile: what it gathers ────────────────────────

/// MODE 0: one radius ( `R( b, 0 )` ), masses 1 / n; 1: a radius and a mass each ( `R( b, i )`, `Ms( b, i )` ), fixed;
/// 2: the same, with the derivatives with respect to them
template<int MODE>
struct Ball {
    double px, py, r, rc, mass;
    template<class VP,class VR,class VM>
    SDOT_P1F_HD Ball( const Profile &pr, const VP &P, const VR &R, const VM &Ms, SI b, SI i, SI N ) {
        px = P( b, i, 0 ); py = P( b, i, 1 );
        r = MODE == 0 ? R( b, 0 ) : R( b, i );
        rc = r < pr.max_w ? r : pr.max_w;
        mass = MODE == 0 ? 1.0 / double( N ) : Ms( b, i );
    }
};

/// adds to `o [ 3 ]` ( d/dx, d/dy, d/dr ) and `om` ( d/dmass ) what the row `g` of view a gives the ball ( `O`: the float
/// of the sums over the views )
template<int P,int MODE,class Geo,class Gr,class O>
SDOT_P1F_HD void gather_view( const Profile &pr, const Geo &geo, SI a, const Ball<MODE> &bl, const Gr &g, O *o, double &om ) {
    using A = typename Prec<P>::A;
    double s, scale, ds[ 2 ], dsc[ 2 ];
    geo.project( a, bl.px, bl.py, s, scale, ds, dsc );
    const double w = bl.rc * scale, iw = 1 / w;
    A gs, gw;
    double gm;
    gather_ball<P,MODE == 2>( pr, g, s, w, iw, gs, gw, gm );
    const A sc = A( bl.mass * iw ), dLds = gs * sc, dLdw = gw * sc;       // ( d/ds of the masses of the cells: - f / w )
    o[ 0 ] += O( dLds * A( ds[ 0 ] ) + dLdw * A( bl.rc * dsc[ 0 ] ) );
    o[ 1 ] += O( dLds * A( ds[ 1 ] ) + dLdw * A( bl.rc * dsc[ 1 ] ) );
    o[ 2 ] += O( dLdw * A( scale ) );
    om += gm;                                                 // ( in double: the normalization of the masses takes a mean off )
}

// ── CPU ───────────────────────────────────────────────────────────────────────
//
// points [ B, n, 2 ], radii [ B, n ] ( or [ B, 1 ] ), masses [ B, n ] ( MODE 0: unused ), data [ B, A, M ],
// loss [ B, A ], grad_points [ B, n, 2 ], grad_radii and grad_masses [ B, n ] ( MODE 2 ). The rows split between the
// threads, each with its own row and its own gradients, added at the end.

template<int K,int MODE,int P = 0,int BLOCK = 256,int TILE = 0,int DEBUG = 0>
void loss_balls( const CpuQueue &queue, const auto &points, const auto &radii, const auto &masses, const auto &normals, const auto &params,
                 const auto &data, auto &&loss, auto &&grad_points, auto &&grad_radii, auto &&grad_masses, auto && /* rows: the card's */ ) {
    const auto Pt = view( points );
    const auto R = view( radii );
    const auto Ms = view( masses );
    const auto D = view( data );
    const auto L = view( loss );
    const auto GP = view( grad_points );
    const auto GR = view( grad_radii );
    const auto GM = view( grad_masses );
    const SI B = points.shape( 0 ), N = points.shape( 1 ), A = normals.shape( 0 ), M = data.shape( 2 );
    const Profile pr = read_profile( view( params ), M, K );
    const Parallel geo{ view( normals ) };
    const SI rows = B * A;
    const int nt = int( std::min<SI>( std::max( queue.nb_workers(), 1 ), std::max<SI>( rows, 1 ) ) );
    std::vector<std::vector<double>> acc( nt, std::vector<double>( size_t( B * N * 4 ), 0.0 ) );
    queue.run_threads( nt, [&]( int th ) {
        std::vector<double> row( static_cast<size_t>( M ), 0.0 );
        auto &g = acc[ th ];
        for ( SI r = rows * th / nt; r < rows * ( th + 1 ) / nt; ++r ) {
            const SI b = r / A, a = r % A;
            std::fill( row.begin(), row.end(), 0.0 );
            for ( SI i = 0; i < N; ++i ) {
                const Ball<MODE> bl( pr, Pt, R, Ms, b, i, N );
                double s, scale, ds[ 2 ], dsc[ 2 ];
                geo.project( a, bl.px, bl.py, s, scale, ds, dsc );
                const double w = bl.rc * scale;
                deposit_ball<P>( pr, [&]( SI j, double m ) { row[ j ] += m; }, s, w, 1 / w, bl.mass );
            }
            L( b, a ) = transport_row( pr, [&]( SI j ) -> double & { return row[ j ]; }, [&]( SI j ) { return double( D( b, a, j ) ); } );
            for ( SI i = 0; i < N; ++i ) {
                double *o = &g[ size_t( 4 * ( b * N + i ) ) ];
                gather_view<P>( pr, geo, a, Ball<MODE>( pr, Pt, R, Ms, b, i, N ), [&]( SI e ) { return row[ e ]; }, o, o[ 3 ] );
            }
        }
    } );
    for ( SI q = 0; q < B * N; ++q ) {
        double o[ 4 ] = { 0, 0, 0, 0 };
        for ( auto &g : acc )
            for ( int c = 0; c < 4; ++c )
                o[ c ] += g[ size_t( 4 * q + c ) ];
        const SI b = q / N, i = q % N;
        GP( b, i, 0 ) = o[ 0 ];
        GP( b, i, 1 ) = o[ 1 ];
        if constexpr ( MODE == 2 ) {
            GR( b, i ) = R( b, i ) < pr.max_w ? o[ 2 ] : 0.0;
            GM( b, i ) = o[ 3 ];
        }
    }
}

// ── FIELDS: what the views ask of each ball ( to colour the shapes with; `otrec.loss.fields` ) ──────────────────────
//
// For each ball, over the views: sum delta n ( `fb [ B, n, 2 ]` ), sum delta^2 ( `fs [ B, n, 0 ]` ), sum of what the data
// see under it against what the balls put there ( `fs [ B, n, 1 ]` ), and the number of views where that is under 0.25
// ( `fs [ B, n, 2 ]` ). delta = - ( d loss / d u ) S / ( 2 m ): the move the view asks of the ball along its axis ( S
// the mass of the lifted image of the view ). In double. The rows: the derivative g ( `H` ), the image before the
// transport ( `I` ), and per row S0, Sn ( the masses of the image and of the data, `RS` ).

/// the sums of one ball over one view ( the row a: g, image, data, and its masses S0, Sn )
template<int MODE,class Geo,class Gr,class Im,class Da>
SDOT_P1F_HD void fields_view( const Profile &pr, const Geo &geo, SI a, const Ball<MODE> &bl, const Gr &g, const Im &img, const Da &dat, double S0, double Sn,
                              double *fb, double *fs ) {
    double s, scale, ds[ 2 ], dsc[ 2 ];
    geo.project( a, bl.px, bl.py, s, scale, ds, dsc );
    const double w = bl.rc * scale, iw = 1 / w, S = S0 * ( 1 + pr.bg ), lift = pr.bg * S0 / double( pr.M );
    double gs, gw, gm;
    gather_ball<0,false>( pr, g, s, w, iw, gs, gw, gm );
    const double delta = - gs * iw * S / 2;
    fb[ 0 ] += delta * ds[ 0 ];
    fb[ 1 ] += delta * ds[ 1 ];
    fs[ 0 ] += delta * delta;
    SI j0;
    const SI n = pr.cells( s, w, j0 );
    double G_prev = pr.G( pr.edge_t( j0, s, iw ) ), seen = 0;
    for ( SI l = 0; l < n; ++l ) {
        const double G_next = pr.G( pr.edge_t( j0 + l + 1, s, iw ) );
        seen += ( G_next - G_prev ) * ( dat( j0 + l ) / Sn ) / ( ( img( j0 + l ) + lift ) / S );
        G_prev = G_next;
    }
    fs[ 1 ] += seen;
    fs[ 2 ] += seen < 0.25 ? 1.0 : 0.0;
}

template<int K,int MODE>
void fields_balls( const CpuQueue &queue, const auto &points, const auto &radii, const auto &masses, const auto &normals, const auto &params,
                   const auto &data, auto &&fb, auto &&fs, auto && /* rows: the card's */, auto && /* image */, auto && /* row sums */ ) {
    const auto Pt = view( points );
    const auto R = view( radii );
    const auto Ms = view( masses );
    const auto D = view( data );
    const auto FB = view( fb );
    const auto FS = view( fs );
    const SI B = points.shape( 0 ), N = points.shape( 1 ), A = normals.shape( 0 ), M = data.shape( 2 );
    const Profile pr = read_profile( view( params ), M, K );
    const Parallel geo{ view( normals ) };
    const SI rows = B * A;
    const int nt = int( std::min<SI>( std::max( queue.nb_workers(), 1 ), std::max<SI>( rows, 1 ) ) );
    std::vector<std::vector<double>> acc( nt, std::vector<double>( size_t( B * N * 5 ), 0.0 ) );
    queue.run_threads( nt, [&]( int th ) {
        std::vector<double> row( static_cast<size_t>( M ), 0.0 ), img( static_cast<size_t>( M ), 0.0 );
        auto &o = acc[ th ];
        for ( SI r = rows * th / nt; r < rows * ( th + 1 ) / nt; ++r ) {
            const SI b = r / A, a = r % A;
            std::fill( row.begin(), row.end(), 0.0 );
            for ( SI i = 0; i < N; ++i ) {
                const Ball<MODE> bl( pr, Pt, R, Ms, b, i, N );
                double s, scale, ds[ 2 ], dsc[ 2 ];
                geo.project( a, bl.px, bl.py, s, scale, ds, dsc );
                const double w = bl.rc * scale;
                deposit_ball<0>( pr, [&]( SI j, double m ) { row[ j ] += m; }, s, w, 1 / w, bl.mass );
            }
            img = row;
            double S0 = 0, Sn = 0;
            for ( SI c = 0; c < M; ++c ) { S0 += row[ c ]; Sn += double( D( b, a, c ) ); }
            transport_row( pr, [&]( SI j ) -> double & { return row[ j ]; }, [&]( SI j ) { return double( D( b, a, j ) ); } );
            for ( SI i = 0; i < N; ++i ) {
                double *q = &o[ size_t( 5 * ( b * N + i ) ) ];
                fields_view( pr, geo, a, Ball<MODE>( pr, Pt, R, Ms, b, i, N ), [&]( SI e ) { return row[ e ]; }, [&]( SI j ) { return img[ j ]; },
                             [&]( SI j ) { return double( D( b, a, j ) ); }, S0, Sn, q, q + 2 );
            }
        }
    } );
    for ( SI q = 0; q < B * N; ++q ) {
        double o[ 5 ] = { 0, 0, 0, 0, 0 };
        for ( auto &g : acc )
            for ( int c = 0; c < 5; ++c )
                o[ c ] += g[ size_t( 5 * q + c ) ];
        const SI b = q / N, i = q % N;
        FB( b, i, 0 ) = o[ 0 ]; FB( b, i, 1 ) = o[ 1 ];
        FS( b, i, 0 ) = o[ 2 ]; FS( b, i, 1 ) = o[ 3 ]; FS( b, i, 2 ) = o[ 4 ];
    }
}

// ── CUDA ──────────────────────────────────────────────────────────────────────
#if defined( __CUDACC__ )

/// a block: one problem b and a tile of `TA` views ( its rows in shared memory, `M` each )
/// DEBUG ( timing only, results wrong ): 1, no transport; 2, no gather; 4, the deposit without atomics ( racy )
template<int K,int MODE,int P,int BLOCK,int DEBUG,class VP,class VR,class VM,class VD,class VL,class VGP,class VGR,class VGM>
__global__ void __launch_bounds__( BLOCK ) loss_balls_kernel( View<double,1> PP, Parallel geo, SI N, SI A, SI M, int TA, VP Pt, VR R, VM Ms, VD D,
                                                              VL L, VGP GP, VGR GR, VGM GM ) {
    extern __shared__ double rows[];
    const Profile pr = read_profile( PP, M, K );
    const SI tiles = ( A + TA - 1 ) / TA, b = blockIdx.x / tiles, a0 = SI( blockIdx.x % tiles ) * TA;
    const int ta = int( A - a0 < TA ? A - a0 : TA );
    for ( SI q = threadIdx.x; q < SI( ta ) * M; q += blockDim.x )
        rows[ q ] = 0;
    __syncthreads();

    // the deposit of every ball in the rows of the tile
    for ( SI i = threadIdx.x; i < N; i += blockDim.x ) {
        const Ball<MODE> bl( pr, Pt, R, Ms, b, i, N );
        for ( int t = 0; t < ta; ++t ) {
            double s, scale, ds[ 2 ], dsc[ 2 ];
            geo.project( a0 + t, bl.px, bl.py, s, scale, ds, dsc );
            const double w = bl.rc * scale;
            double *row = rows + t * M;
            if constexpr ( DEBUG & 4 )
                deposit_ball<P>( pr, [&]( SI j, double m ) { row[ j ] += m; }, s, w, 1 / w, bl.mass );
            else
                deposit_ball<P>( pr, [&]( SI j, double m ) { atomicAdd( row + j, m ); }, s, w, 1 / w, bl.mass );
        }
    }
    __syncthreads();

    // the transport of each row ( a thread each )
    if ( ! ( DEBUG & 1 ) && threadIdx.x < ta ) {
        double *row = rows + threadIdx.x * M;
        const SI a = a0 + threadIdx.x;
        L( b, a ) = transport_row( pr, [&]( SI j ) -> double & { return row[ j ]; }, [&]( SI j ) { return double( D( b, a, j ) ); } );
    }
    __syncthreads();

    // what each ball receives, over the views of the tile
    if constexpr ( DEBUG & 2 )
        return;
    for ( SI i = threadIdx.x; i < N; i += blockDim.x ) {
        const Ball<MODE> bl( pr, Pt, R, Ms, b, i, N );
        typename Prec<P>::A o[ 3 ] = { 0, 0, 0 };
        double om = 0;
        for ( int t = 0; t < ta; ++t ) {
            const double *row = rows + t * M;
            gather_view<P>( pr, geo, a0 + t, bl, [&]( SI e ) { return row[ e ]; }, o, om );
        }
        atomicAdd( &GP( b, i, 0 ), double( o[ 0 ] ) );
        atomicAdd( &GP( b, i, 1 ), double( o[ 1 ] ) );
        if constexpr ( MODE == 2 ) {
            if ( bl.r < pr.max_w )
                atomicAdd( &GR( b, i ), double( o[ 2 ] ) );
            atomicAdd( &GM( b, i ), om );
        }
    }
}

// THE SPLIT PATH ( the default on the card ): three kernels in the same call, the rows in GLOBAL memory ( `rows`, a
// scratch output [ B, A, M ], a few MB: in the L2 cache ). The deposit and the gather are then as parallel as the
// separate kernels of `Projected1d.h` ( a thread per ball and per tile of views, ~ 4096 blocks: each thread keeps its
// ball over many views ), the transport a thread per row. Measured ( 2080 Ti, 100k balls, 768 x 512, PRECISION 2 ): the
// tiles in shared memory above take 28.5 ms ( deposit 15, transport 1, gather 12.6: their ~ 50k threads reload a ball
// every few views, and wait for it ), the three kernels of the separate path 22 ms with the transport in torch.

/// the views a thread goes through: enough blocks to fill the card ( ~ 4096 ), never more tiles than views
inline SI view_tiles( SI nb_ball_blocks, SI A ) {
    const SI want = ( 4096 + nb_ball_blocks - 1 ) / nb_ball_blocks;
    return std::max<SI>( 1, std::min<SI>( A, want ) );
}

template<int K,int MODE,int P,int BLOCK,class VP,class VR,class VM,class VH>
__global__ void __launch_bounds__( BLOCK ) split_deposit_kernel( View<double,1> PP, Parallel geo, SI N, SI A, SI M, SI nbb, SI tiles, VP Pt, VR R, VM Ms, VH H ) {
    const Profile pr = read_profile( PP, M, K );
    const SI q = blockIdx.x, ob = q % nbb, tile = ( q / nbb ) % tiles, b = q / ( nbb * tiles ), i = ob * BLOCK + threadIdx.x;
    if ( i >= N )
        return;
    const Ball<MODE> bl( pr, Pt, R, Ms, b, i, N );
    for ( SI a = A * tile / tiles; a < A * ( tile + 1 ) / tiles; ++a ) {
        double s, scale, ds[ 2 ], dsc[ 2 ];
        geo.project( a, bl.px, bl.py, s, scale, ds, dsc );
        const double w = bl.rc * scale;
        deposit_ball<P>( pr, [&]( SI j, double m ) { atomicAdd( &H( b, a, j ), m ); }, s, w, 1 / w, bl.mass );
    }
}

template<class VH,class VD,class VL>
__global__ void split_transport_kernel( View<double,1> PP, SI A, SI M, int K, VH H, VD D, VL L, SI rows ) {
    const SI r = SI( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( r >= rows )
        return;
    const Profile pr = read_profile( PP, M, K );
    const SI b = r / A, a = r % A;
    L( b, a ) = transport_row( pr, [&]( SI j ) -> double & { return H( b, a, j ); }, [&]( SI j ) { return double( D( b, a, j ) ); } );
}

template<int K,int MODE,int P,int BLOCK,class VP,class VR,class VM,class VH,class VGP,class VGR,class VGM>
__global__ void __launch_bounds__( BLOCK ) split_gather_kernel( View<double,1> PP, Parallel geo, SI N, SI A, SI M, SI nbb, SI tiles, VP Pt, VR R, VM Ms, VH H,
                                                                VGP GP, VGR GR, VGM GM ) {
    const Profile pr = read_profile( PP, M, K );
    const SI q = blockIdx.x, ob = q % nbb, tile = ( q / nbb ) % tiles, b = q / ( nbb * tiles ), i = ob * BLOCK + threadIdx.x;
    if ( i >= N )
        return;
    const Ball<MODE> bl( pr, Pt, R, Ms, b, i, N );
    typename Prec<P>::A o[ 3 ] = { 0, 0, 0 };
    double om = 0;
    for ( SI a = A * tile / tiles; a < A * ( tile + 1 ) / tiles; ++a )
        gather_view<P>( pr, geo, a, bl, [&]( SI e ) { return H( b, a, e ); }, o, om );
    atomicAdd( &GP( b, i, 0 ), double( o[ 0 ] ) );
    atomicAdd( &GP( b, i, 1 ), double( o[ 1 ] ) );
    if constexpr ( MODE == 2 ) {
        if ( bl.r < pr.max_w )
            atomicAdd( &GR( b, i ), double( o[ 2 ] ) );
        atomicAdd( &GM( b, i ), om );
    }
}

/// `rows [ B, A, M ]`: a scratch, zeroed by the caller ( TILE < 0: the tiles in shared memory instead, `-TILE` views each )
template<int K,int MODE,int P = 0,int BLOCK = 256,int TILE = 0,int DEBUG = 0,class Queue> requires ( ! std::is_same_v<Queue,CpuQueue> )
void loss_balls( const Queue &queue, const auto &points, const auto &radii, const auto &masses, const auto &normals, const auto &params,
                 const auto &data, auto &&loss, auto &&grad_points, auto &&grad_radii, auto &&grad_masses, auto &&rows ) {
    const SI B = points.shape( 0 ), N = points.shape( 1 ), A = normals.shape( 0 ), M = data.shape( 2 );
    const auto Pt = view( points );
    const auto R = view( radii );
    const auto Ms = view( masses );
    const auto D = view( data );
    const auto L = view( loss );
    const auto GP = view( grad_points );
    const auto GR = view( grad_radii );
    const auto GM = view( grad_masses );
    const auto PP = view( params );
    const Parallel geo{ view( normals ) };
    if constexpr ( TILE < 0 ) {
        // the tiles: `-TILE` views a block, their rows in shared memory, never more than 48 kB of them
        const int TA = int( std::max<SI>( 1, std::min<SI>( -TILE, ( 48 * 1024 ) / ( 8 * M ) ) ) );
        launch_kernel( queue, &loss_balls_kernel<K,MODE,P,BLOCK,DEBUG,decltype( Pt ),decltype( R ),decltype( Ms ),decltype( D ),decltype( L ),decltype( GP ),decltype( GR ),decltype( GM )>,
                       int( B * ( ( A + TA - 1 ) / TA ) ), BLOCK, int( TA * M * sizeof( double ) ),
                       PP, geo, N, A, M, TA, Pt, R, Ms, D, L, GP, GR, GM );
    } else {
        const auto H = view( rows );
        const SI nbb = ( N + BLOCK - 1 ) / BLOCK, tiles = TILE > 0 ? std::min<SI>( TILE, A ) : view_tiles( nbb * B, A );
        launch_kernel( queue, &split_deposit_kernel<K,MODE,P,BLOCK,decltype( Pt ),decltype( R ),decltype( Ms ),decltype( H )>,
                       int( B * nbb * tiles ), BLOCK, 0, PP, geo, N, A, M, nbb, tiles, Pt, R, Ms, H );
        launch_kernel( queue, &split_transport_kernel<decltype( H ),decltype( D ),decltype( L )>,
                       int( ( B * A + 63 ) / 64 ), 64, 0, PP, A, M, K, H, D, L, B * A );
        launch_kernel( queue, &split_gather_kernel<K,MODE,P,BLOCK,decltype( Pt ),decltype( R ),decltype( Ms ),decltype( H ),decltype( GP ),decltype( GR ),decltype( GM )>,
                       int( B * nbb * tiles ), BLOCK, 0, PP, geo, N, A, M, nbb, tiles, Pt, R, Ms, H, GP, GR, GM );
    }
}


// fields on the card: the deposit of the split path, a transport that keeps the image and the masses of each row, and
// a kernel a thread per ball and per tile of views

template<class VH,class VD,class VL,class VI,class VS>
__global__ void fields_transport_kernel( View<double,1> PP, SI A, SI M, int K, VH H, VD D, VI I, VS RS, SI rows ) {
    const SI r = SI( blockIdx.x ) * blockDim.x + threadIdx.x;
    if ( r >= rows )
        return;
    const Profile pr = read_profile( PP, M, K );
    const SI b = r / A, a = r % A;
    double S0 = 0, Sn = 0;
    for ( SI c = 0; c < M; ++c ) {
        const double v = H( b, a, c );
        I( b, a, c ) = v;
        S0 += v;
        Sn += double( D( b, a, c ) );
    }
    RS( b, a, 0 ) = S0;
    RS( b, a, 1 ) = Sn;
    transport_row( pr, [&]( SI j ) -> double & { return H( b, a, j ); }, [&]( SI j ) { return double( D( b, a, j ) ); } );
}

template<int K,int MODE,int BLOCK,class VP,class VR,class VM,class VH,class VI,class VD,class VS,class VFB,class VFS>
__global__ void __launch_bounds__( BLOCK ) fields_kernel( View<double,1> PP, Parallel geo, SI N, SI A, SI M, SI nbb, SI tiles, VP Pt, VR R, VM Ms,
                                                          VH H, VI I, VD D, VS RS, VFB FB, VFS FS ) {
    const Profile pr = read_profile( PP, M, K );
    const SI q = blockIdx.x, ob = q % nbb, tile = ( q / nbb ) % tiles, b = q / ( nbb * tiles ), i = ob * BLOCK + threadIdx.x;
    if ( i >= N )
        return;
    const Ball<MODE> bl( pr, Pt, R, Ms, b, i, N );
    double fb[ 2 ] = { 0, 0 }, fs[ 3 ] = { 0, 0, 0 };
    for ( SI a = A * tile / tiles; a < A * ( tile + 1 ) / tiles; ++a )
        fields_view( pr, geo, a, bl, [&]( SI e ) { return H( b, a, e ); }, [&]( SI j ) { return I( b, a, j ); }, [&]( SI j ) { return double( D( b, a, j ) ); },
                     RS( b, a, 0 ), RS( b, a, 1 ), fb, fs );
    atomicAdd( &FB( b, i, 0 ), fb[ 0 ] ); atomicAdd( &FB( b, i, 1 ), fb[ 1 ] );
    atomicAdd( &FS( b, i, 0 ), fs[ 0 ] ); atomicAdd( &FS( b, i, 1 ), fs[ 1 ] ); atomicAdd( &FS( b, i, 2 ), fs[ 2 ] );
}

/// `rows`, `image [ B, A, M ]` and `sums [ B, A, 2 ]`: scratches, zeroed by the caller
template<int K,int MODE,class Queue> requires ( ! std::is_same_v<Queue,CpuQueue> )
void fields_balls( const Queue &queue, const auto &points, const auto &radii, const auto &masses, const auto &normals, const auto &params,
                   const auto &data, auto &&fb, auto &&fs, auto &&rows, auto &&image, auto &&sums ) {
    constexpr int BLOCK = 256;
    const SI B = points.shape( 0 ), N = points.shape( 1 ), A = normals.shape( 0 ), M = data.shape( 2 );
    const auto Pt = view( points );
    const auto R = view( radii );
    const auto Ms = view( masses );
    const auto D = view( data );
    const auto H = view( rows );
    const auto I = view( image );
    const auto RS = view( sums );
    const auto FB = view( fb );
    const auto FS = view( fs );
    const auto PP = view( params );
    const Parallel geo{ view( normals ) };
    const SI nbb = ( N + BLOCK - 1 ) / BLOCK, tiles = view_tiles( nbb * B, A );
    launch_kernel( queue, &split_deposit_kernel<K,MODE,0,BLOCK,decltype( Pt ),decltype( R ),decltype( Ms ),decltype( H )>,
                   int( B * nbb * tiles ), BLOCK, 0, PP, geo, N, A, M, nbb, tiles, Pt, R, Ms, H );
    launch_kernel( queue, &fields_transport_kernel<decltype( H ),decltype( D ),void,decltype( I ),decltype( RS )>,
                   int( ( B * A + 63 ) / 64 ), 64, 0, PP, A, M, K, H, D, I, RS, B * A );
    launch_kernel( queue, &fields_kernel<K,MODE,BLOCK,decltype( Pt ),decltype( R ),decltype( Ms ),decltype( H ),decltype( I ),decltype( D ),decltype( RS ),decltype( FB ),decltype( FS )>,
                   int( B * nbb * tiles ), BLOCK, 0, PP, geo, N, A, M, nbb, tiles, Pt, R, Ms, H, I, D, RS, FB, FS );
}

#endif

} // namespace projected1d_fused
} // namespace sdot
