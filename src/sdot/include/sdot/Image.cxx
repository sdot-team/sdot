#pragma once

#include <loom/support/common_macros.h> // HD

#include <loom/support/algorithms/CartesianIndices.h> // iterate the (dynamic-rank) cell grid
#include <loom/support/containers/IotaTensor.h>       // IotaTensor<TF> (knots = 0,1,2) -- default `knots`
#include <loom/support/containers/Vector.h>           // Vector<TF,d>::zeros()          -- default `origin`
#include <loom/support/containers/Matrix.h>
#include <loom/support/atomic_add.h>
#include <limits>
#include "cell/Ids.h"
#include "ConstantDensity.h"
#include "CstUdPiece.h"
#include "LinUdPiece.h"
#include "Image.h"
#include "PolynomialPixel.h"
#include <cmath>
#include <cassert>

#define UTP SDOT_TEMPLATE_DECL_FOR_Image
#define DTP Image<SDOT_TEMPLATE_ARGS_FOR_Image>

namespace sdot {

UTP HD auto DTP::with_defaults( auto &&cont ) const {
    // Substitute one absent member per step, then recurse -- exactly the shape of
    // `Cell::init_as_hypercube`'s default handling, but rebuilding the WHOLE aggregate (a new
    // template instantiation, deduced by C++20 aggregate CTAD, like the generated `kernel_form`)
    // rather than passing individual defaults down. We must rebuild, not mutate: a `NoneTensor` has
    // a fixed type and no `operator=`, so replacing its VALUE means replacing its TYPE.
    //
    // The brace-init is POSITIONAL, so its order must match `SDOT_ATTRIBUTES_OF_Image` (the field
    // declaration order across the MRO, axes skipped): target_mass, nb_dims, shape, values, origin,
    // frame, knots, current_mass, nb_cells_cum, cell_cum_mass, nb_coeffs, coeffs. `::sdot::Image` (qualified) names the
    // TEMPLATE so CTAD re-deduces; bare `Image` would mean the current instantiation and defeat the
    // substitution.
    // ( `ramp`, a C++ member out of the positional list, is copied by hand: see `Image.h` )
    auto again = [&]( auto &&img ) { img.ramp = ramp; return img.with_defaults( FORWARD( cont ) ); };
    if constexpr ( ! DECAYED_TYPE_OF( origin )::is_valid )
        return again( ::sdot::Image{ target_mass, nb_dims, shape, values, Vector<TF,ct_dim>::zeros(), frame, knots, current_mass, nb_cells_cum, cell_cum_mass, nb_coeffs, coeffs } );
    else if constexpr ( ! DECAYED_TYPE_OF( frame )::is_valid )
        return again( ::sdot::Image{ target_mass, nb_dims, shape, values, origin, Matrix<TF,ct_dim>::identity(), knots, current_mass, nb_cells_cum, cell_cum_mass, nb_coeffs, coeffs } );
    else if constexpr ( ! DECAYED_TYPE_OF( knots )::is_valid )
        return again( ::sdot::Image{ target_mass, nb_dims, shape, values, origin, frame, IotaTensor<TF>{}, current_mass, nb_cells_cum, cell_cum_mass, nb_coeffs, coeffs } );
    else
        return cont( *this );
}

UTP HD typename DTP::TF DTP::measure() const {
    return with_defaults( []( auto &&img ) {
        using ImgT = DECAYED_TYPE_OF( img );
        using TF = typename ImgT::TF;
        constexpr int d = ImgT::ct_dim;

        // |det(frame)| -- rebuild a dense d*d matrix so this works whether `frame` is a `Matrix`
        // (the identity default) or a user-given `TensorView`.
        const auto F = Matrix<TF,d>::with_func( [&]( auto r, auto c ) { return TF( img.frame( r, c ) ); } );
        const TF fdet = std::abs( F.determinant() );

        if constexpr ( ImgT::is_polynomial ) {
            // the mass of a pixel is its volume times the mean of its Bernstein coefficients ( `int B_{ i, P } = 1 / ( P + 1 )` )
            constexpr PI nc = DECAYED_TYPE_OF( img.nb_coeffs )::value;
            auto shape = img.coeffs.shape();
            CartesianIndices<DECAYED_TYPE_OF( shape )> cells{ shape };
            double sum = 0;                   // ( in double: in float, the sum of many pixels drifts )
            for ( PI flat = 0; flat < cells.size(); ++flat )
                sum += cells[ flat ].values.apply_values( [&]( auto ...i ) {
                    PI axis = 0;
                    TF spacing = 1;
                    ( ( axis < PI( d ) ? ( spacing *= TF( img.knots( axis, i + 1 ) ) - TF( img.knots( axis, i ) ), 0 ) : 0, ++axis ), ... );
                    return TF( img.coeffs( i... ) ) * spacing;
                } );
            return TF( sum * double( fdet ) / double( nc ) );
        } else {
            // sum over the cells (extent per axis = number of cells along it = `values.shape()`).
            // Each cell contributes value * |det(frame)| * Prod_axis ( knots(axis, i+1) - knots(axis, i) ).
            // `values` is an ordinary dense rank-d view (the `img_pos` AxisList unrolls into d named
            // dimensions), so it and `knots` are both indexed positionally.
            auto shape = img.values.shape();
            CartesianIndices<DECAYED_TYPE_OF( shape )> cells{ shape };
            double sum = 0;                   // ( in double: in float, the sum of many pixels drifts )
            for ( PI flat = 0; flat < cells.size(); ++flat ) {
                // `cells[ flat ]` now returns a `Coords` ( a NAMED multi-index ) and no longer the
                // bare tuple: unpacking is requested from its `values`.
                const TF cell = cells[ flat ].values.apply_values( [&]( auto ...i ) {
                    // running axis counter over the index pack (axes 0..d-1, in order)
                    PI axis = 0;
                    TF spacing = 1;
                    ( ( spacing *= TF( img.knots( axis, i + 1 ) ) - TF( img.knots( axis, i ) ), ++axis ), ... );
                    return TF( img.values( i... ) ) * spacing;
                } );
                sum += double( cell * fdet );
            }
            return TF( sum );
        }
    } );
}

namespace detail {
    /// a sub-box of a pixel under the RAMP ( `Image.h`, `ramp` ): degree 1 along each axis. A stage of the solver only: no
    /// derivative goes through it ( the adjoints read the image at `ramp = 0` )
    template<class TF,int D,class Idx>
    struct RampPixel : PolynomialPixel<TF,D,1,Idx> {
        static constexpr bool is_ramp = true;              ///< ( what an integrand that takes constant densities only skips )
        HD void add_value_grad_at( auto &&, const auto &, TF ) const { assert( false && "no derivative through the ramp of an image" ); }
        HD void integrate_over_simplex_bwd( auto &&... ) const { assert( false && "no derivative through the ramp of an image" ); }
    };

    /// A RUNTIME multi-index (`Vector<SI,d>`) turned into a `Tuple`, which `TensorView::offset` unpacks
    /// by itself -- enough to write `values( k )` with a `k` whose components are computed.
    /// Same `Ct` recursion as `unravel_index` (see `CartesianIndices.h`).
    HD auto image_index_tuple( const auto &k, auto &&acc, auto done ) {
        constexpr int i = DECAYED_TYPE_OF( done )::value;
        constexpr int n = DECAYED_TYPE_OF( k.size() )::value;
        if constexpr ( i == n )
            return acc;
        else
            return image_index_tuple( k, acc.with_appended_value( SI( k[ i ] ) ), Ct<int,i+1>() );
    }
}

UTP HD SI DTP::knot_index( SI axis, TF t, SI nb_cells ) const {
    SI b = 0, e = nb_cells;                 // we search in [ b, e ), invariant: the answer is in it
    while ( e - b > 1 ) {
        const SI m = ( b + e ) / 2;
        if ( TF( knots( axis, m ) ) <= t ) b = m;
        else                               e = m;
    }
    return b;
}

UTP HD void DTP::for_each_piece( const auto &cell, auto &&ws, auto &&func ) const {
    with_defaults( [&]( auto &&img ) { img._for_each_piece( cell, ws, func, Matrix<TF,ct_dim>::identity(), Vector<TF,ct_dim>::zeros() ); } );
}

UTP HD void DTP::for_each_piece_moved( const auto &cell, auto &&ws, auto &&func, const auto &L, const auto &s ) const {
    with_defaults( [&]( auto &&img ) { img._for_each_piece( cell, ws, func, L, s ); } );
}

UTP HD void DTP::_for_each_piece( const auto &cell, auto &&ws, auto &&func, const auto &L, const auto &s ) const {
    constexpr int d = ct_dim;
    static_assert( d >= 1 );

    if ( cell.nb_vertices() == 0 )
        return;

    // ---- the grid planes, in PHYSICAL coordinates.
    // The convention is that of `measure`: `x = origin + F^T t`, so `t = ( F^T )^-1 ( x - origin )`
    // and row `a` of `( F^T )^-1` is COLUMN `a` of `F^-1`, i.e. `solve_ge( F, e_a )`.
    // So nothing is inverted: d solves, one per axis, and tile `k` is the intersection of the
    // bands `knots( a, k_a ) <= n_a . ( x - origin ) <= knots( a, k_a + 1 )`.
    // the image moved by `x -> L x + s`: its pixel edges ( the rows of `F` ) turned by `L`, its origin moved
    const auto F = Matrix<TF,d>::with_func( [&]( auto r, auto c ) {
        TF v = 0;
        for ( PI q = 0; q < PI( d ); ++q )
            v += TF( frame( r, q ) ) * TF( L( c, q ) );
        return v;
    } );
    const auto og = Vector<TF,d>::with_func( [&]( PI c ) {
        TF v = TF( s[ c ] );
        for ( PI q = 0; q < PI( d ); ++q )
            v += TF( L( c, q ) ) * TF( origin( q ) );
        return v;
    } );

    Vector<Vector<TF,d>,d> nrm;
    Vector<TF,d> shift;                     // `n_a . origin`, which shifts the plane offset
    for ( PI a = 0; a < d; ++a ) {
        nrm[ a ] = Matrix<TF,d>::solve_ge( F, Vector<TF,d>::with_value_at( a, TF( 1 ) ) );
        shift[ a ] = dot( nrm[ a ], og );
    }

    // ---- WHICH tiles to try: those that the bounding box of the cell meets.
    // An UNBOUNDED cell has no box -- its vertices are those of a stop-gap simplex
    // (`Cell::init_as_unbounded`) and bound nothing. We then sweep the whole image, which stays
    // CORRECT (the image has compact support, so the integral is finite even on an
    // infinite cell) and costs only time -- a domain (`box = ...`) removes the case.
    Vector<SI,d> k0, k1;
    Vector<TF,d> t_lo, t_hi;                // the extent of the cell along each axis, in grid coordinates
    const bool bounded = cell.bounded();
    for ( PI a = 0; a < d; ++a ) {
        const SI nb = nb_pixels( int( a ) );
        if ( nb <= 0 )
            return;
        if ( ! bounded ) {
            k0[ a ] = 0;
            k1[ a ] = nb;
            t_lo[ a ] = - std::numeric_limits<TF>::max();
            t_hi[ a ] =   std::numeric_limits<TF>::max();
            continue;
        }

        const SI nv = cell.nb_vertices();
        TF t_min = 0, t_max = 0;
        for ( SI v = 0; v < nv; ++v ) {
            TF t = - shift[ a ];
            for ( PI c = 0; c < d; ++c )
                t += nrm[ a ][ c ] * TF( cell.coord( int( v ), int( c ) ) );
            if ( v == 0 || t < t_min ) t_min = t;
            if ( v == 0 || t > t_max ) t_max = t;
        }

        // outside the image on that side: nothing at all to integrate.
        if ( t_max < TF( knots( a, 0 ) ) || t_min > TF( knots( a, nb ) ) )
            return;

        t_lo[ a ] = t_min;
        t_hi[ a ] = t_max;
        // ( a tile that the cell enters by a rounding only is not one of its tiles: see the cuts below )
        using TKc = std::decay_t<decltype( cell.coord( 0, 0 ) )>;
        const TF tk = TF( 64 ) * TF( std::numeric_limits<TKc>::epsilon() );
        k0[ a ] = knot_index( a, t_min + tk * ( 1 + ( t_min < 0 ? -t_min : t_min ) ), nb );
        k1[ a ] = knot_index( a, t_max - tk * ( 1 + ( t_max < 0 ? -t_max : t_max ) ), nb ) + 1;
    }

    // ---- one piece per tile. Odometer counter over `[ k0, k1 )`, and 2d cuts per tile.
    //
    // A RECURSIVE descent (first cut the slab of axis 0, then subdivide inside it) would do
    // ~2 cuts per tile instead of 2d, and would prune empty slabs in one go; it requires
    // however a spare cell PER LEVEL, whereas this one requires two in all. To be
    // done the day the profile calls for it -- the Python-side contract (`extra_cuts_per_piece`) does not
    // change.
    Vector<SI,d> k = k0;
    for ( PI a = 0; a < d; ++a )
        if ( k0[ a ] >= k1[ a ] )
            return;

    using TK = std::decay_t<decltype( cell.coord( 0, 0 ) )>;
    const TF tol = TF( 64 ) * TF( std::numeric_limits<TK>::epsilon() );

    // ---- THE RAMP ( `Image.h` ): the tile `k` cut into sub-boxes, `[ lo, lo + e ]`, the centre, `[ hi - e, hi ]` along each axis
    // where the density varies there, each one a piece of its own ( at most 2d cuts, as a tile: `extra_cuts_per_piece` holds ).
    // `e` is the ramp in grid units at that knot, clipped at half of each pixel it touches -- the same on both sides
    if constexpr ( ! is_polynomial ) {
        if ( ramp > 0 ) {
            TF unit[ d ];                                     // the length of a grid unit along each axis ( a row of `F` )
            for ( PI a = 0; a < d; ++a ) {
                TF l2 = 0;
                for ( PI c = 0; c < d; ++c ) l2 += F( a, c ) * F( a, c );
                unit[ a ] = std::sqrt( l2 );
            }
            auto value_at = [&]( const Vector<SI,d> &q ) { return TF( values( detail::image_index_tuple( q, tuple(), Ct<int,0>() ) ) ); };
            auto differ = [&]( TF u, TF v ) { const TF m = ( u < 0 ? -u : u ) > ( v < 0 ? -v : v ) ? ( u < 0 ? -u : u ) : ( v < 0 ? -v : v ); return ( u > v ? u - v : v - u ) > TF( 1e-12 ) * m; };
            // the density varies across side `sg` of axis `a` of tile `k`: some pair across that face, the other axes within one pixel
            auto varies = [&]( const Vector<SI,d> &kk, PI a, int sg ) {
                Vector<SI,d> o;
                for ( PI b = 0; b < d; ++b ) o[ b ] = b == a ? 0 : -1;
                while ( true ) {
                    Vector<SI,d> p = kk, q;
                    bool in = true;
                    for ( PI b = 0; b < d; ++b ) { p[ b ] += o[ b ]; in &= p[ b ] >= 0 && p[ b ] < nb_pixels( int( b ) ); }
                    if ( in ) {
                        q = p;
                        q[ a ] += sg;
                        if ( differ( value_at( p ), value_at( q ) ) ) return true;
                    }
                    PI b = 0;
                    while ( b < d && ( b == a || ++o[ b ] > 1 ) ) { if ( b != a ) o[ b ] = -1; ++b; }
                    if ( b == d ) return false;
                }
            };
            // the half-width of the ramp at knot `j` of axis `a` ( between pixels `j - 1` and `j` ), in grid units
            auto half = [&]( PI a, SI j ) {
                TF e = ramp / unit[ a ];
                const TF wl = TF( knots( a, j ) ) - TF( knots( a, j - 1 ) ), wr = TF( knots( a, j + 1 ) ) - TF( knots( a, j ) );
                e = e < wl / 2 ? e : wl / 2;
                return e < wr / 2 ? e : wr / 2;
            };
            // along each axis, up to 3 intervals; at each end, the pixels ( offsets -1, 0, 1 ) that mix and their weights
            struct Iv { TF lo, hi; int nb[ 2 ]; SI off[ 2 ][ 2 ]; TF w[ 2 ][ 2 ]; };
            while ( true ) {
                Iv iv[ d ][ 3 ];
                int ni[ d ];
                for ( PI a = 0; a < d; ++a ) {
                    const SI nb = nb_pixels( int( a ) );
                    const TF lo = TF( knots( a, k[ a ] ) ), hi = TF( knots( a, k[ a ] + 1 ) );
                    const TF el = k[ a ] > 0 && varies( k, a, -1 ) ? half( a, k[ a ] ) : TF( 0 );
                    const TF er = k[ a ] + 1 < nb && varies( k, a, +1 ) ? half( a, k[ a ] + 1 ) : TF( 0 );
                    int n = 0;
                    if ( el > 0 ) iv[ a ][ n++ ] = Iv{ lo, lo + el, { 2, 1 }, { { -1, 0 }, { 0, 0 } }, { { TF( 0.5 ), TF( 0.5 ) }, { TF( 1 ), TF( 0 ) } } };
                    if ( hi - er > lo + el ) iv[ a ][ n++ ] = Iv{ lo + el, hi - er, { 1, 1 }, { { 0, 0 }, { 0, 0 } }, { { TF( 1 ), TF( 0 ) }, { TF( 1 ), TF( 0 ) } } };
                    if ( er > 0 ) iv[ a ][ n++ ] = Iv{ hi - er, hi, { 1, 2 }, { { 0, 0 }, { 0, 1 } }, { { TF( 1 ), TF( 0 ) }, { TF( 0.5 ), TF( 0.5 ) } } };
                    ni[ a ] = n;
                }
                Vector<SI,d> m;
                for ( PI a = 0; a < d; ++a ) m[ a ] = 0;
                while ( true ) {
                    // the sub-box `m`: cut out of the cell as a tile is
                    bool alive = true, fitted = true, started = false;
                    for ( PI a = 0; a < d && alive; ++a ) {
                        const Iv &I = iv[ a ][ m[ a ] ];
                        const TF tol_hi = tol * ( 1 + ( I.hi < 0 ? -I.hi : I.hi ) ), tol_lo = tol * ( 1 + ( I.lo < 0 ? -I.lo : I.lo ) );
                        if ( t_hi[ a ] > I.hi + tol_hi ) {
                            fitted = started ? ws.cut( nrm[ a ], I.hi + shift[ a ] ) : ws.start( cell, nrm[ a ], I.hi + shift[ a ] );
                            started = true;
                            if ( ! fitted ) break;
                            if ( ws.nb_vertices() == 0 ) { alive = false; break; }
                        }
                        if ( t_lo[ a ] < I.lo - tol_lo ) {
                            fitted = started ? ws.cut( - nrm[ a ], - I.lo - shift[ a ] ) : ws.start( cell, - nrm[ a ], - I.lo - shift[ a ] );
                            started = true;
                            if ( ! fitted ) break;
                            if ( ws.nb_vertices() == 0 ) { alive = false; break; }
                        }
                    }
                    if ( ! fitted )
                        return;
                    if ( alive ) {
                        // its Bernstein coefficients ( degree 1 along each axis ): at each corner `al`, the product of the weights
                        const auto idx = detail::image_index_tuple( k, tuple(), Ct<int,0>() );
                        detail::RampPixel<TF,d,DECAYED_TYPE_OF( idx )> density{ PolynomialPixel<TF,d,1,DECAYED_TYPE_OF( idx )>{ .idx = idx } };
                        density.for_each_alpha( [&]( int flat, const int *al ) {
                            TF sum = 0;
                            int ch[ d ];
                            for ( PI a = 0; a < d; ++a ) ch[ a ] = 0;
                            while ( true ) {
                                Vector<SI,d> q = k;
                                TF w = 1;
                                for ( PI a = 0; a < d; ++a ) {
                                    const Iv &I = iv[ a ][ m[ a ] ];
                                    q[ a ] += I.off[ al[ a ] ][ ch[ a ] ];
                                    w *= I.w[ al[ a ] ][ ch[ a ] ];
                                }
                                sum += w * value_at( q );
                                PI a = 0;
                                while ( a < d && ++ch[ a ] >= iv[ a ][ m[ a ] ].nb[ al[ a ] ] ) { ch[ a ] = 0; ++a; }
                                if ( a == PI( d ) ) break;
                            }
                            density.c[ flat ] = sum;
                        } );
                        bool flat_density = true;
                        for ( int j = 1; j < density.nb_coeffs; ++j ) flat_density &= ! differ( density.c[ j ], density.c[ 0 ] );
                        if ( flat_density ) {                   // ( the centre, and every sub-box without a jump: the fast path )
                            auto cst = ConstantDensity{ density.c[ 0 ], [&]( auto &&, TF ) {
                                assert( false && "no derivative through the ramp of an image" );
                            } };
                            if ( started ) ws.with_current( [&]( const auto &piece ) { func( piece, cst ); } );
                            else           func( cell, cst );
                        } else {
                            for ( PI a = 0; a < d; ++a ) {
                                const Iv &I = iv[ a ][ m[ a ] ];
                                density.nrm[ a ] = nrm[ a ];
                                density.off[ a ] = I.lo + shift[ a ];
                                density.inv_h[ a ] = TF( 1 ) / ( I.hi - I.lo );
                            }
                            if ( started ) ws.with_current( [&]( const auto &piece ) { func( piece, density ); } );
                            else           func( cell, density );
                        }
                    }
                    PI a = 0;
                    while ( a < d && ++m[ a ] >= ni[ a ] ) { m[ a ] = 0; ++a; }
                    if ( a == PI( d ) ) break;
                }
                PI a = 0;
                while ( a < d && ++k[ a ] >= k1[ a ] ) { k[ a ] = k0[ a ]; ++a; }
                if ( a == PI( d ) )
                    return;
            }
        }
    }

    while ( true ) {
        // A plane that every vertex already satisfies cuts nothing: it is not made. In the common case
        // of a cell that lies inside its tile ( one tile for the whole domain, and the interior cells ) NO plane is
        // made, nothing is copied, and the cell itself is the piece.
        bool alive = true, fitted = true, started = false;
        for ( PI a = 0; a < d && alive; ++a ) {
            const TF lo = TF( knots( a, k[ a ] ) ) + shift[ a ];
            const TF hi = TF( knots( a, k[ a ] + 1 ) ) + shift[ a ];
            const TF tile_lo = TF( knots( a, k[ a ] ) ), tile_hi = TF( knots( a, k[ a ] + 1 ) );

            // those planes carry `PIECE`: they face no seed, and that is exactly what
            // the adjoint reads to know that their share goes nowhere (see
            // `PowerDiagram::scatter_cell_grad`).
            //
            // Up to the rounding of the kernel float: a cell that starts from the image ( its domain ) has vertices ON the sides of
            // the grid, and a frame that is not diagonal puts them a rounding outside -- cutting there again leaves a twin of
            // each, flat simplices whose orientation the adjoint of a quadrature cannot read ( `d|det|`, the moments )
            const TF tol_hi = tol * ( 1 + ( tile_hi < 0 ? -tile_hi : tile_hi ) ), tol_lo = tol * ( 1 + ( tile_lo < 0 ? -tile_lo : tile_lo ) );
            if ( t_hi[ a ] > tile_hi + tol_hi ) {
                fitted = started ? ws.cut( nrm[ a ], hi ) : ws.start( cell, nrm[ a ], hi );
                started = true;
                if ( ! fitted ) break;
                if ( ws.nb_vertices() == 0 ) { alive = false; break; }
            }
            if ( t_lo[ a ] < tile_lo - tol_lo ) {
                fitted = started ? ws.cut( - nrm[ a ], - lo ) : ws.start( cell, - nrm[ a ], - lo );
                started = true;
                if ( ! fitted ) break;
                if ( ws.nb_vertices() == 0 ) { alive = false; break; }
            }
        }

        // a cut that did not fit: the missing capacity is recorded, the host will relaunch with
        // double and this result will be thrown away. We stop here rather than carrying on with a
        // geometry that no longer exists.
        if ( ! fitted )
            return;

        if ( alive ) {
            const auto idx = detail::image_index_tuple( k, tuple(), Ct<int,0>() );
            if constexpr ( is_polynomial ) {
                // a polynomial on the pixel, in its own coordinates `( n_a . x - lo_a ) / h_a` ( `PolynomialPixel.h` )
                PolynomialPixel<TF,d,degree,DECAYED_TYPE_OF( idx )> density{ .idx = idx };
                for ( int j = 0; j < density.nb_coeffs; ++j )
                    density.c[ j ] = TF( coeffs( idx.with_appended_value( SI( j ) ) ) );
                for ( PI a = 0; a < d; ++a ) {
                    density.nrm[ a ] = nrm[ a ];
                    density.off[ a ] = TF( knots( a, k[ a ] ) ) + shift[ a ];
                    density.inv_h[ a ] = TF( 1 ) / ( TF( knots( a, k[ a ] + 1 ) ) - TF( knots( a, k[ a ] ) ) );
                }
                if ( started ) ws.with_current( [&]( const auto &piece ) { func( piece, density ); } );
                else           func( cell, density );
            } else {
                const TF value = TF( values( idx ) );
                auto density = ConstantDensity{ value, [&]( auto &&grad_dist, TF g ) {
                    // the gradient sink of the piece: `d mass / d values( k )` is the volume of the
                    // piece, and `k` is what this closure knows and the caller does not.
                    // Atomic: several work-items integrate cells that touch the same tile.
                    if constexpr ( DECAYED_TYPE_OF( grad_dist.values )::is_valid )
                        atomic_add( grad_dist.values( idx ).ref(), g );
                } };
                if ( started ) ws.with_current( [&]( const auto &piece ) { func( piece, density ); } );
                else           func( cell, density );
            }
        }

        PI a = 0;
        while ( a < d && ++k[ a ] >= k1[ a ] ) { k[ a ] = k0[ a ]; ++a; }
        if ( a == PI( d ) )
            break;
    }
}

UTP HD void DTP::add_affine_grad( auto &&grad_dist, const auto &acc ) const {
    constexpr int d = ct_dim;
    if constexpr ( DECAYED_TYPE_OF( grad_dist.frame )::is_valid ) {
        // `d J / d F = F^-T X^T`: `( F^-T X^T )_ij = sum_k ( F^-1 )_ki X_jk`, the rows of `F^-1` by `d` solves with `F^T`
        const auto Ft = Matrix<TF,d>::with_func( [&]( auto r, auto c ) { return TF( frame( c, r ) ); } );
        for ( int k = 0; k < d; ++k ) {
            const auto col = Matrix<TF,d>::solve_ge( Ft, Vector<TF,d>::with_value_at( k, TF( 1 ) ) );   // ( F^-1 )_{k, .}
            for ( int i = 0; i < d; ++i )
                for ( int j = 0; j < d; ++j )
                    atomic_add( grad_dist.frame( i, j ).ref(), TF( col[ i ] * acc.X[ j ][ k ] ) );
        }
    }
    if constexpr ( DECAYED_TYPE_OF( grad_dist.origin )::is_valid )
        for ( int a = 0; a < d; ++a )
            atomic_add( grad_dist.origin( a ).ref(), TF( acc.y[ a ] ) );
}

UTP HD void DTP::measure_frame_bwd( auto &&grad_frame, auto &&grad_mass ) const {
    if constexpr ( DECAYED_TYPE_OF( grad_frame )::is_valid && DECAYED_TYPE_OF( frame )::is_valid ) {
        constexpr int d = ct_dim;
        const TF gm = TF( grad_mass ) * measure();
        const auto Ft = Matrix<TF,d>::with_func( [&]( auto r, auto c ) { return TF( frame( c, r ) ); } );
        for ( int j = 0; j < d; ++j ) {
            const auto col = Matrix<TF,d>::solve_ge( Ft, Vector<TF,d>::with_value_at( j, TF( 1 ) ) );   // column `j` of `F^-T`
            for ( int i = 0; i < d; ++i )
                grad_frame( i, j ) = TF( grad_frame( i, j ) ) + gm * col[ i ];
        }
    }
}

UTP HD void DTP::measure_bwd( auto &&grad_values, auto &&grad_mass ) const {
    // `mass` is linear in `values` (see `measure`): mass = Sum_c values(c) * |det(frame)| * spacing(c),
    // so grad_values(c) = grad_mass * |det(frame)| * spacing(c). Guarded at compile time: an
    // unperturbed `values` arrives as a `NoneTensor` (no `operator=`), so the block must vanish.
    if constexpr ( DECAYED_TYPE_OF( grad_values )::is_valid ) {
        with_defaults( [&]( auto &&img ) {
            using ImgT = DECAYED_TYPE_OF( img );
            using TF = typename ImgT::TF;
            constexpr int d = ImgT::ct_dim;

            const auto F = Matrix<TF,d>::with_func( [&]( auto r, auto c ) { return TF( img.frame( r, c ) ); } );
            const TF fdet = std::abs( F.determinant() );
            const TF gm = grad_mass; // scalar cotangent of `current_mass`

            auto shape = img.values.shape();
            CartesianIndices<DECAYED_TYPE_OF( shape )> cells{ shape };
            for ( PI flat = 0; flat < cells.size(); ++flat ) {
                // same reason as in the forward: `cells[ flat ]` is a `Coords`, and
                // unpacking is requested from its `values`.
                cells[ flat ].values.apply_values( [&]( auto ...i ) {
                    PI axis = 0;
                    TF spacing = 1;
                    ( ( spacing *= TF( img.knots( axis, i + 1 ) ) - TF( img.knots( axis, i ) ), ++axis ), ... );
                    grad_values( i... ) = gm * fdet * spacing;
                } );
            }
        } );
    }
}

UTP HD void DTP::measure_coeffs_bwd( auto &&grad_coeffs, auto &&grad_mass ) const {
    // `mass = |det F| sum_pixels spacing mean( coeffs )`: linear, `d mass / d coeffs( k, j ) = |det F| spacing( k ) / nb_coeffs`
    if constexpr ( DECAYED_TYPE_OF( grad_coeffs )::is_valid && is_polynomial ) {
        with_defaults( [&]( auto &&img ) {
            using ImgT = DECAYED_TYPE_OF( img );
            using TF = typename ImgT::TF;
            constexpr int d = ImgT::ct_dim;
            constexpr PI nc = DECAYED_TYPE_OF( img.nb_coeffs )::value;
            const auto F = Matrix<TF,d>::with_func( [&]( auto r, auto c ) { return TF( img.frame( r, c ) ); } );
            const TF gm = TF( grad_mass ) * std::abs( F.determinant() ) / TF( nc );
            auto shape = img.coeffs.shape();
            CartesianIndices<DECAYED_TYPE_OF( shape )> cells{ shape };
            for ( PI flat = 0; flat < cells.size(); ++flat )
                cells[ flat ].values.apply_values( [&]( auto ...i ) {
                    PI axis = 0;
                    TF spacing = 1;
                    ( ( axis < PI( d ) ? ( spacing *= TF( img.knots( axis, i + 1 ) ) - TF( img.knots( axis, i ) ), 0 ) : 0, ++axis ), ... );
                    grad_coeffs( i... ) = gm * spacing;
                } );
        } );
    }
}

// The unidimensional-piece walk runs in PHYSICAL detector coordinates: cell edge i sits at
// `x(i) = origin(0) + frame(0,0) * knots(i)` (1D), so `origin`/`frame` are honored exactly as in
// `measure` (whose cell volume is |det(frame)| * knot spacing). A cell's mass is therefore
// `frame(0,0) * ( knots(i+1) - knots(i) ) * value` -- integral of the density over the physical
// width -- while `y` stays the DENSITY (so `mass / y` is a physical length). The default
// origin=0 / frame=identity collapses `x(i)` back to `knots(i)`, reproducing the plain-knot walk.
// Callers reach these through `with_defaults`, so `origin`/`frame`/`knots` are always populated. The knots are read
// as `knots( 0, i )`, the axis first: those of a polynomial image are given per axis.
// Assumes `frame(0,0) > 0` (an increasing detector axis), like the sorted-by-position diracs.
UTP HD typename DTP::TF DTP::udp_left( SI c ) const {
    if constexpr ( is_polynomial ) return TF( coeffs( c, 0 ) );
    else                           return TF( values[ c ] );
}

UTP HD typename DTP::TF DTP::udp_slope( SI c ) const {
    if constexpr ( is_polynomial ) return ( TF( coeffs( c, 1 ) ) - TF( coeffs( c, 0 ) ) ) / ( frame( 0, 0 ) * ( knots( 0, c + 1 ) - knots( 0, c ) ) );
    else                           return TF( 0 );
}

// ( in double: in float, the mass of a pixel of a tail, h times a density of 1e-38, is a denormal with few digits left )
UTP HD double DTP::udp_cell_mass( SI c ) const {
    const double h = double( frame( 0, 0 ) ) * ( double( knots( 0, c + 1 ) ) - double( knots( 0, c ) ) );
    if constexpr ( is_polynomial ) return h * ( double( coeffs( c, 0 ) ) + double( coeffs( c, 1 ) ) ) / 2;
    else                           return h * double( values[ c ] );
}

UTP HD SI DTP::udp_nb_cells() const {
    if constexpr ( is_polynomial ) return SI( coeffs.shape( 0 ) );
    else                           return SI( values.size() );
}

// the length from a point of density `y` ( slope `s` in its pixel ) that holds the mass `m`: `m / y` on a constant
// pixel, the root of `y u + s u^2 / 2 = m` on an affine one, `2 m / ( y + sqrt( y^2 + 2 s m ) )` ( no cancellation )
// ( in double, for the same reason: a small mass over a small density )
UTP HD typename DTP::TF DTP::udp_length( TF y, TF s, double m ) const {
    if constexpr ( is_polynomial ) {
        const double d = double( y ) * double( y ) + 2 * double( s ) * m;
        using std::sqrt;
        const double den = double( y ) + ( d > 0 ? sqrt( d ) : 0.0 );
        return den > 0 ? TF( 2 * m / den ) : TF( 0 );
    } else
        return TF( m / double( y ) );
}

UTP HD auto DTP::udp_piece( TF x0, TF x1, const auto &udp ) const {
    if constexpr ( is_polynomial ) return LinUdPiece<TF>{ .x0 = x0, .x1 = x1, .y0 = udp.y, .s = udp.s };
    else                           return CstUdPiece<TF>{ .x0 = x0, .x1 = x1, .y = udp.y };
}

UTP HD auto DTP::udp_start() const {
    const TF sc = frame( 0, 0 );
    const TF of = origin( 0 );
    return Udp{
        .index = 0,
        .pos = of + sc * knots( 0, 0 ),
        .mass = udp_cell_mass( 0 ),
        .y = udp_left( 0 ),
        .s = udp_slope( 0 )
    };
}

UTP HD auto DTP::udp_cont( auto &&udp, auto mass_to_take, auto &&cb_parts ) const {
    const SI nb_cells = udp_nb_cells();
    const TF sc = frame( 0, 0 );
    const TF of = origin( 0 );

    // Consume whole cells while the current one cannot satisfy the whole request.
    // The `udp.index + 1 < nb_cells` guard keeps every access in bounds: the caller
    // only ever asks for the total remaining mass, so on the last cell we fall through
    // to the partial take below (and a floating-point overshoot cannot walk past the end).
    while( mass_to_take >= udp.mass && udp.index + 1 < nb_cells ) {
        mass_to_take -= udp.mass;

        const SI new_index = udp.index + 1;
        const TF x1 = of + sc * knots( 0, new_index );
        // Emit even when y == 0: it contributes nothing to the cost / barycenter (`w2_dist` and
        // `first_moment` carry the y factor) but its `second_moment_about` feeds the direct term of
        // d cost / d y_c, which is non-zero on a zero-density cell.
        cb_parts( udp_piece( udp.pos, x1, udp ) );

        udp.index = new_index;
        udp.pos = x1;
        udp.y = udp_left( new_index );
        udp.s = udp_slope( new_index );

        udp.mass = udp_cell_mass( new_index );
    }

    // Partial take inside the current cell. Interior zero-density cells are already SKIPPED by the
    // loop above (their mass is 0, so `mass_to_take >= udp.mass` always advances past them); the only
    // way the density can be 0 here ( and not rise in the cell ) is a TRAILING run of empty cells at the
    // target's right end. There is no mass to place there -- the residual `mass_to_take` is the
    // floating-point remainder of an exhausted target -- so we skip the take rather than divide by zero.
    if ( udp.y > 0 || udp.s > 0 ) {
        // The new position is computed FROM THE START OF THE PIXEL, from the mass taken in it so far -- not by adding the
        // length of this take to `udp.pos`: in single precision, at coordinates of the order of 1, a sum of lengths drifts
        // by a fraction of a pixel and the position no longer matches the mass. Bounded by the end of the pixel ( a rounding
        // remainder taken from a nearly empty pixel, a density of 1e-70 in the tail of a gaussian, would otherwise give a
        // length far outside it ).
        const SI c = udp.index;
        const TF x_cell = of + sc * knots( 0, c ), h = sc * ( knots( 0, c + 1 ) - knots( 0, c ) );
        const double rem = udp.mass - double( mass_to_take );
        const TF L0 = udp_length( udp_left( c ), udp.s, udp_cell_mass( c ) - rem ), L = L0 < h ? L0 : h;
        const TF x1 = x_cell + L > udp.pos ? x_cell + L : udp.pos;
        cb_parts( udp_piece( udp.pos, x1, udp ) );
        udp.pos = x1;
        udp.y = udp_left( c ) + udp.s * L;
        udp.mass = rem; // remaining mass of the current cell, for the next dirac
    }
}

UTP HD auto DTP::udp_at( auto &&cell_cum_mass, auto target_mass ) const {
    const SI nb_cells = udp_nb_cells();
    const TF sc = frame( 0, 0 );
    const TF of = origin( 0 );

    // Binary search for the SMALLEST cell index `c` with `cell_cum_mass(c+1) >= target_mass` -- NOT
    // the largest `c` with `cell_cum_mass(c) <= target_mass`. The two rules agree everywhere EXCEPT
    // across a run of zero-mass cells sharing the same cumulative value as `target_mass`, and only the
    // smallest-c rule reproduces what the sequential walk actually does there: `udp_start()` never
    // pre-advances past a leading zero-density cell (advancing only ever happens inside `udp_cont`'s
    // own while-loop, driven by an actual dirac's `mass_to_take`), and a boundary landing exactly on a
    // cell edge resolves to the LAST cell whose mass ended there (`udp.y` possibly nonzero, `udp.mass`
    // exactly 0), matching `udp_cont`'s own state right after a partial-take finishes exactly on an
    // edge -- the following zero-mass run is only entered by the NEXT `udp_cont` call, exactly like the
    // sequential algorithm's own next call would. For any `c > 0` chosen by this rule, cell `c` PROVABLY
    // carries mass (`cell_cum_mass(c) < target_mass` forces `cell_cum_mass(c) < cell_cum_mass(c+1)`) --
    // only `c == 0` can be empty, and only when `target_mass == 0` exactly, guarded below like
    // `udp_cont`'s own partial-take guard.
    SI lo = 0, hi = nb_cells - 1; // invariant: answer in [lo,hi]
    while ( lo < hi ) {
        const SI mid = lo + ( hi - lo ) / 2;
        if ( double( cell_cum_mass( mid + 1 ) ) >= double( target_mass ) ) hi = mid;
        else                                                       lo = mid + 1;
    }
    const SI c = lo; // a float overshoot past the array's own total (independent chunked-summation
                      // rounding between `cell_cum_mass` and the caller's own target mass) cannot walk
                      // past the end -- same bound-safety reasoning as `udp_cont`'s `index+1 < nb_cells`.

    const double base = double( cell_cum_mass( c ) );
    const TF y    = udp_left( c );
    const TF s    = udp_slope( c );
    const TF x0   = of + sc * knots( 0, c );
    const TF L0   = ( y > 0 || s > 0 ) ? udp_length( y, s, double( target_mass ) - base ) : TF( 0 );
    const TF h    = sc * ( knots( 0, c + 1 ) - knots( 0, c ) );
    const TF L    = L0 < h ? L0 : h;                                 // ( bounded by the pixel, as in `udp_cont` )
    return Udp{
        .index = c,
        .pos   = x0 + L,
        .mass  = double( cell_cum_mass( c + 1 ) ) - double( target_mass ), // array-consistent (not recomputed
                                                                    // from knots*y), stays coherent with
                                                                    // whatever rounding built the array
        .y     = y + s * L,
        .s     = s
    };
}

// Sequential build of `cell_cum_mass` -- exactly the cell-mass formula `udp_start`/`udp_cont` use
// inline above, but walked once end-to-end and cached (see `Image.py`'s `cell_cum_mass`
// `ComputedAttribute`) instead of being rebuilt by every `SdotPlan1d` forward/backward call. One
// thread does a whole angle -- `nb_cells` is small enough that no cooperative scan is worth it.
UTP HD void DTP::fill_cell_cum_mass( auto &&cell_cum_mass ) const {
    with_defaults( [&]( auto &&img ) {
        const SI nb_cells = img.udp_nb_cells();
        auto cell_mass = [&]( SI c ) { return img.udp_cell_mass( c ); };   // ( in double )

        double running = 0;                  // ( in double, and stored as such: in float, the sum of many pixels drifts )
        for ( SI c = 0; c < nb_cells; ++c ) {
            cell_cum_mass( c ) = running;
            running += cell_mass( c );
        }
        cell_cum_mass( nb_cells ) = running; // sentinel: total target mass
    } );
}

}

#undef UTP
#undef DTP
