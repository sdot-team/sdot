#pragma once

// the members + generated methods (`operator()`, `kernel_form`, ...) as macros this struct
// drops in, plus the axes it names -- written to the build include tree by `CallArg_Aggregate`.
#include <sdot/generated/aggregates/Image.h>
#include <loom/support/common_macros.h>
#include <type_traits>
#include "PolynomialPixel.h"

namespace sdot {

SDOT_TEMPLATE_DECL_FOR_Image
struct Image {
    SDOT_ATTRIBUTES_OF_Image

    SCInt  ct_dim       = DECAYED_TYPE_OF( nb_dims )::value;

    /// a POLYNOMIAL image ( `Image( degree = P )`, `P > 0` ): `coeffs [ pixels..., ( P + 1 )^d ]`, Bernstein, `values` unset
    /// ( `PolynomialPixel.h` ). Otherwise `values [ pixels... ]`, one constant per pixel
    static constexpr bool is_polynomial = DECAYED_TYPE_OF( coeffs )::is_valid;
    using  TF           = typename std::conditional_t<is_polynomial, DECAYED_TYPE_OF( coeffs ), DECAYED_TYPE_OF( values )>::TF;
    /// the degree along each axis: `( P + 1 )^d = nb_coeffs`
    static constexpr int degree = [] {
        if constexpr ( is_polynomial ) {
            constexpr int n = DECAYED_TYPE_OF( nb_coeffs )::value;
            for ( int p = 1; ; ++p ) {
                int r = 1;
                for ( int a = 0; a < ct_dim; ++a ) r *= p + 1;
                if ( r == n ) return p;
                if ( r > n ) return -1;
            }
        } else
            return 0;
    }();
    static_assert( degree >= 0, "Image: `nb_coeffs` is not a `( P + 1 )^d`" );

    /// the number of pixels along axis `a`
    HD SI nb_pixels( int a ) const {
        if constexpr ( is_polynomial ) return SI( coeffs.shape( a ) );
        else                           return SI( values.shape( a ) );
    }

    /// this distribution CUTS the cell into pieces ( one per block ): the integrator reserves
    /// a spare cell for it in the scratch ( `PowerDiagram::nb_work_cells` )
    static constexpr bool cuts_pieces = true;

    /// what this density's adjoint derives, for loom's `computed_grads` ( `flags`: the distribution's, under one
    /// output ). The masses are never read by the integration: their derivative is known to be zero.
    static void computed_grads( auto &&flags ) {
        // the value of each pixel ( `for_each_piece`'s `add_value_grad` ), or its coefficients -- its geometry: `affine_computed_grads`
        if constexpr ( requires { flags.values; } ) flags.values = true;
        if constexpr ( requires { flags.coeffs; } ) flags.coeffs = true;
        if constexpr ( requires { flags.current_mass; } ) flags.current_mass = true;
        if constexpr ( requires { flags.target_mass; } )  flags.target_mass  = true;
    }
    /// ... and its frame and origin, when the integration knows how ( not with a periodicity nor a sink: decided in Python )
    static void affine_computed_grads( auto &&flags ) {
        if constexpr ( requires { flags.frame; } )  flags.frame  = true;
        if constexpr ( requires { flags.origin; } ) flags.origin = true;
    }

    // ---- THE FRAME AND THE ORIGIN, differentiated ( `diagram::AffineAcc` ) ---------------------------------------------
    // The image is the affine image `x = origin + F^T t` of a fixed grid: the integral of a cell moves with `F` and `origin`
    // as `dJ = tr( X^T dA A^-1 ) + y . d origin`, `A = F^T` -- `X`, `y` accumulated by `diagram::integrate_bwd_into` -- so
    // `d J / d F = F^-T X^T`, `d J / d origin = y`.
    static constexpr bool is_affine = true;
    HD static constexpr bool affine_wanted( const auto &grad_dist ) {
        if constexpr ( requires { grad_dist.frame; grad_dist.origin; } )
            return DECAYED_TYPE_OF( grad_dist.frame )::is_valid || DECAYED_TYPE_OF( grad_dist.origin )::is_valid;
        else
            return false;
    }
    HD void affine_origin( auto *o ) const {
        for ( int a = 0; a < ct_dim; ++a )
            o[ a ] = 0;
        if constexpr ( DECAYED_TYPE_OF( origin )::is_valid )
            for ( int a = 0; a < ct_dim; ++a )
                o[ a ] = TF( origin( a ) );
    }
    HD void add_affine_grad( auto &&grad_dist, const auto &acc ) const;

    // build a FULLY-POPULATED image -- each of `origin` / `frame` / `knots` that is unbound
    // (a `NoneTensor`) replaced by its documented default -- and hand it to `cont`. Lets the
    // methods below be written ONCE against a complete image instead of gating on `is_valid`.
    HD auto   with_defaults( auto &&cont ) const;

    // total measure: sum over pixels of value * pixel volume ( a polynomial pixel: the mean of its Bernstein coefficients )
    HD TF  measure   () const;

    // ---- integrating a CELL against this image --------------------------------------------------
    // The image as a DISTRIBUTION: what `PowerDiagram::measures` calls when given
    // one. The contract (that of any distribution) is written in `distributions/Distribution.py`;
    // it fits in one sentence: cut `cell` into pieces on which the density is CONSTANT,
    // and pass each one to `func( piece, value, add_value_grad )`.
    //
    // Here a piece is `cell INTERSECT grid block`. The cutting is therefore only 2d cuts
    // per block met, made in the two spare cells that the caller provides
    // (`PieceWorkspace.h`) -- the original cell is never touched, which makes it possible
    // to open one piece after another without copying it.
    HD void   for_each_piece ( const auto &cell, auto &&ws, auto &&func ) const;
    HD void   _for_each_piece( const auto &cell, auto &&ws, auto &&func, const auto &L, const auto &s ) const;

    // the same against the image MOVED by the isometry `x -> L x + s` ( a `Matrix` and a `Vector`: periodicity, `Periodic.h` )
    HD void   for_each_piece_moved( const auto &cell, auto &&ws, auto &&func, const auto &L, const auto &s ) const;

    // the largest `i` such that `knots( axis, i ) <= t`, clamped to `[ 0, nb_cells - 1 ]` (the
    // knots are increasing, hence a binary search -- a production image has many blocks).
    HD SI  knot_index   ( SI axis, TF t, SI nb_cells ) const;

    // adjoint of `measure` w.r.t. `values`: since mass = Sum_c values(c) * cell_volume(c) is linear
    // in `values`, grad_values(c) = grad_mass * cell_volume(c). `grad_mass` is the scalar cotangent.
    HD void   measure_bwd  ( auto &&grad_values, auto &&grad_mass ) const;
    HD void   measure_coeffs_bwd( auto &&grad_coeffs, auto &&grad_mass ) const;

    // ... and w.r.t. the frame: mass = |det F| * ( the rest ), so grad_frame = grad_mass * mass * F^-T
    HD void   measure_frame_bwd( auto &&grad_frame, auto &&grad_mass ) const;

    // unidimension piece
    struct Udp          { SI index; TF pos; TF mass; TF y; };
    HD auto   udp_start () const;
    HD auto   udp_cont  ( auto &&udp, auto mass_to_take, auto &&cb_parts ) const;

    // jump directly to the `Udp` state a sequential walk would be in after having consumed exactly
    // `target_mass` -- via binary search over a PRECOMPUTED per-cell cumulative-mass array (built by
    // `fill_cell_cum_mass`), instead of walking there step by step. Lets independent work-items each
    // resume the walk at their own chunk's boundary in parallel.
    HD auto   udp_at    ( auto &&cell_cum_mass, auto target_mass ) const;

    // fills `cell_cum_mass[0..nb_cells]` (exclusive prefix of each cell's mass, sentinel = total
    // mass) -- cached on the PYTHON side as `Image.cell_cum_mass` (a `ComputedAttribute`, see
    // `distributions/Image.py`) so `SdotPlan1d` reads it ready-made instead of rebuilding it on every
    // forward/backward call. Sequential (one thread does a whole angle): `nb_cells` is small enough
    // that no cooperative scan is worth the complexity here.
    HD void   fill_cell_cum_mass( auto &&cell_cum_mass ) const;
};

}

#include "Image.cxx"

// its convolved density, for the transport solver ( `sdotplan/Convolved.h` )
#include "sdotplan/convolved/Image.h"
