#pragma once

#include <sdot/generated/aggregates/ProjectedSumOfAnisotropicTents.h>
#include <loom/support/common_macros.h>
#include <loom/support/atomic_add.h>

#include <cmath>
#include <type_traits>

namespace sdot {

SDOT_TEMPLATE_DECL_FOR_ProjectedSumOfAnisotropicTents
struct ProjectedSumOfAnisotropicTents {
    SDOT_ATTRIBUTES_OF_ProjectedSumOfAnisotropicTents

    SCInt ct_proj = DECAYED_TYPE_OF( proj_dims )::value;  // ambient space
    using TF      = DECAYED_TYPE_OF( points )::TF;

    /// the projected center of tent i on THIS item's line
    HD TF position( SI i ) const {
        TF s = 0;
        for ( SI d = 0; d < ct_proj; ++d )
            s += TF( points( ::num_dirac = i, proj_dim = d ) ) * TF( normal( proj_dim = d ) );
        return s;
    }
    /// coordinate d of the shared ambient point i
    HD TF point( SI i, SI d ) const { return TF( points( ::num_dirac = i, proj_dim = d ) ); }
    /// the half-width of tent i on THIS item's line: | M_i^T n |, M_i its matrix ( an ellipse's shadow ); it depends on the item
    HD TF half_width( SI i ) const {
        TF w2 = 0;
        for ( SI c = 0; c < ct_proj; ++c ) {
            TF u = 0;
            for ( SI r = 0; r < ct_proj; ++r )
                u += TF( matrices( ::num_dirac = i, proj_dim = r, proj_dim_b = c ) ) * TF( normal( proj_dim = r ) );
            w2 += u * u;
        }
        return std::sqrt( w2 );
    }
    /// the half-widths are NOT shared by the items ( the card's kernels, which read them from the first item, refuse this type )
    static constexpr bool shared_half_width = false;
    HD TF mass( SI i ) const { return TF( weights( ::num_dirac = i ) ); }
    HD SI nb_tents() const { return SI( weights.size() ); }

    /// the 3 n breakpoints of this item, unsorted: position, slope jump of the summed density, id ( 3 i + 0 / 1 / 2 ),
    /// the masses normalized by `total`
    HD void fill_breakpoints( auto &&pos, auto &&dslope, auto &&id, double total ) const {
        const SI n = nb_tents();
        for ( SI i = 0; i < n; ++i ) {
            const double s = double( position( i ) ), w = double( half_width( i ) ), k = double( mass( i ) ) / total / ( w * w );
            pos( 3 * i + 0 ) = TF( s - w ); dslope( 3 * i + 0 ) = TF(  k     ); id( 3 * i + 0 ) = 3 * i + 0;
            pos( 3 * i + 1 ) = TF( s     ); dslope( 3 * i + 1 ) = TF( -2 * k ); id( 3 * i + 1 ) = 3 * i + 1;
            pos( 3 * i + 2 ) = TF( s + w ); dslope( 3 * i + 2 ) = TF(  k     ); id( 3 * i + 2 ) = 3 * i + 2;
        }
    }

    /// what the transport derives here, for loom's `computed_grads`: the ambient points ( through the projection ) and
    /// the half-widths -- not the projection direction, not the masses ( yet )
    static void computed_grads( auto &&flags ) {
        flags.points      = true;
        flags.matrices    = true;
    }

    /// d cost / d position(i) -> the shared ambient point, d cost / d half_width(i) -> the matrix: ATOMIC,
    /// every item contributes. Compile-time no-ops when not wanted.
    HD void add_position_grad( auto &&grad_src, SI i, TF g ) const {
        if constexpr ( DECAYED_TYPE_OF( grad_src.points )::is_valid )
            for ( SI d = 0; d < ct_proj; ++d )
                atomic_add( grad_src.points( ::num_dirac = i, proj_dim = d ).ref(),
                            std::decay_t<decltype( grad_src.points( ::num_dirac = i, proj_dim = d ).ref() )>( g * TF( normal( proj_dim = d ) ) ) );
    }
    /// d cost / d half_width(i) = g -> the matrix: g n ( M^T n )^T / w, atomic
    HD void add_half_width_grad( auto &&grad_src, SI i, TF g ) const {
        if constexpr ( requires { grad_src.matrices; } && DECAYED_TYPE_OF( grad_src.matrices )::is_valid ) {
            const TF w = half_width( i );
            for ( SI c = 0; c < ct_proj; ++c ) {
                TF u = 0;
                for ( SI r = 0; r < ct_proj; ++r )
                    u += TF( matrices( ::num_dirac = i, proj_dim = r, proj_dim_b = c ) ) * TF( normal( proj_dim = r ) );
                for ( SI r = 0; r < ct_proj; ++r )
                    atomic_add( grad_src.matrices( ::num_dirac = i, proj_dim = r, proj_dim_b = c ).ref(),
                                std::decay_t<decltype( grad_src.matrices( ::num_dirac = i, proj_dim = r, proj_dim_b = c ).ref() )>( g * TF( normal( proj_dim = r ) ) * u / w ) );
            }
        }
    }

    /// the d-th component of this item's direction, and the gradient of the shared ambient point i along d ( atomic )
    HD TF normal_component( SI d ) const { return TF( normal( proj_dim = d ) ); }
    HD void add_point_grad( auto &&grad_src, SI i, SI d, TF g ) const {
        if constexpr ( DECAYED_TYPE_OF( grad_src.points )::is_valid )
            atomic_add( grad_src.points( ::num_dirac = i, proj_dim = d ).ref(), std::decay_t<decltype( grad_src.points( ::num_dirac = i, proj_dim = d ).ref() )>( g ) );
    }

    /// clear the shared gradient buffers ONCE before the atomic adds ( a fresh FFI buffer is not zeroed )
    void zero_grads( auto &&queue, auto &&grad_src ) const {
        if constexpr ( DECAYED_TYPE_OF( grad_src.points )::is_valid )
            grad_src.points.fill_with( queue, TF( 0 ) );
        if constexpr ( requires { grad_src.matrices; } && DECAYED_TYPE_OF( grad_src.matrices )::is_valid )
            grad_src.matrices.fill_with( queue, TF( 0 ) );
    }
};

}
