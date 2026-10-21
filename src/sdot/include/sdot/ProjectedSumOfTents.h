#pragma once

#include <sdot/generated/aggregates/ProjectedSumOfTents.h>
#include <loom/support/common_macros.h>
#include <loom/support/atomic_add.h>

namespace sdot {

SDOT_TEMPLATE_DECL_FOR_ProjectedSumOfTents
struct ProjectedSumOfTents {
    SDOT_ATTRIBUTES_OF_ProjectedSumOfTents

    SCInt ct_proj = DECAYED_TYPE_OF( proj_dims )::value;  // ambient space
    using TF      = DECAYED_TYPE_OF( points )::TF;

    /// the projected center of tent i on THIS item's line
    HD TF position( SI i ) const {
        TF s = 0;
        for ( SI d = 0; d < ct_proj; ++d )
            s += TF( points( ::num_dirac = i, proj_dim = d ) ) * TF( normal( proj_dim = d ) );
        return s;
    }
    HD TF half_width( SI i ) const { return TF( half_widths( ::num_dirac = i ) ); }
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
        flags.half_widths = true;
    }

    /// d cost / d position(i) -> the shared ambient point, d cost / d half_width(i) -> the shared half-width: ATOMIC,
    /// every item contributes. Compile-time no-ops when not wanted.
    HD void add_position_grad( auto &&grad_src, SI i, TF g ) const {
        if constexpr ( DECAYED_TYPE_OF( grad_src.points )::is_valid )
            for ( SI d = 0; d < ct_proj; ++d )
                atomic_add( grad_src.points( ::num_dirac = i, proj_dim = d ).ref(), TF( g * TF( normal( proj_dim = d ) ) ) );
    }
    HD void add_half_width_grad( auto &&grad_src, SI i, TF g ) const {
        if constexpr ( DECAYED_TYPE_OF( grad_src.half_widths )::is_valid )
            atomic_add( grad_src.half_widths( ::num_dirac = i ).ref(), g );
    }

    /// clear the shared gradient buffers ONCE before the atomic adds ( a fresh FFI buffer is not zeroed )
    HD void zero_grads( auto &&queue, auto &&grad_src ) const {
        if constexpr ( DECAYED_TYPE_OF( grad_src.points )::is_valid )
            grad_src.points.fill_with( queue, TF( 0 ) );
        if constexpr ( DECAYED_TYPE_OF( grad_src.half_widths )::is_valid )
            grad_src.half_widths.fill_with( queue, TF( 0 ) );
    }
};

}
