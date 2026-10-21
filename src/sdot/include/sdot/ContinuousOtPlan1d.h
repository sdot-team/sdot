#pragma once

#include <sdot/generated/aggregates/ContinuousOtPlan1d.h>
#include <sdot/ot1d/tents_1d.h>
#include <loom/support/common_macros.h>

namespace sdot {

/// sort the indices `order( 0 .. n-1 )` by `key( order( k ) )`, in place: a heap sort on accessors ( no pointer, no
/// allocation, the same on the host and on one thread of the card )
HD void heap_sort_by_key( auto &&order, auto &&key, SI n ) {
    auto sift = [&]( SI start, SI end ) {
        SI root = start;
        while ( 2 * root + 1 < end ) {
            SI child = 2 * root + 1;
            if ( child + 1 < end && key( SI( order( child ) ) ) < key( SI( order( child + 1 ) ) ) )
                ++child;
            if ( !( key( SI( order( root ) ) ) < key( SI( order( child ) ) ) ) )
                return;
            const SI t = SI( order( root ) ); order( root ) = SI( order( child ) ); order( child ) = t;
            root = child;
        }
    };
    for ( SI start = n / 2 - 1; start >= 0; --start )
        sift( start, n );
    for ( SI end = n - 1; end > 0; --end ) {
        const SI t = SI( order( 0 ) ); order( 0 ) = SI( order( end ) ); order( end ) = t;
        sift( 0, end );
    }
}

SDOT_TEMPLATE_DECL_FOR_ContinuousOtPlan1d
struct ContinuousOtPlan1d {
    SDOT_ATTRIBUTES_OF_ContinuousOtPlan1d

    using TF = DECAYED_TYPE_OF( cost )::TF;

    /// the transport of THIS item ( one view ): the tents of `src_dist` against the image `dst_dist`, through the generic
    /// 1D transport; `on_tent( i, mass_i, moments_i )` per tent ( `ot1d::TentMoments`, the masses normalized ). Returns the
    /// cost. `bp_*` and `sums`: this thread's scratch rows ( 3 n breakpoints, their sums ).
    HD double run( auto &&bp_pos, auto &&bp_dslope, auto &&bp_id, auto &&sums, auto &&on_tent ) const {
        const SI n = src_dist.nb_tents();
        double total = 0;
        for ( SI i = 0; i < n; ++i )
            total += double( src_dist.mass( i ) );
        src_dist.fill_breakpoints( bp_pos, bp_dslope, bp_id, total );
        heap_sort_by_key( bp_id, [&]( SI j ) { return double( bp_pos( j ) ); }, 3 * n );

        auto src = ot1d::tent_stream( [&]( long k ) { return double( bp_pos( SI( bp_id( k ) ) ) ); },
                                      [&]( long k ) { return double( bp_dslope( SI( bp_id( k ) ) ) ); },
                                      [&]( long k ) { return int( bp_id( k ) ); }, long( 3 * n ) );
        ot1d::Sums1d S;
        dst_dist.with_defaults( [&]( auto &&img ) {
            const SI nb = img.values.size();
            const double sc = double( img.frame( 0, 0 ) ), of = double( img.origin( 0 ) );
            double tot = 0;
            for ( SI k = 0; k < nb; ++k )
                tot += sc * ( double( img.knots( k + 1 ) ) - double( img.knots( k ) ) ) * double( img.values[ k ] );
            auto dst = ot1d::func_stream0( [&]( long k ) {
                ot1d::PolyPiece1d<double,0> p;
                p.x0 = of + sc * double( img.knots( k ) );
                p.x1 = of + sc * double( img.knots( k + 1 ) );
                p.c[ 0 ] = double( img.values[ k ] ) / tot;
                return p;
            }, long( nb ) );
            S = ot1d::transport_1d<3>( src, dst, [&]( int id, const ot1d::Sums1d &s ) {
                sums( id, 0 ) = s.P0; sums( id, 1 ) = s.P1; sums( id, 2 ) = s.P2; sums( id, 3 ) = s.phi;
            } );
        } );

        auto at = [&]( SI id ) {
            ot1d::Sums1d s;
            s.P0 = double( sums( id, 0 ) ); s.P1 = double( sums( id, 1 ) ); s.P2 = double( sums( id, 2 ) ); s.phi = double( sums( id, 3 ) );
            return s;
        };
        for ( SI i = 0; i < n; ++i ) {
            const auto m = ot1d::tent_moments( double( src_dist.position( i ) ), double( src_dist.half_width( i ) ),
                                               at( 3 * i ), at( 3 * i + 1 ), at( 3 * i + 2 ) );
            on_tent( i, double( src_dist.mass( i ) ) / total, m );
        }
        return S.cost;
    }

    /// the forward: the cost, and the barycenters / potentials ( the mass derivative, centered ) when they are outputs
    HD void update_outputs( auto &&bp_pos, auto &&bp_dslope, auto &&bp_id, auto &&sums ) {
        double phi_mean = 0;
        const double c = run( bp_pos, bp_dslope, bp_id, sums, [&]( SI i, double mi, const ot1d::TentMoments &m ) {
            if constexpr ( DECAYED_TYPE_OF( barycenters )::is_valid )
                barycenters( ::num_dirac = i ) = TF( m.b );
            if constexpr ( DECAYED_TYPE_OF( potentials )::is_valid ) {
                potentials( ::num_dirac = i ) = TF( m.phi );
                phi_mean += mi * m.phi;
            }
        } );
        if constexpr ( DECAYED_TYPE_OF( potentials )::is_valid )
            for ( SI i = 0; i < SI( potentials.size() ); ++i )
                potentials( ::num_dirac = i ) = TF( double( potentials( ::num_dirac = i ) ) - phi_mean );
        nb_diracs.set( src_dist.nb_diracs );
        cost = TF( c );
    }

    /// what the adjoint derives ( loom's `computed_grads` ): what the source says
    static void update_outputs_computed_grads( auto &&flags ) {
        DECAYED_TYPE_OF( src_dist )::computed_grads( flags.src_dist );
    }

    /// the adjoint of the cost: the sweep again, then per tent d / d s_i = 2 m_i ( s_i - b_i ) through the projection and
    /// d / d w_i = ( 2 m_i / w_i ) ( w_i^2 / 6 - Q_i ), accumulated by atomics on the shared points / half-widths
    HD void update_outputs_bwd( auto &&grad_plan, auto &&bp_pos, auto &&bp_dslope, auto &&bp_id, auto &&sums ) const {
        const double g = double( grad_plan.cost );
        run( bp_pos, bp_dslope, bp_id, sums, [&]( SI i, double mi, const ot1d::TentMoments &m ) {
            const double s = double( src_dist.position( i ) ), w = double( src_dist.half_width( i ) );
            src_dist.add_position_grad( grad_plan.src_dist, i, TF( g * 2 * mi * ( s - m.b ) ) );
            src_dist.add_half_width_grad( grad_plan.src_dist, i, TF( g * 2 * mi / w * ( w * w / 6 - m.Q ) ) );
        } );
    }
};

}
