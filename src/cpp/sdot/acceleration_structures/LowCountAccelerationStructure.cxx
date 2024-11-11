#pragma once

#include <tl/support/operators/product.h>
#include <tl/support/operators/norm_2.h>

#include "LowCountAccelerationStructure.h"
#include "CellTraversalError.h"

#include "../support/spawn.h"

#include "../CutType.h"

#include <tl/support/P.h>
#include <cmath>

namespace sdot {

#define DTP template<class TCell>
#define UTP LowCountAccelerationStructure<TCell>

DTP UTP::LowCountAccelerationStructure( const PoomVec<Pt> &positions, const PoomVec<TF> &weights, const Vec<Trans> &transformations ) : transformations( transformations ) {
    position_and_weights.resize( positions.size() );

    positions.get_values_by_chuncks( [&]( CstSpanView<Pt> positions ) {
        for( PI index = positions.beg_index(); index < positions.end_index(); ++index )
            position_and_weights[ index ].position = positions[ index ];
    } );

    weights.get_values_by_chuncks( [&]( CstSpanView<TF> weights ) {
        for( PI index = weights.beg_index(); index < weights.end_index(); ++index )
            position_and_weights[ index ].weight = weights[ index ];
    } );
}

DTP UTP::~LowCountAccelerationStructure() {
}

DTP void UTP::display( Displayer &ds ) const {
    DS_OJBECT( position_and_weights, transformations );
}

DTP PI UTP::recommended_nb_threads() const {
    return std::thread::hardware_concurrency();
}

DTP void UTP::make_cuts_from( PI i0, TCell &cell, Vec<PI> &buf ) {
    // "virtual" indices of points to be tested (includes virtual points)
    buf.clear();
    for( PI n1 = 0; n1 < position_and_weights.size() * ( 1 + transformations.size() ); ++n1 )
        if ( n1 != i0 )
            buf << n1;

    // get position from "virtual" index
    auto pos = [&]( PI n ) {
        const PI i = n % position_and_weights.size(), t = n / position_and_weights.size();
        const Pt res = position_and_weights[ i ].position;
        return t ? transformations[ t - 1 ]( res ) : res;  
    };

    // sort indices
    std::sort( buf.begin(), buf.end(), [&]( PI na, PI nb ) -> bool {
        return norm_2_p2( pos( na ) - cell.info.p0 ) < norm_2_p2( pos( nb ) - cell.info.p0 );
    } );

    // make the cuts
    const Pt p0 = cell.info.p0;
    const TF w0 = cell.info.w0;
    for( PI n1 : buf ) {
        const PI i1 = n1 % position_and_weights.size();
        const TF w1 = position_and_weights[ i1 ].weight;
        const Pt p1 = pos( n1 );

        const Pt dir = p1 - p0;

        auto n = norm_2_p2( dir );
        auto s0 = sp( dir, p0 );
        auto s1 = sp( dir, p1 );

        auto off = s0 + ( 1 + ( w0 - w1 ) / n ) / 2 * ( s1 - s0 );

        cell.cut( dir, off, { CutType::Dirac, p1, w1, i1 } );
    }
}

DTP int UTP::for_each_cell( const TCell &base_cell, const std::function<void( TCell &cell, int num_thread )> &f, int max_nb_threads ) {
    // for each box...
    int error = 0;
    spawn( [&]( int num_thread, int nb_threads ) {
        try {
            TCell local_cell;
            Vec<PI> buf( FromReservationSize(), position_and_weights.size() * ( 1 + transformations.size() ) );

            // "primary" points (not the virtual ones)
            PI beg_i0 = position_and_weights.size() * ( num_thread + 0 ) / nb_threads;
            PI end_i0 = position_and_weights.size() * ( num_thread + 1 ) / nb_threads;
            for( PI i0 = beg_i0; i0 < end_i0; ++i0 ) {
                if ( error )
                    return;

                // copy of base cell
                local_cell.get_geometrical_data_from( base_cell );

                Paw &paw = position_and_weights[ i0 ];
                local_cell.info.p0 = paw.position;
                local_cell.info.w0 = paw.weight;
                local_cell.info.i0 = i0;

                // cuts with points from b0
                make_cuts_from( i0, local_cell, buf );

                // callback
                f( local_cell, num_thread );
            }
        } catch ( CellTraversalError e ) {
            error = e.error;
        }
    }, max_nb_threads ? max_nb_threads : recommended_nb_threads() );

    return error;
}

#undef DTP
#undef UTP

} // namespace sdot
