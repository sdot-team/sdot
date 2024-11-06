#pragma once

#include "../support/Mpi.h"
#include "RegularGrid.h"

#include <tl/support/P.h>

namespace sdot {

#define DTP template<class TF,int nb_dims>
#define UTP RegularGrid<TF,nb_dims>

DTP PI UTP::RegularGrid::end_index() const {
    return product( nb_divs );
}

DTP PI UTP::RegularGrid::index( const Pt &pos, int dim ) const {
    return min( nb_divs[ dim ] - 1, PI( ( pos[ dim ] - limits[ 0 ][ dim ] ) / steps[ dim ] ) );
}

DTP PI UTP::RegularGrid::index( const Pt &pos ) const {
    PI res = index( pos, 0 );
    for( PI d = 1, m = 1; d < nb_dims; ++d )
        res += ( m *= nb_divs[ d - 1 ] ) * index( pos, d );
    return res;
}

DTP UTP::RegularGrid( const auto &points, const Vec<Trans> &transformations, TF nb_diracs_per_box ) : transformations( transformations ) {
    nb_base_points = points.global_size();
    nb_glob_points = nb_base_points * ( 1 + transformations.size() );

    // local min_pos, max_pos
    bool seen = false;
    points.get_local_content( [&]( const auto &content ) {
        if ( content.size() == 0 )
            return;
        if ( ! seen ) {
            limits[ 0 ] = *content.begin();
            limits[ 1 ] = limits[ 0 ];
            seen = true;
        }
        for( const auto &p : content ) {
            limits[ 0 ] = min( limits[ 0 ], p );
            limits[ 1 ] = max( limits[ 1 ], p );
        }
    } );

    // global min_pos ans max_pos
    limits = mpi->reduction<Vec<Pt,2>>( limits, []( Vec<Pt,2> &a, const Vec<Pt,2> &b ) {
        a[ 0 ] = min( a[ 0 ], b[ 0 ] );
        a[ 1 ] = max( a[ 1 ], b[ 1 ] );
    }, {} );

    //
    TF step = pow( nb_glob_points / ( nb_diracs_per_box * product( limits[ 1 ] - limits[ 0 ] ) ), double( 1 ) / nb_dims );
    for( PI d = 0; d < nb_dims; ++d ) {
        nb_divs[ d ] = max( TF( 1 ), ceil( ( limits[ 1 ][ d ] - limits[ 0 ][ d ] ) / step ) );
        steps[ d ] = TF( limits[ 1 ][ d ] - limits[ 0 ][ d ] ) / nb_divs[ d ];
    }

    // get local nb diracs for each cell (stored in this->offsets)
    offsets.resize( product( nb_divs ) + 1, 0 );
    for( PI i = 0, s = dv.local_size(); i < s; ++i )
        ++offsets[ index( dv.local_pos( i ) ) ];

    // exclusive scan
    if ( mpi->size() > 1 )
        TODO;
    for( PI i = 0, a = 0; i < offsets.size(); ++i )
        offsets[ i ] = exchange( a, a + offsets[ i ] );

    // get points and inds
    points.resize( dv.local_size() );
    inds.resize( dv.local_size() );
    for( PI i = 0, s = dv.local_size(); i < s; ++i ) {
        Pt pt = dv.local_pos( i );

        PI o = offsets[ index( pt ) ]++;
        points[ o ] = pt;
        inds[ o ] = i;
    }

    // remake the exclusive scan
    for( PI i = 0, o = 0; i < offsets.size() - 1; ++i )
        offsets[ i ] = std::exchange( o, offsets[ i ] );
}

DTP void UTP::display( Displayer &ds ) const {
    DS_OJBECT( limits );
}

#undef DTP
#undef UTP

} // namespace sdot
