#pragma once

#include "../support/Mpi.h"
#include "RegularGrid.h"

namespace sdot {

#define DTP template<class TF,int nb_dims>
#define UTP RegularGrid<TF,nb_dims>

DTP PI UTP::RegularGrid::end_index() const {
    return product( nb_divs );
}

DTP PI UTP::RegularGrid::index( const Pt &pos, int dim ) const {
    return min( nb_divs[ dim ] - 1, PI( ( pos[ dim ] - min_pos[ dim ] ) / steps[ dim ] ) );
}

DTP PI UTP::RegularGrid::index( const Pt &pos ) const {
    PI res = index( pos, 0 );
    for( PI d = 1, m = 1; d < nb_dims; ++d )
        res += ( m *= nb_divs[ d - 1 ] ) * index( pos, d );
    return res;
}

DTP UTP::RegularGrid( const auto &points, const Vec<Trans> &transformations ) : transformations( transformations ) {
    nb_base_points = points.global_size();

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
    limits = mpi->reduction( limits, []( auto &a, const auto &b ) {
        a[ 0 ] = min( a[ 0 ], b[ 0 ] );
        a[ 1 ] = max( a[ 1 ], b[ 1 ] );
    } );

    P( limits );
}

DTP void UTP::display( Displayer &ds ) const {
    DS_OJBECT( limits );
}

#undef DTP
#undef UTP

} // namespace sdot
