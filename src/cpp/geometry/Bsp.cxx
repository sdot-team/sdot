#pragma once

#include "Bsp.h"

namespace sdot {

#define UTP template<class AdditionalPtData,class TF,int ct_dim>
#define DTP Bsp<AdditionalPtData,TF,ct_dim>

UTP DTP::Bsp( PI dim ) : dim( dim ) {
    nodes.push_back( dim );
}

UTP DTP::AvgData DTP::avg_data_for( TensorView<const TF,2> points ) const {
    ASSERT( points.size( 1 ) == dim );
    AvgData res( dim );

    const PI n = points.size( 0 );
    for( PI i = 0; i < n; ++i )
        for( PI j = 0; j < dim; ++j )
            res[ j ] += points( i, j );

    return res;
}

UTP DTP::CovData DTP::cov_data_for( TensorView<const TF,2> points ) const {
    ASSERT( points.size( 1 ) == dim );
    CovData res( dim * ( dim + 1 ) / 2 );

    Pt mu( dim );
    P( points.size( 0 ) );
    for( PI i = 0, n = points.size( 0 ); i < n; ++i )
        for( PI r = 0, o = 0; r < dim; ++r )
            for( PI c = 0; c <= r; ++c, ++o )
                res[ o ] += ( points( i, r ) - mu[ r ] ) * ( points( i, c ) - mu[ c ] );

    return res;
}


#undef UTP
#undef DTP

} // namespace sdot

template<class AdditionalPtData,class TF,int dim>
std::ostream &operator<<( std::ostream &os, const sdot::Bsp<AdditionalPtData,TF,dim> &p ) {
    return os << "pouet";
}
