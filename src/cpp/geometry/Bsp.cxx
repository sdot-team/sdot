#pragma once

#include "Bsp.h"

namespace sdot {

#define UTP template<class AdditionalPtData,class TF,int ct_dim>
#define DTP Bsp<AdditionalPtData,TF,ct_dim>

UTP DTP::Bsp( PI nb_points, PI dim ) : nb_points( nb_points ), dim( dim ) {
    nodes.push_back( dim );
}

UTP PI DTP::cell_number( Pt pos ) const {
    PI res = 0;
    while ( true ) {
        const Node &node = nodes[ res ];
        if ( node.final() )
            break;
        const TF s = dot( node.split_dir, pos ) - node.split_dot;
        res = node.child_indices[ PI( s > 0 ) ];
    }
    return res;
}

UTP auto DTP::avg_reduction( const std::vector<AvgData> &a, const std::vector<AvgData> &b ) -> std::vector<AvgData> {
    ASSERT( a.size() == b.size() );
    std::vector<AvgData> res;
    res.reserve( a.size() );
    for( PI i = 0; i < a.size(); ++i )
        res.emplace_back( a[ i ].sum + b[ i ].sum, a[ i ].len + b[ i ].len );
    return res;
}

UTP auto DTP::cov_reduction( const std::vector<CovData> &a, const std::vector<CovData> &b ) -> std::vector<CovData> {
    ASSERT( a.size() == b.size() );
    std::vector<CovData> res;
    res.reserve( a.size() );
    for( PI i = 0; i < a.size(); ++i )
        res.emplace_back( a[ i ].sum + b[ i ].sum, a[ i ].len + b[ i ].len );
    return res;
}

UTP auto DTP::avg_data_for( TensorView<const TF,2> points ) const -> std::vector<AvgData> {
    ASSERT( points.size( 1 ) == dim );

    // for each point, we find the cell, and we add the position in the corresponding AvgData
    std::vector<AvgData> res( nodes.size() );
    const PI nb_points = points.size( 0 );
    for( PI num_point = 0; num_point < nb_points; ++num_point ) {
        Pt pos( dim );
        for( PI j = 0; j < dim; ++j )
            pos[ j ] += points( num_point, j );

        AvgData &av = res[ cell_number( pos ) ];
        av.sum += pos;
        av.len += 1;
    }

    return res;
}

UTP std::vector<typename DTP::CovData> DTP::cov_data_for( TensorView<const TF,2> points, const std::vector<AvgData> &avg_data ) const {
    std::vector<CovData> res( nodes.size(), CovData{ dim * ( dim + 1 ) / 2 } );
    std::vector<Pt> avg( nodes.size() );
    ASSERT( avg_data.size() == nodes.size() );
    ASSERT( points.size( 1 ) == dim );

    for( PI num_node = 0; num_node < nodes.size(); ++num_node )
        avg[ num_node ] = avg_data[ num_node ].sum / TF( avg_data[ num_node ].len );

    for( PI num_point = 0; num_point < nb_points; ++num_point ) {
        Pt pos( dim );
        for( PI j = 0; j < dim; ++j )
            pos[ j ] += points( num_point, j );

        const PI cn = cell_number( pos );
        const Pt &av = avg[ cn ];
        CovData &cd = res[ cn ];

        for( PI r = 0, o = 0; r < dim; ++r )
            for( PI c = 0; c <= r; ++c, ++o )
                cd.sum[ o ] += ( pos[ r ] - av[ r ] ) * ( pos[ c ] - av[ c ] );
        cd.len += 1;
    }

    return res;
}

UTP typename DTP::Pt DTP::split_dir( const std::vector<CovData> &cov ) {
    cov.
    return { 17, 8 };
}

#undef UTP
#undef DTP

} // namespace sdot

template<class AdditionalPtData,class TF,int dim>
std::ostream &operator<<( std::ostream &os, const sdot::Bsp<AdditionalPtData,TF,dim> &p ) {
    return os << "pouet";
}
