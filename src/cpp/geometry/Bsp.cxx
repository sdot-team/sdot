#pragma once

#include "Bsp.h"

namespace sdot {

#define UTP template<class AdditionalPtData,class TF,int ct_dim>
#define DTP Bsp<AdditionalPtData,TF,ct_dim>

UTP DTP::Bsp( TensorView<const TF,3> all_the_paths, TensorView<const PI,1> indices, TensorView<const TF,2> points, TensorView<const TF,2> path ) : nb_points( points.size( 0 ) ), dim( points.size( 1 ) ) {
    // root node
    nodes.push_back( dim );

    // construction of the base nodes using all_the_paths
    P( all_the_paths.size( 0 ) );
    P( all_the_paths.size( 1 ) );
    P( all_the_paths );
    for( PI i = 0; i < all_the_paths.size( 0 ); ++i )
        add_path( all_the_paths.row( i ) );
}

UTP void DTP::add_path( TensorView<const TF,2> path ) {
    P( path );
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

UTP bool DTP::is_in_charge_of( const Pt &pos ) const {
    return true;
}

UTP auto DTP::sum_pos_for( TensorView<const TF,2> points ) const -> Pt {
    ASSERT( points.size( 1 ) == dim );
    Pt res( dim );

    // for each point, we find the cell, and we add the position in the corresponding AvgData
    const PI nb_points = points.size( 0 );
    for( PI num_point = 0; num_point < nb_points; ++num_point ) {
        Pt pos( dim );
        for( PI d = 0; d < dim; ++d )
            pos[ d ] = points( num_point, d );

        if ( is_in_charge_of( pos ) )
            res += pos;
    }

    return res;
}

UTP SimpleSquareMatrix<TF> DTP::sum_cov_for( TensorView<const TF,2> points, const Pt &avg ) const {
    ASSERT( points.size( 1 ) == dim );
    SimpleSquareMatrix<TF> res( dim );

    const PI nb_points = points.size( 0 );
    for( PI num_point = 0; num_point < nb_points; ++num_point ) {
        Pt pos( dim );
        for( PI d = 0; d < dim; ++d )
            pos[ d ] = points( num_point, d );

        if ( is_in_charge_of( pos ) )
            for( PI r = 0; r < dim; ++r )
                for( PI c = 0; c < dim; ++c )
                    res( r, c ) += ( pos[ r ] - avg[ r ] ) * ( pos[ c ] - avg[ c ] );
    }

    return res;
}

UTP auto DTP::split_hst_for( TensorView<const TF,2> points, const Pt &split_dir, TF split_beg, TF split_end, PI nb_bins ) const -> std::vector<TF> {
    using namespace std;
    std::vector<TF> res( nb_bins, 0 );

    const PI nb_points = points.size( 0 );
    for( PI num_point = 0; num_point < nb_points; ++num_point ) {
        Pt pos( dim );
        for( PI d = 0; d < dim; ++d )
            pos[ d ] = points( num_point, d );

        if ( is_in_charge_of( pos ) ) {
            TF n = ( dot( pos, split_dir ) - split_beg ) * nb_bins / ( split_end - split_beg );
            PI i = max( TF( 0 ), min( TF( nb_bins - 1 ), n ) );
            ++res[ i ];
        }
    }

    return res;
}

#undef UTP
#undef DTP

} // namespace sdot

template<class AdditionalPtData,class TF,int dim>
std::ostream &operator<<( std::ostream &os, const sdot::Bsp<AdditionalPtData,TF,dim> &p ) {
    return os << "pouet";
}
