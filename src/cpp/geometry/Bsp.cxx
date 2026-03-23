#pragma once

#include "Bsp.h"

namespace sdot {

#define UTP template<class AdditionalPtData,class TF,int ct_dim>
#define DTP Bsp<AdditionalPtData,TF,ct_dim>

UTP DTP::Bsp( TensorView<const TF,3> all_the_paths, TensorView<const PI,1> indices, TensorView<const TF,2> points, TensorView<const TF,2> path, PI max_points_per_cell ) : nb_points( points.size( 0 ) ), dim( points.size( 1 ) ) {
    // copy point data
    pt_data.resize( points.size( 0 ) );
    for( PI i = 0; i < points.size( 0 ); ++i ) {
        for( PI d = 0; d < dim; ++d )
            pt_data[ i ].position[ d ] = points( i, d );
        pt_data[ i ].index = indices[ i ];
    }

    // root node
    nodes.push_back( dim );

    // construction of the base nodes using `all_the_paths`
    for( PI i = 0; i < all_the_paths.size( 0 ); ++i )
        add_path( all_the_paths.row( i ), i );

    // make the local nodes
    PI node_index = 0;
    for( PI r = 0; r < path.size( 0 ); ++r )
        node_index = nodes[ node_index ].child_indices[ path( r, dim + 1 ) ];
    fill_node( node_index, 0, pt_data.size(), max_points_per_cell );

    // get the node cells

}

UTP void DTP::fill_node( PI node_index, PI beg_pt_data, PI end_pt_data, PI max_points_per_cell ) {
    using namespace std;

    Node &node = nodes[ node_index ];
    node.beg_pt_data = beg_pt_data;
    node.end_pt_data = end_pt_data;

    const PI nb_points = end_pt_data - beg_pt_data;
    if ( nb_points <= max_points_per_cell )
        return;

    // avg
    Pt avg( dim );
    for( PI num_point = 0; num_point < nb_points; ++num_point )
        avg += pt_data[ num_point ].position;
    avg /= nb_points;

    // cov — fill lower triangular, then symmetrize and divide
    SimpleSquareMatrix<TF> cov( dim );
    for( PI num_point = 0; num_point < nb_points; ++num_point )
        for( PI r = 0; r < dim; ++r )
            for( PI c = 0; c <= r; ++c )
                cov( r, c ) += ( pt_data[ num_point ].position[ r ] - avg[ r ] ) * ( pt_data[ num_point ].position[ c ] - avg[ c ] );
    for( PI r = 0; r < dim; ++r )
        for( PI c = 0; c <= r; ++c )
            cov( c, r ) = cov( r, c ) = cov( r, c ) / TF( nb_points );

    // split dir — dominant eigenvector
    auto es = cov.eigen_system();
    PI n = 0;
    for( PI i = 1; i < dim; ++i )
        if( es.values[ i ] > es.values[ n ] )
            n = i;
    for( PI d = 0; d < dim; ++d )
        node.split_dir[ d ] = es.vectors( n, d );

    //
    const TF eig_val = es.values[ n ];
    const TF avg_dot = dot( node.split_dir, avg );
    const TF beg_dot = avg_dot - 15 * eig_val;
    const TF end_dot = avg_dot + 15 * eig_val;
    std::vector<PI> bins( 256 );
    for( PI num_point = 0; num_point < nb_points; ++num_point ) {
        TF p = dot( pt_data[ num_point ].position, node.split_dir ) - beg_dot;
        p = max( TF( 0 ), min( TF( bins.size() - 1 ), bins.size() * p / ( end_dot - beg_dot ) ) );
        ++bins[ p ];
    }

    //
    node.split_dot = avg_dot;
    for( PI b = 0, acc = 0; b < bins.size(); ++b ) {
        if ( acc >= nb_points / 2 ) {
            node.split_dot = beg_dot + ( end_dot - beg_dot ) * b / bins.size();
            break;
        }
        acc += bins[ b ];
    }

    //
    PI beg = beg_pt_data;
    PI end = end_pt_data;
    while ( beg < end ) {
        const TF sp = dot( pt_data[ beg ].position, node.split_dir ) - node.split_dot;
        if ( sp > 0 )
            std::swap( pt_data[ beg ], pt_data[ --end ] );
        else
            ++beg;
    }
    if ( beg == beg_pt_data || beg == end_pt_data )
        throw std::runtime_error( "TODO: points in the same place" );

    //
    PI n0 = nodes.size() + 0;
    PI n1 = nodes.size() + 1;
    node.child_indices = { n0, n1 };
    nodes.emplace_back( dim );
    nodes.emplace_back( dim );

    //
    fill_node( n0, beg_pt_data, beg, max_points_per_cell );
    fill_node( n1, beg, end_pt_data, max_points_per_cell );
}

UTP void DTP::add_path( TensorView<const TF,2> path, PI num_bsp ) {
    Node *node = &nodes[ 0 ];
    for( PI r = 0; r < path.size( 0 ); ++r ) {
        // if we already have the child, go to it
        const PI side = path( r, dim + 1 );
        if ( PI child_index = node->child_indices[ side ] ) {
            node = &nodes[ child_index ];
            continue;
        }

        // else, add the new node
        for( PI d = 0; d < dim; ++d )
            node->split_dir[ d ] = path( r, d );
        node->split_dot = path( r, dim );
        node->num_bsp = num_bsp;

        node->child_indices[ side ] = nodes.size();
        node = &nodes.emplace_back( dim );
    }

    node->num_bsp = num_bsp;
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

UTP void DTP::display_rec( std::ostream &os, PI node_index, std::string prefix ) const {
    const Node &node = nodes[ node_index ];
    if ( node.final() ) {
        if ( PI nb_points = node.nb_points() )
            os << prefix << "nb_points: " << nb_points;
        else
            os << prefix << "num_bsp: " << node.num_bsp;
        return;
    }

    os << prefix << "dir: " << node.split_dir << " dot: " << node.split_dot;
    for( PI num_split = 0; num_split < 2; ++num_split )
        if ( node.child_indices[ num_split ] > node_index )
            display_rec( os << "\n", node.child_indices[ num_split ], prefix + "  " );
}

#undef UTP
#undef DTP

} // namespace sdot

template<class AdditionalPtData,class TF,int dim>
std::ostream &operator<<( std::ostream &os, const sdot::Bsp<AdditionalPtData,TF,dim> &p ) {
    p.display_rec( os, 0 );
    return os;
}
