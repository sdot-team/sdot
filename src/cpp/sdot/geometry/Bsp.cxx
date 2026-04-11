#pragma once

//#include "../support/P.h"
#include "Bsp.h"

namespace sdot {

#define UTP template<class AdditionalPtData,class TF,int ct_dim,class Arch>
#define DTP Bsp<AdditionalPtData,TF,ct_dim,Arch>

UTP DTP::Bsp( const auto &node_summary, PI node_index, TensorView<const TF,2,Arch> positions, TensorView<const PI,1,Arch> indices, TensorView<const TF,1,Arch> weights, PI max_points_per_cell ) : nb_points( positions.size( 0 ) ), dim( positions.size( 1 ) ), pf( dim ) {
    // pt_data
    const PI nb_points = positions.size( 0 );
    pt_data.reserve( nb_points );
    for( PI num_point = 0; num_point < positions.size( 0 ); ++num_point ) {
        pt_data.push_back( {
            .additional_data = 0,
            .position = positions.row( num_point ),
            .weight = weights[ num_point ],
            .index = indices[ num_point ],
        } );
    }

    // cells
    cells.reserve( node_summary.size() );
    for( PI num_node = 0; num_node < node_summary.size(); ++num_node )
        cells.push_back( std::get<1>( node_summary[ num_node ] ) );

    //
    nodes.resize( node_summary.size() );
    for( PI num_node = 0; num_node < node_summary.size(); ++num_node ) {
        const auto &src = node_summary[ num_node ];
        Node &dst = nodes[ num_node ];

        const PI ci0 = std::get<0>( src )[ 0 ];
        const PI ci1 = std::get<0>( src )[ 1 ];

        // split ?
        if ( ci0 ) {
            dst.type = NodeType::Split;
            dst.data.split.child_indices = { ci0, ci1 };
            continue;
        }

        // final ?
        if ( num_node == node_index ) {
            dst.type = NodeType::Final;
            dst.data.final.beg_pt_data = 0;
            dst.data.final.end_pt_data = nb_points;
            continue;
        }

        // -> ext
        dst.type = NodeType::Ext;
        dst.data.ext.num_item = ci1;
    }

    // make the local nodes
    update_children_of( node_index, max_points_per_cell );
}

UTP void DTP::display_vtk( VtkOutput &vo ) const {
    for( PI num_node = 0; num_node < nodes.size(); ++num_node ) {
        cells[ num_node ].display_vtk( vo );

        const Node &node = nodes[ num_node ];
        if ( node.type == NodeType::Final )
            for( PI i = node.data.final.beg_pt_data; i < node.data.final.end_pt_data; ++i )
               vo.add_point( pt_data[ i ].position );
    }
}

UTP void DTP::update_children_of( PI node_index, PI max_points_per_cell ) {
    using namespace std;

    // small enough ?
    Node &node = nodes[ node_index ];
    ASSERT( node.type == NodeType::Final );
    const PI beg_pt_data = node.data.final.beg_pt_data;
    const PI end_pt_data = node.data.final.end_pt_data;
    if ( end_pt_data - beg_pt_data <= max_points_per_cell )
        return;

    //

    // // avg
    // Pt avg( dim );
    // for( PI num_point = 0; num_point < nb_points; ++num_point )
    //     avg += pt_data[ num_point ].position;
    // avg /= nb_points;

    // // cov — fill lower triangular, then symmetrize and divide
    // SimpleSquareMatrix<TF,-1,Arch> cov( dim );
    // for( PI num_point = 0; num_point < nb_points; ++num_point )
    //     for( PI r = 0; r < dim; ++r )
    //         for( PI c = 0; c <= r; ++c )
    //             cov( r, c ) += ( pt_data[ num_point ].position[ r ] - avg[ r ] ) * ( pt_data[ num_point ].position[ c ] - avg[ c ] );
    // for( PI r = 0; r < dim; ++r )
    //     for( PI c = 0; c <= r; ++c )
    //         cov( c, r ) = cov( r, c ) = cov( r, c ) / TF( nb_points );

    // // split dir — dominant eigenvector
    // auto es = cov.eigen_system();
    // PI n = 0;
    // for( PI i = 1; i < dim; ++i )
    //     if( es.values[ i ] > es.values[ n ] )
    //         n = i;
    // for( PI d = 0; d < dim; ++d )
    //     node.split_dir[ d ] = es.vectors( n, d );

    // //
    // const TF eig_val = es.values[ n ];
    // const TF avg_dot = dot( node.split_dir, avg );
    // const TF beg_dot = avg_dot - 15 * eig_val;
    // const TF end_dot = avg_dot + 15 * eig_val;

    Pt min_pos = pt_data[ beg_pt_data ].position;
    Pt max_pos = pt_data[ beg_pt_data ].position;
    for( PI i = beg_pt_data + 1; i < end_pt_data; ++i ) {
        min_pos = min( min_pos, pt_data[ i ].position );
        max_pos = max( max_pos, pt_data[ i ].position );
    }

    PI d = ( max_pos - min_pos ).arg_max();
    Pt split_dir = pf.value_at( d, 1 );
    TF beg_dot = dot( split_dir, min_pos );
    TF end_dot = dot( split_dir, max_pos );

    std::vector<PI> bins( 256 );
    for( PI num_point = beg_pt_data; num_point < end_pt_data; ++num_point ) {
        TF p = dot( pt_data[ num_point ].position, split_dir ) - beg_dot;
        p = std::max( TF( 0 ), std::min( TF( bins.size() - 1 ), bins.size() * p / ( end_dot - beg_dot ) ) );
        ++bins[ p ];
    }

    //
    TF split_dot = ( beg_dot + end_dot ) / 2;
    for( PI b = 0, acc = 0; b < bins.size(); ++b ) {
        if ( acc >= nb_points / 2 ) {
            split_dot = beg_dot + ( end_dot - beg_dot ) * b / bins.size();
            break;
        }
        acc += bins[ b ];
    }

    //
    PI beg = beg_pt_data;
    PI end = end_pt_data;
    while ( beg < end ) {
        const TF sp = dot( pt_data[ beg ].position, split_dir ) - split_dot;
        if ( sp > 0 )
            std::swap( pt_data[ beg ], pt_data[ --end ] );
        else
            ++beg;
    }
    if ( beg == node.data.final.beg_pt_data || beg == node.data.final.end_pt_data )
        throw std::runtime_error( "TODO: points in the same place" );

    //
    PI n0 = nodes.size() + 0;
    PI n1 = nodes.size() + 1;
    node.type = NodeType::Split;
    node.data.split.child_indices = { n0, n1 };

    nodes.push_back( { .type = NodeType::Final, .data = { .final = { beg_pt_data, beg } } } );
    nodes.push_back( { .type = NodeType::Final, .data = { .final = { beg, end_pt_data } } } );

    cells.push_back( make_new_cell( cells[ node_index ], + split_dir, beg_pt_data, beg ) );
    cells.push_back( make_new_cell( cells[ node_index ], - split_dir, beg, end_pt_data ) );

    //
    update_children_of( n0, max_points_per_cell );
    update_children_of( n1, max_points_per_cell );
}

UTP typename DTP::Cell DTP::make_new_cell( const Cell &base_cell, const Pt &split_dir, PI beg, PI end ) {
    using namespace std;

    Cell new_cell = cells[ 0 ];
    auto cut = [&]( const Pt split_dir ) {
        TF sp = dot( pt_data[ beg ].position, split_dir );
        for( PI ind = beg + 1; ind < end; ++ind )
            sp = max( sp, dot( pt_data[ ind ].position, split_dir ) );
        new_cell.cut( split_dir, sp, {} );
    };

    base_cell.for_each_cut( [&]( const Pt split_dir, TF, const auto & ) {
        cut( split_dir );
    } );
    cut( split_dir );

    return new_cell;
}

UTP void DTP::display_rec( std::ostream &os, PI node_index, std::string prefix ) const {
    const Node &node = nodes[ node_index ];
    switch ( node.type ) {
        case NodeType::Split:
            os << prefix << "split:";
            for( PI num_split = 0; num_split < 2; ++num_split )
                display_rec( os << "\n", node.data.split.child_indices[ num_split ], prefix + "  " );
            return;
        case NodeType::Final:
            os << prefix << "final( beg: " << node.data.final.beg_pt_data << ", end: " << node.data.final.end_pt_data << " )";
            return;
        case NodeType::Ext:
            os << prefix << "ext( " << node.data.ext.num_item << " )";
            return;
    }
}

UTP void DTP::for_each_cell( const auto &primitive, const auto &sorted_potentials, auto &&func ) {
    using TR = DECAYED_TYPE_OF( TF( 0 ) + sorted_potentials( 0 ) );

    sdot::Cell<TR,ct_dim,Arch> base_cell = primitive.template englobing_cell<TR>( dim, {}, { .global_dirac_index = PI( -1 ) } );

    for( PI n0 = 0; n0 < pt_data.size(); ++n0 ) {
        const TR p0 = sorted_potentials( n0 );
        const Pt v0 = pt_data[ n0 ].position;
        const TR w0 = pt_data[ n0 ].weight;
        const PI i0 = pt_data[ n0 ].index;

        sdot::Cell<TR,ct_dim,Arch> cell = base_cell;
        cell.info.global_dirac_index = i0;
        cell.info.local_dirac_index = n0;
        cell.info.dirac_position = v0;
        cell.info.dirac_weight = w0;
        cell.info.potential = w0;

        for( PI n1 = 0; n1 < pt_data.size(); ++n1 ) {
            if ( n0 == n1 )
                continue;
            const TR p1 = sorted_potentials( n1 );
            const Pt v1 = pt_data[ n1 ].position;
            const TR w1 = pt_data[ n1 ].weight;
            const PI i1 = pt_data[ n1 ].index;

            const Pt dir = v1 - v0;

            auto n = norm_2_p2( dir );
            auto s0 = dot( dir, v0 );
            auto s1 = dot( dir, v1 );

            auto off = s0 + ( 1 + ( p0 - p1 ) / n ) / 2 * ( s1 - s0 );

            cell.cut( dir, off, {
                .global_dirac_index = i1,
                .local_dirac_index = n1,
                .dirac_position = v1,
                .dirac_weight = w1,
                .potential = p1
            } );
        }

        func( cell );
    }
}

UTP auto DTP::remake_cell( const auto &cell, const auto &primitive, const auto &sorted_potentials ) {
    using TR = DECAYED_TYPE_OF( TF( 0 ) + sorted_potentials( 0 ) );

    const PI n0 = cell.info.local_dirac_index;
    const TR p0 = sorted_potentials( n0 );
    const Pt v0 = pt_data[ n0 ].position;
    const TR w0 = pt_data[ n0 ].weight;
    const PI i0 = pt_data[ n0 ].index;

    sdot::Cell<TR,ct_dim,Arch> res = primitive.template englobing_cell<TR>( dim, {}, { .global_dirac_index = PI( -1 ) } );
    res.info.global_dirac_index = i0;
    res.info.local_dirac_index = n0;
    res.info.dirac_position = v0;
    res.info.dirac_weight = w0;
    res.info.potential = w0;

    cell.for_each_cut( [&]( const auto &cut_dir, const auto &cut_dot, const auto &cut_info ) {
        if ( cut_info.global_dirac_index == PI( -1 ) ) {
            res.cut( cut_dir, cut_dot, {
                .global_dirac_index = cut_info.global_dirac_index,
                .local_dirac_index = cut_info.local_dirac_index,
                .dirac_position = cut_info.dirac_position,
                .dirac_weight = cut_info.dirac_weight,
                .potential = cut_info.potential
            } );
            return;
        }

        const PI n1 = cut_info.local_dirac_index;
        const TR p1 = sorted_potentials( n1 );
        const Pt v1 = pt_data[ n1 ].position;
        const TR w1 = pt_data[ n1 ].weight;
        const PI i1 = pt_data[ n1 ].index;

        const Pt new_dir = v1 - v0;

        auto n = norm_2_p2( new_dir );
        auto s0 = dot( new_dir, v0 );
        auto s1 = dot( new_dir, v1 );

        auto new_dot = s0 + ( 1 + ( p0 - p1 ) / n ) / 2 * ( s1 - s0 );

        res.cut( new_dir, new_dot, {
            .global_dirac_index = i1,
            .local_dirac_index = n1,
            .dirac_position = v1,
            .dirac_weight = w1,
            .potential = p1
        } );
    } );

    return res;
}

// UTP std::ostream &operator<<( std::ostream &os, const DTP &p )

#undef UTP
#undef DTP

} // namespace sdot

// template<class AdditionalPtData,class TF,int dim,class Arch>
// std::ostream &operator<<( std::ostream &os, const sdot::Bsp<AdditionalPtData,TF,dim,Arch> &p ) {
//     p.display_rec( os, 0 );
//     return os;
// }
