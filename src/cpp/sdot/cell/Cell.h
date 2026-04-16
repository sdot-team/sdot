#pragma once

#include "../support/TensorView.h"

namespace sdot {

///
template<class TF,int ct_dim,class Arch>
struct Cell {
    static constexpr SI   INFINITE          = -2;
    static constexpr SI   BOUNDARY          = -1;

    PI                    dim               () const { return vertex_positions.size( 1 ); }

    TensorView<TF,2,Arch> vertex_positions; ///< ( vertex_capacity, dim )
    TensorView<SI,2,Arch> vertex_inds;      ///< ( vertex_capacity, dim ) : sorted cut indices for each vertex
    TensorView<SI,2,Arch> edge_links;       ///< ( vertex_capacity, 2 )
    TensorView<TF,2,Arch> cut_planes;       ///< ( cut_capacity, dim + 1 )
    TensorView<SI,1,Arch> cut_ids;          ///< ( cut_capacity )

    TensorView<SI,0,Arch> is_fully_closed;
    TensorView<SI,0,Arch> nb_vertices;
    TensorView<SI,0,Arch> nb_cuts;
};

#define UTP2 template<class TF,class Arch>
#define DTP2 Cell<TF,2,Arch>

#define UTP1 template<class TF,class Arch>
#define DTP1 Cell<TF,2,Arch>

#define UTP template<class TF,int ct_dim,class Arch>
#define DTP Cell<TF,ct_dim,Arch>

UTP void make_aligned_simplex( DTP &cell, SI cut_id ) {
    const PI dim = cell.vertex_positions.size( 1 );
    const PI nb_vertices = dim + 1;
    const PI nb_cuts = dim + 1;

    cell.is_fully_closed() = cut_id != DTP::INFINITE;
    cell.nb_vertices() = nb_vertices;
    cell.nb_cuts() = nb_cuts;

    // vertex_positions
    for( PI num_vertex = 0; num_vertex < dim + 1; ++num_vertex )
        for( PI d = 0; d < dim; ++d )
            cell.vertex_positions( num_vertex, d ) = ( d + 1 == num_vertex );

    // vertex_inds
    for( PI num_vertex = 0; num_vertex < dim + 1; ++num_vertex )
        for( PI d = 0; d < dim; ++d )
            cell.vertex_inds( num_vertex, d ) = d + ( d >= num_vertex );

    // edge_links
    for( PI num_vertex = 0; num_vertex < dim + 1; ++num_vertex ) {
        cell.edge_links( num_vertex, 0 ) = ( num_vertex + nb_vertices - 1 ) % nb_vertices;
        cell.edge_links( num_vertex, 1 ) = ( num_vertex + 1 ) % nb_vertices;
    }

    // cut_planes
    for( PI num_cut = 0; num_cut < dim; ++num_cut ) {
        for( PI d = 0; d < dim; ++d )
            cell.cut_planes( num_cut, d ) = - ( d == num_cut );
        cell.cut_planes( num_cut, dim ) = 0;
    }
    for( PI d = 0; d < dim + 1; ++d )
        cell.cut_planes( dim, d ) = 1;

    // cut_ids
    for( PI num_cut = 0; num_cut < dim; ++num_cut )
        cell.cut_ids( num_cut ) = cut_id;
}

UTP void make_empty_cell( DTP &cell ) {
    make_aligned_simplex( cell, DTP::INFINITE );
}

#undef UTP2
#undef DTP2
#undef UTP1
#undef DTP1
#undef UTP
#undef DTP

} // namespace sdot
