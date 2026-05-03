#include <sdot/generated_includes/Cell.h>
#include "CellBoundary.h"

#define UTP template<int ct_dim,class Arch,class TF,class TI>
#define DTP Cell<ct_dim,Arch,TF,TI>

namespace sdot {

UTP void make_aligned_simplex( DTP &cell, SI cut_id, CtInt<ct_dim> ) {
    const PI dim = cell.vertex_positions.size( 1 );
    const PI nb_edges = ( dim + 1 ) * dim / 2;
    const PI nb_vertices = dim + 1;
    const PI nb_cuts = dim + 1;

    cell.is_fully_closed() = cut_id != CellBoundary::INFINITE;
    cell.nb_vertices = nb_vertices;
    cell.nb_edges = nb_edges;
    cell.nb_cuts = nb_cuts;

    // vertex_positions
    for( PI num_vertex = 0; num_vertex < nb_vertices; ++num_vertex )
        for( PI d = 0; d < dim; ++d )
            cell.vertex_positions( num_vertex, d ) = ( d + 1 == num_vertex );

    // vertex_inds
    if ( dim != 2 )
        for( PI num_vertex = 0; num_vertex < nb_vertices; ++num_vertex )
            for( PI d = 0; d < dim; ++d )
                cell.vertex_indices( num_vertex, d ) = d + ( d >= num_vertex );

    // edge_indices
    if ( dim != 2 ) {
        for ( PI a = 0, o = 0; a < nb_vertices; ++a ) {
            for ( PI b = a + 1; b < nb_vertices; ++b ) {
                if ( a != b ) {
                    cell.edge_indices( o, 0 ) = a;
                    cell.edge_indices( o, 1 ) = b;
                    for( PI d = 0; d < dim - 1; ++d )
                        cell.edge_indices( o, 2 + d ) = d + ( d >= a ) + ( d >= b - 1 );
                    ++o;
                }
            }
        }
    }

    // cut_planes
    if ( dim != 2 ) {
        for( PI num_cut = 0; num_cut < dim; ++num_cut ) {
            for( PI d = 0; d < dim; ++d )
                cell.cut_planes( num_cut, d ) = - ( d == num_cut );
            cell.cut_planes( num_cut, dim ) = 0;
        }
        for( PI d = 0; d < dim + 1; ++d )
            cell.cut_planes( dim, d ) = 1;
    } else {
        cell.cut_planes( 0, 0 ) =  0; cell.cut_planes( 0, 1 ) = -1; cell.cut_planes( 0, 2 ) = 0;
        cell.cut_planes( 1, 0 ) = +1; cell.cut_planes( 1, 1 ) = +1; cell.cut_planes( 1, 2 ) = 1;
        cell.cut_planes( 2, 0 ) = -1; cell.cut_planes( 2, 1 ) =  0; cell.cut_planes( 2, 2 ) = 0;
    }

    // cut_ids
    for( PI num_cut = 0; num_cut < nb_cuts; ++num_cut )
        cell.cut_ids( num_cut ) = cut_id;
}

void make_empty_cell( auto &&p ) {
    make_aligned_simplex( p.cell, p.cell.INFINITE, p.ct_dim );
}

void make_empty_cell_backward( auto && ) {
    TODO; // make_aligned_simplex( p.cell, p.cell.INFINITE );
}

#undef UTP
#undef DTP

} // namespace sdot

