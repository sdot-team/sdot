#include "../Cell/for_each_face.h"
#include "for_each_cell.h"
// #include "../support/P.h"

namespace sdot {

// template<int ct_dim,typename Arch,typename TF,typename TI>
// void display_cell( const Cell<ct_dim,Arch,TF,TI> &cell ) {
//     for( PI i = 0; i < cell.nb_vertices; ++i )
//         P( cell.vertex_positions.row( i ) );
// }

void vtk_cell_faces( auto &&p ) {
    for_each_cell( p.power_diagram, p.cells, p.cut_workspace, [&]( const auto &cell, PI batch_index ) {
        for_each_face( cell, p.cut_workspace.row( batch_index ), [&]( const auto &face_indices ) {
            PI np = p.nb_points.post_increment( face_indices.size() );
            for( PI i = 0; i < face_indices.size(); ++i ) {
                for( PI d = 0; d < p.power_diagram.dim(); ++d )
                    p.points( np + i, d ) = cell.vertex_positions( face_indices[ i ], d );
            }

            PI nf = p.nb_faces.post_increment( 1 + face_indices.size() );
            p.faces( nf++ ) = face_indices.size();
            for( PI i = 0; i < face_indices.size(); ++i )
                p.faces( nf++ ) = np + i;
        } );
    } );
}

void vtk_cell_faces_backward( auto &&p ) {
    TODO;
}

} // namespace sdot
