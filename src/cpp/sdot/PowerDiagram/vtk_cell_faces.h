#include <sdot/generated_includes/PowerDiagram.h>
#include <sdot/generated_includes/BatchOfCell.h>
#include "../Cell/make_aligned_simplex.h"
#include "../Cell/for_each_face.h"
#include "../support/P.h"

namespace sdot {

template<int ct_dim,typename Arch,typename TF,typename TI>
void for_each_cell( const PowerDiagram<ct_dim,ct_dim,ct_dim,Arch,TF,TI> &pd, BatchOfCell<ct_dim,Arch,TF,TI> &cells, auto &&func ) {
    Cell<ct_dim,Arch,TF,TI> cell = cells.row( 0 );
    make_aligned_simplex( cell, CellBoundary::INFINITE, CtInt<ct_dim>() );
    func( cell );
}

void vtk_cell_faces( auto &&p ) {
    for_each_cell( p.power_diagram, p.cells, [&]( const auto &cell ) {
        for_each_face( cell, [&]( const auto &face_indices ) {
            P( p.nb_points.capacity );

            PI np = p.nb_points.post_increment( face_indices.size() );
            for( PI i = 0; i < face_indices.size(); ++i ) {
                for( PI d = 0; d < p.power_diagram.dim(); ++d )
                    p.points( np + i, d ) = cell.vertex_positions( face_indices[ i ], d );
            }

            P( p.nb_faces.capacity );

            PI nf = p.nb_faces.post_increment( 1 + face_indices.size() );
            p.faces( nf++ ) = face_indices.size();
            for( PI i = 0; i < face_indices.size(); ++i )
                p.faces( nf++ ) = face_indices[ i ];
        } );
    } );
}

void vtk_cell_faces_backward( auto &&p ) {
    TODO;
}

} // namespace sdot
