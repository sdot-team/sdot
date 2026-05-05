#include "../Distribution/PolynomialGridWorker.h"
#include "PowerDiagramWorker.h"
// #include "../support/P.h"

namespace sdot {

// template<int ct_dim,typename Arch,typename TF,typename TI>
// void display_cell( const Cell<ct_dim,Arch,TF,TI> &cell ) {
//     for( PI i = 0; i < cell.nb_vertices; ++i )
//         P( cell.vertex_positions.row( i ) );
// }

void vtk_cell_faces( auto &&p ) {
    PowerDiagramWorker pw( p.power_diagram, p.cell_workspace, p.cells );
    with_worker_for( p.target_distribution, [&]( auto &&distribution ) {
        pw.for_each_cell( distribution, [&]( auto &cell_worker, PI /* batch_index */, PI /* point_index */ ) {
            cell_worker.for_each_face( [&]( const auto &face_indices, const auto &face_cuts ) {
                // we do not display pure infinite faces
                if ( face_cuts.size() ) {
                    bool no_disp = true;
                    for( PI num_cut : face_cuts )
                        if ( cell_worker.cut_id( num_cut ) != CellBoundary::INFINITE )
                            no_disp = false;
                    if ( no_disp )
                        return;
                }

                // add vertices
                PI np = p.nb_points.post_increment( face_indices.size() );
                for( PI i = 0; i < face_indices.size(); ++i )
                    p.points.row( np + i ).get_data_from( cell_worker.vertex_position( face_indices[ i ] ) );

                // add connectivity
                PI nf = p.nb_faces.post_increment( 1 + face_indices.size() );
                p.faces( nf++ ) = face_indices.size();
                for( PI i = 0; i < face_indices.size(); ++i )
                    p.faces( nf++ ) = np + i;
            } );

            return 0;
        } );
    } );
}

void vtk_cell_faces_backward( auto &&p ) {
    TODO;
}

} // namespace sdot
