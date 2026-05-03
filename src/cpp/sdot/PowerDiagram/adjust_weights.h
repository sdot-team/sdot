// #include <sdot/generated_includes/PolynomialGrid.h>
// #include "../Cell/for_each_face.h"
#include "../Cell/measure.h"
#include "for_each_cell.h"
#include "../support/P.h"

namespace sdot {

void get_matrix_terms( auto &p, auto &&distribution ) {
    double tot = 0;
    for_each_cell( p.power_diagram, p.cells, p.cell_workspace, distribution, [&]( const auto &cell, PI batch_index ) {
        tot += measure( cell, p.cell_workspace.row( batch_index ), distribution );
        // for_each_face( cell, p.cell_workspace.row( batch_index ), [&]( const auto &face_indices, const auto &face_cuts ) {
        //     // we do not display pure infinite faces
        //     if ( face_cuts.size() ) {
        //         bool no_disp = true;
        //         for( PI num_cut : face_cuts )
        //             if ( cell.cut_ids( num_cut ) != CellBoundary::INFINITE )
        //                 no_disp = false;
        //         if ( no_disp )
        //             return;
        //     }

        //     // add vertices
        //     PI np = p.nb_points.post_increment( face_indices.size() );
        //     for( PI i = 0; i < face_indices.size(); ++i ) {
        //         for( PI d = 0; d < p.power_diagram.dim(); ++d )
        //             p.points( np + i, d ) = cell.vertex_positions( face_indices[ i ], d );
        //     }

        //     // add connectivity
        //     PI nf = p.nb_faces.post_increment( 1 + face_indices.size() );
        //     p.faces( nf++ ) = face_indices.size();
        //     for( PI i = 0; i < face_indices.size(); ++i )
        //         p.faces( nf++ ) = np + i;
        // } );
    } );

    P( tot );
}

void adjust_weights( auto &&p ) {
    with_worker_for( p.target_distribution, [&]( auto &&distribution ) {
        get_matrix_terms( p, distribution );
    } );
}

void adjust_weights_backward( auto &&p ) {
    TODO;
}

} // namespace sdot
