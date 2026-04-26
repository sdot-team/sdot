#include "../cell/for_each_face.h"
#include "../support/P.h"
#include "PowerDiagram.h"

namespace sdot {

template<class TF,class Arch,int ct_dim>
void for_each_cell( const PowerDiagram<TF,Arch,ct_dim> &pd, auto &&func ) {
    Cell<TF,Arch> cell;
    func( cell );
}

void vtk_cell_faces( auto &&p ) {
    // for_each_cell( p.power_diagram, [&]( const auto &cell ) {
    //     for_each_face( cell, [&]( const auto &face_indices ) {

    //     } );
    // } );

    p.points( p.nb_points, 0 ) = 0; p.points( p.nb_points, 1 ) = 0; p.nb_points++;
    p.points( p.nb_points, 0 ) = 1; p.points( p.nb_points, 1 ) = 0; p.nb_points++;
    p.points( p.nb_points, 0 ) = 1; p.points( p.nb_points, 1 ) = 1; p.nb_points++;
    p.points( p.nb_points, 0 ) = 0; p.points( p.nb_points, 1 ) = 1; p.nb_points++;

    p.faces( p.nb_faces ) = 4; p.nb_faces++;
    p.faces( p.nb_faces ) = 0; p.nb_faces++;
    p.faces( p.nb_faces ) = 1; p.nb_faces++;
    p.faces( p.nb_faces ) = 2; p.nb_faces++;
    p.faces( p.nb_faces ) = 3; p.nb_faces++;

    P( p.cells.nb_vertices );
}

void vtk_cell_faces_backward( auto &&p ) {
    TODO;
}

} // namespace sdot
