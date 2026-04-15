#pragma once

#include "support/TensorView.h"

namespace sdot {

template<class TF,int ct_dim,class Arch>
struct Cell {
    TensorView<TF,2,Arch> vertex_positions;
    TensorView<TF,2,Arch> cut_planes;
    TensorView<SI,1,Arch> cut_ids;

    TensorView<SI,0,Arch> nb_vertices;
    TensorView<SI,0,Arch> nb_cuts;
};

template<class TF>
auto make_empty_cell( Cell<TF,2,Cpu> &cell ) {
    const PI dim = cell.vertex_positions.size( 1 );
    cell.nb_vertices() = dim + 1;
    cell.nb_cuts() = dim + 1;

    cell.vertex_positions( 0, 0 ) = 17;
}

} // namespace sdot
