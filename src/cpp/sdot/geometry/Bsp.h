#pragma once

#include "../support/TensorView.h"

namespace sdot {

template<class TF,class Arch,int degree_w_approx=1>
struct Bsp {
    TensorView<SI,1,Arch> sorted_vertex_indices;
    TensorView<SI,2,Arch> cell_indices;
    TensorView<TF,2,Arch> cell_bounds;
    TensorView<SI,0,Arch> nb_cells;
};

} // namespace sdot
