#pragma once

#include "support/TensorView.h"
#include "geometry/Cell.h"
#include "SplineGrid.h"

namespace sdot {

// 1d, order 1 (piecewise affine)
template<class TF,int ct_dim,class Arch>
struct SplineGrid<TF,ct_dim,0,Arch> {
    using  Values        = TensorView<const TF,ct_dim,Arch>;
    using  Bounds        = TensorView<const TF,2,Arch>;
    using  Knots         = TensorView<const TF,1,Arch>;
    using  Cell          = sdot::Cell<TF,ct_dim,Arch>;

    /**/   SplineGrid    ( Values values, Bounds bounds, const std::vector<Knots> &knots );

    Cell   base_cell     ( PI dim, typename Cell::CellInfo cell_info = {}, typename Cell::CutInfo cut_info = {} ) const;

    TF     coeff_values;
    Values values;
    Bounds bounds;
};

} // namespace sdot

#include "SplineGrid_nd_o0.cxx"
