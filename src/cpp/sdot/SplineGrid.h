#pragma once

#include "support/TensorView.h"

namespace sdot {

///
template<class TF,int ct_dim,int order,class Arch>
struct SplineGrid;

//
template<int order,class T,int ct_dim,class Arch> auto spline_grid( CtInt<order>, TensorView<const T,ct_dim,Arch> values, TensorView<const T,2,Arch> bounds, const std::vector<TensorView<const T,1,Arch>> &knots ) {
    return SplineGrid<T,ct_dim,order,Arch>{ values, bounds, knots };
}

} // namespace sdot

#include "SplineGrid_1d_o1.h" // IWYU pragma: export
#include "SplineGrid_nd_o0.h" // IWYU pragma: export
