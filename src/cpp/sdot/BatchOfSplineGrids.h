#pragma once

#include "support/vector_map.h"
#include "SplineGrid.h"

namespace sdot {

template<class TF,int ct_dim,int order,class Arch>
struct BatchOfSplineGrids {
    using  Values = TensorView<const TF,ct_dim+1,Arch>;
    using  Bounds = TensorView<const TF,3,Arch>;
    using  Knots  = std::vector<TensorView<const TF,2,Arch>>;

    auto   row    ( PI i ) const { return SplineGrid<TF,ct_dim,order,Arch>( values.row( i ), bounds.row( i ), vector_map( knots, [i]( const auto &knot ) { return knot.row( i ); } ) ); }

    Values values;
    Bounds bounds;
    Knots  knots;
};

template<int order,int ct_dim_plus_1,class T,class Arch> auto batch_of_spline_grids( CtInt<order>, TensorView<const T,ct_dim_plus_1,Arch> values, TensorView<const T,3,Arch> bounds, const std::vector<TensorView<const T,2,Arch>> &knots ) {
    return BatchOfSplineGrids<T,ct_dim_plus_1-1,order,Arch>{ values, bounds, knots };
}

} // namespace sdot
