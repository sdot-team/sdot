#pragma once

#include "support/vector_map.h"
#include "SplineGrid.h"

namespace sdot {

template<class TF,int order>
struct BatchOfSplineGrids {
    using  Values = TensorView<const TF,2,Cpu>;
    using  Bounds = TensorView<const TF,3,Cpu>;
    using  Knots  = std::vector<TensorView<const TF,2,Cpu>>;

    auto   row    ( PI i ) const { return SplineGrid<TF,order>( values.row( i ), bounds.row( i ), vector_map( knots, [i]( const auto &knot ) { return knot.row( i ); } ) ); }

    Values values;
    Bounds bounds;
    Knots  knots;
};

template<int order,class T> auto batch_of_spline_grids( CtInt<order>, TensorView<const T,2,Cpu> values, TensorView<const T,3,Cpu> bounds, const std::vector<TensorView<const T,2,Cpu>> &knots ) {
    return BatchOfSplineGrids<T,order>{ values, bounds, knots };
}

} // namespace sdot
