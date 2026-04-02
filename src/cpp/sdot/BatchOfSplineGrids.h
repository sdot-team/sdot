#pragma once

#include "support/IndexTensor.h"
#include "SplineGrid.h"

namespace sdot {

template<class TF,int order,class Knots>
struct BatchOfSplineGrids {
    using  Values = TensorView<const TF,2,Cpu>;

    auto   row    ( PI i ) const { return SplineGrid<TF,order,DECAYED_TYPE_OF( knots.row( i ) )>( values.row( i ), knots.row( i ) ); }

    Values values;
    Knots  knots;
};

template<int order,class T,class K> auto batch_of_spline_grids( CtInt<order>, TensorView<const T,2,Cpu> values, K &&knots ) { return BatchOfSplineGrids<T,order,std::decay_t<K>>{ values, FORWARD( knots ) }; }
template<int order,class T> auto batch_of_spline_grids( CtInt<order>, TensorView<const T,2,Cpu> values ) { return BatchOfSplineGrids<T,order,IndexTensor<T,2,1>>{ values, { T( 1 ) / ( values.size( 1 ) - 1 ) } }; }

} // namespace sdot
