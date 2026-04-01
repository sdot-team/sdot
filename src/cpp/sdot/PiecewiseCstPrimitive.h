#pragma once

#include "support/TensorView.h"

namespace sdot {

template<class TF>
struct PiecewiseCstPrimitive {

    TensorView<const TF,1,Cpu> values;
};

template<class TF>
struct BatchOfPiecewiseCstPrimitive {
    auto row( PI i ) const { return PiecewiseCstPrimitive<TF>( values.row( i) ); }

    TensorView<const TF,2,Cpu> values;
};

T_T auto piecewise_cst_primitive( TensorView<const T,2,Cpu> values ) { return BatchOfPiecewiseCstPrimitive<T>{ values }; }

} // namespace sdot

#include "PiecewiseCstPrimitive.cxx"
