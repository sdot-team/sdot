#pragma once

#include "PiecewiseAffineGrid1d.h"

namespace sdot {

template<class T,class Arch>
struct BatchOfPiecewiseAffineGrid1d {
    using   TF       = IntermediateScalarType<std::decay_t<T>,Arch>::type;
    using   TT       = TensorView<T,2,Arch>;

    HD PI   nb_points() const;
    HD PI   nb_rows  () const;
    HD auto row      ( PI num_batch ) const -> PiecewiseAffineGrid1d<T,Arch>;

    auto    masses   () const -> Tensor<TF,1,Arch>;

    TT      xs;      ///<
    TT      ys;      ///<
};

} // namespace sdot

#include "BatchOfPiecewiseAffineGrid1d.cxx"
