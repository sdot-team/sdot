#pragma once

#include "SumOfWeightedDiracs1d.h"

namespace sdot {

//
template<class T,class Arch>
struct BatchOfSumOfWeightedDiracs1d {
    using   TF         = IntermediateScalarType<std::decay_t<T>,Arch>::type;
    using   TT         = TensorView<T,2,Arch>;

    HD PI   nb_diracs  () const;
    HD PI   nb_rows    () const;
    HD auto row        ( PI num_batch ) const -> SumOfWeightedDiracs1d<T,Arch>;

    void    arg_sorts  ( Tensor<PI,2,Arch> &sorted_is, Tensor<std::remove_const_t<T>,2,Arch> &sorted_xs ) const;
    auto    masses     () const -> Tensor<TF,1,Arch>;

    TT      positions; ///<
    TT      weights;   ///<
};

} // namespace sdot

#include "BatchOfSumOfWeightedDiracs1d.cxx"
