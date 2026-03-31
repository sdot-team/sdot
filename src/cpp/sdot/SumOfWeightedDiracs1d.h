#pragma once

#include "support/IntermediateScalarType.h"
#include "support/Tensor.h"
#include <vector>

namespace sdot {

//
template<class T,class Arch>
struct SumOfWeightedDiracs1d {
    using TF         = IntermediateScalarType<std::decay_t<T>,Arch>::type;
    using TT         = TensorView<T,1,Arch>;

    PI    nb_diracs  () const;
    auto  arg_sort   () -> std::vector<PI> const;
    TF    mass       () const;

    TT    positions; ///<
    TT    weights;   ///< 
};

} // namespace sdot

#include "SumOfWeightedDiracs1d.cxx"
