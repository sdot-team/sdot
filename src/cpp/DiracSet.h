#pragma once

#include "support/IntermediateScalarType.h"
#include "support/Tensor.h"
#include <vector>

namespace sdot {

//
template<class T,class Arch>
struct DiracSet {
    using TF        = IntermediateScalarType<std::decay_t<T>,Arch>::type;
    using TT        = TensorView<T,1,Arch>;

    PI    nb_diracs () const;
    auto  arg_sort  () -> std::vector<PI> const;
    TF    mass      () const;

    TT    xs;       ///< positions
    TT    ws;       ///< weights
};

//
template<class T,class Arch>
struct BatchOfDiracSet {
    using TF        = IntermediateScalarType<std::decay_t<T>,Arch>::type;
    using TT        = TensorView<T,2,Arch>;

    PI    nb_rows   () const;
    auto  row       ( PI num_batch ) const -> DiracSet<T,Arch>;

    auto  masses    () const -> Tensor<TF,1,Arch>;

    TT    xs;       ///< positions
    TT    ws;       ///< weights
};

} // namespace sdot

#include "DiracSet.cxx"
