#pragma once

#include "support/IntermediateScalarType.h"
#include "support/TensorView.h"
#include <vector>

namespace sdot {

//
template<class T>
struct DiracSet {
    using TF        = IntermediateScalarType<std::decay_t<T>>::type;
    using TT        = TensorView<T,1>;

    PI    nb_diracs () const;
    auto  arg_sort  () -> std::vector<PI> const;
    TF    mass      () const;

    TT    xs;       ///< positions
    TT    ws;       ///< weights
};

//
template<class T>
struct BatchOfDiracSet {
    using TT        = TensorView<T,2>;

    PI    nb_rows   () const;
    auto  row       ( PI num_batch ) const -> DiracSet<T>;

    TT    xs;       ///< positions
    TT    ws;       ///< weights
};

} // namespace sdot

#include "DiracSet.cxx"
