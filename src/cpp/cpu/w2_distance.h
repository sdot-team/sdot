#pragma once

#include "../support/common_macros.h"
#include "../Affine1d.h"
#include "../DiracSet.h"
// #include <cstddef>

namespace sdot {

/// Wasserstein 2 distance
T_T void w2_distance( BatchOfDiracSet<const T> diracs, BatchOfAffine1d<const T> functions, TensorView<T,1> squared, TensorView<T,2> barycenters, TensorView<T,2> potentials, TensorView<T,3> cuts );
T_T void w2_distance( DiracSet<const T> diracs, Affine1d<const T> functions, TensorView<T,0> squared, TensorView<T,1> barycenters, TensorView<T,1> potentials, TensorView<T,2> cuts );

/// Gradients of Wasserstein 2 distance
T_T void w2_distance_backward( TensorView<const T,1> grad_squared, TensorView<const T,2> grad_barycenters, TensorView<const T,2> barycenters, TensorView<const T,2> potentials, TensorView<const T,3> cuts, BatchOfDiracSet<const T> diracs, BatchOfAffine1d<const T> functions, BatchOfDiracSet<T> grad_diracs, BatchOfAffine1d<T> grad_functions );
T_T void w2_distance_backward( TensorView<const T,0> grad_squared, TensorView<const T,1> grad_barycenters, TensorView<const T,1> barycenters, TensorView<const T,1> potentials, TensorView<const T,2> cuts, DiracSet<const T> diracs, Affine1d<const T> functions, DiracSet<T> grad_diracs, Affine1d<T> grad_functions );

} // namespace sdot

#include "w2_distance.cxx"
