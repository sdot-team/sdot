#pragma once

#include "../support/common_macros.h"
#include "../Affine1DFunction.h"
#include "../DiracSet.h"
// #include <cstddef>

namespace sdot {

/// Wasserstein 2 distance
T_T void w2_distance( DiracSet<const T,0> diracs, Affine1DFunction<const T,0> functions, TensorView<T,0> w2_squared, TensorView<T,1> w2_barycenters );
T_T void w2_distance( DiracSet<const T,1> diracs, Affine1DFunction<const T,1> functions, TensorView<T,1> w2_squared, TensorView<T,2> w2_barycenters );

/// Gradients of Wasserstein 2 distance
T_T void w2_distance_backward( TensorView<const T,1> grad_w2_squared, TensorView<const T,2> grad_w2_barycenters, TensorView<const T,2> w2_barycenters, DiracSet<const T,1> diracs, Affine1DFunction<const T,1> functions, DiracSet<T,1> grad_diracs, Affine1DFunction<T,1> grad_functions );
T_T void w2_distance_backward( TensorView<const T,0> grad_w2_squared, TensorView<const T,1> grad_w2_barycenters, TensorView<const T,1> w2_barycenters, DiracSet<const T,0> diracs, Affine1DFunction<const T,0> functions, DiracSet<T,0> grad_diracs, Affine1DFunction<T,0> grad_functions );

} // namespace sdot

#include "w2_distance.cxx"
