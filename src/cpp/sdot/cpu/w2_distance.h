#pragma once

#include "../support/common_macros.h"
#include "../Affine1d.h"
#include "../DiracSet.h"
// #include <cstddef>

namespace sdot {

/// Wasserstein 2 distance
T_T void w2_distance( BatchOfDiracSet<const T,Cpu > diracs, BatchOfAffine1d<const T,Cpu > functions, TensorView<T,1,Cpu > squared, TensorView<T,2,Cpu > barycenters, TensorView<T,2,Cpu > potentials, TensorView<T,3,Cpu > cuts );
T_T void w2_distance( DiracSet<const T,Cpu> diracs, Affine1d<const T,Cpu> functions, TensorView<T,0,Cpu> squared, TensorView<T,1,Cpu> barycenters, TensorView<T,1,Cpu> potentials, TensorView<T,2,Cpu> cuts );

/// Gradients of Wasserstein 2 distance
T_T void w2_distance_backward( TensorView<const T,1,Cpu > grad_squared, TensorView<const T,2,Cpu > grad_barycenters, TensorView<const T,2,Cpu > barycenters, TensorView<const T,2,Cpu > potentials, TensorView<const T,3,Cpu > cuts, BatchOfDiracSet<const T,Cpu > diracs, BatchOfAffine1d<const T,Cpu > functions, BatchOfDiracSet<T,Cpu > grad_diracs, BatchOfAffine1d<T,Cpu > grad_functions );
T_T void w2_distance_backward( TensorView<const T,0,Cpu> grad_squared, TensorView<const T,1,Cpu> grad_barycenters, TensorView<const T,1,Cpu> barycenters, TensorView<const T,1,Cpu> potentials, TensorView<const T,2,Cpu> cuts, DiracSet<const T,Cpu> diracs, Affine1d<const T,Cpu> functions, DiracSet<T,Cpu> grad_diracs, Affine1d<T,Cpu> grad_functions );

#ifdef __CUDACC__
T_T void w2_distance( BatchOfDiracSet<const T,Cuda> diracs, BatchOfAffine1d<const T,Cuda> functions, TensorView<T,1,Cuda> squared, TensorView<T,2,Cuda> barycenters, TensorView<T,2,Cuda> potentials, TensorView<T,3,Cuda> cuts );
T_T void w2_distance_backward( TensorView<const T,1,Cuda> grad_squared, TensorView<const T,2,Cuda> grad_barycenters, TensorView<const T,2,Cuda> barycenters, TensorView<const T,2,Cuda> potentials, TensorView<const T,3,Cuda> cuts, BatchOfDiracSet<const T,Cuda> diracs, BatchOfAffine1d<const T,Cuda> functions, BatchOfDiracSet<T,Cuda> grad_diracs, BatchOfAffine1d<T,Cuda> grad_functions );
#endif

} // namespace sdot

#include "w2_distance.cxx"
