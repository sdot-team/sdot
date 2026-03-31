#pragma once

#include "../BatchOfPiecewiseAffineFunction1d.h"
#include "../BatchOfSumOfWeightedDiracs1d.h"
// #include <cstddef>

namespace sdot {

/// Wasserstein 2 distance
T_T void w2_distance( BatchOfSumOfWeightedDiracs1d<const T,Cpu > diracs, BatchOfPiecewiseAffineFunction1d<const T,Cpu > functions, TensorView<T,1,Cpu > squared, TensorView<T,2,Cpu > barycenters, TensorView<T,2,Cpu > potentials, TensorView<T,3,Cpu > cuts );
T_T void w2_distance( SumOfWeightedDiracs1d<const T,Cpu> diracs, PiecewiseAffineFunction1d<const T,Cpu> functions, TensorView<T,0,Cpu> squared, TensorView<T,1,Cpu> barycenters, TensorView<T,1,Cpu> potentials, TensorView<T,2,Cpu> cuts );

/// Gradients of Wasserstein 2 distance
T_T void w2_distance_backward( TensorView<const T,1,Cpu > grad_squared, TensorView<const T,2,Cpu > grad_barycenters, TensorView<const T,2,Cpu > barycenters, TensorView<const T,2,Cpu > potentials, TensorView<const T,3,Cpu > cuts, BatchOfSumOfWeightedDiracs1d<const T,Cpu > diracs, BatchOfPiecewiseAffineFunction1d<const T,Cpu > functions, BatchOfSumOfWeightedDiracs1d<T,Cpu > grad_diracs, BatchOfPiecewiseAffineFunction1d<T,Cpu > grad_functions );
T_T void w2_distance_backward( TensorView<const T,0,Cpu> grad_squared, TensorView<const T,1,Cpu> grad_barycenters, TensorView<const T,1,Cpu> barycenters, TensorView<const T,1,Cpu> potentials, TensorView<const T,2,Cpu> cuts, SumOfWeightedDiracs1d<const T,Cpu> diracs, PiecewiseAffineFunction1d<const T,Cpu> functions, SumOfWeightedDiracs1d<T,Cpu> grad_diracs, PiecewiseAffineFunction1d<T,Cpu> grad_functions );

#ifdef __CUDACC__
T_T void w2_distance( BatchOfSumOfWeightedDiracs1d<const T,Cuda> diracs, BatchOfAffine1d<const T,Cuda> functions, TensorView<T,1,Cuda> squared, TensorView<T,2,Cuda> barycenters, TensorView<T,2,Cuda> potentials, TensorView<T,3,Cuda> cuts );
T_T void w2_distance_backward( TensorView<const T,1,Cuda> grad_squared, TensorView<const T,2,Cuda> grad_barycenters, TensorView<const T,2,Cuda> barycenters, TensorView<const T,2,Cuda> potentials, TensorView<const T,3,Cuda> cuts, BatchOfSumOfWeightedDiracs1d<const T,Cuda> diracs, BatchOfAffine1d<const T,Cuda> functions, BatchOfSumOfWeightedDiracs1d<T,Cuda> grad_diracs, BatchOfAffine1d<T,Cuda> grad_functions );
#endif

} // namespace sdot

#include "w2_distance.cxx"
