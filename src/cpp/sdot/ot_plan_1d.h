#pragma once

#include "support/TensorView.h"

namespace sdot {

/// Wasserstein 2 distance
T_T void ot_plan_1d_forward( TensorView<const T,2,Cpu> dirac_xs, TensorView<const T,1,Cpu> dirac_ws, auto &&primitive, TensorView<T,0,Cpu> distance, TensorView<T,2,Cpu> barycenters, TensorView<T,1,Cpu> potentials, TensorView<T,2,Cpu> cuts );
T_T void ot_plan_1d_forward( TensorView<const T,3,Cpu> dirac_xs, TensorView<const T,2,Cpu> dirac_ws, auto &&primitive, TensorView<T,1,Cpu> distance, TensorView<T,3,Cpu> barycenters, TensorView<T,2,Cpu> potentials, TensorView<T,3,Cpu> cuts );

/// Gradients of Wasserstein 2 distance
// T_T void ot_plan_1d_backward( TensorView<const T,1,Cpu> grad_squared, TensorView<const T,2,Cpu> grad_barycenters, TensorView<const T,2,Cpu> barycenters, TensorView<const T,2,Cpu> potentials, TensorView<const T,3,Cpu> cuts, BatchOfSumOfWeightedDiracs1d<const T,Cpu> diracs, BatchOfPiecewiseAffineGrid1d<const T,Cpu> functions, BatchOfSumOfWeightedDiracs1d<T,Cpu> grad_diracs, BatchOfPiecewiseAffineGrid1d<T,Cpu> grad_functions );
// T_T void ot_plan_1d_backward( TensorView<const T,0,Cpu> grad_squared, TensorView<const T,1,Cpu> grad_barycenters, TensorView<const T,1,Cpu> barycenters, TensorView<const T,1,Cpu> potentials, TensorView<const T,2,Cpu> cuts, SumOfWeightedDiracs1d<const T,Cpu> diracs, PiecewiseAffineGrid1d<const T,Cpu> functions, SumOfWeightedDiracs1d<T,Cpu> grad_diracs, PiecewiseAffineGrid1d<T,Cpu> grad_functions );

// #ifdef __CUDACC__
// T_T void ot_plan_1d( BatchOfSumOfWeightedDiracs1d<const T,Cuda> diracs, BatchOfAffine1d<const T,Cuda> functions, TensorView<T,1,Cuda> squared, TensorView<T,2,Cuda> barycenters, TensorView<T,2,Cuda> potentials, TensorView<T,3,Cuda> cuts );
// T_T void ot_plan_1d_backward( TensorView<const T,1,Cuda> grad_squared, TensorView<const T,2,Cuda> grad_barycenters, TensorView<const T,2,Cuda> barycenters, TensorView<const T,2,Cuda> potentials, TensorView<const T,3,Cuda> cuts, BatchOfSumOfWeightedDiracs1d<const T,Cuda> diracs, BatchOfAffine1d<const T,Cuda> functions, BatchOfSumOfWeightedDiracs1d<T,Cuda> grad_diracs, BatchOfAffine1d<T,Cuda> grad_functions );
// #endif

} // namespace sdot

#include "ot_plan_1d.cxx"
