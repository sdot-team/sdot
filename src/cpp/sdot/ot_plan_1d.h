#pragma once

#include "support/TensorView.h"

namespace sdot {

/// Wasserstein 2 distance
T_T void ot_plan_1d_forward( TensorView<const T,2,Cpu> dirac_xs, TensorView<const T,1,Cpu> dirac_ws, auto &&primitive, TensorView<T,0,Cpu> distance, TensorView<T,2,Cpu> barycenters, TensorView<T,1,Cpu> potentials, TensorView<T,2,Cpu> cuts );
T_T void ot_plan_1d_forward( TensorView<const T,3,Cpu> dirac_xs, TensorView<const T,2,Cpu> dirac_ws, auto &&primitive, TensorView<T,1,Cpu> distance, TensorView<T,3,Cpu> barycenters, TensorView<T,2,Cpu> potentials, TensorView<T,3,Cpu> cuts );

/// Gradients of Wasserstein 2 distance
T_T void ot_plan_1d_backward( TensorView<const T,2,Cpu> dirac_xs, TensorView<const T,1,Cpu> dirac_ws, auto &&primitive, TensorView<const T,0,Cpu> distance, TensorView<const T,2,Cpu> barycenters, TensorView<const T,1,Cpu> potentials, TensorView<const T,2,Cpu> cuts, TensorView<const T,0,Cpu> grad_distance, TensorView<const T,2,Cpu> grad_barycenters, TensorView<const T,1,Cpu> grad_potentials, TensorView<const T,2,Cpu> grad_cuts, TensorView<T,2,Cpu> grad_dirac_xs, TensorView<T,1,Cpu> grad_dirac_ws, TensorView<T,1,Cpu> grad_g_values, TensorView<T,2,Cpu> grad_g_bounds, TensorView<T,1,Cpu> grad_g_knots );
T_T void ot_plan_1d_backward( TensorView<const T,3,Cpu> dirac_xs, TensorView<const T,2,Cpu> dirac_ws, auto &&primitive, TensorView<const T,1,Cpu> distance, TensorView<const T,3,Cpu> barycenters, TensorView<const T,2,Cpu> potentials, TensorView<const T,3,Cpu> cuts, TensorView<const T,1,Cpu> grad_distance, TensorView<const T,3,Cpu> grad_barycenters, TensorView<const T,2,Cpu> grad_potentials, TensorView<const T,3,Cpu> grad_cuts, TensorView<T,3,Cpu> grad_dirac_xs, TensorView<T,2,Cpu> grad_dirac_ws, TensorView<T,2,Cpu> grad_g_values, TensorView<T,3,Cpu> grad_g_bounds, TensorView<T,2,Cpu> grad_g_knots );


// #ifdef __CUDACC__
// T_T void ot_plan_1d( BatchOfSumOfWeightedDiracs1d<const T,Cuda> diracs, BatchOfAffine1d<const T,Cuda> functions, TensorView<T,1,Cuda> squared, TensorView<T,2,Cuda> barycenters, TensorView<T,2,Cuda> potentials, TensorView<T,3,Cuda> cuts );
// T_T void ot_plan_1d_backward( TensorView<const T,1,Cuda> grad_squared, TensorView<const T,2,Cuda> grad_barycenters, TensorView<const T,2,Cuda> barycenters, TensorView<const T,2,Cuda> potentials, TensorView<const T,3,Cuda> cuts, BatchOfSumOfWeightedDiracs1d<const T,Cuda> diracs, BatchOfAffine1d<const T,Cuda> functions, BatchOfSumOfWeightedDiracs1d<T,Cuda> grad_diracs, BatchOfAffine1d<T,Cuda> grad_functions );
// #endif

} // namespace sdot

#include "ot_plan_1d.cxx"
