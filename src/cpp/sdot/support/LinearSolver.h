#pragma once

#include "common_types.h"
#include "Arch.h"

namespace sdot {

/** Dispatched linear solver for symmetric positive-definite systems.
    Input: COO triplets (may store extra-diagonal terms twice).
    CPU:  AMGCL AMG-preconditioned CG  (define SDOT_USE_AMGCL, default)
          or Eigen SimplicialLDLT      (fallback without AMGCL)
    CUDA: TODO — amgcl::backend::cuda or cuSPARSE. */
template<class Arch, class TF, class TI>
struct LinearSolver;

template<class TF, class TI>
struct LinearSolver<Cpu, TF, TI> {
    /** Solve A x = rhs for SPD A given in COO format (n × n, nnz terms).
        Duplicate entries are summed.  x need not be initialised on input. */
    static void solve_spd( PI n, PI nnz,
                           const TI *rows, const TI *cols, const TF *vals,
                           const TF *rhs, TF *x );
};

#ifdef __CUDACC__
template<class TF, class TI>
struct LinearSolver<Cuda, TF, TI> {
    static void solve_spd( PI n, PI nnz,
                           const TI *rows, const TI *cols, const TF *vals,
                           const TF *rhs, TF *x );
};
#endif

} // namespace sdot

#include "LinearSolver.cxx" // IWYU pragma: export
