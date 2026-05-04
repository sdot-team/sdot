#pragma once

#include "LinearSolver.h"

#include <Eigen/Sparse>
#include <vector>

#ifdef SDOT_USE_AMGCL
#  include <amgcl/make_solver.hpp>
#  include <amgcl/solver/cg.hpp>
#  include <amgcl/amg.hpp>
#  include <amgcl/coarsening/smoothed_aggregation.hpp>
#  include <amgcl/relaxation/spai0.hpp>
#  include <amgcl/backend/builtin.hpp>
#  include <amgcl/adapter/eigen.hpp>
#else
#  include <Eigen/SparseCholesky>
#endif

namespace sdot {

template<class TF, class TI>
void LinearSolver<Cpu, TF, TI>::solve_spd(
    PI n, PI nnz,
    const TI *rows, const TI *cols, const TF *vals,
    const TF *rhs, TF *x
) {
#ifdef SDOT_USE_AMGCL
    // Build CSR (RowMajor) for the AMGCL Eigen adapter
    Eigen::SparseMatrix<TF, Eigen::RowMajor> A( n, n );
#else
    // Build CSC (ColMajor) for Eigen's direct solver
    Eigen::SparseMatrix<TF> A( n, n );
#endif
    {
        std::vector<Eigen::Triplet<TF>> triplets;
        triplets.reserve( nnz );
        for ( PI i = 0; i < nnz; ++i )
            triplets.emplace_back( int( rows[ i ] ), int( cols[ i ] ), vals[ i ] );
        A.setFromTriplets( triplets.begin(), triplets.end() );
    }
    A.makeCompressed();

#ifdef SDOT_USE_AMGCL
    // AMG-preconditioned conjugate gradient via AMGCL
    // Suited for Poisson-like SPD systems; scales to large problems.
    using Backend = amgcl::backend::builtin<TF>;
    using AMG     = amgcl::amg<Backend,
                        amgcl::coarsening::smoothed_aggregation,
                        amgcl::relaxation::spai0>;
    using Solver  = amgcl::make_solver<AMG, amgcl::solver::cg<Backend>>;

    Solver solver( A );  // AMGCL reads CSR directly from RowMajor Eigen matrix

    std::vector<TF> b( rhs, rhs + n );
    std::vector<TF> sol( n, TF( 0 ) );
    solver( b, sol );
    std::copy( sol.begin(), sol.end(), x );
#else
    // SimplicialLDLT: direct Cholesky — robust fallback for smaller systems
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<TF>> solver;
    solver.compute( A );

    Eigen::Map<const Eigen::Matrix<TF,-1,1>> b_map( rhs, n );
    Eigen::Map<Eigen::Matrix<TF,-1,1>>       x_map( x,   n );
    x_map = solver.solve( b_map );
#endif
}

#ifdef __CUDACC__
// TODO: switch Backend to amgcl::backend::cuda<TF> (requires device-side rhs/x)
//       or use cuSPARSE + cuSOLVER directly.
template<class TF, class TI>
void LinearSolver<Cuda, TF, TI>::solve_spd(
    PI, PI, const TI *, const TI *, const TF *, const TF *, TF *
) {
    TODO;
}
#endif

} // namespace sdot
