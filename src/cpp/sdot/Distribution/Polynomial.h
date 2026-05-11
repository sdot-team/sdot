#pragma once

#include "../support/common_types.h"
#include "../support/Vector.h"
#include "../Cell/Simplex.h"

namespace sdot {

/** Q_k tensor-product basis, lex multi-index (p0,p1,...) each in 0..order.
    Ex order=1 dim=2: coeffs[c] for monomials 1, x1, x0, x0*x1 (last axis fastest). */
template<int order, int dim, typename Arch, typename TF>
struct Polynomial {
    static constexpr int nb_coeffs = pow_rec( order + 1, PI( dim ) );

    Polynomial() : coeffs( Size(), nb_coeffs ) {}

    Vector<TF,Arch,nb_coeffs,Arch> coeffs;
};

/** ∫_{simplex} P(x) dx, simplex = array of dim+1 points */
// template<int dim,class TF,class Arch,int order>
// TF integral( const Simplex<dim,dim+1,TF,Arch> &simplex, const Polynomial<order,dim,Arch,TF> &local_function );

} // namespace sdot

#include "Polynomial.cxx" // IWYU pragma: export
