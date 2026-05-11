#pragma once

#include <sdot/generated_includes/BatchOfCellWorkspace.h>
#include <sdot/generated_includes/PolynomialGrid.h>
#include <sdot/generated_includes/BatchOfCell.h>
#include "../Cell/make_hypercube.h"
#include "Polynomial.h"

namespace sdot {

template<int nb_coeffs, int dim,typename Arch,typename TF,typename TI,typename Knots>
struct PolynomialGridWorker {
    static constexpr int  _order                             () { for ( int k = 1; k <= nb_coeffs; ++k ) { int p = 1; for ( int d = 0; d < dim; ++d ) p *= k; if ( p >= nb_coeffs ) return k - 1; } return -1; } // nb_coeffs == (order+1)^dim: find the integer dim-th root of nb_coeffs minus 1
    static constexpr int  order                              = _order();


    using                 BatchOfCellWorkspace               = sdot::BatchOfCellWorkspace<Arch,TF,TI>;
    using                 PolynomialGrid                     = sdot::PolynomialGrid<nb_coeffs,dim,Arch,TF,TI>;
    using                 BatchOfCell                        = sdot::BatchOfCell<dim,Arch,TF,TI>;
    using                 Polynomial                         = sdot::Polynomial<order,dim,Arch,TF>;
    using                 Cell                               = sdot::Cell<dim,Arch,TF,TI>;
    using                 Pt                                 = Vector<TF,Arch,dim,Arch>;
    using                 Pi                                 = Vector<TI,dim,Arch>;

    void                  with_preparation_for_cell_traversal( auto &cell_workspace, auto &cells, auto &&func );
    void                  for_each_sub_cell                  ( const auto &cell, TI batch_index, auto &&func ); // ( auto &cell_worker, const auto &local_function )
    void                  for_each_index                     ( auto &&func ) const;
    void                  init_cell                          ( Cell &cell );

    TF                    piece_integral                     ( auto index, const Polynomial &pol ) const;
    Polynomial            polynomial                         ( auto position ) const;
    TF                    mass                               ();

    TF                    cut_offset                         ( TI d, Pi index );
    Pt                    cut_plane                          ( TI d ) const;

    PolynomialGrid&       pgrid;                             ///<
    Knots&                knots;                             ///<

    BatchOfCellWorkspace* batch_of_cell_workspace;           ///<
    BatchOfCell*          batch_of_cell;                     ///<
};

template<int nb_coeffs,int dim,typename Arch,typename TF,typename TI>
auto with_worker_for( PolynomialGrid<nb_coeffs,dim,Arch,TF,TI> &polynomial_grid, auto &&func ) {
    BatchOfCellWorkspace<Arch,TF,TI> *w = nullptr;
    BatchOfCell<dim,Arch,TF,TI> *c = nullptr;

    if ( polynomial_grid.knots.is_valid() ) {
        PolynomialGridWorker pw( polynomial_grid, polynomial_grid.knots, w, c );
        return func( pw );
    }

    auto knots = []( TI, TI i ) { return i; };
    PolynomialGridWorker pw( polynomial_grid, knots, w, c );
    return func( pw );
}

} // namespace sdot

#include "PolynomialGridWorker.cxx" // IWYU pragma: export
