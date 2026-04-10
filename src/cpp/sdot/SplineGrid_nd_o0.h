#pragma once

#include "geometry/PointFactory.h"
#include "support/TensorView.h"
#include "geometry/Cell.h"
#include "SplineGrid.h"

namespace sdot {

// nd, order 0 (piecewise constant)
template<class TF,int ct_dim,class Arch>
struct SplineGrid<TF,ct_dim,0,Arch> {
    using        PiFactory      = PointFactory<PI,ct_dim,Arch>;
    using        PfFactory      = PointFactory<TF,ct_dim,Arch>;
    using        Values         = TensorView<const TF,ct_dim,Arch>;
    using        Bounds         = TensorView<const TF,2,Arch>;
    using        Knots          = TensorView<const TF,1,Arch>;
    using        Pi             = Point<PI,ct_dim,Arch>;
    using        Pt             = Point<TF,ct_dim,Arch>;

    /**/         SplineGrid     ( Values values, Bounds bounds, const std::vector<Knots> &knots );

    void         for_each_piece ( const auto &cell, auto &&func ) const;
    TF           knot           ( PI d, PI index ) const;
    PI           dim            () const { return ct_dim; }

    auto         facet_integral ( auto facet, auto value ) const;
    auto         integral       ( auto cell, auto value ) const;

    T_U auto     base_cell      ( PI dim, typename Cell<U,ct_dim,Arch>::CellInfo cell_info = {}, typename Cell<U,ct_dim,Arch>::CutInfo cut_info = {} ) const;

    void         _for_each_piece( const auto &cell, const auto &func, auto beg, auto end, auto &cur, PI d ) const;

    TF           coeff_values;
    Values       values;
    Bounds       bounds;
    PiFactory    pif;
    PfFactory    pf;
};

} // namespace sdot

#include "SplineGrid_nd_o0.cxx"
