#pragma once

#include "support/TensorView.h"
#include "SplineGrid.h"

namespace sdot {

// 1d, order 1 (piecewise affine)
template<class TF,class Arch>
struct SplineGrid<TF,1,1,Arch> {
    using  Values = TensorView<const TF,1,Cpu>;
    using  Bounds = TensorView<const TF,2,Cpu>;
    using  Knots  = TensorView<const TF,1,Cpu>;

    /// Mutable state for traversing the CDF; holds only position, no grid copies
    struct Cursor {
        TF  mass, x0, x1, y0, y1;
        PI  i1;
    };

    /// Sub-interval [x0,x1] of rho, contained within grid cell [k0,k1]
    struct Piece {
        TF  w2_dist( TF ref_x ) const;
        TF  moment () const;

        PI  i1;
        TF  k0, k1;
        TF  x0, x1;
        TF  y0, y1;
    };

    /**/   SplineGrid                    ( Values values, Bounds bounds, const std::vector<Knots> &knots );

    Cursor first_cursor                  () const;
    Cursor last_cursor                   () const;
    PI     nb_values                     () const;
    TF     min_x                         () const;
    TF     max_x                         () const;
    TF     knot                          ( PI i ) const;

    void   apply_normalization_correction( TF g_dist, TF phi_avg, TensorView<TF,1,Cpu> grad_values ) const;
    void   accumulate_gradients_dist     ( const Piece &p, TF g_dist, TF dirac_x, TF potential, TensorView<TF,1,Cpu> grad_values ) const;
    void   take_some_mass                ( Cursor &, TF mass_to_take, auto &&func ) const;

    auto   base_cell                     ( PI dim ) const;

    TF     coeff_values;
    Values values;
    Bounds bounds;
    Knots  _knots;
};

} // namespace sdot

#include "SplineGrid_1d_o1.cxx"
