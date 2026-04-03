#pragma once

#include "support/TensorView.h"
#include "SplineGrid.h"

namespace sdot {

// 1d, order 1 (picewise affine)
template<class TF,class Knots>
struct SplineGrid<TF,1,Knots> {
    using  Values = TensorView<const TF,1,Cpu>;

    struct Piece {
        void   take_some_mass( TF mass_to_take, auto &&func ); // call `func` on sub parts of the piece to take `mass_to_take`.
        TF     value_at      ( TF x ) const;

        TF     coeff_values;
        Values values;
        Knots  knots;
        TF     mass;
        PI     i1;
        TF     x0;
        TF     x1;
        TF     y0;
        TF     y1;
    };

    struct Part {
        TF     w2_dist ( TF dirac_x ) const;
        TF     moment  () const;
        PI     i1;
        TF     x0;
        TF     x1;
        TF     y0;
        TF     y1;
    };

    /**/   SplineGrid ( Values values, Knots knots );

    Piece  first_piece() const;
    Piece  last_piece () const;

    void   accumulate_gradients( const Part &part, TF coeff_x2, TF coeff_x1, TF coeff_x0, TensorView<TF,1,Cpu> grad_values ) const;

    TF     coeff_values;
    Values values;
    Knots  knots;
};

} // namespace sdot

#include "SplineGrid_1d_o1.cxx"
