#pragma once

#include "support/IntermediateScalarType.h"
#include "support/TensorView.h"

#include "PieceOfAffine1d.h"

namespace sdot {

template<class T,class Arch>
struct PiecewiseAffineFunction1d {
    using    TF                       = IntermediateScalarType<std::decay_t<T>,Arch>::type;
    using    Piece                    = PieceOfAffine1d<TF>;
    using    TT                       = TensorView<T,1,Arch>;

    HD void  accumulate_linear_grad_ys( const Piece &piece, TF slope, TF offset, TF scale, auto grad_ys ) const;          ///< backward of moment += piece.moment() with dual variable (slope*t + offset): accumulates into grad_ys[piece.index-1] and grad_ys[piece.index]
    HD void  accumulate_w2_grad_ys    ( const Piece &piece, TF dirac_pos, TF potential, TF g_scale, auto grad_ys ) const; ///< backward of w2 += piece.w2_dist(dirac_pos): accumulates into grad_ys[piece.index-1] and grad_ys[piece.index]
    HD Piece get_first_piece          ( TF point_scale ) const;
    HD TF    take_some_mass           ( Piece &current_piece, TF point_scale, TF mass_to_take, auto &&on_taken_piece ) const; ///< return right position
    HD void  get_next_piece           ( Piece &piece, TF point_scale ) const;
    HD void  get_grad_ys              ( T ratio, auto grad_y ) const;

    HD PI    nb_points                () const;
    HD TF    mass                     () const;

    TT       xs;                      ///<
    TT       ys;                      ///<
};

} // namespace sdot

#include "PiecewiseAffineFunction1d.cxx"
