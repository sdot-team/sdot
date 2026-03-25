#pragma once

#include "support/IntermediateScalarType.h"
#include "support/TensorView.h"

#include "PieceOfAffine1d.h"

namespace sdot {

template<class T>
struct Affine1d;

template<class T>
struct Affine1d {
    using  TF             = IntermediateScalarType<std::decay_t<T>>::type;
    using  Piece          = PieceOfAffine1d<TF>;
    using  TT             = TensorView<T,1>;

    Piece  get_first_piece         ( TF point_scale ) const;
    TF     take_some_mass          ( Piece &current_piece, TF point_scale, TF mass_to_take, auto &&on_taken_piece ) const; ///< return right position
    void   get_next_piece          ( Piece &piece, TF point_scale ) const;
    void   get_grad_ys             ( T ratio, auto grad_y ) const;

    void   accumulate_w2_grad_ys   ( const Piece &piece, TF dirac_pos, TF potential, TF g_scale, auto grad_ys ) const; ///< backward of w2 += piece.w2_dist(dirac_pos): accumulates into grad_ys[piece.index-1] and grad_ys[piece.index]
    void   accumulate_linear_grad_ys( const Piece &piece, TF slope, TF offset, TF scale, auto grad_ys ) const;          ///< backward of moment += piece.moment() with dual variable (slope*t + offset): accumulates into grad_ys[piece.index-1] and grad_ys[piece.index]

    PI     nb_points      () const;
    TF     mass           () const;

    TT     xs;            ///<
    TT     ys;            ///<
};

template<class T>
struct BatchOfAffine1d {
    using  TT             = TensorView<T,2>;

    PI     nb_rows        () const;
    auto   row            ( PI num_batch ) const -> Affine1d<T>;

    TT     xs;            ///<
    TT     ys;            ///<
};

} // namespace sdot

#include "Affine1d.cxx"
