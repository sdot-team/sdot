#pragma once

#include "support/IntermediateScalarType.h"
#include "support/TensorView.h"

#include "PieceOfAffine1d.h"

namespace sdot {

template<class T,int batch_dim=0>
struct Affine1d;

template<class T>
struct Affine1d<T,0> {
    using  TF             = IntermediateScalarType<std::decay_t<T>>::type;
    using  Piece          = PieceOfAffine1d<TF>;
    using  TT             = TensorView<T,1>;

    Piece  get_first_piece() const;
    TF     take_some_mass ( Piece &current_piece, TF mass_to_take, auto &&on_taken_piece ) const; ///< return right position
    void   get_next_piece ( Piece &piece ) const;

    PI     nb_points      () const;
    TF     mass           () const;

    TT     xs;            ///<
    TT     ys;            ///<
};

template<class T>
struct Affine1d<T,1> {
    using  TT             = TensorView<T,2>;

    PI     nb_rows        () const;
    auto   row            ( PI num_batch ) const -> Affine1d<T,0>;

    TT     xs;            ///<
    TT     ys;            ///<
};

} // namespace sdot

#include "Affine1d.cxx"
