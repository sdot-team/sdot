#pragma once

#include "../support/common_macros.h"
#include "TensorView.h"
#include <ostream>
#include <span>

namespace sdot {

/// owning tensor — contiguous row-major by construction, strides stored (no recomputation on access)
template<class T,int ct_rank,class Arch>
class Tensor {
public:
    using       value_type = T;
    using       CTV        = TensorView<const T,ct_rank,Arch>;
    using       TV         = TensorView<T,ct_rank,Arch>;
    using       Storage    = Arch::template Vector<T>::type;
    using       Strides    = typename TV::Strides;
    using       Extent     = typename TV::Sizes;
    using       Byte       = std::byte;

    T_U         Tensor     ( std::initializer_list<std::initializer_list<std::initializer_list<U>>> m );
    T_U         Tensor     ( std::initializer_list<std::initializer_list<U>> m );
    T_U         Tensor     ( std::initializer_list<U> l );
    /* */       Tensor     ( sdot::Shape, Extent ext );
    T_U         Tensor     ( std::span<U> l );
    /* */       Tensor     ();

    CTV         view       () const;
    TV          view       ();

    operator    CTV        () const;
    operator    TV         ();

    const T&    operator() ( PI i0, PI i1, PI i2 ) const;
    const T&    operator() ( PI i0, PI i1 ) const;
    const T&    operator() ( PI i0 ) const;
    const T&    operator() () const;

    T&          operator() ( PI i0, PI i1, PI i2 );
    T&          operator() ( PI i0, PI i1 );
    T&          operator() ( PI i0 );
    T&          operator() ();

    const T&    operator[] ( PI i0 ) const;
    T&          operator[] ( PI i0 );

    auto        row        ( PI index ) const;
    auto        row        ( PI index );

    auto        squeeze    ( PI axis ) const;
    auto        squeeze    ( PI axis );

    bool        empty      () const;

    PI          size       ( PI d ) const;
    PI          size       () const;

    const T*    data       () const;
    T*          data       ();

    const T*    begin      () const;
    T*          begin      ();

    const T*    end        () const;
    T*          end        ();

private:
    static PI   total_size ( const Extent &ext );

    const auto* bytes      () const;
    auto*       bytes      ();

    Extent      extent;
    Strides     strides;
    Storage     storage;
};

} // namespace sdot

template<class T,int ct_rank,class Arch>
std::ostream &operator<<( std::ostream &os, const sdot::Tensor<T,ct_rank,Arch> &t );

#include "Tensor.cxx" // IWYU pragma: export
