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
    /* */       Tensor     ( Rank, PI rank );

    /// Construct from any TensorView (possibly strided, possibly const-qualified element type).
    /// Uses memcpy when source is contiguous, element-wise copy otherwise.
    template<class T2,int ct_rank2>
    /* */       Tensor     ( const TensorView<T2,ct_rank2,Arch> &src );

    CTV         view       () const;
    TV          view       ();

    operator    CTV        () const;
    operator    TV         ();

    bool        is_invalid () const { return false; }


    const T&    operator() ( PI i0, PI i1, PI i2 ) const;
    const T&    operator() ( PI i0, PI i1 ) const;
    const T&    operator() ( PI i0 ) const;
    const T&    operator() () const;

    T&          operator() ( PI i0, PI i1, PI i2 );
    T&          operator() ( PI i0, PI i1 );
    T&          operator() ( PI i0 );
    T&          operator() ();

    /// DsVec-based access (forwards to TensorView)
    const T&    operator() ( const auto &indices ) const;
    T&          operator() ( const auto &indices );

    const T&    operator[] ( PI i0 ) const;
    T&          operator[] ( PI i0 );

    auto        row        ( PI index ) const;
    auto        row        ( PI index );

    auto        squeeze    ( PI axis ) const;
    auto        squeeze    ( PI axis );

    /// TensorView methods forwarded so callers don't need .view()
    void        for_each_index  ( auto &&func, PI sub = 0 ) const;
    auto        unsqueeze       () const;
    bool        is_contiguous   () const;

    Extent      sizes      () const;
    bool        empty      () const;

    PI          size       ( PI d ) const;
    PI          size       () const;
    PI          rank       () const { return extent.size(); }

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
