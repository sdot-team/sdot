#pragma once

#include "../hardware/Ptr.h"
#include "AxisValues.h"
#include "Vector.h"

namespace sdot {

/// view on strided data (strides in bytes, handles non-contiguous arrays)
///   MemorySpace = where the data lives (drives the informed pointer Ptr<...,MemorySpace>).
///   The shape no longer carries any arch/memory tag; it is a parameter of the view.
template<class TF,class _Shape,class _Strides,class _MemorySpace>
class TensorView {
public:
    using            MemorySpace        = _MemorySpace;
    using            value_type         = TF;
    using            Strides            = _Strides;
    using            Shape              = _Shape;

    SCInt            ct_rank            = Shape::ct_rank;
    using            RawByte            = std::conditional_t<std::is_const_v<TF>,const std::byte,std::byte>;
    using            RawPtr             = Ptr<RawByte,_MemorySpace>; ///< informed byte pointer (address + memory space)
    using            TI                 = Shape::TI;

    static HD auto   make_invalid       ( Shape shape, Strides strides, MemorySpace memory_space = {} ) -> TensorView; ///< invalid TensorView — is_valid()==false, _ptr==&_sentinel

    HD               TensorView         ( TF *data, Shape shape, Strides strides, MemorySpace memory_space = {} );

    // operator() produces a new tensor
    HD auto          operator()         ( const auto &indices, auto ...rem ) const requires ( requires { DECAYED_TYPE_OF( indices.size() )::value; } );
    HD auto          operator()         ( const auto &index, auto ...rem ) const;
    HD auto          operator()         () const { return *this; }

    // operator[] allows to get a scalar value/reference for a rank 1 tensor
    HD TF&           operator[]         ( const auto &index ) const;

    // scalar value/reference for a rank 0 tensor
    HD               operator TF        () const { ASSERT( rank() == 0 ); return *data(); }

    HD void          operator=          ( const TensorView &that ) { get_data_from( that, _shape ); }
    HD void          operator=          ( TF value ) { ASSERT( rank() == 0 ); *data() = value; }

    HD void          operator+=         ( TF value ) { ASSERT( rank() == 0 ); *data() += value; }
    HD void          operator-=         ( TF value ) { ASSERT( rank() == 0 ); *data() -= value; }

    HD TF&           item               () const;

    // data copy / transfer — arch-unaware (HD, valid in device code)
    HD void          get_data_from      ( const auto &that, const auto &size_to_take );
    HD void          get_data_from      ( const auto &that );
    HD void          fill_with          ( TF value );

    // data copy / transfer — arch-aware (host only, dispatches to GPU when arch=CudaGpu)
    void             get_data_from      ( const auto &arch, const auto &that ) requires requires { arch.copy( (void*)nullptr, (const void*)nullptr, PI{} ); };
    void             fill_with          ( const auto &arch, TF value );

    // cross-arch copy: dst and src may live on different devices
    void             get_data_from      ( const auto &dst_arch, const auto &src_arch, const auto &that ) requires requires { arch_copy( dst_arch, (void*)nullptr, src_arch, (const void*)nullptr, PI{} ); };

    HD void          spill_to           ( TensorView &that ); ///< copie data of this to that, and use data from that

    // strides
    HD Strides       strides            () const;
    HD SI            stride             ( auto d ) const;

    // shape
    HD PI            nb_items         () const;
    HD auto          shape              ( auto d ) const { return _shape[ d ]; }
    HD Shape         shape              () const { return _shape; }
    HD auto          empty              () const;
    HD auto          size               () const;

    HD bool          not_surely_null    () const { return ! surely_null(); }
    HD bool          surely_null        () const; ///< is_invalid() || Zero tensor
    HD bool          is_invalid         () const; ///<
    HD bool          is_valid           () const; ///<

    HD auto          rank               () const;

    HD TF*           data               () const;

    HD auto          begin              () const;
    HD auto          end                () const;

    HD void          for_each_index     ( auto &&func ) const;

    HD bool          is_contiguous      () const; ///< true iff strides match row-major contiguous layout

    // T_U auto      sum_along_axis_1() const -> Tensor<U,1,Arch>;
    void             apply_cpu_version  ( auto &&func ) const;
    HD auto          unsqueeze          ( auto axis ) const; ///< append a trailing dimension of size 1 (preserves strides)
    HD auto          squeeze            ( auto axis, PI index = 0 ) const;
    HD auto          row                ( PI index ) const;

    void             with_same_shape    ( const auto &arch, auto &&func ) const;

// private:
    static HD RawPtr sentinel           () { return RawPtr( nullptr ) + 1; }

    MemorySpace      _memory_space;     ///<
    RawPtr           _raw_ptr;          ///<
    Strides          _strides;          ///< byte strides
    Shape            _shape;            ///<
};

} // namespace sdot

#include "TensorView.cxx" // IWYU pragma: export
