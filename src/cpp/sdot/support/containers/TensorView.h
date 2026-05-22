#pragma once

#include "../hardware/MemorySpace_GlobalCudaRam.h" // IWYU pragma: export
#include "../hardware/MemorySpace_PinnedCpuRam.h" // IWYU pragma: export
#include "../hardware/MemorySpace_CpuRam.h" // IWYU pragma: export
#include "../hardware/Ptr.h"

#include "contiguous_strides.h" // IWYU pragma: export
#include "Vector.h" // IWYU pragma: export
#include "Tuple.h" // IWYU pragma: export

namespace sdot {


/// view on strided data (strides in bytes, handles non-contiguous arrays)
///   MemorySpace = where the data lives (drives the informed pointer Ptr<...,MemorySpace>).
///   The shape no longer carries any arch/memory tag; it is a parameter of the view.
template<class _TF,class _Shape,class _Strides,class _MemorySpace>
class TensorView {
public:
    using            MemorySpace          = _MemorySpace;
    using            Strides              = _Strides;
    using            Shape                = _Shape;
    using            TF                   = _TF;
    using            TI                   = SI;

    using            value_type           = TF;
    SCInt            ct_rank              = Shape::ct_size;
    using            RawByte              = std::conditional_t<std::is_const_v<TF>,const std::byte,std::byte>;
    using            RawPtr               = Ptr<RawByte,MemorySpace>; ///< informed byte pointer (address + memory space)

    HD               TensorView           ( TF *data, Shape shape, Strides strides, MemorySpace memory_space = {} );
    HD               TensorView           ( const TensorView & ) = default; ///< Eigen-like view semantics: copy-construction shares the data (shallow), while operator= copies the elements (deep). The defaulted copy-ctor also silences -Wdeprecated-copy.

    // static ctors
    static HD auto   make_invalid         ( Shape shape, Strides strides, MemorySpace memory_space = {} ) -> TensorView; ///< invalid TensorView — is_valid()==false, _ptr==&_sentinel

    // info
    MemorySpace      memory_space         () const { return _memory_space; }
    HD PI            nb_items             () const;
    void             display              ( std::ostream &os ) const;

    HD Strides       strides              () const;
    HD SI            stride               ( auto d ) const;

    HD auto          shape                ( auto d ) const { return _shape[ d ]; }
    HD Shape         shape                () const { return _shape; }
    HD auto          empty                () const;
    HD auto          size                 () const;

    HD bool          not_surely_null      () const { return ! surely_null(); }
    HD bool          surely_null          () const; ///< is_invalid() || Zero tensor
    HD bool          is_invalid           () const; ///<
    HD bool          is_valid             () const; ///<

    HD auto          rank                 () const;

    HD auto          data                 () const;

    HD auto          begin                () const;
    HD auto          end                  () const;

    // operator() and operator[] produce a new tensor
    HD auto          operator()           ( const auto &index, auto ...rem ) const;
    HD auto          operator[]           ( const auto &index ) const { return operator()( index ); }
    HD auto          operator()           () const { return *this; }

    // scalar value/reference for a rank 1 tensor
    HD               operator TF          () const { return value(); }
    HD TF            value                () const;
    HD TF&           ref                  () const;

    // reassign
    void             copy_elements_from   ( const auto &that ) const;
    void             operator=            ( const TensorView &that ) { copy_elements_from( that ); }
    HD void          spill_to             ( TensorView &that ); ///< copy data of *this to that, and use data from that

    //     SDOT_DATA_ACCESSOR( void operator=( TF value ), { static_assert( ct_rank == 0 ); *data() = value; } )
    //     SDOT_DATA_ACCESSOR( void operator+=( TF value ), { static_assert( ct_rank == 0 ); *data() += value; } )
    //     SDOT_DATA_ACCESSOR( void operator-=( TF value ), { static_assert( ct_rank == 0 ); *data() -= value; } )
    // #ifdef __CUDACC__
    //     __host__   void  operator=( const auto &that ) requires ( host_accessible_v<MemorySpace>   && requires { that.shape(); } ) { copy_elements_from( that ); }
    //     __device__ void  operator=( const auto &that ) requires ( device_accessible_v<MemorySpace> && requires { that.shape(); } ) { copy_elements_from( that ); }
    // #else
    //     void             operator=( const auto &that ) requires ( host_accessible_v<MemorySpace>   && requires { that.shape(); } ) { copy_elements_from( that ); }
    // #endif

    // data copy / transfer — arch-unaware (HD, valid in device code)
    void             make_accessible      ( auto execution_space, auto &&func ) const;
    HD void          fill_with            ( TF value );

    // data copy / transfer — arch-aware (host only, dispatches to GPU when arch=CudaGpu)
    void             fill_with            ( const auto &arch, TF value );

    // cross-arch copy: dst and src may live on different devices

    // shape

    HD void          for_each_index       ( auto &&func ) const;

    HD bool          is_contiguous        () const; ///< true iff strides match row-major contiguous layout

    // T_U auto      sum_along_axis_1() const -> Tensor<U,1,Arch>;
    void             apply_cpu_version    ( auto &&func ) const;
    HD auto          unsqueeze            ( auto axis ) const; ///< append a trailing dimension of size 1 (preserves strides)
    HD auto          squeeze              ( auto axis, PI index = 0 ) const;
    HD auto          row                  ( PI index ) const;

    void             with_same_shape      ( const auto &arch, auto &&func ) const;

private:
    static HD RawPtr _sentinel            () { return RawPtr( nullptr ) + 1; }

    MemorySpace      _memory_space;       ///<
    RawPtr           _raw_ptr;            ///<
    Strides          _strides;            ///< byte strides
    Shape            _shape;              ///<
};

#undef SDOT_DATA_ACCESSOR

} // namespace sdot

#include "TensorView.cxx" // IWYU pragma: export
