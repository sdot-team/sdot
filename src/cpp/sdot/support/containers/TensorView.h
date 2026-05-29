#pragma once

#include "../hardware/MemorySpace_GlobalCudaRam.h" // IWYU pragma: export
#include "../hardware/MemorySpace_PinnedCpuRam.h" // IWYU pragma: export
#include "../hardware/MemorySpace_CpuRam.h" // IWYU pragma: export
#include "../hardware/Ptr.h"

#include "contiguous_strides.h" // IWYU pragma: export
#include "container_tags.h" // IWYU pragma: export
#include "Vector.h" // IWYU pragma: export
#include "Tuple.h" // IWYU pragma: export

namespace sdot {


/// view on strided data (strides in bytes, handles non-contiguous arrays)
///   MemorySpace = where the data lives (drives the informed pointer Ptr<...,MemorySpace>).
///   The shape no longer carries any arch/memory tag; it is a parameter of the view.
///   Tags...      = compile-time facts about the view (see container_tags.h).
template<class _TF,class _MemorySpace,class _Shape,class _Strides = DECAYED_TYPE_OF( contiguous_strides<_TF>( _Shape{} ) ),class... _Tags>
class TensorView {
public:
    using            MemorySpace          = _MemorySpace;
    using            Strides              = _Strides;
    using            Shape                = _Shape;
    using            TF                   = _TF;
    using            TI                   = SI;

    /// is `Tag` part of this view's compile-time tag set ?
    template<class Tag> static constexpr bool has_tag = contains_tag<Tag,_Tags...>;

    using            value_type           = TF;
    SCInt            ct_rank              = Shape::ct_size;
    using            RawByte              = std::conditional_t<std::is_const_v<TF>,const std::byte,std::byte>;
    using            RawPtr               = Ptr<RawByte,MemorySpace>; ///< informed byte pointer (address + memory space)

    HD               TensorView           ( TF *data, Shape shape, Strides strides, MemorySpace memory_space = {} );
    HD               TensorView           ( const TensorView & ) = default; ///< Eigen-like view semantics: copy-construction shares the data (shallow), while operator= copies the elements (deep). The defaulted copy-ctor also silences -Wdeprecated-copy.

    // static ctors
    static HD auto   make_invalid         ( Shape shape, Strides strides, MemorySpace memory_space = {} ) -> TensorView; ///< invalid TensorView — is_valid()==false, _ptr==&_sentinel

    // generic info
    HD bool          not_surely_null      () const { return ! surely_null(); }
    HD bool          surely_null          () const; ///< is_invalid() || Zero tensor
    HD bool          is_invalid           () const; ///<
    HD bool          is_valid             () const; ///<

    // generic info
    MemorySpace      memory_space         () const { return _memory_space; }
    void             display              ( std::ostream &os ) const;
    HD auto          rank                 () const;

    //
    HD Strides       strides              () const;
    HD auto          stride               ( auto d ) const;

    // shape
    HD void          for_each_index       ( auto &&func ) const;
    HD void          for_each_item        ( auto &&func ) const;
    HD auto          is_contiguous        () const; ///< true iff strides match row-major contiguous layout
    HD auto          all_indices          () const;
    HD auto          nb_items             () const;
    HD auto          shape                ( auto d ) const { return _shape[ d ]; }
    HD Shape         shape                () const { return _shape; }
    HD auto          empty                () const;
    HD auto          size                 () const;

    // content
    HD auto          data                 () const;

    HD auto          begin                () const;
    HD auto          end                  () const;

    // operator() and operator[] produce a new tensor
    HD auto          operator()           ( const auto &index, auto ...rem ) const;
    HD auto          operator[]           ( const auto &index ) const { return operator()( index ); }
    HD auto          operator()           () const { return *this; }

    HD auto          offset               ( const auto &index, auto ...rem ) const;
    HD auto          offset               () const { return *this; }

    // scalar value/reference for a rank 1 tensor
    HD               operator TF          () const { return value(); }
    HD TF            value                () const;
    HD TF&           ref                  () const;

    // reassign
    HD void          copy_elements_from   ( const auto &that );
    HD void          operator-=           ( const auto &that );
    HD void          operator+=           ( const auto &that );
    HD void          operator*=           ( const auto &that );
    HD void          operator/=           ( const auto &that );
    HD void          operator=            ( const auto &that );
    HD void          operator=            ( const TensorView &that );
    HD void          spill_to             ( TensorView &that ); ///< copy data of *this to that, and use data from that

    // data copy / transfer — arch-unaware (HD, valid in device code)
    HD void          make_accessible_mut        ( auto execution_space, auto &&func ) const; ///< read-write: copy in, copy back
    HD void          make_accessible_out        ( auto execution_space, auto &&func ) const; ///< write-only: no copy in, copy back
    HD void          make_accessible_inp        ( auto execution_space, auto &&func ) const; ///< read-only:  transfer in, no copy back
    auto             transfer_cost              ( const auto &execution_context ) const;

    void             with_same_shape            ( const auto &arch, auto &&func ) const;
    HD void          fill_with                  ( TF value );

    //
    HD auto          unsqueeze                  ( auto axis ) const; ///< append a trailing dimension of size 1 (preserves strides)
    HD auto          squeeze                    ( auto axis, PI index = 0 ) const;
    HD auto          row                        ( PI index ) const;

    // compile-time tags (see container_tags.h)
    template<class... ExtraTags>
    HD auto          with_tags            () const; ///< same view, with ExtraTags... added to the tag set (no-op for tags already present)
    HD auto          as_already_parallelized() const; ///< == with_tags<container_tags::has_already_been_parallelized>()

private:
    static HD RawPtr _sentinel            () { return RawPtr( nullptr ) + 1; }
    CPU_ONLY void    transfer_through     ( auto execution_space, bool copy_in, bool copy_back, auto &&func ) const; ///< host-only transfer path: materialize in the exec space, optionally copy in, run, optionally copy back

    MemorySpace      _memory_space;       ///<
    RawPtr           _raw_ptr;            ///<
    Strides          _strides;            ///< byte strides
    Shape            _shape;              ///<
};

#undef SDOT_DATA_ACCESSOR

} // namespace sdot

#include "TensorView.cxx" // IWYU pragma: export
