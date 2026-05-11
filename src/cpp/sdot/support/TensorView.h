#pragma once

#include "AxisTuple.h"
#include "Vector.h"

namespace sdot {

/// view on strided data (strides in bytes, handles non-contiguous arrays)
template<class TF,class _Shape,class _Strides>
class TensorView {
public:
    using            value_type         = TF;
    using            Strides            = _Strides;
    using            Shape              = _Shape;

    SCInt            ct_rank            = Shape::ct_rank;
    using            RawPtr             = std::conditional_t<std::is_const_v<TF>,const std::byte*,std::byte*>;
    using            Arch               = Shape::Arch;
    using            TI                 = Shape::TI;

    static HD auto   make_invalid       ( Shape shape, Strides strides ) -> TensorView; ///< invalid TensorView — is_valid()==false, _ptr==&_sentinel

    HD               TensorView         ( TF *data, Shape shape, Strides strides );

    // operator() produces a new tensor
    HD auto          operator()         ( const auto &indices, auto ...rem ) const requires ( requires { indices.size(); } );
    HD auto          operator()         ( const auto &index, auto ...rem ) const;
    HD auto          operator()         () const { return *this; }

    // operator[] allows to get a scalar value/reference for a rank 1 tensor
    HD TF&           operator[]         ( const auto &index ) const;

    // scalar value/reference for a rank 0 tensor
    HD               operator TF        () const requires ( ct_rank <= 0 ) { ASSERT( rank() == 0 ); return *data(); }
    HD TensorView&   operator=          ( TF value ) requires ( ct_rank <= 0 ) { ASSERT( rank() == 0 ); *data() = value; return *this; }
    HD TF&           item               () const;

    // data copy / transfer
    HD void          get_data_from      ( const auto &that, const auto &size_to_take );
    HD void          get_data_from      ( const auto &that );
    void             fill_with          ( TF value );

    void             spill_to           ( TensorView &that ); ///< copie data of this to that, and use data from that

    // strides
    HD Strides       strides            () const;
    HD SI            stride             ( auto d ) const;

    // shape
    HD PI            total_size         () const;
    HD auto          shape              ( auto d ) const { return _shape[ d ]; }
    Shape            shape              () const { return _shape; }
    HD auto          empty              () const;
    HD auto          size               () const;

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
    auto             squeeze            ( auto axis, PI index = 0 ) const;
    HD auto          row                ( PI index ) const;

    void             with_same_shape    ( auto &&func ) const;

private:
    static std::byte _sentinel;        ///< address used as invalid marker — never points to real data

    RawPtr           _raw_ptr;          ///<
    Strides          _strides;          ///< byte strides
    Shape            _shape;            ///<
    Arch             _arch;             ///<
};

} // namespace sdot

#ifdef __CUDACC__
template<class T,int ct_rank>
std::ostream &operator<<( std::ostream &os, const sdot::TensorView<T,ct_rank,sdot::Cuda> &p );
#endif

#include "TensorView.cxx" // IWYU pragma: export
