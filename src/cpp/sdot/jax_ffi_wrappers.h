#pragma once

/// Convenience helpers: convert typed XLA FFI buffers to sdot TensorView.
///
/// Usage (in a generated XLA FFI handler):
///
///   auto tv = tensor_view_input ( CtType<AxisTuple<PI,Cpu,2>>(), buf     );
///   auto tv = tensor_view_output( CtType<AxisTuple<PI,Cpu,0>>(), res_buf );
///
/// Axes with a compile-time known size use KnownAxisSize in the AxisTuple:
///   auto tv = tensor_view_input( CtType<AxisTuple<PI,Cpu,2,KnownAxisSize<PI,1,3>>>(), buf );

#include "support/DynamicAxis.h"
#include <xla/ffi/api/ffi.h>
#include <utility>

#include "support/CudaGpu.h" // IWYU pragma: export
#include "support/Cpu.h" // IWYU pragma: export

namespace sdot {


// convenience aliases for the two most common dtypes
template<xla::ffi::DataType dtype> struct SdotTypeFor;
template<> struct SdotTypeFor<xla::ffi::DataType::F32> { using type = FP32; };
template<> struct SdotTypeFor<xla::ffi::DataType::F64> { using type = FP64; };
template<> struct SdotTypeFor<xla::ffi::DataType::U32> { using type = PI32; };
template<> struct SdotTypeFor<xla::ffi::DataType::U64> { using type = PI64; };
template<> struct SdotTypeFor<xla::ffi::DataType::S32> { using type = SI32; };
template<> struct SdotTypeFor<xla::ffi::DataType::S64> { using type = SI64; };


// ------------------- zero strides -------------------
// All-zero byte strides: every element aliases data()[0].
// Used for surely-null tensors (SymbolicZero from JAX): not_surely_null() == false.

template<class Shape, std::size_t... Is>
auto _zero_strides_impl( std::index_sequence<Is...> ) {
    using Strides = AxisTuple<typename Shape::TI, typename Shape::Arch, Shape::ct_rank>;
    return Strides( Values(), ( (void)Is, SI(0) )... );
}

template<class Shape>
auto zero_strides() {
    return _zero_strides_impl<Shape>( std::make_index_sequence<Shape::ct_rank>{} );
}


// ------------------- contiguous strides -------------------
// C-contiguous (row-major) byte strides for a given shape AxisTuple.

template<class TF, class Shape, std::size_t... Is>
auto _contiguous_strides_impl( const Shape &shape, std::index_sequence<Is...> ) {
    using Strides = AxisTuple<typename Shape::TI,typename Shape::Arch,Shape::ct_rank>;
    if constexpr ( Shape::ct_rank == 0 ) {
        return Strides( Values() );
    } else {
        SI s[ Shape::ct_rank ];
        s[ Shape::ct_rank - 1 ] = sizeof( TF );
        for ( int i = Shape::ct_rank - 2; i >= 0; --i )
            s[ i ] = s[ i + 1 ] * SI( shape[ i + 1 ] );
        return Strides( Values(), SI( s[ Is ] )... );
    }
}

template<class TF, class Shape>
auto contiguous_strides( const Shape &shape ) {
    return _contiguous_strides_impl<TF>( shape, std::make_index_sequence<Shape::ct_rank>{} );
}


// ------------------- tensor_view_input -------------------

template<class Shape, xla::ffi::DataType dtype>
auto tensor_view_input( CtType<Shape>, xla::ffi::Buffer<dtype> buf, PI8 tensor_type_index = 1 ) {
    using TF = SdotTypeFor<dtype>::type;

    if ( tensor_type_index == 2 ) {
        // surely-null: shape in buf.dimensions()[1:], zero strides, data points to a static TF(0)
        auto shape = [&]<std::size_t... Is>( std::index_sequence<Is...> ) {
            return Shape( Values(), PI( buf.dimensions()[ Is + 1 ] )... );
        }( std::make_index_sequence<Shape::ct_rank>{} );
        auto strides = zero_strides<Shape>();
        using Strides = DECAYED_TYPE_OF( strides );
        static const TF _zero{};
        return TensorView<TF, Shape, Strides>( const_cast<TF *>( &_zero ), shape, strides );
    }

    auto shape = [&]<std::size_t... Is>( std::index_sequence<Is...> ) {
        return Shape( Values(), PI( buf.dimensions()[ Is ] )... );
    }( std::make_index_sequence<Shape::ct_rank>{} );

    auto strides = contiguous_strides<TF>( shape );
    using Strides = DECAYED_TYPE_OF( strides );

    if ( ! tensor_type_index )
        return TensorView<TF,Shape,Strides>::make_invalid( shape, strides );

    return TensorView<TF,Shape,Strides>( const_cast<TF *>( buf.typed_data() ), shape, strides );
}


// ------------------- tensor_view_output -------------------

template<class Shape, xla::ffi::DataType dtype>
auto tensor_view_output( CtType<Shape>, xla::ffi::ResultBuffer<dtype> buf, bool valid = true ) {
    using TF = SdotTypeFor<dtype>::type;

    auto shape = [&]<std::size_t... Is>( std::index_sequence<Is...> ) {
        return Shape( Values(), PI( buf->dimensions()[ Is ] )... );
    }( std::make_index_sequence<Shape::ct_rank>{} );

    auto strides = contiguous_strides<TF>( shape );
    using Strides = DECAYED_TYPE_OF( strides );

    if ( ! valid )
        return TensorView<TF,Shape,Strides>::make_invalid( shape, strides );

    return TensorView<TF,Shape,Strides>( buf->typed_data(), shape, strides );
}


// ------------------- tensor_view_mutable -------------------

template<class Shape, xla::ffi::DataType dtype>
auto tensor_view_mutable( CtType<Shape> cd, xla::ffi::ResultBuffer<dtype> buf_out, xla::ffi::Buffer<dtype> buf_inp, bool valid = true ) {
    if ( ! valid )
        return tensor_view_output( cd, buf_out, false );

    auto out = tensor_view_output( cd, buf_out, true );
    auto inp = tensor_view_input ( cd, buf_inp, true );
    out.get_data_from( inp );
    return out;
}


// ------------------- dynamic_axis -------------------

template<class Shape, xla::ffi::DataType dtype>
auto dynamic_axis_input( CtType<Shape> cd, PI num_dynamic_axis, PI capacity, xla::ffi::Buffer<dtype> buf, bool valid ) {
    return DynamicAxis( num_dynamic_axis, capacity, tensor_view_input( cd, buf, valid ) );
}

template<class Shape, xla::ffi::DataType dtype>
auto dynamic_axis_output( CtType<Shape> cd, PI num_dynamic_axis, PI capacity, xla::ffi::ResultBuffer<dtype> buf, bool valid ) {
    auto res = DynamicAxis( num_dynamic_axis, capacity, tensor_view_output( cd, buf, valid ) );
    if ( valid )
        res.sizes.fill_with( 0 );
    return res;
}

#ifdef __CUDACC__
template<class Shape, xla::ffi::DataType dtype>
auto dynamic_axis_output( CtType<Shape> cd, PI num_dynamic_axis, PI capacity, xla::ffi::ResultBuffer<dtype> buf, bool valid, cudaStream_t stream ) {
    using TF = typename SdotTypeFor<dtype>::type;
    auto res = DynamicAxis( num_dynamic_axis, capacity, tensor_view_output( cd, buf, valid ) );
    if ( valid )
        cudaMemsetAsync( buf->typed_data(), 0, buf->element_count() * sizeof( TF ), stream );
    return res;
}
#endif

template<class Shape, xla::ffi::DataType dtype>
auto dynamic_axis_mutable( CtType<Shape> cd, PI num_dynamic_axis, PI capacity, xla::ffi::Buffer<dtype> buf_inp, bool valid_inp, xla::ffi::ResultBuffer<dtype> buf_out, bool /* valid_out */ ) {
    auto res = DynamicAxis( num_dynamic_axis, capacity, tensor_view_output( cd, buf_out ) );
    if ( valid_inp ) {
        auto inp = tensor_view_input( cd, buf_inp );
        res.sizes.get_data_from( inp );
    }
    return res;
}


// ------------------- first_positive -------------------

SI first_positive( const std::string &name ) { return 0; }
SI first_positive( const std::string &name, SI head, auto...tail ) { return head >= 0 ? head : first_positive( name, tail... ); }

} // namespace sdot
