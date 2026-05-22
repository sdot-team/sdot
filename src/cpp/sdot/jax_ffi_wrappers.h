#pragma once

/// Convenience helpers: convert typed XLA FFI buffers to sdot TensorView.
///
/// Usage (in a generated XLA FFI handler):
///
///   auto tv = tensor_view_input ( CtType<AxisValues<PI,2>>(), buf     );
///   auto tv = tensor_view_output( CtType<AxisValues<PI,0>>(), res_buf );
///
/// Axes with a compile-time known size use StaticAxisValue in the AxisValues:
///   auto tv = tensor_view_input( CtType<AxisValues<PI,2,StaticAxisValue<PI,1,3>>>(), buf );

#include "support/containers/DynamicAxis.h"
#include <xla/ffi/api/ffi.h>
#include <utility>

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
    using Strides = AxisValues<typename Shape::TI,Shape::ct_rank>;
    return Strides( Values(), ( (void)Is, SI(0) )... );
}

template<class Shape>
auto zero_strides() {
    return _zero_strides_impl<Shape>( std::make_index_sequence<Shape::ct_rank>{} );
}


// contiguous_strides<TF>( shape ) now lives in containers/contiguous_strides.h (shared with TensorView)


// ------------------- tensor_view_input -------------------

template<xla::ffi::DataType dtype>
auto tensor_view_input( auto &&shape, const auto &memory_space, xla::ffi::Buffer<dtype> buf, PI8 tensor_type_index = 1 ) {
    using TF = SdotTypeFor<dtype>::type;
    auto strides = contiguous_strides<TF>( shape );

    // invalid
    if ( tensor_type_index == 0 )
        return TensorView<TF,Shape,Strides>::make_invalid( shape, strides );

    // surely-null: shape in buf.dimensions()[1:], zero strides, data points to a static TF(0)
    if ( tensor_type_index == 2 ) {
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

template<class Shape, xla::ffi::DataType dtype, class Arch>
auto dynamic_axis_output( CtType<Shape> cd, PI num_dynamic_axis, PI capacity, xla::ffi::ResultBuffer<dtype> buf, bool valid, const Arch &arch ) {
    using TF = typename SdotTypeFor<dtype>::type;
    auto res = DynamicAxis( num_dynamic_axis, capacity, tensor_view_output( cd, buf, valid ) );
    if ( valid )
        arch.zero_fill( buf->typed_data(), buf->element_count(), sizeof( TF ) );
    return res;
}

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
