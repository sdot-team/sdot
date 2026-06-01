#pragma once

/// Convenience helpers: convert typed XLA FFI buffers to sdot TensorView / ZeroTensor / NoneTensor.
///
/// Usage (in a generated XLA FFI handler):
///
///   auto tv  = tensor_view_inp( CtType<AxisValues<PI,2>>(), memory_space, buf );
///   auto zt  = zero_tensor_inp( CtType<AxisValues<PI,2>>(), buf );          // gradient is zero
///   auto nt  = none_tensor_inp<float>();                                     // gradient absent

#include "support/containers/DynamicAxis.h"
#include "support/containers/NoneTensor.h"  // IWYU pragma: export
#include "support/containers/ZeroTensor.h"  // IWYU pragma: export
#include "support/CtType.h"

#include <xla/ffi/api/ffi.h>

namespace sdot {

// ------------------- tensor_view_inp -------------------

template<class Shape,xla::ffi::DataType dtype>
auto tensor_view_inp( CtType<Shape>, const auto &memory_space, xla::ffi::Buffer<dtype> buf ) {
    using MemorySpace = DECAYED_TYPE_OF( memory_space );
    using TF          = DECAYED_TYPE_OF( *buf.typed_data() );

    Shape shape( Function(), [&]( auto index ) { return buf.dimensions()[ index ]; } );

    auto strides = contiguous_strides<TF>( shape );
    using Strides = DECAYED_TYPE_OF( strides );

    return TensorView<TF,MemorySpace,Shape,Strides>( const_cast<TF *>( buf.typed_data() ), shape, strides, memory_space );
}

// ------------------- zero_tensor_inp -------------------
// The Python side prepends a marker dimension of size 0 to the buffer shape,
// so the actual tensor shape starts at dimensions[1].

template<class Shape,xla::ffi::DataType dtype>
auto zero_tensor_inp( CtType<Shape>, xla::ffi::Buffer<dtype> buf ) {
    using TF = DECAYED_TYPE_OF( *buf.typed_data() );

    Shape shape( Function(), [&]( auto index ) { return buf.dimensions()[ index + 1 ]; } );

    return ZeroTensor<TF,Shape>( shape );
}

// ------------------- none_tensor_inp -------------------

template<class TF>
auto none_tensor_inp() {
    return NoneTensor<TF>{};
}


// ------------------- tensor_view_out -------------------

template<class Shape, xla::ffi::DataType dtype>
auto tensor_view_out( CtType<Shape> cs, const auto &memory_space, xla::ffi::ResultBuffer<dtype> buf ) {
    return tensor_view_inp( cs, memory_space, *buf );
}

// ------------------- tensor_view_mut -------------------

template<class Shape, xla::ffi::DataType dtype>
auto tensor_view_mut( CtType<Shape> cs, const auto &memory_space, xla::ffi::ResultBuffer<dtype> buf_out, xla::ffi::Buffer<dtype> buf_inp ) {
    auto out = tensor_view_out( cs, memory_space, buf_out );
    out.copy_elements_from( tensor_view_inp( cs, memory_space, buf_inp ) );
    return out;
}

// ------------------- dynamic_axis -------------------

template<class Shape, xla::ffi::DataType dtype>
auto dynamic_axis_inp( CtType<Shape> cs, const auto &memory_space, xla::ffi::Buffer<dtype> buf, PI num_dynamic_axis, PI capacity ) {
    return DynamicAxis( num_dynamic_axis, capacity, tensor_view_inp( cs, memory_space, buf ) );
}

template<class Shape, xla::ffi::DataType dtype>
auto dynamic_axis_out( CtType<Shape> cs, const auto &memory_space, xla::ffi::ResultBuffer<dtype> buf, PI num_dynamic_axis, PI capacity ) {
    auto res = DynamicAxis( num_dynamic_axis, capacity, tensor_view_out( cs, memory_space, buf ) );
    res.sizes.fill_with( 0 );
    return res;
}

template<class Shape, xla::ffi::DataType dtype>
auto dynamic_axis_mut( CtType<Shape> cs, const auto &memory_space, xla::ffi::Buffer<dtype> buf_inp, xla::ffi::ResultBuffer<dtype> buf_out, PI num_dynamic_axis, PI capacity ) {
    auto res = dynamic_axis_out( cs, memory_space, buf_out, num_dynamic_axis, capacity );
    res.sizes.copy_elements_from( tensor_view_inp( cs, memory_space, buf_inp ) );
    return res;
}


// ------------------- first_positive -------------------

// SI first_positive( const std::string &/* name */ ) { return 0; }
// SI first_positive( const std::string &name, SI head, auto...tail ) { return head >= 0 ? head : first_positive( name, tail... ); }

} // namespace sdot
