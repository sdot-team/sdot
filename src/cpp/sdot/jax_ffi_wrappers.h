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
#include "support/CtType.h"
#include <xla/ffi/api/ffi.h>

namespace sdot {

enum {
    TENSOR_TYPE_STD = 0,
    TENSOR_TYPE_ZERO = 1,
    TENSOR_TYPE_INVALID = 2,
};

// ------------------- tensor_view_input -------------------

template<class Shape,xla::ffi::DataType dtype>
auto tensor_view_inp( CtType<Shape>, const auto &memory_space, xla::ffi::Buffer<dtype> buf, PI8 tensor_type ) {
    using MemorySpace = DECAYED_TYPE_OF( memory_space );
    using TF = DECAYED_TYPE_OF( *buf.typed_data() );

    // shape (there's a particular case for TENSOR_TYPE_ZERO)
    Shape shape( Function(), [&]( auto index ) {
        const PI offset = ( tensor_type == TENSOR_TYPE_ZERO );
        return buf.dimensions()[ index + offset ];
    } );

    // strides
    auto strides = contiguous_strides<TF>( shape );
    using Strides = DECAYED_TYPE_OF( strides );

    // return type
    using RT = TensorView<TF,MemorySpace,Shape,Strides>;

    // symbolic zero -> clear strides
    if ( tensor_type == TENSOR_TYPE_ZERO ) {
        PI nb_zeros = 0;
        strides.for_each_item( [&]( auto &value ) {
            if constexpr ( requires { DECAYED_TYPE_OF( value )::value; } ) {
                nb_zeros += DECAYED_TYPE_OF( value )::value == 0;
            } else {
                ++nb_zeros;
                value = 0;
            }
        } );
        if ( nb_zeros == 0 )
            ERROR( "symbolic zero tensors are possible if stride can be modified" );

        return RT( zero_for<TF>( memory_space ), shape, strides, memory_space );
    }

    // invalid
    if ( tensor_type == TENSOR_TYPE_INVALID )
        return RT::make_invalid( shape, strides, memory_space );

    // standard
    if ( tensor_type == TENSOR_TYPE_STD )
        return RT( const_cast<TF *>( buf.typed_data() ), shape, strides, memory_space );

    ERROR( "unknown tensor_type" );
    return RT( const_cast<TF *>( buf.typed_data() ), shape, strides, memory_space );
}


// ------------------- tensor_view_output -------------------

template<class Shape, xla::ffi::DataType dtype>
auto tensor_view_out( CtType<Shape> cs, const auto &memory_space, xla::ffi::ResultBuffer<dtype> buf, PI8 tensor_type ) {
    return tensor_view_inp( cs, memory_space, *buf, tensor_type );
}

// ------------------- tensor_view_mutable -------------------

template<class Shape, xla::ffi::DataType dtype>
auto tensor_view_mut( CtType<Shape> cs, const auto &memory_space, xla::ffi::ResultBuffer<dtype> buf_out, xla::ffi::Buffer<dtype> buf_inp, PI8 tensor_type_out, PI8 tensor_type_inp ) {
    auto out = tensor_view_out( cs, memory_space, buf_out, tensor_type_out );

    if ( tensor_type_out != TENSOR_TYPE_INVALID && tensor_type_inp != TENSOR_TYPE_INVALID ) {
        auto inp = tensor_view_inp( cs, memory_space, buf_inp, tensor_type_inp );
        out.copy_elements_from( inp );
    }

    return out;
}

// ------------------- dynamic_axis -------------------

template<class Shape, xla::ffi::DataType dtype>
auto dynamic_axis_inp( CtType<Shape> cs, const auto &memory_space, xla::ffi::Buffer<dtype> buf, PI8 tensor_type, PI num_dynamic_axis, PI capacity ) {
    return DynamicAxis( num_dynamic_axis, capacity, tensor_view_inp( cs, memory_space, buf, tensor_type ) );
}

template<class Shape, xla::ffi::DataType dtype>
auto dynamic_axis_out( CtType<Shape> cs, const auto &memory_space, xla::ffi::ResultBuffer<dtype> buf, PI8 tensor_type, PI num_dynamic_axis, PI capacity ) {
    auto res = DynamicAxis( num_dynamic_axis, capacity, tensor_view_out( cs, memory_space, buf, tensor_type ) );
    if ( tensor_type != TENSOR_TYPE_INVALID )
        res.sizes.fill_with( 0 );
    return res;
}

template<class Shape, xla::ffi::DataType dtype>
auto dynamic_axis_mut( CtType<Shape> cs, const auto &memory_space, xla::ffi::Buffer<dtype> buf_inp, PI8 tensor_type_inp, xla::ffi::ResultBuffer<dtype> buf_out, PI8 tensor_type_out, PI num_dynamic_axis, PI capacity ) {
    auto res = dynamic_axis_out( cs, memory_space, buf_out, tensor_type_out );
    if ( tensor_type_inp != TENSOR_TYPE_INVALID ) {
        auto inp = tensor_view_inp( cs, memory_space, buf_inp, tensor_type_inp );
        res.sizes.copy_elements_from( inp );
    }
    return res;
}


// ------------------- first_positive -------------------

SI first_positive( const std::string &/* name */ ) { return 0; }
SI first_positive( const std::string &name, SI head, auto...tail ) { return head >= 0 ? head : first_positive( name, tail... ); }

} // namespace sdot
