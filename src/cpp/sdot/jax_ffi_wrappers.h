#pragma once

/// Convenience helpers: convert typed XLA FFI buffers to sdot TensorView.
///
/// Usage (in a generated XLA FFI handler):
///
///   auto tv = ffi_tv<TF,2>( arg_0 );   // ffi::Buffer<ffi::DataType::F64> → TensorView<TF,2,Cpu>
///   auto tv = ffi_tv<TF,1>( *res_0 );  // dereference Result<Buffer> first

#include "support/DynamicAxis.h"
#include "support/P.h"
#include <xla/ffi/api/ffi.h>

namespace sdot {


// convenience aliases for the two most common dtypes
template<xla::ffi::DataType dtype> struct SdotTypeFor;
template<> struct SdotTypeFor<xla::ffi::DataType::F32> { using type = FP32; };
template<> struct SdotTypeFor<xla::ffi::DataType::F64> { using type = FP64; };
template<> struct SdotTypeFor<xla::ffi::DataType::U32> { using type = PI32; };
template<> struct SdotTypeFor<xla::ffi::DataType::U64> { using type = PI64; };
template<> struct SdotTypeFor<xla::ffi::DataType::S32> { using type = SI32; };
template<> struct SdotTypeFor<xla::ffi::DataType::S64> { using type = SI64; };


// ------------------- tensor_view -------------------

template<int ndim,xla::ffi::DataType dtype>
auto tensor_view_input( CtInt<ndim>, xla::ffi::Buffer<dtype> buf, bool valid = true ) {
    using TF = SdotTypeFor<dtype>::type;

    if ( ! valid )
        return TensorView<TF,ndim,Cpu>::make_invalid( ndim );

    ASSERT_EQ( ndim, buf.dimensions().size() );
    DsVec<PI,ndim,Cpu> sizes( Size(), ndim );
    for( int i = 0; i < ndim; ++i )
        sizes[ i ] = buf.dimensions()[ i ];

    // XLA FFI guarantees C-contiguous (row-major) layout
    return TensorView<TF,ndim,Cpu>( const_cast<TF *>( buf.typed_data() ), sizes );
}


// xla::ffi::ResultBuffer
template<int ndim,xla::ffi::DataType dtype>
auto tensor_view_output( CtInt<ndim>, xla::ffi::ResultBuffer<dtype> buf, bool valid = true ) {
    using TF = SdotTypeFor<dtype>::type;

    if ( ! valid )
        return TensorView<TF,ndim,Cpu>::make_invalid( ndim );

    ASSERT_EQ( ndim, buf->dimensions().size() );
    DsVec<PI,ndim,Cpu> sizes( Size(), ndim );
    for( int i = 0; i < ndim; ++i )
        sizes[ i ] = buf->dimensions()[ i ];

    // XLA FFI guarantees C-contiguous (row-major) layout
    return TensorView<TF,ndim,Cpu>( buf->typed_data(), sizes );
}

// mutable
template<int ndim,xla::ffi::DataType dtype>
auto tensor_view_mutable( CtInt<ndim>, xla::ffi::ResultBuffer<dtype> buf_out, xla::ffi::Buffer<dtype> buf_inp, bool valid = true ) {
    if ( ! valid )
        return tensor_view( CtInt<ndim>(), buf_out, false );

    auto out = tensor_view( CtInt<ndim>(), buf_out, true );
    auto inp = tensor_view( CtInt<ndim>(), buf_inp, true );
    out.get_data_from( inp );
    return out;
}

// ------------------- dynamic_axis -------------------
template<int ndim,xla::ffi::DataType dtype>
auto dynamic_axis_input( CtInt<ndim> cd, PI num_dynamic_axis, PI capacity, xla::ffi::Buffer<dtype> buf, bool valid ) {
    return DynamicAxis( num_dynamic_axis, capacity, tensor_view_input( cd, buf, valid ) );
}


// xla::ffi::ResultBuffer
template<int ndim,xla::ffi::DataType dtype>
auto dynamic_axis_output( CtInt<ndim> cd, PI num_dynamic_axis, PI capacity, xla::ffi::ResultBuffer<dtype> buf, bool valid ) {
    auto res = DynamicAxis( num_dynamic_axis, capacity, tensor_view_output( cd, buf, valid ) );
    res.sizes.fill_with( 0 );
    return res;
}

// mutable
template<int ndim,xla::ffi::DataType dtype>
auto dynamic_axis_mutable( CtInt<ndim> cd, PI num_dynamic_axis, PI capacity, xla::ffi::Buffer<dtype> buf_inp, bool valid_inp, xla::ffi::ResultBuffer<dtype> buf_out, bool /* valid_out */ ) {
    auto res = DynamicAxis( num_dynamic_axis, capacity, tensor_view_output( cd, buf_out ) );
    if ( valid_inp ) {
        auto inp = tensor_view_output( cd, buf_inp );
        res.sizes.get_data_from( inp );
    }
    return res;
}

// ------------------- first_positive -------------------
SI first_positive() { throw std::runtime_error( "no positive value" ); }
SI first_positive( SI head, auto...tail ) { return head >= 0 ? head : first_positive( tail... ); }

} // namespace sdot
