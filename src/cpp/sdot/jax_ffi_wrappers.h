#pragma once

/// Convenience helpers: convert typed XLA FFI buffers to sdot TensorView.
///
/// Usage (in a generated XLA FFI handler):
///
///   auto tv = ffi_tv<TF,2>( arg_0 );   // ffi::Buffer<ffi::DataType::F64> → TensorView<TF,2,Cpu>
///   auto tv = ffi_tv<TF,1>( *res_0 );  // dereference Result<Buffer> first

#include "support/TensorView.h"
#include <xla/ffi/api/ffi.h>
// #include "support/P.h"

namespace sdot {

// // convenience aliases for the two most common dtypes
template<xla::ffi::DataType dtype> struct SdotTypeFor;
template<> struct SdotTypeFor<xla::ffi::DataType::F32> { using type = FP32; };
template<> struct SdotTypeFor<xla::ffi::DataType::F64> { using type = FP64; };
template<> struct SdotTypeFor<xla::ffi::DataType::U32> { using type = PI32; };
template<> struct SdotTypeFor<xla::ffi::DataType::U64> { using type = PI64; };
template<> struct SdotTypeFor<xla::ffi::DataType::S32> { using type = SI32; };
template<> struct SdotTypeFor<xla::ffi::DataType::S64> { using type = SI64; };


// xla::ffi::Buffer
template<int ndim,xla::ffi::DataType dtype>
auto tensor_view( CtInt<ndim>, xla::ffi::Buffer<dtype> buf, bool valid = true ) {
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
auto tensor_view( CtInt<ndim>, xla::ffi::ResultBuffer<dtype> buf, bool valid = true ) {
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
auto tensor_view( CtInt<ndim>, xla::ffi::ResultBuffer<dtype> buf_out, xla::ffi::Buffer<dtype> buf_inp, bool valid = true ) {
    if ( ! valid )
        return tensor_view( CtInt<ndim>(), buf_out, false );

    auto out = tensor_view( CtInt<ndim>(), buf_out, true );
    auto inp = tensor_view( CtInt<ndim>(), buf_inp, true );
    out.get_data_from( inp );
    return out;
}

bool test_and_shift( auto &mask ) {
    bool res = mask & 1;
    mask >>= 1;
    return res;
}

// ------------------- first_valid_dimension -------------------
PI first_valid_dimension( const PI64 */* u64_input */ ) {
    return 0;
}
template<xla::ffi::DataType dtype>
PI first_valid_dimension( const PI64 *u64_input, const xla::ffi::Buffer<dtype> &buffer, PI validity_index, PI num_axis, auto&& ...tail ) {
    if ( u64_input[ validity_index / 64 ] & ( PI64( 1 ) << ( validity_index % 64 ) ) )
        return buffer.dimensions()[ num_axis ];
    return first_valid_dimension( u64_input, FORWARD( tail )... );
}
template<xla::ffi::DataType dtype>
PI first_valid_dimension( const PI64 *u64_input, xla::ffi::ResultBuffer<dtype> &buffer, PI validity_index, PI num_axis, auto& ...tail ) {
    if ( u64_input[ validity_index / 64 ] & ( PI64( 1 ) << ( validity_index % 64 ) ) )
        return buffer->dimensions()[ num_axis ];
    return first_valid_dimension( u64_input, FORWARD( tail )... );
}

} // namespace sdot
