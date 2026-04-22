#pragma once

/// Convenience helpers: convert typed XLA FFI buffers to sdot TensorView.
///
/// Usage (in a generated XLA FFI handler):
///
///   auto tv = ffi_tv<TF,2>( arg_0 );   // ffi::Buffer<ffi::DataType::F64> → TensorView<TF,2,Cpu>
///   auto tv = ffi_tv<TF,1>( *res_0 );  // dereference Result<Buffer> first

#include "support/TensorView.h"
#include <xla/ffi/api/ffi.h>
// #include "support/jax_ffi.h"

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
        return TensorView<const TF,ndim,Cpu>::make_invalid( ndim );

    ASSERT_EQ( ndim, buf.dimensions().size() );
    DsVec<PI,ndim,Cpu> sizes( Size(), ndim );
    for( int i = 0; i < ndim; ++i )
        sizes[ i ] = buf.dimensions()[ i ];

    // XLA FFI guarantees C-contiguous (row-major) layout
    return TensorView<const TF,ndim,Cpu>( reinterpret_cast<const TF *>( buf.untyped_data() ), sizes );
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
    return TensorView<TF,ndim,Cpu>( reinterpret_cast<TF *>( buf->untyped_data() ), sizes );
}

bool test_and_shift( auto &mask ) {
    bool res = mask & 1;
    mask >>= 1;
    return res;
}

} // namespace sdot
