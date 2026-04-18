#pragma once

/// Convenience helpers: convert typed XLA FFI buffers to sdot TensorView.
///
/// Usage (in a generated XLA FFI handler):
///
///   auto tv = ffi_tv<TF,2>( arg_0 );   // ffi::Buffer<ffi::DataType::F64> → TensorView<TF,2,Cpu>
///   auto tv = ffi_tv<TF,1>( *res_0 );  // dereference Result<Buffer> first

#include "support/jax_ffi.h"
#include "support/TensorView.h"

namespace sdot {

template<typename T, int Rank, typename BufT>
static auto ffi_tv( BufT &&buf ) {
    PI ndim = buf.dimensions().size();
    DsVec<PI,Rank,Cpu> sizes( Size(), ndim );
    for ( int i = 0; i < (int)ndim; ++i )
        sizes[ i ] = buf.dimensions()[ i ];

    // XLA FFI guarantees C-contiguous (row-major) layout
    return TensorView<T,Rank,Cpu>( reinterpret_cast<T*>( buf.untyped_data() ), sizes );
}

// convenience aliases for the two most common dtypes
template<int Rank, typename BufT> static auto ffi_tvf( BufT &&buf ) { return ffi_tv<double,Rank>( std::forward<BufT>( buf ) ); }
template<int Rank, typename BufT> static auto ffi_tvi( BufT &&buf ) { return ffi_tv<PI,Rank>( std::forward<BufT>( buf ) ); }

} // namespace sdot
