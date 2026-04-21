// #pragma once

// #include <xla/ffi/api/ffi.h>
// #include "TensorView.h"
// #include <optional>
// #include <numeric>

// namespace sdot {

// namespace ffi = xla::ffi;

// template<typename T, int Rank, typename Arch>
// struct JAX_FFI_Buffer_To_TensorView;

// template<typename T, int Rank>
// struct JAX_FFI_Buffer_To_TensorView<T, Rank, Cpu> {
//     static auto get(const ffi::AnyBuffer& buffer) {
//         PI r = buffer.dimensions().size();
//         DsVec<PI, Rank, Cpu> sizes(Size(), r);
//         for (int i = 0; i < (int)r; ++i) {
//             sizes[i] = buffer.dimensions()[i];
//         }

//         // JAX FFI doesn't seem to have strides in this version, assuming contiguous
//         return TensorView<T, Rank, Cpu>(reinterpret_cast<T*>(buffer.untyped_data()), sizes);
//     }

//     static auto get(ffi::Result<ffi::AnyBuffer> buffer) {
//         return get(*buffer);
//     }
// };

// } // namespace sdot
