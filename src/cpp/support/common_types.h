#pragma once

#include <cstddef>
#include <cstdint>

namespace sdot {

using FP64 = double;
using FP32 = float;

using PI32 = std::int32_t;
using PI64 = std::int64_t;

using SI = std::ptrdiff_t;
using PI = std::size_t;

template<class T> struct CtType {};

inline static const char *type_name( CtType<FP64> ) { return "FP64"; }
inline static const char *type_name( CtType<FP32> ) { return "FP32"; }

// ctor args
struct Extent {};

}
