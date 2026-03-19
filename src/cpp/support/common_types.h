#pragma once

#include <cstddef>

namespace sdot {

using FP64 = double;
using FP32 = float;

using PI = std::size_t;

template<class T> struct CtType {};

inline static const char *type_name( CtType<FP64> ) { return "FP64"; }
inline static const char *type_name( CtType<FP32> ) { return "FP32"; }

}
