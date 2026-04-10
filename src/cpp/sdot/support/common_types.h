#pragma once

#include "common_macros.h"
#include <iostream>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <array>

namespace sdot {

using FP64 = double;
using FP32 = float;

using PI32 = std::int32_t;
using PI64 = std::int64_t;

using SI = std::ptrdiff_t;
using PI = std::size_t;

template<class T> struct CtType {};
template<int i> struct CtInt {};

inline static const char *type_name( CtType<FP64> ) { return "FP64"; }
inline static const char *type_name( CtType<FP32> ) { return "FP32"; }

// ctor args
struct Shape {};

template<class TF> TF factorial( TF n ) {
    TF res = 1;
    for ( PI i = 2; i <= n; ++i )
        res *= TF( i );
    return res;
}

} // namespace sdot


T_Up std::ostream &operator<<( std::ostream &os, const std::array<U,p> &v ) {
    for( sdot::PI i = 0; i < v.size(); ++i )
        os << ( i ? ", " : "" ) << v[ i ];
    return os;
}

T_T std::ostream &operator<<( std::ostream &os, const std::vector<T> &p ) {
    for( sdot::PI i = 0; i < p.size(); ++i )
        os << ( i ? ", " : "" ) << p[ i ];
    return os;
}
