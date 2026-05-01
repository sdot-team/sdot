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

using PI8  = std::uint8_t;
using PI32 = std::uint32_t;
using PI64 = std::uint64_t;

using SI8  = std::int8_t;
using SI32 = std::int32_t;
using SI64 = std::int64_t;

using SI = long long;
using PI = std::size_t;

template<class T> struct CtType {};
template<int i> struct CtInt { static constexpr int value = i; constexpr operator int() const { return i; } };

inline static const char *type_name( CtType<FP64> ) { return "FP64"; }
inline static const char *type_name( CtType<FP32> ) { return "FP32"; }

// ctor args
struct SizeAndCtorArgs {};
struct Reserved {};
struct Values {};
struct Shape {};
struct Rank {};
struct Size {};

template<class TF> TF factorial( TF n ) {
    TF res = 1;
    for ( PI i = 2; i <= n; ++i )
        res *= TF( i );
    return res;
}

template<class TF> constexpr TF pow_rec( TF v, PI n ) {
    return n ? pow_rec( v, n - 1 ) * v : 1;
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
