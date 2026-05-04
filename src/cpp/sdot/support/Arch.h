#pragma once

#include "common_types.h"
#include <vector>

#ifdef __CUDACC__
#include <thrust/device_vector.h>
#endif

namespace sdot {

#ifdef __CUDACC__
struct Cuda {
    template<class T> struct Vector { using type = thrust::device_vector<T>; };

    static auto        raw_ptr( auto &&vec ) { return thrust::raw_pointer_cast( vec.data() ); }
    static const char *name   () { return "cuda"; }
};
#endif

struct Cpu {
    T_T struct         Vector          { using type = std::vector<T>; };

    T_T void           with_reservation( PI size, auto &&func ) const { T *res = new T[ size ]; func( res ); delete [] res; }
    static auto        raw_ptr         ( auto &&vec ) { return vec.data(); }
    static const char* name            () { return "cpu"; }

};

} // namespace sdot
