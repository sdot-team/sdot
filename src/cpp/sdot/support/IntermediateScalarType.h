#pragma once

#include "Cpu.h"

namespace sdot {

///
template<class T,class Arch>
struct IntermediateScalarType {
    using type = T;
};

// specializations
template<> struct IntermediateScalarType<float,Cpu> { using type = double; };

} // namespace sdot
