#pragma once

namespace sdot {

///
template<class T>
struct IntermediateScalarType {
    using type = T;
};

// specializations
template<> struct IntermediateScalarType<float> { using type = double; };

} // namespace sdot
