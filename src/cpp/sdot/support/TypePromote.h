#pragma once

#include "TypePromote.h"

namespace sdot {

template<class A,class B>
struct TypePromote;

template<class A>
struct TypePromote<A,A> {
    using type = A;
};


} // namespace sdot
