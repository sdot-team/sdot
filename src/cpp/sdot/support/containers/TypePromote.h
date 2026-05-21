#pragma once

#include "../common_types.h"
#include "Void.h"

namespace sdot {

template<class ...Args>
struct TypePromote;

template<class A,class B,class ...Tail>
struct TypePromote<A,B,Tail...> {
    using type = TypePromote<typename TypePromote<A,B>::type,Tail...>;
};

template<class A> struct TypePromote<A> { using type = A; };

template<class A> struct TypePromote<A,A> { using type = A; };

template<class A> struct TypePromote<A,Void> { using type = A; };
template<class A> struct TypePromote<Void,A> { using type = A; };

template<> struct TypePromote<SI32,SI64> { using type = SI64; };
template<> struct TypePromote<SI64,SI32> { using type = SI64; };

template<> struct TypePromote<PI32,PI64> { using type = PI64; };
template<> struct TypePromote<PI64,PI32> { using type = PI64; };

} // namespace sdot
