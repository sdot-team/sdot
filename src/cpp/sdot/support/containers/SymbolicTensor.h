#pragma once

#include "common_macros.h"
#include "common_types.h"
#include <functional>

namespace sdot {

template<class T,int ct_rank>
class SymbolicTensor;

template<class T>
class SymbolicTensor<T,1> {
public:
    using          Function  = std::function<T( PI )>;

    HD auto        operator()( PI i0 ) const { return function( i0 ); }
    HD auto        operator[]( PI i0 ) const { return function( i0 ); }
    HD PI          rank      () const { return 1; }

    Function       function;
};

template<class T>
class SymbolicTensor<T,2> {
public:
    using          Function  = std::function<T( PI, PI )>;

    HD auto        operator()( PI i0, PI i1 ) const { return function( i0, i1 ); }
    HD PI          rank      () const { return 2; }
    HD auto        row       ( PI r ) const { return SymbolicTensor<T,1>{ .function = [&f=this->function,r]( PI i ) { return f( r, i ); } }; }

    Function       function;
};

template<class T>
class SymbolicTensor<T,3> {
public:
    using          Function  = std::function<T( PI, PI, PI )>;

    HD auto        operator()( PI i0, PI i1, PI i2 ) const { return function( i0, i1, i2 ); }
    HD PI          rank      () const { return 3; }
    HD auto        row       ( PI r ) const { return SymbolicTensor<T,2>{ .function = [&f=this->function,r]( PI i0, PI i1 ) { return f( r, i0, i1 ); } }; }

    Function       function;
};

} // namespace sdot

