#pragma once

// #include "common_macros.h"
#include "common_types.h"
#include <algorithm>
#include <array>

namespace sdot {

/// simple view on contiguous data
template<class T,int dim>
class TensorView {
public:
    using  Extent    = std::array<PI,dim>;

    /* */  TensorView( T *data, Extent extent ) : extent( extent ), ptr( data ) {}
    /* */  TensorView( T *data, PI size ) : extent{ size }, ptr( data ) {}

    T&     operator()( PI i0, PI i1 ) const { return ptr[ i0 * extent[ 0 ] + i1 ]; }
    T&     operator()( PI i0 ) const { return ptr[ i0 ]; }
    T&     operator()() const { return ptr[ 0 ]; }

    T&     operator[]( PI i0 ) const { return ptr[ i0 ]; }

    bool   empty     () const { return dim == 0 ? false : std::none_of( extent.begin(), extent.end(), []( auto a ) { return a != 0; } ); }

    PI     size      ( PI d ) const { return extent[ d ]; }
    PI     size      () const { return size( 0 ); }

    auto   row       ( PI index ) const -> TensorView<T,dim-1> { std::array<PI,dim-1> nextent; PI off = index; for( PI i = 1; i < dim; ++i ) { nextent[ i - 1 ] = extent[ i ]; off *= extent[ i ]; } return { ptr + off, nextent }; }

private:
    Extent extent;   ///<
    T*     ptr;      ///<
};

// Helper to get a row from a 2D mdspan since std::submdspan is C++26

} // namespace sdot

#include "DiracSet.cxx"
