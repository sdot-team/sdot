#pragma once

#include "TensorView.h"

namespace sdot {

//
struct DynamicSizeException {
    PI num_dynamic_axis;
    PI needed_size;
};


//
template<class TI,int rank,class Arch>
class DynamicAxis {
public:
    using        Sizes         = TensorView<TI,rank,Arch>;

    // slicing/subparts
    auto         operator()    ( auto...indices ) const { constexpr int new_rank = rank - int( sizeof...( indices ) ); return DynamicAxis<TI,new_rank,Arch>( num_dynamic_axis, capacity, sizes.partial( indices... ) ); }
    auto         row           ( auto index ) const { return operator()( index ); }

    // info
    bool         is_invalid    () const { return sizes.is_invalid(); }
    bool         is_valid      () const { return sizes.is_valid(); }
    PI           size          ( PI ind ) const { return sizes.size( ind ); }

    // assuming rank == 0
    PI           post_increment( PI value ) { PI res = sizes(); operator=( res + value ); return res; }
    PI           operator++    () { PI res = sizes() + 1; operator=( res ); return res; }
    PI           operator++    ( int ) { PI res = sizes(); operator=( res + 1 ); return res; }
    PI           operator--    () { PI res = sizes() - 1; operator=( res ); return res; }
    PI           operator--    ( int ) { PI res = sizes(); operator=( res - 1 ); return res; }
    DynamicAxis& operator=     ( PI new_size ) { if ( new_size > capacity ) overflow( new_size ); sizes() = new_size; return *this; }
    operator     PI            () const { return sizes(); }

    // exception
    void         overflow      ( PI needed_size ) { throw DynamicSizeException( num_dynamic_axis, needed_size ); }

    const PI     num_dynamic_axis;
    const PI     capacity;
    Sizes        sizes;
};

} // namespace sdot
