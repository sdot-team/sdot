#pragma once

#include "TensorView.h"

namespace sdot {

//
struct DynamicSizeException {
    PI num_dynamic_axis;
    PI needed_size;
};


//
template<class TI,class Shape,class Strides>
class DynamicAxis {
public:
    using        Sizes          = TensorView<TI,Shape,Strides>;

    // slicing/subparts
    auto         operator()     ( auto...indices ) const { auto new_sizes = sizes( indices... ); using TT = DECAYED_TYPE_OF( new_sizes ); return DynamicAxis<TI,typename TT::Shape,typename TT::Strides>( num_dynamic_axis, capacity, new_sizes ); }
    auto         row            ( auto index ) const { return operator()( index ); }

    // info
    bool         is_invalid     () const { return sizes.is_invalid(); }
    bool         is_valid       () const { return sizes.is_valid(); }
    PI           shape          ( PI ind ) const { return sizes.shape( ind ); }
    PI           size           ( PI ind ) const { return sizes.size( ind ); }

    // assuming rank == 0
    PI           post_increment ( PI value ) { PI res = sizes.item(); operator=( res + value ); return res; }
    PI           operator++     () { PI res = sizes.item() + 1; operator=( res ); return res; }
    PI           operator++     ( int ) { PI res = sizes.item(); operator=( res + 1 ); return res; }
    PI           operator--     () { PI res = sizes.item() - 1; operator=( res ); return res; }
    PI           operator--     ( int ) { PI res = sizes.item(); operator=( res - 1 ); return res; }
    DynamicAxis& operator=      ( PI new_size ) { if ( new_size > capacity ) overflow( new_size ); sizes.item() = new_size; return *this; }
    operator     PI             () const { return sizes.item(); }

    // exception
    void         overflow       ( PI needed_size ) { throw DynamicSizeException( num_dynamic_axis, needed_size ); }

    // creation
    void         with_same_shape( auto &&func ) const { sizes.with_same_shape( [&]( auto &sizes ) { DynamicAxis da( num_dynamic_axis, capacity, sizes ); func( da ); } ); }

    const PI     num_dynamic_axis;
    const PI     capacity;
    Sizes        sizes;
};

} // namespace sdot
