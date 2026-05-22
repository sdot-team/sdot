#pragma once

#include "TensorView.h"

namespace sdot {

//
struct DynamicSizeException {
    PI num_dynamic_axis;
    PI needed_size;
};


//
template<class TI,class MemorySpace,class Shape,class Strides=DECAYED_TYPE_OF( contiguous_strides<TI>( Shape() ) )>
class DynamicAxis {
public:
    using           Sizes          = TensorView<TI,Shape,Strides,MemorySpace>;

    // slicing/subparts
    HD auto         operator()     ( auto...indices ) const { auto new_sizes = sizes( indices... ); using TT = DECAYED_TYPE_OF( new_sizes ); return DynamicAxis<TI,typename TT::Shape,typename TT::Strides,MemorySpace>( num_dynamic_axis, capacity, new_sizes ); }
    HD auto         row            ( auto index ) const { return operator()( index ); }

    // info
    HD bool         is_invalid     () const { return sizes.is_invalid(); }
    HD bool         is_valid       () const { return sizes.is_valid(); }
    HD auto         shape          ( auto ind ) const { return sizes.shape( ind ); }
    HD auto         size           ( auto ind ) const { return sizes.size( ind ); }

    // assuming rank == 0
    HD PI           post_increment ( PI value ) { PI res = sizes.item(); operator=( res + value ); return res; }
    HD PI           operator++     () { PI res = sizes.item() + 1; operator=( res ); return res; }
    HD PI           operator++     ( int ) { PI res = sizes.item(); operator=( res + 1 ); return res; }
    HD PI           operator--     () { PI res = sizes.item() - 1; operator=( res ); return res; }
    HD PI           operator--     ( int ) { PI res = sizes.item(); operator=( res - 1 ); return res; }
    HD DynamicAxis& operator=      ( PI new_size ) { if ( new_size > capacity ) overflow( new_size ); sizes.item() = new_size; return *this; }
    HD operator     PI             () const { return sizes.item(); }

    // exception
    HD void         overflow       ( PI needed_size ) {
        #ifndef __CUDACC__
        throw DynamicSizeException( num_dynamic_axis, needed_size );
        #endif
    }

    // creation
    HD void         with_same_shape( auto &&func ) const { sizes.with_same_shape( [&]( auto &sizes ) { DynamicAxis da( num_dynamic_axis, capacity, sizes ); func( da ); } ); }

    const PI        num_dynamic_axis;
    const PI        capacity;
    Sizes           sizes;
};

} // namespace sdot
