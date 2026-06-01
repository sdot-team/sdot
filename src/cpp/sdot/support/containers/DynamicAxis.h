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
    using           Sizes          = TensorView<TI,MemorySpace,Shape,Strides>;

    // slicing/subparts
    HD auto         operator()     ( auto...indices ) const { auto new_sizes = sizes( indices... ); using TT = DECAYED_TYPE_OF( new_sizes ); return DynamicAxis<TI,MemorySpace,typename TT::Shape,typename TT::Strides>( num_dynamic_axis, capacity, new_sizes ); }
    HD auto         row            ( auto index ) const { return operator()( index ); }

    // info
    HD auto         transfer_cost  ( const auto &execution_context ) const { return sizes.transfer_cost( execution_context ); }
    HD bool         is_invalid     () const { return sizes.is_invalid(); }
    HD bool         is_valid       () const { return sizes.is_valid(); }
    HD auto         shape          ( auto ind ) const { return sizes.shape( ind ); }
    HD auto         size           ( auto ind ) const { return sizes.size( ind ); }

    // assuming rank == 0
    HD PI           post_increment ( PI value ) { PI res = sizes.value(); operator=( res + value ); return res; }
    HD PI           operator++     () { PI res = sizes.value() + 1; operator=( res ); return res; }
    HD PI           operator++     ( int ) { PI res = sizes.value(); operator=( res + 1 ); return res; }
    HD PI           operator--     () { PI res = sizes.value() - 1; operator=( res ); return res; }
    HD PI           operator--     ( int ) { PI res = sizes.value(); operator=( res - 1 ); return res; }
    HD DynamicAxis& operator=      ( PI new_size ) { if ( new_size > capacity ) overflow( new_size ); sizes = new_size; return *this; }
    HD operator     PI             () const { return sizes.value(); }


    // exception
    HD void         overflow       ( PI needed_size ) {
        info( needed_size, capacity );
        #ifndef __CUDACC__
        throw DynamicSizeException( num_dynamic_axis, needed_size );
        #endif
    }

    // creation
    HD void         with_same_shape( auto &&func ) const { sizes.with_same_shape( [&]( auto &sizes ) { DynamicAxis da( num_dynamic_axis, capacity, sizes ); func( da ); } ); }

    const PI        num_dynamic_axis = 0;
    const PI        capacity = 0;
    Sizes           sizes;
};

} // namespace sdot
