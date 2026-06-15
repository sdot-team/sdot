#pragma once

#include "TensorView.h"

namespace sdot {

//
struct DynamicSizeException {
    PI num_dynamic_axis;
    PI needed_size;
};


// forward declaration: rebuild a DynamicAxis from a (possibly tag-rewritten) sizes view, keeping
// all of its TensorView template parameters — including the `Tags...` ( AxisNames ).
template<class TI,class MemorySpace,class Shape,class Strides,class... Tags>
HD auto dynamic_axis_from_sizes( PI num_dynamic_axis, PI capacity, const TensorView<TI,MemorySpace,Shape,Strides,Tags...> &sizes );

//
template<class TI,class MemorySpace,class Shape,class Strides=DECAYED_TYPE_OF( contiguous_strides<TI>( Shape() ) ),class... Tags>
class DynamicAxis {
public:
    using           Sizes          = TensorView<TI,MemorySpace,Shape,Strides,Tags...>;

    // slicing/subparts ( tags are carried through: the new sizes view keeps / rewrites them )
    T_VA HD auto    operator()     ( A...indices ) const { return dynamic_axis_from_sizes( num_dynamic_axis, capacity, sizes( indices... ) ); }
    T_U  HD auto    row            ( U index ) const { return operator()( index ); }

    // named squeeze ( for a batched aggregate's generated squeeze ): resolve the named axis on the
    // sizes view ( its AxisNames tag maps the name to its position — no leading assumption ).
    T_U  HD auto    squeeze        ( U name, PI index = 0 ) const { return dynamic_axis_from_sizes( num_dynamic_axis, capacity, sizes.squeeze( name, index ) ); }

    // same axis with ExtraTags added ( e.g. AxisNames<…> ): delegated to the sizes view
    template<class... ExtraTags>
    HD auto         with_tags      () const { return dynamic_axis_from_sizes( num_dynamic_axis, capacity, sizes.template with_tags<ExtraTags...>() ); }

    // info
    T_U  HD auto    transfer_cost  ( const U &execution_context ) const { return sizes.transfer_cost( execution_context ); }
    HD bool         is_invalid     () const { return sizes.is_invalid(); }
    HD bool         is_valid       () const { return sizes.is_valid(); }
    T_U  HD auto    shape          ( U ind ) const { return sizes.shape( ind ); }
    T_U  HD auto    size           ( U ind ) const { return sizes.size( ind ); }

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
        throw DynamicSizeException{ num_dynamic_axis, needed_size };
        #endif
    }

    // creation
    //HD void       with_same_shape( auto &&func ) const { sizes.with_same_shape( [&]( auto &sizes ) { DynamicAxis da( num_dynamic_axis, capacity, sizes ); func( da ); } ); }

    const PI        num_dynamic_axis = 0;
    const PI        capacity = 0;
    Sizes           sizes;
};

// deduction guide: aggregate CTAD is C++20; this keeps DynamicAxis( n, capa, tensor_view ) working in C++17
template<class TI,class MemorySpace,class Shape,class Strides,class... Tags>
DynamicAxis( PI, PI, TensorView<TI,MemorySpace,Shape,Strides,Tags...> ) -> DynamicAxis<TI,MemorySpace,Shape,Strides,Tags...>;

// rebuild a DynamicAxis from a sizes view, preserving every TensorView template parameter ( Tags
// included ) — used by operator()/squeeze/with_tags so a derived axis keeps its AxisNames.
template<class TI,class MemorySpace,class Shape,class Strides,class... Tags>
HD auto dynamic_axis_from_sizes( PI num_dynamic_axis, PI capacity, const TensorView<TI,MemorySpace,Shape,Strides,Tags...> &sizes ) {
    return DynamicAxis<TI,MemorySpace,Shape,Strides,Tags...>{ num_dynamic_axis, capacity, sizes };
}

} // namespace sdot
