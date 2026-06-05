#pragma once

#include "transfer_cost.h"

namespace sdot {

template<class ES,class V,class I,class O,class F>
HD void make_accessible( const ES &execution_space, V &&value, I /* inp */, O /* out */, F &&func ) {
    constexpr bool available = std::is_trivial<DECAYED_TYPE_OF( value )>::value ||
                               DECAYED_TYPE_OF( transfer_cost( execution_space, value ) )::value == 0;
    if constexpr ( available ) {
        func( FORWARD( value ) );
    // else if constexpr ( requires { for_each_attribute( value, []( auto, auto & ) {} ); } ) {
    //     auto raw_copy = value;
    //     for_each_attribute( raw_copy, []( auto, auto &attribute ) {
    //         ...
    //     } );
    } else
        value.todo_make_accessible();
}

} // namespace sdot


