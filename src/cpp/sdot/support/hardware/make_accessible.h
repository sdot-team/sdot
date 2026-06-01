#pragma once

#include "transfer_cost.h"

namespace sdot {

HD void make_accessible( const auto &execution_space, auto &&value, auto /* inp */, auto /* out */, auto &&func ) {
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


