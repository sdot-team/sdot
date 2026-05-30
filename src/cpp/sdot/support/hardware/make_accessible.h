#pragma once

#include "../common_macros.h"

namespace sdot {

HD void make_accessible( const auto &/* execution_space */, auto &&value, auto /* inp */, auto /* out */, auto &&func ) {
    if constexpr ( std::is_pod<DECAYED_TYPE_OF( value )>::value )
        func( FORWARD( value ) );
    else if constexpr ( requires { for_each_attribute( value, []( auto, auto & ) {} ); } ) {
        auto raw_copy = value;
        for_each_attribute( raw_copy, []( auto, auto &attribute ) {
            ...
        } );
    } else
        value.todo_make_accessible();
}

} // namespace sdot


