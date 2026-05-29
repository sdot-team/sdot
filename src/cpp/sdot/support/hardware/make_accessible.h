#pragma once

#include "../common_macros.h"

namespace sdot {

HD void make_accessible_inp( const auto &execution_space, auto &&value, auto &&func ) {
    if constexpr ( requires { value.make_accessible_inp( execution_space, FORWARD( func ) ); } )
        value.make_accessible_inp( execution_space, FORWARD( func ) );
    else
        static_assert( "TODO" ); // func( FORWARD( value ) );
}

HD void make_accessible_out( const auto &execution_space, auto &&value, auto &&func ) {
    if constexpr ( requires { value.make_accessible_out( execution_space, FORWARD( func ) ); } )
        value.make_accessible_out( execution_space, FORWARD( func ) );
    else
        static_assert( "TODO" ); // func( FORWARD( value ) );
}

HD void make_accessible_mut( const auto &execution_space, auto &&value, auto &&func ) {
    if constexpr ( requires { value.make_accessible_mut( execution_space, FORWARD( func ) ); } )
        value.make_accessible_mut( execution_space, FORWARD( func ) );
    else
        static_assert( "TODO" ); // func( FORWARD( value ) );
}

} // namespace sdot


